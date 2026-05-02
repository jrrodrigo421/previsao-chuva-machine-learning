from __future__ import annotations

import io
import time
import urllib.error
import urllib.request

import pandas as pd

from config import DATA_DIR, LAT_CEMADEN, LON_CEMADEN, NASA_PARAMS


def _parse_nasa_hourly_csv(raw: str) -> pd.DataFrame:
    lines = raw.replace("\r\n", "\n").split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("YEAR,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["-999"])
    df = df.rename(columns={"PRECTOTCORR": "PRECTOT"})
    return df


def fetch_month(year: int, month: int, retries: int = 4) -> pd.DataFrame:
    from calendar import monthrange

    start = f"{year}{month:02d}01"
    last = monthrange(year, month)[1]
    end = f"{year}{month:02d}{last:02d}"
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters={NASA_PARAMS}&community=AG"
        f"&longitude={LON_CEMADEN}&latitude={LAT_CEMADEN}"
        f"&start={start}&end={end}&format=CSV&time-standard=UTC"
    )
    wait = 3.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return _parse_nasa_hourly_csv(raw)
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(wait)
            wait *= 1.5
    raise RuntimeError("unreachable")


def load_or_fetch_nasa_hourly(year: int = 2024) -> pd.DataFrame:
    """Baixa (ou lê do cache) a série horária NASA POWER em UTC.

    O sufixo `_utc` no nome do cache invalida caches antigos que
    foram salvos com o padrão LST default da API.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"nasa_hourly_{year}_pampulha_utc.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["ts_utc"])

    parts = [fetch_month(year, m) for m in range(1, 13)]
    df = pd.concat(parts, ignore_index=True)
    df["ts_utc"] = pd.to_datetime(
        dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR), utc=True
    )
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"])
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    load_or_fetch_nasa_hourly()

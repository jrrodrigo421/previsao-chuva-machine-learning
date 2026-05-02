from __future__ import annotations

import pandas as pd

from config import CEMADEN_DIR, CEMADEN_STATION


def load_cemaden_station_daily() -> pd.Series:
    """Soma diária de valorMedida (mm) na estação Pampulha.

    `datahora` do CEMADEN é publicada em UTC; convertemos para
    America/Sao_Paulo antes de agrupar por dia civil, garantindo
    coerência com INMET e NASA.
    """
    files = sorted(CEMADEN_DIR.glob("*.csv"))
    chunks = []
    usecols = ["codEstacao", "datahora", "valorMedida"]
    for path in files:
        for ch in pd.read_csv(
            path,
            sep=";",
            decimal=",",
            usecols=usecols,
            chunksize=300_000,
            encoding="latin-1",
            low_memory=False,
        ):
            sub = ch[ch["codEstacao"] == CEMADEN_STATION].copy()
            if sub.empty:
                continue
            ts_utc = pd.to_datetime(
                sub["datahora"],
                format="%Y-%m-%d %H:%M:%S.%f",
                errors="coerce",
                utc=True,
            )
            sub = sub.assign(ts_utc=ts_utc).dropna(subset=["ts_utc"])
            sub["day"] = pd.to_datetime(
                sub["ts_utc"].dt.tz_convert("America/Sao_Paulo").dt.date
            )
            chunks.append(sub[["day", "valorMedida"]])

    if not chunks:
        raise RuntimeError("Nenhuma linha CEMADEN para a estação configurada.")
    all_d = pd.concat(chunks, ignore_index=True)
    daily = all_d.groupby("day", as_index=True)["valorMedida"].sum()
    daily.name = "cemaden_mm_day"
    return daily.sort_index()

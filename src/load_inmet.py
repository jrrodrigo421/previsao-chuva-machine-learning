from __future__ import annotations

import pandas as pd

from config import INMET_BH


def load_inmet_bh_daily() -> pd.DataFrame:
    raw = pd.read_csv(
        INMET_BH,
        sep=";",
        skiprows=8,
        encoding="latin-1",
        low_memory=False,
    )
    raw.columns = [str(c).strip() for c in raw.columns]
    dcol, hcol = raw.columns[0], raw.columns[1]
    hora = (
        raw[hcol]
        .astype(str)
        .str.replace("UTC", "", regex=False)
        .str.strip()
        .str.zfill(4)
    )
    dt = pd.to_datetime(
        raw[dcol].astype(str).str.strip() + " " + hora,
        format="%Y/%m/%d %H%M",
        utc=True,
    )
    raw = raw.assign(ts_utc=dt).dropna(subset=["ts_utc"])
    raw["day"] = pd.to_datetime(
        raw["ts_utc"].dt.tz_convert("America/Sao_Paulo").dt.date
    )
    meta = {dcol, hcol, "ts_utc", "day"}
    val_cols = [c for c in raw.columns if c not in meta]
    for c in val_cols:
        raw[c] = pd.to_numeric(
            raw[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    agg_spec: dict[str, str] = {val_cols[0]: "sum"}
    for c in val_cols[1:]:
        agg_spec[c] = "mean"
    daily = raw.groupby("day", as_index=True).agg(agg_spec)
    daily.columns = [f"inmet_{i}" for i in range(len(val_cols))]
    return daily.sort_index()

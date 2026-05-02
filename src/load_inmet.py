from __future__ import annotations

import pandas as pd

from config import INMET_BH

INMET_FEATURES: list[tuple[str | None, str | None]] = [
    ("inmet_precip", "sum"),
    ("inmet_pressao", "mean"),
    ("inmet_pressao_max", "mean"),
    ("inmet_pressao_min", "mean"),
    ("inmet_radiacao", "mean"),
    ("inmet_temp_ar", "mean"),
    ("inmet_temp_orvalho", "mean"),
    ("inmet_temp_max", "mean"),
    ("inmet_temp_min", "mean"),
    ("inmet_orvalho_max", "mean"),
    ("inmet_orvalho_min", "mean"),
    ("inmet_umidade_max", "mean"),
    ("inmet_umidade_min", "mean"),
    ("inmet_umidade", "mean"),
    (None, None),
    ("inmet_vento_rajada", "mean"),
    ("inmet_vento_vel", "mean"),
]


def load_inmet_bh_daily() -> pd.DataFrame:
    """Lê o CSV horário do INMET (encoding latin-1, vírgula decimal),
    converte para America/Sao_Paulo e agrega por dia civil.

    Decisões de coluna:
    - Direção do vento (índice 14) é descartada por ser variável circular
      (média aritmética de graus é incorreta); usar `nasa_U10M`/`nasa_V10M`.
    - A última coluna fantasma (gerada pelo `;` final do header) é removida.
    """
    raw = pd.read_csv(
        INMET_BH,
        sep=";",
        skiprows=8,
        encoding="latin-1",
        low_memory=False,
    )
    raw = raw.dropna(axis=1, how="all")
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
    if len(val_cols) != len(INMET_FEATURES):
        raise ValueError(
            f"INMET: esperava {len(INMET_FEATURES)} colunas de medidas, "
            f"encontrei {len(val_cols)}. Header pode ter mudado."
        )

    keep = {src: spec for src, spec in zip(val_cols, INMET_FEATURES) if spec[0]}
    for src in keep:
        raw[src] = pd.to_numeric(
            raw[src].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    agg_spec = {src: how for src, (_, how) in keep.items()}
    daily = raw.groupby("day", as_index=True).agg(agg_spec)
    daily.columns = [keep[c][0] for c in daily.columns]
    return daily.sort_index()

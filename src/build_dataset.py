from __future__ import annotations

import pandas as pd

from config import OUTPUT_DIR
from fetch_nasa import load_or_fetch_nasa_hourly
from load_cemaden import load_cemaden_station_daily
from load_inmet import load_inmet_bh_daily


def nasa_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Agrega séries horárias NASA por dia civil em America/Sao_Paulo."""
    h = hourly.copy()
    ts = pd.to_datetime(h["ts_utc"], utc=True).dt.tz_convert("America/Sao_Paulo")
    h["day"] = pd.to_datetime(ts.dt.date)
    skip = {"ts_utc", "day"}
    cols = [c for c in h.columns if c not in skip]
    out = h.groupby("day", as_index=False)[cols].mean()
    out.columns = ["day"] + [f"nasa_{c}" for c in cols]
    return out.set_index("day").sort_index()


def build_final_dataset() -> pd.DataFrame:
    cemaden = load_cemaden_station_daily()
    inmet = load_inmet_bh_daily()
    nasa_h = load_or_fetch_nasa_hourly()
    nasa_d = nasa_hourly_to_daily(nasa_h)
    nasa_d["nasa_PRECTOT_roll7"] = (
        nasa_d["nasa_PRECTOT"].rolling(7, min_periods=1).mean()
    )

    cemaden_feats = pd.DataFrame(
        {
            "cemaden_lag1": cemaden,
            "cemaden_lag2": cemaden.shift(1),
            "cemaden_lag3": cemaden.shift(2),
            "cemaden_roll3": cemaden.rolling(3, min_periods=1).mean(),
            "cemaden_roll7": cemaden.rolling(7, min_periods=1).mean(),
            "target_mm_next_day": cemaden.shift(-1),
        }
    )
    df = cemaden_feats.join(inmet, how="inner").join(nasa_d, how="inner")
    df = df.dropna(subset=["target_mm_next_day"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dataset_final.csv"
    df.reset_index(names=["day"]).to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    build_final_dataset()

from __future__ import annotations

import pandas as pd

from config import OUTPUT_DIR
from fetch_nasa import load_or_fetch_nasa_hourly
from load_cemaden import load_cemaden_station_daily
from load_inmet import load_inmet_bh_daily


def nasa_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["day"] = pd.to_datetime(h["ts"]).dt.normalize()
    skip = {"ts", "day"}
    cols = [c for c in h.columns if c not in skip]
    agg = {c: "mean" for c in cols}
    out = h.groupby("day", as_index=False).agg(agg)
    out.columns = ["day"] + [f"nasa_{c}" for c in cols]
    out["day"] = pd.to_datetime(out["day"]).dt.normalize()
    return out.set_index("day").sort_index()


def build_final_dataset() -> pd.DataFrame:
    cemaden = load_cemaden_station_daily()
    inmet = load_inmet_bh_daily()
    nasa_h = load_or_fetch_nasa_hourly()
    nasa_d = nasa_hourly_to_daily(nasa_h)
    nasa_d["nasa_PRECTOT_roll7"] = (
        nasa_d["nasa_PRECTOT"].rolling(7, min_periods=1).mean()
    )

    base = pd.DataFrame({"target_mm_next_day": cemaden.shift(-1)})
    df = base.join(inmet, how="inner").join(nasa_d, how="inner")
    df = df.dropna(subset=["target_mm_next_day"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dataset_final.csv"
    df.reset_index(names=["day"]).to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    build_final_dataset()

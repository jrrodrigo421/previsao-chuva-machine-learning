"""Previsão com modelo persistido (`train.py` sem novo fit)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import MODEL_METADATA_PATH, MODEL_PATH, OUTPUT_DIR
from persistence import carregar_metadados, carregar_pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gera pred_mm_next_day usando output/modelo_chuva.joblib."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV com colunas compatíveis (obrigatório: todas as features do metadata).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "previsoes.csv",
    )
    args = p.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Treine antes (train.py): não há {MODEL_PATH}")

    meta = carregar_metadados(MODEL_METADATA_PATH)
    cols: list[str] = meta["feature_columns"]
    df = pd.read_csv(args.input)
    if "day" in df.columns:
        df = df.assign(day=pd.to_datetime(df["day"]))

    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas exigidas pelo modelo: {sorted(missing)}")

    pipe = carregar_pipeline(MODEL_PATH)
    X = df[cols].astype(float).to_numpy()
    pred = pipe.predict(X)

    out_cols: dict[str, object] = {"pred_mm_next_day": pred}
    if "day" in df.columns:
        out_cols["day"] = df["day"].values
    pd.DataFrame(out_cols).to_csv(args.output, index=False)
    print(f"gravado {args.output.resolve()} ({len(pred)} linhas)")


if __name__ == "__main__":
    main()

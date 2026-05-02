from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from config import OUTPUT_DIR, REPORTS_DIR


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "dataset_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"Execute build_dataset primeiro: falta {path}")

    df = pd.read_csv(path, parse_dates=["day"])
    target = "target_mm_next_day"
    y = df[target].astype(float)
    X = df.drop(columns=[target, "day"], errors="ignore")

    null_report = X.isna().sum().sort_values(ascending=False)
    null_report.to_csv(REPORTS_DIR / "nulos_por_coluna.csv")
    X.dtypes.astype(str).to_csv(REPORTS_DIR / "tipos_colunas.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(y, kde=True, ax=ax, bins=40)
    ax.set_title("Distribuição da target (mm no dia seguinte)")
    ax.set_xlabel("target_mm_next_day")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "distribuicao_target.png", dpi=150)
    plt.close(fig)

    num = X.select_dtypes(include=[np.number]).copy()
    num = num.loc[:, ~num.isna().all()]
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlação entre variáveis numéricas")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "matriz_correlacao.png", dpi=150)
    plt.close(fig)

    tgt_corr = num.corrwith(y).sort_values(key=abs, ascending=False)
    tgt_corr.to_csv(REPORTS_DIR / "correlacao_com_target.csv")
    fig, ax = plt.subplots(figsize=(10, max(6, len(tgt_corr) * 0.12)))
    tgt_corr.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Correlação com a target")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "correlacao_target.png", dpi=150)
    plt.close(fig)

    feature_names = list(num.columns)
    X_arr = num.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)

    split = int(len(df) * 0.8)
    X_train, X_test = X_arr[:split], X_arr[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]

    gb = GradientBoostingRegressor(
        random_state=42,
        max_depth=4,
        learning_rate=0.06,
        n_estimators=200,
        subsample=0.9,
    )
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("gb", gb),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, pred)

    Xt_train = pipe.named_steps["imp"].transform(X_train)
    Xt_test = pipe.named_steps["imp"].transform(X_test)
    gb_fit: GradientBoostingRegressor = pipe.named_steps["gb"]
    staged_train = np.array(
        [
            mean_squared_error(y_train, p)
            for p in gb_fit.staged_predict(Xt_train)
        ]
    )
    staged_test = np.array(
        [mean_squared_error(y_test, p) for p in gb_fit.staged_predict(Xt_test)]
    )

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
        "n_train": int(split),
        "n_test": int(len(df) - split),
    }
    (REPORTS_DIR / "metricas_modelo.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "metricas_modelo.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(staged_train, label="MSE treino (por estágio)")
    ax.plot(staged_test, label="MSE teste (por estágio)")
    ax.set_xlabel("Estágio (árvore)")
    ax.set_ylabel("MSE")
    ax.set_title("Evolução do erro — Gradient Boosting (staged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "curva_erro_estagios.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.35, s=12)
    lim = max(float(np.max(y_test)), float(np.max(pred)), 1e-6)
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--")
    ax.set_xlabel("Observado (mm)")
    ax.set_ylabel("Previsto (mm)")
    ax.set_title("Teste: real vs previsto")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "real_vs_previsto.png", dpi=150)
    plt.close(fig)

    fi = pd.Series(gb_fit.feature_importances_, index=feature_names).sort_values(
        ascending=True
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(fi) * 0.11)))
    fi.plot(kind="barh", ax=ax, color="darkgreen")
    ax.set_title("Importância das features (Gradient Boosting)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "importancia_features.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

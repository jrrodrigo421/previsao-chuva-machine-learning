"""Treino, relatórios, persistência opcional do Pipeline e modo --eval-only."""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from config import MODEL_METADATA_PATH, MODEL_PATH, OUTPUT_DIR, REPORTS_DIR
from persistence import (
    carregar_metadados,
    carregar_pipeline,
    file_sha256,
    ler_backup_metadados,
    montar_metadados,
    salvar_pipeline,
)


TARGET = "target_mm_next_day"


def _regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    vy = float(np.var(y_true))
    r2_val: float | None
    if vy > 0:
        r2_val = float(r2_score(y_true, y_pred))
    else:
        r2_val = None
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2_val}


def _plot_bundle(
    y: pd.Series,
    num: pd.DataFrame,
    y_test: np.ndarray,
    pred: np.ndarray,
    staged_train: np.ndarray,
    staged_test: np.ndarray,
    gb_fit,
    feature_names: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(y, kde=True, ax=ax, bins=40)
    ax.set_title("Distribuição da target (mm no dia seguinte)")
    ax.set_xlabel(TARGET)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "distribuicao_target.png", dpi=150)
    plt.close(fig)

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


def _run_cv_benchmarks(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lag_idx: int,
    n_splits: int,
    random_state: int = 42,
) -> tuple[list[dict[str, object]], dict[str, float | None | int]]:
    tsc = TimeSeriesSplit(n_splits=n_splits)

    gb_cv = GradientBoostingRegressor(
        random_state=random_state,
        max_depth=4,
        learning_rate=0.06,
        n_estimators=200,
        subsample=0.9,
    )
    pipe_cv = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("gb", gb_cv),
        ]
    )

    rows_cv: list[dict[str, object]] = []
    for fold, (tr_ix, te_ix) in enumerate(tsc.split(X_arr, y_arr), start=1):
        X_tr, X_te = X_arr[tr_ix], X_arr[te_ix]
        y_tr, y_te = y_arr[tr_ix], y_arr[te_ix]
        pipe_cv.fit(X_tr, y_tr)
        pred_gb = pipe_cv.predict(X_te)
        m_tr = float(np.mean(y_tr))
        pred_fold_mean = np.full_like(y_te, fill_value=m_tr, dtype=float)
        pred_fold_persist = (
            X_te[:, lag_idx].astype(float)
            if lag_idx >= 0
            else np.zeros_like(y_te)
        )

        row: dict[str, object] = {
            "fold": fold,
            "n_train": len(tr_ix),
            "n_test": len(te_ix),
        }
        for key, vals in [
            ("model", _regression_metrics(y_te, pred_gb)),
            ("baseline_mean", _regression_metrics(y_te, pred_fold_mean)),
            (
                "baseline_persist",
                _regression_metrics(y_te, pred_fold_persist)
                if lag_idx >= 0
                else {},
            ),
        ]:
            for mname, mv in vals.items():
                row[f"{key}_{mname}"] = mv
        rows_cv.append(row)

    def _mean_std(metric_key: str) -> tuple[float | None, float | None]:
        vals_f = []
        for r in rows_cv:
            v = r.get(metric_key)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                vals_f.append(float(v))
        if not vals_f:
            return None, None
        arr = np.array(vals_f, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    cv_summary: dict[str, float | None | int] = {"n_splits": n_splits}
    for prefix in ("model", "baseline_mean", "baseline_persist"):
        for suf in ("MAE", "MSE", "RMSE", "R2"):
            key_out = f"cv_{prefix}_{suf}"
            m, s = _mean_std(f"{prefix}_{suf}")
            cv_summary[f"{key_out}_mean"] = m
            cv_summary[f"{key_out}_std"] = s

    return rows_cv, cv_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino / relatório de chuva dia+1.")
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Treina mas não grava modelo_chuva.joblib nem metadata.",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Carrega modelo salvo, regera métricas e gráficos (sem novo fit).",
    )
    p.add_argument(
        "--compare-prev",
        action="store_true",
        help="Depois do treino/gravação, imprime comparação com metadata .backup.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "dataset_final.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Execute build_dataset primeiro: falta {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["day"])
    y = df[TARGET].astype(float)
    X_raw = df.drop(columns=[TARGET, "day"], errors="ignore")

    null_report = X_raw.isna().sum().sort_values(ascending=False)
    null_report.to_csv(REPORTS_DIR / "nulos_por_coluna.csv")
    X_raw.dtypes.astype(str).to_csv(REPORTS_DIR / "tipos_colunas.csv")

    num_full = X_raw.select_dtypes(include=[np.number]).copy()
    num_full = num_full.loc[:, ~num_full.isna().all()]
    feat_default = list(num_full.columns)

    split = int(len(df) * 0.8)
    n_samples = len(y)

    rows_cv: list[dict[str, object]] | None = None
    eval_extra: dict[str, object] = {}
    cv_summary: dict[str, float | None | int] = {}

    if args.eval_only:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modo --eval-only mas não há modelo: {MODEL_PATH}. Treine primeiro."
            )
        meta = carregar_metadados(MODEL_METADATA_PATH)
        expected: list[str] = meta["feature_columns"]
        missing = set(expected) - set(df.columns)
        if missing:
            raise ValueError(
                f"--eval-only: faltam colunas no dataset VS modelo: "
                f"{sorted(missing)}"
            )
        pipe = carregar_pipeline(MODEL_PATH)

        df_X = df[expected].astype(float)
        feature_names = expected
        num = df_X
        X_arr = df_X.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        X_train, X_test = X_arr[:split], X_arr[split:]
        y_train, y_test = y_arr[:split], y_arr[split:]

        pred = pipe.predict(X_test)
        gb_fit = pipe.named_steps["gb"]

        Xt_train = pipe.named_steps["imp"].transform(X_train)
        Xt_test = pipe.named_steps["imp"].transform(X_test)
        staged_train = np.array(
            [
                mean_squared_error(y_train, p)
                for p in gb_fit.staged_predict(Xt_train)
            ]
        )
        staged_test = np.array(
            [mean_squared_error(y_test, p) for p in gb_fit.staged_predict(Xt_test)]
        )

        try:
            lag_idx = feature_names.index("cemaden_lag1")
        except ValueError:
            lag_idx = -1

        eval_extra = {
            "eval_only": True,
            "skipped_cv": True,
            "model_trained_sha256_saved": meta.get("dataset_sha256"),
            "trained_at_utc_saved_model": meta.get("trained_at_utc"),
        }
        cv_summary = {"n_splits_eval_only_skipped": 0}
    else:
        feature_names = feat_default
        num = num_full
        X_arr = num.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        X_train, X_test = X_arr[:split], X_arr[split:]
        y_train, y_test = y_arr[:split], y_arr[split:]

        try:
            lag_idx = feature_names.index("cemaden_lag1")
        except ValueError:
            lag_idx = -1

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

        Xt_train = pipe.named_steps["imp"].transform(X_train)
        Xt_test = pipe.named_steps["imp"].transform(X_test)
        gb_fit = pipe.named_steps["gb"]
        staged_train = np.array(
            [
                mean_squared_error(y_train, p)
                for p in gb_fit.staged_predict(Xt_train)
            ]
        )
        staged_test = np.array(
            [mean_squared_error(y_test, p) for p in gb_fit.staged_predict(Xt_test)]
        )

        n_splits = min(5, max(3, n_samples // 70))
        rows_cv, cv_summary = _run_cv_benchmarks(X_arr, y_arr, lag_idx, n_splits)

    gb_fit_for_plots = pipe.named_steps["gb"]
    y_mean_train = float(np.mean(y_train))
    pred_holdout_mean = np.full_like(y_test, fill_value=y_mean_train, dtype=float)
    pred_holdout_persist = (
        X_test[:, lag_idx].astype(float) if lag_idx >= 0 else np.zeros_like(y_test)
    )

    m_model = _regression_metrics(y_test, pred)
    m_mean = _regression_metrics(y_test, pred_holdout_mean)
    m_persist = (
        _regression_metrics(y_test, pred_holdout_persist)
        if lag_idx >= 0
        else {}
    )

    metrics: dict[str, object] = {
        "MAE": m_model["MAE"],
        "MSE": m_model["MSE"],
        "RMSE": m_model["RMSE"],
        "R2": m_model["R2"],
        "n_train": split,
        "n_test": int(len(df) - split),
        **eval_extra,
    }
    for k, v in m_mean.items():
        metrics[f"holdout_baseline_mean_{k}"] = v
    if lag_idx >= 0 and m_persist:
        for k, v in m_persist.items():
            metrics[f"holdout_baseline_persist_{k}"] = v
    metrics.update(cv_summary)

    (REPORTS_DIR / "metricas_modelo.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "metricas_modelo.csv", index=False)

    if rows_cv is not None:
        pd.DataFrame(rows_cv).to_csv(REPORTS_DIR / "metricas_cv_fold.csv", index=False)
    elif args.eval_only:
        pd.DataFrame(
            [{"note": "CV omitido em --eval-only (modelo pré-treinado fixo)."}]
        ).to_csv(REPORTS_DIR / "metricas_cv_fold.csv", index=False)

    _plot_bundle(
        y,
        num,
        y_test,
        pred,
        staged_train,
        staged_test,
        gb_fit_for_plots,
        feature_names,
    )

    ds_hash = file_sha256(csv_path)
    gb_hyper = pipe.named_steps["gb"].get_params()
    extra_meta = {
        "gradient_boosting_params": {
            k: gb_hyper[k]
            for k in (
                "n_estimators",
                "learning_rate",
                "max_depth",
                "subsample",
                "random_state",
            )
            if k in gb_hyper
        },
    }

    if not args.no_save and not args.eval_only:
        md = montar_metadados(
            dataset_path=csv_path,
            dataset_rows=len(df),
            dataset_sha256=ds_hash,
            target_column=TARGET,
            feature_columns=feature_names,
            metrics=metrics,
            sklearn_version=sklearn.__version__,
            extra_meta=extra_meta,
        )
        md["reports_metricas_relative"] = "reports/metricas_modelo.json"
        salvar_pipeline(pipe, MODEL_PATH, MODEL_METADATA_PATH, md)

    if args.compare_prev:
        prev = ler_backup_metadados(MODEL_METADATA_PATH)
        if prev is None:
            print(
                "compare-prev: não há modelo_chuva.metadata.json.backup.",
                file=sys.stderr,
            )
        else:
            pn = prev.get("metrics_holdout_summary", {})
            keys = ("MAE", "RMSE", "R2")
            lines = ["compare-prev (metadata backup -> métricas atuais no holdout):"]
            for k in keys:
                if k not in pn:
                    continue
                lines.append(f"  {k}: {pn[k]} -> {metrics.get(k)}")
            print("\n".join(lines))


if __name__ == "__main__":
    main()

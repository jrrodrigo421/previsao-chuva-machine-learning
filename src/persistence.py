"""Persistência do Pipeline (treino → disco) via joblib + metadados JSON."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _backup_file(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".backup"))


def salvar_pipeline(
    pipe: Pipeline,
    modelo_path: Path,
    meta_path: Path,
    metadata: dict[str, Any],
    compress: int = 3,
) -> None:
    modelo_path.parent.mkdir(parents=True, exist_ok=True)
    _backup_file(modelo_path)
    _backup_file(meta_path)
    joblib.dump(pipe, modelo_path, compress=compress)
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def carregar_pipeline(modelo_path: Path) -> Pipeline:
    return joblib.load(modelo_path)


def carregar_metadados(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def montar_metadados(
    *,
    dataset_path: Path,
    dataset_rows: int,
    dataset_sha256: str,
    target_column: str,
    feature_columns: list[str],
    metrics: dict[str, Any],
    sklearn_version: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "sklearn": sklearn_version,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": dataset_rows,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "metrics_holdout_summary": {
            k: metrics[k]
            for k in (
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "n_train",
                "n_test",
            )
            if k in metrics
        },
        "metrics_full_metrics_modelo_keys": sorted(metrics.keys()),
        **(extra_meta or {}),
    }


def ler_backup_metadados(meta_path: Path) -> dict[str, Any] | None:
    bk = meta_path.with_name(meta_path.name + ".backup")
    if not bk.exists():
        return None
    return json.loads(bk.read_text(encoding="utf-8"))

"""Executa construção do dataset e treino do modelo."""

from __future__ import annotations

from build_dataset import build_final_dataset
from train import main as train_main


def main() -> None:
    build_final_dataset()
    train_main()


if __name__ == "__main__":
    main()

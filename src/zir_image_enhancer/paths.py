"""Where things live: the dataset, the official starter repo and our local outputs.

Defaults match the team layout (dataset in ``data/``, starter repo in ``local/``).
Any of them can be overridden from ``.env`` or the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: str) -> Path:
    # Relative values are read from the repo root, so scripts work from any folder.
    path = Path(os.environ.get(name, default))
    return path if path.is_absolute() else _REPO_ROOT / path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    data_root: Path
    starter_root: Path
    runs_root: Path

    @classmethod
    def from_env(cls) -> ProjectPaths:
        # .env holds FAL_KEY and optional path overrides; it is ignored by Git.
        load_dotenv(_REPO_ROOT / ".env")
        return cls(
            repo_root=_REPO_ROOT,
            data_root=_env_path("ZIR_DATA_ROOT", "data/sartiq_neural_wave_2026"),
            starter_root=_env_path("ZIR_STARTER_ROOT", "local/neural-wave-2026"),
            runs_root=_env_path("ZIR_RUNS_ROOT", "local/runs"),
        )

    @property
    def holdout_val(self) -> Path:
        return self.starter_root / "holdout" / "val.jsonl"

    @property
    def holdout_train(self) -> Path:
        return self.starter_root / "holdout" / "train.jsonl"

    @property
    def lora_root(self) -> Path:
        """Datasets, training runs and weights of the fine-tuned packshot LoRA."""
        return self.repo_root / "local" / "lora"

    @property
    def lora_registry(self) -> Path:
        """The active adapter (written by scripts/lora/train_fal.py or register_lora.py)."""
        return self.lora_root / "registry.json"

    @property
    def fal_cache(self) -> Path:
        return self.repo_root / "local" / "fal_cache"

    @property
    def fal_ledger(self) -> Path:
        return self.repo_root / "local" / "fal_ledger.jsonl"

    @property
    def experiments_csv(self) -> Path:
        return self.repo_root / "experiments" / "results.csv"

    def scorer_python(self) -> Path:
        """Interpreter of the starter repo's venv, where the official scorer is installed."""
        # Windows venvs keep the interpreter in Scripts/, everything else in bin/.
        for relative in (".venv/Scripts/python.exe", ".venv/bin/python"):
            candidate = self.starter_root / relative
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No scorer venv in {self.starter_root}. Follow the starter README to create it."
        )

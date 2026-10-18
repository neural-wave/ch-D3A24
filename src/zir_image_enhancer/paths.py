"""Where the pipeline keeps its local files: the fal cache, the cost ledger and app runs.

Everything lives under ``local/``, which Git ignores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path

    @classmethod
    def from_env(cls) -> ProjectPaths:
        # .env holds FAL_KEY and the budget limit; it is ignored by Git.
        load_dotenv(_REPO_ROOT / ".env")
        return cls(repo_root=_REPO_ROOT)

    @property
    def local(self) -> Path:
        return self.repo_root / "local"

    @property
    def fal_cache(self) -> Path:
        """Every generated image, keyed by the request, so the same call is never paid twice."""
        return self.local / "fal_cache"

    @property
    def vision_cache(self) -> Path:
        """Every spec and gate answer, keyed by the prompt and the image bytes."""
        return self.local / "reconstruction_cache"

    @property
    def fal_ledger(self) -> Path:
        """One line per paid call; the budget limit is checked against its total."""
        return self.local / "fal_ledger.jsonl"

    @property
    def app_runs(self) -> Path:
        return self.local / "app_runs"

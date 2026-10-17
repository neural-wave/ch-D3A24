"""Studio pipeline: phone photos of one product in, one studio packshot out.

``StudioPipeline`` is the single seam shared by the Gradio app (``app.py``) and the batch
runner (``cli.py`` / ``zir-run``). It runs the reconstruction harness: garment spec (vision),
studio prompt with geometry, flattening and details, named detail photos, quality gate with
one repair, and an optional relight pass.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction.photos import Role, role_from_filename
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionResult,
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.vision import VisionClient

try:  # iPhone photos arrive as HEIC; the opener is optional.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - only when pillow-heif is absent
    pass

#: The view name the challenge expects in ``submissions/<team>/<id>/<view>.<ext>``.
DEFAULT_VIEW = "front"


class MissingKeyError(RuntimeError):
    """FAL_KEY is not set, so nothing can be generated."""


@dataclass(frozen=True)
class StudioResult:
    image: Path
    front: Path  # the photo the spec used as the front, for before/after views
    log: tuple[str, ...]
    cost_usd: float
    seconds: float


class StudioRunner(Protocol):
    def run(
        self, photos: Sequence[Path], name: str, roles: Sequence[str] = (), relight: bool = False
    ) -> StudioResult: ...


class StudioPipeline:
    def __init__(
        self,
        settings: ReconstructionSettings | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        self._paths = paths or ProjectPaths.from_env()
        if not os.environ.get("FAL_KEY"):
            raise MissingKeyError("FAL_KEY is not set. Put it in .env at the repo root.")
        self._settings = settings or ReconstructionSettings()
        budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", 70.0))
        ledger = CostLedger(self._paths.fal_ledger)
        cache = self._paths.repo_root / "local" / "reconstruction_cache"
        self._generator = FalGenerator(
            cache_dir=self._paths.fal_cache, ledger=ledger, budget_usd=budget
        )
        self._vision = VisionClient(ledger, budget, cache)
        self._work_root = self._paths.repo_root / "local" / "app_runs"

    def run(
        self, photos: Sequence[Path], name: str, roles: Sequence[str] = (), relight: bool = False
    ) -> StudioResult:
        """``roles`` optionally names each photo ("front", "back", "detail chest logo")."""
        if not photos:
            raise ValueError("no photos given")
        workdir = self._work_root / name
        inputs = prepare_photos(photos, workdir / "inputs", roles)
        settings = replace(self._settings, relight=relight)
        reconstructor = Reconstructor(settings, self._generator, self._vision)
        started = time.perf_counter()
        result = reconstructor.run(GarmentItem(name, inputs, None), workdir / "attempts")
        return StudioResult(
            result.final,
            result.spec.front,
            describe(result),
            result.cost_usd,
            time.perf_counter() - started,
        )


def prepare_photos(
    photos: Sequence[Path], folder: Path, roles: Sequence[str] = ()
) -> tuple[Path, ...]:
    """Upright RGB JPEG copies, named so the photo roles survive (01_front, 03_detail_logo).

    Phone photos come as HEIC, rotated by EXIF, or with an alpha channel; the edit model
    reads none of that reliably, so every photo is normalised once here.
    """
    folder.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, photo in enumerate(photos, start=1):
        role = roles[index - 1].strip() if index - 1 < len(roles) else ""
        if role:
            stem = f"{index:02d}_{_slug(role)}"
        elif role_from_filename(Path(photo)).role is not Role.UNKNOWN:
            # Already named by the user (01_front.jpg, 03_detail_chest-logo.jpg).
            stem = Path(photo).stem
        else:
            stem = f"{index:02d}_{_slug(Path(photo).stem)}"
        target = folder / f"{stem}.jpg"
        ImageOps.exif_transpose(Image.open(photo)).convert("RGB").save(target, quality=95)
        prepared.append(target)
    return tuple(prepared)


def describe(result: ReconstructionResult) -> tuple[str, ...]:
    """What the pipeline understood and decided, in plain lines for the app and the logs."""
    spec = result.spec
    lines = [
        f"Front photo: {spec.front.name}",
        f"Product: {spec.category}" + (f" ({spec.construction})" if spec.construction else ""),
    ]
    if len(spec.pieces) > 1:
        lines.append(f"Pieces: {', '.join(spec.pieces)}")
    if spec.photo_setup:
        lines.append(f"Photo setup: {spec.photo_setup}")
    if spec.details:
        lines.append("Details to keep: " + "; ".join(d.name for d in spec.details))
    if spec.used_fallback:
        lines.append(f"Spec unavailable, studio prompt only ({spec.note})")
    for number, attempt in enumerate(result.attempts, start=1):
        if attempt.gate is None:
            verdict = "not checked"
        elif attempt.gate.problems:
            verdict = "problems: " + ", ".join(attempt.gate.problems)
        else:
            verdict = "passed"
        lines.append(f"Attempt {number} ({attempt.image.stem}): {verdict}")
    lines.append(f"Kept: {result.final.stem}")
    lines.append(f"Cost: ${result.cost_usd:.3f}, time {result.seconds:.0f} s")
    return tuple(lines)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "photo"

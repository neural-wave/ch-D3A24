"""Studio pipeline: phone photos of one product in, one studio packshot out.

``StudioPipeline`` is the single seam shared by the Gradio app (``app.py``) and the batch
runner (``cli.py`` / ``zir-run``): it prepares the photos and runs the reconstructor
(``reconstruction/reconstructor.py``) in a working folder under ``local/app_runs/``.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator, Generator
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

_EXIF_ORIENTATION = 0x0112


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
    def run(self, photos: Sequence[Path], name: str, roles: Sequence[str] = ()) -> StudioResult: ...


class StudioPipeline:
    def __init__(
        self,
        settings: ReconstructionSettings | None = None,
        paths: ProjectPaths | None = None,
        *,
        generator: Generator | None = None,
        vision: VisionClient | None = None,
    ) -> None:
        """``generator`` and ``vision`` replace the hosted models (for tests); by default
        they are built from FAL_KEY."""
        self._paths = paths or ProjectPaths.from_env()
        if generator is None and not os.environ.get("FAL_KEY"):
            raise MissingKeyError("FAL_KEY is not set. Put it in .env at the repo root.")
        self._settings = settings or ReconstructionSettings()
        budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", 70.0))
        ledger = CostLedger(self._paths.fal_ledger)
        self._generator = generator or FalGenerator(
            cache_dir=self._paths.fal_cache, ledger=ledger, budget_usd=budget
        )
        self._vision = vision or VisionClient(ledger, budget, self._paths.vision_cache)

    def run(self, photos: Sequence[Path], name: str, roles: Sequence[str] = ()) -> StudioResult:
        """``roles`` optionally names each photo ("front", "back", "detail chest logo")."""
        if not photos:
            raise ValueError("no photos given")
        workdir = self._paths.app_runs / name
        inputs = prepare_photos(photos, workdir / "inputs", roles)
        reconstructor = Reconstructor(self._settings, self._generator, self._vision)
        result = reconstructor.run(inputs, workdir / "attempts")
        return StudioResult(
            result.final, result.spec.front, describe(result), result.cost_usd, result.seconds
        )


def prepare_photos(
    photos: Sequence[Path], folder: Path, roles: Sequence[str] = ()
) -> tuple[Path, ...]:
    """Upright RGB JPEG copies, named so the photo roles survive (01_front, 03_detail_logo).

    Phone photos come as HEIC, rotated by EXIF, or with an alpha channel; the edit model
    reads none of that reliably, so those are converted once here. A photo that already is an
    upright RGB JPEG is copied byte for byte: saving it again would lose a little detail and
    change the cache key of every call made with it.
    """
    folder.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, photo in enumerate(photos, start=1):
        photo = Path(photo)
        role = roles[index - 1].strip() if index - 1 < len(roles) else ""
        if role:
            stem = f"{index:02d}_{_slug(role)}"
        elif role_from_filename(photo).role is not Role.UNKNOWN:
            # Already named by the user (01_front.jpg, 03_detail_chest-logo.jpg).
            stem = photo.stem
        else:
            stem = f"{index:02d}_{_slug(photo.stem)}"
        target = folder / f"{stem}.jpg"
        with Image.open(photo) as image:
            ready = (
                image.format == "JPEG"
                and image.mode == "RGB"
                and image.getexif().get(_EXIF_ORIENTATION, 1) == 1
            )
            if not ready:
                ImageOps.exif_transpose(image).convert("RGB").save(target, quality=95)
        if ready:
            shutil.copyfile(photo, target)
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
        problems = attempt.gate.problems
        verdict = ("problems: " + ", ".join(problems)) if problems else "passed"
        lines.append(f"Attempt {number} ({attempt.image.stem}): {verdict}")
    lines.append(f"Kept: {result.final.stem}")
    lines.append(f"Cost: ${result.cost_usd:.3f}, time {result.seconds:.0f} s")
    return tuple(lines)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "photo"

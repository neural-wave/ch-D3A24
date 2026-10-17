"""Studio pipeline: phone photos of one product in, one studio packshot out.

``StudioPipeline`` is the single seam shared by the Gradio app (``app.py``) and the batch
runner (``cli.py`` / ``zir-run``). It runs the reconstruction harness in one of its three
versions (v3 "jury" by default: one-shot, jury gate, then an edit or a redraw only for the
garments with defects; see ``reconstruction/reconstructor.py``), or v4: the nine versions of
``selection.V4_VERSIONS``, ranked by the embedding experts and checked by the gate.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator, Generator
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction.photos import Role, role_from_filename, roles_from_filenames
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionResult,
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.spec import GarmentSpec, extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient
from zir_image_enhancer.selection import (
    POOL_SCALES,
    V4_VERSIONS,
    Candidate,
    Pick,
    Version,
    VersionSelector,
    gate_pick,
)

try:  # iPhone photos arrive as HEIC; the opener is optional.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - only when pillow-heif is absent
    pass

#: The view name the challenge expects in ``submissions/<team>/<id>/<view>.<ext>``.
DEFAULT_VIEW = "front"
#: The mode name of v4; the other modes are the reconstructor's (oneshot, harness, jury).
V4_MODE = "v4"
#: How many of the best ranked v4 candidates the gate may check, as in the batch run.
GATE_TOP = 3


class MissingKeyError(RuntimeError):
    """FAL_KEY is not set, so nothing can be generated."""


class MissingExtraError(RuntimeError):
    """v4 needs the local embedding models of the ``select`` extra (torch, CLIP)."""


@dataclass(frozen=True)
class StudioResult:
    image: Path
    front: Path  # the photo the spec used as the front, for before/after views
    log: tuple[str, ...]
    cost_usd: float
    seconds: float


class StudioRunner(Protocol):
    def run(
        self,
        photos: Sequence[Path],
        name: str,
        roles: Sequence[str] = (),
        mode: str | None = None,
    ) -> StudioResult: ...


class Ranker(Protocol):
    def rank(
        self, items: Sequence[GarmentItem], candidates: Mapping[str, Sequence[Candidate]]
    ) -> dict[str, list[Candidate]]: ...


class StudioPipeline:
    def __init__(
        self,
        settings: ReconstructionSettings | None = None,
        paths: ProjectPaths | None = None,
        *,
        generator: Generator | None = None,
        vision: VisionClient | None = None,
        selector: Ranker | None = None,
    ) -> None:
        """``generator``, ``vision`` and ``selector`` replace the real services (for tests);
        by default they are built from FAL_KEY and, for v4, on its first run."""
        self._paths = paths or ProjectPaths.from_env()
        if generator is None and not os.environ.get("FAL_KEY"):
            raise MissingKeyError("FAL_KEY is not set. Put it in .env at the repo root.")
        self._settings = settings or ReconstructionSettings()
        budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", 70.0))
        ledger = CostLedger(self._paths.fal_ledger)
        cache = self._paths.repo_root / "local" / "reconstruction_cache"
        self._generator = generator or FalGenerator(
            cache_dir=self._paths.fal_cache, ledger=ledger, budget_usd=budget
        )
        self._vision = vision or VisionClient(ledger, budget, cache)
        self._selector = selector
        self._work_root = self._paths.repo_root / "local" / "app_runs"

    def run(
        self,
        photos: Sequence[Path],
        name: str,
        roles: Sequence[str] = (),
        mode: str | None = None,
    ) -> StudioResult:
        """``roles`` optionally names each photo ("front", "back", "detail chest logo");
        ``mode`` picks the version (oneshot, harness, jury, v4) instead of the default."""
        if not photos:
            raise ValueError("no photos given")
        workdir = self._work_root / name
        item = GarmentItem(name, prepare_photos(photos, workdir / "inputs", roles), None)
        started = time.perf_counter()
        if mode == V4_MODE:
            return self._run_v4(item, workdir, started)
        settings = replace(self._settings, mode=mode) if mode else self._settings
        result = Reconstructor(settings, self._generator, self._vision).run(
            item, workdir / "attempts"
        )
        return StudioResult(
            result.final,
            result.spec.front,
            describe(result),
            result.cost_usd,
            time.perf_counter() - started,
        )

    def _run_v4(self, item: GarmentItem, workdir: Path, started: float) -> StudioResult:
        selector = self._get_selector()  # loads the local models before paying for anything
        # The nine versions read the same spec: asking once first lets them all hit the cache.
        spec = extract_spec(roles_from_filenames(item.inputs), self._vision)
        results: dict[str, ReconstructionResult] = {}
        failures: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=len(V4_VERSIONS)) as pool:
            futures = {
                version.name: pool.submit(
                    self._run_version, item, workdir / "v4" / version.name, version
                )
                for version in V4_VERSIONS
            }
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as error:  # noqa: BLE001
                    # One refused version (fal's content filter) should not lose the other eight.
                    failures[name] = error
        if not results:
            raise next(iter(failures.values()))

        candidates = [Candidate(name, result.final) for name, result in results.items()]
        ranking = selector.rank([item], {item.item_id: candidates})[item.item_id]
        pick = gate_pick(ranking, spec, self._vision, GATE_TOP)
        # Cached calls cost nothing, so the shared spec is counted once, here.
        cost = spec.cost_usd + sum(r.cost_usd for r in results.values()) + pick.cost_usd
        seconds = time.perf_counter() - started
        log = describe_v4(spec, ranking, pick, failures, cost, seconds)
        return StudioResult(pick.chosen.image, spec.front, log, cost, seconds)

    def _run_version(
        self, item: GarmentItem, workdir: Path, version: Version
    ) -> ReconstructionResult:
        settings = replace(
            self._settings,
            mode=version.mode,
            studio_style=version.style,
            seed=version.seed,
            # The one-shots are not checked one by one: the gate checks the best ranked ones.
            gate=version.mode != "oneshot",
        )
        return Reconstructor(settings, self._generator, self._vision).run(item, workdir)

    def _get_selector(self) -> Ranker:
        if self._selector is None:
            try:
                from zir_image_enhancer.embeddings import EmbeddingExperts
            except ImportError as error:
                raise MissingExtraError(
                    "v4 needs the local embedding models. Start the app with "
                    "`uv run --extra select zir-app`."
                ) from error
            # Fixed scales: one garment alone has to rank as it would inside the batch.
            self._selector = VersionSelector(EmbeddingExperts(self._paths), scales=POOL_SCALES)
        return self._selector


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
    lines = list(_spec_lines(result.spec))
    for number, attempt in enumerate(result.attempts, start=1):
        if attempt.gate is None:
            verdict = "not checked"
        else:
            verdict = _verdict(attempt.gate.problems)
        lines.append(f"Attempt {number} ({attempt.image.stem}): {verdict}")
    lines.append(f"Kept: {result.final.stem}")
    lines.append(f"Cost: ${result.cost_usd:.3f}, time {result.seconds:.0f} s")
    return tuple(lines)


def describe_v4(
    spec: GarmentSpec,
    ranking: Sequence[Candidate],
    pick: Pick,
    failures: Mapping[str, Exception],
    cost_usd: float,
    seconds: float,
) -> tuple[str, ...]:
    """The v4 decision in plain lines: versions made, style ranking, gate checks, the pick."""
    lines = list(_spec_lines(spec))
    lines.append(f"Versions generated: {len(ranking)} of {len(V4_VERSIONS)}")
    for name, error in failures.items():
        lines.append(f"Version {name} failed: {type(error).__name__}: {error}")
    lines.append("Style ranking: " + ", ".join(c.version for c in ranking))
    for check in pick.checks:
        lines.append(f"Jury check of {check.version}: {_verdict(check.problems)}")
    lines.append(f"Kept: {pick.chosen.version}")
    lines.append(f"Cost: ${cost_usd:.3f}, time {seconds:.0f} s")
    return tuple(lines)


def _spec_lines(spec: GarmentSpec) -> tuple[str, ...]:
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
    return tuple(lines)


def _verdict(problems: Sequence[str]) -> str:
    return ("problems: " + ", ".join(problems)) if problems else "passed"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "photo"

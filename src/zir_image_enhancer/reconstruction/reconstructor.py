"""One garment through the reconstruction pipeline: spec, references, generation, gate.

photos -> spec + input assessment (vision) -> references (front, pipeline-picked details)
       -> generate (nano-banana) -> quality gate -> one repair if it fails
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult, Generator
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.reconstruction.gate import GateResult, check
from zir_image_enhancer.reconstruction.photos import Role, unlabelled
from zir_image_enhancer.reconstruction.preprocess import (
    PreparedReferences,
    PreprocessSettings,
    prepare_references,
)
from zir_image_enhancer.reconstruction.prompt import build_harness_prompt
from zir_image_enhancer.reconstruction.references import (
    Reference,
    closeup_label,
    detail_references,
)
from zir_image_enhancer.reconstruction.spec import GarmentSpec, extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient


@dataclass(frozen=True)
class ReconstructionSettings:
    model_key: str = "nano-banana-2.1-portrait"
    use_detail_photos: bool = True
    gate: bool = True
    max_references: int = 4
    seed: int = 0
    #: OpenCV preprocessing of the generation references (see ``reconstruction.preprocess``).
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)


@dataclass(frozen=True)
class Attempt:
    prompt: str
    references: tuple[str, ...]
    image: Path
    gate: GateResult | None
    cost_usd: float


@dataclass(frozen=True)
class ReconstructionResult:
    final: Path
    spec: GarmentSpec
    attempts: tuple[Attempt, ...]
    seconds: float
    cost_usd: float = field(default=0.0)
    #: Capture statistics of the front photo, recorded when preprocessing ran.
    preprocess_stats: Mapping[str, float] = field(default_factory=dict)


class Reconstructor:
    def __init__(
        self,
        settings: ReconstructionSettings,
        generator: Generator,
        vision: VisionClient,
    ) -> None:
        self._settings = settings
        self._generator = generator
        self._vision = vision
        self._model = get_model(settings.model_key)

    def run(self, item: GarmentItem, workdir: Path) -> ReconstructionResult:
        started = time.perf_counter()
        spec = extract_spec(unlabelled(item.inputs), self._vision)
        workdir.mkdir(parents=True, exist_ok=True)
        references, stats = self._references(spec, workdir)

        first = self._attempt(spec, references, workdir, sample=0, fix="")
        attempts = [first]
        final = first
        if first.gate is not None and first.gate.problems:
            fix = first.gate.fix or "; ".join(first.gate.problems)
            repair = self._attempt(spec, references, workdir, sample=1, fix=fix)
            attempts.append(repair)
            # Keep the repair unless the gate finds more problems (severe ones count double).
            if repair.gate is None or _weight(repair.gate) <= _weight(first.gate):
                final = repair

        cost = spec.cost_usd + sum(
            a.cost_usd + (a.gate.cost_usd if a.gate else 0.0) for a in attempts
        )
        return ReconstructionResult(
            final.image,
            spec,
            tuple(attempts),
            time.perf_counter() - started,
            cost,
            preprocess_stats=stats,
        )

    def _references(
        self, spec: GarmentSpec, workdir: Path
    ) -> tuple[tuple[Reference, ...], Mapping[str, float]]:
        """The front photo and the detail close-ups sent to the model.

        With preprocessing on, the front and the detail photos are normalised, and every spec
        detail with a box is cropped from the enhanced front photo at full resolution so the
        model can read small logos and text. Without it the original photos are sent as before.
        """
        preprocess = self._settings.preprocess
        if not preprocess.enabled:
            references = [Reference(spec.front, "the whole garment from the front")]
            if self._settings.use_detail_photos:
                references += detail_references(spec, self._settings.max_references - 1)
            return tuple(references), {}

        detail_photos: tuple[tuple[str, Path], ...] = ()
        if self._settings.use_detail_photos:
            detail_photos = tuple(
                (photo.name, photo.path) for photo in spec.photos_with(Role.DETAIL)
            )
        prepared: PreparedReferences = prepare_references(
            front=spec.front,
            out_dir=workdir / "references",
            settings=preprocess,
            detail_photos=detail_photos,
            details=tuple((detail.name, detail.box) for detail in spec.details),
        )
        references = [Reference(prepared.front, "the whole garment from the front")]
        references += [
            Reference(path, closeup_label(name)) for name, path in prepared.closeups
        ]
        return tuple(references[: self._settings.max_references]), prepared.stats

    def _attempt(
        self,
        spec: GarmentSpec,
        references: tuple[Reference, ...],
        workdir: Path,
        sample: int,
        fix: str,
    ) -> Attempt:
        prompt = build_harness_prompt(spec, references, fix)
        request = GenerationRequest(
            self._model,
            prompt,
            tuple(ref.path for ref in references),
            self._settings.seed + sample,
            sample,
        )
        result: GenerationResult = self._generator.generate(request)
        image = workdir / f"attempt_{sample + 1}.png"
        image.write_bytes(result.png_bytes)
        gate = check(spec.front, image, spec, self._vision) if self._settings.gate else None
        return Attempt(prompt, tuple(str(r.path) for r in references), image, gate, result.cost_usd)


def _weight(gate: GateResult) -> int:
    return len(gate.problems) + len(gate.severe)

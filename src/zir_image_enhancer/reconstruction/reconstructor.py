"""One garment through the reconstruction pipeline: spec, references, generation, gate.

photos -> spec (vision) -> references (front, named detail photos)
       -> generate (nano-banana) -> quality gate -> one repair if it fails
       -> optional relight pass (same model, packshot as the only reference)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult, Generator
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.generation.prompts import build_prompt
from zir_image_enhancer.reconstruction.gate import GateResult, check
from zir_image_enhancer.reconstruction.photos import roles_from_filenames
from zir_image_enhancer.reconstruction.prompt import (
    build_harness_prompt,
    build_relight_prompt,
)
from zir_image_enhancer.reconstruction.references import Reference, named_detail_references
from zir_image_enhancer.reconstruction.spec import GarmentSpec, extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient


@dataclass(frozen=True)
class ReconstructionSettings:
    model_key: str = "nano-banana-2.1-portrait"
    use_detail_photos: bool = True
    gate: bool = True
    max_references: int = 4
    seed: int = 0
    # "harness": the studio prompt plus geometry, studio light and details from the spec, and
    # the detail photos the user named. "style": studio prompt and front photo only, spec only
    # for the gate (-0.37 on the holdout). The fully spec-written prompt (-3.0) was removed.
    prompt_mode: str = "harness"
    # Second pass that only changes the light. In one edit the model keeps the light of the
    # phone photo; once the shape is done, a packshot in, packshot out edit has less to change.
    relight: bool = False


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
        spec = extract_spec(roles_from_filenames(item.inputs), self._vision)
        references: tuple[Reference, ...] = (
            Reference(spec.front, "the whole garment from the front"),
        )
        if self._settings.prompt_mode == "harness" and self._settings.use_detail_photos:
            references += named_detail_references(spec, self._settings.max_references - 1)

        workdir.mkdir(parents=True, exist_ok=True)
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
        if self._settings.relight:
            relit = self._relight(spec, final, workdir)
            attempts.append(relit)
            # The relight must not break what the gate already accepted (shape, details).
            if (
                relit.gate is None
                or final.gate is None
                or _weight(relit.gate) <= _weight(final.gate)
            ):
                final = relit

        cost = spec.cost_usd + sum(
            a.cost_usd + (a.gate.cost_usd if a.gate else 0.0) for a in attempts
        )
        return ReconstructionResult(
            final.image, spec, tuple(attempts), time.perf_counter() - started, cost
        )

    def _attempt(
        self,
        spec: GarmentSpec,
        references: tuple[Reference, ...],
        workdir: Path,
        sample: int,
        fix: str,
    ) -> Attempt:
        if self._settings.prompt_mode == "style":
            prompt = build_prompt("studio-style-v2", len(references))
            if fix:
                prompt += f" A previous attempt had these problems; fix them: {fix}"
        else:
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

    def _relight(self, spec: GarmentSpec, final: Attempt, workdir: Path) -> Attempt:
        prompt = build_relight_prompt()
        sample = 2
        request = GenerationRequest(
            self._model, prompt, (final.image,), self._settings.seed + sample, sample
        )
        result = self._generator.generate(request)
        image = workdir / "attempt_relight.png"
        image.write_bytes(result.png_bytes)
        # Gate against the real photo, not against the packshot it came from.
        gate = check(spec.front, image, spec, self._vision) if self._settings.gate else None
        return Attempt(prompt, (str(final.image),), image, gate, result.cost_usd)


def _weight(gate: GateResult) -> int:
    return len(gate.problems) + len(gate.severe)

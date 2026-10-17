"""One garment through the reconstruction pipeline: spec, references, generation, gate.

photos -> spec (vision) -> references (front, named detail photos)
       -> generate (nano-banana) -> quality gate -> one repair if it fails

Three versions, kept side by side so they can be compared (holdout scores in the README):
- "oneshot" (v1): studio prompt and the front photo, one generation, gate only reported.
- "harness" (v2): studio prompt plus the spec's proportions, details and sets, with the named
  detail photos; any gate problem gets one repair.
- "jury" (v3): the one-shot first, because it scores best on the metric. When the gate finds
  a defect the jury looks at, a detail (logo, text, closures, seams, texture) is fixed by
  editing the one-shot packshot, and a global one (colour, proportions, view, crop) by
  drawing it again with the harness. The one with fewer defects is kept (a tie keeps the
  one-shot).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult, Generator
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.generation.prompts import build_studio_prompt
from zir_image_enhancer.reconstruction.gate import GateResult, check
from zir_image_enhancer.reconstruction.photos import roles_from_filenames
from zir_image_enhancer.reconstruction.prompt import build_edit_repair_prompt, build_harness_prompt
from zir_image_enhancer.reconstruction.references import Reference, named_detail_references
from zir_image_enhancer.reconstruction.spec import GarmentSpec, extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient

MODES = ("oneshot", "harness", "jury")

# Problems that affect the whole image. In v3 they are repaired by drawing the garment again;
# the others (logos, text, closures, seams, texture, details) by editing the one-shot packshot.
# On the holdout, editing kept the shape and fixed details (t0087 +0.8, t0327 +0.9) but could
# not fix a colour (t0314 -4.1, where regenerating gave +10.4).
GLOBAL_PROBLEMS = (
    "different garment",
    "not the front",
    "colour changed",
    "proportions differ",
    "lighting not studio",
    "extra items",
    "missing piece",
    "cropped",
)


@dataclass(frozen=True)
class ReconstructionSettings:
    model_key: str = "nano-banana-2.1-portrait"
    use_detail_photos: bool = True
    gate: bool = True
    max_references: int = 4
    seed: int = 0
    mode: str = "jury"  # "oneshot", "harness" or "jury"; see the module docstring


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
        if settings.mode not in MODES:
            raise ValueError(f"Unknown mode '{settings.mode}'. Choose one of: {', '.join(MODES)}")
        self._model = get_model(settings.model_key)

    def run(self, item: GarmentItem, workdir: Path) -> ReconstructionResult:
        started = time.perf_counter()
        spec = extract_spec(roles_from_filenames(item.inputs), self._vision)
        references: tuple[Reference, ...] = (
            Reference(spec.front, "the whole garment from the front"),
        )
        if self._settings.use_detail_photos:
            references += named_detail_references(spec, self._settings.max_references - 1)

        workdir.mkdir(parents=True, exist_ok=True)
        front_only = references[:1]
        if self._settings.mode == "oneshot":
            attempts = [self._attempt(spec, front_only, workdir, 0, build_studio_prompt())]
            final = attempts[0]
        elif self._settings.mode == "jury":
            first = self._attempt(spec, front_only, workdir, 0, build_studio_prompt())
            attempts, final = [first], first
            if first.gate is not None and first.gate.problems:
                fix = first.gate.fix or "; ".join(first.gate.problems)
                if any(problem in GLOBAL_PROBLEMS for problem in first.gate.problems):
                    # Colour, shape or view is wrong everywhere: draw it again from the photo.
                    prompt = build_harness_prompt(spec, references, fix)
                    repair = self._attempt(spec, references, workdir, 1, prompt)
                else:
                    # Only details are wrong: edit the one-shot packshot (image 1) so shape,
                    # colour and framing stay, with the real photos as the reference.
                    edit_refs = (Reference(first.image, "the packshot to edit"), *references)
                    prompt = build_edit_repair_prompt(references, fix)
                    repair = self._attempt(spec, edit_refs, workdir, 1, prompt)
                attempts.append(repair)
                # Strictly fewer defects: on a tie the one-shot stays, it scores better.
                if repair.gate is not None and _weight(repair.gate) < _weight(first.gate):
                    final = repair
        else:
            first = self._attempt(
                spec, references, workdir, 0, build_harness_prompt(spec, references)
            )
            attempts, final = [first], first
            if first.gate is not None and first.gate.problems:
                fix = first.gate.fix or "; ".join(first.gate.problems)
                prompt = build_harness_prompt(spec, references, fix)
                repair = self._attempt(spec, references, workdir, 1, prompt)
                attempts.append(repair)
                # Keep the repair unless the gate finds more problems (severe ones count double).
                if repair.gate is None or _weight(repair.gate) <= _weight(first.gate):
                    final = repair

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
        prompt: str,
    ) -> Attempt:
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

"""One garment through the pipeline: spec, one-shot, jury gate, and a repair only if needed.

photos -> spec (vision) -> one-shot packshot (nano-banana) -> jury gate (vision)
       -> no defect: done
       -> a detail is wrong (logo, text, closures, seams, texture): edit the packshot
       -> the whole image is wrong (colour, proportions, view, crop): draw it again
       -> keep the one with fewer defects (a tie keeps the one-shot)

The one-shot goes first because it scores best on the metric; the repair only steps in for
the defects the jury looks at.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult, Generator
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.generation.prompts import build_studio_prompt
from zir_image_enhancer.reconstruction.gate import GateResult, check
from zir_image_enhancer.reconstruction.photos import roles_from_filenames
from zir_image_enhancer.reconstruction.prompt import build_edit_prompt, build_redraw_prompt
from zir_image_enhancer.reconstruction.references import Reference, named_detail_references
from zir_image_enhancer.reconstruction.spec import GarmentSpec, extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient

# Problems that affect the whole image, repaired by drawing the garment again; the others
# (logos, text, closures, seams, texture, details) by editing the one-shot packshot. On the
# holdout, editing kept the shape and fixed details (t0087 +0.8, t0327 +0.9) but could not
# fix a colour (t0314 -4.1, where drawing it again gave +10.4).
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

# The front photo plus up to three named close-ups go with a repair.
_MAX_REFERENCES = 4


@dataclass(frozen=True)
class ReconstructionSettings:
    model_key: str = "nano-banana-2.1-portrait"
    seed: int = 0
    # "flat" (laid flat, from above), "ghost" (invisible mannequin) or "auto": ghost only when
    # the photo shows the garment on a mannequin. On the 10 holdout garments shot on a
    # mannequin ghost scored +1.3 (7 of 10 wins); on hanger and flat photos it lost 6 to 8.
    studio_style: str = "auto"


@dataclass(frozen=True)
class Attempt:
    prompt: str
    references: tuple[str, ...]
    image: Path
    gate: GateResult
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

    def run(self, photos: Sequence[Path], workdir: Path) -> ReconstructionResult:
        started = time.perf_counter()
        spec = extract_spec(roles_from_filenames(tuple(photos)), self._vision)
        references: tuple[Reference, ...] = (
            Reference(spec.front, "the whole garment from the front"),
            *named_detail_references(spec, _MAX_REFERENCES - 1),
        )
        workdir.mkdir(parents=True, exist_ok=True)
        style = _style_for(self._settings.studio_style, spec)

        first = self._attempt(spec, references[:1], workdir, 0, build_studio_prompt(style=style))
        attempts, final = [first], first
        if first.gate.problems:
            fix = first.gate.fix or "; ".join(first.gate.problems)
            if any(problem in GLOBAL_PROBLEMS for problem in first.gate.problems):
                # Colour, shape or view is wrong everywhere: draw it again from the photo.
                prompt = build_redraw_prompt(spec, references, fix, style)
                repair = self._attempt(spec, references, workdir, 1, prompt)
            else:
                # Only details are wrong: edit the one-shot packshot (image 1) so shape,
                # colour and framing stay, with the real photos as the reference.
                edit_refs = (Reference(first.image, "the packshot to edit"), *references)
                prompt = build_edit_prompt(references, fix)
                repair = self._attempt(spec, edit_refs, workdir, 1, prompt)
            attempts.append(repair)
            # Strictly fewer defects: on a tie the one-shot stays, it scores better.
            if repair.gate.weight < first.gate.weight:
                final = repair

        cost = spec.cost_usd + sum(a.cost_usd + a.gate.cost_usd for a in attempts)
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
        gate = check(spec.front, image, spec, self._vision)
        return Attempt(prompt, tuple(str(r.path) for r in references), image, gate, result.cost_usd)


def _style_for(setting: str, spec: GarmentSpec) -> str:
    if setting != "auto":
        return setting
    return "ghost" if spec.photo_setup == "mannequin" else "flat"

"""Reconstruction prompts: the studio prompt plus short constraints from the garment spec."""

from __future__ import annotations

from collections.abc import Sequence

from zir_image_enhancer.generation.prompts import build_prompt
from zir_image_enhancer.reconstruction.gate import proportion_text
from zir_image_enhancer.reconstruction.references import Reference
from zir_image_enhancer.reconstruction.spec import GarmentSpec

_STUDIO_LIGHT = (
    "Light it like a high-key e-commerce studio photo: bright, soft and even light with lifted "
    "shadows and clean whites, so the real colours read clearly. It must not look darker, "
    "dimmer or more contrasty than a catalogue photo, whatever the light in the reference."
)


def build_harness_prompt(spec: GarmentSpec, references: Sequence[Reference], fix: str = "") -> str:
    """The studio prompt (best on the holdout) plus short constraints from the spec.

    The fully spec-written prompt scored 3 points lower, so the proven prompt stays the base
    and the spec only adds what the photo alone does not enforce: geometry, studio light and
    the details to keep.
    """
    names = ", ".join(ref.label.removeprefix("a close-up of the ") for ref in references[1:])
    lines = [build_prompt("studio-style-v2", len(references), len(references) - 1, names)]
    proportions = proportion_text(spec)
    if proportions:
        lines.append(
            proportions.replace("Proportions in image 1:", "Keep the proportions of image 1:")
        )
    flatten = _FLATTEN.get(spec.photo_setup)
    if flatten:
        lines.append(flatten)
    lines.append(_STUDIO_LIGHT)
    if spec.details:
        keep = "; ".join(f"{d.name} ({d.location}): {d.description}" for d in spec.details)
        lines.append(f"Keep these details exactly: {keep}.")
    if fix:
        lines.append(f"A previous attempt had these problems; fix them: {fix}")
    return " ".join(lines)


# The edit model keeps the shape it sees: a garment shot on a mannequin came out with a bust
# and a waist. Saying how the photo was taken tells it which shape to undo.
_FLATTEN: dict[str, str] = {
    "mannequin": (
        "Image 1 shows the garment on a mannequin: flatten it completely, as the empty garment "
        "lying flat. No bust, waist or hip shaping; side seams run straight from armhole to hem."
    ),
    "worn": (
        "Image 1 shows the garment worn by a person: remove the body and flatten the empty "
        "garment completely. No bust, waist or hip shaping; side seams run straight."
    ),
    "hanger": (
        "Image 1 shows the garment on a hanger: remove the hanger and its sag; the shoulders lie "
        "flat and level and the garment is not stretched downwards."
    ),
    "flat": "Image 1 already shows the garment flat: keep its outline and only straighten it.",
}

_RELIGHT = (
    "Change only the lighting of this packshot: make it a high-key e-commerce studio photo with "
    "bright, soft, even light, neutral white balance, lifted shadows and a clean light-grey "
    "background (RGB 240,240,240). Do not change the garment's shape, size, position, folds, "
    "pattern, print, details or its real colours. Output only the relit packshot."
)


def build_relight_prompt() -> str:
    """Second pass: the shape is already done, so the model only has to fix the light."""
    return _RELIGHT

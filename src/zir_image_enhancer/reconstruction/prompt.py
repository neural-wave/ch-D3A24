"""Reconstruction prompts: the studio prompt plus short constraints from the garment spec."""

from __future__ import annotations

from collections.abc import Sequence

from zir_image_enhancer.generation.prompts import build_studio_prompt
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
    names = [ref.label.removeprefix("a close-up of the ") for ref in references[1:]]
    lines = [build_studio_prompt(names)]
    proportions = proportion_text(spec)
    if proportions:
        lines.append(
            proportions.replace("Proportions in image 1:", "Keep the proportions of image 1:")
        )
    if len(spec.pieces) > 1:
        lines.append(pieces_text(spec.pieces))
    lines.append(_STUDIO_LIGHT)
    if spec.details:
        keep = "; ".join(f"{d.name} ({d.location}): {d.description}" for d in spec.details)
        lines.append(f"Keep these details exactly: {keep}.")
    if fix:
        lines.append(f"A previous attempt had these problems; fix them: {fix}")
    return " ".join(lines)


def pieces_text(pieces: Sequence[str]) -> str:
    """Sets (a bikini) are one product: the studio prompt's "only the garment" must not drop a
    piece, so the prompt says what the garment is made of."""
    names = ", ".join(pieces[:-1]) + f" and {pieces[-1]}"
    return (
        f'This product is a set of {len(pieces)} pieces: {names}. Here "the garment" means '
        "the whole set: show every piece in the same packshot, each laid flat in front view, "
        "arranged like a catalogue set (top above bottom, pairs side by side), at the same "
        "scale and not overlapping."
    )

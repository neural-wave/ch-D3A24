"""The two repair prompts: draw the garment again, or edit the one-shot packshot."""

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


def build_redraw_prompt(
    spec: GarmentSpec, references: Sequence[Reference], fix: str, style: str = "flat"
) -> str:
    """Repair for a global defect: the studio prompt plus short constraints from the spec
    and the gate's fix, with the front photo and the named close-ups.

    The fully spec-written prompt scored 3 points lower, so the proven prompt stays the base
    and the spec only adds what the photo alone does not enforce: geometry, studio light and
    the details to keep.
    """
    names = [ref.label.removeprefix("a close-up of the ") for ref in references[1:]]
    lines = [build_studio_prompt(names, style)]
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


def build_edit_prompt(references: Sequence[Reference], fix: str) -> str:
    """Repair for detail defects: edit the one-shot packshot instead of drawing it again.

    Regenerating from the phone photo fixed the detail but moved shape, colour and framing
    (3 of 4 holdout repairs lost points). Image 1 is the packshot to keep; the real photos
    only show what the flagged details must look like.
    """
    closeups = [ref.label.removeprefix("a close-up of the ") for ref in references[1:]]
    photos = "Image 2 is a real photo of the same garment"
    if closeups:
        photos += f"; images 3 to {len(references) + 1} are close-ups of its " + ", ".join(closeups)
    return (
        f"Image 1 is a studio packshot of a garment. {photos}. Edit image 1 only to fix these "
        f"problems, using the real photos as the reference for how the garment looks: {fix}. "
        "Keep everything else in image 1 exactly as it is: the garment's shape, size and "
        "position, its colours, folds and texture, the background and the framing, unless a "
        "problem above is about one of them. Output only the edited packshot."
    )

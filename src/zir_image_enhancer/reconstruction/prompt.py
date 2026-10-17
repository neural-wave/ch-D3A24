"""Reconstruction prompt, built from the garment spec instead of one generic instruction.

Structure from the architecture: what to preserve, what to change, what to avoid, plus the
studio catalogue style that scored best on the holdout (studio-style-v2).
"""

from __future__ import annotations

from collections.abc import Sequence

from zir_image_enhancer.reconstruction.colour import MeasuredColour
from zir_image_enhancer.reconstruction.references import Reference
from zir_image_enhancer.reconstruction.spec import GarmentSpec

_STYLE = (
    "Present it as a fashion catalogue packshot: front view, centred and symmetric, laid flat "
    "and shot straight from above on a plain light-grey studio background (RGB 240,240,240), "
    "soft even diffused light, subtle natural fabric volume, no cast shadow. Long sleeves hang "
    "straight down along the sides of the body; short sleeves point slightly downwards; "
    "buttons and zips are closed; trouser legs are straight and parallel; dresses and skirts "
    "hang straight. The whole garment is visible with a margin, filling most of a portrait "
    "frame."
)


def build_reconstruction_prompt(
    spec: GarmentSpec,
    references: Sequence[Reference],
    colours: Sequence[MeasuredColour] = (),
    fix: str = "",
) -> str:
    lines = ["Reconstruct this exact garment as a studio flat-lay packshot."]
    lines.append(
        "Reference images: "
        + "; ".join(f"image {i} is {ref.label}" for i, ref in enumerate(references, start=1))
        + "."
    )
    garment = f"The garment is a {spec.category}"
    if spec.construction:
        garment += f" ({spec.construction})"
    if spec.material:
        garment += f", made of {spec.material}"
    if spec.pattern:
        garment += f", {spec.pattern}"
    lines.append(garment + ".")

    preserve = [
        "Preserve exactly: the real colour and shade, material and texture, pattern, "
        "proportions, construction and every detail below."
    ]
    if colours:
        palette = ", ".join(f"{c.hex} ({round(c.share * 100)}%)" for c in colours)
        preserve.append(
            f"Measured garment colours in image 1: {palette}. Keep these colours; take colour "
            "and brightness from image 1 only."
        )
    for detail in spec.details:
        where = f" on the {detail.location}" if detail.location else ""
        preserve.append(f"- {detail.name}{where}: {detail.description}.")
    lines.extend(preserve)

    lines.append(
        "Change only the pose, perspective, presentation and background needed for the packshot."
    )
    lines.append(_STYLE)
    lines.append(
        "Avoid: inventing construction that is not visible, adding accessories, branding, text "
        "or other garments, smoothing away the fabric texture, and stylized lighting that changes "
        "the apparent colour. Remove any person, mannequin, hanger, price tag, sticker and floor."
    )
    if fix:
        lines.append(f"A previous attempt had these problems; fix them: {fix}")
    return "\n".join(lines)

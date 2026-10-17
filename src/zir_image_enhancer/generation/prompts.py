"""The studio prompt the harness builds on. Earlier prompts and their results: docs/PROMPTS.md."""

from __future__ import annotations

from collections.abc import Sequence

# Written after looking at the training targets: front view, symmetric, long sleeves hanging
# straight down at the sides, closures done up, trouser legs straight and parallel, light-grey
# studio background, portrait frame. One prompt with rules per garment type, so no classifier
# is needed in front of it. Best one-shot prompt on the holdout (61.82).
_STUDIO_STYLE_V2 = (
    "Turn the {source} of this garment into a fashion catalogue packshot. "
    "{references}"
    "Show only the garment, front view, centred and symmetric, laid flat and shot straight "
    "from above on a plain light-grey studio background (RGB 240,240,240), with soft, even, "
    "diffused light, a subtle natural fabric volume and no cast shadow. Style it neatly: long "
    "sleeves hang straight down along the sides of the body, slightly apart from it; short "
    "sleeves point slightly downwards; buttons and zips are closed and the collar is tidy; "
    "trousers, jeans and shorts have both legs straight down and parallel; dresses and skirts "
    "hang straight with a flat hem. The whole garment is visible with a margin on every side "
    "and fills most of a portrait frame. Keep exactly the same garment: colour and shade, "
    "fabric and texture, pattern, print, logos and small details, length and proportions. "
    "Remove any person, mannequin, hanger, price tag, floor and props, and never add "
    "accessories, other garments, text or watermarks."
)

# Used when the user named detail photos. The model is told what each image is, so it copies
# the real letters and shapes instead of guessing them from a small area of the full photo.
_DETAIL_REFERENCES = (
    "Image 1 shows the whole garment from the front. Images 2 to {last} are close-ups of "
    "details on the front of the same garment ({details}). Use the close-ups only for the "
    "shapes of those details: same letters, numbers and drawing, at the position and size they "
    "have in image 1. Take every colour and the brightness of the garment and of its details "
    "from image 1 only, because the close-ups may be lit differently. Output one front view. "
)


# Most adult garments in the training targets are not flat-lays seen from above: they are
# shot straight on with the volume of an invisible mannequin (rounded shoulders, sleeves
# falling close to the body, trousers hanging). This variant asks for that look.
_STUDIO_GHOST_V1 = (
    "Turn the {source} of this garment into an e-commerce product photo in the "
    "invisible-mannequin style. "
    "{references}"
    "Show only the garment, front view, shot straight on at chest height (not from above), "
    "centred and symmetric, with the natural three-dimensional volume it has when worn by an "
    "invisible mannequin: level, rounded shoulders, sleeves falling naturally close to the "
    "sides of the body, buttons and zips closed and the collar tidy; trousers, jeans and "
    "shorts hang straight with both legs parallel; dresses and skirts fall naturally. Plain "
    "seamless light-grey studio background (RGB 240,240,240), soft, even, diffused light and "
    "only a faint soft shadow under the garment. The whole garment is visible with a margin on "
    "every side and fills most of a portrait frame. Keep exactly the same garment: colour and "
    "shade, fabric and texture, pattern, print, logos and small details, length and "
    "proportions. Remove any person, mannequin, hanger, price tag, floor and props, and never "
    "add accessories, other garments, text or watermarks."
)

# The official hosted baseline prompt, verbatim (68.71 on dev with the first photo). Our
# studio-style-v2 beat it on the holdout but not on dev (68.62), so dev-tuned variants start
# from it. It has no {references} slot: these variants send the front photo only.
_BASELINE = (
    "Turn this photograph of a single garment into a clean e-commerce flat-lay packshot. "
    "Lay the garment perfectly flat and centred on a plain, uniform light-grey background "
    "(RGB 245,245,245), seen straight from above with no perspective distortion. Use soft, "
    "even studio lighting with no visible shadow on the background. Keep the exact same "
    "garment, colour, material, pattern and proportions as the original. Remove the person, "
    "mannequin, hanger, floor, furniture and background; do not add props, text, logos, "
    "watermarks or reflections. Output only the final packshot image."
)

# B: colour and LPIPS lose the most points, so ask the model to change less of the garment.
_BASELINE_KEEP = _BASELINE.replace(
    " Output only the final packshot image.",
    " Change as little as possible of the garment itself: keep its real colours and "
    "brightness, fabric texture, stitching, seams, logos and print exactly as in the photo; "
    "only remove what is around it and lay it flat. Output only the final packshot image.",
)

# C: the same with a pure white background; generating it white can change how the model
# exposes the garment, unlike painting the background afterwards (that lowered the score).
_BASELINE_WHITE = _BASELINE.replace(
    "light-grey background (RGB 245,245,245)", "pure white background (RGB 255,255,255)"
)

STYLES = {
    "flat": _STUDIO_STYLE_V2,
    "ghost": _STUDIO_GHOST_V1,
    "baseline": _BASELINE,
    "baseline-keep": _BASELINE_KEEP,
    "baseline-white": _BASELINE_WHITE,
}


def build_studio_prompt(detail_names: Sequence[str] = (), style: str = "flat") -> str:
    """The studio prompt for the front photo plus one close-up per detail name.

    ``flat`` is ``studio-style-v2`` (laid flat, from above); ``ghost`` the invisible-mannequin
    look most adult targets have.
    """
    template = STYLES[style]
    if detail_names:
        references = _DETAIL_REFERENCES.format(
            last=len(detail_names) + 1, details=", ".join(detail_names)
        )
        return template.format(source="reference photos", references=references)
    return template.format(source="photo", references="")

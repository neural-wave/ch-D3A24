"""Prompt templates. Each one has a stable key so runs and results stay comparable."""

from __future__ import annotations

# Copied from the official hosted baseline (baselines/baseline_nanobanana.py in the starter
# repo), so a run with this prompt reproduces the 68.71 dev reference.
_BASELINE = (
    "Turn this photograph of a single garment into a clean e-commerce flat-lay packshot. "
    "Lay the garment perfectly flat and centred on a plain, uniform light-grey background "
    "(RGB 245,245,245), seen straight from above with no perspective distortion. Use soft, "
    "even studio lighting with no visible shadow on the background. Keep the exact same "
    "garment, colour, material, pattern and proportions as the original. Remove the person, "
    "mannequin, hanger, floor, furniture and background; do not add props, text, logos, "
    "watermarks or reflections. Output only the final packshot image."
)

# Written for the scorer: the whole garment visible (a cut garment is a real error), white
# background with no shadow (u2net can swallow a shadow into the mask), and colour kept
# as-is (colour is 40% of the score).
_STUDIO_WHITE_V1 = (
    "Create a professional e-commerce flat-lay packshot of the garment in the {source}. "
    "{references}"
    "Show the whole garment, front side up, laid perfectly flat and symmetric, seen straight "
    "from above, centred with a clear empty margin on every side. Arrange sleeves, straps and "
    "legs neatly as in a catalogue flat-lay. Pure white seamless background, soft even studio "
    "lighting, no shadows, no reflections. Keep exactly the same garment: the same colour and "
    "shade, fabric texture, pattern, print, stitching, buttons and proportions. Remove any "
    "person, mannequin, hanger, price tag, floor and props. Do not add text, logos or "
    "watermarks. Output only the final packshot image."
)

# Written after looking at the training targets: front view, symmetric, long sleeves hanging
# straight down at the sides, closures done up, trouser legs straight and parallel, light-grey
# studio background, portrait frame. One prompt with rules per garment type, so no classifier
# is needed in front of it.
_SARTIQ_STYLE_V2 = (
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

_MULTI_REFERENCES = (
    "All {count} photos show the same single garment: the first is the main view, the others "
    "are other angles or close-ups. Use all of them to understand the garment, and output one "
    "front view. "
)

TEMPLATES: dict[str, str] = {
    "baseline": _BASELINE,
    "studio-white-v1": _STUDIO_WHITE_V1,
    "sartiq-style-v2": _SARTIQ_STYLE_V2,
}


def build_prompt(template_key: str, n_references: int) -> str:
    if template_key not in TEMPLATES:
        raise KeyError(f"Unknown prompt '{template_key}'. Choose one of: {', '.join(TEMPLATES)}")
    template = TEMPLATES[template_key]
    if template_key == "baseline":
        # The baseline prompt is kept verbatim on purpose, whatever the number of photos.
        return template
    if n_references > 1:
        return template.format(
            source="reference photos",
            references=_MULTI_REFERENCES.format(count=n_references),
        )
    return template.format(source="photo", references="")

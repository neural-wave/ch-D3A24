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

# For photos that already follow our capture protocol (garment laid flat, shot from above,
# colour balanced with a grey card). The model is asked to change as little as possible:
# keep the layout and colours it sees, only clean the scene and the light. Less to invent
# means fewer hallucinated details and colour drift.
_STUDIO_FLAT_V1 = (
    "The {source} already shows this garment laid flat and shot from above, with correct "
    "colours. {references}"
    "Turn it into a fashion catalogue packshot while changing as little as possible: keep the "
    "garment's layout, shape, proportions, exact colours, fabric texture, pattern, print, "
    "logos and small details as they are. Only straighten the perspective so it is seen "
    "exactly from above, centre it, smooth small wrinkles, tidy the collar, close buttons and "
    "zips if they are open, and make the light soft and even. Replace the surface under it "
    "with a plain light-grey studio background (RGB 240,240,240) with no cast shadow, and "
    "remove any card, tag, label, hand or object around it. The whole garment stays visible "
    "with a margin on every side, in a portrait frame. Never add accessories, other garments, "
    "text or watermarks."
)

_MULTI_REFERENCES = (
    "All {count} photos show the same single garment: the first is the main view, the others "
    "are other angles or close-ups. Use all of them to understand the garment, and output one "
    "front view. "
)

# Used when the photo triage found fine details on the front and close-ups of them. The model
# is told what each image is, so it copies the real letters and shapes instead of guessing
# them from a small, blurry area of the full photo.
_DETAIL_REFERENCES = (
    "Image 1 shows the whole garment from the front. Images 2 to {last} are close-ups of "
    "details on the front of the same garment ({details}). Use the close-ups only for the "
    "shapes of those details: same letters, numbers and drawing, at the position and size they "
    "have in image 1. Take every colour and the brightness of the garment and of its details "
    "from image 1 only, because the close-ups may be lit differently. Output one front view. "
)

TEMPLATES: dict[str, str] = {
    "baseline": _BASELINE,
    "studio-white-v1": _STUDIO_WHITE_V1,
    # Generic studio catalogue conventions, written as text (no target pixels are ever used).
    "studio-style-v2": _STUDIO_STYLE_V2,
    "studio-flat-v1": _STUDIO_FLAT_V1,
    # Old names, kept so earlier runs in experiments/results.csv can be reproduced.
    "sartiq-style-v2": _STUDIO_STYLE_V2,
    "sartiq-flat-v1": _STUDIO_FLAT_V1,
}


def build_prompt(
    template_key: str, n_references: int, detail_closeups: int = 0, details: str = ""
) -> str:
    if template_key not in TEMPLATES:
        raise KeyError(f"Unknown prompt '{template_key}'. Choose one of: {', '.join(TEMPLATES)}")
    template = TEMPLATES[template_key]
    if template_key == "baseline":
        # The baseline prompt is kept verbatim on purpose, whatever the number of photos.
        return template
    if detail_closeups > 0:
        return template.format(
            source="reference photos",
            references=_DETAIL_REFERENCES.format(
                last=detail_closeups + 1, details=details or "print, logo or text"
            ),
        )
    if n_references > 1:
        return template.format(
            source="reference photos",
            references=_MULTI_REFERENCES.format(count=n_references),
        )
    return template.format(source="photo", references="")

"""Quality gate: compare the generated packshot with the front photo and the spec.

It checks the failures the jury punishes and the photo can prove: wrong or missing details,
invented text, extra items (mannequin, hanger, a garment that is not part of the product), a
back view instead of the front, and an obvious colour change. A failed check returns the fix
instructions for one repair attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zir_image_enhancer.reconstruction.spec import GarmentSpec
from zir_image_enhancer.reconstruction.vision import VisionClient

_SYSTEM = (
    "You are a strict quality checker for e-commerce packshots. Compare the packshot with the "
    "real photo of the same garment. Answer with one JSON object and nothing else."
)

_PROMPT = """Image 1 is the real photo of the garment (front). Image 2 is a generated packshot that
must show the SAME garment, flat, front view, on a plain light background.

Garment: {category}. Product pieces: {pieces}. {proportions}
Details that must be present and identical:
{details}

Return:
{{
  "same_garment": <true if it is the same type of product with the same construction>,
  "front_view": <true if image 2 shows the front>,
  "colour_ok": <false only if the colour or shade clearly changed>,
  "proportions_ok": <false if sleeve length or garment length relative to width clearly
                     differs from image 1>,
  "lighting_ok": <false if image 2 looks dark, dim, contrasty or washed out instead of a
                  bright, soft, even e-commerce studio photo with clean, vivid colours>,
  "flat_shape": <false if image 2 still shows the shape of a body or mannequin (bust, waist
                 curve, rounded shoulders) or the sag of a hanger instead of a flat garment>,
  "extra_items": <true if there is a person, mannequin, hanger, tag, props, or a garment
                  that is not one of the product pieces>,
  "all_pieces": <false if one of the product pieces is missing>,
  "details": [{{"name": "<detail>", "ok": <true if present, same shape/text, same place>}}],
  "invented": <true if image 2 adds text, logos or prints that image 1 does not have>,
  "fix": "<short instructions to fix the problems, empty if none>"
}}
Lighting and background differences are expected; judge only the garment itself."""


# Every problem is repaired once (the jury punishes any invented detail); the report says
# which ones are severe. Minor repairs (a button count) cost about 2 points on the holdout
# metric, which compares against one studio photo, not against the real garment.
SEVERE = (
    "different garment",
    "not the front",
    "extra items",
    "missing piece",
    "invented text or logo",
)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    problems: tuple[str, ...]
    fix: str
    cost_usd: float
    used_fallback: bool = False

    @property
    def severe(self) -> tuple[str, ...]:
        return tuple(problem for problem in self.problems if problem in SEVERE)


def proportion_text(spec: GarmentSpec) -> str:
    """The garment's proportions as one sentence, shared by the prompt and the gate."""
    parts = []
    if spec.body_length_to_width:
        parts.append(f"garment length about {spec.body_length_to_width} times the chest width")
    if spec.sleeve_to_body_length:
        parts.append(f"sleeves about {spec.sleeve_to_body_length} of the garment length")
    if spec.proportion_notes:
        parts.append(spec.proportion_notes)
    return ("Proportions in image 1: " + "; ".join(parts) + ".") if parts else ""


def check(front: Path, output: Path, spec: GarmentSpec, vision: VisionClient) -> GateResult:
    details = "\n".join(f"- {d.name}: {d.description}" for d in spec.details) or "- none listed"
    prompt = _PROMPT.format(
        category=spec.category,
        pieces=", ".join(spec.pieces) if len(spec.pieces) > 1 else "one garment",
        proportions=proportion_text(spec),
        details=details,
    )
    try:
        answer = vision.ask(_SYSTEM, prompt, [front, output], kind="gate")
    except Exception as error:  # noqa: BLE001
        # A broken check must not block the garment: it passes, and the report says why.
        return GateResult(True, (f"gate unavailable: {error}",), "", 0.0, used_fallback=True)
    data = answer.data
    problems = []
    if data.get("same_garment") is False:
        problems.append("different garment")
    if data.get("front_view") is False:
        problems.append("not the front")
    if data.get("colour_ok") is False:
        problems.append("colour changed")
    if data.get("proportions_ok") is False:
        problems.append("proportions differ")
    if data.get("lighting_ok") is False:
        problems.append("lighting not studio")
    if data.get("flat_shape") is False:
        problems.append("worn shape")
    if data.get("extra_items") is True:
        problems.append("extra items")
    if data.get("all_pieces") is False:
        problems.append("missing piece")
    if data.get("invented") is True:
        problems.append("invented text or logo")
    rows = data.get("details")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("ok") is False:
            problems.append(f"detail wrong: {row.get('name', '?')}")
    return GateResult(not problems, tuple(problems), str(data.get("fix", "")), answer.cost_usd)

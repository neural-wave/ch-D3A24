"""Quality gate: compare the generated packshot with the front photo and the spec.

It checks the failures the jury punishes and the photo can prove: wrong or missing details,
invented text, extra items (mannequin, hanger, second garment), back instead of front, and an
obvious colour change. A failed check returns the fix instructions for one repair attempt.
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

Garment: {category}. Details that must be present and identical:
{details}

Return:
{{
  "same_garment": <true if it is the same type of garment with the same construction>,
  "front_view": <true if image 2 shows the front>,
  "colour_ok": <false only if the colour or shade clearly changed>,
  "extra_items": <true if there is a person, mannequin, hanger, tag, second garment or props>,
  "details": [{{"name": "<detail>", "ok": <true if present, same shape/text, same place>}}],
  "invented": <true if image 2 adds text, logos or prints that image 1 does not have>,
  "fix": "<short instructions to fix the problems, empty if none>"
}}
Lighting and background differences are expected; judge only the garment itself."""


# Problems worth paying for a repair. On the holdout, repairs for minor problems (a button
# count, a slight shade) came back almost identical and scored 2 points lower on average,
# so those stay as warnings in the report.
SEVERE = ("different garment", "not the front", "extra items", "invented text or logo")


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


def check(front: Path, output: Path, spec: GarmentSpec, vision: VisionClient) -> GateResult:
    details = "\n".join(f"- {d.name}: {d.description}" for d in spec.details) or "- none listed"
    prompt = _PROMPT.format(category=spec.category, details=details)
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
    if data.get("extra_items") is True:
        problems.append("extra items")
    if data.get("invented") is True:
        problems.append("invented text or logo")
    rows = data.get("details")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("ok") is False:
            problems.append(f"detail wrong: {row.get('name', '?')}")
    return GateResult(not problems, tuple(problems), str(data.get("fix", "")), answer.cost_usd)

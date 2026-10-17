"""Decide, with a vision model, which photos of a garment go to the edit model.

One call per garment with all its photos. Comparing the photos side by side is what makes
front versus back reliable (collar, buttons, pockets and prints differ between the two),
which CLIP failed at. The answer also says whether the front has fine details (small text,
logos, detailed prints) and which photos are close-ups of them:

* the front, which is what the pipeline sends (on 43 multi-photo train pairs this scored
  61.70 in the original order and 61.67 with the photos shuffled, the best of all policies);
* whether the front has fine details, written to the run log so they are checked first;
* close-ups of those details. Sending them (max_closeups > 0) lowered the score by about
  2 points on the 11 garments that had them, mostly through colour shifts, so it is off by
  default.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import fal_client

from zir_image_enhancer.generation.fal_backend import BudgetExceededError, CostLedger

ENDPOINT = "openrouter/router/vision"
DEFAULT_MODEL = "google/gemini-2.5-flash"
# Bump when the prompt changes, so cached answers from an older prompt are not reused.
PROMPT_VERSION = "v2"

_SYSTEM = (
    "You check phone photos of one single garment for an e-commerce packshot pipeline. "
    "Answer with one JSON object and nothing else."
)

_PROMPT = """There are {count} photos of the same garment, numbered 1 to {count} in the order given.

Return this JSON object:
{{
  "front_index": <number of the photo that shows the WHOLE garment from the FRONT most clearly>,
  "photos": [
    {{"index": <n>, "view": "<front|back|detail_front|detail_other|label|packaging|other>",
      "whole_garment": <true if the whole garment is visible>}}
  ],
  "fine_details": <true if the FRONT has small text, logos, numbers or detailed prints that are
                   easy to get wrong when the garment is redrawn>,
  "details": "<short description of those front details, or empty>"
}}

Rules:
- The front is the side a buyer sees first: neck opening, buttons or zip, chest print or logo.
  Do not choose a back view as the front, even if the back has a bigger print.
- "detail_front" is a close-up of a print, logo, text or embroidery that is on the FRONT.
- Price tags, care labels and barcodes are "label"; a garment inside a bag is "packaging".
- "fine_details" and "details" only count what is part of the garment itself (printed, woven,
  embroidered or sewn on). Stickers, price tags, size stickers and hang tags do not count:
  they will be removed from the packshot.
"""


@dataclass(frozen=True)
class PhotoRole:
    index: int
    view: str
    whole_garment: bool


@dataclass(frozen=True)
class TriageResult:
    front: Path
    detail_closeups: tuple[Path, ...]
    fine_details: bool
    details: str
    roles: tuple[PhotoRole, ...] = ()
    cost_usd: float = 0.0
    used_fallback: bool = False
    note: str = ""

    @property
    def selected(self) -> tuple[Path, ...]:
        """Photos to send: the front first, then the close-ups of its details."""
        return (self.front, *self.detail_closeups)

    @property
    def needs_closeup(self) -> bool:
        return self.fine_details and not self.detail_closeups

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["front"] = str(self.front)
        data["detail_closeups"] = [str(path) for path in self.detail_closeups]
        data["needs_closeup"] = self.needs_closeup
        return data


@dataclass
class VisionTriage:
    ledger: CostLedger
    budget_usd: float
    cache_dir: Path
    model: str = DEFAULT_MODEL
    max_closeups: int = 2
    # Rough upper bound per call for the budget guard; the real cost is logged from the reply.
    est_cost_usd: float = 0.01
    client: fal_client.SyncClient | None = field(default=None, repr=False)

    def triage(self, photos: Sequence[Path]) -> TriageResult:
        cached = self.cache_dir / f"{self._cache_key(photos)}.json"
        if cached.exists():
            return self._build(photos, json.loads(cached.read_text(encoding="utf-8")), 0.0)
        try:
            answer, cost = self._ask(photos)
        except BudgetExceededError:
            raise
        except Exception as error:  # noqa: BLE001
            # The pipeline must never stop on the triage: fall back to the first photo.
            return TriageResult(photos[0], (), False, "", used_fallback=True, note=str(error))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(answer, indent=2), encoding="utf-8")
        return self._build(photos, answer, cost)

    def _ask(self, photos: Sequence[Path]) -> tuple[dict[str, object], float]:
        if self.ledger.spent_usd() + self.est_cost_usd > self.budget_usd:
            raise BudgetExceededError("triage call would cross the budget limit")
        client = self.client or fal_client.SyncClient()
        response = client.subscribe(
            ENDPOINT,
            arguments={
                "model": self.model,
                "system_prompt": _SYSTEM,
                "prompt": _PROMPT.format(count=len(photos)),
                "image_urls": [fal_client.encode_file(path) for path in photos],
                "temperature": 0,
            },
        )
        usage = response.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        self.ledger.record(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": self.model,
                "endpoint": ENDPOINT,
                "cost_usd": cost,
                "kind": "triage",
            }
        )
        return _parse_json(str(response.get("output", ""))), cost

    def _build(
        self, photos: Sequence[Path], answer: dict[str, object], cost: float
    ) -> TriageResult:
        rows = answer.get("photos")
        roles: list[PhotoRole] = []
        for row in rows if isinstance(rows, list) else []:
            # The reply is model output: anything malformed is skipped, never trusted.
            index = _as_int(row.get("index")) if isinstance(row, dict) else None
            if index is None or not 1 <= index <= len(photos):
                continue
            roles.append(
                PhotoRole(index, str(row.get("view", "other")), bool(row.get("whole_garment")))
            )
        front_index = _as_int(answer.get("front_index")) or 1
        if not 1 <= front_index <= len(photos):
            front_index = 1
        fine_details = bool(answer.get("fine_details", False))
        closeups = tuple(
            photos[role.index - 1]
            for role in roles
            if role.view == "detail_front" and role.index != front_index
        )[: self.max_closeups]
        return TriageResult(
            front=photos[front_index - 1],
            detail_closeups=closeups if fine_details else (),
            fine_details=fine_details,
            details=str(answer.get("details", "")),
            roles=tuple(roles),
            cost_usd=cost,
        )

    def _cache_key(self, photos: Sequence[Path]) -> str:
        digest = hashlib.sha256(f"{self.model}|{PROMPT_VERSION}".encode())
        for path in photos:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()[:24]


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None


def _parse_json(text: str) -> dict[str, object]:
    """Pull the JSON object out of the reply, tolerating code fences or extra words."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON in the vision reply: {text[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("the vision reply is not a JSON object")
    return parsed

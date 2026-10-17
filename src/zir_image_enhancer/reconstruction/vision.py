"""One small client for every vision-model call (spec extraction and quality gate).

Answers are cached by prompt and image content, and every paid call goes to the shared cost
ledger, so the vision steps follow the same budget rules as generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import fal_client

from zir_image_enhancer.generation.fal_backend import BudgetExceededError, CostLedger

ENDPOINT = "openrouter/router/vision"
DEFAULT_MODEL = "google/gemini-2.5-flash"


@dataclass(frozen=True)
class VisionAnswer:
    data: dict[str, object]
    cost_usd: float
    from_cache: bool


@dataclass
class VisionClient:
    ledger: CostLedger
    budget_usd: float
    cache_dir: Path
    model: str = DEFAULT_MODEL
    # Upper bound per call for the budget guard; the real cost comes back in the reply.
    est_cost_usd: float = 0.01
    client: fal_client.SyncClient | None = field(default=None, repr=False)

    def ask(self, system: str, prompt: str, images: Sequence[Path], kind: str) -> VisionAnswer:
        cached = self.cache_dir / f"{kind}-{self._key(system, prompt, images)}.json"
        if cached.exists():
            return VisionAnswer(json.loads(cached.read_text(encoding="utf-8")), 0.0, True)
        if self.ledger.spent_usd() + self.est_cost_usd > self.budget_usd:
            raise BudgetExceededError(f"{kind} call would cross the budget limit")
        client = self.client or fal_client.SyncClient()
        response = client.subscribe(
            ENDPOINT,
            arguments={
                "model": self.model,
                "system_prompt": system,
                "prompt": prompt,
                "image_urls": [fal_client.encode_file(path) for path in images],
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
                "kind": kind,
            }
        )
        data = parse_json(str(response.get("output", "")))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return VisionAnswer(data, cost, False)

    def _key(self, system: str, prompt: str, images: Sequence[Path]) -> str:
        digest = hashlib.sha256(f"{self.model}|{system}|{prompt}".encode())
        for path in images:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()[:24]


def parse_json(text: str) -> dict[str, object]:
    """Pull the JSON object out of a reply, tolerating code fences or extra words."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON in the vision reply: {text[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("the vision reply is not a JSON object")
    return parsed


def as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None

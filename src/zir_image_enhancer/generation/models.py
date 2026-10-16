"""Hosted edit models we compare, and how to call each one."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    endpoint: str
    max_references: int
    # Per call, used only by the budget guard. These are estimates from fal's pages;
    # check the model page before trusting them for planning.
    est_cost_usd: float
    supports_seed: bool
    extra_arguments: Mapping[str, object] = field(default_factory=dict)

    def build_arguments(
        self, prompt: str, image_urls: Sequence[str], seed: int | None
    ) -> dict[str, object]:
        arguments: dict[str, object] = {
            "prompt": prompt,
            "image_urls": list(image_urls)[: self.max_references],
            **self.extra_arguments,
        }
        if seed is not None and self.supports_seed:
            arguments["seed"] = seed
        return arguments


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        # The official baseline model (about $0.06 per edit according to the organisers).
        ModelSpec("nano-banana-2.1", "google/nano-banana-2.1/edit", 14, 0.06, False),
        ModelSpec("nano-banana-pro", "fal-ai/nano-banana-pro/edit", 14, 0.15, False),
        ModelSpec("flux-2", "fal-ai/flux-2/edit", 9, 0.05, True),
        ModelSpec("seedream-4.5", "fal-ai/bytedance/seedream/v4.5/edit", 10, 0.05, True),
    )
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}")
    return MODELS[key]

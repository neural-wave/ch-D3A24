"""The hosted edit model and how to call it (other models we tried: README findings)."""

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
    # True only for the fine-tuned packshot LoRA: generation/lora.py attaches the registered
    # adapter as the ``loras`` argument, and refuses to run when none is registered.
    requires_lora: bool = False
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
        # Its web search option stays off (the default): retrieving product images is
        # against the rules and disqualifies the garment.
        ModelSpec("nano-banana-2.1", "google/nano-banana-2.1/edit", 14, 0.06, True),
        # Same model with a portrait canvas like the targets (about 2:3). With "auto" it
        # keeps the input's aspect, and landscape inputs came back as wide, spread-out
        # garments that lost silhouette IoU against the narrow targets.
        ModelSpec(
            "nano-banana-2.1-portrait",
            "google/nano-banana-2.1/edit",
            14,
            0.06,
            True,
            extra_arguments={"aspect_ratio": "2:3"},
        ),
        # The fine-tuned packshot LoRA trained from the example pairs (docs/LORA.md), used
        # exactly like any other model: official spec/prompt/gate stay untouched. The
        # endpoint is the klein edit model with LoRA support; the adapter itself is not in
        # this spec (generation/lora.py adds it, and the cache key covers it), so the same
        # spec can point at any registered adapter. 4 references max, like the trainer.
        # The canvas is the exact 2:3 one the pairs are trained on: 832x1248 is the largest
        # 2:3 size divisible by 16 whose area stays under klein's 1024^2 cap. Billing is
        # $0.02 per megapixel of input and output with each input resized to 1 MP, so the
        # worst case (4 references + ~1 MP output) is about $0.10.
        ModelSpec(
            "sartiq-lora-klein",
            "fal-ai/flux-2/klein/9b/base/edit/lora",
            4,
            0.10,
            True,
            requires_lora=True,
            extra_arguments={"image_size": {"width": 832, "height": 1248}},
        ),
    )
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}")
    return MODELS[key]

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
            {"aspect_ratio": "2:3"},
        ),
        # The other models also get a portrait canvas, so the comparison is between models
        # and not between canvas shapes. Prices are from each model's fal page.
        ModelSpec(
            "nano-banana-pro",
            "fal-ai/nano-banana-pro/edit",
            14,
            0.15,
            True,
            {"aspect_ratio": "2:3"},
        ),
        # FLUX.2 [dev] edit reads at most 4 images and bills $0.012 per megapixel on every
        # input (resized to 1 MP) and on the output, so 4 inputs + a 1.6 MP output is ~$0.07.
        ModelSpec(
            "flux-2",
            "fal-ai/flux-2/edit",
            4,
            0.07,
            True,
            {"image_size": {"width": 1024, "height": 1536}},
        ),
        # Open-weight Qwen-Image-Edit-2511 (Apache 2.0) through fal, so it can be tested in
        # minutes instead of setting up a GPU. $0.035 per megapixel; 1024x1536 is ~1.6 MP.
        ModelSpec(
            "qwen-2511",
            "fal-ai/qwen-image-edit-2511/lora",
            3,
            0.06,
            True,
            {"image_size": {"width": 1024, "height": 1536}},
        ),
        # Same model with the community "Extract-Outfit" LoRA (Apache 2.0), trained to turn a
        # worn garment into a flat mockup; the closest public match to our task.
        ModelSpec(
            "qwen-2511-outfit",
            "fal-ai/qwen-image-edit-2511/lora",
            3,
            0.06,
            True,
            {
                "image_size": {"width": 1024, "height": 1536},
                "loras": [
                    {
                        "path": "https://huggingface.co/prithivMLmods/QIE-2511-Extract-Outfit"
                        "/resolve/main/QIE-2511-Extract-Outfit-4200.safetensors",
                        "scale": 1.0,
                    }
                ],
            },
        ),
        # Seedream needs at least 1920 px per side (or a large total pixel count).
        ModelSpec(
            "seedream-4.5",
            "fal-ai/bytedance/seedream/v4.5/edit",
            10,
            0.04,
            True,
            {"image_size": {"width": 1920, "height": 2880}},
        ),
    )
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}")
    return MODELS[key]

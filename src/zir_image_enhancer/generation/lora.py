"""Where the fine-tuned Sartiq LoRA lives, and how it reaches the generation request.

The packshot LoRA trained from the example pairs (see ``docs/LORA.md``) is used by
``StudioPipeline`` exactly like any other hosted model: the spec, the references, the
prompt, the quality gate and the repair loop stay untouched. Only the model changes —
``fal-ai/flux-2/klein/9b/base/edit/lora`` receives the adapter as its ``loras``
argument.

The adapter is resolved when the Reconstructor starts (so a swap needs no restart):

1. ``ZIR_LORA_PATH`` from the environment — a fal URL, a HuggingFace repo id
   (``owner/repo``) or a local ``.safetensors`` path; ``ZIR_LORA_SCALE`` overrides the
   scale (default 1.0);
2. ``local/lora/registry.json``, written by ``scripts/lora/train_fal.py`` when a
   training run finishes and by ``scripts/lora/register_lora.py`` for weights trained
   elsewhere (e.g. on vast.ai).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from zir_image_enhancer.generation.models import ModelSpec
from zir_image_enhancer.paths import ProjectPaths


class LoraNotAvailableError(RuntimeError):
    """The selected model needs the fine-tuned LoRA and no adapter is registered."""


@dataclass(frozen=True)
class LoraBinding:
    """One registered adapter: where it is, how strongly to apply it, where it came from."""

    path: str
    scale: float = 1.0
    name: str = ""


def load_binding(paths: ProjectPaths | None = None) -> LoraBinding | None:
    """Read the active adapter from the environment, then from the local registry."""
    env_path = os.environ.get("ZIR_LORA_PATH")
    if env_path:
        scale = _float_or(os.environ.get("ZIR_LORA_SCALE"), 1.0)
        return LoraBinding(env_path, scale, name="ZIR_LORA_PATH")
    registry = (paths or ProjectPaths.from_env()).lora_registry
    if registry.is_file():
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        path = str(data.get("path") or "")
        if path:
            return LoraBinding(path, _float_or(data.get("scale"), 1.0), str(data.get("name") or ""))
    return None


def attach_binding(spec: ModelSpec, binding: LoraBinding) -> ModelSpec:
    """The same spec, with the adapter added as the ``loras`` argument.

    The binding is part of ``extra_arguments`` on purpose: the generator's cache key
    covers them, so two adapters (or two scales) never share a cached image.
    """
    loras = [{"path": binding.path, "scale": binding.scale}]
    return replace(spec, extra_arguments={**spec.extra_arguments, "loras": loras})


def resolved_model(spec: ModelSpec, paths: ProjectPaths | None = None) -> ModelSpec:
    """Attach the registered LoRA when the spec needs one; leave every other model as is."""
    if not spec.requires_lora:
        return spec
    binding = load_binding(paths)
    if binding is None:
        registry = (paths or ProjectPaths.from_env()).lora_registry
        raise LoraNotAvailableError(
            f"model '{spec.key}' needs the fine-tuned LoRA, but no adapter is registered. "
            "Train one (scripts/lora/train_fal.py --live) or register existing weights "
            f"(scripts/lora/register_lora.py), or point ZIR_LORA_PATH at one. "
            f"Registry looked up in {registry}."
        )
    return attach_binding(spec, binding)


def _float_or(value: object, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def registry_path(paths: ProjectPaths | None = None) -> Path:
    """Where the active adapter is registered (for messages and training scripts)."""
    return (paths or ProjectPaths.from_env()).lora_registry

"""Tests for the fine-tuned packshot LoRA: resolution, attachment, fail-fast."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from zir_image_enhancer.generation.fal_backend import GenerationRequest
from zir_image_enhancer.generation.lora import (
    LoraBinding,
    LoraNotAvailableError,
    attach_binding,
    load_binding,
    resolved_model,
)
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.pipeline import StudioPipeline


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(tmp_path, tmp_path / "data", tmp_path / "starter", tmp_path / "runs")


def _register(paths: ProjectPaths, path: str = "https://cdn.example.com/lora.safetensors") -> None:
    registry = paths.lora_registry
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps({"path": path, "scale": 1.0, "name": "test"}), encoding="utf-8"
    )


def _clear_lora_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZIR_LORA_PATH", raising=False)
    monkeypatch.delenv("ZIR_LORA_SCALE", raising=False)


def test_plain_model_passes_through_without_a_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_lora_env(monkeypatch)
    spec = get_model("nano-banana-2.1-portrait")
    assert resolved_model(spec, _paths(tmp_path)) is spec


def test_registry_binding_reaches_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_lora_env(monkeypatch)
    paths = _paths(tmp_path)
    _register(paths, "https://cdn.example.com/sartiq.safetensors")
    spec = resolved_model(get_model("sartiq-lora-klein"), paths)

    assert spec.endpoint == "fal-ai/flux-2/klein/9b/base/edit/lora"
    arguments = spec.build_arguments("prompt", ["u"], 3)
    assert arguments["loras"] == [
        {"path": "https://cdn.example.com/sartiq.safetensors", "scale": 1.0}
    ]
    assert arguments["image_size"] == {"width": 832, "height": 1248}
    assert arguments["seed"] == 3


def test_env_path_wins_over_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    _register(paths, "https://registry.example.com/lora.safetensors")
    monkeypatch.setenv("ZIR_LORA_PATH", "owner/repo")
    monkeypatch.setenv("ZIR_LORA_SCALE", "0.75")

    assert load_binding(paths) == LoraBinding("owner/repo", 0.75, name="ZIR_LORA_PATH")


def test_missing_binding_fails_with_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_lora_env(monkeypatch)
    with pytest.raises(LoraNotAvailableError) as error:
        resolved_model(get_model("sartiq-lora-klein"), _paths(tmp_path))

    message = str(error.value)
    assert "sartiq-lora-klein" in message and "train_fal.py" in message


def test_cache_key_covers_the_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_lora_env(monkeypatch)
    image = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 6), "red").save(image)
    spec = get_model("sartiq-lora-klein")
    first = attach_binding(spec, LoraBinding("https://a.example.com/lora.safetensors", 1.0))
    second = attach_binding(spec, LoraBinding("https://b.example.com/lora.safetensors", 1.0))

    request_a = GenerationRequest(first, "p", (image,), None)
    request_b = GenerationRequest(second, "p", (image,), None)
    assert request_a.cache_key() != request_b.cache_key()


def test_pipeline_takes_the_model_from_env_and_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setenv("ZIR_MODEL_KEY", "sartiq-lora-klein")
    _clear_lora_env(monkeypatch)
    paths = _paths(tmp_path)

    with pytest.raises(LoraNotAvailableError):
        StudioPipeline(paths=paths)

    _register(paths)
    pipeline = StudioPipeline(paths=paths)
    assert pipeline._settings.model_key == "sartiq-lora-klein"  # noqa: SLF001

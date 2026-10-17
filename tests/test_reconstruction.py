import io
from pathlib import Path
from typing import Any, cast

from PIL import Image

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult
from zir_image_enhancer.reconstruction.gate import check
from zir_image_enhancer.reconstruction.prompt import build_harness_prompt, build_relight_prompt
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.references import Reference
from zir_image_enhancer.reconstruction.spec import GarmentSpec
from zir_image_enhancer.reconstruction.vision import VisionAnswer, VisionClient


class FakeVision:
    """Answers by call kind, so the tests never touch the network."""

    def __init__(self, answers: dict[str, dict[str, object]]) -> None:
        self._answers = answers
        self.calls: list[str] = []

    def ask(self, system: str, prompt: str, images: Any, kind: str) -> VisionAnswer:
        self.calls.append(kind)
        return VisionAnswer(self._answers.get(kind, {}), 0.0, False)


class FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        buffer = io.BytesIO()
        Image.new("RGB", (8, 12), "white").save(buffer, format="PNG")
        return GenerationResult(request, buffer.getvalue(), False, 0.04, 1.0)


def _spec(front: Path, setup: str) -> GarmentSpec:
    return GarmentSpec(front, (), "T-shirt", "", "", "", (), photo_setup=setup)


def test_harness_prompt_flattens_mannequin_shape(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "front.jpg", "mannequin")
    prompt = build_harness_prompt(spec, [Reference(spec.front, "the front")])
    assert "No bust, waist or hip shaping" in prompt
    # Unknown setups add nothing instead of a wrong instruction.
    other = build_harness_prompt(_spec(spec.front, "other"), [Reference(spec.front, "x")])
    assert "No bust" not in other


def test_gate_reports_worn_shape(tmp_path: Path) -> None:
    vision = FakeVision({"gate": {"flat_shape": False, "fix": "flatten it"}})
    result = check(
        tmp_path / "a.jpg",
        tmp_path / "b.png",
        _spec(tmp_path / "a.jpg", "worn"),
        cast(VisionClient, vision),
    )
    assert result.problems == ("worn shape",)
    assert not result.passed


def test_relight_pass_uses_packshot_as_only_reference(tmp_path: Path) -> None:
    front = tmp_path / "01_front.jpg"
    Image.new("RGB", (8, 12), "red").save(front)
    vision = FakeVision({"spec": {"category": "T-shirt"}, "gate": {}})
    generator = FakeGenerator()
    settings = ReconstructionSettings(relight=True)
    reconstructor = Reconstructor(settings, generator, cast(VisionClient, vision), tmp_path)
    result = reconstructor.run(GarmentItem("g1", (front,), None), tmp_path / "work")

    assert len(generator.requests) == 2
    relight = generator.requests[1]
    assert relight.prompt == build_relight_prompt()
    assert relight.images == (tmp_path / "work" / "attempt_1.png",)
    assert result.final.name == "attempt_relight.png"

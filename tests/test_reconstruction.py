import io
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult
from zir_image_enhancer.generation.prompts import build_studio_prompt
from zir_image_enhancer.reconstruction.gate import check
from zir_image_enhancer.reconstruction.photos import Role, RolePhoto
from zir_image_enhancer.reconstruction.prompt import build_harness_prompt
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.references import Reference
from zir_image_enhancer.reconstruction.spec import GarmentSpec, extract_spec
from zir_image_enhancer.reconstruction.vision import VisionAnswer, VisionClient


class FakeVision:
    """Answers by call kind, so the tests never touch the network."""

    def __init__(self, answers: dict[str, dict[str, object]]) -> None:
        self._answers = answers
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def ask(self, system: str, prompt: str, images: Any, kind: str) -> VisionAnswer:
        self.calls.append(kind)
        self.prompts.append(prompt)
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


def test_gate_problem_triggers_one_repair_with_the_fix(tmp_path: Path) -> None:
    front = tmp_path / "01_front.jpg"
    Image.new("RGB", (8, 12), "red").save(front)
    vision = FakeVision(
        {"spec": {"category": "T-shirt"}, "gate": {"colour_ok": False, "fix": "redder"}}
    )
    generator = FakeGenerator()
    settings = ReconstructionSettings(mode="harness")
    reconstructor = Reconstructor(settings, generator, cast(VisionClient, vision))
    result = reconstructor.run(GarmentItem("g1", (front,), None), tmp_path / "work")

    assert len(generator.requests) == 2
    assert generator.requests[1].prompt.endswith("fix them: redder")
    # Same number of problems: the repair is kept.
    assert result.final.name == "attempt_2.png"


def test_spec_reads_the_pieces_of_a_set(tmp_path: Path) -> None:
    vision = FakeVision({"spec": {"category": "bikini", "pieces": ["top", "bottom", ""]}})
    spec = extract_spec((RolePhoto(tmp_path / "a.jpg", Role.UNKNOWN),), cast(VisionClient, vision))
    assert spec.pieces == ("top", "bottom")


def test_set_keeps_every_piece_in_prompt_and_gate(tmp_path: Path) -> None:
    front = tmp_path / "a.jpg"
    spec = GarmentSpec(front, (), "bikini", "", "", "", (), pieces=("bikini top", "bikini bottom"))
    prompt = build_harness_prompt(spec, [Reference(front, "the front")])
    assert "set of 2 pieces: bikini top and bikini bottom" in prompt

    vision = FakeVision({"gate": {"all_pieces": False}})
    result = check(front, tmp_path / "b.png", spec, cast(VisionClient, vision))
    assert "bikini top, bikini bottom" in vision.prompts[0]
    assert result.severe == ("missing piece",)


def test_single_garment_prompt_has_no_set_line(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "a.jpg", "flat")
    prompt = build_harness_prompt(spec, [Reference(spec.front, "the front")])
    assert "set of" not in prompt


def test_harness_prompt_does_not_flatten(tmp_path: Path) -> None:
    # Flattening lines cost 1.2 points on the holdout and changed details on mannequin shots.
    spec = _spec(tmp_path / "front.jpg", "mannequin")
    prompt = build_harness_prompt(spec, [Reference(spec.front, "the front")])
    assert "mannequin: flatten" not in prompt


def _front(tmp_path: Path) -> Path:
    front = tmp_path / "01_front.jpg"
    Image.new("RGB", (8, 12), "red").save(front)
    return front


def test_oneshot_mode_generates_once_from_the_front(tmp_path: Path) -> None:
    front = _front(tmp_path)
    vision = FakeVision({"spec": {"category": "T-shirt"}, "gate": {"colour_ok": False}})
    generator = FakeGenerator()
    settings = ReconstructionSettings(mode="oneshot")
    Reconstructor(settings, generator, cast(VisionClient, vision)).run(
        GarmentItem("g1", (front,), None), tmp_path / "work"
    )
    assert len(generator.requests) == 1
    assert generator.requests[0].prompt == build_studio_prompt()


def test_jury_mode_keeps_the_oneshot_on_a_tie(tmp_path: Path) -> None:
    front = _front(tmp_path)
    vision = FakeVision({"spec": {"category": "T-shirt"}, "gate": {"seams_ok": False}})
    generator = FakeGenerator()
    settings = ReconstructionSettings(mode="jury")
    result = Reconstructor(settings, generator, cast(VisionClient, vision)).run(
        GarmentItem("g1", (front,), None), tmp_path / "work"
    )
    assert len(generator.requests) == 2
    assert generator.requests[0].prompt == build_studio_prompt()
    assert "seams differ" in generator.requests[1].prompt
    # The repair edits the one-shot packshot, with the real photo as reference.
    assert generator.requests[1].images[0].name == "attempt_1.png"
    assert generator.requests[1].images[1] == front
    # Both attempts have the same defect, so the one-shot (better on the metric) stays.
    assert result.final.name == "attempt_1.png"


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        Reconstructor(
            ReconstructionSettings(mode="best"), FakeGenerator(), cast(VisionClient, FakeVision({}))
        )


def test_jury_mode_regenerates_for_a_colour_problem(tmp_path: Path) -> None:
    front = _front(tmp_path)
    vision = FakeVision({"spec": {"category": "T-shirt"}, "gate": {"colour_ok": False}})
    generator = FakeGenerator()
    settings = ReconstructionSettings(mode="jury")
    Reconstructor(settings, generator, cast(VisionClient, vision)).run(
        GarmentItem("g1", (front,), None), tmp_path / "work"
    )
    # A colour problem is global: the repair starts again from the photo, not the packshot.
    assert generator.requests[1].images[0] == front


def test_auto_style_uses_the_invisible_mannequin_only_for_mannequin_photos(tmp_path: Path) -> None:
    front = _front(tmp_path)
    for setup, expected in (("mannequin", "invisible-mannequin"), ("hanger", "laid flat")):
        vision = FakeVision({"spec": {"category": "top", "photo_setup": setup}, "gate": {}})
        generator = FakeGenerator()
        Reconstructor(
            ReconstructionSettings(mode="oneshot"), generator, cast(VisionClient, vision)
        ).run(GarmentItem("g1", (front,), None), tmp_path / setup)
        assert expected in generator.requests[0].prompt

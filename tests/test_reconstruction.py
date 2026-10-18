import io
from pathlib import Path
from typing import Any, cast

from PIL import Image

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult
from zir_image_enhancer.reconstruction.gate import check
from zir_image_enhancer.reconstruction.photos import Role, RolePhoto
from zir_image_enhancer.reconstruction.prompt import build_harness_prompt
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.references import Reference, detail_references
from zir_image_enhancer.reconstruction.spec import (
    GarmentSpec,
    PhotoAssessment,
    extract_spec,
    select_photos,
)
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
    reconstructor = Reconstructor(ReconstructionSettings(), generator, cast(VisionClient, vision))
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


def _photo(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    Image.new("RGB", (8, 12), "white").save(path)
    return path


def _scores(index: int, role: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "index": index,
        "role": role,
        "quality": 0.9,
        "garment_visibility": 0.9,
        "colour": 0.9,
        "texture": 0.9,
        "details": 0.9,
        "silhouette": 0.9,
        "geometry": 0.9,
        "note": "",
    }
    row.update(overrides)
    return row


def test_assessment_drops_a_label_photo(tmp_path: Path) -> None:
    front = _photo(tmp_path, "a.jpg")
    label = _photo(tmp_path, "b.jpg")
    vision = FakeVision(
        {
            "assess": {
                "images": [
                    _scores(1, "front"),
                    _scores(
                        2,
                        "label",
                        quality=0.9,
                        garment_visibility=0.05,
                        colour=0.8,
                        texture=0.1,
                        details=0.1,
                        silhouette=0.0,
                        geometry=0.0,
                    ),
                ]
            },
            "spec": {"category": "T-shirt", "front_index": 1},
        }
    )
    spec = extract_spec(
        (RolePhoto(front, Role.UNKNOWN), RolePhoto(label, Role.UNKNOWN)),
        cast(VisionClient, vision),
    )
    assert [photo.path for photo in spec.roles] == [front]
    assert [a.path for a in spec.dropped] == [str(label)]


def test_pipeline_names_detail_closeups_and_sends_them(tmp_path: Path) -> None:
    front = _photo(tmp_path, "a.jpg")
    logo = _photo(tmp_path, "b.jpg")
    vision = FakeVision(
        {
            "assess": {"images": [_scores(1, "front"), _scores(2, "detail")]},
            "spec": {
                "category": "T-shirt",
                "front_index": 1,
                "roles": [
                    {"index": 1, "role": "front"},
                    {"index": 2, "role": "detail", "name": "chest logo"},
                ],
            },
        }
    )
    spec = extract_spec(
        (RolePhoto(front, Role.UNKNOWN), RolePhoto(logo, Role.UNKNOWN)),
        cast(VisionClient, vision),
    )
    refs = detail_references(spec, 3)
    assert [ref.path for ref in refs] == [logo]
    assert refs[0].label == "a close-up of the chest logo"


def test_assessment_keeps_a_whole_garment_back_view(tmp_path: Path) -> None:
    front = _photo(tmp_path, "a.jpg")
    back = _photo(tmp_path, "b.jpg")
    vision = FakeVision(
        {
            "assess": {
                "images": [
                    _scores(1, "front"),
                    _scores(
                        2,
                        "back",
                        quality=0.3,
                        garment_visibility=0.6,
                        colour=0.3,
                        texture=0.2,
                        details=0.1,
                        silhouette=0.4,
                        geometry=0.4,
                    ),
                ]
            },
            "spec": {"category": "T-shirt"},
        }
    )
    spec = extract_spec(
        (RolePhoto(front, Role.UNKNOWN), RolePhoto(back, Role.UNKNOWN)),
        cast(VisionClient, vision),
    )
    assert {photo.path for photo in spec.roles} == {front, back}


def test_assessment_never_drops_every_photo(tmp_path: Path) -> None:
    first = _photo(tmp_path, "a.jpg")
    second = _photo(tmp_path, "b.jpg")
    zero = dict(
        quality=0.0,
        garment_visibility=0.0,
        colour=0.0,
        texture=0.0,
        details=0.0,
        silhouette=0.0,
        geometry=0.0,
    )
    kept, _ = select_photos(
        (RolePhoto(first, Role.UNKNOWN), RolePhoto(second, Role.UNKNOWN)),
        (
            PhotoAssessment(1, str(first), role="other", **zero),
            PhotoAssessment(2, str(second), role="artifact", **zero),
        ),
    )
    assert len(kept) == 1


def test_redundant_photo_is_worth_half() -> None:
    full = dict(
        role="detail",
        quality=1.0,
        garment_visibility=1.0,
        colour=1.0,
        texture=1.0,
        details=1.0,
        silhouette=1.0,
        geometry=1.0,
    )
    best = PhotoAssessment(1, "a.jpg", **full)
    duplicate = PhotoAssessment(2, "b.jpg", redundant=True, **full)
    assert duplicate.informational_value == round(best.informational_value * 0.5, 3)


def test_single_photo_skips_the_assessment_call(tmp_path: Path) -> None:
    photo = _photo(tmp_path, "a.jpg")
    vision = FakeVision({"spec": {"category": "T-shirt"}})
    extract_spec((RolePhoto(photo, Role.UNKNOWN),), cast(VisionClient, vision))
    assert "assess" not in vision.calls


def test_harness_prompt_does_not_flatten(tmp_path: Path) -> None:
    # Flattening lines cost 1.2 points on the holdout and changed details on mannequin shots.
    spec = _spec(tmp_path / "front.jpg", "mannequin")
    prompt = build_harness_prompt(spec, [Reference(spec.front, "the front")])
    assert "mannequin: flatten" not in prompt

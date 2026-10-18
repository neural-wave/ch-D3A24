import io
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.pipeline import MissingKeyError, StudioPipeline, prepare_photos
from zir_image_enhancer.reconstruction.photos import Role, role_from_filename
from zir_image_enhancer.reconstruction.vision import VisionAnswer, VisionClient


class FakeVision:
    def __init__(self, answers: dict[str, dict[str, object]]) -> None:
        self._answers = answers

    def ask(self, system: str, prompt: str, images: Any, kind: str) -> VisionAnswer:
        return VisionAnswer(self._answers.get(kind, {}), 0.0, False)


class FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        buffer = io.BytesIO()
        Image.new("RGB", (8, 12), "white").save(buffer, format="PNG")
        return GenerationResult(request, buffer.getvalue(), False, 0.04, 1.0)


def _photo(path: Path, mode: str = "RGB") -> Path:
    Image.new(mode, (6, 4)).save(path)
    return path


def test_roles_typed_in_the_app_become_file_names(tmp_path: Path) -> None:
    photos = [_photo(tmp_path / "IMG_1.png", "RGBA"), _photo(tmp_path / "IMG_2.png")]
    prepared = prepare_photos(photos, tmp_path / "out", ["front", "Detail chest logo"])
    assert [p.name for p in prepared] == ["01_front.jpg", "02_detail-chest-logo.jpg"]
    detail = role_from_filename(prepared[1])
    assert detail.role is Role.DETAIL and detail.name == "chest logo"


def test_named_files_keep_their_role_and_others_keep_their_order(tmp_path: Path) -> None:
    photos = [_photo(tmp_path / "02_back.jpg"), _photo(tmp_path / "IMG 7.jpg")]
    prepared = prepare_photos(photos, tmp_path / "out")
    assert [p.name for p in prepared] == ["02_back.jpg", "02_img-7.jpg"]
    assert Image.open(prepared[0]).mode == "RGB"


def test_a_clean_jpeg_is_copied_and_others_are_converted(tmp_path: Path) -> None:
    jpeg, png = _photo(tmp_path / "a.jpg"), _photo(tmp_path / "b.png", "RGBA")
    prepared = prepare_photos([jpeg, png], tmp_path / "out")
    # Same bytes, so every cached call made with the original photo is reused.
    assert prepared[0].read_bytes() == jpeg.read_bytes()
    with Image.open(prepared[1]) as converted:
        assert (converted.format, converted.mode) == ("JPEG", "RGB")


def test_pipeline_needs_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        StudioPipeline(paths=ProjectPaths(tmp_path))


def test_run_returns_the_packshot_and_what_it_decided(tmp_path: Path) -> None:
    generator = FakeGenerator()
    pipeline = StudioPipeline(
        paths=ProjectPaths(tmp_path),
        generator=generator,
        vision=cast(VisionClient, FakeVision({"spec": {"category": "T-shirt"}})),
    )
    result = pipeline.run([_photo(tmp_path / "front.jpg")], "g1")
    assert len(generator.requests) == 1
    assert result.image == tmp_path / "local" / "app_runs" / "g1" / "attempts" / "attempt_1.png"
    assert "Product: T-shirt" in result.log and "Kept: attempt_1" in result.log

from pathlib import Path

import pytest
from PIL import Image

from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.pipeline import MissingKeyError, StudioPipeline, prepare_photos
from zir_image_enhancer.reconstruction.photos import Role, role_from_filename


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


def test_pipeline_needs_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    paths = ProjectPaths(tmp_path, tmp_path, tmp_path, tmp_path)
    with pytest.raises(MissingKeyError):
        StudioPipeline(paths=paths)

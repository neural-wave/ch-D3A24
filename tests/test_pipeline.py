from pathlib import Path

import pytest
from PIL import Image

from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.pipeline import MissingKeyError, StudioPipeline, prepare_photos


def _photo(path: Path, mode: str = "RGB") -> Path:
    Image.new(mode, (6, 4)).save(path)
    return path


def test_prepare_photos_normalises_and_numbers_in_order(tmp_path: Path) -> None:
    photos = [_photo(tmp_path / "IMG_1.png", "RGBA"), _photo(tmp_path / "IMG_2.png")]
    prepared = prepare_photos(photos, tmp_path / "out")
    assert [p.name for p in prepared] == ["01_img-1.jpg", "02_img-2.jpg"]
    assert Image.open(prepared[0]).mode == "RGB"


def test_file_names_do_not_carry_roles(tmp_path: Path) -> None:
    # "02_back.jpg" is just a name now; the pipeline decides what the photo is.
    photos = [_photo(tmp_path / "02_back.jpg"), _photo(tmp_path / "IMG 7.jpg")]
    prepared = prepare_photos(photos, tmp_path / "out")
    assert [p.name for p in prepared] == ["01_02-back.jpg", "02_img-7.jpg"]
    assert Image.open(prepared[0]).mode == "RGB"


def test_pipeline_needs_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    paths = ProjectPaths(tmp_path, tmp_path, tmp_path)
    with pytest.raises(MissingKeyError):
        StudioPipeline(paths=paths)

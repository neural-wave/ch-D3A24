"""Tests for the OpenCV preprocessing: no network, all images are built in memory."""

import io
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult
from zir_image_enhancer.reconstruction.preprocess import (
    PreprocessSettings,
    add_grain,
    auto_tone,
    box_to_pixels,
    crop_around,
    dominant_colours,
    fit_max_side,
    flatten_illumination,
    garment_mask,
    gray_world,
    image_stats,
    mask_bbox,
    neutral_white_balance,
    prepare_references,
    read_image,
    to_u8,
    write_image,
)
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.vision import VisionAnswer, VisionClient


def _save(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def _solid(height: int, width: int, colour: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    return image


def test_read_write_roundtrip(tmp_path: Path) -> None:
    original = _solid(20, 30, (10, 200, 30))
    path = write_image(tmp_path / "a.jpg", original, quality=100)
    again = read_image(path)
    assert again.shape == (20, 30, 3)
    assert np.abs(again.astype(int) - original.astype(int)).mean() < 4  # JPEG is lossy


def test_fit_max_side_downscales_long_edge_only() -> None:
    assert fit_max_side(_solid(100, 50, (0, 0, 0)), 40).shape[:2] == (40, 20)
    assert fit_max_side(_solid(20, 10, (0, 0, 0)), 40).shape[:2] == (20, 10)


def test_to_u8_casts_dtype() -> None:
    assert to_u8(np.array([[300.0, -5.0]])).dtype == np.uint8


def test_gray_world_removes_a_colour_cast() -> None:
    warm = _solid(40, 40, (150, 140, 120))
    fixed = gray_world(warm, strength=1.0)
    means = fixed.reshape(-1, 3).mean(axis=0)
    assert (means.max() - means.min()) < 4  # channels evened out
    assert image_stats(warm)["cast_b"] > 0  # the cast was real to begin with


def test_gray_world_clamps_so_a_strong_colour_survives() -> None:
    red = _solid(40, 40, (200, 20, 20))
    fixed = gray_world(red, strength=1.0, max_gain=0.3).reshape(-1, 3).astype(float).mean(axis=0)
    assert 139 <= fixed[0] <= 141  # red only reduced by the 30% cap
    assert fixed[1] <= 26.5  # green only raised by the cap


def test_gray_world_off_is_a_no_op() -> None:
    image = _solid(8, 8, (10, 20, 30))
    assert np.array_equal(gray_world(image, strength=0.0), image)


def test_neutral_white_balance_removes_a_cast() -> None:
    warm = _solid(40, 40, (150, 140, 120))
    fixed = neutral_white_balance(warm, strength=1.0)
    means = fixed.reshape(-1, 3).mean(axis=0)
    assert (means.max() - means.min()) < 4  # channels evened out


def test_neutral_white_balance_leaves_a_saturated_garment_alone() -> None:
    # A red garment on a near-neutral grey floor: the backdrop is what gets measured, so the
    # red must not be dragged towards grey the way a whole-scene average would.
    scene = _solid(120, 120, (185, 185, 185))
    scene[20:100, 20:100] = (200, 20, 20)
    balanced = neutral_white_balance(scene, strength=1.0)
    red = balanced[60, 60].astype(int)
    assert red[0] > 190 and red[1] < 35 and red[2] < 35
    assert abs(int(balanced[5, 5, 0]) - int(balanced[5, 5, 2])) < 6  # grey stays grey


def test_neutral_white_balance_off_is_a_no_op() -> None:
    image = _solid(8, 8, (10, 20, 30))
    assert np.array_equal(neutral_white_balance(image, strength=0.0), image)


def test_flatten_illumination_evens_a_light_gradient() -> None:
    # A dark-to-bright ramp stands in for the photographer's shadow or a lamp on one side.
    ramp = np.tile(np.linspace(60, 200, 200, dtype=np.uint8), (200, 1))
    scene = np.repeat(ramp[:, :, None], 3, axis=2)
    before = scene.mean(axis=2).mean(axis=1)  # column means, left dark -> right bright
    after = flatten_illumination(scene, strength=1.0).mean(axis=2).mean(axis=1)
    assert after.std() < before.std() * 0.25  # the ramp is mostly gone
    assert abs(after.mean() - before.mean()) < 8  # global exposure is preserved


def test_flatten_illumination_keeps_hue() -> None:
    scene = _solid(80, 80, (190, 90, 60))
    flattened = flatten_illumination(scene, strength=1.0)
    ratio_before = scene[..., 0].mean() / scene[..., 2].mean()
    ratio_after = flattened[..., 0].mean() / flattened[..., 2].mean()
    assert abs(ratio_after - ratio_before) < 0.05


def test_flatten_illumination_off_is_a_no_op() -> None:
    image = _solid(8, 8, (10, 20, 30))
    assert np.array_equal(flatten_illumination(image, strength=0.0), image)


def test_auto_tone_stretches_a_flat_image() -> None:
    flat = np.linspace(100, 150, 40 * 40 * 3).reshape(40, 40, 3).astype(np.uint8)
    stretched = auto_tone(flat)
    assert int(stretched.max()) - int(stretched.min()) > 200  # near full range now


def test_add_grain_is_reproducible_and_seed_dependent() -> None:
    image = _solid(30, 30, (120, 120, 120))
    first = add_grain(image, sigma=3.0, seed=7)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert np.array_equal(first, add_grain(image, sigma=3.0, seed=7))
    assert not np.array_equal(first, add_grain(image, sigma=3.0, seed=8))
    assert np.array_equal(image, add_grain(image, sigma=0.0, seed=7))


def test_garment_mask_finds_a_centred_square() -> None:
    scene = _solid(200, 200, (130, 130, 130))
    scene[50:150, 50:150] = (240, 240, 240)
    mask = garment_mask(scene)
    assert mask is not None
    assert 0.20 <= float((mask > 0).mean()) <= 0.30
    assert mask_bbox(mask) == (49, 49, 150, 150) or mask_bbox(mask) == (50, 50, 149, 149)


def test_garment_mask_gives_up_on_a_uniform_image() -> None:
    assert garment_mask(_solid(120, 120, (128, 128, 128))) is None


def test_dominant_colours_reads_the_garment_colour() -> None:
    scene = _solid(120, 120, (135, 135, 135))
    scene[30:90, 30:90] = (220, 20, 20)
    mask = garment_mask(scene)
    palette = dominant_colours(scene, mask, k=3)
    hex_code = palette[0][0]
    red, green, blue = (int(hex_code[i : i + 2], 16) for i in (1, 3, 5))
    assert red > 180 and green < 70 and blue < 70
    assert palette[0][1] > 0.5


def test_box_to_pixels_pads_and_clamps() -> None:
    assert box_to_pixels((250, 300, 450, 600), 1000, 1000, 0.3) == (210, 160, 690, 540)
    assert box_to_pixels((-50, -50, 2000, 2000), 100, 100, 0.5) == (0, 0, 100, 100)


def test_crop_around_adds_a_margin() -> None:
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    assert crop_around(image, (100, 100, 200, 200), margin=0.5).shape[:2] == (200, 200)


def test_prepare_references_writes_front_and_detail_crop(tmp_path: Path) -> None:
    photo = _save(tmp_path / "front.jpg", _solid(320, 240, (90, 90, 90)))
    prepared = prepare_references(
        front=photo,
        out_dir=tmp_path / "refs",
        settings=PreprocessSettings(),
        details=(("chest logo", (100, 300, 300, 600)),),
    )
    assert prepared.front.exists() and prepared.front.name == "front.jpg"
    assert [name for name, _ in prepared.closeups] == ["chest logo"]
    assert prepared.closeups[0][1].exists()
    assert prepared.stats["width"] == 240.0 and prepared.stats["height"] == 320.0


def test_prepare_references_detail_crops_can_be_disabled(tmp_path: Path) -> None:
    photo = _save(tmp_path / "front.jpg", _solid(320, 240, (90, 90, 90)))
    prepared = prepare_references(
        front=photo,
        out_dir=tmp_path / "refs",
        settings=PreprocessSettings(detail_crops=False),
        details=(("chest logo", (100, 300, 300, 600)),),
    )
    assert prepared.closeups == ()


def test_crop_to_garment_trims_the_reference_and_shifts_detail_crops(tmp_path: Path) -> None:
    scene = _solid(400, 400, (205, 205, 205))
    scene[100:300, 120:280] = (30, 30, 200)  # a blue garment on a grey floor
    photo = _save(tmp_path / "front.jpg", scene)
    details = (("graphic", (300, 350, 700, 650)),)

    plain = prepare_references(front=photo, out_dir=tmp_path / "a", settings=PreprocessSettings())
    tight = prepare_references(
        front=photo,
        out_dir=tmp_path / "b",
        settings=PreprocessSettings(crop_to_garment=True),
        details=details,
    )

    assert read_image(tight.front).shape[0] < read_image(plain.front).shape[0]
    assert read_image(tight.front).shape[1] < read_image(plain.front).shape[1]
    assert [name for name, _ in tight.closeups] == ["graphic"]  # box survived the shift
    assert tight.closeups[0][1].exists()


def test_prepare_references_names_user_detail_photos(tmp_path: Path) -> None:
    front = _save(tmp_path / "front.jpg", _solid(320, 240, (90, 90, 90)))
    logo = _save(tmp_path / "logo.jpg", _solid(200, 200, (200, 40, 40)))
    prepared = prepare_references(
        front=front,
        out_dir=tmp_path / "refs",
        settings=PreprocessSettings(),
        detail_photos=(("chest logo", logo),),
    )
    assert [name for name, _ in prepared.closeups] == ["chest logo"]
    assert prepared.closeups[0][1].name == "photo_01.jpg"


# --- integration with the reconstructor ------------------------------------------------------


class _FakeVision:
    def __init__(self, answers: dict[str, dict[str, object]]) -> None:
        self._answers = answers

    def ask(self, system: str, prompt: str, images: Any, kind: str) -> VisionAnswer:
        return VisionAnswer(self._answers.get(kind, {}), 0.0, False)


class _FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        buffer = io.BytesIO()
        Image.new("RGB", (8, 12), "white").save(buffer, format="PNG")
        return GenerationResult(request, buffer.getvalue(), False, 0.04, 1.0)


def _front(tmp_path: Path) -> Path:
    return _save(tmp_path / "01_front.jpg", _solid(360, 240, (95, 90, 85)))


def test_reconstructor_sends_enhanced_references_and_detail_crops(tmp_path: Path) -> None:
    front = _front(tmp_path)
    vision = _FakeVision(
        {
            "spec": {
                "category": "T-shirt",
                "details": [
                    {"name": "chest logo", "description": "ACE", "box": [200, 300, 400, 600]}
                ],
            }
        }
    )
    generator = _FakeGenerator()
    reconstructor = Reconstructor(
        ReconstructionSettings(), generator, cast(VisionClient, vision)
    )
    result = reconstructor.run(GarmentItem("g1", (front,), None), tmp_path / "work")

    sent = [Path(image) for image in generator.requests[0].images]
    assert sent[0].parent.name == "references" and sent[0].name == "front.jpg"
    assert any(path.name.startswith("crop_") for path in sent)
    assert result.preprocess_stats["width"] == 240.0
    assert front not in sent  # the original is not sent when preprocessing is on


def test_reconstructor_sends_the_original_when_preprocessing_is_off(tmp_path: Path) -> None:
    front = _front(tmp_path)
    vision = _FakeVision({"spec": {"category": "T-shirt"}})
    generator = _FakeGenerator()
    settings = ReconstructionSettings(preprocess=PreprocessSettings(enabled=False))
    reconstructor = Reconstructor(settings, generator, cast(VisionClient, vision))
    reconstructor.run(GarmentItem("g1", (front,), None), tmp_path / "work")

    sent = [Path(image) for image in generator.requests[0].images]
    assert sent == [front]

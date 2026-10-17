import io
import json
from pathlib import Path

import pytest
from PIL import Image

from zir_image_enhancer.dataset import load_manifest
from zir_image_enhancer.generation.fal_backend import (
    BudgetExceededError,
    CostLedger,
    FalGenerator,
    GenerationRequest,
    to_png_bytes,
)
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.generation.prompts import build_studio_prompt


def _jpeg(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 96), colour).save(path, format="JPEG")
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _jpeg(root / "train/t0001/input_1.jpg", (200, 10, 10))
    _jpeg(root / "train/t0001/target.jpg", (220, 220, 220))
    for index in range(1, 4):
        _jpeg(root / f"dev/d001/input_{index}.jpg", (10, 10 * index, 200))
    (root / "train/pairs.jsonl").write_text(
        json.dumps(
            {
                "pair_id": "t0001",
                "inputs": ["train/t0001/input_1.jpg"],
                "target": "train/t0001/target.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "dev/inputs.jsonl").write_text(
        json.dumps(
            {
                "garment_id": "d001",
                "view": "front",
                "inputs": [f"dev/d001/input_{i}.jpg" for i in range(1, 4)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_load_manifest_handles_pairs_and_inputs(dataset: Path) -> None:
    pairs = load_manifest(dataset / "train/pairs.jsonl", dataset)
    garments = load_manifest(dataset / "dev/inputs.jsonl", dataset)
    assert pairs[0].item_id == "t0001" and pairs[0].target is not None
    assert garments[0].item_id == "d001" and garments[0].target is None
    assert len(garments[0].inputs) == 3


def test_cache_key_depends_on_sample_and_image_bytes(dataset: Path) -> None:
    model = get_model("nano-banana-2.1")
    image = dataset / "train/t0001/input_1.jpg"
    first = GenerationRequest(model, "p", (image,), None, sample=0)
    assert first.cache_key() == GenerationRequest(model, "p", (image,), None, sample=0).cache_key()
    assert first.cache_key() != GenerationRequest(model, "p", (image,), None, sample=1).cache_key()


def test_portrait_model_sends_seed_and_canvas() -> None:
    arguments = get_model("nano-banana-2.1-portrait").build_arguments("p", ["u"], 7)
    assert arguments["seed"] == 7 and arguments["aspect_ratio"] == "2:3"


def test_budget_guard_refuses_before_calling(dataset: Path, tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "ledger.jsonl")
    ledger.record({"cost_usd": 69.99})

    class ExplodingClient:
        def subscribe(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise AssertionError("must not be called once the budget is exhausted")

    generator = FalGenerator(tmp_path / "cache", ledger, budget_usd=70.0, client=ExplodingClient())  # type: ignore[arg-type]
    request = GenerationRequest(
        get_model("nano-banana-2.1"), "p", (dataset / "train/t0001/input_1.jpg",), None
    )
    with pytest.raises(BudgetExceededError):
        generator.generate(request)


def test_to_png_flattens_transparency_onto_white() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(buffer, format="PNG")
    pixel = Image.open(io.BytesIO(to_png_bytes(buffer.getvalue()))).getpixel((0, 0))
    assert pixel == (255, 255, 255)


def test_detail_prompt_names_the_closeups() -> None:
    prompt = build_studio_prompt(["chest logo", "small chest text"])
    assert "Images 2 to 3 are close-ups" in prompt and "small chest text" in prompt
    assert "close-ups" not in build_studio_prompt()


def test_gate_repairs_only_severe_problems() -> None:
    from zir_image_enhancer.reconstruction.gate import GateResult

    minor = GateResult(False, ("detail wrong: buttons", "colour changed"), "", 0.0)
    severe = GateResult(False, ("extra items", "colour changed"), "", 0.0)
    assert minor.severe == () and severe.severe == ("extra items",)


def test_photo_roles_from_file_names() -> None:
    from zir_image_enhancer.reconstruction.photos import Role, role_from_filename

    detail = role_from_filename(Path("03_detail_chest-logo.jpg"))
    assert detail.role is Role.DETAIL and detail.name == "chest logo"
    assert role_from_filename(Path("01_front.jpg")).role is Role.FRONT
    assert role_from_filename(Path("input_1.jpg")).role is Role.UNKNOWN


def test_small_holdout_is_a_stable_half_of_the_holdout(tmp_path: Path) -> None:
    from zir_image_enhancer.dataset import split_manifest
    from zir_image_enhancer.paths import ProjectPaths

    paths = ProjectPaths(tmp_path, tmp_path, tmp_path / "starter", tmp_path)
    paths.holdout_val.parent.mkdir(parents=True)
    rows = [json.dumps({"pair_id": f"t{i:04d}", "inputs": []}) for i in range(6)]
    paths.holdout_val.write_text("\n".join(rows) + "\n", encoding="utf-8")
    small = split_manifest("holdout-small", paths).read_text(encoding="utf-8").splitlines()
    assert len(small) == 3 and set(small) <= set(rows)
    # Written once: a second call returns the same pairs.
    assert split_manifest("holdout-small", paths).read_text(encoding="utf-8").splitlines() == small

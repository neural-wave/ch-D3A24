import io
from pathlib import Path

import pytest
from PIL import Image

from zir_image_enhancer.generation.fal_backend import (
    BudgetExceededError,
    CostLedger,
    FalGenerator,
    GenerationRequest,
    to_png_bytes,
)
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.generation.prompts import build_studio_prompt
from zir_image_enhancer.reconstruction.gate import GateResult
from zir_image_enhancer.reconstruction.photos import Role, role_from_filename

MODEL = "nano-banana-2.1-portrait"


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    path = tmp_path / "input_1.jpg"
    Image.new("RGB", (64, 96), (200, 10, 10)).save(path, format="JPEG")
    return path


def test_cache_key_depends_on_sample_and_image_bytes(photo: Path) -> None:
    model = get_model(MODEL)
    first = GenerationRequest(model, "p", (photo,), None, sample=0)
    assert first.cache_key() == GenerationRequest(model, "p", (photo,), None, sample=0).cache_key()
    assert first.cache_key() != GenerationRequest(model, "p", (photo,), None, sample=1).cache_key()


def test_portrait_model_sends_seed_and_canvas() -> None:
    arguments = get_model(MODEL).build_arguments("p", ["u"], 7)
    assert arguments["seed"] == 7 and arguments["aspect_ratio"] == "2:3"


def test_budget_guard_refuses_before_calling(photo: Path, tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "ledger.jsonl")
    ledger.record({"cost_usd": 69.99})

    class ExplodingClient:
        def subscribe(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise AssertionError("must not be called once the budget is exhausted")

    generator = FalGenerator(tmp_path / "cache", ledger, budget_usd=70.0, client=ExplodingClient())  # type: ignore[arg-type]
    with pytest.raises(BudgetExceededError):
        generator.generate(GenerationRequest(get_model(MODEL), "p", (photo,), None))


def test_to_png_flattens_transparency_onto_white() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(buffer, format="PNG")
    pixel = Image.open(io.BytesIO(to_png_bytes(buffer.getvalue()))).getpixel((0, 0))
    assert pixel == (255, 255, 255)


def test_detail_prompt_names_the_closeups() -> None:
    prompt = build_studio_prompt(["chest logo", "small chest text"])
    assert "Images 2 to 3 are close-ups" in prompt and "small chest text" in prompt
    assert "close-ups" not in build_studio_prompt()


def test_severe_problems_count_double() -> None:
    minor = GateResult(("detail wrong: buttons", "colour changed"), "", 0.0)
    severe = GateResult(("extra items", "colour changed"), "", 0.0)
    assert minor.severe == () and severe.severe == ("extra items",)
    assert (minor.weight, severe.weight) == (2, 3)


def test_photo_roles_from_file_names() -> None:
    detail = role_from_filename(Path("03_detail_chest-logo.jpg"))
    assert detail.role is Role.DETAIL and detail.name == "chest logo"
    assert role_from_filename(Path("01_front.jpg")).role is Role.FRONT
    assert role_from_filename(Path("input_1.jpg")).role is Role.UNKNOWN

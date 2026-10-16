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
    GenerationResult,
    to_png_bytes,
)
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.generation.prompts import build_prompt
from zir_image_enhancer.generation.runner import (
    GenerationRun,
    InputSelection,
    RunConfig,
    select_inputs,
)


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


class FakeGenerator:
    """Returns a grey PNG and counts calls, so runs can be tested without the network."""

    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (128, 128, 128)).save(buffer, format="PNG")
        return GenerationResult(request, buffer.getvalue(), False, request.model.est_cost_usd, 0.1)


def test_load_manifest_handles_pairs_and_inputs(dataset: Path) -> None:
    pairs = load_manifest(dataset / "train/pairs.jsonl", dataset)
    garments = load_manifest(dataset / "dev/inputs.jsonl", dataset)
    assert pairs[0].item_id == "t0001" and pairs[0].target is not None
    assert garments[0].item_id == "d001" and garments[0].target is None
    assert len(garments[0].inputs) == 3


def test_baseline_prompt_is_verbatim_and_v1_mentions_all_photos() -> None:
    assert build_prompt("baseline", 3) == build_prompt("baseline", 1)
    assert "All 3 photos" in build_prompt("studio-white-v1", 3)
    assert "All" not in build_prompt("studio-white-v1", 1)


def test_cache_key_depends_on_sample_and_image_bytes(dataset: Path) -> None:
    model = get_model("nano-banana-2.1")
    image = dataset / "train/t0001/input_1.jpg"
    first = GenerationRequest(model, "p", (image,), None, sample=0)
    assert first.cache_key() == GenerationRequest(model, "p", (image,), None, sample=0).cache_key()
    assert first.cache_key() != GenerationRequest(model, "p", (image,), None, sample=1).cache_key()


def test_seed_only_sent_to_models_that_take_it() -> None:
    assert "seed" in get_model("nano-banana-pro").build_arguments("p", ["u"], 7)
    assert get_model("flux-2").build_arguments("p", ["u"], 7)["seed"] == 7


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


def test_run_writes_scorer_layout_and_respects_input_selection(
    dataset: Path, tmp_path: Path
) -> None:
    items = load_manifest(dataset / "dev/inputs.jsonl", dataset)
    fake = FakeGenerator()
    config = RunConfig(
        "test_run", "flux-2", "studio-white-v1", InputSelection.ALL, samples=2, seed_base=10
    )
    summary = GenerationRun(config, fake, tmp_path / "runs").execute(items)

    assert summary.generated == 2 and summary.failed == 0
    assert (tmp_path / "runs/test_run/cand_0/d001/front.png").exists()
    assert (tmp_path / "runs/test_run/cand_1/d001/front.png").exists()
    assert len(fake.calls[0].images) == 3
    assert [call.seed for call in fake.calls] == [10, 11]


def test_parallel_run_writes_every_candidate(dataset: Path, tmp_path: Path) -> None:
    items = load_manifest(dataset / "dev/inputs.jsonl", dataset)
    fake = FakeGenerator()
    config = RunConfig("parallel", "flux-2", "studio-white-v1", InputSelection.FIRST, samples=4)
    summary = GenerationRun(config, fake, tmp_path / "runs").execute(items, workers=3)

    assert summary.generated == 4
    meta_lines = (tmp_path / "runs/parallel/meta.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(meta_lines) == 4
    assert all((tmp_path / f"runs/parallel/cand_{k}/d001/front.png").exists() for k in range(4))


def test_to_png_flattens_transparency_onto_white() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(buffer, format="PNG")
    pixel = Image.open(io.BytesIO(to_png_bytes(buffer.getvalue()))).getpixel((0, 0))
    assert pixel == (255, 255, 255)


def test_filtered_inputs_use_the_selector_and_respect_model_limit(dataset: Path) -> None:
    item = load_manifest(dataset / "dev/inputs.jsonl", dataset)[0]
    reversed_selector = lambda photos: tuple(reversed(photos))  # noqa: E731
    chosen = select_inputs(item, InputSelection.FILTERED, get_model("flux-2"), reversed_selector)
    assert chosen[0].name == "input_3.jpg"
    with pytest.raises(ValueError):
        select_inputs(item, InputSelection.FILTERED, get_model("flux-2"))


def test_best_input_is_the_selectors_first_choice(dataset: Path) -> None:
    item = load_manifest(dataset / "dev/inputs.jsonl", dataset)[0]
    reversed_selector = lambda photos: tuple(reversed(photos))  # noqa: E731
    chosen = select_inputs(item, InputSelection.BEST, get_model("flux-2"), reversed_selector)
    assert [path.name for path in chosen] == ["input_3.jpg"]


def test_grey_card_removes_a_warm_cast() -> None:
    np = pytest.importorskip("numpy")
    from zir_image_enhancer.preprocess.white_balance import CardSettings, GreyCardBalancer

    # A warm scene: grey card and backdrop both pushed towards orange.
    pixels = np.full((400, 600, 3), (200, 170, 140), dtype=np.uint8)
    pixels[150:250, 200:400] = (40, 90, 160)  # the garment
    pixels[330:390, 530:590] = (150, 120, 90)  # the card, bottom-right
    balancer = GreyCardBalancer(CardSettings(corner="bottom-right", size=0.15, inset=0.02))
    result = balancer.balance(Image.fromarray(pixels))
    assert result.applied
    card = (
        np.asarray(
            GreyCardBalancer(CardSettings(cover=False)).balance(Image.fromarray(pixels)).image
        )[350:370, 550:570]
        .reshape(-1, 3)
        .mean(axis=0)
    )
    assert card.max() - card.min() < 4  # neutral after balancing


def test_grey_card_left_alone_when_no_card() -> None:
    np = pytest.importorskip("numpy")
    from zir_image_enhancer.preprocess.white_balance import GreyCardBalancer

    noise = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    assert not GreyCardBalancer().balance(Image.fromarray(noise)).applied

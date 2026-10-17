import io
import zlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import GenerationRequest, GenerationResult
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.pipeline import StudioPipeline
from zir_image_enhancer.reconstruction.spec import GarmentSpec
from zir_image_enhancer.reconstruction.vision import VisionAnswer, VisionClient
from zir_image_enhancer.selection import V4_VERSIONS, Candidate, VersionSelector, gate_pick

CLEAN: dict[str, object] = {}
COLOUR = {"colour_ok": False}  # one problem
CROPPED = {"whole_visible": False}  # a severe problem: counts double


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class FakeEmbedder:
    """A fixed random vector per image path, so the tests need no torch."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._style = (_unit(rng.normal(size=(20, 8))), _unit(rng.normal(size=(20, 8))))

    def embed(self, images: Any) -> tuple[np.ndarray, np.ndarray]:
        vectors = [
            np.random.default_rng(zlib.crc32(str(p).encode())).normal(size=(2, 8)) for p in images
        ]
        return _unit(np.array([v[0] for v in vectors])), _unit(np.array([v[1] for v in vectors]))

    def house_style(self) -> tuple[np.ndarray, np.ndarray]:
        return self._style


class SequenceVision:
    """Gate answers in the order given, one per check."""

    def __init__(self, answers: list[dict[str, object]]) -> None:
        self._answers = list(answers)

    def ask(self, system: str, prompt: str, images: Any, kind: str) -> VisionAnswer:
        return VisionAnswer(self._answers.pop(0), 0.001, False)


class FakeVision:
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


class ReverseRanker:
    """Ranks the versions backwards, so the pick is not just the first one generated."""

    def rank(self, items: Any, candidates: Any) -> dict[str, list[Candidate]]:
        return {item.item_id: list(reversed(candidates[item.item_id])) for item in items}


def test_fixed_scales_rank_one_garment_like_the_whole_pool(tmp_path: Path) -> None:
    items = [GarmentItem(f"g{i}", (tmp_path / f"g{i}.jpg",), None) for i in range(4)]
    candidates = {
        item.item_id: [Candidate(v, tmp_path / item.item_id / f"{v}.png") for v in "abcde"]
        for item in items
    }
    experts = FakeEmbedder()
    batch = VersionSelector(experts)
    together = batch.rank(items, candidates)
    # The app sees one garment at a time; with the pool's spreads it must rank the same way.
    alone = VersionSelector(experts, scales=batch.signals(items, candidates).std(axis=0))
    for item in items:
        assert alone.rank([item], candidates) == {item.item_id: together[item.item_id]}


@pytest.mark.parametrize(
    ("answers", "top", "chosen", "checked"),
    [
        ([CROPPED, COLOUR, CLEAN], 3, "c", 3),  # the first clean one
        ([COLOUR, CLEAN, CLEAN], 3, "b", 2),  # stops at the first clean one
        ([CROPPED, COLOUR, CLEAN], 2, "b", 2),  # none clean: the fewest defects
        ([COLOUR, CROPPED, COLOUR], 3, "a", 3),  # a tie keeps the better ranked one
    ],
)
def test_gate_keeps_the_first_clean_candidate_or_the_fewest_defects(
    tmp_path: Path, answers: list[dict[str, object]], top: int, chosen: str, checked: int
) -> None:
    spec = GarmentSpec(tmp_path / "01_front.jpg", (), "T-shirt", "", "", "", ())
    ranking = [Candidate(v, tmp_path / f"{v}.png") for v in "abc"]
    pick = gate_pick(ranking, spec, cast(VisionClient, SequenceVision(answers)), top)
    assert pick.chosen.version == chosen
    assert len(pick.checks) == checked


def test_v4_makes_every_version_and_keeps_the_ranked_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)  # fakes only, so no key is needed
    photo = tmp_path / "01_front.jpg"
    Image.new("RGB", (8, 12), "red").save(photo)
    vision = FakeVision({"spec": {"category": "T-shirt"}})
    generator = FakeGenerator()
    pipeline = StudioPipeline(
        paths=ProjectPaths(tmp_path, tmp_path, tmp_path, tmp_path),
        generator=generator,
        vision=cast(VisionClient, vision),
        selector=ReverseRanker(),
    )
    result = pipeline.run([photo], "g1", mode="v4")

    # Every version is a different prompt or seed; a clean gate means no repair in v3.
    assert len({(r.prompt, r.seed) for r in generator.requests}) == len(V4_VERSIONS)
    assert len(generator.requests) == len(V4_VERSIONS)
    assert result.image.parent.name == V4_VERSIONS[-1].name
    assert f"Kept: {V4_VERSIONS[-1].name}" in result.log
    # v3 checks its own one-shot, then the gate checks the top pick, which passes.
    assert vision.calls.count("gate") == 2

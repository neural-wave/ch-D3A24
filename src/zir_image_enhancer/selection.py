"""v4: several versions of the same packshot, ranked by house style, the best checked by the gate.

nano-banana gives a different image on every call, and on our holdout the best of nine would
score far above any single one. v4 generates the versions below and picks one without the
target: two embedding experts (``embeddings.py``) score how close each candidate is to the
house style of the training studio photos and to the garment's own photo. The jury gate then
checks the top candidates in that order, so a stylish but wrong image is not kept.

This module needs only numpy; the experts (torch, CLIP, AlexNet) live in ``embeddings.py``
and are loaded only when a ranking is asked for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.reconstruction.gate import check
from zir_image_enhancer.reconstruction.spec import GarmentSpec
from zir_image_enhancer.reconstruction.vision import VisionClient


@dataclass(frozen=True)
class Version:
    name: str
    mode: str  # reconstructor mode: "oneshot", "harness" or "jury"
    style: str  # studio prompt style, see generation/prompts.py
    seed: int


# Distinct prompts and seeds: the selector gained about +0.2 per distinct version on the
# holdout (2 to 9 versions), so the pool is as varied as the budget allows.
V4_VERSIONS = (
    Version("v3", "jury", "auto", 0),
    Version("flat_s1", "oneshot", "flat", 1),
    Version("flat_s2", "oneshot", "flat", 2),
    Version("A", "oneshot", "baseline", 0),
    Version("A_s1", "oneshot", "baseline", 1),
    Version("B", "oneshot", "baseline-keep", 0),
    Version("B_s1", "oneshot", "baseline-keep", 1),
    Version("C", "oneshot", "baseline-white", 0),
    Version("C_s1", "oneshot", "baseline-white", 1),
)

# Spread of each signal (CLIP style, CLIP identity, AlexNet style, AlexNet identity) over the
# 38 holdout garments with 9 versions each. Batch v4 standardises the signals over all its
# garments at once; one garment alone (the app) has too few candidates for that, and ranking
# it with its own spread dropped the holdout pick from 63.01 to 61.70. Inside one garment
# only the ratio between the spreads matters, so with these fixed it ranks like the batch.
POOL_SCALES = (0.0423, 0.0649, 0.0468, 0.0773)


@dataclass(frozen=True)
class Candidate:
    version: str
    image: Path


@dataclass(frozen=True)
class GateCheck:
    version: str
    problems: tuple[str, ...]


@dataclass(frozen=True)
class Pick:
    chosen: Candidate
    checks: tuple[GateCheck, ...]
    cost_usd: float


class Embedder(Protocol):
    def embed(self, images: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]: ...

    def house_style(self) -> tuple[np.ndarray, np.ndarray]: ...


class VersionSelector:
    """Ranks the candidates of each garment, best first, without looking at any target.

    With ``scales`` unset, each signal is standardised over every candidate passed to
    ``rank`` (the batch, many garments at once); with ``scales`` set, each signal is divided
    by its fixed spread, so a single garment ranks as it would inside a batch.
    """

    def __init__(
        self,
        experts: Embedder,
        neighbours: int = 10,
        scales: Sequence[float] | None = None,
    ) -> None:
        self._experts = experts
        self._neighbours = neighbours
        self._scales = scales
        self._style_clip, self._style_alex = experts.house_style()

    def rank(
        self, items: Sequence[GarmentItem], candidates: Mapping[str, Sequence[Candidate]]
    ) -> dict[str, list[Candidate]]:
        pool = _pool(items, candidates)
        ranked: dict[str, list[Candidate]] = {item.item_id: [] for item in items}
        if not pool:
            return ranked
        matrix = self._signals(items, pool)
        if self._scales is None:
            # Standardised over the whole pool so neither expert dominates.
            value = ((matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)).sum(axis=1)
        else:
            value = (matrix / np.asarray(self._scales)).sum(axis=1)
        # A stable sort, so ties keep the order of the versions.
        for k in sorted(range(len(pool)), key=lambda k: -value[k]):
            item, cand = pool[k]
            ranked[item.item_id].append(cand)
        return ranked

    def signals(
        self, items: Sequence[GarmentItem], candidates: Mapping[str, Sequence[Candidate]]
    ) -> np.ndarray:
        """One row per candidate (garment by garment, in the given order), one column per
        signal; ``signals(...).std(axis=0)`` is how ``POOL_SCALES`` was measured."""
        return self._signals(items, _pool(items, candidates))

    def _signals(
        self, items: Sequence[GarmentItem], pool: list[tuple[GarmentItem, Candidate]]
    ) -> np.ndarray:
        clip, alex = self._experts.embed([cand.image for _, cand in pool])
        in_clip, in_alex = self._experts.embed([item.inputs[0] for item in items])
        photo = {item.item_id: index for index, item in enumerate(items)}
        columns = []
        for emb, style, inputs in (
            (clip, self._style_clip, in_clip),
            (alex, self._style_alex, in_alex),
        ):
            # House style: mean similarity to the closest training studio photos.
            columns.append(np.sort(emb @ style.T, axis=1)[:, -self._neighbours :].mean(axis=1))
            # Identity: similarity to the garment's own first photo.
            columns.append(
                np.array([emb[k] @ inputs[photo[item.item_id]] for k, (item, _) in enumerate(pool)])
            )
        return np.stack(columns, axis=1)


def candidates_in_folders(
    items: Sequence[GarmentItem], versions: Mapping[str, Path]
) -> dict[str, list[Candidate]]:
    """Candidates from batch runs: one folder per version, with <garment>/<view>.png inside."""
    return {
        item.item_id: [
            Candidate(name, folder / item.item_id / f"{item.view}.png")
            for name, folder in versions.items()
            # A version can miss a garment (fal's content filter refused one); skip it.
            if (folder / item.item_id / f"{item.view}.png").exists()
        ]
        for item in items
    }


def gate_pick(
    ranking: Sequence[Candidate], spec: GarmentSpec, vision: VisionClient, top: int = 3
) -> Pick:
    """The first of the ``top`` candidates the gate finds clean, else the one with the fewest
    defects (severe ones count double; a tie keeps the better ranked one).

    The ranking only knows style; the gate compares with the real photo, so a top candidate
    with a cut sleeve or a missing logo gives way to the next one.
    """
    if not ranking:
        raise ValueError("no candidates to pick from")
    best: tuple[int, Candidate] | None = None
    checks: list[GateCheck] = []
    cost = 0.0
    for cand in ranking[:top]:
        gate = check(spec.front, cand.image, spec, vision)
        checks.append(GateCheck(cand.version, gate.problems))
        cost += gate.cost_usd
        if best is None or gate.weight < best[0]:
            best = (gate.weight, cand)
        if not gate.problems:
            break
    return Pick(best[1] if best else ranking[0], tuple(checks), cost)


def _pool(
    items: Sequence[GarmentItem], candidates: Mapping[str, Sequence[Candidate]]
) -> list[tuple[GarmentItem, Candidate]]:
    return [(item, cand) for item in items for cand in candidates.get(item.item_id, ())]

"""Reading the challenge manifests into one item type for every split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from zir_image_enhancer.paths import ProjectPaths

DEFAULT_VIEW = "front"


@dataclass(frozen=True)
class GarmentItem:
    item_id: str
    inputs: tuple[Path, ...]
    target: Path | None
    view: str = DEFAULT_VIEW


def load_manifest(manifest: Path, data_root: Path) -> list[GarmentItem]:
    """Load train/holdout pairs or dev/test inputs.

    Paths inside a manifest are relative to the dataset root.
    """
    items: list[GarmentItem] = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        # Train and holdout rows use pair_id; dev and test rows use garment_id.
        item_id = row.get("pair_id") or row.get("garment_id")
        if not item_id:
            raise ValueError(f"{manifest}:{line_no} has no pair_id or garment_id")
        target = row.get("target")
        items.append(
            GarmentItem(
                item_id=item_id,
                inputs=tuple(data_root / path for path in row["inputs"]),
                target=data_root / target if target else None,
                view=row.get("view", DEFAULT_VIEW),
            )
        )
    return items


def split_manifest(split: str, paths: ProjectPaths) -> Path:
    """Manifest path for a named split. 'holdout' is the validation side of our grouped split."""
    if split == "holdout-small":
        return small_holdout(paths)
    manifests = {
        "holdout": paths.holdout_val,
        "holdout-train": paths.holdout_train,
        "train": paths.data_root / "train" / "pairs.jsonl",
        "dev": paths.data_root / "dev" / "inputs.jsonl",
        "test": paths.data_root / "test" / "inputs.jsonl",
    }
    if split not in manifests:
        names = ", ".join([*manifests, "holdout-small"])
        raise KeyError(f"Unknown split '{split}'. Choose one of: {names}")
    return manifests[split]


def small_holdout(paths: ProjectPaths) -> Path:
    """Half of the holdout (19 of 38 pairs, about 5% of the training pairs), written once.

    It is a fixed subset of the full holdout rather than a new split, so every score we
    already have can be compared on the same garments. Pairs are picked by a hash of their
    id, which does not depend on file order. A run costs about half: ~$1.1 instead of ~$2.3.
    """
    target = paths.holdout_small
    if not target.exists():
        lines = [
            line for line in paths.holdout_val.read_text(encoding="utf-8").splitlines() if line
        ]

        def rank(line: str) -> str:
            return hashlib.sha1(str(json.loads(line)["pair_id"]).encode()).hexdigest()

        chosen = sorted(lines, key=rank)[: len(lines) // 2]
        target.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return target

"""Reading the challenge manifests into one item type for every split."""

from __future__ import annotations

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
    manifests = {
        "holdout": paths.holdout_val,
        "holdout-train": paths.holdout_train,
        "train": paths.data_root / "train" / "pairs.jsonl",
        "dev": paths.data_root / "dev" / "inputs.jsonl",
        "test": paths.data_root / "test" / "inputs.jsonl",
    }
    if split not in manifests:
        raise KeyError(f"Unknown split '{split}'. Choose one of: {', '.join(manifests)}")
    return manifests[split]

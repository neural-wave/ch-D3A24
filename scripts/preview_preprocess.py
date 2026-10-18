"""Build a before/after sheet so the preprocessing can be judged by eye.

For every garment the sheet shows up to four columns: the original front photo, the garment
mask OpenCV found, the enhanced reference sent to the model, and next to it the real studio
target when the split has one. It never calls a paid model.

    python scripts/preview_preprocess.py --split holdout --limit 8
    python scripts/preview_preprocess.py --manifest local/captures/inputs.jsonl --grain 0

Writes ``local/preprocess_preview/<garment>.png``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from zir_image_enhancer.dataset import GarmentItem, load_manifest, split_manifest
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction import preprocess as pp


def _row(images: list[tuple[str, np.ndarray]], height: int = 420) -> np.ndarray:
    """A horizontal strip of labelled tiles, all scaled to the same height."""
    tiles = []
    for label, bgr in images:
        scale = height / bgr.shape[0]
        tile = cv2.resize(bgr, (max(1, round(bgr.shape[1] * scale)), height))
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        tiles.append(tile)
    return cv2.hconcat(tiles)


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def sheet(item: GarmentItem, settings: pp.PreprocessSettings) -> np.ndarray:
    front = pp.read_image(item.inputs[0])
    mask = pp.garment_mask(front)
    enhanced = pp.enhance_full(front, settings, mask)
    enhanced = pp.finalize(enhanced, settings, pp.grain_seed(item.inputs[0]))

    tiles: list[tuple[str, np.ndarray]] = [("original", _to_bgr(front))]
    if mask is not None:
        overlay = _to_bgr(front).copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        tiles.append(("mask", overlay))
    tiles.append(("enhanced", _to_bgr(enhanced)))
    if item.target is not None and item.target.is_file():
        tiles.append(("target", _to_bgr(pp.read_image(item.target))))
    return _row(tiles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("local/preprocess_preview"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--grain", type=float, default=None)
    parser.add_argument("--no-white-balance", action="store_true")
    parser.add_argument("--wb-method", choices=pp.WHITE_BALANCE_METHODS, default=None)
    parser.add_argument("--flatten", type=float, default=None, metavar="STRENGTH")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    items = load_manifest(args.manifest or split_manifest(args.split, paths), paths.data_root)
    items = items[: args.limit] if args.limit is not None else items

    settings = pp.PreprocessSettings()
    if args.grain is not None:
        settings = replace(settings, grain_sigma=args.grain)
    if args.no_white_balance:
        settings = replace(settings, white_balance=0.0)
    if args.wb_method is not None:
        settings = replace(settings, white_balance_method=args.wb_method)
    if args.flatten is not None:
        settings = replace(settings, flatten_illumination=args.flatten)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in items:
        try:
            image = sheet(item, settings)
        except Exception as error:  # noqa: BLE001 - one bad garment must not stop the sheet
            print(f"  ! {item.item_id}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        cv2.imwrite(str(args.out / f"{item.item_id}.png"), image)
        written += 1
    print(f"wrote {written} sheet(s) to {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

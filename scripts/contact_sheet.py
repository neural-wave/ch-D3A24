"""One image per garment: its input photos, every candidate of a run, and the target if known.

The numbers say which config wins; the sheets say why. Sheets are written to
``local/runs/<run>/sheets/`` and never leave the machine (they contain dataset images).

    python scripts/contact_sheet.py --run holdout_nb21_baseline --split holdout
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from zir_image_enhancer.dataset import GarmentItem, load_manifest, split_manifest
from zir_image_enhancer.paths import ProjectPaths

TILE = 320
LABEL_HEIGHT = 22


def _tile(path: Path, label: str) -> Image.Image:
    tile = Image.new("RGB", (TILE, TILE + LABEL_HEIGHT), (90, 90, 90))
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((TILE, TILE))
    tile.paste(image, ((TILE - image.width) // 2, LABEL_HEIGHT + (TILE - image.height) // 2))
    ImageDraw.Draw(tile).text((6, 4), label, fill=(255, 255, 255))
    return tile


def build_sheet(item: GarmentItem, run_dir: Path) -> Image.Image | None:
    tiles = [_tile(path, f"input {index + 1}") for index, path in enumerate(item.inputs)]
    candidates = sorted(run_dir.glob(f"cand_*/{item.item_id}/{item.view}.png"))
    if not candidates:
        return None
    tiles += [_tile(path, path.parts[-3]) for path in candidates]
    if item.target is not None:
        tiles.append(_tile(item.target, "target"))
    sheet = Image.new("RGB", (TILE * len(tiles), TILE + LABEL_HEIGHT), (40, 40, 40))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * TILE, 0))
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="holdout")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    run_dir = paths.runs_root / args.run
    out_dir = run_dir / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in load_manifest(split_manifest(args.split, paths), paths.data_root):
        sheet = build_sheet(item, run_dir)
        if sheet is not None:
            sheet.save(out_dir / f"{item.item_id}.jpg", quality=88)
            written += 1
    print(f"{written} sheets in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

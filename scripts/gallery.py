"""One folder per garment with its input photos, every output we generated, and the target.

    python scripts/gallery.py
    python scripts/gallery.py --runs holdout_nb21p_v2_all length_test --ids t0162 t0025

Writes local/gallery/<garment>/ with input_<n>.jpg, <run>.jpg (one per run and candidate),
target.jpg when there is one, and sheet.jpg: inputs in the left column, outputs on the right,
each labelled with its run. Images are shrunk to keep the folder light; it stays in local/
(the dataset must not leave this machine).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from zir_image_enhancer.paths import ProjectPaths

# Runs that only copy or post-process another run's outputs; they add nothing to look at.
_SKIP_PREFIXES = ("post_", "holdout_cutout", "probe", "bias_probe", "holdout_nb21p_v1_post")
_LONG_SIDE = 768
_TILE = 300  # tile height in the sheet
_LABEL_HEIGHT = 22


@dataclass
class GarmentPhotos:
    inputs: list[Path]
    target: Path | None
    outputs: dict[str, Path] = field(default_factory=dict)  # label -> image


class Gallery:
    def __init__(self, paths: ProjectPaths, out_dir: Path) -> None:
        self._paths = paths
        self._out_dir = out_dir
        self._garments: dict[str, GarmentPhotos] = {}

    def load_manifests(self) -> None:
        """Inputs and targets of every garment we may have generated (train, dev, test, ours)."""
        data = self._paths.data_root
        manifests = [
            data / "train" / "pairs.jsonl",
            data / "dev" / "inputs.jsonl",
            data / "test" / "inputs.jsonl",
        ]
        manifests += sorted((self._paths.repo_root / "local" / "captures").glob("inputs.jsonl"))
        for manifest in manifests:
            if not manifest.is_file():
                continue
            for line in manifest.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                garment = str(row.get("pair_id") or row.get("garment_id"))
                inputs = [self._resolve(value) for value in row.get("inputs", [])]
                target = self._resolve(row["target"]) if row.get("target") else None
                self._garments.setdefault(garment, GarmentPhotos(inputs, target))

    def collect_runs(self, runs: list[str] | None) -> None:
        names = runs or sorted(
            path.name
            for path in self._paths.runs_root.iterdir()
            if path.is_dir() and not path.name.startswith(_SKIP_PREFIXES)
        )
        for run in names:
            run_dir = self._paths.runs_root / run
            candidates = sorted(run_dir.glob("cand_*"))
            for candidate in candidates:
                label = run if len(candidates) == 1 else f"{run}_{candidate.name}"
                for image in candidate.glob("*/*.png"):
                    garment = self._garments.get(image.parent.name)
                    if garment is not None:
                        garment.outputs[label] = image

    def write(self, ids: list[str] | None) -> int:
        written = 0
        for garment_id, photos in sorted(self._garments.items()):
            if ids and garment_id not in ids:
                continue
            if not photos.outputs:
                continue
            folder = self._out_dir / garment_id
            folder.mkdir(parents=True, exist_ok=True)
            for index, path in enumerate(photos.inputs, start=1):
                _shrink(path).save(folder / f"input_{index}.jpg", quality=90)
            for label, path in photos.outputs.items():
                _shrink(path).save(folder / f"{label}.jpg", quality=90)
            if photos.target is not None:
                _shrink(photos.target).save(folder / "target.jpg", quality=90)
            self._sheet(photos).save(folder / "sheet.jpg", quality=88)
            written += 1
        return written

    def _sheet(self, photos: GarmentPhotos) -> Image.Image:
        left = [(f"input {i}", p) for i, p in enumerate(photos.inputs, start=1)]
        right = sorted(photos.outputs.items())
        if photos.target is not None:
            right.append(("target", photos.target))
        columns = 4
        tile_w = int(_TILE * 2 / 3)  # portrait tiles, like the outputs
        rows_right = -(-len(right) // columns)
        height = max(len(left), rows_right) * (_TILE + _LABEL_HEIGHT)
        width = (1 + columns) * tile_w + 20
        sheet = Image.new("RGB", (width, height), (40, 40, 40))
        for row, (label, path) in enumerate(left):
            _paste(sheet, path, label, (0, row * (_TILE + _LABEL_HEIGHT)), tile_w)
        for index, (label, path) in enumerate(right):
            x = tile_w + 20 + (index % columns) * tile_w
            y = (index // columns) * (_TILE + _LABEL_HEIGHT)
            _paste(sheet, path, label, (x, y), tile_w)
        return sheet

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self._paths.data_root / path


def _shrink(path: Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((_LONG_SIDE, _LONG_SIDE))
    return image


def _paste(sheet: Image.Image, path: Path, label: str, origin: tuple[int, int], width: int) -> None:
    tile = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    tile.thumbnail((width - 6, _TILE - 6))
    x, y = origin
    sheet.paste(tile, (x + (width - tile.width) // 2, y + (_TILE - tile.height) // 2))
    ImageDraw.Draw(sheet).text((x + 4, y + _TILE + 4), label[:36], fill=(230, 230, 230))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="*", default=None, help="default: every run")
    parser.add_argument("--ids", nargs="*", default=None, help="default: every garment")
    parser.add_argument("--out", type=Path, default=None, help="default: local/gallery")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    gallery = Gallery(paths, args.out or paths.repo_root / "local" / "gallery")
    gallery.load_manifests()
    gallery.collect_runs(args.runs)
    count = gallery.write(args.ids)
    print(f"{count} garment folders in {args.out or 'local/gallery'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

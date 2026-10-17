"""Side-by-side pages to judge runs by eye: one row per garment, one column per run.

    python scripts/compare_sheet.py --runs holdout_nb21p_v2_all holdout_harness holdout_harness_v2

Writes local/compare/<name>/page_<n>.jpg. Each row: the first input photo, each run's output
with its holdout score (when scored), and the target. Missing outputs are left blank.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from zir_image_enhancer.paths import ProjectPaths

_TILE_H = 260
_TILE_W = 175
_HEADER = 24
_LABEL = 18


class CompareSheet:
    def __init__(self, paths: ProjectPaths, runs: list[str], rows_per_page: int) -> None:
        self._paths = paths
        self._runs = runs
        self._rows_per_page = rows_per_page
        self._scores = {run: self._load_scores(run) for run in runs}

    def write(self, out_dir: Path) -> int:
        pairs = [
            json.loads(line)
            for line in self._paths.holdout_val.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = 0
        for start in range(0, len(pairs), self._rows_per_page):
            chunk = pairs[start : start + self._rows_per_page]
            self._page(chunk).save(out_dir / f"page_{pages + 1}.jpg", quality=88)
            pages += 1
        return pages

    def _page(self, pairs: list[dict[str, object]]) -> Image.Image:
        columns = ["input", *self._runs, "target"]
        width = len(columns) * _TILE_W + 60
        height = _HEADER + len(pairs) * (_TILE_H + _LABEL)
        page = Image.new("RGB", (width, height), (40, 40, 40))
        draw = ImageDraw.Draw(page)
        for col, name in enumerate(columns):
            draw.text((60 + col * _TILE_W + 4, 6), name[:28], fill=(240, 240, 240))
        for row, pair in enumerate(pairs):
            garment = str(pair["pair_id"])
            y = _HEADER + row * (_TILE_H + _LABEL)
            draw.text((4, y + _TILE_H // 2), garment, fill=(240, 240, 240))
            raw_inputs = pair.get("inputs")
            inputs = raw_inputs if isinstance(raw_inputs, list) else []
            cells: list[tuple[Path | None, str]] = [
                (self._paths.data_root / str(inputs[0]) if inputs else None, "")
            ]
            for run in self._runs:
                found = sorted((self._paths.runs_root / run / "cand_0" / garment).glob("*.png"))
                score = self._scores[run].get(garment)
                cells.append((found[0] if found else None, f"{score:.1f}" if score else ""))
            cells.append((self._paths.data_root / str(pair["target"]), ""))
            for col, (path, label) in enumerate(cells):
                x = 60 + col * _TILE_W
                if path is not None and path.exists():
                    _paste(page, path, (x, y))
                if label:
                    draw.text((x + 4, y + _TILE_H + 2), label, fill=(255, 220, 120))
        return page

    def _load_scores(self, run: str) -> dict[str, float]:
        path = self._paths.runs_root / run / "scores" / "cand_0" / "pairs.csv"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        # Garments that were never generated are scored as empty; do not show those.
        run_dir = self._paths.runs_root / run / "cand_0"
        return {
            row["pair_id"]: float(row["score"])
            for row in rows
            if (run_dir / row["pair_id"]).exists()
        }


def _paste(page: Image.Image, path: Path, origin: tuple[int, int]) -> None:
    tile = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    tile.thumbnail((_TILE_W - 6, _TILE_H - 6))
    x, y = origin
    page.paste(tile, (x + (_TILE_W - tile.width) // 2, y + (_TILE_H - tile.height) // 2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--name", default=None, help="default: the last run's name")
    parser.add_argument("--rows", type=int, default=10, help="garments per page")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    out_dir = paths.repo_root / "local" / "compare" / (args.name or args.runs[-1])
    pages = CompareSheet(paths, args.runs, args.rows).write(out_dir)
    print(f"{pages} page(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

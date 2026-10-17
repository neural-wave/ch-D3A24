"""Batch runner: turn a manifest of garments into the challenge submission layout.

The official harness (``scripts/eval_val.py`` and ``scripts/make_submission.py``)
consumes ``<out>/<team>/<garment_id>/<view>.<ext>``. This CLI reads either manifest
the dataset ships, runs :func:`zir_image_enhancer.pipeline.studio_shot` on each
garment's photos, and writes that layout, so the same output can be self-scored
and validated.

    zir-run --manifest data/sartiq_neural_wave_2026/dev/inputs.jsonl \
            --dataset data/sartiq_neural_wave_2026 --out submissions --team zir

Manifest rows (paths are relative to the dataset root):

    train/pairs.jsonl        {"pair_id": "t0001", "inputs": [...], "target": ...}
    dev|test/inputs.jsonl    {"garment_id": "d001", "view": "front", "inputs": [...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from zir_image_enhancer.pipeline import DEFAULT_VIEW, studio_shot

try:  # iPhone photos arrive as HEIC; the opener is optional.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - only when pillow-heif is absent
    pass

_SPLIT_DIRS = {"train", "dev", "test"}


def _resolve_dataset_root(manifest: Path, explicit: str | None) -> Path:
    """Dataset root for a manifest: explicit flag, sibling ``meta.json``, or split parent."""
    if explicit:
        return Path(explicit)
    meta = manifest.parent / "meta.json"
    if meta.is_file():
        try:
            recorded = json.loads(meta.read_text(encoding="utf-8-sig")).get("dataset_root")
        except (OSError, ValueError):
            recorded = None
        if recorded:
            root = Path(str(recorded))
            return root if root.is_absolute() else (meta.parent / root)
    if manifest.parent.name in _SPLIT_DIRS:
        return manifest.parent.parent
    return manifest.parent


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: {path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"error: {path}:{lineno}: expected a JSON object")
        garment_id = row.get("garment_id") or row.get("pair_id")
        if not garment_id:
            raise SystemExit(f"error: {path}:{lineno}: row needs a garment_id (or pair_id)")
        inputs = row.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise SystemExit(f"error: {path}:{lineno}: row needs a non-empty 'inputs' list")
        rows.append(row)
    return rows


def _resolve_input(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zir-run",
        description="Run the pipeline over a manifest into the challenge submission layout.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="train/pairs.jsonl or dev|test/inputs.jsonl",
    )
    parser.add_argument("--dataset", default=None, help="dataset root (holds train/, dev/, test/)")
    parser.add_argument("--out", required=True, help="submission root (holds <team>/ folders)")
    parser.add_argument("--team", required=True, help="team folder name, e.g. zir")
    parser.add_argument("--view", default=DEFAULT_VIEW, help=f"view name (default: {DEFAULT_VIEW})")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be written without running the pipeline",
    )
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"error: --manifest is not a file: {manifest}")

    dataset_root = _resolve_dataset_root(manifest, args.dataset)
    rows = _load_manifest(manifest)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    out_root = Path(args.out) / args.team
    print(f"zir-run: {len(rows)} garment(s) from {manifest} (dataset {dataset_root})")
    written = 0
    for row in rows:
        garment_id = str(row.get("garment_id") or row.get("pair_id"))
        view = str(row.get("view") or args.view)
        inputs = [_resolve_input(str(v), dataset_root) for v in row["inputs"]]
        missing = [p for p in inputs if not p.is_file()]
        if missing:
            raise SystemExit(f"error: {garment_id}: input not found: {missing[0]}")

        dest = out_root / garment_id / f"{view}.png"
        if args.dry_run:
            print(f"  {garment_id}: {inputs[0]} -> {dest}")
            continue

        result = studio_shot(_load_rgb(inputs[0]))
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(dest)
        written += 1
        print(f"  [{written}/{len(rows)}] {garment_id}: -> {dest}")

    if not args.dry_run:
        print(f"zir-run: wrote {written} image(s) under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

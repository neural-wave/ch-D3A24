"""Batch runner: turn a manifest of garments into the challenge submission layout.

The official harness (``scripts/eval_val.py`` and ``scripts/make_submission.py``)
consumes ``<out>/<team>/<garment_id>/<view>.<ext>``. This CLI reads either manifest
the dataset ships, runs :class:`zir_image_enhancer.pipeline.StudioPipeline` on all of each
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
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from zir_image_enhancer.pipeline import (
    DEFAULT_VIEW,
    MissingKeyError,
    StudioPipeline,
    StudioResult,
    StudioRunner,
)

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


def main(argv: list[str] | None = None, pipeline: StudioRunner | None = None) -> int:
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
    parser.add_argument("--workers", type=int, default=4, help="garments processed in parallel")
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
    jobs: list[tuple[str, list[Path], Path]] = []
    for row in rows:
        garment_id = str(row.get("garment_id") or row.get("pair_id"))
        view = str(row.get("view") or args.view)
        inputs = [_resolve_input(str(v), dataset_root) for v in row["inputs"]]
        missing = [p for p in inputs if not p.is_file()]
        if missing:
            raise SystemExit(f"error: {garment_id}: input not found: {missing[0]}")
        jobs.append((garment_id, inputs, out_root / garment_id / f"{view}.png"))

    if args.dry_run:
        for garment_id, inputs, dest in jobs:
            print(f"  {garment_id}: {len(inputs)} photo(s) -> {dest}")
        return 0

    if pipeline is None:
        try:
            pipeline = StudioPipeline()
        except MissingKeyError as error:
            raise SystemExit(f"error: {error}") from error
    runner = pipeline

    def one(job: tuple[str, list[Path], Path]) -> tuple[str, Path, StudioResult | Exception]:
        garment_id, inputs, dest = job
        try:
            return garment_id, dest, runner.run(inputs, garment_id)
        except Exception as error:  # noqa: BLE001
            # One failed garment must not stop the batch; it is reported at the end.
            return garment_id, dest, error

    written, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for garment_id, dest, result in pool.map(one, jobs):
            if isinstance(result, Exception):
                failed += 1
                print(f"  ! {garment_id}: {type(result).__name__}: {result}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result.image, dest)
            written += 1
            print(f"  [{written}/{len(jobs)}] {garment_id}: -> {dest} (${result.cost_usd:.3f})")

    print(f"zir-run: wrote {written} image(s) under {out_root}, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

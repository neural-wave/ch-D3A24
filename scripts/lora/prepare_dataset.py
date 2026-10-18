#!/usr/bin/env python3
"""Build the packshot LoRA training dataset from the example pairs (see docs/LORA.md).

Reads the frozen training split (``sartiq_lora_unified/holdout/train.jsonl``), keeps only the
hand-checked labels (ok / multi_input), refuses to touch anything that appears in the
validation split, and exports two equivalent layouts from the same letterboxed 2:3 pairs:

    out/fal/    ROOT_start.jpg, ROOT_start2..4.jpg, ROOT_end.jpg, ROOT.txt   (+ fal.zip)
    out/aitk/   target/ROOT.jpg + target/ROOT.txt, reference_1..4/ROOT.jpg

The fal layout is the archive ``scripts/lora/train_fal.py`` uploads; the ai-toolkit layout is
what ``scripts/lora/train_vast.py`` bundles for a vast.ai GPU. Nothing here touches the
network or spends credits; rembg is used only when installed (it is in the ``scorer`` extra),
otherwise a simple border-colour mask stands in.

The canvas is 832x1248: the largest 2:3 size divisible by 16 under klein's 1024^2 cap, which
matches the example targets (~2:3, light-grey ~240 background) and the ``sartiq-lora-klein``
canvas in ``zir_image_enhancer.generation.models``.

    python scripts/lora/prepare_dataset.py                # both layouts, full training split
    python scripts/lora/prepare_dataset.py --limit 20     # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:  # rembg is optional: the scorer extra ships it, the base install does not.
    from rembg import new_session, remove

    _HAS_REMBG = True
except ImportError:  # pragma: no cover - only when rembg is absent
    _HAS_REMBG = False

_REPO_ROOT = Path(__file__).resolve().parents[2]

_REMBG_SESSION: object | None = None
_REMBG_TRIED = False

#: Labels the humans kept in labels.csv; anything else (bad, set) is not training data.
KEEP_LABELS = {"ok", "multi_input"}

#: One caption per pair, chosen deterministically from these and conditioned on the category.
#: They are edit instructions (what to do), varied on purpose: the pipeline's own prompt is
#: not one of them, so the LoRA learns the transformation instead of one exact sentence.
CAPTIONS = (
    "Turn this photo of a {c} into a professional fashion catalogue packshot: only the {c}, "
    "front view, laid flat and centred on a plain light-grey studio background. Keep the exact "
    "colour, fabric texture, pattern, print and details.",
    "Clean e-commerce packshot from this product photo: a {c}, front view, laid flat and "
    "centred on a plain light-grey studio background, the whole {c} visible. Keep the exact "
    "colour, fabric, print and proportions; remove the person, hanger, tags and props.",
    "Convert these product photos into one studio packshot of a {c}: front view, laid flat and "
    "centred on a plain light-grey background with soft even light. The other photos are "
    "close-ups of the same {c}: use them only for the details. Keep the exact colour, texture "
    "and print.",
    "Make a studio flat-lay packshot of a {c} from this photo: only the {c}, front view, "
    "centred on a light-grey studio background, the whole garment visible with a margin on "
    "every side. Keep the exact colour, fabric, pattern and every small detail.",
    "Product photo -> catalogue packshot: a {c}, front view, laid flat and centred on a plain "
    "light-grey studio background. Keep the exact colour, texture, print and proportions; show "
    "no person, mannequin, floor or props.",
)

#: Captions that only make sense when a second photo exists (detail close-ups).
MULTI_CAPTIONS = (CAPTIONS[2],)

_GREY = (240, 240, 240)


def parse_size(text: str) -> tuple[int, int]:
    """``"832x1248"`` -> (832, 1248)."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"size must look like 832x1248, got {text!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive, got {text!r}")
    return width, height


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_csv_column(path: Path, key: str, value: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row[value] for row in csv.DictReader(handle) if row.get(key)}


def load_rgb(path: Path) -> Image.Image | None:
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except (OSError, ValueError):
        return None


def letterbox(image: Image.Image, size: tuple[int, int], fill: tuple[int, int, int]) -> Image.Image:
    """Fit the image on a canvas of ``size`` without cropping, centred, on ``fill``."""
    width, height = size
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", size, fill)
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    return canvas


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..255 (h, w, 3) -> CIELAB D65 (h, w, 3); the quality filter's delta E is in here."""
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    lab = np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )
    return lab


def border_pixels(rgb: np.ndarray, margin: int = 12) -> np.ndarray:
    top, bottom = rgb[:margin], rgb[-margin:]
    left, right = rgb[:, :margin], rgb[:, -margin:]
    return np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)]
    )


def _rembg_session() -> object | None:
    """A cached rembg session when rembg is installed, else None."""
    global _REMBG_SESSION, _REMBG_TRIED
    if not _HAS_REMBG:
        return None
    if not _REMBG_TRIED:
        _REMBG_TRIED = True
        models_dir = _REPO_ROOT / "local" / "neural-wave-2026" / "scorer" / ".models"
        if not os.environ.get("U2NET_HOME") and models_dir.is_dir():
            os.environ["U2NET_HOME"] = str(models_dir)
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception:  # noqa: BLE001 - optional accelerator, fall back to the heuristic
            _REMBG_SESSION = None
    return _REMBG_SESSION


def garment_mask(image: Image.Image) -> np.ndarray:
    """Boolean mask of the garment, via rembg when available, else by distance from the border."""
    session = _rembg_session()
    if session is not None:
        try:
            small = image.copy()
            small.thumbnail((1024, 1024))
            mask = remove(small, session=session, only_mask=True)
            grown = mask.resize(image.size, Image.Resampling.NEAREST).convert("L")
            return np.asarray(grown) > 127
        except Exception:  # noqa: BLE001 - optional accelerator, fall back to the heuristic
            pass
    rgb = np.asarray(image, dtype=np.float64)
    background = np.median(border_pixels(rgb.astype(np.uint8)), axis=0)
    return np.sqrt(np.sum((rgb - background) ** 2, axis=-1)) > 40.0


def analyse(image: Image.Image, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Garment area share and mean garment colour in Lab, for selection and delta E."""
    rgb = np.asarray(image, dtype=np.uint8)
    area = float(np.mean(mask))
    lab = rgb_to_lab(rgb)
    garment = lab[mask]
    mean_lab = garment.mean(axis=0) if garment.shape[0] > 50 else np.array([50.0, 0.0, 0.0])
    return area, mean_lab


def target_quality(image: Image.Image, mask: np.ndarray, min_side: int = 300) -> str | None:
    """Return a reason when the packshot target is unusable, else None."""
    if min(image.size) < min_side:
        return f"resolution {image.width}x{image.height}"
    borders = border_pixels(np.asarray(image, dtype=np.uint8))
    if float(borders.mean()) < 210.0 or float(borders.std()) > 40.0:
        return "background is not a flat light grey"
    area = float(np.mean(mask))
    if area < 0.08 or area > 0.92:
        return f"garment area {area:.2f}"
    return None


def caption_for(pair_id: str, category: str, n_inputs: int) -> str:
    """One deterministic instruction per pair; multi-photo pairs get the close-up line sometimes."""
    choices = CAPTIONS if n_inputs == 1 else CAPTIONS + MULTI_CAPTIONS
    return random.Random(f"sartiq-lora-{pair_id}").choice(choices).format(c=category)


@dataclass
class Candidate:
    """One usable input photo of a pair, with the numbers that rank it."""

    rel: str
    image: Image.Image
    suitability: float


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_folder(folder: Path, archive: Path) -> None:
    """Zip the layout folders with plain file names at the archive root (fal wants that)."""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder).as_posix())


def make_preview(out: Path, samples: list[dict[str, Any]], count: int = 24) -> Path | None:
    """A grey grid with up to four references and the target of each picked sample, for eyeballing."""
    if not samples:
        return None
    pick = random.Random(0).sample(samples, min(count, len(samples)))
    cell = (160, 240)
    cols = 5
    sheet = Image.new("RGB", (cols * cell[0], len(pick) * cell[1]), (128, 128, 128))
    for row, sample in enumerate(pick):
        folder = out / "fal"
        roots = [str(name) for name in sample["refs_fal"]] + [str(sample["end_fal"])]
        for col, name in enumerate(roots):
            sheet.paste(Image.open(folder / name).resize(cell), (col * cell[0], row * cell[1]))
    preview = out / "preview.jpg"
    sheet.save(preview, quality=85)
    return preview


def export(
    args: argparse.Namespace, pairs: list[dict[str, object]], val_ids: set[str]
) -> dict[str, object]:
    labels = load_csv_column(args.labels, "pair_id", "label")
    categories = load_csv_column(args.categories, "pair_id", "category")
    write_fal = args.layout in ("fal", "both")
    write_aitk = args.layout in ("aitk", "both")

    if write_fal:
        shutil.rmtree(args.out / "fal", ignore_errors=True)
        (args.out / "fal").mkdir(parents=True, exist_ok=True)
    if write_aitk:
        shutil.rmtree(args.out / "aitk", ignore_errors=True)
        (args.out / "aitk" / "target").mkdir(parents=True, exist_ok=True)
        for index in range(1, args.max_inputs + 1):
            (args.out / "aitk" / f"reference_{index}").mkdir(exist_ok=True)

    skipped = {
        "val": 0,
        "label": 0,
        "limit": 0,
        "missing": 0,
        "target_quality": 0,
        "no_inputs": 0,
        "input_quality": 0,
    }
    samples: list[dict[str, Any]] = []
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for pair in pairs:
            pair_id = str(pair.get("pair_id") or pair.get("garment_id") or "")
            if not pair_id or not pair.get("target"):
                skipped["missing"] += 1
                continue
            if pair_id in val_ids:
                # The guard the whole exercise depends on: never train on the 38 held-out pairs.
                skipped["val"] += 1
                continue
            label = labels.get(pair_id, "ok")
            if label not in KEEP_LABELS:
                skipped["label"] += 1
                continue
            if args.limit is not None and len(samples) >= args.limit:
                skipped["limit"] += 1
                continue

            target = load_rgb(args.dataset / str(pair["target"]))
            if target is None:
                skipped["missing"] += 1
                continue
            target_mask = garment_mask(target)
            _area, target_lab = analyse(target, target_mask)
            if not args.no_quality_filter:
                reason = target_quality(target, target_mask)
                if reason:
                    print(f"  - {pair_id}: target skipped ({reason})")
                    skipped["target_quality"] += 1
                    continue

            candidates: list[Candidate] = []
            for rel in [str(value) for value in pair.get("inputs", [])]:
                image = load_rgb(args.dataset / rel)
                if image is None:
                    continue
                area, mean_lab = analyse(image, garment_mask(image))
                delta_e = float(np.linalg.norm(mean_lab - target_lab))
                if not args.no_quality_filter and delta_e > 45.0:
                    skipped["input_quality"] += 1
                    continue
                candidates.append(Candidate(rel, image, area * 100.0 - delta_e * 1.2))
            if not candidates:
                skipped["no_inputs"] += 1
                continue
            candidates.sort(key=lambda candidate: candidate.suitability, reverse=True)
            used = candidates[: args.max_inputs]

            category = categories.get(pair_id, "garment")
            caption = caption_for(pair_id, category, len(used))

            sample: dict[str, Any] = {
                "pair_id": pair_id,
                "category": category,
                "caption": caption,
                "inputs": [candidate.rel for candidate in used],
                "target": str(pair["target"]),
            }
            if write_fal:
                fal_dir = args.out / "fal"
                refs_fal: list[str] = []
                for index, candidate in enumerate(used, start=1):
                    suffix = "_start" if index == 1 else f"_start{index}"
                    name = f"{pair_id}{suffix}.jpg"
                    letterbox(candidate.image, args.size, _GREY).save(fal_dir / name, quality=92)
                    refs_fal.append(name)
                end_fal = f"{pair_id}_end.jpg"
                letterbox(target, args.size, _GREY).save(fal_dir / end_fal, quality=92)
                (fal_dir / f"{pair_id}.txt").write_text(caption + "\n", encoding="utf-8")
                sample["refs_fal"] = refs_fal
                sample["end_fal"] = end_fal
            if write_aitk:
                aitk_dir = args.out / "aitk"
                boxed = letterbox(target, args.size, _GREY)
                boxed.save(aitk_dir / "target" / f"{pair_id}.jpg", quality=92)
                (aitk_dir / "target" / f"{pair_id}.txt").write_text(
                    caption + "\n", encoding="utf-8"
                )
                for index, candidate in enumerate(used, start=1):
                    path = aitk_dir / f"reference_{index}" / f"{pair_id}.jpg"
                    letterbox(candidate.image, args.size, _GREY).save(path, quality=92)
                sample["aitk"] = {
                    "target": f"target/{pair_id}.jpg",
                    "references": [
                        f"reference_{i}/{pair_id}.jpg" for i in range(1, len(used) + 1)
                    ],
                }
            samples.append(sample)
            manifest.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if len(samples) % 50 == 0:
                print(f"  ... {len(samples)} pairs exported")

    result: dict[str, object] = {"samples": len(samples), "skipped": skipped}
    if write_fal:
        archive = args.out / "fal.zip"
        zip_folder(args.out / "fal", archive)
        result["fal_zip"] = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_of(archive),
        }
        preview = make_preview(args.out, samples)
        result["preview"] = str(preview) if preview else None
    return result


def main(argv: list[str] | None = None) -> int:
    starter = _REPO_ROOT / "sartiq_lora_unified"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, default=starter / "holdout" / "train.jsonl")
    parser.add_argument("--val", type=Path, default=starter / "holdout" / "val.jsonl")
    parser.add_argument("--dataset", type=Path, default=_REPO_ROOT / "data" / "sartiq_neural_wave_2026")
    parser.add_argument("--labels", type=Path, default=starter / "labels.csv")
    parser.add_argument("--categories", type=Path, default=starter / "categories.csv")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "local" / "lora" / "dataset")
    parser.add_argument("--layout", choices=["fal", "aitk", "both"], default="both")
    parser.add_argument("--max-inputs", type=int, default=4, help="references per pair (fal max 4)")
    parser.add_argument("--size", type=parse_size, default=(832, 1248), help="canvas, e.g. 832x1248")
    parser.add_argument("--limit", type=int, default=None, help="export only the first N pairs")
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="keep targets/inputs the automatic checks would drop",
    )
    args = parser.parse_args(argv)

    if not args.pairs.is_file():
        raise SystemExit(f"error: --pairs is not a file: {args.pairs}")
    if not args.dataset.is_dir():
        raise SystemExit(f"error: --dataset is not a directory: {args.dataset}")
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = load_jsonl(args.pairs)
    val_ids = {str(row.get("pair_id") or row.get("garment_id") or "") for row in load_jsonl(args.val)}
    print(f"pairs: {len(pairs)} from {args.pairs}")
    print(f"holdout guard: {len(val_ids)} validation pair ids excluded")
    result = export(args, pairs, val_ids)

    info = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pairs_file": str(args.pairs),
        "val_file": str(args.val),
        "dataset_root": str(args.dataset),
        "labels_file": str(args.labels),
        "categories_file": str(args.categories),
        "layout": args.layout,
        "canvas": {"width": args.size[0], "height": args.size[1]},
        "max_inputs": args.max_inputs,
        "quality_filter": not args.no_quality_filter,
        **result,
    }
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    skipped = result["skipped"]
    assert isinstance(skipped, dict)
    print(f"exported {result['samples']} pairs to {args.out}")
    print(
        "skipped: "
        + ", ".join(f"{key}={value}" for key, value in skipped.items() if value)
        + " (val, label, missing, quality, no_inputs, input_quality, limit)"
    )
    if "fal_zip" in result:
        zip_info = result["fal_zip"]
        assert isinstance(zip_info, dict)
        print(f"fal.zip: {zip_info['bytes'] / 1024 / 1024:.1f} MB, sha256 {zip_info['sha256'][:16]}...")
    print(f"preview: {result.get('preview')}")
    print("next: scripts/lora/train_fal.py (dry-run first) or scripts/lora/train_vast.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

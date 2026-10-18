"""Quick input-vs-preprocessed comparison for one photo, a folder or a glob.

For every image it writes a strip to ``local/preprocess_compare/<name>.png`` with the original
on the left and the reference the model will actually receive on the right (optionally with a
garment mask and an amplified difference panel), and prints the capture statistics of both.

    python scripts/compare_preprocess.py local/captures/IMG_1.jpg
    python scripts/compare_preprocess.py local/captures --diff
    python scripts/compare_preprocess.py "data/.../train/t0011/*.jpg" --with-mask

No paid model is called: this is only OpenCV.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from zir_image_enhancer.reconstruction import preprocess as pp

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _expand(paths: list[str]) -> list[Path]:
    """Files from explicit paths, folders and shell-style globs, without duplicates."""
    found: list[Path] = []
    for value in paths:
        matches = glob.glob(value, recursive=True)
        candidates = (Path(match) for match in matches) if matches else (Path(value),)
        for candidate in candidates:
            if candidate.is_dir():
                found += sorted(
                    child
                    for child in candidate.iterdir()
                    if child.suffix.lower() in _IMAGE_SUFFIXES
                )
            elif candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
                found.append(candidate)
    seen: set[Path] = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _label(bgr: np.ndarray, text: str) -> np.ndarray:
    """Black-outlined label so it reads on any background."""
    for thickness, colour in ((3, (15, 15, 15)), (1, (250, 250, 250))):
        cv2.putText(bgr, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, thickness)
    return bgr


def strip(panels: list[tuple[str, np.ndarray]], height: int) -> np.ndarray:
    """A horizontal strip of labelled BGR tiles, all scaled to the same height."""
    tiles = []
    for label, bgr in panels:
        scale = height / bgr.shape[0]
        tile = cv2.resize(bgr, (max(1, round(bgr.shape[1] * scale)), height))
        tiles.append(_label(tile, label))
    return cv2.hconcat(tiles)


def difference(before: np.ndarray, after: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Amplified absolute pixel difference as a heatmap (BGR)."""
    height, width = after.shape[:2]
    resized = cv2.resize(before, (width, height), interpolation=cv2.INTER_AREA)
    delta = cv2.absdiff(resized, after)
    gray = cv2.cvtColor(delta, cv2.COLOR_RGB2GRAY)
    amplified = pp.to_u8(np.clip(gray.astype(np.float32) * gain, 0, 255))
    return cv2.applyColorMap(amplified, cv2.COLORMAP_TURBO)


def compare_one(
    path: Path,
    out_dir: Path,
    settings: pp.PreprocessSettings,
    *,
    height: int = 520,
    with_diff: bool = False,
    with_mask: bool = False,
) -> dict[str, float]:
    """Write one comparison strip and return before/after statistics."""
    original = pp.read_image(path)
    mask = pp.garment_mask(original)
    enhanced = pp.enhance_full(original, settings, mask)
    enhanced = pp.finalize(enhanced, settings, pp.grain_seed(path))

    panels: list[tuple[str, np.ndarray]] = [("input", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))]
    if with_mask and mask is not None:
        overlay = cv2.cvtColor(original, cv2.COLOR_RGB2BGR).copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(
            np.uint8
        )
        panels.append(("mask", overlay))
    panels.append(("preprocessed", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)))
    if with_diff:
        panels.append(("difference x4", difference(original, enhanced)))

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = strip(panels, height)
    destination = out_dir / f"{path.stem}.png"
    ok, buffer = cv2.imencode(".png", sheet)
    if not ok:
        raise ValueError(f"cannot encode PNG: {destination}")
    buffer.tofile(str(destination))

    before = pp.image_stats(original)
    after = pp.image_stats(enhanced)
    original_small: np.ndarray
    if original.shape[:2] != enhanced.shape[:2]:
        original_small = cv2.resize(
            original, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_AREA
        )
    else:
        original_small = original
    delta = float(np.abs(original_small.astype(np.float32) - enhanced.astype(np.float32)).mean())
    return {
        "brightness_before": before["brightness"],
        "brightness_after": after["brightness"],
        "contrast_before": before["contrast"],
        "contrast_after": after["contrast"],
        "mean_abs_diff": round(delta, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", help="image files, folders or globs")
    parser.add_argument("--out", type=Path, default=Path("local/preprocess_compare"))
    parser.add_argument("--height", type=int, default=520, help="strip height in pixels")
    parser.add_argument("--diff", action="store_true", help="add an amplified difference panel")
    parser.add_argument("--with-mask", action="store_true", help="add the garment mask panel")
    parser.add_argument("--grain", type=float, default=None, help="grain sigma in 0-255 levels")
    parser.add_argument("--sharpen", type=float, default=None)
    parser.add_argument(
        "--wb-method",
        choices=pp.WHITE_BALANCE_METHODS,
        default=None,
        help="illuminant estimator for the white balance (default: gray_world)",
    )
    parser.add_argument(
        "--flatten",
        type=float,
        default=None,
        metavar="STRENGTH",
        help="illumination flattening in [0, 1]; evens out cast shadows and vignetting",
    )
    parser.add_argument("--no-white-balance", action="store_true")
    parser.add_argument("--no-tone", action="store_true")
    args = parser.parse_args(argv)

    settings = pp.PreprocessSettings()
    if args.grain is not None:
        settings = replace(settings, grain_sigma=args.grain)
    if args.sharpen is not None:
        settings = replace(settings, sharpen=args.sharpen)
    if args.wb_method is not None:
        settings = replace(settings, white_balance_method=args.wb_method)
    if args.flatten is not None:
        settings = replace(settings, flatten_illumination=args.flatten)
    if args.no_white_balance:
        settings = replace(settings, white_balance=0.0)
    if args.no_tone:
        settings = replace(settings, auto_tone=False)

    photos = _expand(args.photos)
    if not photos:
        parser.error("no image files matched")

    for photo in photos:
        stats = compare_one(
            photo,
            args.out,
            settings,
            height=args.height,
            with_diff=args.diff,
            with_mask=args.with_mask,
        )
        print(
            f"{photo.name}: brightness {stats['brightness_before']:g} -> "
            f"{stats['brightness_after']:g}, contrast {stats['contrast_before']:g} -> "
            f"{stats['contrast_after']:g}, mean |diff| {stats['mean_abs_diff']:g}"
        )
    print(f"wrote {len(photos)} strip(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

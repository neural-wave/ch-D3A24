"""Build a manifest from our own captures, one folder per garment id.

    local/captures/g01/01_front.jpg
    local/captures/g01/02_detail.jpg
    local/captures/g02/...

    python scripts/make_capture_manifest.py --photos local/captures

Photos are listed in name order, and the protocol says the first one is the front, so name
them so the front sorts first. HEIC is converted to JPEG next to the original, because the
edit model and the scorer do not read HEIC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def to_readable(path: Path) -> Path:
    if path.suffix.lower() not in {".heic", ".heif"}:
        return path
    converted = path.with_suffix(".jpg")
    if not converted.exists():
        # EXIF rotation is applied here, so the photo is upright in every later step.
        ImageOps.exif_transpose(Image.open(path)).convert("RGB").save(converted, quality=95)
    return converted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--photos", type=Path, required=True, help="one sub-folder per garment")
    parser.add_argument("--out", type=Path, default=None, help="default: <photos>/inputs.jsonl")
    args = parser.parse_args(argv)
    register_heif_opener()

    rows = []
    for folder in sorted(path for path in args.photos.iterdir() if path.is_dir()):
        photos = sorted(
            path for path in folder.iterdir() if path.suffix.lower() in PHOTO_EXTENSIONS
        )
        readable = []
        for photo in photos:
            converted = to_readable(photo)
            if converted not in readable:
                readable.append(converted)
        if not readable:
            print(f"  ! {folder.name}: no photos")
            continue
        # Absolute paths: the loader joins them to the dataset root, and an absolute path wins.
        rows.append(
            {
                "garment_id": folder.name,
                "view": "front",
                "inputs": [str(path.resolve()) for path in readable],
            }
        )
        print(f"  {folder.name}: {len(readable)} photo(s), first = {readable[0].name}")

    out = args.out or args.photos / "inputs.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"{len(rows)} garments -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

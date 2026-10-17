"""Pack one generated image per garment into the leaderboard ZIP and check the limits first.

    python scripts/make_zip.py --images local/runs/dev_pipeline/cand_0 --out oneshot.zip
    python scripts/make_zip.py --images local/submit/harness --out local/submit/harness.zip

The ZIP holds <garment>/front.png at its root, one per garment in the split's manifest.
Limits from the submission page: PNG, JPG or WebP; at most 20 MiB per image; sides between
256 and 6144 px; at most 24 MP; aspect ratio at most 3:1; at most 100 MiB per ZIP.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from PIL import Image

from zir_image_enhancer.paths import ProjectPaths

_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}
_MAX_IMAGE_BYTES = 20 * 2**20
_MAX_ZIP_BYTES = 100 * 2**20


def check_image(path: Path) -> list[str]:
    problems = []
    if path.suffix.lower() not in _FORMATS:
        problems.append(f"format {path.suffix}")
    if path.stat().st_size > _MAX_IMAGE_BYTES:
        problems.append("larger than 20 MiB")
    with Image.open(path) as image:
        width, height = image.size
    if min(width, height) < 256 or max(width, height) > 6144:
        problems.append(f"size {width}x{height}")
    if width * height > 24_000_000:
        problems.append("more than 24 MP")
    if max(width, height) / min(width, height) > 3:
        problems.append("aspect ratio above 3:1")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True, help="folder with <garment>/<view>.*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="our own photos; overrides --split"
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    manifest = args.manifest or paths.data_root / args.split / "inputs.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    errors = []
    files: list[tuple[Path, str]] = []
    for row in rows:
        garment, view = str(row["garment_id"]), str(row.get("view") or "front")
        found = sorted(
            p for p in (args.images / garment).glob(f"{view}.*") if p.suffix.lower() in _FORMATS
        )
        if not found:
            errors.append(f"{garment}: missing")
            continue
        errors += [f"{garment}: {problem}" for problem in check_image(found[0])]
        files.append((found[0], f"{garment}/{view}{found[0].suffix.lower()}"))
    if errors:
        print("not packed:\n  " + "\n  ".join(errors))
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in files:
            archive.write(source, name)
    size = args.out.stat().st_size
    if size > _MAX_ZIP_BYTES:
        print(f"{args.out} is {size / 2**20:.1f} MiB, above the 100 MiB limit")
        return 1
    print(f"{args.out}: {len(files)} images, {size / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

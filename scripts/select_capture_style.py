"""Write a sub-manifest of pairs whose first photo was taken in a given capture style.

Used to test prompts made for our own capture protocol (garment laid flat, shot from
above) on example pairs that already look like that, since our own photos have no
reference to score against.

    python scripts/select_capture_style.py --manifest local/neural-wave-2026/holdout/train.jsonl \
        --style flat --out local/neural-wave-2026/holdout/train_flat.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps

from zir_image_enhancer.paths import ProjectPaths

STYLES: dict[str, str] = {
    "flat": "a photo of a garment lying flat on a floor or a table, seen from above",
    "hanger": "a photo of a garment hanging on a hanger",
    "mannequin": "a photo of a garment on a mannequin or a dress form",
    "worn": "a photo of a person wearing a garment",
    "studio": "a product photo of a garment on a plain white background",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--style", choices=sorted(STYLES), default="flat")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-probability", type=float, default=0.5)
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        text = model.encode_text(tokenizer(list(STYLES.values())))
    text = text / text.norm(dim=-1, keepdim=True)
    style_index = list(STYLES).index(args.style)

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = []
    counts = dict.fromkeys(STYLES, 0)
    for row in rows:
        photo = ImageOps.exif_transpose(Image.open(paths.data_root / row["inputs"][0]))
        with torch.no_grad():
            features = model.encode_image(preprocess(photo.convert("RGB")).unsqueeze(0))
        features = features / features.norm(dim=-1, keepdim=True)
        probabilities = (100.0 * features @ text.T).softmax(dim=-1)[0]
        counts[list(STYLES)[int(probabilities.argmax())]] += 1
        if float(probabilities[style_index]) >= args.min_probability:
            selected.append(row)

    args.out.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    print("first-photo styles:", counts)
    print(f"{len(selected)} pairs with '{args.style}' >= {args.min_probability} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

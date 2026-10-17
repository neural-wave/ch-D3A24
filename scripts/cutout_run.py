"""Deterministic runs (no generative model, no cost), in the same layout as generation runs.

    # The input photo itself, cut out and framed on the background: a zero-cost pipeline
    python scripts/cutout_run.py --source inputs --run holdout_cutout_inputs

    # Post-process an existing generation run: same background and framing for every output
    python scripts/cutout_run.py --source run:holdout_nb21p_v1_all --run holdout_nb21p_v1_post

Score the result with ``scripts/score_run.py --run <run>``.
"""

from __future__ import annotations

import argparse
import json

from PIL import Image

from zir_image_enhancer.dataset import load_manifest, split_manifest
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.preprocess.cutout import DEFAULT_SEGMENTER, CutoutSettings, GarmentCutter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--source", required=True, help="'inputs' or 'run:<run name>'")
    parser.add_argument("--run", required=True, help="name of the new run")
    parser.add_argument("--background", type=int, default=240, help="grey level of the background")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--segmenter", default=DEFAULT_SEGMENTER)
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    settings = CutoutSettings(
        background=(args.background,) * 3, margin=args.margin, min_long_side=0
    )
    cutter = GarmentCutter(settings, segmenter=args.segmenter)
    run_dir = paths.runs_root / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": args.run,
                "model_key": "cutout",
                "prompt_key": args.source,
                "inputs": "first",
                "samples": 1,
                "background": args.background,
                "segmenter": args.segmenter,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (run_dir / "meta.jsonl").open("w", encoding="utf-8") as meta:
        for item in load_manifest(split_manifest(args.split, paths), paths.data_root):
            if args.source == "inputs":
                source = item.inputs[0]
            else:
                source = paths.runs_root / args.source.removeprefix("run:") / "cand_0"
                source = source / item.item_id / f"{item.view}.png"
            if not source.exists():
                print(f"  ! {item.item_id}: no source {source}")
                continue
            result = cutter.cut(Image.open(source))
            destination = run_dir / "cand_0" / item.item_id / f"{item.view}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(destination)
            meta.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "sample": 0,
                        "cost_usd": 0.0,
                        "mask_fraction": round(result.mask_fraction, 4),
                        "fallback": result.used_fallback,
                    }
                )
                + "\n"
            )
            print(f"  {item.item_id}{'  (fallback)' if result.used_fallback else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

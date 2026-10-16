"""End-to-end pipeline for a split: generate the packshots, then export the submission.

    python scripts/run_pipeline.py --split dev
    python scripts/run_pipeline.py --split test

Uses the configuration that scored best on our holdout (see the README):
nano-banana 2.1 on a 2:3 portrait canvas, the Sartiq-style prompt and one photo: the first
one that shows the whole garment, skipping tags, bags and close-ups (local CLIP). More photos
scored lower on multi-photo train pairs, and CLIP alone cannot tell a front from a back.
Results are cached, so a rerun only pays for garments that changed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

DEFAULT_MODEL = "nano-banana-2.1-portrait"
DEFAULT_PROMPT = "sartiq-style-v2"
DEFAULT_INPUTS = "first-usable"
DEFAULT_SEED = 0


def run_step(script: str, arguments: list[str]) -> None:
    # Each step is a script of its own, so it can also be run and debugged by hand.
    subprocess.run([sys.executable, str(SCRIPTS / script), *arguments], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--run", default=None, help="run name (default: <split>_pipeline)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--inputs",
        default=None,
        choices=["first", "first-usable", "best", "all", "filtered"],
        help=(
            f"which photos to send (default: '{DEFAULT_INPUTS}'; on test the second supplied "
            "photo is always the back, which can leak a back print onto the front)"
        ),
    )
    parser.add_argument(
        "--captures",
        type=Path,
        default=None,
        help="manifest of our own photos (make_capture_manifest.py) instead of the supplied ones",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="e.g. sartiq-flat-v1 for flat-lay captures"
    )
    parser.add_argument(
        "--wb-card",
        default=None,
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        help="white balance from the grey card in this corner before generation",
    )
    parser.add_argument("--dry-run", action="store_true", help="show the plan and cost only")
    args = parser.parse_args(argv)

    inputs = args.inputs or DEFAULT_INPUTS
    run_name = args.run or f"{args.split}_pipeline"
    generate_args = [
        "--split", args.split,
        "--model", DEFAULT_MODEL,
        "--prompt", args.prompt,
        "--inputs", inputs,
        "--seed-base", str(DEFAULT_SEED),
        "--workers", str(args.workers),
        "--run", run_name,
    ]  # fmt: skip
    if args.captures is not None:
        # The garment ids stay the official ones (g01..g15), so export works unchanged.
        generate_args += ["--manifest", str(args.captures)]
    if args.wb_card is not None:
        generate_args += ["--preprocess", "wb-card", "--wb-corner", args.wb_card]
    if args.dry_run:
        run_step("generate.py", [*generate_args, "--dry-run"])
        return 0
    run_step("generate.py", generate_args)
    run_step("export_submission.py", ["--run", run_name, "--split", args.split])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

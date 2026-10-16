"""Generate packshot candidates for a split with one model and one prompt.

Examples (from the repo root):

    # Reproduce the official baseline on our holdout (about 38 calls, ~$2.30)
    python scripts/generate.py --split holdout --model nano-banana-2.1 --prompt baseline \
        --inputs first --run holdout_nb21_baseline

    # Always check the plan and the cost first: no network call, no key needed
    python scripts/generate.py --split holdout --model flux-2 --prompt studio-white-v1 \
        --inputs all --samples 2 --limit 15 --run holdout_flux2_v1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from zir_image_enhancer.dataset import load_manifest, split_manifest
from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator
from zir_image_enhancer.generation.models import MODELS
from zir_image_enhancer.generation.prompts import TEMPLATES
from zir_image_enhancer.generation.runner import (
    GenerationRun,
    InputSelection,
    PhotoSelector,
    RunConfig,
)
from zir_image_enhancer.paths import ProjectPaths

DEFAULT_BUDGET_USD = 70.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split", default="holdout", help="holdout, holdout-train, train, dev or test"
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--prompt", default="studio-white-v1", choices=sorted(TEMPLATES))
    parser.add_argument(
        "--inputs", default="first", choices=[mode.value for mode in InputSelection]
    )
    parser.add_argument("--samples", type=int, default=1, help="candidates per garment (K)")
    parser.add_argument(
        "--seed-base", type=int, default=None, help="seed of sample 0, if the model takes seeds"
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N garments")
    parser.add_argument("--ids", nargs="*", default=None, help="only these item ids")
    parser.add_argument("--run", required=True, help="run name; outputs go to local/runs/<run>/")
    parser.add_argument("--workers", type=int, default=4, help="parallel fal calls")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="any manifest file; overrides --split"
    )
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=["none", "cutout", "crop", "denoise"],
        help=(
            "cutout: garment cut out on white; crop: framed around the garment, real "
            "background kept (both upscale small photos); denoise: mild OpenCV denoise."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost, call nothing"
    )
    return parser.parse_args(argv)


def build_preprocessor(name: str, paths: ProjectPaths) -> Callable[[Path], Path] | None:
    # Heavy dependencies are imported only when the option is used, so plain runs do not
    # need rembg, OpenCV or torch installed.
    cache_dir = paths.repo_root / "local" / "preprocessed"
    if name == "none":
        return None
    if name == "denoise":
        from zir_image_enhancer.preprocess.denoise import Denoiser

        return Denoiser(cache_dir=cache_dir).denoise_file

    from zir_image_enhancer.preprocess.cutout import CutoutSettings, GarmentCutter

    settings = CutoutSettings(
        background=(255, 255, 255),
        margin=0.08,
        min_long_side=1024,
        keep_background=(name == "crop"),
    )
    return GarmentCutter(settings, cache_dir=cache_dir).cut_file


def build_photo_selector(inputs: str) -> PhotoSelector | None:
    if inputs != InputSelection.FILTERED.value:
        return None
    from zir_image_enhancer.preprocess.photo_filter import PhotoFilter

    photo_filter = PhotoFilter()
    decisions: dict[tuple[Path, ...], tuple[Path, ...]] = {}

    def select(photos: tuple[Path, ...]) -> tuple[Path, ...]:
        # plan() runs more than once per run (cost estimate, then execution); remember
        # each decision so CLIP runs once per garment and the log is printed once.
        if photos not in decisions:
            decision = photo_filter.select(photos)
            decisions[photos] = decision.kept
            for verdict in decision.verdicts:
                mark = "keep" if verdict.path in decision.kept else "drop"
                print(
                    f"  filter {mark} {verdict.path.parent.name}/{verdict.path.name}: "
                    f"{verdict.label} (front {verdict.front_probability:.2f})"
                )
        return decisions[photos]

    return select


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.from_env()
    manifest = args.manifest or split_manifest(args.split, paths)
    items = load_manifest(manifest, paths.data_root)
    if args.ids:
        wanted = set(args.ids)
        items = [item for item in items if item.item_id in wanted]
    if args.limit is not None:
        items = items[: args.limit]

    config = RunConfig(
        name=args.run,
        model_key=args.model,
        prompt_key=args.prompt,
        inputs=InputSelection(args.inputs),
        samples=args.samples,
        seed_base=args.seed_base,
        preprocess=args.preprocess,
    )
    budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", DEFAULT_BUDGET_USD))
    ledger = CostLedger(paths.fal_ledger)
    preprocessor = build_preprocessor(args.preprocess, paths)
    photo_selector = build_photo_selector(args.inputs)

    if args.dry_run:
        # Built without a generator, so a dry run never needs FAL_KEY.
        run = GenerationRun(config, None, paths.runs_root, preprocessor, photo_selector)
        planned = run.plan(items)
        print(f"run      : {run.run_dir}")
        print(f"garments : {len(items)}   calls: {len(planned)}")
        print(f"cost     : up to ${run.estimated_cost_usd(items):.2f} (cached calls are free)")
        print(f"spent    : ${ledger.spent_usd():.2f} of ${budget:.2f}")
        print("prompt   :", planned[0][1].prompt if planned else "-")
        return 0

    if not os.environ.get("FAL_KEY"):
        print("error: FAL_KEY is not set. Put it in .env at the repo root.", file=sys.stderr)
        return 2

    generator = FalGenerator(cache_dir=paths.fal_cache, ledger=ledger, budget_usd=budget)
    run = GenerationRun(config, generator, paths.runs_root, preprocessor, photo_selector)
    summary = run.execute(items, workers=args.workers)
    print(
        f"done: {summary.generated} generated, {summary.from_cache} from cache, "
        f"{summary.failed} failed, ${summary.cost_usd:.2f}, {summary.seconds:.0f}s"
    )
    print(f"total spent on this machine: ${ledger.spent_usd():.2f} of ${budget:.2f}")
    if summary.stopped_by_budget:
        print("stopped: budget limit reached", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

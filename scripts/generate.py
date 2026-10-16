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

from zir_image_enhancer.dataset import load_manifest, split_manifest
from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator
from zir_image_enhancer.generation.models import MODELS
from zir_image_enhancer.generation.prompts import TEMPLATES
from zir_image_enhancer.generation.runner import GenerationRun, InputSelection, RunConfig
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
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost, call nothing"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.from_env()
    items = load_manifest(split_manifest(args.split, paths), paths.data_root)
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
    )
    budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", DEFAULT_BUDGET_USD))
    ledger = CostLedger(paths.fal_ledger)

    if args.dry_run:
        # Built without a generator, so a dry run never needs FAL_KEY.
        run = GenerationRun(config, generator=None, runs_root=paths.runs_root)
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
    run = GenerationRun(config, generator, paths.runs_root)
    summary = run.execute(items)
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

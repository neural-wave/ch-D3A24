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
import json
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
    TriagedPhotos,
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
        choices=["none", "cutout", "crop", "denoise", "wb-card"],
        help=(
            "cutout: garment cut out on white; crop: framed around the garment, real "
            "background kept (both upscale small photos); denoise: mild OpenCV denoise; "
            "wb-card: white balance from a grey card in a fixed corner, card painted out."
        ),
    )
    parser.add_argument(
        "--max-closeups",
        type=int,
        default=0,
        help=(
            "close-ups of front details sent with --inputs triage (default 0: on 11 multi-photo "
            "train pairs they lowered the score by about 2 points)"
        ),
    )
    parser.add_argument(
        "--wb-corner",
        default="bottom-right",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        help="corner that holds the grey card (with --preprocess wb-card)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost, call nothing"
    )
    return parser.parse_args(argv)


def build_preprocessor(
    name: str, paths: ProjectPaths, wb_corner: str = "bottom-right"
) -> Callable[[Path], Path] | None:
    # Heavy dependencies are imported only when the option is used, so plain runs do not
    # need rembg, OpenCV or torch installed.
    cache_dir = paths.repo_root / "local" / "preprocessed"
    if name == "none":
        return None
    if name == "denoise":
        from zir_image_enhancer.preprocess.denoise import Denoiser

        return Denoiser(cache_dir=cache_dir).denoise_file
    if name == "wb-card":
        from zir_image_enhancer.preprocess.white_balance import CardSettings, GreyCardBalancer

        return GreyCardBalancer(CardSettings(corner=wb_corner), cache_dir=cache_dir).balance_file

    from zir_image_enhancer.preprocess.cutout import CutoutSettings, GarmentCutter

    settings = CutoutSettings(
        background=(255, 255, 255),
        margin=0.08,
        min_long_side=1024,
        keep_background=(name == "crop"),
    )
    return GarmentCutter(settings, cache_dir=cache_dir).cut_file


def build_photo_selector(inputs: str) -> PhotoSelector | None:
    modes = {
        InputSelection.FILTERED.value: "select",
        InputSelection.BEST.value: "select",
        InputSelection.FIRST_USABLE.value: "first_usable",
    }
    if inputs not in modes:
        return None
    from zir_image_enhancer.preprocess.photo_filter import PhotoFilter

    photo_filter = PhotoFilter()
    decisions: dict[tuple[Path, ...], tuple[Path, ...]] = {}

    def select(photos: tuple[Path, ...]) -> tuple[Path, ...]:
        # plan() runs more than once per run (cost estimate, then execution); remember
        # each decision so CLIP runs once per garment and the log is printed once.
        if photos not in decisions:
            decision = getattr(photo_filter, modes[inputs])(photos)
            decisions[photos] = decision.kept
            for verdict in decision.verdicts:
                mark = "keep" if verdict.path in decision.kept else "drop"
                print(
                    f"  filter {mark} {verdict.path.parent.name}/{verdict.path.name}: "
                    f"{verdict.label} (front {verdict.front_probability:.2f})"
                )
        return decisions[photos]

    return select


class TriageLog:
    """Runs the vision triage once per garment and keeps every decision for the run log."""

    def __init__(
        self, ledger: CostLedger, budget: float, paths: ProjectPaths, max_closeups: int = 0
    ) -> None:
        from zir_image_enhancer.preprocess.photo_triage import VisionTriage

        self._triage = VisionTriage(
            ledger, budget, paths.repo_root / "local" / "triage_cache", max_closeups=max_closeups
        )
        self._decisions: dict[tuple[Path, ...], dict[str, object]] = {}
        self._photos: dict[tuple[Path, ...], TriagedPhotos] = {}

    def __call__(self, photos: tuple[Path, ...]) -> TriagedPhotos:
        if photos not in self._photos:
            result = self._triage.triage(photos)
            self._photos[photos] = TriagedPhotos(
                result.selected, len(result.detail_closeups), result.details
            )
            self._decisions[photos] = {"garment": photos[0].parent.name, **result.to_json()}
            closeups = ", ".join(path.name for path in result.detail_closeups) or "none"
            print(
                f"  triage {photos[0].parent.name}: front {result.front.name}, "
                f"close-ups {closeups}{'  (fallback)' if result.used_fallback else ''}"
            )
            if result.fine_details:
                # Not an error: a flag for the report and the UI, so a person (or a later
                # verification step) checks these details in the output first.
                print(
                    f"  DETAILS {photos[0].parent.name}: fine details on the front "
                    f"({result.details}); check them in the output"
                )
        return self._photos[photos]

    def write(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = (json.dumps(decision) + "\n" for decision in self._decisions.values())
        (run_dir / "triage.jsonl").write_text("".join(lines), encoding="utf-8")


def build_triager(
    inputs: str, ledger: CostLedger, budget: float, paths: ProjectPaths, max_closeups: int
) -> TriageLog | None:
    if inputs != InputSelection.TRIAGE.value:
        return None
    return TriageLog(ledger, budget, paths, max_closeups)


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
    preprocessor = build_preprocessor(args.preprocess, paths, args.wb_corner)
    photo_selector = build_photo_selector(args.inputs)

    if args.dry_run:
        # Built without a generator, so a dry run never needs FAL_KEY.
        triager = build_triager(args.inputs, ledger, budget, paths, args.max_closeups)
        run = GenerationRun(config, None, paths.runs_root, preprocessor, photo_selector, triager)
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
    triager = build_triager(args.inputs, ledger, budget, paths, args.max_closeups)
    run = GenerationRun(config, generator, paths.runs_root, preprocessor, photo_selector, triager)
    summary = run.execute(items, workers=args.workers)
    if triager is not None:
        triager.write(run.run_dir)
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

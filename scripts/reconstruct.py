"""Run the reconstruction pipeline over a split or manifest, in the scorer's layout.

    python scripts/reconstruct.py --split holdout --run holdout_recon
    python scripts/reconstruct.py --split holdout --run holdout_recon_nogate --no-gate

Writes local/runs/<run>/cand_0/<id>/front.png (scored by score_run.py) and
local/runs/<run>/reports/<id>.json (spec, references, prompts, gate results).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from zir_image_enhancer.dataset import GarmentItem, load_manifest, split_manifest
from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction.preprocess import (
    WHITE_BALANCE_METHODS,
    PreprocessSettings,
)
from zir_image_enhancer.reconstruction.reconstructor import (
    ReconstructionResult,
    ReconstructionSettings,
    Reconstructor,
)
from zir_image_enhancer.reconstruction.vision import VisionClient


def report(item: GarmentItem, result: ReconstructionResult) -> dict[str, object]:
    return {
        "garment": item.item_id,
        "spec": result.spec.to_json(),
        "preprocess": dict(result.preprocess_stats),
        "attempts": [
            {
                "references": list(a.references),
                "prompt": a.prompt,
                "gate": asdict(a.gate) if a.gate else None,
            }
            for a in result.attempts
        ],
        "final": result.final.name,
        "cost_usd": round(result.cost_usd, 4),
        "seconds": round(result.seconds, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="nano-banana-2.1-portrait")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-detail-photos", action="store_true")
    parser.add_argument(
        "--no-preprocess", action="store_true", help="send the original photos, unprocessed"
    )
    parser.add_argument(
        "--no-detail-crops", action="store_true", help="do not crop spec details from the front"
    )
    parser.add_argument(
        "--crop-to-garment", action="store_true", help="frame the garment tightly (tested worse)"
    )
    parser.add_argument(
        "--wb-method",
        choices=WHITE_BALANCE_METHODS,
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
    parser.add_argument(
        "--grain", type=float, default=None, help="controlled grain sigma in 0-255 levels"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    if not os.environ.get("FAL_KEY"):
        print("error: FAL_KEY is not set. Put it in .env at the repo root.", file=sys.stderr)
        return 2
    items = load_manifest(args.manifest or split_manifest(args.split, paths), paths.data_root)
    if args.ids:
        items = [item for item in items if item.item_id in set(args.ids)]
    items = items[: args.limit] if args.limit is not None else items

    preprocess = PreprocessSettings()
    if args.no_preprocess:
        preprocess = replace(preprocess, enabled=False)
    if args.no_detail_crops:
        preprocess = replace(preprocess, detail_crops=False)
    if args.crop_to_garment:
        preprocess = replace(preprocess, crop_to_garment=True)
    if args.wb_method is not None:
        preprocess = replace(preprocess, white_balance_method=args.wb_method)
    if args.flatten is not None:
        preprocess = replace(preprocess, flatten_illumination=args.flatten)
    if args.grain is not None:
        preprocess = replace(preprocess, grain_sigma=args.grain)

    settings = ReconstructionSettings(
        model_key=args.model,
        use_detail_photos=not args.no_detail_photos,
        gate=not args.no_gate,
        preprocess=preprocess,
    )
    budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", 70.0))
    ledger = CostLedger(paths.fal_ledger)
    cache = paths.repo_root / "local" / "reconstruction_cache"
    vision = VisionClient(ledger, budget, cache)
    generator = FalGenerator(cache_dir=paths.fal_cache, ledger=ledger, budget_usd=budget)
    reconstructor = Reconstructor(settings, generator, vision)

    run_dir = paths.runs_root / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": args.run,
        "model_key": args.model,
        "prompt_key": "reconstruction",
        "inputs": "harness",
        "samples": 1,
        **{key: value for key, value in asdict(settings).items() if key != "model_key"},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def one(item: GarmentItem) -> tuple[GarmentItem, ReconstructionResult]:
        return item, reconstructor.run(item, run_dir / "attempts" / item.item_id)

    failed = 0
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        (run_dir / "meta.jsonl").open("w", encoding="utf-8") as meta,
    ):
        futures = [pool.submit(one, item) for item in items]
        for future in as_completed(futures):
            try:
                item, result = future.result()
            except Exception as error:  # noqa: BLE001
                failed += 1
                print(f"  ! {type(error).__name__}: {error}")
                continue
            destination = run_dir / "cand_0" / item.item_id / f"{item.view}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result.final.read_bytes())
            data = report(item, result)
            (run_dir / "reports").mkdir(exist_ok=True)
            (run_dir / "reports" / f"{item.item_id}.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            gates = [a.gate for a in result.attempts if a.gate is not None]
            meta.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "sample": 0,
                        "cost_usd": data["cost_usd"],
                        "seconds": data["seconds"],
                        "attempts": len(result.attempts),
                        "gate_problems": list(gates[0].problems) if gates else [],
                        "details": [d.name for d in result.spec.details],
                    }
                )
                + "\n"
            )
            problems = ", ".join(gates[0].problems) if gates and gates[0].problems else "ok"
            print(f"  {item.item_id}: {len(result.attempts)} attempt(s), gate: {problems}")

    print(f"done: {len(items) - failed} garments, {failed} failed")
    print(f"total spent on this machine: ${ledger.spent_usd():.2f} of ${budget:.2f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

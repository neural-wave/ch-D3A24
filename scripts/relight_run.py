"""Relight packshots that already exist, to test the second pass with one call per garment.

    python scripts/relight_run.py --from-run holdout_nb21p_v2_all --run relight_test \
        --ids t0162 t0216 t0209 t0033 t0021 t0314
    python scripts/score_run.py --run relight_test --pairs local/runs/relight_test/pairs.jsonl

cand_0 is a copy of the original outputs and cand_1 the relit ones, so score_run.py compares
both on the same garments in one go.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

from zir_image_enhancer.generation.fal_backend import CostLedger, FalGenerator, GenerationRequest
from zir_image_enhancer.generation.models import get_model
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction.prompt import build_relight_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from-run", required=True, help="run whose cand_0 gets relit")
    parser.add_argument("--run", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--model", default="nano-banana-2.1-portrait")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    if not os.environ.get("FAL_KEY"):
        print("error: FAL_KEY is not set. Put it in .env at the repo root.", file=sys.stderr)
        return 2
    source = paths.runs_root / args.from_run / "cand_0"
    run_dir = paths.runs_root / args.run
    wanted = set(args.ids)

    # Only the chosen garments are scored; the full holdout would count the rest as missing.
    pairs = [
        line
        for line in paths.holdout_val.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["pair_id"] in wanted
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pairs.jsonl").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    config = {
        "name": args.run,
        "model_key": args.model,
        "prompt_key": "relight-v1",
        "inputs": f"packshot from {args.from_run}",
        "samples": 2,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    budget = float(os.environ.get("ZIR_FAL_BUDGET_USD", 70.0))
    ledger = CostLedger(paths.fal_ledger)
    generator = FalGenerator(cache_dir=paths.fal_cache, ledger=ledger, budget_usd=budget)
    model = get_model(args.model)
    prompt = build_relight_prompt()

    def one(item_id: str) -> dict[str, object]:
        images = sorted((source / item_id).glob("*.png"))
        if not images:
            return {"item_id": item_id, "error": "no output in source run"}
        original = images[0]
        for cand in ("cand_0", "cand_1"):
            (run_dir / cand / item_id).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, run_dir / "cand_0" / item_id / original.name)
        try:
            result = generator.generate(GenerationRequest(model, prompt, (original,), args.seed, 0))
        except Exception as error:  # noqa: BLE001
            # One failed call (balance, timeout) must not lose the rest of the batch.
            shutil.rmtree(run_dir / "cand_0" / item_id)
            shutil.rmtree(run_dir / "cand_1" / item_id)
            return {"item_id": item_id, "error": f"{type(error).__name__}: {error}"}
        (run_dir / "cand_1" / item_id / original.name).write_bytes(result.png_bytes)
        return {
            "item_id": item_id,
            "sample": 1,
            "cost_usd": result.cost_usd,
            "seconds": result.seconds,
            "from_cache": result.from_cache,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, sorted(wanted)))
    with (run_dir / "meta.jsonl").open("w", encoding="utf-8") as meta:
        for row in rows:
            if "error" in row:
                print(f"  ! {row['item_id']}: {row['error']}")
                continue
            meta.write(json.dumps(row) + "\n")
    print(f"total spent on this machine: ${ledger.spent_usd():.2f} of ${budget:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

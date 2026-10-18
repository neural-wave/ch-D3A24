#!/usr/bin/env python3
"""Train the fine-tuned Sartiq packshot LoRA on fal.ai (docs/LORA.md has the full story).

It uploads the ``fal.zip`` produced by ``scripts/lora/prepare_dataset.py`` to the klein 9B
edit trainer, follows the job to the end, downloads the adapter into ``local/lora/weights``
and registers it in ``local/lora/registry.json`` so the pipeline's ``sartiq-lora-klein`` model
picks it up with no code change.

Costs real money, so the script is dry-run by default:

    python scripts/lora/train_fal.py                    # prints the plan, spends nothing
    python scripts/lora/train_fal.py --live             # uploads, trains, registers
    python scripts/lora/train_fal.py --fetch local/lora/runs/<run>.json   # resume polling

Training is billed by fal as ``0.002 * steps * reference_multiplier`` (2.11 / 3.44 / 5.09 /
6.95 for 1 / 2 / 3 / 4 references per pair); the estimate is written to the shared cost
ledger when the job is submitted, so the pipeline's budget guard sees it too.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fal_client
import httpx

from zir_image_enhancer.generation.fal_backend import CostLedger
from zir_image_enhancer.paths import ProjectPaths

#: fal's klein 9B base trainer, edit mode (verified schema: image_data_url, steps, lr).
TRAINER_ENDPOINT = "fal-ai/flux-2-klein-9b-base-trainer/edit"

#: fal bills 0.002 USD per step times this multiplier, one entry per reference count per pair.
REFERENCE_MULTIPLIERS = {1: 2.11, 2: 3.44, 3: 5.09, 4: 6.95}

DEFAULT_STEPS = 1500
_START_RE = re.compile(r"^(?P<root>.+?)_start(?P<index>\d*)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    paths = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zip", type=Path, default=paths.lora_root / "dataset" / "fal.zip")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="100..20000, step of 100")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (fal default 5e-5)")
    parser.add_argument("--name", default="sartiq_packshot", help="run name / weights file name")
    parser.add_argument(
        "--output-lora-format", choices=["fal", "comfy"], default="fal", help="weight naming scheme"
    )
    parser.add_argument("--fetch", type=Path, default=None, help="resume polling a run json")
    parser.add_argument("--max-budget", type=float, default=None, help="hard USD cap for this call")
    parser.add_argument(
        "--live", action="store_true", help="actually upload and train (spends credits)"
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="download the weights but do not touch local/lora/registry.json",
    )
    return parser.parse_args(argv)


def inspect_zip(zip_path: Path) -> dict[str, Any]:
    """Count image pairs, captions and the references-per-pair distribution of a fal archive."""
    if not zip_path.is_file():
        raise SystemExit(f"error: dataset zip not found: {zip_path}. Run prepare_dataset.py first.")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    images = [name for name in names if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    captions = [name for name in names if name.lower().endswith(".txt")]
    refs: Counter[str] = Counter()
    ends = 0
    for name in images:
        stem = Path(name).stem
        if stem.endswith("_end"):
            ends += 1
            refs.setdefault(stem[: -len("_end")], 0)
            continue
        match = _START_RE.match(stem)
        if match:
            refs[match.group("root")] += 1
    distribution = Counter(refs.values())
    return {
        "pairs": ends,
        "images": len(images),
        "captions": len(captions),
        "references_per_pair": dict(sorted(distribution.items())),
        "max_references": max(distribution, default=0),
        "size_mb": zip_path.stat().st_size / (1024 * 1024),
    }


def estimate_cost(steps: int, references: int) -> float:
    """fal's published formula, conservatively using the largest reference count in the set."""
    multiplier = REFERENCE_MULTIPLIERS.get(max(1, min(4, references)), REFERENCE_MULTIPLIERS[4])
    return round(0.002 * steps * multiplier, 2)


def validate(steps: int, lr: float | None) -> None:
    if not 100 <= steps <= 20000 or steps % 100 != 0:
        raise SystemExit("error: --steps must be a multiple of 100 between 100 and 20000")
    if lr is not None and not 0.0 < lr <= 1e-2:
        raise SystemExit("error: --lr must be a small positive number (e.g. 5e-5)")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def print_logs(logs: list[Any], seen: set[str]) -> None:
    for entry in logs:
        message = str(entry.get("message", entry)) if isinstance(entry, dict) else str(entry)
        if message not in seen:
            seen.add(message)
            print(f"[fal] {message}")


def wait_for_result(
    endpoint: str, request_id: str, run_file: Path, run: dict[str, Any]
) -> dict[str, Any]:
    """Poll a submitted training job to completion, printing new logs as they appear."""
    seen: set[str] = set()
    while True:
        try:
            status = fal_client.status(endpoint, request_id, with_logs=True)
        except fal_client.FalClientHTTPError as error:
            if error.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            print(f"[fal] status check failed ({error.status_code}), retrying")
            time.sleep(15)
            continue
        if isinstance(status, fal_client.Queued):
            print(f"[fal] queued, position {status.position}")
        elif isinstance(status, fal_client.InProgress):
            print_logs(list(status.logs or []), seen)
        elif isinstance(status, fal_client.Completed):
            print_logs(list(status.logs or []), seen)
            if status.error:
                run["error"] = f"{status.error_type}: {status.error}"
                run["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
                save_json(run_file, run)
                raise SystemExit(f"error: training failed: {run['error']} (state kept in {run_file})")
            result = dict(fal_client.result(endpoint, request_id))
            run["result"] = result
            run["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
            save_json(run_file, run)
            return result
        time.sleep(15)


def download_weights(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for block in response.iter_bytes(1024 * 1024):
                handle.write(block)
    return destination


def finish_run(
    paths: ProjectPaths,
    run: dict[str, Any],
    run_file: Path,
    name: str,
    register: bool,
) -> Path | None:
    """Download the trained adapter and make it the active one, unless told not to."""
    result = run.get("result") or {}
    lora_file = result.get("diffusers_lora_file") or {}
    url = str(lora_file.get("url") or "")
    if not url:
        print(f"warning: no diffusers_lora_file in the result; state kept in {run_file}")
        return None
    destination = paths.lora_root / "weights" / f"{name}.safetensors"
    local_path: Path | None = None
    try:
        download_weights(url, destination)
        print(f"weights: {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")
        local_path = destination
        run["local_path"] = str(destination)
    except (httpx.HTTPError, OSError) as error:
        print(f"warning: could not download the weights ({error}); registering the fal url instead")
    save_json(run_file, run)

    if not register:
        print("registration skipped (--no-register)")
        return local_path
    binding = {
        "path": str(local_path) if local_path is not None else url,
        "scale": 1.0,
        "name": name,
        "endpoint": run.get("endpoint", TRAINER_ENDPOINT),
        "steps": run.get("steps"),
        "references": run.get("references"),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": str(run_file),
    }
    save_json(paths.lora_registry, binding)
    print(f"registered: {paths.lora_registry} -> {binding['path']}")
    print("use it with:  ZIR_MODEL_KEY=sartiq-lora-klein  (or zir-run --model sartiq-lora-klein)")
    return local_path


def live_run(
    paths: ProjectPaths,
    args: argparse.Namespace,
    stats: dict[str, Any],
    estimate: float,
    budget: float,
) -> int:
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("error: FAL_KEY is not set. Put it in .env at the repo root.")
    ledger = CostLedger(paths.fal_ledger)
    spent = ledger.spent_usd()
    if spent + estimate > budget:
        raise SystemExit(
            f"error: spent ${spent:.2f} of the ${budget:.2f} budget; this run (~${estimate:.2f}) "
            "would cross it. Raise ZIR_FAL_BUDGET_USD / --max-budget if that is intended."
        )

    payload: dict[str, Any] = {
        "image_data_url": "",  # filled after the upload
        "steps": args.steps,
        "output_lora_format": args.output_lora_format,
    }
    if args.lr is not None:
        payload["learning_rate"] = args.lr

    print(f"uploading {args.zip} ({stats['size_mb']:.1f} MB) to fal storage ...")
    payload["image_data_url"] = fal_client.upload_file(str(args.zip))
    handle = fal_client.submit(TRAINER_ENDPOINT, arguments=payload)

    run_file = paths.lora_root / "runs" / f"{args.name}_{args.steps}s_{handle.request_id}.json"
    run: dict[str, Any] = {
        "endpoint": TRAINER_ENDPOINT,
        "request_id": handle.request_id,
        "name": args.name,
        "steps": args.steps,
        "references": stats["max_references"],
        "learning_rate": args.lr,
        "estimated_cost_usd": estimate,
        "payload": payload,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_json(run_file, run)
    # The money is committed at submission: record the estimate now so the budget guard
    # and the team ledger see it even if this script is interrupted while polling.
    ledger.record(
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": "sartiq-lora-klein",
            "endpoint": TRAINER_ENDPOINT,
            "kind": "lora_training_estimate",
            "cache_key": f"lora:{handle.request_id}",
            "cost_usd": estimate,
            "seconds": 0.0,
            "steps": args.steps,
            "references": stats["max_references"],
            "run": str(run_file),
        }
    )
    print(f"submitted: {handle.request_id} -> {run_file}")
    print("training a 9B LoRA usually takes 30-90 minutes; polling every 15 s (Ctrl+C is safe)")

    result = wait_for_result(TRAINER_ENDPOINT, handle.request_id, run_file, run)
    metrics = result.get("metrics") or {}
    if metrics:
        print(f"metrics: {json.dumps(metrics, sort_keys=True)[:400]}")
    finish_run(paths, run, run_file, args.name, register=not args.no_register)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate(args.steps, args.lr)
    paths = ProjectPaths.from_env()
    budget = args.max_budget if args.max_budget is not None else float(
        os.environ.get("ZIR_FAL_BUDGET_USD", 70.0)
    )

    if args.fetch is not None:
        run_file = args.fetch
        if not run_file.is_file():
            raise SystemExit(f"error: run file not found: {run_file}")
        run = json.loads(run_file.read_text(encoding="utf-8"))
        endpoint = str(run.get("endpoint", TRAINER_ENDPOINT))
        request_id = str(run.get("request_id") or "")
        if not request_id:
            raise SystemExit(f"error: {run_file} has no request_id")
        print(f"resuming {request_id} on {endpoint} (no new spend)")
        if "result" not in run:
            wait_for_result(endpoint, request_id, run_file, run)
        name = str(run.get("name") or args.name)
        finish_run(paths, run, run_file, name, register=not args.no_register)
        return 0

    stats = inspect_zip(args.zip)
    estimate = estimate_cost(args.steps, int(stats["max_references"]))
    print(f"dataset: {args.zip} ({stats['size_mb']:.1f} MB, {stats['pairs']} pairs, "
          f"{stats['images']} images, {stats['captions']} captions)")
    print(f"references per pair: {stats['references_per_pair']} (max {stats['max_references']})")
    print(f"trainer: {TRAINER_ENDPOINT}")
    print(f"run: {args.name}, steps {args.steps}, lr {args.lr or 'fal default (5e-5)'}, "
          f"format {args.output_lora_format}")
    print(f"estimated cost: ${estimate:.2f} (fal formula 0.002 * steps * reference multiplier)")

    if not args.live:
        print("dry-run: nothing was uploaded and no credits were spent.")
        print("payload on --live: image_data_url=<uploaded zip>, "
              f"steps={args.steps}, output_lora_format={args.output_lora_format}"
              + (f", learning_rate={args.lr}" if args.lr is not None else ""))
        print("add --live when you are ready to spend the credits.")
        return 0

    return live_run(paths, args, stats, estimate, budget)


if __name__ == "__main__":
    raise SystemExit(main())

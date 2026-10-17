"""Score a generation run on the holdout with the official scorer and log it.

Each ``cand_<k>`` folder is scored as its own "team" by the starter repo's
``eval_val.py``. On top of that we report the best-of-K score: what we would get if
the selector always picked the best candidate, which is the ceiling for selection.

    python scripts/score_run.py --run holdout_nb21_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from zir_image_enhancer.paths import ProjectPaths

SUBSCORES = ("colour", "lpips", "clip", "geometry", "ssim")
RESULT_COLUMNS = (
    "at",
    "run",
    "model",
    "prompt",
    "inputs",
    "samples",
    "pairs",
    "score",
    *SUBSCORES,
    "best_of_k",
    "usd_per_garment",
    "s_per_call",
)


def score_candidate(
    paths: ProjectPaths, run_dir: Path, candidate: str, jobs: int, pairs: Path
) -> Path:
    out_dir = run_dir / "scores" / candidate
    command = [
        str(paths.scorer_python()),
        str(paths.starter_root / "scripts" / "eval_val.py"),
        "--pairs",
        str(pairs),
        "--preds",
        str(run_dir),
        "--team",
        candidate,
        "--out",
        str(out_dir),
        "--dataset",
        str(paths.data_root),
        "--jobs",
        str(jobs),
    ]
    subprocess.run(command, check=True)
    return out_dir


def read_team_row(score_dir: Path) -> dict[str, str]:
    with (score_dir / "leaderboard.csv").open(encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def read_pair_scores(score_dir: Path) -> dict[str, float]:
    with (score_dir / "pairs.csv").open(encoding="utf-8") as handle:
        return {row["pair_id"]: float(row["score"]) for row in csv.DictReader(handle)}


def run_costs(run_dir: Path) -> tuple[float, float, int]:
    """Total cost, mean seconds per paid call, and number of garments, from meta.jsonl."""
    cost = 0.0
    seconds: list[float] = []
    garments: set[str] = set()
    for line in (run_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        garments.add(str(entry["item_id"]))
        cost += float(entry.get("cost_usd", 0.0))
        if not entry.get("from_cache") and entry.get("seconds"):
            seconds.append(float(entry["seconds"]))
    mean_seconds = sum(seconds) / len(seconds) if seconds else 0.0
    return cost, mean_seconds, len(garments)


def append_result(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run name under local/runs/")
    parser.add_argument("--jobs", type=int, default=1, help="scorer workers (~1.4 GB RAM each)")
    parser.add_argument(
        "--pairs", type=Path, default=None, help="pairs manifest to score on (default: holdout)"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="do not append to experiments/results.csv"
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    run_dir = paths.runs_root / args.run
    candidates = sorted(path.name for path in run_dir.glob("cand_*") if path.is_dir())
    if not candidates:
        print(f"error: no cand_* folders in {run_dir}", file=sys.stderr)
        return 2

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    team_rows: list[dict[str, str]] = []
    per_pair: list[dict[str, float]] = []
    for candidate in candidates:
        score_dir = score_candidate(
            paths, run_dir, candidate, args.jobs, args.pairs or paths.holdout_val
        )
        team_rows.append(read_team_row(score_dir))
        per_pair.append(read_pair_scores(score_dir))

    # The average candidate is what a random pick would score; best-of-K is the ceiling.
    pair_ids = sorted(set().union(*per_pair))
    mean_score = sum(float(row["score"]) for row in team_rows) / len(team_rows)
    best_of_k = sum(max(scores.get(pid, 0.0) for scores in per_pair) for pid in pair_ids) / len(
        pair_ids
    )
    cost, seconds_per_call, n_garments = run_costs(run_dir)

    row: dict[str, object] = {
        "at": datetime.now(UTC).isoformat(timespec="minutes"),
        "run": args.run,
        "model": config["model_key"],
        "prompt": config["prompt_key"],
        "inputs": config["inputs"],
        "samples": config["samples"],
        "pairs": len(pair_ids),
        "score": round(mean_score, 2),
        **{
            name: round(sum(float(r[name]) for r in team_rows) / len(team_rows), 4)
            for name in SUBSCORES
        },
        "best_of_k": round(best_of_k, 2),
        "usd_per_garment": round(cost / max(n_garments, 1), 3),
        "s_per_call": round(seconds_per_call, 1),
    }
    for key, value in row.items():
        print(f"{key:>15}: {value}")
    if not args.no_log:
        append_result(paths.experiments_csv, row)
        print(f"logged to {paths.experiments_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

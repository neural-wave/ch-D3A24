"""v4 end to end: several versions per garment, one pick, the jury gate, the ZIP.

    uv run --extra select python scripts/run_v4.py --split test --dry-run
    uv run --extra select python scripts/run_v4.py --split test
    uv run --extra select python scripts/run_v4.py --name ours --manifest ours.jsonl

Steps, each an existing script run in order (every paid call is cached, so a rerun is free):
1. scripts/reconstruct.py once per version below, into local/runs/<name>_<version>/
2. scripts/select_versions.py: CLIP + AlexNet rank the versions per garment, the jury gate
   checks the top 3 and keeps the first without defects, into local/runs/<name>_v4/
3. scripts/make_zip.py: local/submit/<team>_<name>_v4.zip, with the submission limits checked
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from zir_image_enhancer.dataset import load_manifest, split_manifest
from zir_image_enhancer.paths import ProjectPaths

SCRIPTS = Path(__file__).resolve().parent
COST_PER_IMAGE = 0.06  # nano-banana 2.1 at about 1K; the gate and spec add about $0.002 each


@dataclass(frozen=True)
class Version:
    name: str
    mode: str
    style: str
    seed: int


# Distinct prompts and seeds: the selector gained about +0.2 per distinct version on the
# holdout (2 to 9 versions), so the pool is as varied as the budget allows.
VERSIONS = (
    Version("v3", "jury", "auto", 0),
    Version("flat_s1", "oneshot", "flat", 1),
    Version("flat_s2", "oneshot", "flat", 2),
    Version("A", "oneshot", "baseline", 0),
    Version("A_s1", "oneshot", "baseline", 1),
    Version("B", "oneshot", "baseline-keep", 0),
    Version("B_s1", "oneshot", "baseline-keep", 1),
    Version("C", "oneshot", "baseline-white", 0),
    Version("C_s1", "oneshot", "baseline-white", 1),
)


def run(command: list[str]) -> None:
    print("  $", " ".join(command[1:]), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="test", help="dev or test (ignored with --manifest)")
    parser.add_argument("--manifest", type=Path, default=None, help="our own photos")
    parser.add_argument("--name", default=None, help="run prefix (default: the split)")
    parser.add_argument("--team", default="d3a24")
    parser.add_argument("--versions", type=int, default=len(VERSIONS), help="use the first N")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the cost")
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    manifest = args.manifest or split_manifest(args.split, paths)
    garments = len(load_manifest(manifest, paths.data_root))
    name = args.name or args.split
    versions = VERSIONS[: args.versions]
    print(
        f"v4 on {garments} garments, {len(versions)} versions: at most "
        f"${garments * len(versions) * COST_PER_IMAGE:.2f} (cached results are free)"
    )
    if args.dry_run:
        for version in versions:
            print(
                f"  {version.name}: --mode {version.mode} --studio-style {version.style} "
                f"--seed {version.seed}"
            )
        return 0

    source = ["--manifest", str(manifest)] if args.manifest else ["--split", args.split]
    folders = []
    for version in versions:
        run_name = f"{name}_{version.name}"
        command = [
            sys.executable,
            str(SCRIPTS / "reconstruct.py"),
            *source,
            "--run",
            run_name,
            "--mode",
            version.mode,
            "--studio-style",
            version.style,
            "--seed",
            str(version.seed),
            "--workers",
            str(args.workers),
        ]
        if version.mode == "oneshot":
            command.append("--no-gate")  # the selector gates the pick; no need to gate every one
        run(command)
        folders.append(f"{version.name}={paths.runs_root / run_name / 'cand_0'}")

    picked = f"{name}_v4"
    run(
        [
            sys.executable,
            str(SCRIPTS / "select_versions.py"),
            *source,
            "--run",
            picked,
            "--gate-top",
            "3",
            "--versions",
            *folders,
        ]
    )
    zip_path = paths.repo_root / "local" / "submit" / f"{args.team}_{name}_v4.zip"
    zip_source = ["--manifest", str(manifest)] if args.manifest else ["--split", args.split]
    run(
        [
            sys.executable,
            str(SCRIPTS / "make_zip.py"),
            *zip_source,
            "--images",
            str(paths.runs_root / picked / "cand_0"),
            "--out",
            str(zip_path),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

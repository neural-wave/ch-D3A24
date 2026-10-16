"""Copy one candidate of a run into the official submission layout and validate it.

    python scripts/export_submission.py --run dev_nb21_baseline --split dev

Writes ``local/submissions/<team>/<garment_id>/front.png`` for every garment of the
split and runs the starter repo's ``make_submission.py`` on it. A garment without an
output is reported loudly, because a missing garment scores 0.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from zir_image_enhancer.dataset import load_manifest, split_manifest
from zir_image_enhancer.paths import ProjectPaths

DEFAULT_TEAM = "d3a24"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run name under local/runs/")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--candidate", default="cand_0", help="which candidate folder to export")
    parser.add_argument("--team", default=os.environ.get("ZIR_TEAM", DEFAULT_TEAM))
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    source = paths.runs_root / args.run / args.candidate
    submissions = paths.repo_root / "local" / "submissions"
    items = load_manifest(split_manifest(args.split, paths), paths.data_root)

    missing: list[str] = []
    for item in items:
        output = source / item.item_id / f"{item.view}.png"
        if not output.exists():
            missing.append(item.item_id)
            continue
        destination = submissions / args.team / item.item_id / f"{item.view}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale outputs with other extensions: the scorer would read a .png first,
        # but the validator rejects duplicates, so we keep exactly one file per garment.
        for stale in destination.parent.glob(f"{item.view}.*"):
            stale.unlink()
        shutil.copyfile(output, destination)

    print(f"exported {len(items) - len(missing)} of {len(items)} to {submissions / args.team}")
    if missing:
        print(f"MISSING (score 0 if submitted like this): {', '.join(missing)}", file=sys.stderr)

    validation = subprocess.run(
        [
            str(paths.scorer_python()),
            str(paths.starter_root / "scripts" / "make_submission.py"),
            "--submissions",
            str(submissions),
            "--team",
            args.team,
            "--dataset",
            str(paths.data_root),
        ],
        check=False,
    )
    return 1 if missing or validation.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

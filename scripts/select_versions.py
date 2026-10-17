"""v4: pick, per garment, one packshot among several versions, then check it with the jury gate.

    python scripts/select_versions.py --split test --run test_v4 \
        --versions v1=local/runs/test_pipeline/cand_0 v3=local/runs/test_v3/cand_0 ...

Each version is a folder with <garment>/front.png. Two embedding experts rank the candidates
of a garment without its target: CLIP (what the garment is, overall style) and AlexNet (the
LPIPS backbone: texture and local structure). A candidate scores high when it is close to the
house style, measured as the mean similarity to its 10 nearest studio photos of the training
side of our holdout split, and close to the garment's own input photo.

On the 38-garment holdout this picks 63.01 among 9 versions against 61.90 for the best single
version, and the gain grows with every distinct version in the pool (about +0.2 each). The
jury gate then checks the top candidates in rank order and keeps the first one without
defects, so a stylish but wrong image (a cut sleeve, a missing logo) is not published.
The selector itself lives in the package (selection.py, embeddings.py), shared with the app.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from zir_image_enhancer.dataset import load_manifest, split_manifest
from zir_image_enhancer.embeddings import EmbeddingExperts
from zir_image_enhancer.generation.fal_backend import CostLedger
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction.photos import roles_from_filenames
from zir_image_enhancer.reconstruction.spec import extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient
from zir_image_enhancer.selection import (
    GateCheck,
    VersionSelector,
    candidates_in_folders,
    gate_pick,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--versions", nargs="+", required=True, help="name=folder with <id>/front.png"
    )
    parser.add_argument(
        "--gate-top", type=int, default=3, help="candidates the gate may check (0: none)"
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_env()
    items = load_manifest(args.manifest or split_manifest(args.split, paths), paths.data_root)
    versions = {
        name: paths.repo_root / folder for name, folder in (v.split("=", 1) for v in args.versions)
    }
    # No fixed scales: the batch standardises over all its garments, as evaluated.
    ranked = VersionSelector(EmbeddingExperts(paths)).rank(
        items, candidates_in_folders(items, versions)
    )

    vision = VisionClient(
        CostLedger(paths.fal_ledger), 1e9, paths.repo_root / "local" / "reconstruction_cache"
    )
    out_dir = paths.runs_root / args.run
    log = []
    for item in items:
        order = ranked[item.item_id]
        chosen = order[0]
        checks: tuple[GateCheck, ...] = ()
        if args.gate_top > 0:
            spec = extract_spec(roles_from_filenames(item.inputs), vision)
            pick = gate_pick(order, spec, vision, args.gate_top)
            chosen, checks = pick.chosen, pick.checks
        target = out_dir / "cand_0" / item.item_id / f"{item.view}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chosen.image, target)
        log.append(
            {
                "garment": item.item_id,
                "ranking": [c.version for c in order],
                "chosen": chosen.version,
                "gate": [{"version": c.version, "problems": list(c.problems)} for c in checks],
            }
        )
        top = ", ".join(c.version for c in order[:3])
        print(f"  {item.item_id}: {chosen.version} (ranking {top})")
    (out_dir / "picks.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (out_dir / "meta.jsonl").write_text(
        "".join(
            json.dumps({"item_id": i.item_id, "sample": 0, "cost_usd": 0.0}) + "\n" for i in items
        ),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "name": args.run,
                "model_key": "pool",
                "prompt_key": "v4 select: " + ",".join(versions),
                "inputs": "pool",
                "samples": 1,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

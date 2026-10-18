#!/usr/bin/env python3
"""Package the packshot LoRA training for a vast.ai GPU (see docs/LORA.md).

Writes ``local/lora/vast/``: the ai-toolkit training config, a bootstrap ``run.sh``, a
plain-text runbook and ``bundle.zip`` with everything a fresh instance needs -- the
dataset layout from ``scripts/lora/prepare_dataset.py`` (``target/`` with captions plus
``reference_1..4/``), the config and the script. This machine only writes files: nothing
is uploaded, rented or spent here; the credits go to vast.ai when *you* start the run.

The recipe follows BFL's klein "Graphic_Impressions" config (rank 128/64, conv 64/32,
adamw8bit, lr 1e-4, ``timestep_type: shift``, ``content_or_style: balanced``, qfloat8)
on the 832x1248 canvas the pairs are exported on. ``run.sh`` installs ai-toolkit with
the experimental manager (``python3 -m manager install``) and runs ``run.py`` headless.

    python scripts/lora/train_vast.py                   # 9B base, 2000 steps
    python scripts/lora/train_vast.py --base 4b         # Apache-2.0 model, 12GB+ GPUs
    python scripts/lora/train_vast.py --steps 3000 --save-every 250
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zir_image_enhancer.paths import ProjectPaths

#: Divisor of the dataset canvas and of the buckets; drives ai-toolkit's ``bucket_tolerance``
#: and the ``divisibility`` of toolkit/buckets.py, which must produce the canvas itself.
BUCKET_TOLERANCE = 16

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

#: Fallback when dataset_info.json is missing; prepare_dataset.py writes this canvas.
DEFAULT_CANVAS = (832, 1248)

#: The two klein base variants ai-toolkit trains (extensions_built_in/.../flux2); the
#: hardware numbers are BFL's official minima for each base.
BASES: dict[str, dict[str, Any]] = {
    "9b": {
        "model": "black-forest-labs/FLUX.2-klein-base-9B",
        "arch": "flux2_klein_9b",
        "vram": 22,
        "ram": 64,
        "disk": 100,
        "licence": "FLUX Non-Commercial License (do not publish the adapter)",
        "gated": "the repo is gated: accept its licence and set HF_TOKEN",
    },
    "4b": {
        "model": "black-forest-labs/FLUX.2-klein-base-4B",
        "arch": "flux2_klein_4b",
        "vram": 12,
        "ram": 32,
        "disk": 60,
        "licence": "Apache-2.0",
        "gated": "public repo, no HF token needed",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    paths = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", choices=sorted(BASES), default="9b", help="klein variant")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=paths.lora_root / "dataset",
        help="prepare_dataset.py output (holds aitk/ and dataset_info.json)",
    )
    parser.add_argument("--out", type=Path, default=paths.lora_root / "vast")
    parser.add_argument(
        "--val",
        type=Path,
        default=paths.repo_root / "sartiq_lora_unified" / "holdout" / "val.jsonl",
        help="frozen validation split: pairs in here must never end up in the bundle",
    )
    parser.add_argument("--name", default="sartiq_packshot", help="run name / file name")
    parser.add_argument("--steps", type=int, default=2000, help="training steps (100..20000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (BFL: 8e-5..1e-4)")
    parser.add_argument("--save-every", type=int, default=250, help="checkpoint + sample interval")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=128, help="LoRA linear rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA linear alpha")
    parser.add_argument("--conv-rank", type=int, default=64)
    parser.add_argument("--conv-alpha", type=int, default=32)
    parser.add_argument(
        "--timestep-type",
        choices=["shift", "weighted", "sigmoid", "linear"],
        default="shift",
        help="noise schedule of the flowmatch sampler (BFL used shift)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="bucket cap; default: a value that buckets the canvas exactly",
    )
    parser.add_argument(
        "--sample-prompts",
        type=int,
        default=3,
        help="sample prompts baked into the config (0 disables sampling)",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="set low_vram + layer_offloading in the config (slower, less VRAM)",
    )
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise SystemExit("error: --name may only contain letters, digits, dot, dash, underscore")
    if not 100 <= args.steps <= 20000:
        raise SystemExit("error: --steps must be between 100 and 20000")
    if not 0 < args.save_every <= args.steps:
        raise SystemExit("error: --save-every must be positive and at most --steps")
    if not 1e-6 <= args.lr <= 1e-2:
        raise SystemExit("error: --lr must be a small positive number (e.g. 1e-4)")
    if args.sample_prompts < 0:
        raise SystemExit("error: --sample-prompts cannot be negative")
    if args.resolution is not None and args.resolution <= 0:
        raise SystemExit("error: --resolution must be positive")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise SystemExit("error: --batch-size and --grad-accum must be at least 1")
    for flag in ("rank", "alpha", "conv_rank", "conv_alpha"):
        if getattr(args, flag) <= 0:
            raise SystemExit(f"error: --{flag.replace('_', '-')} must be positive")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def find_dataset(dataset_root: Path) -> tuple[Path, list[Path], dict[str, Any]]:
    """Return (aitk root, non-empty reference dirs, dataset_info) for a prepare_dataset export."""
    aitk = dataset_root / "aitk"
    if not (aitk / "target").is_dir():
        raise SystemExit(
            f"error: {aitk / 'target'} not found. Run scripts/lora/prepare_dataset.py first."
        )
    info: dict[str, Any] = {}
    info_path = dataset_root / "dataset_info.json"
    if info_path.is_file():
        try:
            loaded = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            info = loaded
    layout = str(info.get("layout") or "")
    if layout not in ("", "aitk", "both"):
        raise SystemExit(
            f"error: dataset_info.json says layout={layout!r}; re-export with --layout aitk"
        )
    references = [
        folder
        for folder in sorted(aitk.glob("reference_*"))
        if folder.is_dir() and any(folder.iterdir())
    ]
    if not references:
        raise SystemExit(f"error: no reference_* folders with images under {aitk}.")
    return aitk, references, info


def dataset_info(
    dataset_root: Path, aitk: Path, info: dict[str, Any]
) -> tuple[tuple[int, int], int]:
    """Canvas and target image count; the canvas comes from dataset_info.json when present."""
    canvas_info = info.get("canvas")
    canvas = DEFAULT_CANVAS
    if isinstance(canvas_info, dict):
        width, height = canvas_info.get("width"), canvas_info.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            canvas = (width, height)
    images = [
        path
        for path in (aitk / "target").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise SystemExit(f"error: no images under {aitk / 'target'}.")
    return canvas, len(images)


def bucket_for(
    width: int, height: int, resolution: int, divisibility: int = BUCKET_TOLERANCE
) -> tuple[int, int]:
    """Mirror of ai-toolkit's toolkit/buckets.py::get_bucket_for_image_size.

    With it we can pick a ``resolution`` whose bucket is exactly the exported canvas,
    so training never rescales or crops the pairs.
    """
    total_pixels = width * height
    max_pixels = resolution * resolution
    target_pixels = min(total_pixels, max_pixels)
    scaler = (target_pixels / total_pixels) ** 0.5
    w_raw = width * scaler / divisibility
    h_raw = height * scaler / divisibility
    candidates = [
        (math.floor(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.floor(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
    ]
    capped = [(w, h) for w, h in candidates if w > 0 and h > 0 and w * h <= max_pixels]
    if not capped:
        capped = [
            (
                max(divisibility, math.floor(w_raw) * divisibility),
                max(divisibility, math.floor(h_raw) * divisibility),
            )
        ]
    return min(capped, key=lambda wh: abs(wh[0] * wh[1] - target_pixels))


def resolve_resolution(canvas: tuple[int, int], requested: int | None) -> int:
    """A ``datasets.resolution`` whose bucket is exactly the canvas (no resize, no crop)."""
    width, height = canvas
    for value in (requested, max(width, height)):
        if value is not None and bucket_for(width, height, value) == canvas:
            return value
    smallest = math.ceil(math.sqrt(width * height) / BUCKET_TOLERANCE) * BUCKET_TOLERANCE
    if bucket_for(width, height, smallest) == canvas:
        return smallest
    raise SystemExit(
        f"error: the {width}x{height} canvas cannot be bucketed exactly at any resolution "
        f"(tried {requested}, {max(width, height)}, {smallest}); re-export the dataset on a "
        f"multiple-of-{BUCKET_TOLERANCE} canvas."
    )


def load_samples(dataset_root: Path) -> list[dict[str, Any]]:
    """Manifest rows that have ai-toolkit reference paths, in manifest order."""
    samples: list[dict[str, Any]] = []
    for row in load_jsonl(dataset_root / "manifest.jsonl"):
        aitk = row.get("aitk")
        if not isinstance(aitk, dict):
            continue
        refs = [str(ref) for ref in aitk.get("references") or []]
        if refs:
            row["aitk"] = {"target": str(aitk.get("target") or ""), "references": refs}
            samples.append(row)
    return samples


def assert_holdout_clean(samples: list[dict[str, Any]], val_path: Path) -> None:
    """Belt and braces: the bundle must never contain a held-out pair."""
    val_ids = {
        str(row.get("pair_id") or row.get("garment_id") or "") for row in load_jsonl(val_path)
    }
    val_ids.discard("")
    if not val_ids:
        print(f"warning: no validation ids read from {val_path}; the holdout guard is off")
        return
    leaked = [
        str(row.get("pair_id") or "")
        for row in samples
        if str(row.get("pair_id")) in val_ids
    ]
    if leaked:
        raise SystemExit(
            "error: the dataset contains held-out pairs ("
            + ", ".join(leaked[:5])
            + "); re-run prepare_dataset.py"
        )
    print(f"holdout guard: {len(val_ids)} validation pair ids, none in the dataset")


def pick_sample_prompts(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Up to ``count`` manifest rows; two-reference pairs first so ctrl_img_2 is exercised."""
    rank = {2: 0, 1: 1, 3: 2, 4: 3}
    ordered = sorted(samples, key=lambda row: rank.get(len(row["aitk"]["references"]), 9))
    return ordered[:count]


def yaml_quote(text: str) -> str:
    """A single-quoted YAML scalar; captions are plain ASCII but be safe about quotes."""
    return "'" + text.replace("'", "''") + "'"


def render_config(
    args: argparse.Namespace,
    base: dict[str, Any],
    canvas: tuple[int, int],
    resolution: int,
    reference_count: int,
    sample_prompts: list[dict[str, Any]],
) -> str:
    """The ai-toolkit config; __BUNDLE_DIR__ is replaced by run.sh on the instance."""
    width, height = canvas
    low_vram = "true" if args.low_vram else "false"
    control_block = "\n".join(
        f'            - "__BUNDLE_DIR__/dataset/reference_{index}"'
        for index in range(1, reference_count + 1)
    )
    sample_lines: list[str] = []
    for sample in sample_prompts:
        refs = [str(ref) for ref in sample["aitk"]["references"]]
        sample_lines.append(f"          - prompt: {yaml_quote(str(sample['caption']))}")
        sample_lines.append(f'            ctrl_img_1: "__BUNDLE_DIR__/dataset/{refs[0]}"')
        if len(refs) > 1:
            sample_lines.append(f'            ctrl_img_2: "__BUNDLE_DIR__/dataset/{refs[1]}"')
    sample_block = "\n".join(sample_lines) if sample_lines else "          []"
    sampling_off = (
        "" if sample_lines else "\n        disable_sampling: true  # no samples requested"
    )
    return f"""---
# Generated by scripts/lora/train_vast.py for ostris/ai-toolkit (docs/LORA.md).
# run.sh replaces __BUNDLE_DIR__ with the folder this bundle was unzipped into.
job: extension
config:
  # Folder and file name of the run on the instance.
  name: "{args.name}"
  process:
    - type: "diffusion_trainer"
      training_folder: "__BUNDLE_DIR__/output"
      device: "cuda:0"
      network:
        type: "lora"
        linear: {args.rank}
        linear_alpha: {args.alpha}
        conv: {args.conv_rank}
        conv_alpha: {args.conv_alpha}
      save:
        dtype: "bf16"
        save_every: {args.save_every}
        max_step_saves_to_keep: 8
      datasets:
        # target/ has the packshot + caption of each pair; reference_N/ the input photos
        # of the same pair (same file stem; missing files are skipped by the loader).
        - folder_path: "__BUNDLE_DIR__/dataset/target"
          control_path:
{control_block}
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          resolution: {resolution}
          bucket_tolerance: {BUCKET_TOLERANCE}
      train:
        batch_size: {args.batch_size}
        gradient_accumulation_steps: {args.grad_accum}
        steps: {args.steps}
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: {args.lr:g}
        timestep_type: "{args.timestep_type}"
        content_or_style: "balanced"
        dtype: "bf16"{sampling_off}
      model:
        name_or_path: "{base['model']}"
        arch: "{base['arch']}"
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
        low_vram: {low_vram}
        layer_offloading: {low_vram}
      sample:
        sampler: "flowmatch"
        sample_every: {args.save_every}
        width: {width}
        height: {height}
        samples:
{sample_block}
        neg: ""
        seed: 42
        walk_seed: false
        guidance_scale: 4
        sample_steps: 28
meta:
  name: "{args.name}"
  version: "1.0"
"""


#: Bootstrap for the rented instance; __NAME__ gets the run name baked in here.
RUN_SH = """#!/usr/bin/env bash
# Sartiq packshot LoRA: clone ai-toolkit, install the environment, train. See README.txt.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE_DIR

AITK_REPO="${AITK_REPO:-https://github.com/ostris/ai-toolkit.git}"
AITK_DIR="${BUNDLE_DIR}/ai-toolkit"
CONFIG="${BUNDLE_DIR}/configs/run_config.yaml"

echo "bundle: ${BUNDLE_DIR}"

# 1. bake this folder's absolute paths into the config
python3 - <<'PY'
import os, pathlib
d = pathlib.Path(os.environ["BUNDLE_DIR"])
src = d / "configs" / "aitk_sartiq_packshot.yaml"
dst = d / "configs" / "run_config.yaml"
dst.write_text(
    src.read_text(encoding="utf-8").replace("__BUNDLE_DIR__", d.as_posix()),
    encoding="utf-8",
)
print(f"config: {dst}")
PY

# 2. ostris/ai-toolkit (main; set AITK_REF to a tag or commit to pin the version)
if [ ! -d "${AITK_DIR}/.git" ]; then
  git clone --depth 1 "${AITK_REPO}" "${AITK_DIR}"
fi
if [ -n "${AITK_REF:-}" ]; then
  git -C "${AITK_DIR}" fetch origin "${AITK_REF}"
  git -C "${AITK_DIR}" checkout FETCH_HEAD
fi

# 3. environment (the experimental manager creates .venv with the right torch build)
cd "${AITK_DIR}"
if [ ! -x .venv/bin/python ] || [ "${AITK_FORCE_INSTALL:-0}" = "1" ]; then
  python3 -m manager install
fi
if [ ! -x .venv/bin/python ]; then
  echo "error: .venv/bin/python is missing after install; see README.txt (manual install)" >&2
  exit 1
fi

# 4. train
mkdir -p "${BUNDLE_DIR}/output"
echo "checkpoints and samples -> ${BUNDLE_DIR}/output/__NAME__"
.venv/bin/python run.py "${CONFIG}"
"""


def render_readme(
    args: argparse.Namespace, base: dict[str, Any], canvas: tuple[int, int], pairs: int
) -> str:
    width, height = canvas
    base_desc = f"{args.base} ({base['model']})"
    return f"""SARTIQ PACKSHOT LORA on vast.ai - base {base_desc}
{'=' * (39 + len(base_desc))}

Fine-tunes the packshot LoRA from the {pairs} example pairs (canvas {width}x{height}, letterboxed
on light grey) with ostris/ai-toolkit, so the pipeline can switch to the fine-tuned model
instead of the hosted baseline. Nothing secret is inside: no keys, no tokens.

Contents
  dataset/   ai-toolkit layout: target/ holds the packshot + caption .txt of each pair;
             reference_1..N/ hold the input photos (same file stem as the target; a pair
             with fewer photos is simply missing files in the higher folders).
  configs/   aitk_sartiq_packshot.yaml - the training config, with __BUNDLE_DIR__
             placeholders that run.sh replaces with this folder's real path.
  run.sh     clone + install ai-toolkit, then train. Read it: it is short.
  README.txt this file.

Hardware (https://docs.bfl.ml/flux_2/flux2_klein_training)
  {args.base}: >= {base['vram']}GB VRAM, >= {base['ram']}GB RAM, >= {base['disk']}GB disk.
  The base weights (~35GB for 9B), the text encoder, cached latents and the checkpoints
  live on disk, so do not rent less than {base['disk']}GB.
  Licence: {base['licence']}.

1. Rent
   vast.ai -> any recent NVIDIA on-demand box matching the table above; a plain PyTorch
   image with python3 and git is enough, the toolkit installs its own environment.
   Copy the ssh command from the instance page ("Connect").

2. Upload and unpack (from your machine)
   scp -P <ssh-port> bundle.zip root@<host>:/workspace/bundle.zip
   ssh -p <ssh-port> root@<host>
   cd /workspace && unzip -q bundle.zip       # or: python3 -m zipfile -e bundle.zip .

3. Train
   {base['gated']}
   bash run.sh
   First it downloads the weights and installs ai-toolkit (several minutes), then trains
   {args.steps} steps. A sample grid is written every {args.save_every} steps to
   output/{args.name}/samples/ - the first one is the base model, before any step.

4. Watch the samples, pick a checkpoint
   The LoRA should learn one thing: input photo(s) -> flat-lay packshot on light grey,
   keeping colour, fabric, print and proportions. Overfitting shows up as prints or
   details drifting from the reference photos; quality on this dataset usually peaks
   between 1000 and 1750 steps. Ctrl+C is safe (wait for a running save to finish);
   running run.sh again resumes from the last checkpoint.
   Checkpoints: output/{args.name}/{args.name}.safetensors is the latest, and
   output/{args.name}/{args.name}_000001000.safetensors keep every {args.save_every} steps.

5. Download, then stop the instance
   scp -P <ssh-port> root@<host>:/workspace/output/{args.name}/{args.name}_000001250.safetensors .
   Stop or destroy the instance in the vast.ai dashboard - billing keeps running while
   it is up.

6. Register it in the repo and use it
   python scripts/lora/register_lora.py --weights {args.name}_000001250.safetensors
   python -m zir_image_enhancer.cli --model sartiq-lora-klein --manifest ... --out ... --team ...
   register_lora.py copies the weights into local/lora/weights and uploads them to fal
   storage, then writes local/lora/registry.json; the fal.ai training alternative and
   all the details are in docs/LORA.md.

Troubleshooting
  - "python3 -m manager install" failed: install manually, inside ai-toolkit/:
      python3 -m venv .venv
      .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
      .venv/bin/pip install -r requirements.txt
    (adjust cu124 to the image's CUDA; .venv is what run.sh looks for)
  - CUDA out of memory: in configs/aitk_sartiq_packshot.yaml set low_vram: true and
    layer_offloading: true, or regenerate the bundle with --base 4b or --low-vram.
  - HF 401/403 on the 9B: accept the licence on the model page, then export HF_TOKEN.
  - The samples never improve: check that run_config.yaml paths point at dataset/ (run.sh
    prints the generated config), and that each target has a caption .txt next to it.
"""


def build_bundle(
    out_dir: Path,
    dataset_root: Path,
    config_text: str,
    run_sh: str,
    readme: str,
) -> tuple[Path, dict[str, int]]:
    """Zip README.txt, run.sh, configs/ and dataset/ (root-level names) into bundle.zip."""
    archive = out_dir / "bundle.zip"
    stats = {"dataset_files": 0}
    run_info = zipfile.ZipInfo("run.sh")
    run_info.external_attr = 0o755 << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme)
        zf.writestr(run_info, run_sh)
        zf.writestr("configs/aitk_sartiq_packshot.yaml", config_text)
        for source, arcname in (
            (dataset_root / "manifest.jsonl", "dataset/manifest.jsonl"),
            (dataset_root / "dataset_info.json", "dataset/dataset_info.json"),
        ):
            if source.is_file():
                zf.write(source, arcname)
        for path in sorted((dataset_root / "aitk").rglob("*")):
            if path.is_file():
                zf.write(path, "dataset/" + path.relative_to(dataset_root / "aitk").as_posix())
                stats["dataset_files"] += 1
    stats["bytes"] = archive.stat().st_size
    return archive, stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate(args)
    base = BASES[args.base]

    aitk, references, info = find_dataset(args.dataset)
    canvas, pairs = dataset_info(args.dataset, aitk, info)
    resolution = resolve_resolution(canvas, args.resolution)
    if args.resolution is not None and resolution != args.resolution:
        print(
            f"warning: --resolution {args.resolution} would rescale the pairs; "
            f"using {resolution}"
        )
    width, height = canvas
    bucket = bucket_for(width, height, resolution)
    if bucket != canvas:
        raise SystemExit(
            f"error: internal check failed: resolution {resolution} buckets to {bucket}"
        )

    samples = load_samples(args.dataset)
    assert_holdout_clean(samples, args.val)
    picked = pick_sample_prompts(samples, args.sample_prompts)

    config_text = render_config(args, base, canvas, resolution, len(references), picked)
    run_sh = RUN_SH.replace("__NAME__", args.name)
    readme = render_readme(args, base, canvas, pairs)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aitk_sartiq_packshot.yaml").write_text(config_text, encoding="utf-8")
    (out_dir / "run.sh").write_text(run_sh, encoding="utf-8")
    (out_dir / "run.sh").chmod(0o755)
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")
    archive, stats = build_bundle(out_dir, args.dataset, config_text, run_sh, readme)

    ref_histogram: dict[int, int] = {}
    for sample in samples:
        count = len(sample["aitk"]["references"])
        ref_histogram[count] = ref_histogram.get(count, 0) + 1
    print(f"dataset: {args.dataset} ({pairs} pairs, {len(references)} reference folders, "
          f"{width}x{height})")
    print(f"references per pair: {dict(sorted(ref_histogram.items()))}")
    print(f"bucket: resolution {resolution} -> {bucket[0]}x{bucket[1]} (exact, no resize/crop)")
    print(f"recipe: {args.base} {base['model']}, rank {args.rank}/{args.alpha} "
          f"conv {args.conv_rank}/{args.conv_alpha}, {args.steps} steps, lr {args.lr:g}, "
          f"save every {args.save_every}, timestep {args.timestep_type}")
    print(f"samples: {len(picked)} prompts baked into the config"
          + (" (sampling disabled)" if not picked else ""))
    print(f"files: {out_dir}/aitk_sartiq_packshot.yaml, run.sh, README.txt")
    print(f"bundle: {archive} ({stats['bytes'] / 1024 / 1024:.1f} MB, "
          f"{stats['dataset_files']} dataset files)")
    print(f"generated: {datetime.now(UTC).isoformat(timespec='seconds')} (no upload, no spend)")
    print("next: upload bundle.zip to a vast.ai instance matching README.txt and run: bash run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

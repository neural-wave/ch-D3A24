#!/usr/bin/env python3
"""Register existing packshot LoRA weights as the active adapter (docs/LORA.md).

``scripts/lora/train_fal.py`` registers its own result; this is for weights trained somewhere
else (the vast.ai path, a teammate's machine, HuggingFace). The pipeline only reads
``local/lora/registry.json`` (or ``ZIR_LORA_PATH``), so one command is enough for every tool.

    python scripts/lora/register_lora.py --show
    python scripts/lora/register_lora.py --weights local/lora/weights/sartiq.safetensors
    python scripts/lora/register_lora.py --hub owner/repo --scale 0.8
    python scripts/lora/register_lora.py --url https://cdn.example.com/sartiq.safetensors
    python scripts/lora/register_lora.py --clear

A local ``--weights`` file is copied into ``local/lora/weights`` and uploaded to fal storage
(free, needs FAL_KEY) so the hosted endpoint can read it; with ``--no-upload`` the local path
itself is registered instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from zir_image_enhancer.generation.lora import load_binding
from zir_image_enhancer.paths import ProjectPaths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--weights", type=Path, help="local .safetensors to copy and upload")
    source.add_argument("--url", help="http(s) url of the weights")
    source.add_argument("--hub", help="HuggingFace repo id, owner/repo")
    source.add_argument("--show", action="store_true", help="print the active binding and exit")
    source.add_argument("--clear", action="store_true", help="remove the registry and exit")
    parser.add_argument("--name", default="", help="run name (defaults to the file name)")
    parser.add_argument("--scale", type=float, default=1.0, help="adapter scale, 0.0 to 4.0")
    parser.add_argument(
        "--no-upload", action="store_true", help="register the local path instead of uploading"
    )
    return parser.parse_args(argv)


def resolve_path(args: argparse.Namespace, paths: ProjectPaths) -> tuple[str, str]:
    """Return (path-to-register, source-kind) for the requested weights."""
    if args.weights is not None:
        weights = args.weights.resolve()
        if not weights.is_file():
            raise SystemExit(f"error: weights file not found: {weights}")
        if weights.suffix != ".safetensors":
            print(f"warning: {weights.name} does not look like a .safetensors file")
        name = args.name or weights.stem
        stable = paths.lora_root / "weights" / f"{name}.safetensors"
        stable.parent.mkdir(parents=True, exist_ok=True)
        if weights != stable:
            shutil.copyfile(weights, stable)
        if args.no_upload:
            return str(stable), "local"
        import fal_client  # imported late so --show/--clear never need a key

        print(f"uploading {stable} to fal storage ...")
        return fal_client.upload_file(str(stable)), "upload"
    if args.url:
        return args.url, "url"
    if args.hub:
        return args.hub, "hub"
    raise SystemExit("error: pass --weights, --url or --hub (or --show / --clear)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.from_env()

    if args.show:
        binding = load_binding(paths)
        if binding is None:
            print(f"no adapter registered ({paths.lora_registry} does not exist)")
            return 1
        print(json.dumps({"path": binding.path, "scale": binding.scale, "name": binding.name}, indent=2))
        print(f"registry: {paths.lora_registry}")
        return 0
    if args.clear:
        if paths.lora_registry.is_file():
            paths.lora_registry.unlink()
            print(f"removed {paths.lora_registry}")
        else:
            print(f"nothing to remove ({paths.lora_registry} does not exist)")
        return 0

    if not 0.0 <= args.scale <= 4.0:
        raise SystemExit("error: --scale must be between 0.0 and 4.0 (fal's limit)")

    path, source = resolve_path(args, paths)
    name = args.name or Path(path).stem if source == "local" else (
        args.name or Path(path).stem or path
    )
    binding = {
        "path": path,
        "scale": args.scale,
        "name": name,
        "source": source,
        "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    paths.lora_registry.parent.mkdir(parents=True, exist_ok=True)
    paths.lora_registry.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    print(f"registered: {paths.lora_registry}")
    print(json.dumps(binding, indent=2))
    print("use it with:  ZIR_MODEL_KEY=sartiq-lora-klein  (or zir-run --model sartiq-lora-klein)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

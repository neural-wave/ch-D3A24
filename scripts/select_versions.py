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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms

from zir_image_enhancer.dataset import GarmentItem, load_manifest, split_manifest
from zir_image_enhancer.generation.fal_backend import CostLedger
from zir_image_enhancer.paths import ProjectPaths
from zir_image_enhancer.reconstruction.gate import GateResult, check
from zir_image_enhancer.reconstruction.photos import roles_from_filenames
from zir_image_enhancer.reconstruction.spec import extract_spec
from zir_image_enhancer.reconstruction.vision import VisionClient

_ALEX_FILE = "torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
_ALEX_BLOCKS = {1, 4, 7, 9, 11}  # the five conv blocks LPIPS reads


@dataclass(frozen=True)
class Candidate:
    version: str
    image: Path


class EmbeddingExperts:
    """CLIP and AlexNet embeddings, L2-normalised, with the training targets cached on disk."""

    def __init__(self, paths: ProjectPaths) -> None:
        self._paths = paths
        self._clip, _, self._clip_tf = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._clip.eval()
        self._alex = _load_alexnet(paths)
        self._alex_tf = transforms.Compose(
            [
                transforms.Resize((288, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def embed(self, images: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        clip_parts, alex_parts = [], []
        for start in range(0, len(images), 32):
            batch = [
                ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                for p in images[start : start + 32]
            ]
            with torch.no_grad():
                feats = self._clip.encode_image(torch.stack([self._clip_tf(im) for im in batch]))
                clip_parts.append(torch.nn.functional.normalize(feats, dim=-1).numpy())
                alex_parts.append(
                    self._alex_features(torch.stack([self._alex_tf(im) for im in batch]))
                )
        return np.concatenate(clip_parts), np.concatenate(alex_parts)

    def house_style(self) -> tuple[np.ndarray, np.ndarray]:
        """Embeddings of the training studio photos (never a scored or submitted garment).

        With our holdout split present, only its training side is used, so holdout scores
        stay honest; without it (a fresh clone, the container), every training pair is.
        """
        source = self._paths.holdout_train
        if not source.exists():
            source = self._paths.data_root / "train" / "pairs.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        targets = [self._paths.data_root / row["target"] for row in rows]
        key = hashlib.sha256("|".join(str(t) for t in targets).encode()).hexdigest()[:16]
        cache = self._paths.repo_root / "local" / f"house_style_{key}.npz"
        if cache.exists():
            data = np.load(cache)
            return data["clip"], data["alex"]
        clip, alex = self.embed(targets)
        np.savez(cache, clip=clip, alex=alex)
        return clip, alex

    def _alex_features(self, x: torch.Tensor) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for index, layer in enumerate(self._alex.features):
                x = layer(x)
                if index in _ALEX_BLOCKS:
                    # Channel means and spreads per block: a texture signature, like LPIPS sees.
                    feat = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)
                    parts.append(torch.nn.functional.normalize(feat, dim=-1))
        return torch.nn.functional.normalize(torch.cat(parts, dim=1), dim=-1).numpy()


def _load_alexnet(paths: ProjectPaths) -> torchvision.models.AlexNet:
    """The AlexNet the official scorer loads for LPIPS, from the first cache that has it."""
    model = torchvision.models.alexnet(weights=None)
    for root in (
        paths.starter_root / "scorer" / ".models",
        paths.repo_root / "tools" / "scorer" / ".models",
        Path(torch.hub.get_dir()).parent,
    ):
        weights = root / _ALEX_FILE
        if weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            return model.eval()
    # Not cached anywhere yet: torchvision downloads the same file from pytorch.org once.
    return torchvision.models.alexnet(weights="DEFAULT").eval()


class VersionSelector:
    def __init__(self, experts: EmbeddingExperts, neighbours: int = 10) -> None:
        self._experts = experts
        self._neighbours = neighbours
        self._style_clip, self._style_alex = experts.house_style()

    def rank(
        self, items: list[GarmentItem], versions: dict[str, Path]
    ) -> dict[str, list[Candidate]]:
        """Candidates of every garment, best first."""
        cands = [
            (item, Candidate(name, folder / item.item_id / f"{item.view}.png"))
            for item in items
            for name, folder in versions.items()
            # A version can miss a garment (fal's content filter refused one); skip it.
            if (folder / item.item_id / f"{item.view}.png").exists()
        ]
        clip, alex = self._experts.embed([c.image for _, c in cands])
        in_clip, in_alex = self._experts.embed([item.inputs[0] for item in items])
        first_photo = {item.item_id: i for i, item in enumerate(items)}
        value = np.zeros(len(cands))
        for emb, style, inputs in (
            (clip, self._style_clip, in_clip),
            (alex, self._style_alex, in_alex),
        ):
            prior = np.sort(emb @ style.T, axis=1)[:, -self._neighbours :].mean(axis=1)
            identity = np.array(
                [emb[k] @ inputs[first_photo[item.item_id]] for k, (item, _) in enumerate(cands)]
            )
            # Each signal is standardised over the whole pool so neither expert dominates.
            value += _standardise(prior) + _standardise(identity)
        ranked: dict[str, list[Candidate]] = {}
        for item in items:
            mine = [k for k, (other, _) in enumerate(cands) if other.item_id == item.item_id]
            ranked[item.item_id] = [cands[k][1] for k in sorted(mine, key=lambda k: -value[k])]
        return ranked


def _standardise(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-9)


def _weight(gate: GateResult) -> int:
    return len(gate.problems) + len(gate.severe)


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
    ranked = VersionSelector(EmbeddingExperts(paths)).rank(items, versions)

    vision = VisionClient(
        CostLedger(paths.fal_ledger), 1e9, paths.repo_root / "local" / "reconstruction_cache"
    )
    out_dir = paths.runs_root / args.run
    log = []
    for item in items:
        order = ranked[item.item_id]
        chosen, checked = order[0], []
        if args.gate_top > 0:
            spec = extract_spec(roles_from_filenames(item.inputs), vision)
            best: tuple[int, Candidate] | None = None
            for cand in order[: args.gate_top]:
                gate = check(spec.front, cand.image, spec, vision)
                checked.append({"version": cand.version, "problems": list(gate.problems)})
                if not gate.problems:
                    best = (0, cand)
                    break
                if best is None or _weight(gate) < best[0]:
                    best = (_weight(gate), cand)
            chosen = best[1] if best else order[0]
        target = out_dir / "cand_0" / item.item_id / f"{item.view}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chosen.image, target)
        log.append(
            {
                "garment": item.item_id,
                "ranking": [c.version for c in order],
                "chosen": chosen.version,
                "gate": checked,
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

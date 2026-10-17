"""The two embedding experts of the v4 selector: CLIP and AlexNet, local and on CPU.

CLIP sees what the garment is and the overall style; AlexNet is the LPIPS backbone, so it sees
texture and local structure the way the scorer does. Both need the ``select`` extra (torch);
the rest of the package does not import this module unless v4 runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open_clip
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms

from zir_image_enhancer.paths import ProjectPaths

_ALEX_FILE = "torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
_ALEX_BLOCKS = {1, 4, 7, 9, 11}  # the five conv blocks LPIPS reads


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

    def embed(self, images: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
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
        if not source.exists():
            raise FileNotFoundError(
                f"v4 learns the house style from the training studio photos, and {source} "
                "is missing. Put the challenge dataset under data/ (see the README)."
            )
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        targets = [self._paths.data_root / row["target"] for row in rows]
        key = hashlib.sha256("|".join(str(t) for t in targets).encode()).hexdigest()[:16]
        cache = self._paths.repo_root / "local" / f"house_style_{key}.npz"
        if cache.exists():
            data = np.load(cache)
            return data["clip"], data["alex"]
        clip, alex = self.embed(targets)
        cache.parent.mkdir(parents=True, exist_ok=True)
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

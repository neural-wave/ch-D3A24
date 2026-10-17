"""Fallback photo choice when the vision triage fails (local CLIP, no cost).

Phone sets often include price tags, care labels, a garment still in its bag and close-ups.
Those photos give the model nothing for the front packshot, so the first whole view is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps

# One text prompt per kind of photo. CLIP compares each photo with all of them and the
# highest probability wins. Wording follows CLIP's usual "a photo of ..." style.
_LABELS: dict[str, str] = {
    "front": "a photo of a whole garment seen from the front",
    "back": "a photo of the back of a garment",
    "tag": "a close-up photo of a clothing label or a price tag with a barcode",
    "packaging": "a photo of a folded garment inside a plastic bag",
    "detail": "a close-up photo of fabric texture, a seam or a button",
}


@dataclass(frozen=True)
class PhotoVerdict:
    path: Path
    label: str
    front_probability: float


@dataclass(frozen=True)
class FilterDecision:
    kept: tuple[Path, ...]
    verdicts: tuple[PhotoVerdict, ...]


class PhotoFilter:
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        # The same CLIP the official scorer uses; it is already cached on most machines.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        self._model = model.eval()
        self._preprocess = preprocess
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            text = self._model.encode_text(tokenizer(list(_LABELS.values())).to(device))
        self._text = text / text.norm(dim=-1, keepdim=True)

    def classify(self, path: Path) -> PhotoVerdict:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        batch = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        probabilities = (100.0 * features @ self._text.T).softmax(dim=-1)[0]
        labels = list(_LABELS)
        best = int(probabilities.argmax())
        return PhotoVerdict(path, labels[best], float(probabilities[labels.index("front")]))

    def first_usable(self, photos: Sequence[Path]) -> FilterDecision:
        """The first photo, in the given order, that shows a whole garment.

        CLIP cannot reliably tell the front of a hanging garment from its back (on the
        test set it picked the back for 5 of 15 garments), but it does spot tags, bags and
        close-ups. So the photo order decides front versus back, and CLIP only skips the
        photos that are clearly not a full view.
        """
        if len(photos) <= 1:
            return FilterDecision(tuple(photos), ())
        verdicts: list[PhotoVerdict] = []
        for path in photos:
            verdict = self.classify(path)
            verdicts.append(verdict)
            if verdict.label in ("front", "back"):
                return FilterDecision((path,), tuple(verdicts))
        best = max(verdicts, key=lambda v: v.front_probability)
        return FilterDecision((best.path,), tuple(verdicts))

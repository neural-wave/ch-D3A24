"""Measured garment colours: the main colours of the front photo, as hex codes and shares.

Used in two places: the generation prompt (so the model does not have to guess the shade) and
the quality gate (so a colour drift can be caught). Measured on the garment pixels only, with
the same segmenter the scorer uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove
from skimage import color


@dataclass(frozen=True)
class MeasuredColour:
    hex: str
    share: float  # fraction of the garment pixels


class ColourMeter:
    def __init__(self, clusters: int = 3, min_share: float = 0.08) -> None:
        self._clusters = clusters
        self._min_share = min_share
        self._session = new_session("u2net")

    def measure(
        self, photo: Path, box: tuple[int, int, int, int] | None = None
    ) -> tuple[MeasuredColour, ...]:
        image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
        if box is not None:
            # The segmenter alone kept doors and floors on cluttered photos (a lilac romper
            # measured brown); cropping to the garment box from the spec first avoids that.
            ymin, xmin, ymax, xmax = box
            image = image.crop(
                (
                    int(xmin / 1000 * image.width),
                    int(ymin / 1000 * image.height),
                    int(xmax / 1000 * image.width),
                    int(ymax / 1000 * image.height),
                )
            )
        image.thumbnail((768, 768))  # colours do not need full resolution; this keeps it fast
        rgb = np.asarray(image, dtype=np.float64) / 255.0
        mask = np.asarray(remove(image, session=self._session, only_mask=True)) > 127
        if mask.mean() < 0.02:
            mask = np.ones(mask.shape, dtype=bool)  # segmentation failed: use the whole photo
        lab = color.rgb2lab(rgb)[mask]
        centres, shares = _kmeans(lab, self._clusters)
        found = [
            MeasuredColour(_to_hex(centre), float(share))
            for centre, share in zip(centres, shares, strict=True)
            if share >= self._min_share
        ]
        return tuple(sorted(found, key=lambda c: c.share, reverse=True))


def _kmeans(points: np.ndarray, k: int, iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    # Plain k-means in Lab, where distances match how different two colours look.
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), size=min(len(points), 20000), replace=False)]
    centres = sample[rng.choice(len(sample), size=k, replace=False)]
    for _ in range(iterations):
        labels = np.argmin(((sample[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(labels == j):
                centres[j] = sample[labels == j].mean(axis=0)
    shares = np.bincount(labels, minlength=k) / len(labels)
    return centres, shares


def _to_hex(lab: np.ndarray) -> str:
    rgb = np.clip(color.lab2rgb(lab.reshape(1, 1, 3)).reshape(3), 0, 1)
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)

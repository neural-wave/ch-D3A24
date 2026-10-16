"""White balance from a grey card (or white paper) placed in a fixed corner of the photo.

Capture protocol: the card sits in one corner and fills a square of at least ~15% of the
shorter side. The colour of the card is measured there, and the whole photo is scaled so
the card comes out neutral. Brightness is kept, only the colour cast changes.

Why it is worth it even though other input changes hurt the holdout score: the scorer
compares colour against a studio photo of the real garment (40% of the score), and a warm
or blue cast in the capture is the one thing the edit model cannot know about. A card
measured in the same light is ground truth for the cast.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


@dataclass(frozen=True)
class CardSettings:
    corner: str = "bottom-right"
    # Side of the corner square that holds the card, as a fraction of the shorter side.
    size: float = 0.15
    # Distance from the image edge, so a dark border or vignetting is not measured.
    inset: float = 0.02
    # Paint the card out with the surrounding backdrop colour, so the edit model does not
    # keep it as a prop in the packshot.
    cover: bool = True
    # A card is uniform; if the measured square varies more than this (sRGB, 0-1), the card
    # is probably not there and the photo is left untouched.
    max_std: float = 0.08

    def key(self) -> str:
        return f"{self.corner}-s{self.size}-i{self.inset}-c{self.cover}"


@dataclass(frozen=True)
class BalanceResult:
    image: Image.Image
    gains: tuple[float, float, float]
    card_srgb: tuple[float, float, float]
    applied: bool


class GreyCardBalancer:
    def __init__(self, settings: CardSettings | None = None, cache_dir: Path | None = None) -> None:
        self._settings = settings or CardSettings()
        if self._settings.corner not in CORNERS:
            raise ValueError(f"corner must be one of {CORNERS}")
        self._cache_dir = cache_dir

    def card_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        side = int(round(self._settings.size * min(width, height)))
        inset = int(round(self._settings.inset * min(width, height)))
        left = inset if "left" in self._settings.corner else width - inset - side
        top = inset if "top" in self._settings.corner else height - inset - side
        return left, top, left + side, top + side

    def balance(self, photo: Image.Image) -> BalanceResult:
        rgb = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"), dtype=np.float64) / 255.0
        height, width = rgb.shape[:2]
        left, top, right, bottom = self.card_box(width, height)
        # The centre half of the square only, so a slightly misplaced card still works.
        quarter_w, quarter_h = (right - left) // 4, (bottom - top) // 4
        patch = rgb[top + quarter_h : bottom - quarter_h, left + quarter_w : right - quarter_w]
        pixels = patch.reshape(-1, 3)
        # Clipped pixels have lost their colour information, so they cannot be measured.
        pixels = pixels[(pixels.max(axis=1) < 0.98) & (pixels.min(axis=1) > 0.02)]
        card = np.median(pixels, axis=0) if len(pixels) else np.zeros(3)
        spread = float(np.max(np.std(patch, axis=(0, 1))))
        if len(pixels) < 50 or spread > self._settings.max_std:
            untouched = Image.fromarray((rgb * 255).round().astype(np.uint8))
            return BalanceResult(untouched, (1.0, 1.0, 1.0), _triple(card), False)

        # Gains are computed in linear light, where white balance is a per-channel scale
        # (as in the camera). The target keeps the card's own brightness.
        card_linear = srgb_to_linear(card)
        gains = card_linear.mean() / np.maximum(card_linear, 1e-6)
        balanced = linear_to_srgb(srgb_to_linear(rgb) * gains)
        if self._settings.cover:
            balanced = self._cover_card(balanced, (left, top, right, bottom))
        image = Image.fromarray((balanced * 255).round().astype(np.uint8))
        return BalanceResult(image, _triple(gains), _triple(card), True)

    def balance_file(self, path: Path) -> Path:
        """Balanced copy of a file, cached on disk by content and settings."""
        if self._cache_dir is None:
            raise RuntimeError("balance_file needs a cache_dir")
        digest = hashlib.sha256(path.read_bytes())
        digest.update(self._settings.key().encode())
        cached = self._cache_dir / f"wb-{digest.hexdigest()[:24]}.png"
        if not cached.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            result = self.balance(Image.open(path))
            if not result.applied:
                print(f"  ! no usable card in {path.name}; photo left as it is")
            result.image.save(cached)
        return cached

    @staticmethod
    def _cover_card(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = box
        height, width = rgb.shape[:2]
        ring = max(4, (right - left) // 8)
        outer = rgb[
            max(0, top - ring) : min(height, bottom + ring),
            max(0, left - ring) : min(width, right + ring),
        ]
        # Median of the band around the card = the backdrop colour next to it.
        band = np.concatenate(
            [
                outer[:ring].reshape(-1, 3),
                outer[-ring:].reshape(-1, 3),
                outer[:, :ring].reshape(-1, 3),
                outer[:, -ring:].reshape(-1, 3),
            ]
        )
        covered = rgb.copy()
        covered[top:bottom, left:right] = np.median(band, axis=0)
        return covered

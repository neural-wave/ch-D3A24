"""Cut the garment out of a photo and place it, framed, on a plain background.

Used in three places, always without any generative model:

* before generation, so the edit model only sees the garment (no floor, door or clutter);
* as a fully deterministic pipeline (the cut-out itself is the packshot);
* after generation, to give every output the same clean background and framing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove

DEFAULT_SEGMENTER = "isnet-general-use"


@dataclass(frozen=True)
class CutoutSettings:
    background: tuple[int, int, int] = (255, 255, 255)
    # Margin around the garment as a fraction of its longest side. The scorer crops to the
    # mask anyway; the margin only keeps the garment away from the border (a cut garment
    # is a real error).
    margin: float = 0.08
    # Small phone photos are upscaled (plain Lanczos, no AI) so the edit model gets more
    # pixels on the garment. 0 disables it.
    min_long_side: int = 1024
    # Below or above these mask fractions the segmentation probably failed; the photo is
    # then used as it is instead of a broken cut-out.
    min_mask_fraction: float = 0.02
    max_mask_fraction: float = 0.95
    # True: only crop around the garment and keep the real background. Removing the
    # background before generation cost 3.9 points on the holdout; this tells whether the
    # loss came from the removal itself or from the framing.
    keep_background: bool = False

    def key(self) -> str:
        return f"bg{self.background}-m{self.margin}-ls{self.min_long_side}-kb{self.keep_background}"


@dataclass(frozen=True)
class CutoutResult:
    image: Image.Image
    mask_fraction: float
    used_fallback: bool


class GarmentCutter:
    def __init__(
        self,
        settings: CutoutSettings | None = None,
        segmenter: str = DEFAULT_SEGMENTER,
        cache_dir: Path | None = None,
    ) -> None:
        self._settings = settings or CutoutSettings()
        self._segmenter = segmenter
        self._cache_dir = cache_dir
        # The ONNX session is the slow part to create, so it is built once and reused.
        self._session = new_session(segmenter)

    @property
    def settings(self) -> CutoutSettings:
        return self._settings

    def cut(self, photo: Image.Image) -> CutoutResult:
        rgb = ImageOps.exif_transpose(photo).convert("RGB")
        soft = (
            np.asarray(remove(rgb, session=self._session, only_mask=True), dtype=np.float32) / 255.0
        )
        hard = soft > 0.5
        fraction = float(hard.mean())
        if not (self._settings.min_mask_fraction <= fraction <= self._settings.max_mask_fraction):
            return CutoutResult(self._upscale(rgb), fraction, used_fallback=True)

        if self._settings.keep_background:
            return CutoutResult(self._upscale(self._crop(rgb, hard)), fraction, False)

        pixels = np.asarray(rgb, dtype=np.float32)
        background = np.array(self._settings.background, dtype=np.float32)
        alpha = soft[..., None]
        composite = pixels * alpha + background * (1.0 - alpha)
        framed = self._frame(composite.round().astype(np.uint8), hard)
        return CutoutResult(self._upscale(framed), fraction, used_fallback=False)

    def cut_file(self, path: Path) -> Path:
        """Cut-out of a file, cached on disk by content, segmenter and settings."""
        if self._cache_dir is None:
            raise RuntimeError("cut_file needs a cache_dir")
        digest = hashlib.sha256(path.read_bytes())
        digest.update(f"{self._segmenter}|{self._settings.key()}".encode())
        cached = self._cache_dir / f"{digest.hexdigest()[:24]}.png"
        if not cached.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self.cut(Image.open(path)).image.save(cached)
        return cached

    def _frame(self, pixels: np.ndarray, mask: np.ndarray) -> Image.Image:
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        garment = Image.fromarray(pixels[top:bottom, left:right])
        pad = int(round(self._settings.margin * max(garment.size)))
        canvas = Image.new(
            "RGB",
            (garment.width + 2 * pad, garment.height + 2 * pad),
            self._settings.background,
        )
        canvas.paste(garment, (pad, pad))
        return canvas

    def _crop(self, rgb: Image.Image, mask: np.ndarray) -> Image.Image:
        """Crop the original photo around the garment, margin included, inside the image."""
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        pad = int(round(self._settings.margin * max(bottom - top, right - left)))
        box = (
            max(0, left - pad),
            max(0, top - pad),
            min(rgb.width, right + pad),
            min(rgb.height, bottom + pad),
        )
        return rgb.crop(box)

    def _upscale(self, image: Image.Image) -> Image.Image:
        target = self._settings.min_long_side
        longest = max(image.size)
        if target <= 0 or longest >= target:
            return image
        scale = target / longest
        size = (round(image.width * scale), round(image.height * scale))
        return image.resize(size, Image.Resampling.LANCZOS)

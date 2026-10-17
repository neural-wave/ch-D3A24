"""Classic denoising of input photos before generation (OpenCV non-local means, CPU).

The question it answers: does a cleaner input make the edit model invent less? The filter
is mild on purpose, because LPIPS rewards real fabric texture and a strong denoise would
erase it together with the noise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class DenoiseSettings:
    luma_strength: float = 5.0
    colour_strength: float = 5.0
    template_window: int = 7
    search_window: int = 21

    def key(self) -> str:
        return f"h{self.luma_strength}-hc{self.colour_strength}"


class Denoiser:
    def __init__(
        self, settings: DenoiseSettings | None = None, cache_dir: Path | None = None
    ) -> None:
        self._settings = settings or DenoiseSettings()
        self._cache_dir = cache_dir

    def denoise(self, photo: Image.Image) -> Image.Image:
        rgb = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        clean = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            self._settings.luma_strength,
            self._settings.colour_strength,
            self._settings.template_window,
            self._settings.search_window,
        )
        return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

    def denoise_file(self, path: Path) -> Path:
        """Denoised copy of a file, cached on disk by content and settings."""
        if self._cache_dir is None:
            raise RuntimeError("denoise_file needs a cache_dir")
        digest = hashlib.sha256(path.read_bytes())
        digest.update(self._settings.key().encode())
        cached = self._cache_dir / f"dn-{digest.hexdigest()[:24]}.png"
        if not cached.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self.denoise(Image.open(path)).save(cached)
        return cached

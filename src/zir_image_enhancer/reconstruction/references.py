"""Reference images for generation: the front photo, named detail photos, and detail crops.

Crops come from the front photo itself, so they share its light and colour (close-ups taken
separately were lit differently and pulled the colour off by about 2 points). On a 12 MP
capture a crop also keeps the real letters of a small print, which the edit model would
otherwise see at about 1 megapixel.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from zir_image_enhancer.reconstruction.photos import Role
from zir_image_enhancer.reconstruction.spec import GarmentSpec

MIN_CROP_SIDE = 96  # smaller crops add nothing the full photo does not already show


@dataclass(frozen=True)
class Reference:
    path: Path
    label: str  # how the prompt refers to it, e.g. "close-up of the chest logo"


def build_references(
    spec: GarmentSpec,
    cache_dir: Path,
    max_references: int = 4,
    use_detail_photos: bool = True,
    use_crops: bool = True,
) -> tuple[Reference, ...]:
    references = [Reference(spec.front, "the whole garment from the front")]
    if use_detail_photos:
        # Only photos the user named as details; unnamed close-ups from the dataset were
        # measured to hurt (different light).
        for photo in spec.photos_with(Role.DETAIL):
            if photo.name:
                references.append(Reference(photo.path, f"a close-up of the {photo.name}"))
    if use_crops:
        for detail in spec.details:
            crop = _crop(spec.front, detail.box, cache_dir)
            if crop is not None:
                references.append(Reference(crop, f"a crop of the {detail.name} from image 1"))
    return tuple(references[:max_references])


def _crop(front: Path, box: tuple[int, int, int, int] | None, cache_dir: Path) -> Path | None:
    if box is None:
        return None
    image = ImageOps.exif_transpose(Image.open(front)).convert("RGB")
    ymin, xmin, ymax, xmax = box
    # 15% margin so the crop shows the detail with a bit of the garment around it.
    pad_y, pad_x = (ymax - ymin) * 0.15, (xmax - xmin) * 0.15
    left = max(0, int((xmin - pad_x) / 1000 * image.width))
    top = max(0, int((ymin - pad_y) / 1000 * image.height))
    right = min(image.width, int((xmax + pad_x) / 1000 * image.width))
    bottom = min(image.height, int((ymax + pad_y) / 1000 * image.height))
    if min(right - left, bottom - top) < MIN_CROP_SIDE:
        return None
    digest = hashlib.sha256(front.read_bytes())
    digest.update(str(box).encode())
    path = cache_dir / f"crop-{digest.hexdigest()[:24]}.png"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(path)
    return path

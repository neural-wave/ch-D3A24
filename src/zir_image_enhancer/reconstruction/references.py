"""Reference images for generation: the front photo and the detail photos the user named.

Unnamed close-ups from the dataset were measured to hurt (lit differently, -2 points), so
only photos named in the file name (03_detail_chest-logo.jpg) are sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zir_image_enhancer.reconstruction.photos import Role
from zir_image_enhancer.reconstruction.spec import GarmentSpec


@dataclass(frozen=True)
class Reference:
    path: Path
    label: str  # how the prompt refers to it, e.g. "close-up of the chest logo"


def named_detail_references(spec: GarmentSpec, limit: int) -> tuple[Reference, ...]:
    """Detail photos the user named in the file name (03_detail_chest-logo.jpg)."""
    named = [photo for photo in spec.photos_with(Role.DETAIL) if photo.name]
    return tuple(
        Reference(photo.path, f"a close-up of the {photo.name}") for photo in named[:limit]
    )

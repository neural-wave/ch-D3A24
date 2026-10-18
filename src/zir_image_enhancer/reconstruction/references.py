"""Reference images for generation: the front photo and the pipeline-picked detail close-ups.

The pipeline classifies the photos itself (assessment and spec vision calls) and names the
close-ups it finds, so no user labelling is needed. Close-ups are sent only when the pipeline
classified the photo as a detail and described what it shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zir_image_enhancer.reconstruction.photos import Role
from zir_image_enhancer.reconstruction.spec import GarmentSpec


@dataclass(frozen=True)
class Reference:
    path: Path
    label: str  # how the prompt refers to it, e.g. "a close-up of the chest logo"


def closeup_label(name: str) -> str:
    """How a close-up is named in the prompt; the studio prompt strips this prefix."""
    return f"a close-up of the {name}" if name else "a close-up of a garment detail"


def detail_references(spec: GarmentSpec, limit: int) -> tuple[Reference, ...]:
    """Detail close-ups the pipeline picked and named, ready to guide generation."""
    details = spec.photos_with(Role.DETAIL)
    return tuple(
        Reference(photo.path, closeup_label(photo.name)) for photo in details[:limit]
    )

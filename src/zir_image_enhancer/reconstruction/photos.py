"""Photo roles: which photo is the front, the back, or a detail close-up.

Every role is decided by the pipeline from the images themselves (the assessment and spec
vision calls); the user never labels photos. ``Role.UNKNOWN`` is only the starting state
before the pipeline has decided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Role(Enum):
    FRONT = "front"
    BACK = "back"
    DETAIL = "detail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RolePhoto:
    path: Path
    role: Role
    name: str = ""  # for details: what the close-up shows, e.g. "chest logo"


def unlabelled(paths: Sequence[Path]) -> tuple[RolePhoto, ...]:
    """Wrap raw photo paths; the pipeline assigns every role itself."""
    return tuple(RolePhoto(path, Role.UNKNOWN) for path in paths)

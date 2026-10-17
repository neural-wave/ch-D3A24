"""Photo roles: front, back, or a named detail.

Our own captures say the role in the file name, so nothing has to be guessed:

    01_front.jpg
    02_back.jpg
    03_detail_chest-logo.jpg      -> detail named "chest logo"

Photos without a role in the name (the dataset) get it from the vision model.
"""

from __future__ import annotations

import re
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
    name: str = ""  # for details: what the photo shows, e.g. "chest logo"


_NAMED = re.compile(r"^(?:\d+[_-])?(front|back|detail)(?:[_-](.+))?$", re.IGNORECASE)


def role_from_filename(path: Path) -> RolePhoto:
    match = _NAMED.match(path.stem)
    if match is None:
        return RolePhoto(path, Role.UNKNOWN)
    name = (match.group(2) or "").replace("-", " ").replace("_", " ").strip()
    return RolePhoto(path, Role(match.group(1).lower()), name)


def roles_from_filenames(paths: tuple[Path, ...]) -> tuple[RolePhoto, ...]:
    return tuple(role_from_filename(path) for path in paths)

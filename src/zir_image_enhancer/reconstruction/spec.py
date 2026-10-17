"""Garment spec: what the garment is, what must be kept, and where the details are.

One vision call per garment with all its photos. It decides the front (unless the file names
already say it), describes identity and construction, and lists the distinctive details with
a box on the front photo, so they can be cropped at full resolution and checked afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from zir_image_enhancer.reconstruction.photos import Role, RolePhoto
from zir_image_enhancer.reconstruction.vision import VisionClient, as_int

_SYSTEM = (
    "You describe one single garment from phone photos for a packshot reconstruction pipeline. "
    "Answer with one JSON object and nothing else. Only describe what the photos show."
)

_PROMPT = """There are {count} photos of the same garment, numbered 1 to {count}.
{known_roles}
Return this JSON object:
{{
  "front_index": <photo that shows the WHOLE garment from the FRONT most clearly>,
  "roles": [{{"index": <n>, "role": "<front|back|detail|label|packaging|other>"}}],
  "garment_box": [ymin, xmin, ymax, xmax],
  "category": "<e.g. crew-neck T-shirt, wide-leg trousers, cropped cardigan>",
  "construction": "<neckline, closures (number of buttons, zip), sleeves, pockets, hem, length>",
  "material": "<fabric and texture, e.g. ribbed knit, washed denim, satin>",
  "pattern": "<solid, stripes, print..., with colours named>",
  "details": [
    {{"name": "<short name>", "location": "<where on the garment>",
      "description": "<exact text or shape, colours>",
      "box": [ymin, xmin, ymax, xmax]}}
  ],
  "proportions": {{
    "body_length_to_width": <garment length (shoulder to hem) / chest width on photo front_index>,
    "sleeve_to_body_length": <sleeve length / garment length, or null without sleeves>,
    "notes": "<where sleeves end and where the hem falls, e.g. sleeves end at the hem>"
  }},
  "photo_lighting": "<the light in the photos, e.g. warm indoor light, dim, flash, daylight>",
  "photo_setup": "<how photo front_index shows the garment: flat|hanger|mannequin|worn|other>",
  "uncertain": ["<things the photos do not show clearly>"]
}}

Rules:
- The front has the neck opening, buttons or zip, chest print or logo. Never pick a back view
  as the front.
- "details": only things on the FRONT that are part of the garment (printed, woven,
  embroidered, sewn on: logos, text, prints, buttons, trims). At most 5. Not stickers, price
  tags or hang tags. "box" is on photo front_index, in 0-1000 coordinates; null if unsure.
- "garment_box": the garment only (no hanger, mannequin or person) on photo front_index,
  in 0-1000 coordinates.
- Quote any text exactly as written; say "unreadable" if you cannot read it.
"""


@dataclass(frozen=True)
class DetailSpec:
    name: str
    location: str
    description: str
    box: tuple[int, int, int, int] | None  # ymin, xmin, ymax, xmax in 0-1000 on the front photo


@dataclass(frozen=True)
class GarmentSpec:
    front: Path
    roles: tuple[RolePhoto, ...]
    category: str
    construction: str
    material: str
    pattern: str
    details: tuple[DetailSpec, ...]
    garment_box: tuple[int, int, int, int] | None = None  # on the front photo, 0-1000
    body_length_to_width: float | None = None
    sleeve_to_body_length: float | None = None
    proportion_notes: str = ""
    photo_lighting: str = ""
    photo_setup: str = ""  # flat, hanger, mannequin, worn or other
    uncertain: tuple[str, ...] = ()
    cost_usd: float = 0.0
    used_fallback: bool = False
    note: str = field(default="", compare=False)

    def photos_with(self, role: Role) -> tuple[RolePhoto, ...]:
        return tuple(photo for photo in self.roles if photo.role is role)

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["front"] = str(self.front)
        data["roles"] = [
            {"path": str(p.path), "role": p.role.value, "name": p.name} for p in self.roles
        ]
        return data


def extract_spec(photos: tuple[RolePhoto, ...], vision: VisionClient) -> GarmentSpec:
    """Ask the vision model once; on any failure fall back to a minimal spec (never stop)."""
    known = [
        f"Photo {i} is the {p.role.value}" + (f" ({p.name})" if p.name else "") + "."
        for i, p in enumerate(photos, start=1)
        if p.role is not Role.UNKNOWN
    ]
    known_roles = ("The user labelled some photos: " + " ".join(known) + "\n") if known else ""
    prompt = _PROMPT.format(count=len(photos), known_roles=known_roles)
    try:
        answer = vision.ask(_SYSTEM, prompt, [p.path for p in photos], kind="spec")
    except Exception as error:  # noqa: BLE001
        return _fallback(photos, str(error))
    return _build(photos, answer.data, answer.cost_usd)


def _build(photos: tuple[RolePhoto, ...], data: dict[str, object], cost: float) -> GarmentSpec:
    labelled_front = [i for i, p in enumerate(photos) if p.role is Role.FRONT]
    if labelled_front:
        # What the user said wins over what the model guessed.
        front_index = labelled_front[0]
    else:
        front_index = (as_int(data.get("front_index")) or 1) - 1
        if not 0 <= front_index < len(photos):
            front_index = 0

    guessed = {
        as_int(row.get("index")): str(row.get("role", ""))
        for row in _list(data.get("roles"))
        if isinstance(row, dict)
    }
    roles = []
    for i, photo in enumerate(photos):
        if photo.role is not Role.UNKNOWN:
            roles.append(photo)
            continue
        role = Role.FRONT if i == front_index else _role(guessed.get(i + 1, ""))
        roles.append(RolePhoto(photo.path, role))

    raw_proportions = data.get("proportions")
    proportions = raw_proportions if isinstance(raw_proportions, dict) else {}
    details = []
    for row in _list(data.get("details"))[:5]:
        if not isinstance(row, dict):
            continue
        details.append(
            DetailSpec(
                name=str(row.get("name", "")),
                location=str(row.get("location", "")),
                description=str(row.get("description", "")),
                box=_box(row.get("box")),
            )
        )
    return GarmentSpec(
        front=photos[front_index].path,
        roles=tuple(roles),
        category=str(data.get("category", "garment")),
        construction=str(data.get("construction", "")),
        material=str(data.get("material", "")),
        pattern=str(data.get("pattern", "")),
        details=tuple(details),
        garment_box=_box(data.get("garment_box")),
        body_length_to_width=_ratio(proportions.get("body_length_to_width")),
        sleeve_to_body_length=_ratio(proportions.get("sleeve_to_body_length")),
        proportion_notes=str(proportions.get("notes") or ""),
        photo_lighting=str(data.get("photo_lighting") or ""),
        photo_setup=str(data.get("photo_setup") or "").lower(),
        uncertain=tuple(str(item) for item in _list(data.get("uncertain"))),
        cost_usd=cost,
    )


def _fallback(photos: tuple[RolePhoto, ...], note: str) -> GarmentSpec:
    front = next((p.path for p in photos if p.role is Role.FRONT), photos[0].path)
    return GarmentSpec(front, photos, "garment", "", "", "", (), used_fallback=True, note=note)


def _ratio(value: object) -> float | None:
    try:
        ratio = float(str(value))
    except ValueError:
        return None
    # Outside this range the estimate is a misread, not a garment proportion.
    return round(ratio, 2) if 0.2 <= ratio <= 5.0 else None


def _role(value: str) -> Role:
    mapping = {"front": Role.FRONT, "back": Role.BACK, "detail": Role.DETAIL}
    return mapping.get(value.lower(), Role.UNKNOWN)


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _box(value: object) -> tuple[int, int, int, int] | None:
    numbers = [as_int(v) for v in _list(value)]
    if len(numbers) != 4 or any(n is None for n in numbers):
        return None
    ymin, xmin, ymax, xmax = (max(0, min(1000, n or 0)) for n in numbers)
    if ymax - ymin < 10 or xmax - xmin < 10:
        return None
    return ymin, xmin, ymax, xmax


def spec_summary(spec: GarmentSpec) -> str:
    """One line per field, for logs and the report."""
    return json.dumps(
        {
            "category": spec.category,
            "details": [d.name for d in spec.details],
            "front": spec.front.name,
        }
    )

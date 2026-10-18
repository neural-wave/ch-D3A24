"""Garment spec: input assessment, then what the garment is and where its details are.

Input assessment runs first (architecture §1): one vision call scores every photo for
technical quality, garment visibility and the information it carries (colour, texture,
details, silhouette, geometry). Clearly useless photos (price tags, hang tags, packaging,
receipts, blurred artifacts, duplicates) fall below a relaxed floor and are dropped.

One second vision call then describes the garment from the photos that survived, decides the
front, describes identity and construction, and names the detail close-ups so they can be
sent as generation references, and lists the distinctive details with a box on the front
photo, so they can be cropped at full resolution and checked afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from zir_image_enhancer.reconstruction.photos import Role, RolePhoto
from zir_image_enhancer.reconstruction.vision import VisionClient, as_int

_SYSTEM = (
    "You describe one single product (a garment, or an accessory such as glasses, a bag or a "
    "belt) from phone photos for a packshot reconstruction pipeline. "
    "Answer with one JSON object and nothing else. Only describe what the photos show."
)

_PROMPT = """There are {count} photos of the same garment, numbered 1 to {count}.
Return this JSON object:
{{
  "front_index": <photo that shows the WHOLE garment from the FRONT most clearly>,
  "roles": [{{"index": <n>, "role": "<front|back|detail|label|packaging|other>",
             "name": "<only for role detail: what the close-up shows, e.g. chest logo>"}}],
  "garment_box": [ymin, xmin, ymax, xmax],
  "category": "<e.g. crew-neck T-shirt, wide-leg trousers, cropped cardigan, sunglasses>",
  "pieces": ["<each separate piece of this product, e.g. bikini top, bikini bottom>"],
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
- "pieces": a product sold as a set (bikini, two-piece set, pyjamas, pair of socks) lists
  every piece; a single garment has one piece. Clothes only worn with it (jeans under a top),
  the mannequin and the hanger are not pieces.
- "roles": classify every photo yourself. "name" is only for a "detail" close-up and must say
  what that close-up shows (e.g. "chest logo", "sleeve cuff"), so it can be sent as a reference.
"""


# --- Input assessment (architecture §1) ------------------------------------------------------

_ASSESS_SYSTEM = (
    "You audit the input photos of one single product for packshot reconstruction, before it "
    "is described. Score every photo on how useful it is as evidence of the product. "
    "Answer with one JSON object and nothing else. Judge each photo on its own; do not "
    "describe the product."
)

_ASSESS_PROMPT = """There are {count} photos of the same product, numbered 1 to {count}.
Score every photo and return this JSON object:
{{
  "images": [
    {{
      "index": <n>,
      "role": "<front|back|detail|label|packaging|other|artifact>",
      "quality": <0..1 sharp, well lit, complete; low if blurry, badly exposed or cropped>,
      "garment_visibility": <0..1 how much of the actual product is shown, unoccluded>,
      "colour": <0..1 how reliably it shows the true colours>,
      "texture": <0..1 how reliably it shows fabric and surface texture>,
      "details": <0..1 how reliably it shows small details (print, stitching, buttons, labels)>,
      "silhouette": <0..1 how reliably it shows the overall shape and outline>,
      "geometry": <0..1 how reliably it shows proportions, seams and construction>,
      "redundant": <true if it repeats another photo's view, light and information>,
      "note": "<short reason, or what this photo uniquely adds>"
    }}
  ]
}}

Rules:
- Score every photo from 1 to {count}, including the ones you would throw away.
- Use the whole 0 to 1 range; do not give every photo the same scores.
- A photo of only a price tag, hang tag, packaging, receipt or barcode has almost no garment
  visibility or information: score it low and never call it a front or a back.
- A close-up is not low value: score it high on the one or two features it clearly shows and
  low on the rest.
- When two photos repeat the same view and light, mark the weaker one "redundant": true.
- "quality" is about the photo; "garment_visibility" is about how much garment is in frame.
"""

#: How much each information category counts toward the aggregate informational value (§2).
_INFO_WEIGHTS: dict[str, float] = {
    "colour": 0.25,
    "texture": 0.20,
    "details": 0.20,
    "silhouette": 0.18,
    "geometry": 0.17,
}

#: Only photos below this informational value are dropped. Deliberately relaxed: it should
#: remove clear non-evidence (labels, tags, packaging, receipts, artifacts, duplicates) and
#: keep any photo a human would still call useful.
MIN_INFORMATIONAL_VALUE = 0.25

#: A whole-garment photo is never dropped however the model scored it, so a bad estimate
#: cannot leave the pipeline without a full view of the product.
_PROTECTED_ASSESSMENT_ROLES = frozenset({"front", "back"})


@dataclass(frozen=True)
class PhotoAssessment:
    """One photo's evidence value, as scored by the assessment vision call (all in [0,1])."""

    index: int  # 1-based, as shown to the vision model
    path: str
    role: str = ""
    quality: float = 0.0
    garment_visibility: float = 0.0
    colour: float = 0.0
    texture: float = 0.0
    details: float = 0.0
    silhouette: float = 0.0
    geometry: float = 0.0
    redundant: bool = False
    note: str = ""

    @property
    def information(self) -> float:
        """Weighted blend of the information categories, regardless of photo quality."""
        return round(
            sum(getattr(self, key) * weight for key, weight in _INFO_WEIGHTS.items()), 3
        )

    @property
    def informational_value(self) -> float:
        """How much this photo is worth as evidence, combining quality, visibility and
        information. A redundant photo is halved: keep the better copy, not both."""
        value = (
            self.quality
            * (0.3 + 0.7 * self.information)
            * (0.3 + 0.7 * self.garment_visibility)
        )
        if self.redundant:
            value *= 0.5
        return round(value, 3)


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
    pieces: tuple[str, ...] = ()  # more than one for sets such as a bikini
    uncertain: tuple[str, ...] = ()
    cost_usd: float = 0.0
    used_fallback: bool = False
    note: str = field(default="", compare=False)
    #: Every photo that was assessed, including the ones dropped from ``roles``.
    assessments: tuple[PhotoAssessment, ...] = ()

    def photos_with(self, role: Role) -> tuple[RolePhoto, ...]:
        return tuple(photo for photo in self.roles if photo.role is role)

    @property
    def dropped(self) -> tuple[PhotoAssessment, ...]:
        """Photos that the input assessment removed as non-evidence."""
        kept = {str(photo.path) for photo in self.roles}
        return tuple(a for a in self.assessments if a.path not in kept)

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["front"] = str(self.front)
        data["roles"] = [
            {"path": str(p.path), "role": p.role.value, "name": p.name} for p in self.roles
        ]
        data["assessments"] = [
            {**asdict(a), "informational_value": a.informational_value} for a in self.assessments
        ]
        return data


def extract_spec(photos: tuple[RolePhoto, ...], vision: VisionClient) -> GarmentSpec:
    """Assess the photos (§1), then extract the spec from the ones worth keeping (§2).

    On any failure the pipeline falls back to a minimal spec (it never stops).
    """
    assessments: tuple[PhotoAssessment, ...] = ()
    selected = photos
    cost = 0.0
    if len(photos) > 1:
        assessments, cost = assess_photos(photos, vision)
        selected, _ = select_photos(photos, assessments)
        if not selected:
            selected = photos

    prompt = _PROMPT.format(count=len(selected))
    try:
        answer = vision.ask(_SYSTEM, prompt, [p.path for p in selected], kind="spec")
    except Exception as error:  # noqa: BLE001
        return _fallback(selected, str(error), assessments=assessments, cost=cost)
    return _build(selected, answer.data, cost + answer.cost_usd, assessments)


def assess_photos(
    photos: tuple[RolePhoto, ...], vision: VisionClient
) -> tuple[tuple[PhotoAssessment, ...], float]:
    """Score every photo for evidence value. A failed assessment keeps every photo."""
    prompt = _ASSESS_PROMPT.format(count=len(photos))
    try:
        answer = vision.ask(_ASSESS_SYSTEM, prompt, [p.path for p in photos], kind="assess")
    except Exception:  # noqa: BLE001
        return (), 0.0
    return _parse_assessments(photos, answer.data), answer.cost_usd


def select_photos(
    photos: tuple[RolePhoto, ...],
    assessments: tuple[PhotoAssessment, ...],
    threshold: float = MIN_INFORMATIONAL_VALUE,
) -> tuple[tuple[RolePhoto, ...], tuple[PhotoAssessment, ...]]:
    """Drop only photos that clearly carry no evidence (§1).

    A photo is kept when it is a whole-garment view (front or back) or scores at or above the
    relaxed threshold. A photo the assessment never scored is kept (conservative). At least
    one photo always survives.
    """
    by_index = {a.index: a for a in assessments}
    if not by_index:
        return photos, assessments

    kept_photos: list[RolePhoto] = []
    kept_assessments: list[PhotoAssessment] = []
    for i, photo in enumerate(photos, start=1):
        assessment = by_index.get(i)
        if assessment is None:
            kept_photos.append(photo)
            continue
        whole_garment = assessment.role.strip().lower() in _PROTECTED_ASSESSMENT_ROLES
        if whole_garment or assessment.informational_value >= threshold:
            kept_photos.append(photo)
            kept_assessments.append(assessment)

    if not kept_photos:
        index, best = max(by_index.items(), key=lambda item: item[1].informational_value)
        return (photos[index - 1],), (best,)
    return tuple(kept_photos), tuple(kept_assessments)


def _parse_assessments(
    photos: tuple[RolePhoto, ...], data: dict[str, object]
) -> tuple[PhotoAssessment, ...]:
    rows: dict[int, PhotoAssessment] = {}
    for row in _list(data.get("images")):
        if not isinstance(row, dict):
            continue
        index = as_int(row.get("index"))
        if index is None or not 1 <= index <= len(photos):
            continue
        rows[index] = PhotoAssessment(
            index=index,
            path=str(photos[index - 1].path),
            role=str(row.get("role", "")).lower(),
            quality=_score(row.get("quality")),
            garment_visibility=_score(row.get("garment_visibility")),
            colour=_score(row.get("colour")),
            texture=_score(row.get("texture")),
            details=_score(row.get("details")),
            silhouette=_score(row.get("silhouette")),
            geometry=_score(row.get("geometry")),
            redundant=row.get("redundant") is True,
            note=str(row.get("note", "")),
        )
    return tuple(rows[index] for index in sorted(rows))


def _build(
    photos: tuple[RolePhoto, ...],
    data: dict[str, object],
    cost: float,
    assessments: tuple[PhotoAssessment, ...] = (),
) -> GarmentSpec:
    front_index = (as_int(data.get("front_index")) or 1) - 1
    if not 0 <= front_index < len(photos):
        front_index = 0

    guessed: dict[int, tuple[str, str]] = {}
    for row in _list(data.get("roles")):
        if isinstance(row, dict):
            index = as_int(row.get("index"))
            if index is not None:
                guessed[index] = (str(row.get("role", "")), str(row.get("name", "")))

    roles = []
    for i, photo in enumerate(photos):
        role_text, name = guessed.get(i + 1, ("", ""))
        role = Role.FRONT if i == front_index else _role(role_text)
        roles.append(RolePhoto(photo.path, role, name if role is Role.DETAIL else ""))

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
        pieces=tuple(str(piece) for piece in _list(data.get("pieces")) if piece)[:4],
        uncertain=tuple(str(item) for item in _list(data.get("uncertain"))),
        cost_usd=cost,
        assessments=assessments,
    )


def _fallback(
    photos: tuple[RolePhoto, ...],
    note: str,
    assessments: tuple[PhotoAssessment, ...] = (),
    cost: float = 0.0,
) -> GarmentSpec:
    front = photos[0].path
    return GarmentSpec(
        front,
        photos,
        "garment",
        "",
        "",
        "",
        (),
        cost_usd=cost,
        used_fallback=True,
        note=note,
        assessments=assessments,
    )


def _score(value: object) -> float:
    """A 0..1 score from the model, tolerating a 0..100 reply or a missing value."""
    try:
        number = float(str(value))
    except ValueError:
        return 0.0
    if number > 1.0:
        number /= 100.0
    return round(max(0.0, min(1.0, number)), 3)


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

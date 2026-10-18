"""OpenCV preprocessing: turn phone photos into the best references the edit model can get.

The image model (nano-banana) was trained on ordinary photographs and it copies whatever it
sees. A phone photo differs from the studio target in ways a few classic CV operations can
remove cheaply, *before* the model runs, so the model spends its capacity on the garment
instead of on the capture:

* a colour cast (warm tungsten, green fluorescent, blue shade) that the model copies into
  the packshot, so the colour the jury sees is wrong;
* a dark or flat exposure, so the model returns a dimmer packshot than a catalogue photo;
* a garment that is small in frame, surrounded by hanger, floor, wall or person;
* a small detail (a chest logo, a wash label, a zip puller) that is only a handful of pixels
  wide, so the model redraws the letters and shapes from memory and gets them wrong.

The module also adds *controlled* photographic noise (a small Gaussian grain) so an input
that has been through several resizes and JPEG round-trips stays inside the natural-image
distribution the model was trained on, instead of looking like a synthetically smoothed
image it may treat as an illustration.

It is deliberately conservative and every step can be turned off. Earlier experiments in this
repo showed that heavy-handed processing *lowered* the holdout score: cut-out photos scored
57.16 (against 61.82), a photo cropped around the garment 60.50, denoised 59.95, and measured
hex colours written into the prompt 57.16. The defaults here therefore only remove capture
artefacts and add information; they do not restyle or flatten the garment. See
``docs/PREPROCESSING.md``.

Typical use::

    from zir_image_enhancer.reconstruction.preprocess import (
        PreprocessSettings, prepare_references,
    )
    prepared = prepare_references(
        front=front_jpeg,
        detail_photos=(("chest logo", logo_jpeg),),
        details=(("chest logo", detail_box),),
        out_dir=workdir / "references",
        settings=PreprocessSettings(),
    )

Preview one image from the shell::

    python -m zir_image_enhancer.reconstruction.preprocess in.jpg out.jpg --stats
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
Bool = NDArray[np.bool_]

#: A box as the vision model reports it: ymin, xmin, ymax, xmax on a 0-1000 grid.
Box = tuple[int, int, int, int]

#: How ``PreprocessSettings.white_balance_method`` estimates the scene illuminant.
WHITE_BALANCE_METHODS = ("gray_world", "neutral_pixels")


def to_u8(array: NDArray[np.generic] | NDArray[np.bool_]) -> U8:
    """Cast any numeric array to an 8-bit image.

    OpenCV's numpy stubs widen the dtype of many results (``np.clip`` and ``cv2`` ops return
    ``integer | floating``), so every 8-bit return goes through here.
    """
    return np.ascontiguousarray(array, dtype=np.uint8)


@dataclass(frozen=True)
class PreprocessSettings:
    """What to do to a photo before it becomes a generation reference.

    Everything is an independent switch so a change can be A/B tested with
    ``scripts/reconstruct.py``; the defaults are the conservative set described in the module
    docstring.
    """

    enabled: bool = True

    # --- capture normalisation (remove phone-camera artefacts, keep the garment's colour) ---
    white_balance: float = 0.5  # gray-world strength in [0, 1]; 0 disables
    white_balance_max_gain: float = 0.30  # per-channel gain clamp, so a red garment survives
    #: ``"gray_world"`` (the historical default) or ``"neutral_pixels"``. The latter measures
    #: the illuminant from near-neutral, bright pixels only, so a warm floor or a beige wall
    #: cannot pull a white garment blue the way a whole-scene average can.
    white_balance_method: str = "gray_world"
    #: Illumination flattening strength in [0, 1]; 0 disables. Divides out the low-frequency
    #: light field (the photographer's cast shadow, vignetting, a lamp to one side) so the
    #: garment is evenly lit the way a studio light bank lights it. Luminance-only and
    #: mean-preserving, so it does not change hue or global exposure. Previewed: a cast shadow
    #: that the model would otherwise copy as a fold disappears; kept off by default until it
    #: is A/B tested end to end.
    flatten_illumination: float = 0.0
    auto_tone: bool = True  # gentle percentile exposure/contrast stretch
    denoise: bool = False  # edge-preserving denoise; off, it flattened fabric texture
    sharpen: float = 0.3  # mild unsharp after downscale
    grain_sigma: float = 1.0  # controlled sensor noise, in 0-255 levels; 0 disables
    jpeg_quality: int = 95

    # --- geometry and framing ---
    max_side: int = 1536  # cap the long edge of a reference sent to the model
    min_detail_side: int = 320  # upscale tiny detail crops so the model can read them
    detail_margin: float = 0.30  # context added around a detail box, as a fraction of its size
    crop_to_garment: bool = False  # tight framing tested worse (60.50 vs 61.82): off by default
    garment_margin: float = 0.15

    # --- information extracted with OpenCV ---
    detail_crops: bool = True  # crop the front photo at the spec detail boxes (architecture 3)
    palette: bool = False  # estimate dominant colours; kept for experiments, off by default


@dataclass(frozen=True)
class PreparedReferences:
    """Files written for one garment: the front reference and its detail close-ups."""

    front: Path
    #: (name, path) for close-ups. ``name`` is empty when the pipeline could not name it.
    closeups: tuple[tuple[str, Path], ...]
    #: Capture statistics of the *original* front photo, for logs and reports.
    stats: Mapping[str, float]


# --- image I/O -----------------------------------------------------------------------------


def read_image(path: Path) -> U8:
    """Read a colour image as RGB. Uses ``np.fromfile`` so unicode paths work."""
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return to_u8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def write_image(path: Path, rgb: U8, quality: int = 95) -> Path:
    """Write an RGB image as JPEG, creating parent folders. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"cannot encode JPEG: {path}")
    buffer.tofile(str(path))
    return path


def fit_max_side(rgb: U8, max_side: int) -> U8:
    """Downscale so the long edge is at most ``max_side``; never upscale."""
    height, width = rgb.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return to_u8(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))


# --- capture normalisation -----------------------------------------------------------------


def gray_world(rgb: U8, strength: float, max_gain: float = 0.30, sample: Bool | None = None) -> U8:
    """Pull the colour towards neutral to remove a cast.

    Gray-world assumes the scene averages to grey, which is false for a red dress filling the
    frame, so the correction is blended by ``strength`` and each gain is clamped to
    ``1 +/- max_gain``. Pass a ``sample`` mask (True = pixel to measure, e.g. the background)
    when a garment mask is available.
    """
    if strength <= 0.0:
        return rgb
    flat = rgb.reshape(-1, 3).astype(np.float32)
    measured = flat if sample is None else flat[sample.reshape(-1)]
    if measured.shape[0] < 32:  # too little to trust; fall back to the whole image
        measured = flat
    means = measured.mean(axis=0)
    gains = float(means.mean()) / np.maximum(means, 1.0)
    gains = 1.0 + strength * (gains - 1.0)
    gains = np.clip(gains, 1.0 - max_gain, 1.0 + max_gain)
    out = flat * gains
    return to_u8(np.clip(out, 0, 255).reshape(rgb.shape))


def neutral_white_balance(
    rgb: U8,
    strength: float,
    max_gain: float = 0.30,
    low_saturation: float = 0.35,
    high_value: float = 0.55,
    sample: Bool | None = None,
) -> U8:
    """White balance from near-neutral, reasonably bright pixels.

    Gray-world averages the whole scene, so a warm wood floor or a beige wall is mistaken for
    the light and a white garment comes out blue. The grey-pixel/white-patch family instead
    trusts only pixels that *look* like the light: low saturation and decent brightness. A
    white or grey garment (which is what most studio shots are) dominates that set, so it is
    corrected towards neutral without dragging its own colour, and a strongly coloured garment
    barely moves because few of its pixels are picked. Falls back to the low-saturation set,
    then to the sampled region, only when too few pixels qualify.
    """
    if strength <= 0.0:
        return rgb
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    valid = (
        np.ones(saturation.shape, dtype=bool)
        if sample is None
        else sample.reshape(saturation.shape).astype(bool)
    )
    picked = valid & (saturation < low_saturation) & (value > high_value)
    if int(picked.sum()) < 64:
        picked = valid & (saturation < low_saturation)
    if int(picked.sum()) < 64:
        picked = valid
    flat = rgb.reshape(-1, 3).astype(np.float32)
    measured = flat[picked.reshape(-1)]
    means = measured.mean(axis=0)
    gains = float(means.mean()) / np.maximum(means, 1.0)
    gains = 1.0 + strength * (gains - 1.0)
    gains = np.clip(gains, 1.0 - max_gain, 1.0 + max_gain)
    return to_u8(np.clip(flat * gains, 0, 255).reshape(rgb.shape))


def flatten_illumination(
    rgb: U8, strength: float, sigma_fraction: float = 0.25, mask: U8 | None = None
) -> U8:
    """Divide out the low-frequency illumination field (even the light like a studio bank).

    A phone photo of a garment on the floor carries the photographer's shadow, a lamp on one
    side or a vignette. The model cannot tell that from real folds in the fabric, so it copies
    it into the packshot. Flattening computes a heavily blurred copy of the lightness (the
    light field), then scales the image by ``mean / field``: shadows are lifted, hot spots are
    pulled down, and because only the *lightness* ratio is applied the hue is untouched and the
    mean exposure is preserved. With a garment ``mask`` the correction is faded out over the
    background, so the studio backdrop the model invents is not given a brightened halo.
    """
    if strength <= 0.0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    height, width = lightness.shape[:2]
    sigma = max(1.0, sigma_fraction * max(height, width))
    field = cv2.GaussianBlur(lightness, (0, 0), sigma)
    field = np.maximum(field, 8.0)
    gain = float(field.mean()) / field
    if mask is not None:
        weight = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), sigma * 0.5)
        gain = 1.0 + weight * (gain - 1.0)
    gain = 1.0 + strength * (gain - 1.0)
    out = rgb.astype(np.float32) * gain[:, :, None]
    return to_u8(np.clip(out, 0, 255))


def auto_tone(rgb: U8, low: float = 1.0, high: float = 99.0) -> U8:
    """Stretch the L channel between two percentiles to lift a dark or flat exposure.

    A percentile stretch is gentle: it does not change local contrast (no halos, no synthetic
    look) and it is skipped when the image already spans most of the range.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    lo, hi = np.percentile(lightness, [low, high])
    if hi - lo < 16.0:
        return rgb
    lab[:, :, 0] = np.clip((lightness - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return to_u8(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def denoise(rgb: U8) -> U8:
    """Edge-preserving denoise (bilateral). Off by default: it flattens fabric texture."""
    return to_u8(cv2.bilateralFilter(rgb, 5, 32, 32))


def unsharp(rgb: U8, amount: float) -> U8:
    """Mild unsharp mask, to restore a little crispness lost to downscaling."""
    if amount <= 0.0:
        return rgb
    blurred = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    return to_u8(cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0))


def add_grain(rgb: U8, sigma: float, seed: int) -> U8:
    """Add reproducible Gaussian grain, keeping a processed image photographic.

    The seed is derived from the source bytes, so a run is reproducible and the fal cache key
    still matches across machines.
    """
    if sigma <= 0.0:
        return rgb
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, rgb.shape).astype(np.float32)
    return to_u8(np.clip(rgb.astype(np.float32) + noise, 0, 255))


# --- feature extraction --------------------------------------------------------------------


def garment_mask(rgb: U8, base_threshold: float = 14.0, spread_factor: float = 1.5) -> U8 | None:
    """Best-effort foreground mask of the garment, or ``None`` when unsure.

    Estimates the background colour from a border ring, thresholds the colour distance in LAB,
    keeps the largest connected component and fills its holes. Robust to a plain wall, a floor
    or a mannequin, but returns ``None`` on a busy or ambiguous scene rather than guess.
    """
    height, width = rgb.shape[:2]
    small = fit_max_side(rgb, 256)
    small = to_u8(cv2.GaussianBlur(small, (0, 0), 1.2))
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    border = max(2, round(0.08 * min(h, w)))
    ring = np.zeros((h, w), dtype=bool)
    ring[:border, :] = ring[-border:, :] = ring[:, :border] = ring[:, -border:] = True
    background = lab[ring]
    median = np.median(background, axis=0)
    spread = float(np.median(np.linalg.norm(background - median, axis=1)))

    distance = np.linalg.norm(lab - median, axis=2)
    threshold = max(base_threshold, spread_factor * spread + 8.0)
    foreground = to_u8(distance > threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = to_u8(cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel))
    foreground = to_u8(cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2))

    # OpenCV 5's stubs type the image dtype invariantly, so a uint8 image is rejected here.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(  # type: ignore[call-overload]
        foreground, 8
    )
    if count <= 1:
        return None
    # A garment usually sits inside the frame; the background it is lying on touches the edge.
    # Prefer the largest blob that does not, so a floor or door is not mistaken for the garment.
    blobs = stats[1:]
    areas = blobs[:, cv2.CC_STAT_AREA]
    touches_border = (
        (blobs[:, cv2.CC_STAT_LEFT] <= 0)
        | (blobs[:, cv2.CC_STAT_TOP] <= 0)
        | (blobs[:, cv2.CC_STAT_LEFT] + blobs[:, cv2.CC_STAT_WIDTH] >= w - 1)
        | (blobs[:, cv2.CC_STAT_TOP] + blobs[:, cv2.CC_STAT_HEIGHT] >= h - 1)
    )
    inside = np.flatnonzero(~touches_border)
    candidates = inside if inside.size else np.arange(areas.size)
    index = 1 + int(candidates[np.argmax(areas[candidates])])
    component = to_u8(labels == index)
    fraction = stats[index, cv2.CC_STAT_AREA] / (h * w)
    if not 0.02 <= fraction <= 0.95:
        return None

    # Fill holes: flood the outside from a corner, whatever stays 0 is enclosed by the garment.
    flood = component.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    component = to_u8(np.maximum(component, to_u8(flood == 0)))

    return to_u8(cv2.resize(component * 255, (width, height), interpolation=cv2.INTER_NEAREST))


def mask_bbox(mask: U8) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) bounding box of the mask's non-zero pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("empty mask")
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def dominant_colours(
    rgb: U8, mask: U8 | None = None, k: int = 4, max_samples: int = 4000
) -> tuple[tuple[str, float], ...]:
    """k-means palette of the garment, as (hex, share) sorted by share.

    Intended for reports and experiments; it is not written into the generation prompt by
    default (measured hex colours scored worse).
    """
    pixels = rgb.reshape(-1, 3)
    if mask is not None:
        pixels = pixels[mask.reshape(-1) > 0]
    if pixels.shape[0] < k:
        return ()
    if pixels.shape[0] > max_samples:
        picks = np.linspace(0, pixels.shape[0] - 1, max_samples).astype(int)
        pixels = pixels[picks]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # bestLabels=None is accepted at runtime; OpenCV 5's stubs do not model it.
    _, labels, centers = cv2.kmeans(  # type: ignore[call-overload]
        pixels.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    total = max(1, int(counts.sum()))
    palette = []
    for index in np.argsort(-counts):
        r, g, b = (int(round(value)) for value in centers[index])
        palette.append((f"#{r:02x}{g:02x}{b:02x}", round(int(counts[index]) / total, 3)))
    return tuple(palette)


def image_stats(rgb: U8) -> dict[str, float]:
    """Cheap measurements of the capture: blur, exposure, contrast and colour cast."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    height, width = gray.shape[:2]
    return {
        "width": float(width),
        "height": float(height),
        "blur": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(lab[:, :, 0].std()), 1),
        "cast_a": round(float(lab[:, :, 1].mean()) - 128.0, 1),
        "cast_b": round(float(lab[:, :, 2].mean()) - 128.0, 1),
    }


# --- boxes and crops -----------------------------------------------------------------------


def box_to_pixels(box: Box, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    """Convert a 0-1000 vision box to a padded, clamped pixel box (x0, y0, x1, y1)."""
    ymin, xmin, ymax, xmax = box
    x0, x1 = xmin / 1000.0 * width, xmax / 1000.0 * width
    y0, y1 = ymin / 1000.0 * height, ymax / 1000.0 * height
    padding = margin * max(x1 - x0, y1 - y0)
    x0, x1 = x0 - padding, x1 + padding
    y0, y1 = y0 - padding, y1 + padding

    def clamp(value: float, top: int) -> int:
        return int(round(max(0.0, min(float(top), value))))

    return clamp(x0, width), clamp(y0, height), clamp(x1, width), clamp(y1, height)


def crop(rgb: U8, box: tuple[int, int, int, int]) -> U8:
    """Crop an (x0, y0, x1, y1) box; raises when the box is empty."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"degenerate crop box: {box}")
    return rgb[y0:y1, x0:x1]


def crop_around(rgb: U8, box: tuple[int, int, int, int], margin: float) -> U8:
    """Crop with a margin (fraction of the box's own size) added on every side."""
    x0, y0, x1, y1 = box
    pad = margin * max(x1 - x0, y1 - y0)
    height, width = rgb.shape[:2]
    return crop(
        rgb,
        (
            max(0, int(round(x0 - pad))),
            max(0, int(round(y0 - pad))),
            min(width, int(round(x1 + pad))),
            min(height, int(round(y1 + pad))),
        ),
    )


def clip_crop(rgb: U8, x0: int, y0: int, x1: int, y1: int) -> U8 | None:
    """Crop a pixel box clipped to the image; ``None`` when too little of it remains."""
    height, width = rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return to_u8(rgb[y0:y1, x0:x1])


# --- the reference pipeline ----------------------------------------------------------------


def enhance_full(rgb: U8, settings: PreprocessSettings, mask: U8 | None) -> U8:
    """Colour and tone normalisation at full resolution, before any cropping or downscale.

    Keeping this step on the full image means the front reference and the detail crops cut
    from it share exactly the same colour treatment, as the prompt requires.
    """
    background = (mask == 0) if mask is not None else None
    if settings.flatten_illumination > 0.0:
        rgb = flatten_illumination(rgb, settings.flatten_illumination, mask=mask)
    balance = (
        neutral_white_balance if settings.white_balance_method == "neutral_pixels" else gray_world
    )
    out = balance(
        rgb,
        settings.white_balance,
        settings.white_balance_max_gain,
        sample=background,
    )
    if settings.auto_tone:
        out = auto_tone(out)
    if settings.denoise:
        out = denoise(out)
    return out


def finalize(
    rgb: U8, settings: PreprocessSettings, seed: int, *, min_side: int | None = None
) -> U8:
    """Downscale for the model, restore crispness, then add controlled grain."""
    out = fit_max_side(rgb, settings.max_side)
    if min_side is not None:
        height, width = out.shape[:2]
        if max(height, width) < min_side:
            scale = min_side / max(height, width)
            out = to_u8(
                cv2.resize(
                    out,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            )
    out = unsharp(out, settings.sharpen)
    return add_grain(out, settings.grain_sigma, seed)


def grain_seed(path: Path) -> int:
    """A stable grain seed from the file bytes (not the path), so runs stay reproducible."""
    digest = hashlib.sha256(Path(path).read_bytes()).digest()
    return int.from_bytes(digest[:4], "big")


def prepare_references(
    *,
    front: Path,
    out_dir: Path,
    settings: PreprocessSettings,
    detail_photos: Sequence[tuple[str, Path]] = (),
    details: Sequence[tuple[str, Box | None]] = (),
) -> PreparedReferences:
    """Write the generation references for one garment and return their paths.

    Produces:

    * the front reference, enhanced and capped to ``max_side``;
    * one close-up per user detail photo, enhanced the same way;
    * one close-up per spec detail that has a box, cropped at full resolution from the
      *enhanced* front photo so it matches image 1 exactly (architecture section 3).
    """
    front_rgb = read_image(front)
    full_height, full_width = front_rgb.shape[:2]
    mask = garment_mask(front_rgb)
    stats = image_stats(front_rgb)
    if mask is not None:
        stats["garment_fraction"] = round(float((mask > 0).mean()), 3)

    seed = grain_seed(front)
    enhanced = enhance_full(front_rgb, settings, mask)

    # Optional tight framing. Detail boxes are in the original frame, so remember the offset
    # and shift them below; else crops would point at the wrong place after the trim.
    origin_x = origin_y = 0
    if settings.crop_to_garment and mask is not None:
        x0, y0, x1, y1 = mask_bbox(mask)
        pad = round(settings.garment_margin * max(x1 - x0, y1 - y0))
        origin_x, origin_y = max(0, x0 - pad), max(0, y0 - pad)
        enhanced = crop(
            enhanced,
            (origin_x, origin_y, min(x1 + pad, full_width), min(y1 + pad, full_height)),
        )

    front_out = finalize(enhanced, settings, seed)
    write_image(out_dir / "front.jpg", front_out, settings.jpeg_quality)

    closeups: list[tuple[str, Path]] = []
    for number, (name, path) in enumerate(detail_photos, start=1):
        photo = enhance_full(read_image(path), settings, None)
        write_image(
            out_dir / f"photo_{number:02d}.jpg",
            finalize(photo, settings, grain_seed(path)),
            settings.jpeg_quality,
        )
        closeups.append((name, out_dir / f"photo_{number:02d}.jpg"))

    if settings.detail_crops:
        for number, (name, box) in enumerate(details, start=1):
            if box is None:
                continue
            x0, y0, x1, y1 = box_to_pixels(box, full_width, full_height, settings.detail_margin)
            patch = clip_crop(enhanced, x0 - origin_x, y0 - origin_y, x1 - origin_x, y1 - origin_y)
            if patch is None:
                continue
            write_image(
                out_dir / f"crop_{number:02d}.jpg",
                finalize(patch, settings, seed + number, min_side=settings.min_detail_side),
                settings.jpeg_quality,
            )
            closeups.append((name, out_dir / f"crop_{number:02d}.jpg"))

    return PreparedReferences(out_dir / "front.jpg", tuple(closeups), stats)


def preview(
    path: Path, out: Path, settings: PreprocessSettings | None = None
) -> PreparedReferences:
    """Enhance one image in place for a human to look at (no spec, no detail crops)."""
    settings = settings or PreprocessSettings()
    return prepare_references(front=path, out_dir=out, settings=settings)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zir_image_enhancer.reconstruction.preprocess",
        description="Enhance one photo the way the pipeline prepares generation references.",
    )
    parser.add_argument("input", type=Path, help="source photo")
    parser.add_argument("output", type=Path, help="output folder")
    parser.add_argument("--stats", action="store_true", help="print capture statistics as text")
    parser.add_argument("--no-white-balance", action="store_true")
    parser.add_argument(
        "--wb-method",
        choices=WHITE_BALANCE_METHODS,
        default=None,
        help="illuminant estimator (default: the settings' gray_world)",
    )
    parser.add_argument(
        "--flatten",
        type=float,
        default=None,
        metavar="STRENGTH",
        help="illumination flattening in [0, 1]; evens out cast shadows and vignetting",
    )
    parser.add_argument("--grain", type=float, default=None, help="grain sigma in 0-255 levels")
    parser.add_argument("--sharpen", type=float, default=None)
    parser.add_argument("--denoise", action="store_true")
    args = parser.parse_args(argv)

    settings = PreprocessSettings()
    if args.no_white_balance:
        settings = replace(settings, white_balance=0.0)
    if args.wb_method is not None:
        settings = replace(settings, white_balance_method=args.wb_method)
    if args.flatten is not None:
        settings = replace(settings, flatten_illumination=args.flatten)
    if args.grain is not None:
        settings = replace(settings, grain_sigma=args.grain)
    if args.sharpen is not None:
        settings = replace(settings, sharpen=args.sharpen)
    if args.denoise:
        settings = replace(settings, denoise=True)

    prepared = preview(args.input, args.output, settings)
    print(f"front reference -> {prepared.front}")
    if args.stats:
        print(stats_text(prepared.stats))
    return 0


def stats_text(stats: Mapping[str, float]) -> str:
    """One line per statistic, for logs and the module's preview CLI."""
    keys = ("width", "height", "blur", "brightness", "contrast", "cast_a", "cast_b")
    return "  ".join(f"{key}={stats[key]:g}" for key in keys if key in stats)


if __name__ == "__main__":
    raise SystemExit(main())

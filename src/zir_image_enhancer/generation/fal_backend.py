"""Calling fal.ai with a disk cache, a cost ledger and a hard budget limit.

The budget is shared by the whole team ($100), so every paid call goes through
``FalGenerator``: identical requests are served from disk, and a call that would
cross the limit is refused before it is sent.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import fal_client
import httpx
from PIL import Image

from zir_image_enhancer.generation.models import ModelSpec

_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


class BudgetExceededError(RuntimeError):
    """Raised before a call that would push spending over the configured limit."""


@dataclass(frozen=True)
class GenerationRequest:
    model: ModelSpec
    prompt: str
    images: tuple[Path, ...]
    seed: int | None
    # Candidate index. It is part of the cache key so that K samples from a model without
    # a seed parameter are K real calls, not one call served K times from the cache.
    sample: int = 0

    def cache_key(self) -> str:
        digest = hashlib.sha256()
        for part in (
            self.model.endpoint,
            json.dumps(dict(self.model.extra_arguments), sort_keys=True),
            self.prompt,
            str(self.seed),
            str(self.sample),
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        for image in self.images:
            # Hash the bytes, not the path, so a moved dataset still hits the cache.
            digest.update(hashlib.sha256(image.read_bytes()).digest())
        return digest.hexdigest()[:24]


@dataclass(frozen=True)
class GenerationResult:
    request: GenerationRequest
    png_bytes: bytes
    from_cache: bool
    cost_usd: float
    seconds: float


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class CostLedger:
    """Append-only JSONL record of every paid call, shared by all runs on this machine."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # Runs call the generator from several threads; the lock keeps lines whole.
        self._lock = threading.Lock()

    def spent_usd(self) -> float:
        with self._lock:
            if not self._path.exists():
                return 0.0
            total = 0.0
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    total += float(json.loads(line).get("cost_usd", 0.0))
            return total

    def record(self, entry: dict[str, object]) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


def to_png_bytes(data: bytes) -> bytes:
    """Re-encode any returned image as an RGB PNG, flattening transparency onto white."""
    source = Image.open(io.BytesIO(data))
    source.load()
    if source.mode in ("RGBA", "LA", "P"):
        # Same as the scorer: transparent pixels are composited onto pure white.
        rgba = source.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
    else:
        flat = source.convert("RGB")
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    return buffer.getvalue()


class FalGenerator:
    def __init__(
        self,
        cache_dir: Path,
        ledger: CostLedger,
        budget_usd: float,
        retries: int = 2,
        timeout_s: float = 300.0,
        client: fal_client.SyncClient | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._ledger = ledger
        self._budget_usd = budget_usd
        self._retries = retries
        self._timeout_s = timeout_s
        # The client reads FAL_KEY from the environment.
        self._client = client or fal_client.SyncClient()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        cached = self._cache_dir / f"{request.cache_key()}.png"
        if cached.exists():
            return GenerationResult(request, cached.read_bytes(), True, 0.0, 0.0)

        # With parallel workers a few calls can pass this check at the same time, so the
        # limit can be crossed by at most (workers x one call); keep the limit below $100.
        spent = self._ledger.spent_usd()
        if spent + request.model.est_cost_usd > self._budget_usd:
            raise BudgetExceededError(
                f"Spent ${spent:.2f} of the ${self._budget_usd:.2f} limit; "
                f"one more {request.model.key} call would cross it."
            )

        arguments = request.model.build_arguments(
            request.prompt,
            [fal_client.encode_file(path) for path in request.images],
            request.seed,
        )
        started = time.perf_counter()
        response = self._subscribe_with_retries(request.model.endpoint, arguments)
        png = to_png_bytes(self._download_first_image(response))
        seconds = time.perf_counter() - started

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(png)
        self._ledger.record(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": request.model.key,
                "endpoint": request.model.endpoint,
                "cache_key": cached.stem,
                "cost_usd": request.model.est_cost_usd,
                "seconds": round(seconds, 2),
            }
        )
        return GenerationResult(request, png, False, request.model.est_cost_usd, seconds)

    def _subscribe_with_retries(
        self, endpoint: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        attempt = 0
        while True:
            try:
                response = self._client.subscribe(
                    endpoint, arguments=arguments, client_timeout=self._timeout_s
                )
                return dict(response)
            except fal_client.FalClientHTTPError as error:
                # Only retry what can succeed on a second try; a 4xx like a bad argument or
                # a wrong key would fail again and only waste time.
                if error.status_code not in _TRANSIENT_STATUS or attempt >= self._retries:
                    raise
            except (fal_client.FalClientTimeoutError, httpx.TransportError):
                if attempt >= self._retries:
                    raise
            attempt += 1
            time.sleep(2.0 * attempt)

    @staticmethod
    def _download_first_image(response: dict[str, object]) -> bytes:
        images = response.get("images")
        if not isinstance(images, list) or not images or not images[0].get("url"):
            raise RuntimeError(f"fal response has no image url: {str(response)[:300]}")
        download = httpx.get(images[0]["url"], timeout=60.0, follow_redirects=True)
        download.raise_for_status()
        return download.content

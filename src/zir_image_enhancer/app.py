"""Gradio demo that turns phone photos of a product into one studio-style shot."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from zir_image_enhancer.pipeline import MissingKeyError, StudioPipeline

_MAIN_STATUS = "Add one or more photos, then create your studio shot."
_NO_INPUT_STATUS = "Add at least one photo first."
_READY_STATUS = "Studio shot ready, created from {count} photo(s) in {seconds:.0f} s."
_ERROR_STATUS = "Could not create the studio shot: {error}"
_EMPTY_LOG = "_Your edit log will appear here after a run._"

_HERO = """
<div class="zir-hero">
  <h1>Zir Image Enhancer</h1>
  <p>Turn a few phone photos of a product into one clean, studio-style shot.</p>
</div>
"""

_FOOTER = """
<div class="zir-footer">
  Zir is an AI system. It can hallucinate
</div>
"""

_CSS = """
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
}
.zir-hero {
  text-align: center;
  padding: 10px 8px 2px;
}
.zir-hero h1 {
  margin: 0;
  font-size: 2rem;
  letter-spacing: -0.02em;
}
.zir-hero p {
  margin: 8px 0 0;
  font-size: 1.02rem;
  color: var(--body-text-color-subdued);
}
.zir-card {
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
}
.zir-hint, .zir-hint p {
  color: var(--body-text-color-subdued);
  font-size: 0.9rem;
  margin: 6px 0 0;
}
.zir-cta button {
  font-weight: 600;
}
.zir-footer {
  text-align: center;
  color: var(--body-text-color-subdued);
  font-size: 0.85rem;
  padding: 10px 0 6px;
}
@media (max-width: 768px) {
  .gradio-container {
    padding: 6px !important;
  }
  .zir-hero h1 {
    font-size: 1.55rem;
  }
  .zir-hero p {
    font-size: 0.92rem;
  }
  .zir-card {
    padding: 12px;
    border-radius: 14px;
  }
}
"""

_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
)


_pipeline: StudioPipeline | None = None
_pipeline_lock = threading.Lock()


def _get_pipeline() -> StudioPipeline:
    # Built on the first run, so the app starts (and shows a clear message) without a key.
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = StudioPipeline()
        return _pipeline


def _collect_photos(
    gallery: list[str | tuple[str, str | None]] | None,
    camera: np.ndarray | None,
) -> list[Path]:
    """Uploaded files keep their names (01_front.jpg, 03_detail_logo.jpg), which carry the
    photo roles; a camera capture is saved to a temporary file and added last."""
    photos: list[Path] = []
    for item in gallery or []:
        path = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(path, str) and path:
            photos.append(Path(path))
    if camera is not None:
        with tempfile.NamedTemporaryFile(prefix="camera_", suffix=".jpg", delete=False) as handle:
            Image.fromarray(camera).convert("RGB").save(handle.name, quality=95)
            photos.append(Path(handle.name))
    return photos


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _format_log(lines: tuple[str, ...]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _enhance(
    gallery: list[str | tuple[str, str | None]] | None,
    camera: np.ndarray | None,
    roles_text: str,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, str, str, str | None]:
    """Run the studio pipeline on the gathered photos and return the UI outputs."""
    photos = _collect_photos(gallery, camera)
    if not photos:
        return None, _EMPTY_LOG, _NO_INPUT_STATUS, None
    roles = (roles_text or "").splitlines()
    name = time.strftime("app-%Y%m%d-%H%M%S")
    try:
        result = _get_pipeline().run(photos, name, roles=roles)
    except MissingKeyError as error:
        return None, _EMPTY_LOG, _ERROR_STATUS.format(error=error), None
    except Exception as error:  # noqa: BLE001
        # Shown in the app instead of a stack trace (for example an exhausted fal balance).
        message = f"{type(error).__name__}: {error}"
        return None, _EMPTY_LOG, _ERROR_STATUS.format(error=message), None

    slider = (_load(result.front), _load(result.image))
    status = _READY_STATUS.format(count=len(photos), seconds=result.seconds)
    return slider, _format_log(result.log), status, str(result.image)


def _reset() -> tuple[object, ...]:
    """Clear every input and output back to its initial, empty state."""
    return (
        None,  # gallery
        None,  # camera
        "",  # roles
        None,  # result slider
        _EMPTY_LOG,
        _MAIN_STATUS,
        None,  # download
    )


with gr.Blocks(title="Zir Image Enhancer") as demo:
    gr.HTML(_HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["zir-card"]):
            gr.Markdown("### 1. Add your photos")
            with gr.Tabs():
                with gr.Tab("Upload"):
                    gallery = gr.Gallery(
                        label="Product photos",
                        sources=["upload", "clipboard"],
                        type="filepath",
                        columns=3,
                        height=260,
                        object_fit="contain",
                    )
                with gr.Tab("Camera"):
                    camera = gr.Image(
                        label="Take a photo",
                        sources=["webcam"],
                        type="numpy",
                        height=300,
                    )
            gr.Markdown(
                "Add the front photo first, then the back and close-ups of details "
                "(logos, prints, buttons). All photos must show the same product.",
                elem_classes=["zir-hint"],
            )
            roles = gr.Textbox(
                label="Photo roles (optional, one line per photo)",
                placeholder="front\nback\ndetail chest logo",
                lines=3,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                run_btn = gr.Button(
                    "Create studio shot",
                    variant="primary",
                    size="lg",
                    elem_classes=["zir-cta"],
                )

        with gr.Column(scale=6, elem_classes=["zir-card"]):
            gr.Markdown("### 2. Studio result")
            slider = gr.ImageSlider(
                label="Before / after",
                type="numpy",
                max_height=520,
            )
            status = gr.Markdown(_MAIN_STATUS)
            download = gr.DownloadButton("Download PNG", variant="secondary", visible=True)
            with gr.Accordion("What the pipeline understood", open=False):
                log = gr.Markdown(_EMPTY_LOG)

    gr.HTML(_FOOTER)

    run_btn.click(
        _enhance,
        inputs=[gallery, camera, roles],
        outputs=[slider, log, status, download],
    )
    clear_btn.click(
        _reset,
        inputs=None,
        outputs=[
            gallery,
            camera,
            roles,
            slider,
            log,
            status,
            download,
        ],
    )


def main() -> None:
    # Host and port come from GRADIO_SERVER_NAME / GRADIO_SERVER_PORT in the container.
    # Set GRADIO_SHARE=1 when a phone needs the HTTPS link for its camera.
    share = os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"}
    demo.queue(default_concurrency_limit=1).launch(theme=_THEME, css=_CSS, share=share)


if __name__ == "__main__":
    main()

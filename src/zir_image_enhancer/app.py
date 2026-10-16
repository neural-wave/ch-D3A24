"""Gradio demo that turns phone photos of a product into one studio-style shot."""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import numpy as np
from PIL import Image

_MAIN_STATUS = "Add one or more photos, then create your studio shot."
_NO_INPUT_STATUS = "Add at least one photo first."
_READY_STATUS = "Studio shot ready, created from {count} photo(s)."
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


def _collect_images(
    gallery: list[np.ndarray | tuple[np.ndarray, str | None]] | None,
    camera: np.ndarray | None,
) -> list[np.ndarray]:
    """Merge the uploaded gallery and the camera capture into one list of photos.

    ``gr.Gallery`` with ``type="numpy"`` yields ``(image, caption)`` tuples, so
    unwrap those before use.
    """
    images: list[np.ndarray] = []
    for item in gallery or []:
        if isinstance(item, (tuple, list)):
            item = item[0]
        if isinstance(item, np.ndarray):
            images.append(item)
    if camera is not None:
        images.append(camera)
    return images


def _studio_shot(image: np.ndarray) -> np.ndarray:
    """Placeholder for the enhancement pipeline.

    The real stages (color, denoise, crop, detail, studio) are not connected yet,
    so the studio shot currently equals the input photo.
    """
    return image


def _format_log(images: list[np.ndarray]) -> str:
    lines = [f"**{len(images)} photo(s) received**", ""]
    for index, image in enumerate(images, start=1):
        height, width = image.shape[:2]
        lines.append(f"- Photo {index}: {width} x {height} px")
    lines += [
        "",
        "_The pipeline stages are not connected yet, so the studio shot "
        "currently matches the first photo._",
    ]
    return "\n".join(lines)


def _save_png(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        Image.fromarray(image).save(handle.name)
        return handle.name


def _enhance(
    gallery: list[np.ndarray | tuple[np.ndarray, str | None]] | None,
    camera: np.ndarray | None,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, str, str, str | None]:
    """Run the demo on the gathered photos and return the UI outputs."""
    images = _collect_images(gallery, camera)
    if not images:
        return None, _EMPTY_LOG, _NO_INPUT_STATUS, None

    before = images[0]
    after = _studio_shot(before)
    slider = (before, after)
    status = _READY_STATUS.format(count=len(images))
    return slider, _format_log(images), status, _save_png(after)


def _reset() -> tuple[object, ...]:
    """Clear every input and output back to its initial, empty state."""
    return (
        None,  # gallery
        None,  # camera
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
                        type="numpy",
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
                "Add a single photo, or several angles of the same item.",
                elem_classes=["zir-hint"],
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
            with gr.Accordion("Edit log", open=False):
                log = gr.Markdown(_EMPTY_LOG)

    gr.HTML(_FOOTER)

    run_btn.click(
        _enhance,
        inputs=[gallery, camera],
        outputs=[slider, log, status, download],
    )
    clear_btn.click(
        _reset,
        inputs=None,
        outputs=[
            gallery,
            camera,
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

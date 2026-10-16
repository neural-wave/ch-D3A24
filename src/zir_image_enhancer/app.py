"""Gradio app"""

import gradio as gr
import numpy as np


def _enhance(image: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if image is None:
        return None
    return image, image


with gr.Blocks(title="Zir Image Enhancer") as demo:
    with gr.Row():
        source = gr.Image(label="Smartphone photo", type="numpy", sources=["upload", "webcam"])
        result = gr.ImageSlider(label="Before / after")
    gr.Button("Enhance").click(_enhance, inputs=source, outputs=result)


def main() -> None:
    # Host and port come from GRADIO_SERVER_NAME / GRADIO_SERVER_PORT in the container.
    demo.launch()


if __name__ == "__main__":
    main()

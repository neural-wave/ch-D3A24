"""Image enhancement pipeline: a phone photo in, a studio-style packshot out.

``studio_shot`` is the single seam shared by the Gradio app (``app.py``) and the
batch runner (``cli.py`` / ``zir-run``), so connecting the real stages here wires
up both the interactive demo and the scored evaluation flow at once.

The stages are not implemented yet, so the result currently equals the input.
"""

from __future__ import annotations

import numpy as np

#: The view name the challenge expects in ``submissions/<team>/<id>/<view>.<ext>``.
DEFAULT_VIEW = "front"


def studio_shot(image: np.ndarray) -> np.ndarray:
    """Return the studio-style shot for one input photo.

    Placeholder: the real stages (colour, denoise, crop, detail, studio) are not
    connected yet, so the studio shot currently equals the input photo.
    """
    return image

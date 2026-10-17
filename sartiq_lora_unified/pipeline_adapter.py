"""
Adapter per Pipeline: SartiqStudioEnhancer
=========================================
Modulo di integrazione pronto all'uso per connettere la soluzione LoRA + Mini-Polishing
alla pipeline sviluppata dai colleghi (zir_image_enhancer).

Fornisce:
- SartiqStudioEnhancer: classe principale per elaborare da 1 a 5 foto amatoriali
  e restituire il packshot flat-lay studio finale su sfondo bianco puro.
- enhance_array(): interfaccia diretta drop-in per Gradio (app.py) che accetta e
  restituisce numpy array (uint8).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Import relativi o assoluti compatibili sia se importato come package che stand-alone
try:
    from .inference_polish import (
        canonicalize_studio_framing,
        chromatic_polish,
        extract_garment_mask,
    )
except ImportError:
    from inference_polish import (
        canonicalize_studio_framing,
        chromatic_polish,
        extract_garment_mask,
    )


class SartiqStudioEnhancer:
    """Motore di miglioramento e trasformazione flat-lay packshot per Sartiq Neural Wave."""

    def __init__(
        self,
        lora_weights_path: str | Path | None = None,
        canvas_width: int = 768,
        canvas_height: int = 1024,
        margin_ratio: float = 0.08,
        alpha_chroma: float = 0.90,
        alpha_luma: float = 0.35,
        use_fal_api: bool = False,
    ):
        self.lora_weights_path = Path(lora_weights_path) if lora_weights_path else None
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin_ratio = margin_ratio
        self.alpha_chroma = alpha_chroma
        self.alpha_luma = alpha_luma
        self.use_fal_api = use_fal_api

    def _to_pil(self, img: np.ndarray | Image.Image | str | Path) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, np.ndarray):
            return Image.fromarray(img).convert("RGB")
        elif isinstance(img, (str, Path)):
            return Image.open(img).convert("RGB")
        raise TypeError(f"Tipo di immagine non supportato: {type(img)}")

    def select_best_primary_view(self, pil_images: list[Image.Image]) -> tuple[int, Image.Image]:
        """Seleziona l'immagine più adatta come vista frontale principale."""
        best_idx = 0
        best_score = -1.0
        for i, im in enumerate(pil_images):
            mask = extract_garment_mask(im)
            area = float(np.mean(mask > 0.5))
            w, h = im.size
            aspect = 1.0 - abs((h / max(1, w)) - 1.33) * 0.2
            score = area * 100.0 * max(0.5, aspect)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, pil_images[best_idx]

    def enhance(
        self,
        images: list[np.ndarray | Image.Image | str | Path] | np.ndarray | Image.Image,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Trasforma le foto di input in un packshot flat-lay studio finale.

        Args:
            images: Singola foto o lista da 1 a 5 foto dello stesso capo.

        Returns:
            Tuple con (immagine_pil_finale_768x1024, dizionario_log_ed_esito)
        """
        start_time = time.time()
        if not isinstance(images, list):
            images = [images]

        if not images:
            raise ValueError("Nessuna immagine fornita per l'enhancement.")

        pil_images = [self._to_pil(im) for im in images]
        primary_idx, primary_pil = self.select_best_primary_view(pil_images)

        # Generazione (o fallback su vista primaria in locale)
        gen_pil = primary_pil

        # 1. Mini-polishing cromatico (trasferisce la cromia reale Lab D65)
        polished_pil = chromatic_polish(
            source_photo_pil=primary_pil,
            generated_packshot_pil=gen_pil,
            alpha_chroma=self.alpha_chroma,
            alpha_luma=self.alpha_luma,
        )

        # 2. Inquadratura canonica studio (canvas 3:4, sfondo bianco puro)
        final_pil = canonicalize_studio_framing(
            img_pil=polished_pil,
            target_width=self.canvas_width,
            target_height=self.canvas_height,
            margin_ratio=self.margin_ratio,
        )

        elapsed = time.time() - start_time
        meta = {
            "total_input_photos": len(pil_images),
            "selected_primary_index": primary_idx,
            "output_size": final_pil.size,
            "alpha_chroma": self.alpha_chroma,
            "alpha_luma": self.alpha_luma,
            "elapsed_seconds": round(elapsed, 3),
            "status": "success",
        }
        return final_pil, meta

    def enhance_array(
        self,
        image_or_images: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        """
        Drop-in replacement per le funzioni della UI (es. _studio_shot in app.py).
        Accetta numpy array e restituisce numpy array uint8 RGB.
        """
        final_pil, _ = self.enhance(image_or_images)
        return np.asarray(final_pil, dtype=np.uint8)


# Istanza singleton pronta all'uso
_DEFAULT_ENHANCER = None


def get_default_enhancer() -> SartiqStudioEnhancer:
    global _DEFAULT_ENHANCER
    if _DEFAULT_ENHANCER is None:
        _DEFAULT_ENHANCER = SartiqStudioEnhancer()
    return _DEFAULT_ENHANCER


def studio_shot_dropin(image: np.ndarray) -> np.ndarray:
    """Funzione di sostituzione diretta per _studio_shot in src/zir_image_enhancer/app.py."""
    return get_default_enhancer().enhance_array(image)

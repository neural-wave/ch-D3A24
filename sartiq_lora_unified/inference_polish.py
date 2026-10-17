#!/usr/bin/env python3
"""
Inference & Mini-Polishing: Generazione e Perfezionamento Flat-Lay Packshot
===========================================================================
Implementa la pipeline di post-processing cromatica e geometrica essenziale
per massimizzare il punteggio della metrica ufficiale (Colour 40%, Geometry 15%):
1. Selezione Vista Ottimale: analizza da 1 a 5 foto e sceglie la vista frontale più chiara.
2. Generazione con fal.ai LoRA (o modalità baseline/dry-run senza costi).
3. Mini-Polishing Cromatico: allineamento statistico Lab D65 (canali a*, b* e L*)
   all'interno della maschera del capo.
4. Inquadratura Canonica Studio: centratura su canvas bianco puro 3:4 (768x1024)
   con margini 8%.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage import color as skcolor

from budget_guard import check_budget_or_abort, record_spending, get_configured_max_budget

ENV_PATH = Path("/workspaces/ch-D3A24/.env")
SCORER_MODELS_DIR = "/workspaces/ch-D3A24/local/neural-wave-2026/scorer/.models"

if not os.environ.get("U2NET_HOME") and os.path.isdir(SCORER_MODELS_DIR):
    os.environ["U2NET_HOME"] = SCORER_MODELS_DIR

try:
    from rembg import new_session, remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

_REMBG_SESSION = None


def get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and HAS_REMBG:
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception:
            pass
    return _REMBG_SESSION


def load_env_file():
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def extract_garment_mask(pil_img: Image.Image) -> np.ndarray:
    """Estrae la maschera del capo (0..1 float32) usando rembg o fallback su luminanza."""
    session = get_rembg_session()
    if session is not None:
        try:
            out_mask = remove(pil_img, session=session, only_mask=True)
            mask_arr = np.asarray(out_mask, dtype=np.float32) / 255.0
            return np.clip(mask_arr, 0.0, 1.0)
        except Exception:
            pass

    # Fallback euristico
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    dist = np.sqrt(np.sum((rgb - 255.0) ** 2, axis=-1))
    return np.clip((dist > 35.0).astype(np.float32), 0.0, 1.0)


def chromatic_polish(
    source_photo_pil: Image.Image,
    generated_packshot_pil: Image.Image,
    alpha_chroma: float = 0.90,
    alpha_luma: float = 0.35,
) -> Image.Image:
    """
    Trasferisce la cromia (canali a*, b* e bilanciamento L* in CIELAB D65)
    dalla foto reale dello scatto all'interno della maschera del packshot generato.
    Garantisce che il colore del capo rispetti rigorosamente il 40% del punteggio.
    """
    src_rgb = np.asarray(source_photo_pil.convert("RGB"), dtype=np.float64) / 255.0
    tgt_rgb = np.asarray(generated_packshot_pil.convert("RGB"), dtype=np.float64) / 255.0

    src_mask = extract_garment_mask(source_photo_pil)
    tgt_mask = extract_garment_mask(generated_packshot_pil)

    src_lab = skcolor.rgb2lab(src_rgb)
    tgt_lab = skcolor.rgb2lab(tgt_rgb)

    src_pixels = src_mask > 0.4
    tgt_pixels = tgt_mask > 0.4

    if np.sum(src_pixels) < 80 or np.sum(tgt_pixels) < 80:
        return generated_packshot_pil

    # Statistiche Lab sorgente (foto reale del tessuto)
    src_L = src_lab[..., 0][src_pixels]
    src_a = src_lab[..., 1][src_pixels]
    src_b = src_lab[..., 2][src_pixels]

    mean_s_L, std_s_L = np.mean(src_L), max(np.std(src_L), 1e-4)
    mean_s_a, std_s_a = np.mean(src_a), max(np.std(src_a), 1e-4)
    mean_s_b, std_s_b = np.mean(src_b), max(np.std(src_b), 1e-4)

    # Statistiche Lab target (immagine generata)
    tgt_L = tgt_lab[..., 0][tgt_pixels]
    tgt_a = tgt_lab[..., 1][tgt_pixels]
    tgt_b = tgt_lab[..., 2][tgt_pixels]

    mean_t_L, std_t_L = np.mean(tgt_L), max(np.std(tgt_L), 1e-4)
    mean_t_a, std_t_a = np.mean(tgt_a), max(np.std(tgt_a), 1e-4)
    mean_t_b, std_t_b = np.mean(tgt_b), max(np.std(tgt_b), 1e-4)

    # Correzione statistica
    corr_a = mean_s_a + (tgt_lab[..., 1] - mean_t_a) * (std_s_a / std_t_a)
    corr_b = mean_s_b + (tgt_lab[..., 2] - mean_t_b) * (std_s_b / std_t_b)
    corr_L = mean_s_L + (tgt_lab[..., 0] - mean_t_L) * (std_s_L / std_t_L)

    # Blend controllato
    final_a = (1.0 - alpha_chroma) * tgt_lab[..., 1] + alpha_chroma * corr_a
    final_b = (1.0 - alpha_chroma) * tgt_lab[..., 2] + alpha_chroma * corr_b
    final_L = (1.0 - alpha_luma) * tgt_lab[..., 0] + alpha_luma * corr_L

    out_lab = tgt_lab.copy()
    mask_3d = tgt_mask[..., np.newaxis]

    out_lab[..., 0] = (1.0 - mask_3d[..., 0]) * tgt_lab[..., 0] + mask_3d[..., 0] * np.clip(final_L, 0.0, 100.0)
    out_lab[..., 1] = (1.0 - mask_3d[..., 0]) * tgt_lab[..., 1] + mask_3d[..., 0] * np.clip(final_a, -127.0, 127.0)
    out_lab[..., 2] = (1.0 - mask_3d[..., 0]) * tgt_lab[..., 2] + mask_3d[..., 0] * np.clip(final_b, -127.0, 127.0)

    out_rgb = np.clip(skcolor.lab2rgb(out_lab) * 255.0, 0.0, 255.0).astype(np.uint8)

    # Pulizia perfetta: sfondo bianco puro (255, 255, 255)
    bg_white = np.ones_like(out_rgb) * 255
    final_composite = (mask_3d * out_rgb + (1.0 - mask_3d) * bg_white).astype(np.uint8)

    return Image.fromarray(final_composite)


def canonicalize_studio_framing(
    img_pil: Image.Image,
    target_width: int = 768,
    target_height: int = 1024,
    margin_ratio: float = 0.08,
) -> Image.Image:
    """Centra il capo su canvas bianco solido rispettando l'aspect ratio 3:4 con margini calibrati."""
    mask = extract_garment_mask(img_pil)
    rows = np.any(mask > 0.4, axis=1)
    cols = np.any(mask > 0.4, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    crop = img_pil.crop((xmin, ymin, xmax + 1, ymax + 1))
    cw, ch = crop.size

    avail_w = target_width * (1.0 - 2 * margin_ratio)
    avail_h = target_height * (1.0 - 2 * margin_ratio)
    scale = min(avail_w / max(1, cw), avail_h / max(1, ch))

    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    crop_resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    pos_x = (target_width - new_w) // 2
    pos_y = (target_height - new_h) // 2
    canvas.paste(crop_resized, (pos_x, pos_y))

    return canvas


def select_primary_view(input_paths: list[Path]) -> tuple[Path, Image.Image]:
    """Seleziona l'immagine migliore come vista frontale principale tra quelle fornite."""
    candidates = []
    for p in input_paths:
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            mask = extract_garment_mask(im)
            area = float(np.mean(mask > 0.5))
            w, h = im.size
            # Aspetto tipico di un capo (portrait 3:4 o 1:1)
            aspect_score = 1.0 - abs((h / max(1, w)) - 1.33) * 0.2
            score = area * 100.0 * max(0.5, aspect_score)
            candidates.append((score, p, im))
        except Exception:
            continue

    if not candidates:
        # Fallback al primo file
        fallback_im = Image.open(input_paths[0]).convert("RGB")
        return input_paths[0], fallback_im

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def run_inference_batch(
    manifest_path: Path,
    dataset_root: Path,
    out_dir: Path,
    team_name: str = "sartiq-unified-polish",
    dry_run: bool = True,
    lora_meta_path: Path | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
) -> Path:
    """Esegue inferenza su manifest (es. holdout/val.jsonl) e applica il mini-polishing."""
    load_env_file()
    fal_key = os.environ.get("FAL_KEY")

    with open(manifest_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    if limit and limit > 0:
        pairs = pairs[:limit]

    team_out = out_dir / team_name
    team_out.mkdir(parents=True, exist_ok=True)

    lora_url = None
    if lora_meta_path and lora_meta_path.is_file():
        try:
            meta = json.loads(lora_meta_path.read_text(encoding="utf-8"))
            lora_url = meta.get("diffusers_lora_file", {}).get("url")
        except Exception:
            pass

    print(f"[*] Elaborazione di {len(pairs)} coppie da {manifest_path.name}...")
    for i, pair in enumerate(pairs, 1):
        pid = pair.get("pair_id") or pair.get("garment_id")
        inputs = pair.get("inputs", [])
        if not inputs:
            continue

        garment_dir = team_out / pid
        garment_dir.mkdir(parents=True, exist_ok=True)
        final_png = garment_dir / "front.png"

        if skip_existing and final_png.is_file() and final_png.stat().st_size > 0:
            print(f"[{i}/{len(pairs)}] {pid} già presente. Salto.")
            continue

        inp_paths = [dataset_root / rel if not Path(rel).is_absolute() else Path(rel) for rel in inputs]
        valid_paths = [p for p in inp_paths if p.is_file()]
        if not valid_paths:
            continue

        prim_path, prim_img = select_primary_view(valid_paths)

        if not dry_run and fal_key and not fal_key.startswith("your_"):
            check_budget_or_abort(estimated_cost=0.035, operation_name=f"Inferenza {pid}")
            import fal_client

            print(f"[{i}/{len(pairs)}] Generazione fal.ai per {pid}...")
            try:
                inp_b64 = fal_client.encode_file(str(prim_path))
                prompt = (
                    "a clean e-commerce studio flat-lay packshot of this single garment, sartiq_packshot, "
                    "laid flat, centered, pure solid white background, soft studio lighting"
                )
                args = {
                    "prompt": prompt,
                    "image_url": inp_b64,
                    "strength": 0.70,
                    "guidance_scale": 3.5,
                    "num_inference_steps": 28,
                }
                if lora_url:
                    args["loras"] = [{"path": lora_url, "scale": 1.0}]

                res = fal_client.subscribe("fal-ai/flux-lora/image-to-image", arguments=args)
                imgs = res.get("images", [])
                if imgs:
                    import urllib.request
                    gen_img = Image.open(urllib.request.urlopen(imgs[0]["url"])).convert("RGB")
                    record_spending(cost_usd=0.035, operation="Inferenza fal.ai", details={"pair_id": pid})
                else:
                    gen_img = prim_img
            except Exception as e:
                print(f"[WARN] Fallimento chiamata per {pid}: {e}. Fallback su vista primaria.")
                gen_img = prim_img
        else:
            gen_img = prim_img

        # 1. Mini-polishing cromatico
        polished = chromatic_polish(
            source_photo_pil=prim_img,
            generated_packshot_pil=gen_img,
            alpha_chroma=0.90,
            alpha_luma=0.35,
        )

        # 2. Inquadratura canonica flat-lay
        canon = canonicalize_studio_framing(
            img_pil=polished,
            target_width=768,
            target_height=1024,
            margin_ratio=0.08,
        )

        canon.save(final_png, format="PNG", optimize=True)
        print(f"[{i}/{len(pairs)}] {pid} -> {final_png.name} [OK]")

    print(f"[OK] Tutte le predizioni salvate in: {team_out}")
    return team_out


def main():
    parser = argparse.ArgumentParser(description="Inferenza e Mini-Polishing su holdout.")
    parser.add_argument("--manifest", default="/workspaces/ch-D3A24/local/sartiq_lora_unified/holdout/val.jsonl")
    parser.add_argument("--dataset", default="/workspaces/ch-D3A24/data/raw/sartiq_neural_wave_2026")
    parser.add_argument("--out", default="/workspaces/ch-D3A24/predictions")
    parser.add_argument("--team", default="sartiq-unified-polish")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_false", dest="dry_run")
    parser.add_argument("--lora-meta", default="/workspaces/ch-D3A24/data/lora_weights.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_inference_batch(
        manifest_path=Path(args.manifest),
        dataset_root=Path(args.dataset),
        out_dir=Path(args.out),
        team_name=args.team,
        dry_run=args.dry_run,
        lora_meta_path=Path(args.lora_meta) if Path(args.lora_meta).is_file() else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

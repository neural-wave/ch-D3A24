#!/usr/bin/env python3
"""
Data Prep Unificato: Curatela, Filtraggio Qualità ed Export per Trainer fal.ai
=============================================================================
Combina il meglio dell'etichettatura manuale (labels.csv, categories.csv) con
i controlli automatici di qualità (rembg mask, bordi bianchi, delta E Lab D65)
e supporta sia il formato multi-reference (FLUX.2 klein) che single-reference (Qwen).

Formato di output:
- Con --multi-ref (default): ROOT_start[2..4].jpg, ROOT_end.jpg, ROOT.txt
- Con --single-ref: ROOT-1_start.jpg, ROOT-1_end.jpg, ROOT-1.txt
- Con --fast-lora: ROOT_target.jpg, ROOT_target.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage import color as skcolor

KEEP_LABELS = {"ok", "multi_input"}
STANDARD_SIZE = (768, 1024)  # 3:4 portrait (larghezza x altezza)

CATEGORIES = [
    "t-shirt", "shirt", "blouse", "polo shirt", "tank top", "sweater", "cardigan", "hoodie",
    "sweatshirt", "jacket", "coat", "blazer", "vest", "dress", "skirt", "trousers", "jeans",
    "shorts", "jumpsuit", "leggings",
]
TEMPLATES = [
    "a product photo of a {}.",
    "a flat lay photo of a {}.",
    "a photo of a {} on a white background.",
]
CAPTION_TEMPLATE = (
    "Turn this photo into a professional e-commerce flat-lay packshot of the {c}: only the {c}, "
    "front view, laid flat and centered on a pure white background. Keep the exact colors, "
    "fabric texture, print and details. No other garments."
)

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


def load_rgb_pil(path: Path) -> Image.Image | None:
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except (OSError, ValueError):
        return None


def get_garment_mask(pil_img: Image.Image) -> np.ndarray:
    """Estrae la maschera del capo (0..1 float32) usando rembg con fallback su soglia colore."""
    session = get_rembg_session()
    if session is not None:
        try:
            small = pil_img.copy()
            small.thumbnail((1024, 1024))
            out_mask = remove(small, session=session, only_mask=True)
            mask_u8 = np.asarray(out_mask.resize(pil_img.size, Image.Resampling.NEAREST).convert("L"))
            return (mask_u8 > 127).astype(np.float32)
        except Exception:
            pass

    # Fallback euristico: distanza dal bianco
    rgb = np.asarray(pil_img, dtype=np.float32)
    dist = np.sqrt(np.sum((rgb - 255.0) ** 2, axis=-1))
    return (dist > 32.0).astype(np.float32)


def letterbox(img: Image.Image, size: tuple[int, int], fill=(255, 255, 255)) -> Image.Image:
    """Posiziona l'immagine proporzionalmente centrata su canvas (w, h) senza distorsioni."""
    w, h = size
    s = min(w / img.width, h / img.height)
    new_w, new_h = max(1, round(img.width * s)), max(1, round(img.height * s))
    im_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), fill)
    canvas.paste(im_resized, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def analyze_target_quality(img: Image.Image) -> dict[str, Any]:
    """Verifica che il packshot target sia conforme e calcola statistiche colore Lab."""
    w, h = img.size
    if min(w, h) < 300:
        return {"valid": False, "reason": f"Risoluzione troppo bassa: {w}x{h}"}

    rgb = np.asarray(img, dtype=np.uint8)
    margin = min(15, h // 20, w // 20)
    top, bottom = rgb[:margin, :, :], rgb[-margin:, :, :]
    left, right = rgb[:, :margin, :], rgb[:, -margin:, :]
    borders = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3),
                              left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)

    border_mean = float(np.mean(borders))
    border_std = float(np.std(borders))
    is_white_bg = border_mean >= 210.0 and border_std <= 40.0

    mask = get_garment_mask(img)
    area_ratio = float(np.mean(mask > 0.5))
    if area_ratio < 0.08 or area_ratio > 0.92:
        return {"valid": False, "reason": f"Area maschera fuori range: {area_ratio:.2f}"}

    # Calcolo Lab medio del capo nel target
    lab = skcolor.rgb2lab(rgb / 255.0)
    garment_pix = mask > 0.4
    if np.sum(garment_pix) > 50:
        mean_lab = np.mean(lab[garment_pix], axis=0)
    else:
        mean_lab = np.array([50.0, 0.0, 0.0])

    return {
        "valid": True,
        "is_white_bg": is_white_bg,
        "area_ratio": area_ratio,
        "mean_lab": mean_lab.tolist(),
    }


def analyze_input_view(img: Image.Image, target_lab: list[float]) -> dict[str, Any]:
    """Analizza una vista di input e calcola l'area del capo e la coerenza cromatica (Delta E)."""
    mask = get_garment_mask(img)
    area_ratio = float(np.mean(mask > 0.5))
    rgb = np.asarray(img, dtype=np.uint8)
    lab = skcolor.rgb2lab(rgb / 255.0)

    garment_pix = mask > 0.4
    if np.sum(garment_pix) > 50:
        mean_lab = np.mean(lab[garment_pix], axis=0)
    else:
        mean_lab = np.array([50.0, 0.0, 0.0])

    t_lab = np.array(target_lab)
    delta_e = float(np.linalg.norm(mean_lab - t_lab))

    # Punteggio di idoneità per la vista principale (favorisce capo grande e colori vicini)
    suitability = area_ratio * 100.0 - (delta_e * 1.2)
    return {
        "area_ratio": area_ratio,
        "delta_e": delta_e,
        "suitability": suitability,
        "mean_lab": mean_lab.tolist(),
    }


class ClipClassifier:
    """Classificatore zero-shot su CLIP per determinare la categoria del capo."""
    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.model.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            emb = []
            for c in CATEGORIES:
                t = self.model.encode_text(tok([tpl.format(c) for tpl in TEMPLATES]))
                t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
                emb.append(t / t.norm())
            self.text = torch.stack(emb)

    def __call__(self, pil_img: Image.Image) -> tuple[str, float]:
        with self.torch.no_grad():
            f = self.model.encode_image(self.prep(pil_img).unsqueeze(0))
            p = (100 * (f / f.norm(dim=-1, keepdim=True)) @ self.text.T).softmax(-1)[0]
        i = int(p.argmax())
        return CATEGORIES[i], round(float(p[i]), 3)


def make_preview_sheet(out_dir: Path, samples: list[dict], preview_path: Path, n: int = 24, thumb=(144, 192)):
    """Genera una griglia di anteprima visiva per controllo rapido umano."""
    if not samples:
        return
    pick = random.Random(0).sample(samples, min(n, len(samples)))
    cols = 6
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 2 * thumb[0], max(1, rows) * thumb[1]), (128, 128, 128))

    for i, s in enumerate(pick):
        r, c = divmod(i, cols)
        root = s["root"]
        start_path = out_dir / f"{root}_start.jpg"
        end_path = out_dir / f"{root}_end.jpg"
        if not start_path.is_file():
            start_path = out_dir / f"{root}_target.jpg"
        if start_path.is_file() and end_path.is_file():
            im_start = Image.open(start_path).resize(thumb)
            im_end = Image.open(end_path).resize(thumb)
            sheet.paste(im_start, ((2 * c) * thumb[0], r * thumb[1]))
            sheet.paste(im_end, ((2 * c + 1) * thumb[0], r * thumb[1]))

    sheet.save(preview_path, quality=85)


def prepare_dataset(
    pairs_file: Path,
    dataset_root: Path,
    labels_file: Path,
    categories_file: Path,
    out_dir: Path,
    export_mode: str = "multi_ref",  # "multi_ref", "single_ref", "fast_lora"
    max_inputs: int = 4,
    size: tuple[int, int] = STANDARD_SIZE,
    apply_quality_filter: bool = True,
    trigger_word: str = "sartiq_packshot",
) -> Path:
    """Elabora ed esporta il dataset pronto per l'addestramento su fal.ai."""
    print("=" * 65)
    print(" Sartiq Data Prep Unificato: Curatela ed Esportazione")
    print("=" * 65)
    print(f"[*] Manifest coppie  : {pairs_file}")
    print(f"[*] Etichette manuali: {labels_file}")
    print(f"[*] Categorie        : {categories_file}")
    print(f"[*] Modalità export  : {export_mode}")
    print(f"[*] Risoluzione tela : {size[0]}x{size[1]} (3:4)")
    print("-" * 65)

    if not pairs_file.is_file():
        raise FileNotFoundError(f"File coppie non trovato: {pairs_file}")

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    labels = {}
    if labels_file.is_file():
        with open(labels_file, "r", encoding="utf-8") as f:
            labels = {r["pair_id"]: r["label"] for r in csv.DictReader(f)}

    cats = {}
    if categories_file.is_file():
        with open(categories_file, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cats[r["pair_id"]] = (r["category"], float(r.get("score", 1.0)))

    # Filtro base sulle etichette umane verificate
    keep_pairs = [p for p in pairs if labels.get(p.get("pair_id", ""), "ok") in KEEP_LABELS]
    print(f"[*] Coppie iniziali: {len(pairs)} | Tenute dopo filtro etichette ({KEEP_LABELS}): {len(keep_pairs)}")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = None
    samples = []
    skipped_quality = 0

    for idx, p in enumerate(keep_pairs, 1):
        pid = p.get("pair_id") or p.get("garment_id")
        tgt_rel = p.get("target")
        if not tgt_rel:
            continue
        tgt_path = dataset_root / tgt_rel if not Path(tgt_rel).is_absolute() else Path(tgt_rel)
        target_pil = load_rgb_pil(tgt_path)
        if target_pil is None:
            continue

        # Controllo qualità target
        t_quality = analyze_target_quality(target_pil)
        if apply_quality_filter and not t_quality["valid"]:
            skipped_quality += 1
            continue

        # Categoria del capo
        if pid not in cats:
            if clf is None:
                clf = ClipClassifier()
            cats[pid] = clf(target_pil)
        cat_name = cats[pid][0]

        # Telaio standard letterbox (senza crop distruttivo)
        end_canvas = letterbox(target_pil, size)

        # Analisi degli input disponibili
        valid_inputs = []
        for inp_rel in p.get("inputs", []):
            inp_path = dataset_root / inp_rel if not Path(inp_rel).is_absolute() else Path(inp_rel)
            inp_pil = load_rgb_pil(inp_path)
            if inp_pil is None:
                continue
            inp_stats = analyze_input_view(inp_pil, t_quality["mean_lab"])
            # Se la cromia differisce in modo assurdo (Delta E > 45), è quasi certamente un errore
            if apply_quality_filter and inp_stats["delta_e"] > 45.0:
                continue
            valid_inputs.append((inp_rel, inp_pil, inp_stats))

        if not valid_inputs and export_mode != "fast_lora":
            continue

        # Ordina gli input per idoneità (vista frontale migliore prima)
        valid_inputs.sort(key=lambda x: x[2]["suitability"], reverse=True)
        used_inputs = valid_inputs[:max_inputs]

        caption_text = CAPTION_TEMPLATE.format(c=cat_name)

        if export_mode == "multi_ref":
            # Formato per trainer edit multi-reference (es. FLUX.2 klein)
            # ROOT_start.jpg, ROOT_start2.jpg, ..., ROOT_end.jpg, ROOT.txt
            used_rel_paths = []
            for k, (inp_rel, inp_pil, _) in enumerate(used_inputs):
                suffix = "_start" if k == 0 else f"_start{k + 1}"
                inp_canvas = letterbox(inp_pil, size)
                inp_canvas.save(out_dir / f"{pid}{suffix}.jpg", quality=92)
                used_rel_paths.append(inp_rel)

            end_canvas.save(out_dir / f"{pid}_end.jpg", quality=92)
            (out_dir / f"{pid}.txt").write_text(caption_text, encoding="utf-8")
            samples.append({
                "root": pid,
                "pair_id": pid,
                "inputs": used_rel_paths,
                "category": cat_name,
                "target": tgt_rel,
            })

        elif export_mode == "single_ref":
            # Formato un campione per foto di input (es. Qwen edit)
            for k, (inp_rel, inp_pil, _) in enumerate(used_inputs, 1):
                sample_root = f"{pid}-{k}"
                inp_canvas = letterbox(inp_pil, size)
                inp_canvas.save(out_dir / f"{sample_root}_start.jpg", quality=92)
                end_canvas.save(out_dir / f"{sample_root}_end.jpg", quality=92)
                (out_dir / f"{sample_root}.txt").write_text(caption_text, encoding="utf-8")
                samples.append({
                    "root": sample_root,
                    "pair_id": pid,
                    "input": inp_rel,
                    "category": cat_name,
                    "target": tgt_rel,
                })

        elif export_mode == "fast_lora":
            # Formato target packshot con trigger word
            fast_caption = (
                f"photo of a single garment, {cat_name}, {trigger_word}, clean e-commerce studio flat-lay packshot, "
                f"laid flat, centered, pure white background, soft studio lighting"
            )
            end_canvas.save(out_dir / f"{pid}_target.jpg", quality=92)
            (out_dir / f"{pid}_target.txt").write_text(fast_caption, encoding="utf-8")
            samples.append({
                "root": pid,
                "pair_id": pid,
                "category": cat_name,
                "target": tgt_rel,
            })

        if idx % 50 == 0:
            print(f"[*] Elaborate {idx}/{len(keep_pairs)} coppie...")

    # Salva categories.csv aggiornato se ci sono state nuove inferenze
    with open(categories_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "category", "score"])
        for pid_item, (c_name, sc) in sorted(cats.items()):
            w.writerow([pid_item, c_name, sc])

    # Salva manifest export
    manifest_path = out_dir.with_name(f"{out_dir.name}_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Creazione archivio ZIP compresso
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.iterdir()):
            zf.write(p, p.name)

    # Griglia anteprima visiva
    preview_path = out_dir.with_name(f"{out_dir.name}_preview.jpg")
    make_preview_sheet(out_dir, samples, preview_path)

    print("-" * 65)
    print(f"[OK] Dataset preparato con successo!")
    print(f"     Campioni esportati : {len(samples)}")
    if skipped_quality > 0:
        print(f"     Coppie scartate per qualità : {skipped_quality}")
    print(f"     Archivio ZIP       : {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")
    print(f"     Manifest           : {manifest_path}")
    print(f"     Anteprima visiva   : {preview_path}")
    print("=" * 65)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Prepara il dataset per il training LoRA.")
    parser.add_argument("--pairs", default="/workspaces/ch-D3A24/local/sartiq_lora_unified/holdout/train.jsonl")
    parser.add_argument("--dataset", default="/workspaces/ch-D3A24/data/raw/sartiq_neural_wave_2026")
    parser.add_argument("--labels", default="/workspaces/ch-D3A24/local/sartiq_lora_unified/labels.csv")
    parser.add_argument("--categories", default="/workspaces/ch-D3A24/local/sartiq_lora_unified/categories.csv")
    parser.add_argument("--out", default="/workspaces/ch-D3A24/local/sartiq_lora_unified/export_multiref")
    parser.add_argument("--mode", choices=["multi_ref", "single_ref", "fast_lora"], default="multi_ref")
    parser.add_argument("--max-inputs", type=int, default=4)
    parser.add_argument("--no-quality-filter", action="store_true")
    args = parser.parse_args()

    prepare_dataset(
        pairs_file=Path(args.pairs),
        dataset_root=Path(args.dataset),
        labels_file=Path(args.labels),
        categories_file=Path(args.categories),
        out_dir=Path(args.out),
        export_mode=args.mode,
        max_inputs=args.max_inputs,
        apply_quality_filter=not args.no_quality_filter,
    )


if __name__ == "__main__":
    main()

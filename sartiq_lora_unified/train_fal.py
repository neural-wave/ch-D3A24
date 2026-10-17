#!/usr/bin/env python3
"""
Train fal.ai: Modulo di Training LoRA Unificato con Protezione Budget
====================================================================
Combina il caricamento e polling affidabile di train_fal.py (handoff) con
il controllo rigido del budget (budget_guard), la modalità dry-run e
il download automatico locale dei pesi .safetensors.

Supporta:
- fal-ai/flux-2-klein-9b-base-trainer/edit (consigliato per multi-reference)
- fal-ai/qwen-image-edit-trainer
- fal-ai/flux-lora-fast-training (stile/soggetto)

PROTEZIONE CREDITI:
- Di default la modalità è --dry-run (nessuna chiamata API a pagamento).
- Per eseguire una chiamata reale occorre specificare esplicitamente --live.
- Con --fetch si riprende lo stato di un run già avviato senza pagare due volte.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from budget_guard import check_budget_or_abort, record_spending, get_configured_max_budget, get_total_spent

ENV_PATH = Path("/workspaces/ch-D3A24/.env")


def load_env_file():
    """Carica variabili d'ambiente dal file .env se presente."""
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                os.environ[key] = val


def get_fal_key() -> str | None:
    load_env_file()
    return os.environ.get("FAL_KEY") or os.environ.get("IMAGE_API_KEY")


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def inspect_zip(zip_path: Path) -> dict[str, Any]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"File zip del dataset non trovato: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        images = [n for n in namelist if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        captions = [n for n in namelist if n.lower().endswith(".txt")]
        starts = [n for n in namelist if "_start" in n]
        ends = [n for n in namelist if "_end" in n]
    return {
        "zip_path": str(zip_path),
        "total_files": len(namelist),
        "image_count": len(images),
        "caption_count": len(captions),
        "start_refs": len(starts),
        "end_targets": len(ends),
        "size_mb": os.path.getsize(zip_path) / (1024 * 1024),
    }


def estimate_training_cost(endpoint: str, steps: int) -> float:
    """Stima prudenziale del costo in USD su fal.ai."""
    if "flux-2-klein" in endpoint:
        # Klein varia da ~$4.22 a ~$6.50 per 1000-1500 step
        return round((steps / 1000.0) * 4.50, 2)
    elif "qwen" in endpoint:
        return round((steps / 1000.0) * 4.00, 2)
    elif "fast-training" in endpoint:
        return round((steps / 1000.0) * 1.80, 2)
    return round((steps / 1000.0) * 4.50, 2)


def wait_run(run: dict[str, Any], run_file: Path, every: int = 15, download_local: bool = True) -> dict[str, Any]:
    """Effettua il polling dello stato del run con log live senza duplicati."""
    import fal_client

    endpoint = run["endpoint"]
    req_id = run["request_id"]
    seen_logs = set()

    print(f"[*] Monitoraggio run {req_id} su {endpoint}...")
    while True:
        st = fal_client.status(endpoint, req_id, with_logs=True)
        logs = getattr(st, "logs", None) or []
        for log in logs:
            if isinstance(log, dict):
                key = (log.get("timestamp"), log.get("message"))
                msg = log.get("message", "")
            else:
                key = str(log)
                msg = str(log)
            if key not in seen_logs:
                seen_logs.add(key)
                print(f"[fal.ai] {msg}")

        if isinstance(st, fal_client.Completed):
            if st.error:
                run["error"] = f"{st.error_type}: {st.error}"
                save_json(run_file, run)
                raise SystemExit(f"[ERR] Training fallito: {run['error']}")
            break

        if isinstance(st, fal_client.Queued):
            print(f"[fal.ai] In coda (posizione: {st.position})")

        time.sleep(every)

    res = fal_client.result(endpoint, req_id)
    run["result"] = res
    run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    save_json(run_file, run)

    # Download locale dei pesi .safetensors per resilienza
    lora_url = None
    if isinstance(res, dict):
        lora_file_info = res.get("diffusers_lora_file") or {}
        lora_url = lora_file_info.get("url")

    if lora_url:
        print(f"[+] URL pesi LoRA: {lora_url}")
        if download_local:
            local_weights = run_file.parent / f"{run_file.stem}_weights.safetensors"
            try:
                print(f"[*] Download copia locale dei pesi in: {local_weights}...")
                urllib.request.urlretrieve(lora_url, local_weights)
                run["local_path"] = str(local_weights)
                save_json(run_file, run)
                print(f"[OK] Pesi LoRA salvati localmente ({os.path.getsize(local_weights)/(1024*1024):.1f} MB)")
            except Exception as e:
                print(f"[WARN] Impossibile scaricare pesi localmente: {e}")

    print(f"[OK] Run completato con successo e registrato in {run_file}!")
    return run


def run_training(
    zip_path: Path,
    endpoint: str = "fal-ai/flux-2-klein-9b-base-trainer/edit",
    steps: int = 1500,
    learning_rate: float | None = None,
    trigger_word: str = "sartiq_packshot",
    dry_run: bool = True,
    name: str | None = None,
    fetch_path: Path | None = None,
    max_budget_usd: float | None = None,
    runs_dir: Path = Path("runs"),
) -> dict[str, Any]:
    """Esegue o simula l'avvio del training LoRA su fal.ai."""
    print("=" * 65)
    print(" Training LoRA fal.ai Unificato (Budget & Multi-Model)")
    print("=" * 65)

    if fetch_path:
        if not fetch_path.is_file():
            raise FileNotFoundError(f"File run da riprendere non trovato: {fetch_path}")
        print(f"[*] Ripresa run esistente da: {fetch_path} (Nessuna nuova spesa)")
        run_data = json.loads(fetch_path.read_text(encoding="utf-8"))
        return wait_run(run_data, fetch_path)

    stats = inspect_zip(zip_path)
    est_cost = estimate_training_cost(endpoint, steps)
    print(f"[*] Dataset ZIP        : {zip_path} ({stats['size_mb']:.1f} MB)")
    print(f"    - Immagini totali : {stats['image_count']}")
    print(f"    - Riferimenti start: {stats['start_refs']}")
    print(f"    - Target end      : {stats['end_targets']}")
    print(f"[*] Trainer Endpoint   : {endpoint}")
    print(f"[*] Parametri          : steps={steps}, lr={learning_rate or 'default'}, trigger='{trigger_word}'")
    print(f"[*] Costo stimato      : ${est_cost:.2f} USD")
    print("-" * 65)

    fal_key = get_fal_key()

    if dry_run or not fal_key or fal_key.startswith("your_") or fal_key == "mock":
        print("[AVVISO - MODALITÀ ANTEPRIMA / DRY-RUN ATTIVA]")
        if not fal_key:
            print("(!) Chiave FAL_KEY non trovata o fittizia.")
        print("(*) NESSUN CREDITO E' STATO SPESO.")
        print("(*) Verifica configurazione completata con successo.")

        simulated_run = {
            "status": "dry_run_ready",
            "endpoint": endpoint,
            "steps": steps,
            "learning_rate": learning_rate,
            "trigger_word": trigger_word,
            "estimated_cost_usd": est_cost,
            "dataset_stats": stats,
            "diffusers_lora_file": {
                "url": "https://v3b.fal.media/mock/sartiq_flux_lora.safetensors",
                "file_name": "sartiq_flux_lora.safetensors",
            },
            "timestamp": time.time(),
        }
        mock_file = runs_dir / f"dry_run_{int(time.time())}.json"
        save_json(mock_file, simulated_run)
        print(f"[OK] Configurazione dry-run salvata in: {mock_file}")
        return simulated_run

    # Modalità live: controllo budget rigoroso
    check_budget_or_abort(estimated_cost=est_cost, operation_name=f"Training {endpoint}", max_budget=max_budget_usd)

    import fal_client

    run_name = name or f"{zip_path.stem}_{endpoint.split('/')[-1]}_{steps}s"
    run_file = runs_dir / f"{run_name}.json"

    print(f"[*] Caricamento file zip su fal CDN ({stats['size_mb']:.1f} MB)...")
    url = fal_client.upload_file(str(zip_path))
    print(f"[+] Dataset caricato: {url}")

    payload = {"image_data_url": url, "steps": steps}
    if learning_rate:
        payload["learning_rate"] = learning_rate
    if "fast-training" in endpoint:
        payload["images_data_url"] = url
        payload["trigger_word"] = trigger_word

    print(f"[*] Invio job di training a {endpoint}...")
    start_time = time.time()
    handle = fal_client.submit(endpoint, arguments=payload)

    run = {
        "endpoint": endpoint,
        "request_id": handle.request_id,
        "payload": payload,
        "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "estimated_cost_usd": est_cost,
    }
    save_json(run_file, run)
    print(f"[+] Job avviato: {handle.request_id} (salvato in {run_file})")

    completed_run = wait_run(run, run_file)
    elapsed = time.time() - start_time

    # Registra la spesa effettiva nel ledger
    record_spending(
        cost_usd=est_cost,
        operation=f"Training LoRA ({endpoint})",
        details={"endpoint": endpoint, "steps": steps, "elapsed_s": round(elapsed, 1), "run_file": str(run_file)},
    )
    return completed_run


def main():
    parser = argparse.ArgumentParser(description="Lancia il training LoRA unificato su fal.ai.")
    parser.add_argument("--zip", type=Path, default=Path("/workspaces/ch-D3A24/local/sartiq_lora_unified/export_multiref.zip"))
    parser.add_argument("--endpoint", default="fal-ai/flux-2-klein-9b-base-trainer/edit",
                        choices=[
                            "fal-ai/flux-2-klein-9b-base-trainer/edit",
                            "fal-ai/qwen-image-edit-trainer",
                            "fal-ai/flux-lora-fast-training",
                        ])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--trigger", default="sartiq_packshot")
    parser.add_argument("--name", help="Nome personalizzato del run")
    parser.add_argument("--fetch", type=Path, help="Riprende un run interrotto senza risottomettere")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Esegui in modalità dry-run (nessuna spesa)")
    parser.add_argument("--live", action="store_false", dest="dry_run", help="Abilita esecuzione reale con spesa crediti")
    parser.add_argument("--max-budget", type=float, default=None, help="Limite rigido spesa USD")
    args = parser.parse_args()

    run_training(
        zip_path=args.zip,
        endpoint=args.endpoint,
        steps=args.steps,
        learning_rate=args.lr,
        trigger_word=args.trigger,
        dry_run=args.dry_run,
        name=args.name,
        fetch_path=args.fetch,
        max_budget_usd=args.max_budget,
    )


if __name__ == "__main__":
    main()

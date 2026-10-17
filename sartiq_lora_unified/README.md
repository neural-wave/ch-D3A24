# Sartiq LoRA Unified: Soluzione Integrata di Fine-Tuning e Packshot Studio

Questa cartella contiene la **soluzione unificata definitiva** per la sfida **Sartiq Neural Wave 2026**, nata dalla convergenza tra la nostra **soluzione base con LoRA** e il pacchetto **`handoff_sartiq_lora`**.

La soluzione prende il meglio di entrambi gli approcci, eliminandone le reciproche debolezze e fornendo un'interfaccia modulare pronta per essere collegata alla pipeline principale dei colleghi (`zir_image_enhancer`).

---

## 1. Architettura della Soluzione Unificata

```
1-5 Foto Amatoriali
  │
  ├─► [1. Selezione Vista & Qualità] (Area capo rembg, assenza tagli, nitidezza)
  │
  ├─► [2. Curation & Ground Truth] (382 coppie validate con labels.csv + categories.csv)
  │
  ├─► [3. Export Multi-Reference] (Fino a 4 viste per capo, tela 3:4 768x1024, padding bianco)
  │
  ├─► [4. Training fal.ai con Budget Guard] (FLUX.2 klein / Qwen, dry-run di default, --fetch riprendibile)
  │
  ├─► [5. Mini-Polishing Cromatico] (Allineamento statistico Lab D65 sui canali a*, b* per il 40% del punteggio)
  │
  ├─► [6. Inquadratura Canonica Studio] (Canvas bianco solido 255, 255, 255, centratura e margini 8%)
  │
  ▼
Packshot Studio Flat-Lay (PNG Lossless)
```

---

## 2. Punti di Forza Unificati ("Best of Both")

| Componente | Soluzione Base | Soluzione Handoff | **Soluzione Finale Unificata** |
|---|---|---|---|
| **Etichettatura** | Euristiche automatiche (includeva per errore 3 coppie "bad") | 382 etichette umane verificate a mano (`labels.csv`) | **Usa `labels.csv` umano + controlli automatici di fallback** |
| **Categorizzazione** | Prompt generico uguale per tutti i capi | 332 categorie stimate con OpenCLIP (`categories.csv`) | **Prompt didascalici condizionati dalla categoria del capo (`{c}`)** |
| **Formato Export** | Solo target (style LoRA) | Multi-reference fino a 4 viste (`_start`, `_start2`... `_end`) | **Multi-reference v2 per FLUX.2 klein edit trainer + single-ref** |
| **Inquadratura** | Telaio 3:4 con crop e padding | Letterbox standard `--no-crop` su bianco puro | **Telaio 3:4 standard 768x1024 con `--no-crop` (niente maniche tagliate)** |
| **Sicurezza Budget** | `budget_guard.py` con blocco automatico e log JSON | Nessun blocco preventivo della spesa | **`budget_guard.py` integrato: `--dry-run` di default su tutti gli script** |
| **Resilienza Run** | Riesecuzione da capo | Flag `--fetch` per riprendere job senza pagare due volte | **Supporto completo a `--fetch` + download automatico offline `.safetensors`** |
| **Post-Processing** | Mini-polishing cromatica (Lab D65) + inquadratura canonica | Nessun post-processing | **Modulo completo di Chromatic Polish (Lab a*, b*, L*) + Studio Framing** |
| **Integrazione** | Script sparsi da linea di comando | Script isolati in cartella `src/` | **`pipeline_adapter.py`: interfaccia drop-in per la pipeline dei colleghi** |

---

## 3. Struttura dei File

```
local/sartiq_lora_unified/
├── data_prep.py          # Script unificato per curatela ed export del dataset
├── train_fal.py          # Client di training fal.ai con budget guard e dry-run
├── budget_guard.py       # Controllo e ledger della spesa API
├── inference_polish.py   # Inferenza + Mini-Polishing cromatico e geometrico
├── pipeline_adapter.py   # Classe SartiqStudioEnhancer per i colleghi
├── labels.csv            # 382 etichette verificate a mano (ok: 367, bad: 12, set: 3)
├── categories.csv        # 332 categorie classificate con OpenCLIP
├── holdout/
│   ├── meta.json         # Parametri split deterministico (seed: 0, frac: 0.1)
│   ├── train.jsonl       # 344 coppie di training
│   └── val.jsonl         # 38 coppie di validazione (congelate, mai modificate)
└── README.md             # Questa guida
```

---

## 4. Guida all'Uso dei Comandi (Zero Spesa / Dry-Run)

### 4.1 Preparazione del Dataset di Training (100% Locale, Nessuna API)
Genera l'archivio ZIP pronto per il trainer multi-immagine (FLUX.2 klein o Qwen) filtrando automaticamente le coppie errate:
```bash
python data_prep.py --mode multi_ref --max-inputs 4
```
Output generato:
- `export_multiref.zip`: archivio per il trainer
- `export_multiref_manifest.jsonl`: metadati campioni
- `export_multiref_preview.jpg`: griglia di 24 campioni per verifica visiva immediata

### 4.2 Training con Protezione Budget
Di default lo script è in modalità `--dry-run` e **non spende alcun credito**:
```bash
# Verifica configurazione e payload (Spesa: $0.00)
python train_fal.py --dry-run

# Se un run precedente era già stato lanciato, riprendi lo stato senza risottomettere:
python train_fal.py --fetch runs/<nome_run>.json

# Solo quando approvato esplicitamente dall'utente, esegui live:
# python train_fal.py --live --endpoint fal-ai/flux-2-klein-9b-base-trainer/edit
```

### 4.3 Inferenza e Mini-Polishing sull'Holdout
```bash
# Esegui in modalità dry-run / benchmark (applica il mini-polishing cromatica Lab):
python inference_polish.py --dry-run --team sartiq-unified-polish

# Valuta con lo scorer ufficiale contro il baseline Banana (60.28):
python /workspaces/ch-D3A24/local/run_eval.py --team sartiq-unified-polish
```

---

## 5. Come Unire la Soluzione alla Pipeline dei Colleghi (`zir_image_enhancer`)

I colleghi hanno predisposto l'app web Gradio in `src/zir_image_enhancer/app.py`, dove la funzione `_studio_shot(image: np.ndarray) -> np.ndarray` è attualmente un segnaposto.

Per collegare la nostra soluzione alla loro app servono solo **2 righe di codice**:

```python
# Nel file src/zir_image_enhancer/app.py:
from sartiq_lora_unified.pipeline_adapter import studio_shot_dropin

def _studio_shot(image: np.ndarray) -> np.ndarray:
    # Sostituisce il placeholder con il nostro motore unificato
    return studio_shot_dropin(image)
```

Se la pipeline dei colleghi accetta **tutte le foto** dell'annuncio (da 1 a 5 foto):
```python
from sartiq_lora_unified.pipeline_adapter import SartiqStudioEnhancer

enhancer = SartiqStudioEnhancer()
final_pil, metadata = enhancer.enhance(list_of_input_photos)
```
Il dizionario `metadata` contiene l'indice della vista frontale selezionata, le dimensioni di output, i coefficienti cromatici e il tempo di elaborazione, perfetto per alimentare l'"Edit log" dell'interfaccia.

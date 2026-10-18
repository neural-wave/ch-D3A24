# Fine-tuned packshot LoRA

The pipeline can swap its hosted edit model for a LoRA fine-tuned on our own example pairs.
This document is the whole story: what the LoRA is, how it is trained (fal.ai and vast.ai),
how it reaches the request, and what it costs.

What is *not* in here: the `sartiq_lora_unified/` preprocessing and post-processing. That
code was only the starting point; the pipeline keeps its own spec, reference selection,
prompt, quality gate and repair loop, which are tuned to this dataset. Only the fine-tuned
model is used, exactly like any other hosted model.

## Honesty first: the numbers to beat

Every claim about the LoRA must be measured on the frozen holdout (38 pairs grouped by
target) with the official scorer, the same way the existing results were:

| Setup | Score |
|---|---|
| identity (no model) | 39.78 |
| nano-banana 2.1 baseline prompt | 60.22–60.92 |
| one-shot pipeline (`studio-style-v2`) | 61.82 |
| harness with gate and repair | 61.77 |

Gaps under about 1.5 points are noise. The LoRA is worth submitting only if it moves the
holdout number past 61.82 on the same 38 pairs, with the same gate and repair.

The 332 training pairs are the *train* split of that same holdout split
(`sartiq_lora_unified/holdout/train.jsonl`, kept when `labels.csv` says `ok` or
`multi_input`). `scripts/lora/prepare_dataset.py` hard-refuses any `pair_id` that appears in
`holdout/val.jsonl`, so training cannot leak the 38 evaluation pairs by construction.

## The data

`scripts/lora/prepare_dataset.py` exports the training split from
`data/sartiq_neural_wave_2026` into two equivalent layouts from the same processed pairs:

- `local/lora/dataset/fal/` — `PAIR_start.jpg`, `PAIR_start2..4.jpg`, `PAIR_end.jpg`,
  `PAIR.txt` (fal trainer layout, zipped into `fal.zip`)
- `local/lora/dataset/aitk/` — `target/PAIR.jpg` + `target/PAIR.txt` with captions and
  `reference_1..4/PAIR.jpg` (ai-toolkit layout, bundled for vast.ai)

Choices worth knowing:

- **Canvas 832x1248** — the largest 2:3 size divisible by 16 whose area (1,038,336 px)
  stays under klein's 1024² cap; matches the example targets (~2:3, light grey ~240) and
  the portrait frame of the pipeline. Inputs and targets are letterboxed on (240,240,240),
  never cropped.
- **One caption per pair**, chosen deterministically from five edit-instruction phrasings,
  conditioned on the category, no trigger word. The pipeline's own prompt is deliberately
  not among them, so the LoRA learns the transformation, not one sentence.
- **Quality filters** — targets must have a flat light background and a sane garment area;
  inputs whose mean colour is more than ΔE 45 from the target's are dropped; the most
  central/complete photos are kept, up to 4 per pair (the fal trainer's maximum).
- A `preview.jpg` grid and `dataset_info.json` (canvas, counts, script hashes) land next
  to the layouts.

```
python scripts/lora/prepare_dataset.py            # both layouts, full training split
python scripts/lora/prepare_dataset.py --limit 20 # smoke test
```

## Path A — fal.ai (fastest)

The trainer is `fal-ai/flux-2-klein-9b-base-trainer/edit` (9B only; the 4B exists only on
the vast.ai path). It takes the `fal.zip` through the normal fal queue and returns a
`safetensors` adapter.

```
python scripts/lora/train_fal.py                  # dry run: prints the plan, spends nothing
python scripts/lora/train_fal.py --live           # upload, train, download, register
python scripts/lora/train_fal.py --fetch local/lora/runs/<run>.json   # resume polling
```

- fal bills `0.002 × steps × multiplier` where the multiplier depends on the references
  per pair: 2.11 / 3.44 / 5.09 / 6.95 for 1 / 2 / 3 / 4. The default run (1500 steps,
  4 refs worst case) is about **$20.85**; a 1-reference-only export would be $6.33.
- The estimate is written to the shared cost ledger at submission and checked against
  `ZIR_FAL_BUDGET_USD` (default 70) *before* uploading.
- When the job finishes the script downloads the weights to `local/lora/weights/` and
  writes `local/lora/registry.json` — the pipeline picks them up with no code change.
  `--no-register` keeps the files without touching the registry.

## Path B — vast.ai with ai-toolkit ($100 credit)

`scripts/lora/train_vast.py` writes a self-contained bundle for a rented GPU; nothing is
uploaded or spent by the script itself.

```
python scripts/lora/train_vast.py                 # 9B, 2000 steps, rank 128/64
python scripts/lora/train_vast.py --base 4b       # Apache-2.0 model, 12GB+ GPUs
```

It writes `local/lora/vast/{aitk_sartiq_packshot.yaml, run.sh, README.txt, bundle.zip}`.
`README.txt` is the runbook: rent, `scp bundle.zip`, `unzip`, `bash run.sh`, watch the
samples, download the best checkpoint, stop the instance. The bundle contains only data
and scripts (no keys).

- Recipe: follows BFL's klein `Graphic_Impressions` config — LoRA rank 128/64 (conv
  64/32), `adamw8bit`, `lr 1e-4`, `timestep_type: shift`, `content_or_style: balanced`,
  qfloat8 quantization, bf16 saves every 250 steps — on the exact 832x1248 bucket
  (verified against ai-toolkit's `toolkit/buckets.py`, so training never rescales).
- Hardware from BFL: 9B needs ≥22GB VRAM / 64GB RAM / ~100GB disk; 4B needs ≥12GB / 32GB.
  The 9B base is gated on HuggingFace (accept the licence, export `HF_TOKEN`); the 4B is
  public. 9B is FLUX Non-Commercial — keep the adapter internal; 4B is Apache-2.0.
- Cost: on a 4090-class vast.ai instance (~$0.35–0.55/h) a 2000-step run is a few dollars
  of the $100 credit.

Either path ends with a `.safetensors` file; `scripts/lora/register_lora.py` makes it the
active adapter:

```
python scripts/lora/register_lora.py --weights local/lora/weights/sartiq.safetensors
python scripts/lora/register_lora.py --hub owner/repo --scale 0.8
python scripts/lora/register_lora.py --show
```

A `--weights` file is copied into `local/lora/weights/` and uploaded to fal storage (free,
needs `FAL_KEY`) because the hosted endpoint has to read it; `--url` and `--hub` skip the
upload. `--no-upload` registers a local path, which only helps tools running on this
machine — the hosted endpoint cannot read it.

## Using the adapter in the pipeline

The adapter is a registry entry (`local/lora/registry.json`), and the model key
`sartiq-lora-klein` needs one:

| Where | How |
|---|---|
| CLI | `python -m zir_image_enhancer.cli --model sartiq-lora-klein ...` (or `zir-run --model ...`) |
| Environment | `ZIR_MODEL_KEY=sartiq-lora-klein` |
| Makefile | `make predict MODEL=sartiq-lora-klein TEAM=zir-lora` then `make eval TEAM=zir-lora` |
| Adapter override | `ZIR_LORA_PATH` (fal URL, `owner/repo`, local path) + `ZIR_LORA_SCALE` (0–4) |

What changes under the hood is only the model: the spec points at
`fal-ai/flux-2/klein/9b/base/edit/lora`, `generation/lora.py` adds
`loras: [{"path": ..., "scale": ...}]` to the request, and the canvas is fixed to
832x1248. The spec, references, prompt, gate and repair are untouched. The generator's
cache key covers the adapter and its scale, so cached images never mix adapters, and two
registered adapters can be compared without clearing anything. When no adapter is
registered, the run fails immediately with a clear error instead of silently falling back
to the base model.

Evaluation is the normal loop:

```
make predict MODEL=sartiq-lora-klein TEAM=zir-lora
make eval TEAM=zir-lora
```

## Budget

- Every fal call goes through `CostLedger` at `local/fal_ledger.jsonl`; `spent_usd()` sums
  `cost_usd` over all lines. `ZIR_FAL_BUDGET_USD` (default 70 of the team's $100) is the
  hard stop: a run that would cross it refuses to start.
- fal training estimates are written to the same ledger at submission
  (`kind: lora_training_estimate`), so the guard sees the money even if the training
  script is interrupted.
- One `sartiq-lora-klein` edit is billed at $0.02 per megapixel of input+output with each
  input resized to 1MP: 4 references + ~1MP output ≈ **$0.10** per call (worst case
  encoded in the spec, `est_cost_usd=0.10`). A 38-pair holdout evaluation with repair is
  roughly $4–8. Vast.ai billing is per-instance on vast.ai and outside this ledger;
  destroy the instance when done.

## Troubleshooting

- **`LoraNotAvailableError: model 'sartiq-lora-klein' needs the fine-tuned LoRA`** — no
  registry and no `ZIR_LORA_PATH`; train/register first, or `register_lora.py --show` to
  see what is active.
- **fal rejects the adapter** — check the registered path is a fal storage URL or a
  HuggingFace repo id (`register_lora.py --show`). A local path or a stale upload are the
  usual causes; re-run `register_lora.py --weights ...` to re-upload.
- **Training job failed / script interrupted** — the run state (request id, payload,
  logs) is kept in `local/lora/runs/<name>_<steps>s_<id>.json`; resume polling with
  `--fetch` and nothing is spent twice.
- **`error: spent $... of the $70.00 budget`** — intended: raise
  `ZIR_FAL_BUDGET_USD`/`--max-budget` only on purpose.
- **9B training 401/403 on vast.ai** — the HF repo is gated; accept its licence and set
  `HF_TOKEN` before `bash run.sh`.
- **The LoRA is too strong / drifts** — scale it down: `register_lora.py --scale 0.6` or
  `ZIR_LORA_SCALE=0.6`; 0.6–0.8 often keeps the base model's colour fidelity.
- **Weights look wrong** — ai-toolkit saves `output/<name>/<name>.safetensors` plus step
  checkpoints (`<name>_000001250.safetensors`); pick by the sample grid, not the last step.

## Where things live

| Path | What |
|---|---|
| `local/lora/dataset/` | exports: `fal/`, `fal.zip`, `aitk/`, `manifest.jsonl`, `dataset_info.json` |
| `local/lora/vast/` | vast.ai bundle: config, `run.sh`, `README.txt`, `bundle.zip` |
| `local/lora/weights/` | downloaded/registered `.safetensors` files |
| `local/lora/runs/` | fal training run state (resume, logs, result) |
| `local/lora/registry.json` | the active adapter (path + scale + provenance) |
| `local/fal_ledger.jsonl` | every fal cent, including training estimates |

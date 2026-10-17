# Zir Image Enhancer

A local AI image enhancer pipeline built for the [Neuralwave 2026 Hackathon](https://neuralwave.ch) by the `D3A24` team for [Sartiq AI](https://www.sartiq.ai/).

## Quickstart (local, VS Code Dev Container)

### Requirements

- **VS Code** with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`).
- **Docker:**
  - **Windows:** Docker Desktop with the WSL2 backend. For GPU access, an up-to-date NVIDIA driver on Windows is enough; you do not install the NVIDIA Container Toolkit.
  - **Linux:** Docker Engine plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **GPU is optional.** Without an NVIDIA GPU the container still starts and runs on CPU (slow, but fine for UI and code work).

> **Windows tip:** clone the repo inside WSL2 (for example `~/code/zir-image-enhancer`), not under `C:\`. Files on the Windows drive are much slower inside the container.

### Steps

1. Clone this repository and open it in VS Code.
2. Click **"Reopen in Container"** (or run `Dev Containers: Reopen in Container` from the Command Palette). The first build takes a while because it installs every dependency into the image.
3. Inside the container, start the app with hot reloading:

   ```bash
   gradio src/zir_image_enhancer/app.py
   ```

   Or without hot reloading:

   ```bash
   zir-app
   ```

4. Open `http://localhost:7860`.

Model weights are cached in the `zir-model-cache` Docker volume, so they are downloaded only once and never end up in the repo.

## Running on Vast.ai

A Vast.ai instance is already a container, so the Dev Container itself is not used there. Use the same image instead:

1. Build and push the image once (from the repo root):

   ```bash
   docker build -f .devcontainer/Dockerfile -t ghcr.io/<org>/zir-image-enhancer:latest .
   ```

   ```bash
   docker push ghcr.io/<org>/zir-image-enhancer:latest
   ```

2. Create a Vast.ai template with that image, expose port `7860`, and pick an **on-demand** RTX 4090.
3. Connect over SSH, clone the repo into `/workspaces/zir-image-enhancer` and run:

   ```bash
   uv pip install --system --no-deps -e .
   ```

4. Start the app inside `tmux` so it survives SSH disconnects. Use `share=True` in `demo.launch()` (or a Cloudflare tunnel) to get the HTTPS link phones need for the camera.

## Batch runs and self-scoring

The pipeline is exposed both as the Gradio demo and as a batch runner that writes the
challenge submission layout (`submissions/<team>/<garment_id>/<view>.png`):

```bash
zir-run --manifest data/sartiq_neural_wave_2026/dev/inputs.jsonl \
        --dataset data/sartiq_neural_wave_2026 --out submissions --team zir
```

The official self-evaluation harness is vendored under `tools/` and `scripts/` (see
[Credits](#credits-and-provenance)) and wrapped by a `Makefile`:

```bash
make holdout    # carve a self-evaluation holdout out of the example pairs
make run        # run the studio pipeline over dev into submissions/zir/
make eval       # score predictions/zir/ on the holdout with the frozen scorer
make baseline   # reproduce the identity floor to compare against
make validate   # check the submission layout and images before handing in
```

The dataset is expected at `data/sartiq_neural_wave_2026` (gitignored). Override it per
run with `--dataset <root>` or `make DATASET=<root> …`.
## Local layout

Everything under `local/` and `data/` is ignored by Git. Copy `.env.example` to `.env` and fill in `FAL_KEY`.

```
data/sartiq_neural_wave_2026/   the dataset (never committed, delete after the event)
local/neural-wave-2026/         the official starter repo, with its own .venv for the scorer
local/runs/<run>/               runs: cand_<k>/<id>/front.png, scores, reports
local/fal_cache/                every paid fal result, so nothing is paid twice
local/fal_ledger.jsonl          one line per paid call; the budget guard reads it
```

## Run the app

Copy `.env.example` to `.env`, fill in `FAL_KEY`, then from the repo root:

```bash
uv run zir-app
```

Open `http://localhost:7860` (or start it with `GRADIO_SHARE=1` for an https link a phone can open). Upload the front photo first, then the back and close-ups of details, and optionally type one role per photo (`front`, `back`, `detail chest logo`). Detail photos are only used when they have a role, typed in the app or in the file name (`03_detail_chest-logo.jpg`). The log shows what the pipeline understood and what the gate found; each run keeps its photos and attempts in `local/app_runs/<name>/`.

How the pipeline works, step by step: spec (Gemini 2.5 Flash) → studio prompt plus proportions, flattening, details and sets → nano-banana 2.1 → quality gate (Gemini) → one repair if needed. Every prompt we tried, with its results, is in [docs/PROMPTS.md](docs/PROMPTS.md).

### A whole set at once

One folder per garment in `local/captures/`:

```bash
uv run python scripts/make_capture_manifest.py --photos local/captures
```

```bash
uv run zir-run --manifest local/captures/inputs.jsonl --out local/submissions --team d3a24
```

Spending stops at `ZIR_FAL_BUDGET_USD` (counted in `local/fal_ledger.jsonl`). The ledger uses estimated prices and has under-counted the real fal bill.

## Evaluate on the holdout

| Script | Use |
|---|---|
| `reconstruct.py --run <name> [--ids ...]` | The app's pipeline over the holdout, in the scorer's layout, with a report per garment |
| `score_run.py --run <name> [--pairs <file>]` | Official scorer on the run; appends to `experiments/results.csv` |
| `compare_sheet.py --runs <a> <b> ...` | Pages with input, each run's output and score, and the target |
| `gallery.py` | One folder per garment with its inputs, every output and the target |

## Findings (holdout, 38 pairs grouped by target)

Gaps under about 1.5 points are within noise. Official dev numbers: identity 48.37, one-shot baseline 68.71. All rows are in `experiments/results.csv`.

| Config | Score | Note |
|---|---|---|
| Input unchanged | 39.78 | floor |
| Official baseline (nano-banana 2.1, first photo) | 60.22 | |
| **One-shot: portrait canvas, `studio-style-v2`** | **61.82** | +1.6, 25 of 38 wins |
| nano-banana Pro | 61.23 | 2.5x the price |
| Seedream 4.5 / FLUX.2 / Qwen-Image-Edit-2511 | 49.6 / 43.4 / 54.4 | change the garment |
| Harness (spec, gate, repair), 24 garments | 61.77 | one-shot on the same 24: 62.34 |
| Harness v2 (+ flatten, length, vivid colour, single zip) | 60.40 | length rule made tops too long; reverted to harness + flatten + sets |

What was tried and removed from the code (still in the tags `exp-snapshot-2026-10-17` and `pre-cleanup-2026-10-17`):

| Idea | Effect |
|---|---|
| Cut-out, crop or denoise before generation | -3.9, -1.3, -1.9 |
| Spec-written prompt, with measured hex colours | -3.0, -4.7 |
| Colour correction after generation (learned model, average offset) | -6 to -14, -0.6 to -1.2 |
| Light correction after generation (shading removal, shadow lift) | -0.2 to -1.5 |
| CLIP photo choice (`best`, `filtered`) | picked the back on 5 of 15 test garments |
| Relight pass after generation | -1.5 on 8 garments, barely visible |
| One-shot pipeline scripts (`run_pipeline.py`, `generate.py`, triage) | replaced by the harness in the app |

Other results:

- Vision triage picks the front in any photo order (61.67 against 60.22 for the first photo when the photos are shuffled).
- Unnamed close-ups pulled the colour (-2.1), so only named detail photos are used.
- The model already corrects most of the light: garment L* error drops from 12.9 (input) to 5.6 (output). What is left has no fixed sign (7 garments too dark, 9 too light) and cannot be predicted from the photos, so a phone photo alone does not give the studio colour. Capture quality is the lever.
- The harness is 0.6 below the one-shot pipeline on the same 24 garments (within noise). It is kept because it repairs invented or missing details, which the jury judges and the metric barely sees.

## Optional dependency groups

All groups are installed in the image. If you install outside the container:

| Group | What for |
|---|---|
| `scorer` | The vendored frozen Sartiq scorer and its dependencies (scipy, lpips, open_clip_torch, rembg, onnxruntime, tqdm) |
| `dev` | ruff, mypy, pytest, pre-commit |

## Development

```bash
ruff check . && ruff format .
```

```bash
mypy src
```

```bash
pytest
```

## Credits and provenance

The self-evaluation harness is **vendored unmodified** from the official Sartiq Neural
Wave 2026 participant starter repo, [`shootify-io/neural-wave-2026`](https://github.com/shootify-io/neural-wave-2026)
at commit `ce3d736` (16 October 2026), under the Neural Wave 2026 hackathon terms of
service:

| Path | Upstream source | Notes |
|---|---|---|
| `tools/scorer/` | `scorer/` | The **frozen** reference-similarity scorer (`sartiq-score`). Installed from this path via `[tool.uv.sources]`. |
| `tools/baselines/` | `baselines/` | Identity floor + hosted image-model baseline, kept as reference points. |
| `scripts/` | `scripts/` | `make_holdout.py`, `eval_val.py`, `make_submission.py`. Unchanged; the `Makefile` supplies this repo's dataset path. |
| `docs/neural-wave-2026/` | `docs/` | Challenge docs and the problem-statement PDF. |

The scorer is frozen and is not modified here. Dataset images are provided under the
hackathon terms and are **not** redistributed in this repository.

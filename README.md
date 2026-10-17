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

## Local layout

Everything under `local/` and `data/` is ignored by Git:

```
data/sartiq_neural_wave_2026/   the dataset (never committed, delete after the event)
local/neural-wave-2026/         the official starter repo, with its own .venv for the scorer
local/runs/<run>/               generation runs (candidates, scores, contact sheets)
local/fal_cache/                every paid fal result, so nothing is paid twice
local/fal_ledger.jsonl          one line per paid call; the budget guard reads it
```

Copy `.env.example` to `.env` and fill in `FAL_KEY`.

The holdout comes from the starter repo, created once from `local/neural-wave-2026`:

```bash
.venv/Scripts/python.exe scripts/make_holdout.py --pairs ../../data/sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout --group-by-target
```

## The pipeline (current best)

One command per split. It generates every garment and writes a validated submission to `local/submissions/<team>/`:

```bash
python scripts/run_pipeline.py --split dev --dry-run
```

```bash
python scripts/run_pipeline.py --split dev
```

What it does for each garment:

1. Picks one photo: the front, chosen by a vision model (Gemini 2.5 Flash through fal's OpenRouter endpoint, about $0.001 per garment) that sees all the photos of the garment at once (`--inputs triage`). It also flags fine details (small text, logos) in `triage.jsonl` so they are checked first. If the call fails, the first photo is used.
2. Sends them to nano-banana 2.1 (fal) on a 2:3 portrait canvas with the `studio-style-v2` prompt: generic studio catalogue conventions written as rules per garment type (text only, no target images) (sleeves down along the body, closures done up, trouser legs straight and parallel, light-grey background, whole garment visible).
3. Converts the result to PNG and exports it in the official layout, then runs the official validator.

Cost: about $0.06 per garment. Results are cached by content, so reruns are free.

## Running on our own captures (test day)

1. Shoot each garment following the capture protocol: garment laid flat on the backdrop, phone straight above, one soft light, grey card (or white paper) in the **bottom-right** corner, the front photo first.
2. Copy the photos to `local/captures/<garment_id>/` (for example `local/captures/g01/01_front.jpg`). Name them so the front sorts first. HEIC is converted automatically.
3. Build the manifest and run the pipeline on it. The output goes to the same submission folder, with the official ids:

```bash
python scripts/make_capture_manifest.py --photos local/captures
```

```bash
python scripts/run_pipeline.py --split test --captures local/captures/inputs.jsonl --wb-card bottom-right --prompt studio-flat-v1 --run test_captures --dry-run
```

- `--wb-card <corner>` measures the card, removes the colour cast (in linear light, brightness kept) and paints the card out before generation.
- `--prompt studio-flat-v1` tells the model the photo is already a flat-lay with correct colours and asks it to change as little as possible.
- Without `--captures`, the pipeline uses the photos supplied by the organisers.

## Generation experiments

Run from the repo root. Always dry-run first: it prints the number of calls and the cost.

```bash
python scripts/generate.py --split holdout --model nano-banana-2.1 --prompt baseline --inputs first --run holdout_nb21_baseline --dry-run
```

```bash
python scripts/generate.py --split holdout --model nano-banana-2.1 --prompt baseline --inputs first --run holdout_nb21_baseline
```

```bash
python scripts/score_run.py --run holdout_nb21_baseline
```

```bash
python scripts/contact_sheet.py --run holdout_nb21_baseline --split holdout
```

- `generate.py` writes `local/runs/<run>/cand_<k>/<id>/front.png`; `--samples K` gives K candidates per garment, `--inputs all` sends every photo of the garment.
- `score_run.py` scores each candidate with the official scorer and appends one row to `experiments/results.csv`, including `best_of_k` (the score if we always picked the best candidate).
- Spending stops at `ZIR_FAL_BUDGET_USD` (default $70, counted in `local/fal_ledger.jsonl`).
- Models: `nano-banana-2.1` (official baseline), `nano-banana-2.1-portrait` (2:3 canvas), `nano-banana-pro`, `flux-2`, `seedream-4.5`. Prompts: `baseline` (official), `studio-white-v1`, `studio-style-v2`.
- `--preprocess cutout|crop|denoise` changes the photo before it is sent (see findings: all three hurt).
- `--inputs filtered` keeps only full front views, chosen by local CLIP (`preprocess/photo_filter.py`); it drops tags, barcodes, bags and close-ups.
- `--manifest <file>` runs on any manifest; `score_run.py --pairs <file>` scores against it.
- `select_capture_style.py` writes the pairs whose first photo looks like a given capture style (flat, hanger, mannequin, worn, studio).
- `cutout_run.py` makes zero-cost runs without generation (cut-out of the input, or re-framing of a generation run).

## Findings so far (holdout, 38 pairs grouped by target)

Paired differences are against the same 38 garments; with this sample size, gaps under about 1.5 points are within noise.

| Config | Holdout | Colour | LPIPS | CLIP | IoU | $/garment |
|---|---|---|---|---|---|---|
| Input unchanged (identity) | 39.78 | 0.38 | 0.16 | 0.50 | 0.68 | 0 |
| Input cut out on grey, no generation | 40.09 | 0.38 | 0.18 | 0.55 | 0.61 | 0 |
| Official baseline (nano-banana 2.1, first photo) | 60.22 | 0.61 | 0.35 | 0.73 | 0.84 | 0.06 |
| nano-banana 2.1 portrait, official prompt | 60.92 | 0.63 | 0.35 | 0.72 | 0.84 | 0.06 |
| nano-banana 2.1 portrait, `studio-white-v1`, all photos | 61.04 | 0.61 | 0.37 | 0.74 | 0.85 | 0.06 |
| same + cut-out inputs (`--preprocess cutout`) | 57.16 | 0.56 | 0.32 | 0.71 | 0.82 | 0.06 |
| pipeline + crop around the garment, background kept (`--preprocess crop`) | 60.50 | 0.63 | 0.33 | 0.73 | 0.84 | 0.06 |
| pipeline + mild OpenCV denoise (`--preprocess denoise`) | 59.95 | 0.62 | 0.33 | 0.73 | 0.83 | 0.06 |
| same, re-framed on grey after generation | 60.13 | 0.61 | 0.36 | 0.72 | 0.84 | 0.06 |
| nano-banana Pro, `studio-white-v1`, all photos | 61.23 | 0.62 | 0.36 | 0.75 | 0.83 | 0.15 |
| **nano-banana 2.1 portrait, `studio-style-v2`, all photos** | **61.82** | **0.64** | 0.35 | 0.75 | 0.85 | 0.06 |
| Seedream 4.5, `studio-white-v1`, all photos | 49.59 | 0.49 | 0.24 | 0.65 | 0.73 | 0.04 |
| FLUX.2 [dev], `studio-white-v1`, all photos | 43.36 | 0.34 | 0.25 | 0.59 | 0.73 | 0.07 |

What we learned (official dev numbers: identity 48.37, one-shot baseline 68.71):

- **Generation is needed.** Without it the garment keeps the hanger or mannequin pose, and silhouette and LPIPS stay at the level of the raw photo.
- **The studio-style prompt is the best change so far:** +1.6 over the official baseline (wins on 25 of 38 garments) at the same cost.
- **Pro is not worth 2.5x the price** here, and FLUX.2 and Seedream change the garment (wrong colour, extra trousers from the mannequin, a sweater turned into a cardigan).
- **Cutting the garment out before generation hurts** (-3.9, wins on 13 of 38): the model loses context and colour. Re-framing after generation does not help either.
- **Picking among candidates is not worth paying for yet.** The best of 4 nano-banana outputs would score 64.83, but choosing without the reference (scoring each candidate against the garment's own photos) only reaches 61.41 against 60.85 for a random pick.
- **Correcting colour after generation hurts.** A colour model trained on the 344 holdout-train pairs (phone colour to studio colour) predicts worse than nano-banana already does (quantile distance 8.2 vs 3.7) and drops the score to 48-56. The model's outputs are only slightly too light (+2.5 L*) and yellow (+1 b*) on average.
- **Any change to the input photo hurts.** Cut-out on white -3.9, crop with the real background -1.3 (13 of 38 wins), mild denoise -1.9 (14 of 38). The edit model seems to read the whole scene (scale, light, context) and does better with the photo as it is.
- **More photos do not help.** On the 43 multi-photo pairs of holdout-train (never used for tuning), with the pipeline prompt:

  | Photos sent | Score | vs first photo |
  |---|---|---|
  | First photo only | **61.46** | |
  | All photos | 60.71 | -0.76, 20 of 43 wins |
  | CLIP-filtered (no tags, bags, close-ups) | 60.83 | -0.63, 18 of 43 wins |

  The differences are within noise, but none beats the first photo, which is also the cheapest.
- **Choosing the single photo: order matters more than it should.** Same 43 pairs, also with the photos shuffled into a random order (the first photo changes in 33 of 43):

  | Policy | Original order | Shuffled order |
  |---|---|---|
  | First photo (`first`) | **61.46** | 60.22 |
  | First photo that is a whole garment (`first-usable`, pipeline) | **61.46** | 60.30 |
  | Most "front-looking" photo by CLIP (`best`) | 61.19 | **61.19** |
  | **Front chosen by a vision model (`triage`, pipeline)** | **61.70** | **61.67** |

  `best` does not depend on the order, but on the test set it picked the back for 5 of 15 garments (CLIP cannot tell the front of a hanging garment from its back; g12's back has a large print). The vision model compares the photos side by side, picks the front on all 15 test garments, and is the best policy in both orders, so it is the pipeline default. `first-usable` stays as the fallback when the call fails.
- **Close-ups of fine details do not help the score.** Sending the front plus close-ups of its prints or logos (`--max-closeups 2`) lowered the score by 2.09 on the 11 garments that had them (4 of 11 wins); telling the model to take colour only from the full photo brought it to -1.81. The close-ups are often lit differently and pull the colour (a black T-shirt came out grey), and the details were not visibly more faithful.
- **Open model (Qwen-Image-Edit-2511, Apache 2.0, via fal):** 54.37 without LoRA (it changes the garment type: a skirt became a shirt dress), 39.69 with the public "Extract-Outfit" LoRA (it invents a full outfit). A LoRA trained on our pairs is the only open route left.
- **Reconstruction from a garment spec (`scripts/reconstruct.py`).** One vision call extracts category, construction and details (with boxes); a quality gate checks the output. Holdout, paired against the pipeline (61.82):

  | Variant | Score | Diff |
  |---|---|---|
  | Spec in the prompt + detail crops + gate | 58.84 | -2.99 (9 of 38 wins) |
  | Same + measured hex colours | 57.16 | -4.66 |
  | Studio prompt + front only; spec used only by the gate (`--prompt-mode style`) | 61.46 | -0.37 |

  Writing the spec into the prompt forces any spec error and makes the model rely less on the photo. Hex colours carry the photo's colour cast (use them only with `--wb-card`). The gate flags mostly minor issues (button counts); repairing those scored -2, so it now repairs only severe problems (different garment, back view, extra items, invented text) and reports the rest.
- **Prompt for flat-lay captures.** On the 22 holdout-train pairs whose first photo is a flat-lay (picked by CLIP, checked by eye), `studio-flat-v1` scores 61.29 against 60.74 for `studio-style-v2` (+0.54, 11 of 22 wins, within noise) with better texture (LPIPS 0.373 vs 0.349). It is the prompt for our own flat-lay captures.
- **No stable colour bias to correct.** An average offset learned on 43 train outputs (studio photos about 1.9 L* brighter than our outputs) is the opposite of what the holdout showed earlier (outputs 2.5 L* brighter). Applying it to the holdout lowers the score by 0.6 to 1.2 (clearly negative). Post-generation colour tweaks are out.
- **Grey-card white balance cannot be measured on the dataset** (no photo has a card). It is in the pipeline as `--wb-card` and is judged on our own captures against the physical garment.

## Optional dependency groups

All groups are installed in the image. If you install outside the container:

| Group | What for |
|---|---|
| `generative` | Optional creative mode (diffusers, accelerate) |
| `eval` | Metrics and plots (pyiqa, pandas, matplotlib) |
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

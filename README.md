# Zir Image Enhancer

Zir (Ukrainian *зір*: vision, sight) turns a few phone photos of a garment into one clean
studio packshot: front view, on a light background, with the real colours and details kept.

Built by team `D3A24` for the [Sartiq](https://www.sartiq.ai/) challenge at the
[Neural Wave 2026 Hackathon](https://neuralwave.ch).

**Live demo:** <https://huggingface.co/spaces/slymachenko/zir-image-enhancer>

## Deliverables

Everything we hand in is in [`deliverables/`](deliverables/):

| What | Where |
|---|---|
| Final report, with the capture protocol | `deliverables/REPORT.pdf` (coming) |
| Presentation video | link in `deliverables/VIDEO.md` (coming) |
| Packshots for the 15 on-site garments (`g01`–`g15`) | `deliverables/d3a24_test_v3_pro.zip` |
| Packshots for the 27 dev garments (69.77 on the dev leaderboard) | `deliverables/d3a24_dev_v3_pro.zip` |
| The pipeline that made them (v3 with nano-banana Pro) | this repository; see [Run](#run) |
| Live demo | [Hugging Face Space](https://huggingface.co/spaces/slymachenko/zir-image-enhancer) |

- [Deliverables](#deliverables)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run](#run)
- [Evaluate](#evaluate)
- [Results](#results)
- [Project layout](#project-layout)
- [Development](#development)
- [Credits](#credits)

## What it does

You upload the photos you have (on a hanger, on a mannequin, worn, or laid on the floor).
The app returns a catalogue-style image of the same garment, without the person, mannequin
or hanger, and tells you what it understood and what it checked.

Two hosted models do the work, both through [fal.ai](https://fal.ai):

| Model | Role |
|---|---|
| Gemini 2.5 Flash (vision) | Reads the photos before generating (the "spec") and checks the result afterwards (the "gate") |
| nano-banana Pro or 2.1 edit (image) | Draws the packshot from the photos and a written instruction (Pro for our submission; 2.1 is the cheaper default) |

Our code around them (the harness) decides which photos are sent, writes the instruction,
checks the result on what the jury looks at, and repairs it once if needed.

## Architecture

One garment flows through a single seam, `StudioPipeline` (`pipeline.py`), which is shared
by the Gradio app (`app.py`) and the batch runner (`cli.py` / `zir-run`). The stages are:

1. **Prepare** — `pipeline.prepare_photos`.
   *Input:* the raw phone photos (HEIC, EXIF-rotated, or with an alpha channel) and optional
   role labels. *Output:* upright RGB JPEG copies under
   `local/app_runs/<name>/inputs/`, named so the roles survive the round trip
   (`01_front.jpg`, `03_detail_chest-logo.jpg`). A photo that is already an upright RGB JPEG
   is copied byte for byte — re-saving it would lose detail and change the cache key of
   every call made with it.

2. **Spec** — `reconstruction/spec.py` via `reconstruction/vision.py`.
   *Input:* every prepared photo. *Output:* one `GarmentSpec` from a single Gemini 2.5
   Flash call: which photo is the front, the category and construction, the distinctive
   details to keep, the proportions, how the garment is photographed (flat / hanger /
   mannequin / worn), and the pieces of a set. A failed call falls back to a minimal spec
   so a garment is never blocked.

3. **One-shot packshot** — `generation/prompts.py`, `generation/fal_backend.py`,
   `generation/models.py`.
   *Input:* the front photo and a studio prompt. *Output:* the first packshot, drawn by
   nano-banana. The prompt asks for a flat-lay catalogue shot, or the
   invisible-mannequin (`ghost`) look when the spec says the photo shows a garment on a
   mannequin. The one-shot goes first because it scores best on the metric.

4. **Jury gate** — `reconstruction/gate.py`.
   *Input:* the real front photo, the packshot, and the spec's category, pieces, proportions
   and details. *Output:* a `GateResult` listing the defects the jury would notice, plus a
   short `fix`. The gate checks the same garment, front view, nothing cropped, logos and
   text, zips and buttons, seams, texture, colour, proportions, studio light, no person or
   props, no missing piece of a set, every listed detail, and no invented text or logo. A
   broken gate passes the garment and says so in the log.

5. **Repair, only if the gate flagged something** — `reconstruction/reconstructor.py`,
   `reconstruction/prompt.py`.
   *Input:* the flagged defects and the reference photos. A **detail** defect (logo, text,
   closures, seams, texture) is fixed by **editing the packshot**, so shape, colour and
   framing stay, with the real photos as the reference. A **global** defect (colour,
   proportions, view, crop) is fixed by **drawing it again** from the photos. The second
   attempt is gated too, and the pipeline keeps the image with strictly fewer defects; a tie
   keeps the one-shot because it scores better.

6. **Output** — `StudioResult` (the final image, the front photo, a plain-language log, the
   cost and the time). The app shows a before/after slider; `zir-run` writes
   `submissions/<team>/<garment_id>/front.png`.

Both hosted calls go through the same guardrails in `generation/fal_backend.py` and
`reconstruction/vision.py`: identical requests are served from disk
(`local/fal_cache/`, `local/reconstruction_cache/`), every paid call is appended to
`local/fal_ledger.jsonl`, and a call that would cross `ZIR_FAL_BUDGET_USD` is refused
before it is sent.

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.11+ and the project environment for
  you; no conda or system Python setup needed)
- A fal.ai API key with balance
- No GPU: both models run on fal

`uv run zir-app` sets up the app's dependencies on first use. To install everything at once
(the app, the official scorer and the dev tools), run `uv sync --all-extras` or
`make install`.

## Setup

### Container (recommended)

The dev container is uv-native and installs the project environment into `/opt/venv`,
outside the workspace, so it survives rebuilds and works the same on any host.

**VS Code / Dev Containers.** Open the repo and choose **Reopen in Container**. The image
forwards port `7860`, mounts a model-cache volume at `/root/.cache`, and sets
`GRADIO_SERVER_NAME=0.0.0.0` / `GRADIO_SERVER_PORT=7860` so the demo is reachable from the
host. The default image is lean: the app and CLI only call hosted models, so the heavy
`scorer` extra (torch, onnxruntime, open_clip, rembg) is **not** installed.

To self-evaluate inside the container, build the heavier image: set `"SCORER": "1"` under
`build.args` in `.devcontainer/devcontainer.json` and rebuild.

**Plain Docker (any host, e.g. Vast.ai).** Build and run directly:

```bash
# Lean image: app + CLI only.
docker build -f .devcontainer/Dockerfile -t zir .

# Self-evaluation image: adds the official scorer (torch etc.).
docker build -f .devcontainer/Dockerfile --build-arg SCORER=1 -t zir-eval .

docker run --rm -it -p 7860:7860 \
  -v "$PWD:/workspaces/zir-image-enhancer" \
  -v zir-cache:/root/.cache \
  zir
```

The workspace mount brings in your `.env`, `data/`, and generated outputs; the named
`zir-cache` volume keeps model downloads between runs. Inside the container, run the app
with `uv run zir-app` or the CLI with `uv run zir-run ...`.

### Local

```bash
git clone <repo-url> zir-image-enhancer
cd zir-image-enhancer
cp .env.example .env      # then set FAL_KEY
```

`.env` is Git-ignored and holds two values:

| Variable | Meaning |
|---|---|
| `FAL_KEY` | Your fal.ai key. Required to generate anything. |
| `ZIR_FAL_BUDGET_USD` | Hard stop for fal spending, counted in `local/fal_ledger.jsonl`. Default `70`. |

Then install the environment:

```bash
uv sync --all-extras      # app + official scorer + dev tools
# or: make install
```

## Run

### The app (one garment)

```bash
uv run zir-app
```

Open `http://localhost:7860`, upload one to five photos of the same garment (the front
first, then the back and close-ups) and press **Create studio image**.

Tips:

- Detail photos are only used when they have a role. The pipeline reads the role from the
  file name, so name them before uploading (`front.jpg`, `02_back.jpg`,
  `03_detail_chest-logo.jpg`). Otherwise the spec picks the front and only the front photo
  is sent to the generator.
- Add `GRADIO_SHARE=1` before the command to get an https link a phone can open (needed for
  the camera).
- Every run keeps its photos and attempts in `local/app_runs/<name>/`.

### A whole split (the submission)

`zir-run` runs the same pipeline over a manifest and writes the submission layout
`submissions/<team>/<garment_id>/front.png`. Check what it will do first, then run it:

```bash
uv run zir-run --manifest data/sartiq_neural_wave_2026/test/inputs.jsonl \
    --out submissions --team d3a24 --dry-run

uv run zir-run --manifest data/sartiq_neural_wave_2026/test/inputs.jsonl \
    --out submissions --team d3a24
```

Add `--model nano-banana-pro` to use the larger model. Every paid call is cached, so a
rerun is free.

For your own photos, give `zir-run` a manifest with one row per garment:

```json
{"garment_id": "c01", "inputs": ["C:/photos/c01/1.jpg", "C:/photos/c01/2.jpg"]}
```

## Evaluate

Everything you need to predict and score lives in `tools/` (the vendored, frozen scorer
and baselines) and `scripts/` (the starter scoring/holdout/submission scripts). The
`Makefile` wraps them so a full loop is a few commands. The default dataset root is
`data/sartiq_neural_wave_2026` and must directly contain `train/`, `dev/` and `test/`.

### Self-score on a holdout

Only `train/` ships targets, so carve a holdout, predict on it, and score:

```bash
make holdout                    # scripts/make_holdout.py -> holdout/{train,val}.jsonl + meta.json
make predict                    # our pipeline over the holdout  -> predictions/<TEAM>/
make predict-identity           # the identity floor              -> predictions/identity/
make eval TEAM=zir              # scripts/eval_val.py with the frozen scorer
make eval TEAM=identity         # compare against the floor
```

`make eval` writes `val_out/leaderboard.csv`, `scores.jsonl`, `review.csv`, per-pair
`pairs.csv` and a `summary.json` whose mean comes from the official, garment-weighted
aggregate. The scorer needs the real studio photo, so it runs offline only; dev scores come
from the leaderboard.

### Build and validate the submission

```bash
make run                        # zir-run over dev  -> submissions/<TEAM>/
make run SPLIT=test TEAM=d3a24  # or the test split
make validate                   # scripts/make_submission.py checks the layout
make validate ZIP=1             # also writes <TEAM>.zip
```

### Compare against the hosted baseline

```bash
make baseline-nanobanana-dry    # preview requests, no key, no spend
make baseline-nanobanana        # hosted baseline over dev (spends budget)
make predict-nanobanana         # hosted baseline over the holdout
```

### Raw commands (without `make`)

The Makefile only fills in paths; every step is a plain command:

```bash
uv run python scripts/make_holdout.py --pairs data/sartiq_neural_wave_2026/train/pairs.jsonl \
    --frac 0.05 --seed 0 --out holdout --dataset data/sartiq_neural_wave_2026 --group-by-target

uv run --extra scorer python scripts/eval_val.py --pairs holdout/val.jsonl \
    --preds predictions --team zir --out val_out --dataset data/sartiq_neural_wave_2026 --jobs 4

uv run --extra scorer python scripts/make_submission.py --submissions submissions \
    --team d3a24 --dataset data/sartiq_neural_wave_2026 --zip
```

Common Makefile overrides: `TEAM`, `SPLIT`, `MODEL` (`nano-banana-2.1-portrait` or
`nano-banana-pro`), `FRAC`, `SEED`, `JOBS`, `LIMIT`, `ZIP`. For example
`make eval TEAM=zir JOBS=4 LIMIT=8` or `make run MODEL=nano-banana-pro`.

## Results

Scores come from the official scorer (colour 40%, LPIPS 25%, CLIP 15%, silhouette 15%,
SSIM 5%) on a holdout of 38 training pairs grouped by target, and a fixed half of it
(19 pairs) for cheaper runs.

| Version | Holdout score |
|---|---|
| Phone photo unchanged | 39.78 |
| Official baseline (nano-banana 2.1, first photo) | 60.22 |
| v1, one-shot studio prompt | 61.82 |
| v3, one-shot + jury gate + edit or redraw | 61.90 |
| v4, 9 versions + selector | 63.01 |

This repo runs v3. The other versions are kept in the tag `pre-v3-only-2026-10-18`.

| Version | What it does | Holdout (38) | Dev leaderboard |
|---|---|---|---|
| v1 | Studio prompt and the front photo, one generation | 61.82 | 68.62 |
| v2 | Spec details and sets in the prompt, named detail photos, gate, one redraw | 61.77 (24 garments) | 68.30 |
| **v3** | v1, then the jury gate; a detail defect is fixed by editing the packshot, a global one (colour, proportions, view, crop) by drawing it again | **61.90** | **68.75** |
| v4 | 9 versions per garment, one pick by CLIP and AlexNet embeddings, the jury gate on the top 3 | 63.01 | 69.59 |
| **v3 with nano-banana Pro** | v3 as it runs now, with `--model nano-banana-pro` | not run | **69.77** |

v4 scored well, but it makes nine images per garment: about nine times the cost and three
times the time. We kept v3 — one image per garment, repaired only when the jury would see a
defect — and moved it to nano-banana Pro. With the same settings, nano-banana 2.1 scored
68.64 on dev (the 68.75 above drew every garment laid flat; v3 now uses the
invisible-mannequin style for mannequin photos). Pro's first image also had gate defects on
2 of 27 dev garments, against 4.

Cost and time per garment: about $0.16 with nano-banana Pro and $0.06 with nano-banana 2.1
(the default), about 20 seconds, plus a repair for about one garment in ten. The spec and
each gate check add about $0.002.

### What we tried and dropped

All rows are in `experiments/results.csv`, every prompt in `docs/PROMPTS.md`, removed code
in the tags `exp-snapshot-2026-10-17`, `pre-cleanup-2026-10-17` and
`pre-v3-only-2026-10-18`.

| Idea | Result |
|---|---|
| Other generators: nano-banana Pro, FLUX.2, Seedream 4.5, Qwen-Image-Edit 2511 | 61.2, 43.4, 49.6, 54.4 |
| nano-banana Pro versions added to the v4 pool on dev (with and without v3 in Pro) | 69.34 and 68.89 against 69.59 |
| Pre-processing: cut-out, crop, denoise, 2x super-resolution (Real-ESRGAN) | -3.9, -1.3, -1.9, -3.0 |
| Post-processing: colour offsets, a learned colour model, saturation, shading, background tone, stretch, blur, sharpen | -0.1 to -14 |
| A per-garment colour correction predicted from the photo and its background | no better than none (L* error 5.08 vs 5.10) |
| Prompt lines: spec-written prompt, flatten, real length with vivid colour, relight pass | -3.0, -1.2, -1.3, -1.5 |
| Invisible-mannequin style for mannequin photos | +1.3 on those garments, but every reference is flat-lay |
| Picking between versions with a Gemini judge, a scorer run against the input photo, or consensus | at or below a random pick on the holdout |
| A third embedding expert (DINOv2); embeddings before the generator (routing, conditional style) | no gain over CLIP + AlexNet |
| Prompt variants on dev (official prompt, change as little as possible, white background) | 68.59 to 68.66 against 68.62 |

Dev (67% mannequin photos) and test (all hanger photos) are distributed differently from our
holdout (26% mannequin), which is why some holdout gains did not move dev.

## Project layout

| Path | What it is |
|---|---|
| `src/zir_image_enhancer/app.py` | The Gradio app |
| `src/zir_image_enhancer/cli.py` | `zir-run`, the batch runner |
| `src/zir_image_enhancer/pipeline.py` | `StudioPipeline`, shared by the app and `zir-run`: prepares the photos, runs the reconstructor |
| `src/zir_image_enhancer/paths.py` | Where local files live (fal cache, caches, cost ledger, app runs) |
| `src/zir_image_enhancer/reconstruction/` | Spec, gate, repair prompts, reference photos and the v3 loop (`reconstructor.py`) |
| `src/zir_image_enhancer/generation/` | fal client with cache and budget guard, the models and the studio prompts |
| `scripts/eval_val.py`, `make_holdout.py`, `make_submission.py` | Official starter scripts (vendored): holdout, scoring, validation and ZIP |
| `tools/scorer/` | The frozen official scorer (`sartiq-score`) and its docs |
| `tools/baselines/` | The identity floor and the hosted nano-banana baseline |
| `docs/PROMPTS.md` | Every prompt we tried and how it scored |
| `experiments/results.csv` | Every scored run |
| `docs/neural-wave-2026/` | Challenge docs and the problem statement (vendored) |

Dataset images are provided under the hackathon terms and are not redistributed in this
repository; they go in `data/sartiq_neural_wave_2026` (never committed).

## Development

Everything runs through uv: `uv sync --all-extras` (or `make install`) creates the
project's `.venv` with the dev tools and the scorer. The dev tools come from the default
`dev` group, so plain `uv run` already has them.

```bash
uv run ruff check .       # or make fmt / make lint
uv run mypy src           # or make type
uv run pytest             # or make test
make check                # lint + type + test
```

## Credits

The self-evaluation harness is vendored unmodified from the official starter repo
[`shootify-io/neural-wave-2026`](https://github.com/shootify-io/neural-wave-2026) at commit
`ce3d736` (16 October 2026), under the Neural Wave 2026 hackathon terms:

| Path | Upstream | Notes |
|---|---|---|
| `tools/scorer/` | `scorer/` | The frozen scorer (`sartiq-score`), installed through `[tool.uv.sources]` |
| `tools/baselines/` | `baselines/` | Identity floor and hosted baseline |
| `scripts/eval_val.py`, `make_holdout.py`, `make_submission.py` | `scripts/` | Unchanged; the `Makefile` supplies this repo's dataset path |
| `docs/neural-wave-2026/` | `docs/` | Challenge docs and the problem statement |

## License

Our code is released under the [MIT License](LICENSE). The vendored files listed above keep the terms of the starter repo and the hackathon, and the dataset is not part of this repository.

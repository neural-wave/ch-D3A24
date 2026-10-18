# Preprocessing

OpenCV runs on every photo before it is sent to the image model, so the model gets the most
useful, cleanest possible view of the garment. The code lives in
`src/zir_image_enhancer/reconstruction/preprocess.py` and is wired into the reconstructor
(`reconstruction/reconstructor.py`); the Gradio app and `zir-run` get it through
`StudioPipeline` with no extra configuration.

## Where it sits in the pipeline

```
photos -> input assessment + spec (vision, ORIGINAL photos)
       -> preprocess (OpenCV)  <- this stage
       -> references (front + detail close-ups) -> generate -> gate (ORIGINAL front)
```

The vision steps (assessment, spec, gate) always read the **original** photos, because they
are evidence about the real garment. Only the images given to the *generation* model are
preprocessed. The gate still compares the packshot against the original front photo, so a
preprocessing mistake cannot hide a real colour or detail problem from quality control.

## What it does

| Step | Why | Default |
|---|---|---|
| Gray-world white balance | Remove the colour cast of indoor, tungsten, shade or mixed light, so the model does not copy a warm or green tint into the packshot. Blended by `white_balance` and clamped to `white_balance_max_gain` so a garment that fills the frame keeps its own colour. | on, gentle |
| Neutral-pixel white balance | Optional alternative to gray-world (see below). Estimates the light from near-neutral, bright pixels only, so a warm wood floor or a beige wall cannot pull a white garment blue. | **off** — `white_balance_method="neutral_pixels"` to try |
| Illumination flattening | Divide out the low-frequency light field (the photographer's cast shadow, a lamp on one side, a vignette) so the garment is lit evenly like a studio bank. Luminance-only and mean-preserving, so hue and exposure are untouched. This targets what the model cannot tell apart from a real fold in the fabric. | **off** — `flatten_illumination=0.25` to try |
| Percentile tone stretch | Lift a dark or flat phone exposure to the bright range the studio target lives in. An L-channel percentile stretch adds no halos, unlike CLAHE. | on |
| Downscale to `max_side` | Keep the payload small without losing detail the model can use. | 1536 px long edge |
| Unsharp mask | Restore the crispness lost to downscaling. | mild (0.3) |
| Edge-preserving denoise | Optional. | **off** — it flattens fine fabric texture |
| Controlled grain | Add a small, reproducible Gaussian grain so an image that has been through several resizes and JPEG round-trips stays inside the natural-photo distribution the model was trained on, instead of looking like a synthetically smoothed image. Seed comes from the source bytes, so runs are reproducible and cache keys stay stable. | on (σ = 1.0) |
| Garment mask | Best-effort foreground mask from a background-colour threshold in LAB. Used to pick *background* pixels for white balance, and available for a tight crop. Returns nothing when the scene is ambiguous. | on |
| Detail crops | Crop the front photo at the spec detail boxes (`detail_margin` context) at full resolution, cut from the **enhanced** front so they match image 1 exactly. Small crops are upscaled to `min_detail_side`. This is the architecture's texture-reference subsystem: it lets the model copy a logo or a label instead of redrawing it from memory. | on |
| Tight garment crop | Frame the garment closely. | **off** — tested worse, see below |
| Palette | k-means dominant colours of the garment. | **off** — kept for reports/experiments |

All of it returns `PreparedReferences` (paths + capture statistics), and the reconstructor
turns those into the `Reference` list the prompt is built from.

## Why it is conservative

This repo is evidence-driven, and earlier experiments showed that heavy processing *lowers*
the holdout score. The defaults only remove capture artefacts and add information; they do
not restyle the garment.

| Experiment | Score | Note |
|---|---|---|
| `studio-style-v2` (harness base) | **61.82** | no preprocessing |
| photo cropped around the garment | 60.50 | tight framing hurt |
| photo denoised | 59.95 | texture loss hurt |
| cut-out photos as input | 57.16 | background removal hurt the model |
| spec prompt + detail crops | 58.84 | crops bundled with a worse prompt |
| measured hex colours in the prompt | 57.16 | numeric colour hurt |

So the risky steps (denoise, tight crop, palette in the prompt) are **off by default** and
kept as switches, while the safe normalisation and the detail crops are on.

## Turning it off and A/B testing

`PreprocessSettings` is a field on `ReconstructionSettings`, so everything is configurable in
code and from the evaluation script:

```bash
# baseline (no preprocessing) vs the default
uv run python scripts/reconstruct.py --split holdout --run recon_plain --no-preprocess
uv run python scripts/reconstruct.py --split holdout --run recon_pre
# turn individual steps off / tune the grain / try the tight crop
uv run python scripts/reconstruct.py --split holdout --run recon_nocrop --no-detail-crops
uv run python scripts/reconstruct.py --split holdout --run recon_grain0 --grain 0
uv run python scripts/reconstruct.py --split holdout --run recon_tight --crop-to-garment
uv run python scripts/score_run.py --run recon_pre
```

To see the effect without spending anything, build a before/after sheet (original, mask,
enhanced, target) for a split or a capture folder:

```bash
uv run python scripts/preview_preprocess.py --split train --limit 8
uv run python scripts/preview_preprocess.py --manifest local/captures/inputs.jsonl --grain 0
```

For a quick look at one photo (or a folder/glob), `compare_preprocess.py` writes an
input | preprocessed strip, optionally with the mask and an amplified difference panel, and
prints the before/after statistics:

```bash
uv run python scripts/compare_preprocess.py local/captures/IMG_1.jpg --diff --with-mask
uv run python scripts/compare_preprocess.py local/captures --diff
make compare PHOTOS=local/captures
```

Preview one image from the module itself:

```bash
uv run python -m zir_image_enhancer.reconstruction.preprocess input.jpg /tmp/pre --stats
```

## Knobs

`PreprocessSettings` (see the module for the full list and help):

- `enabled`, `white_balance`, `white_balance_max_gain`, `white_balance_method`,
  `flatten_illumination`, `auto_tone`, `denoise`, `sharpen`, `grain_sigma`, `jpeg_quality`
- `max_side`, `min_detail_side`, `detail_margin`, `crop_to_garment`, `garment_margin`
- `detail_crops`, `palette`

## What was added after the first `recon_plain` / `recon_pre` comparison

On 19 holdout garments the default preprocessing scored 58.57 against 60.87 for the untouched
photo, and the loss was almost entirely colour (0.589 vs 0.634): the reference already carried
the capture's warm cast and uneven light, and the model copied it faithfully instead of
studying it. Looking at the pairs, two steps attack exactly that, without touching the parts
that already work (the detail crops):

- **`white_balance_method="neutral_pixels"`** — gray-world trusts the whole scene, so a warm
  floor or wall was read as "the light" and whites came out cool, which is the wrong direction
  and the pattern in the worst garments (most of them had no garment mask, so the fallback
  averaged the entire frame). The neutral-pixel estimator measures only low-saturation, bright
  pixels, so it moves whites that really are off and leaves saturated garments alone.
- **`flatten_illumination`** — the differences the model reproduces most readily are shadows,
  not hues, because a cast shadow looks like a fold. A mean-preserving, luminance-only division
  by a heavily blurred copy of the image removes the shadow the way a studio light removes it,
  and only flattens inside the garment mask so the invented backdrop keeps its own falloff.

Both are off by default and controlled from every entry point, so they can be A/B tested
separately:

```bash
uv run python scripts/reconstruct.py --split holdout --run recon_np --wb-method neutral_pixels
uv run python scripts/reconstruct.py --split holdout --run recon_flat --flatten 0.25
uv run python scripts/reconstruct.py --split holdout --run recon_studio \
    --wb-method neutral_pixels --flatten 0.25
```

```bash
# eyeball the new steps for free, no paid calls
uv run python scripts/preview_preprocess.py --split holdout --limit 8 --wb-method neutral_pixels
uv run python scripts/preview_preprocess.py --split holdout --limit 8 --flatten 0.25
uv run python -m zir_image_enhancer.reconstruction.preprocess in.jpg /tmp/pre --flatten 0.25
```

The honest read of the visual comparison: the unprocessed photo is already a strong studio
reference, because nano-banana normalises exposure and background on its own. Preprocessing
should therefore *add information* (the detail crops) and fix what reads as real garment
shading (the cast shadow), not restyle colour globally. Tone-mapping the whole frame and
gray-world with no mask are the two steps most likely to be doing harm.

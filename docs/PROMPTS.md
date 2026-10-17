# Prompt history

Every prompt we sent to a model, its full text, and the runs it was used in. Scores are the official scorer on our holdout (38 pairs grouped by target) unless the row says otherwise; all rows are in `experiments/results.csv`. Gaps under about 1.5 points are within noise. Removed prompts can be recovered from the tag `exp-snapshot-2026-10-17` or the commits named below.

## Summary

| Prompt | Model | Status | Best result |
|---|---|---|---|
| `baseline` (official) | nano-banana 2.1 | reference only | 60.92 (portrait canvas) |
| `studio-white-v1` | nano-banana 2.1, Pro, FLUX.2, Seedream | removed | 61.23 (Pro) |
| `studio-style-v2` | nano-banana 2.1 | **in use** (base of the harness) | **61.82** |
| `studio-flat-v1` | nano-banana 2.1 | removed | 61.29 on 22 flat-lay pairs |
| Multi-photo wrapper | nano-banana 2.1 | removed | 60.71 on 43 pairs (first photo only: 61.46) |
| Detail close-up wrapper | nano-banana 2.1 | **in use** (named detail photos) | -1.81 with unnamed close-ups |
| Spec-written prompt | nano-banana 2.1 | removed | 58.84 |
| Harness lines | nano-banana 2.1 | **in use**: proportions, studio light, details, pieces | 61.77 on 24 pairs |
| Flatten, length, vivid colour, single zip lines | nano-banana 2.1 | removed | 60.40 (v2), 60.72 (flatten only, v3) |
| Relight | nano-banana 2.1 | removed | -1.46 on 8 pairs |
| Triage | Gemini 2.5 Flash | removed (the spec now picks the front) | 61.67 on shuffled photos |
| Spec | Gemini 2.5 Flash | **in use** | |
| Gate | Gemini 2.5 Flash | **in use**, with the jury's criteria since v4 | v3 (jury mode): 61.76 |

## 1. Generation prompts (one-shot)

### `baseline`

Copied from the official hosted baseline. Sent verbatim, whatever the number of photos.

```text
Turn this photograph of a single garment into a clean e-commerce flat-lay packshot. Lay the garment perfectly flat and centred on a plain, uniform light-grey background (RGB 245,245,245), seen straight from above with no perspective distortion. Use soft, even studio lighting with no visible shadow on the background. Keep the exact same garment, colour, material, pattern and proportions as the original. Remove the person, mannequin, hanger, floor, furniture and background; do not add props, text, logos, watermarks or reflections. Output only the final packshot image.
```

| Run | Setup | Score |
|---|---|---|
| official dev reference | hosted baseline, dev split | 68.71 (dev, not holdout) |
| `holdout_nb21_baseline` | nano-banana 2.1, first photo | 60.22 |
| `holdout_nb21p_baseline` | same on a 2:3 portrait canvas | 60.92 |

### `studio-white-v1`

Written for the scorer: whole garment visible, white background with no shadow (u2net can swallow a shadow into the mask), colour kept.

```text
Create a professional e-commerce flat-lay packshot of the garment in the {source}. {references}Show the whole garment, front side up, laid perfectly flat and symmetric, seen straight from above, centred with a clear empty margin on every side. Arrange sleeves, straps and legs neatly as in a catalogue flat-lay. Pure white seamless background, soft even studio lighting, no shadows, no reflections. Keep exactly the same garment: the same colour and shade, fabric texture, pattern, print, stitching, buttons and proportions. Remove any person, mannequin, hanger, price tag, floor and props. Do not add text, logos or watermarks. Output only the final packshot image.
```

| Run | Setup | Score |
|---|---|---|
| `holdout_nb21p_v1_all` | nano-banana 2.1 portrait, all photos | 61.04 |
| `holdout_nbpro_v1_all` | nano-banana Pro | 61.23 (2.5x the price) |
| `holdout_seedream45_v1_all` | Seedream 4.5 | 49.59 |
| `holdout_flux2_v1_all` | FLUX.2 [dev] | 43.36 |
| `holdout_nb21p_v1_all_cut` | cut-out photos as input | 57.16 |
| `holdout_nb21p_v1_post` | output cut out and re-framed on grey | 60.13 |

### `studio-style-v2` (in use)

Generic catalogue conventions written as rules per garment type (no target images used). In `results.csv` it appears under its old name `sartiq-style-v2`.

```text
Turn the {source} of this garment into a fashion catalogue packshot. {references}Show only the garment, front view, centred and symmetric, laid flat and shot straight from above on a plain light-grey studio background (RGB 240,240,240), with soft, even, diffused light, a subtle natural fabric volume and no cast shadow. Style it neatly: long sleeves hang straight down along the sides of the body, slightly apart from it; short sleeves point slightly downwards; buttons and zips are closed and the collar is tidy; trousers, jeans and shorts have both legs straight down and parallel; dresses and skirts hang straight with a flat hem. The whole garment is visible with a margin on every side and fills most of a portrait frame. Keep exactly the same garment: colour and shade, fabric and texture, pattern, print, logos and small details, length and proportions. Remove any person, mannequin, hanger, price tag, floor and props, and never add accessories, other garments, text or watermarks.
```

| Run | Setup | Score |
|---|---|---|
| `holdout_nb21p_v2_all` | nano-banana 2.1 portrait (the one-shot pipeline) | **61.82** |
| `holdout_nb21p_v2_all_crop` | photo cropped around the garment | 60.50 |
| `holdout_nb21p_v2_all_denoise` | photo denoised | 59.95 |
| `holdout_qwen-2511_v2` | Qwen-Image-Edit-2511 | 54.37 |
| `holdout_qwen-2511-outfit_v2` | Qwen-2511 with the "Extract-Outfit" LoRA | 39.69 |
| `trainmulti_v2_first` | 43 multi-photo pairs, first photo | 61.46 |
| `trainmulti_v2_best` / `_firstusable` | CLIP picks the photo | 61.19 / 61.46 |
| `trainmulti_v2_triage_front` | Gemini picks the front | 61.70 (61.67 shuffled) |
| `trainflat_sartiq-style-v2` | 22 flat-lay pairs | 60.74 |

### `studio-flat-v1`

For photos that already follow our capture protocol (flat, from above, colour balanced). Asks the model to change as little as possible.

```text
The {source} already shows this garment laid flat and shot from above, with correct colours. {references}Turn it into a fashion catalogue packshot while changing as little as possible: keep the garment's layout, shape, proportions, exact colours, fabric texture, pattern, print, logos and small details as they are. Only straighten the perspective so it is seen exactly from above, centre it, smooth small wrinkles, tidy the collar, close buttons and zips if they are open, and make the light soft and even. Replace the surface under it with a plain light-grey studio background (RGB 240,240,240) with no cast shadow, and remove any card, tag, label, hand or object around it. The whole garment stays visible with a margin on every side, in a portrait frame. Never add accessories, other garments, text or watermarks.
```

| Run | Setup | Score |
|---|---|---|
| `trainflat_sartiq-flat-v1` | 22 flat-lay pairs | 61.29 (`studio-style-v2`: 60.74, +0.54, within noise) |

### Reference wrappers

Inserted at `{references}` when more than one photo is sent.

Multi-photo wrapper (removed):

```text
All {count} photos show the same single garment: the first is the main view, the others are other angles or close-ups. Use all of them to understand the garment, and output one front view.
```

| Run | Setup | Score |
|---|---|---|
| `trainmulti_v2_all` | 43 pairs, all photos | 60.71 (first photo only: 61.46) |
| `trainmulti_v2_filtered` | only full front views (CLIP) | 60.83 |

Detail close-up wrapper (in use: the harness sends named detail photos with it). The last sentence about colour was added after the first run.

```text
Image 1 shows the whole garment from the front. Images 2 to {last} are close-ups of details on the front of the same garment ({details}). Use the close-ups only for the shapes of those details: same letters, numbers and drawing, at the position and size they have in image 1. Take every colour and the brightness of the garment and of its details from image 1 only, because the close-ups may be lit differently. Output one front view.
```

| Run | Setup | Score |
|---|---|---|
| `trainmulti_v2_triage` | front + unnamed close-ups, without the colour sentence | -2.09 on the 11 garments with close-ups |
| `trainmulti_v2_triage_colourfix` | with the colour sentence | -1.81 on the same 11 |

## 2. Reconstruction prompts (built from the garment spec)

### Spec-written prompt (removed, commit `da22d1e`)

The whole prompt written from the spec, with optional measured hex colours. Template:

```text
Reconstruct this exact garment as a studio flat-lay packshot.
Reference images: image 1 is the whole garment from the front; image 2 is a crop of the <detail> from image 1; ...
The garment is a <category> (<construction>), made of <material>, <pattern>.
Preserve exactly: the real colour and shade, material and texture, pattern, proportions, construction and every detail below.
Measured garment colours in image 1: <hex> (<share>%), ... Keep these colours; take colour and brightness from image 1 only.   [only with colours]
- <detail> on the <location>: <description>.
Change only the pose, perspective, presentation and background needed for the packshot.
Present it as a fashion catalogue packshot: front view, centred and symmetric, laid flat and shot straight from above on a plain light-grey studio background (RGB 240,240,240), soft even diffused light, subtle natural fabric volume, no cast shadow. Long sleeves hang straight down along the sides of the body; short sleeves point slightly downwards; buttons and zips are closed; trouser legs are straight and parallel; dresses and skirts hang straight. The whole garment is visible with a margin, filling most of a portrait frame.
Avoid: inventing construction that is not visible, adding accessories, branding, text or other garments, smoothing away the fabric texture, and stylized lighting that changes the apparent colour. Remove any person, mannequin, hanger, price tag, sticker and floor.
```

| Run | Setup | Score |
|---|---|---|
| `holdout_recon_nocolour` | spec prompt + detail crops + gate | 58.84 |
| `holdout_recon_colour` | same + measured hex colours | 57.16 |
| `holdout_recon_style_gate` | `studio-style-v2` instead, spec only for the gate | 61.46 |

### Harness lines (added to `studio-style-v2`)

The harness sends `studio-style-v2` (with the detail wrapper when there are named detail photos) and appends these lines in order.

| Line | Status | Text |
|---|---|---|
| Proportions | in use | `Keep the proportions of image 1: garment length about {x} times the chest width; sleeves about {y} of the garment length; {notes}.` |
| Pieces | in use, only for sets | `This product is a set of {n} pieces: {names}. Here "the garment" means the whole set: show every piece in the same packshot, each laid flat in front view, arranged like a catalogue set (top above bottom, pairs side by side), at the same scale and not overlapping.` |
| Flatten, mannequin | removed | `Image 1 shows the garment on a mannequin: flatten it completely, as the empty garment lying flat. No bust, waist or hip shaping; side seams run straight from armhole to hem.` |
| Flatten, worn | removed | `Image 1 shows the garment worn by a person: remove the body and flatten the empty garment completely. No bust, waist or hip shaping; side seams run straight.` |
| Flatten, hanger | removed | `Image 1 shows the garment on a hanger: remove the hanger and its sag; the shoulders lie flat and level and the garment is not stretched downwards.` |
| Flatten, flat | removed | `Image 1 already shows the garment flat: keep its outline and only straighten it.` |
| Studio light | in use | `Light it like a high-key e-commerce studio photo: bright, soft and even light with lifted shadows and clean whites, so the real colours read clearly. It must not look darker, dimmer or more contrasty than a catalogue photo, whatever the light in the reference.` |
| Vivid colour | removed | `Colours look rich, clean and vivid, as under professional studio light: the garment's own hue, never washed out or greyish, and never oversaturated or shifted to another hue.` (appended to the studio light line) |
| Real length | removed | `Keep the garment's real length and width: never shorten, squash or widen it to fill the frame. A long garment (trousers, a maxi or midi dress or skirt) stays long and narrow, with empty background at its sides.` |
| Single zip puller | removed | `Closures exactly as in image 1: the same number of buttons, snaps and zips, and a zip has one single puller unless image 1 clearly shows two.` (two-way zips are real) |
| Details | in use | `Keep these details exactly: {name} ({location}): {description}; ...` |
| Repair | in use, second attempt only | `A previous attempt had these problems; fix them: {fix from the gate}` |

| Run | Lines used | Score |
|---|---|---|
| `holdout_harness` | proportions, studio light, details, repair | 61.77 on 24 pairs (one-shot pipeline on the same 24: 62.34; 14 failed when the balance ran out) |
| `flatten_test` | + flatten, vivid, length, single zip, pieces | 61.44 on 6 mannequin or worn pairs (harness: 61.20) |
| `length_test` | same | 59.87 on 8 long garments (harness: 60.23) |
| `holdout_harness_v2` | same | 60.40 on 38; on the 24 shared with `holdout_harness`: 60.50 against 61.77 |
| `holdout_harness_v3` | proportions, pieces, flatten, studio light, details, repair | 60.72 on 38; on the 24 shared with `holdout_harness`: 60.56 against 61.77 |
| `holdout_jury` (v3, the app's default) | one-shot `studio-style-v2` first; harness lines + fix only when the jury gate fails | 61.76 on 38 (one-shot: 61.82); 4 garments repaired: t0314 +10.4, t0042 -0.6, t0087 -2.6, t0327 -3.7 |

The length line made tops longer and narrower than their targets. Without it (v3) the score stayed low: the flatten line hurt mannequin shots in particular (t0103 -3.6, t0105 -6.8, where a neck scarf turned into a bow, t0327 -5.4), so it was removed too. The vivid line lowered the colour score slightly (0.646 to 0.634 on the 24 shared pairs).

### Relight (removed)

Second pass with the finished packshot as the only image.

```text
Change only the lighting of this packshot: make it a high-key e-commerce studio photo with bright, soft, even light, neutral white balance, lifted shadows and a clean light-grey background (RGB 240,240,240). Do not change the garment's shape, size, position, folds, pattern, print, details or its hue. Output only the relit packshot.
```

| Run | Setup | Score |
|---|---|---|
| `relight_test` | pipeline outputs relit, 8 pairs | 58.76 against 60.22 (4 of 8 wins), barely visible |
| `flatten_relight_test` | harness + relight, 6 pairs | 61.12 against 61.44 without it |

## 3. Vision prompts (Gemini 2.5 Flash)

### Triage (removed, version v2)

Picked the front photo for the one-shot pipeline. The spec prompt now does this. v2 added the rule that stickers and tags do not count as details.

```text
System: You check phone photos of one single garment for an e-commerce packshot pipeline. Answer with one JSON object and nothing else.

There are {count} photos of the same garment, numbered 1 to {count} in the order given.

Return this JSON object:
{
  "front_index": <number of the photo that shows the WHOLE garment from the FRONT most clearly>,
  "photos": [
    {"index": <n>, "view": "<front|back|detail_front|detail_other|label|packaging|other>",
      "whole_garment": <true if the whole garment is visible>}
  ],
  "fine_details": <true if the FRONT has small text, logos, numbers or detailed prints that are
                   easy to get wrong when the garment is redrawn>,
  "details": "<short description of those front details, or empty>"
}

Rules:
- The front is the side a buyer sees first: neck opening, buttons or zip, chest print or logo.
  Do not choose a back view as the front, even if the back has a bigger print.
- "detail_front" is a close-up of a print, logo, text or embroidery that is on the FRONT.
- Price tags, care labels and barcodes are "label"; a garment inside a bag is "packaging".
- "fine_details" and "details" only count what is part of the garment itself (printed, woven,
  embroidered or sewn on). Stickers, price tags, size stickers and hang tags do not count:
  they will be removed from the packshot.
```

| Run | Setup | Score |
|---|---|---|
| `trainmulti_v2_triage_front` | 43 pairs, Gemini picks the front | 61.70 (first photo: 61.46) |
| `trainmulti_v2_triage_front_shuffled` | photos in random order | 61.67 (first photo: 60.22) |

### Spec (in use)

One call with all the photos. Versions: v1 (`da22d1e`) had front, roles, box, category, construction, material, pattern, details and uncertain; v2 (`1fb2f88`) added proportions and photo lighting; v3 (current) added photo setup, pieces and accessories.

```text
System: You describe one single product (a garment, or an accessory such as glasses, a bag or a belt) from phone photos for a packshot reconstruction pipeline. Answer with one JSON object and nothing else. Only describe what the photos show.

There are {count} photos of the same garment, numbered 1 to {count}.
{known_roles: "The user labelled some photos: Photo 1 is the front. ..."}
Return this JSON object:
{
  "front_index": <photo that shows the WHOLE garment from the FRONT most clearly>,
  "roles": [{"index": <n>, "role": "<front|back|detail|label|packaging|other>"}],
  "garment_box": [ymin, xmin, ymax, xmax],
  "category": "<e.g. crew-neck T-shirt, wide-leg trousers, cropped cardigan, sunglasses>",
  "pieces": ["<each separate piece of this product, e.g. bikini top, bikini bottom>"],
  "construction": "<neckline, closures (number of buttons, zip), sleeves, pockets, hem, length>",
  "material": "<fabric and texture, e.g. ribbed knit, washed denim, satin>",
  "pattern": "<solid, stripes, print..., with colours named>",
  "details": [
    {"name": "<short name>", "location": "<where on the garment>",
      "description": "<exact text or shape, colours>",
      "box": [ymin, xmin, ymax, xmax]}
  ],
  "proportions": {
    "body_length_to_width": <garment length (shoulder to hem) / chest width on photo front_index>,
    "sleeve_to_body_length": <sleeve length / garment length, or null without sleeves>,
    "notes": "<where sleeves end and where the hem falls, e.g. sleeves end at the hem>"
  },
  "photo_lighting": "<the light in the photos, e.g. warm indoor light, dim, flash, daylight>",
  "photo_setup": "<how photo front_index shows the garment: flat|hanger|mannequin|worn|other>",
  "uncertain": ["<things the photos do not show clearly>"]
}

Rules:
- The front has the neck opening, buttons or zip, chest print or logo. Never pick a back view
  as the front.
- "details": only things on the FRONT that are part of the garment (printed, woven,
  embroidered, sewn on: logos, text, prints, buttons, trims). At most 5. Not stickers, price
  tags or hang tags. "box" is on photo front_index, in 0-1000 coordinates; null if unsure.
- "garment_box": the garment only (no hanger, mannequin or person) on photo front_index,
  in 0-1000 coordinates.
- Quote any text exactly as written; say "unreadable" if you cannot read it.
- "pieces": a product sold as a set (bikini, two-piece set, pyjamas, pair of socks) lists
  every piece; a single garment has one piece. Clothes only worn with it (jeans under a top),
  the mannequin and the hanger are not pieces.
```

A free check showed that the spec's length-to-width estimate predicts the target shape worse than the generated image already does (correlation 0.80 against 0.85), so it is only used as a soft hint in the prompt and the gate.

### Gate (in use)

Compares the real front photo (image 1) with the packshot (image 2). Versions: v1 (`da22d1e`) checked same garment, front view, colour, extra items, details and invented text; v2 (`1fb2f88`) added proportions and lighting; v3 added pieces; v4 (current) added the jury's criteria: crop, logos and text, closures, seams and texture, with crop and logos or text counted as severe. A flat-shape check, a "washed out / vivid colours" condition on lighting and a closures check were tried in v3 and removed.

```text
System: You are a strict quality checker for e-commerce packshots. Compare the packshot with the real photo of the same garment. Answer with one JSON object and nothing else.

Image 1 is the real photo of the garment (front). Image 2 is a generated packshot that
must show the SAME garment, flat, front view, on a plain light background.

Garment: {category}. Product pieces: {pieces}. {proportions}
Details that must be present and identical:
{details}

Return:
{
  "same_garment": <true if it is the same type of product with the same construction>,
  "front_view": <true if image 2 shows the front>,
  "colour_ok": <false only if the colour or shade clearly changed>,
  "proportions_ok": <false if sleeve length or garment length relative to width clearly
                     differs from image 1>,
  "lighting_ok": <false if image 2 looks dark, dim or contrasty instead of a bright, soft,
                  even e-commerce studio photo>,
  "extra_items": <true if there is a person, mannequin, hanger, tag, props, or a garment
                  that is not one of the product pieces>,
  "all_pieces": <false if one of the product pieces is missing>,
  "whole_visible": <false if any part of the garment is cut off by the image border>,
  "logos_text_ok": <false if a logo, badge or printed or woven text of image 1 is missing,
                    misspelled, moved or redrawn differently>,
  "closures_ok": <false if zips, buttons or snaps differ in number, type or place from image 1>,
  "seams_ok": <false if seams, stitching, hems, pockets or panels clearly differ from image 1>,
  "texture_ok": <false if the fabric texture or knit clearly differs (smoothed away, a
                 different weave or knit)>,
  "details": [{"name": "<detail>", "ok": <true if present, same shape/text, same place>}],
  "invented": <true if image 2 adds text, logos or prints that image 1 does not have>,
  "fix": "<short instructions to fix the problems, empty if none>"
}
Lighting and background differences are expected; judge only the garment itself.
```

Severe problems (count double when choosing between the first attempt and the repair): different garment, not the front, extra items, missing piece, invented text or logo, cropped, logo or text differs.

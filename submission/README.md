# Submission

Both archives use the official layout `d3a24/<garment_id>/front.png`, were checked with `scripts/make_submission.py` (0 errors) and were made by the pipeline in this repo: v3 with nano-banana Pro.

| File | Garments | Score |
|---|---|---|
| `d3a24_test_v3_pro.zip` | the 15 on-site garments, `g01`–`g15` | scored by the organisers (references are private) |
| `d3a24_dev_v3_pro.zip` | the 27 dev garments, `d001`–`d027` | 69.77 on the dev leaderboard |

To rebuild them (every paid call is cached, so a rerun is free):

```bash
uv run zir-run --manifest data/sartiq_neural_wave_2026/test/inputs.jsonl --out submissions --team d3a24 --model nano-banana-pro
uv run --extra scorer python scripts/make_submission.py --submissions submissions --team d3a24 --dataset data/sartiq_neural_wave_2026 --zip
```

Use `dev/inputs.jsonl` in the first command for the dev archive.

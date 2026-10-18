# Self-evaluation harness around the vendored, frozen Sartiq scorer.
#
# Everything goes through uv. `make install` creates .venv with every dependency:
# the `scorer` extra (official scorer + torch) and the default `dev` group (ruff,
# mypy, pytest, pre-commit). Use `uv run` directly for anything not wrapped here.
#
# Typical loops:
#   make holdout predict predict-identity          # self-eval data + predictions
#   make eval TEAM=zir && make eval TEAM=identity  # compare against the floor
#   make run validate                              # build + check the dev submission

UV         ?= uv
RUN         = $(UV) run
RUN_SCORER  = $(UV) run --extra scorer

DATASET    ?= data/sartiq_neural_wave_2026
TEAM       ?= zir
FRAC       ?= 0.1
SEED       ?= 0
JOBS       ?= 1
LIMIT      ?=
NB_TEAM    ?= hosted-edit
NB_PROVIDER ?= fal
NB_LIMIT   ?=

# API keys: the shell environment wins; otherwise fall back to the gitignored
# .env so a hosted run is a single `make` away. Never committed.
FAL_KEY          ?= $(shell sed -n 's/^FAL_KEY=//p' .env 2>/dev/null | head -1)
GEMINI_API_KEY   ?= $(shell sed -n 's/^GEMINI_API_KEY=//p' .env 2>/dev/null | head -1)
GOOGLE_API_KEY   ?= $(shell sed -n 's/^GOOGLE_API_KEY=//p' .env 2>/dev/null | head -1)
OPENAI_API_KEY   ?= $(shell sed -n 's/^OPENAI_API_KEY=//p' .env 2>/dev/null | head -1)
IMAGE_API_KEY    ?= $(shell sed -n 's/^IMAGE_API_KEY=//p' .env 2>/dev/null | head -1)
export FAL_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENAI_API_KEY IMAGE_API_KEY

.PHONY: install holdout holdout-train predict predict-identity predict-nanobanana eval \
        baseline run validate baseline-nanobanana-dry baseline-nanobanana app \
        reconstruct score sheet gallery fmt lint type test check clean

install:  ## Create/refresh .venv with every dependency (scorer + dev)
	$(UV) sync --all-extras

holdout:  ## Carve a self-evaluation holdout out of the example pairs
	$(RUN) python scripts/make_holdout.py \
		--pairs $(DATASET)/train/pairs.jsonl \
		--frac $(FRAC) --seed $(SEED) --out holdout \
		--dataset $(DATASET) --group-by-target

predict:  ## Run the pipeline over the holdout into predictions/<TEAM>/
	$(RUN) python -m zir_image_enhancer.cli \
		--manifest holdout/val.jsonl \
		--dataset $(DATASET) --out predictions --team $(TEAM)

predict-identity:  ## Identity predictions on the holdout (the floor to beat)
	$(RUN) python tools/baselines/baseline_identity.py \
		--manifest holdout/val.jsonl \
		--out predictions --team identity --dataset $(DATASET)

predict-nanobanana:  ## Hosted (fal.ai) predictions on the holdout (spends budget)
	$(RUN) python tools/baselines/baseline_nanobanana.py \
		--manifest holdout/val.jsonl \
		--out predictions --team $(NB_TEAM) --provider $(NB_PROVIDER) \
		--dataset $(DATASET) $(if $(NB_LIMIT),--limit $(NB_LIMIT),)

eval:  ## Score predictions/<TEAM> against the holdout with the official scorer
	$(RUN_SCORER) python scripts/eval_val.py \
		--pairs holdout/val.jsonl --preds predictions \
		--team $(TEAM) --out val_out --dataset $(DATASET) --jobs $(JOBS) \
		$(if $(LIMIT),--limit $(LIMIT),)

baseline:  ## Identity baseline over dev into the submission layout
	$(RUN) python tools/baselines/baseline_identity.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team identity --dataset $(DATASET)

run:  ## Run the pipeline over dev into submissions/<TEAM>/
	$(RUN) python -m zir_image_enhancer.cli \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--dataset $(DATASET) --out submissions --team $(TEAM)

validate:  ## Check submissions/<TEAM>/ layout and images before handing in
	$(RUN) python scripts/make_submission.py \
		--submissions submissions --team $(TEAM) --dataset $(DATASET)

baseline-nanobanana-dry:  ## Preview hosted-baseline requests (no key, no spend)
	$(RUN) python tools/baselines/baseline_nanobanana.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team $(NB_TEAM) --provider $(NB_PROVIDER) \
		--dataset $(DATASET) --dry-run $(if $(NB_LIMIT),--limit $(NB_LIMIT),)

baseline-nanobanana:  ## Run the hosted (fal.ai) baseline over dev (spends budget)
	$(RUN) python tools/baselines/baseline_nanobanana.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team $(NB_TEAM) --provider $(NB_PROVIDER) \
		--dataset $(DATASET) $(if $(NB_LIMIT),--limit $(NB_LIMIT),)

app:  ## Launch the Gradio demo
	$(RUN) python -m zir_image_enhancer.app

reconstruct:  ## Experiment: run the pipeline over a split into local/runs/<RUN_NAME>. Set RUN_NAME=... SPLIT=...
	$(RUN) python scripts/reconstruct.py --split $(or $(SPLIT),holdout) --run $(RUN_NAME)

score:  ## Experiment: score local/runs/<RUN_NAME> on the holdout (see scripts/score_run.py)
	$(RUN_SCORER) python scripts/score_run.py --run $(RUN_NAME)

sheet:  ## Experiment: input/model/target contact sheets for runs (set RUNS="a b")
	$(RUN) python scripts/compare_sheet.py --runs $(RUNS)

gallery:  ## Experiment: per-garment input/output/target galleries (set RUNS="a b", optional)
	$(RUN) python scripts/gallery.py $(if $(RUNS),--runs $(RUNS),)

fmt:  ## Format and fix with ruff
	$(RUN) ruff check --fix .

lint:  ## Lint with ruff
	$(RUN) ruff check .

type:  ## Type-check with mypy
	$(RUN) mypy src scripts

test:  ## Run the test suite
	$(RUN) pytest

check: lint type test  ## Everything CI would run

clean:
	rm -rf holdout val_out val_smoke val_hosted predictions submissions

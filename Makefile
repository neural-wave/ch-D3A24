# Self-evaluation harness around the vendored, frozen Sartiq scorer.
#
# The scripts under scripts/ are copied unmodified from the participant starter
# repo, whose defaults assume a sibling `../sartiq_neural_wave_2026` dataset.
# Passing --dataset / --pairs explicitly here is what adapts them to this repo's
# layout, so the upstream code stays byte-identical (easy to re-vendor later).
#
# Typical loops:
#   make holdout predict predict-identity          # self-eval data + predictions
#   make eval TEAM=zir && make eval TEAM=identity  # compare against the floor
#   make run validate                              # build + check the dev submission

PYTHON      ?= python
UV          ?= uv pip install --system
DATASET     ?= data/sartiq_neural_wave_2026
TEAM        ?= zir
FRAC        ?= 0.1
SEED        ?= 0
JOBS        ?= 1
LIMIT       ?=
NB_TEAM     ?= hosted-edit
NB_PROVIDER ?= fal
NB_LIMIT    ?=

# API keys: the shell environment wins; otherwise fall back to the gitignored
# .env so a hosted run is a single `make` away. Never committed.
FAL_KEY          ?= $(shell sed -n 's/^FAL_KEY=//p' .env 2>/dev/null | head -1)
GEMINI_API_KEY   ?= $(shell sed -n 's/^GEMINI_API_KEY=//p' .env 2>/dev/null | head -1)
GOOGLE_API_KEY   ?= $(shell sed -n 's/^GOOGLE_API_KEY=//p' .env 2>/dev/null | head -1)
OPENAI_API_KEY   ?= $(shell sed -n 's/^OPENAI_API_KEY=//p' .env 2>/dev/null | head -1)
IMAGE_API_KEY    ?= $(shell sed -n 's/^IMAGE_API_KEY=//p' .env 2>/dev/null | head -1)
export FAL_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENAI_API_KEY IMAGE_API_KEY

.PHONY: install holdout predict predict-identity predict-nanobanana eval baseline \
        run validate baseline-nanobanana-dry baseline-nanobanana app clean

install:  ## Install this project plus the vendored scorer and its dependencies
	$(UV) -e ".[scorer]"

holdout:  ## Carve a self-evaluation holdout out of the example pairs
	$(PYTHON) scripts/make_holdout.py \
		--pairs $(DATASET)/train/pairs.jsonl \
		--frac $(FRAC) --seed $(SEED) --out holdout \
		--dataset $(DATASET) --group-by-target

predict:  ## Run the pipeline over the holdout into predictions/<TEAM>/
	$(PYTHON) -m zir_image_enhancer.cli \
		--manifest holdout/val.jsonl \
		--dataset $(DATASET) --out predictions --team $(TEAM)

predict-identity:  ## Identity predictions on the holdout (the floor to beat)
	$(PYTHON) tools/baselines/baseline_identity.py \
		--manifest holdout/val.jsonl \
		--out predictions --team identity --dataset $(DATASET)

predict-nanobanana:  ## Hosted (fal.ai) predictions on the holdout (spends budget)
	$(PYTHON) tools/baselines/baseline_nanobanana.py \
		--manifest holdout/val.jsonl \
		--out predictions --team $(NB_TEAM) --provider $(NB_PROVIDER) \
		--dataset $(DATASET) $(if $(NB_LIMIT),--limit $(NB_LIMIT),)

eval:  ## Score predictions/<TEAM> against the holdout with the official scorer
	$(PYTHON) scripts/eval_val.py \
		--pairs holdout/val.jsonl --preds predictions \
		--team $(TEAM) --out val_out --dataset $(DATASET) --jobs $(JOBS) \
		$(if $(LIMIT),--limit $(LIMIT),)

baseline:  ## Identity baseline over dev into the submission layout
	$(PYTHON) tools/baselines/baseline_identity.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team identity --dataset $(DATASET)

run:  ## Run the pipeline over dev into submissions/<TEAM>/
	$(PYTHON) -m zir_image_enhancer.cli \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--dataset $(DATASET) --out submissions --team $(TEAM)

validate:  ## Check submissions/<TEAM>/ layout and images before handing in
	$(PYTHON) scripts/make_submission.py \
		--submissions submissions --team $(TEAM) --dataset $(DATASET)

baseline-nanobanana-dry:  ## Preview hosted-baseline requests (no key, no spend)
	$(PYTHON) tools/baselines/baseline_nanobanana.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team $(NB_TEAM) --provider $(NB_PROVIDER) \
		--dataset $(DATASET) --dry-run $(if $(NB_LIMIT),--limit $(NB_LIMIT),)

baseline-nanobanana:  ## Run the hosted (fal.ai) baseline over dev (spends budget)
	$(PYTHON) tools/baselines/baseline_nanobanana.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team $(NB_TEAM) --provider $(NB_PROVIDER) \
		--dataset $(DATASET) $(if $(NB_LIMIT),--limit $(NB_LIMIT),)

app:  ## Launch the Gradio demo
	$(PYTHON) -m zir_image_enhancer.app

clean:
	rm -rf holdout val_out val_smoke val_hosted predictions submissions

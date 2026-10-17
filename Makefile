# Self-evaluation harness around the vendored, frozen Sartiq scorer.
#
# The scripts under scripts/ are copied unmodified from the participant starter
# repo, whose defaults assume a sibling `../sartiq_neural_wave_2026` dataset.
# Passing --dataset / --pairs explicitly here is what adapts them to this repo's
# layout, so the upstream code stays byte-identical (easy to re-vendor later).

PYTHON  ?= python
UV      ?= uv pip install --system
DATASET ?= data/sartiq_neural_wave_2026
TEAM    ?= zir
FRAC    ?= 0.1
SEED    ?= 0
JOBS    ?= 1

.PHONY: install holdout eval baseline validate run app clean

install:  ## Install this project plus the vendored scorer and its dependencies
	$(UV) -e ".[scorer]"

holdout:  ## Carve a self-evaluation holdout out of the example pairs
	$(PYTHON) scripts/make_holdout.py \
		--pairs $(DATASET)/train/pairs.jsonl \
		--frac $(FRAC) --seed $(SEED) --out holdout \
		--dataset $(DATASET) --group-by-target

eval:  ## Score predictions/$(TEAM) against the holdout with the official scorer
	$(PYTHON) scripts/eval_val.py \
		--pairs holdout/val.jsonl --preds predictions \
		--team $(TEAM) --out val_out --dataset $(DATASET) --jobs $(JOBS)

baseline:  ## Reproduce the identity baseline over dev into the submission layout
	$(PYTHON) tools/baselines/baseline_identity.py \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--out submissions --team identity --dataset $(DATASET)

run:  ## Run the pipeline over dev into submissions/<TEAM>/
	$(PYTHON) -m zir_image_enhancer.cli \
		--manifest $(DATASET)/dev/inputs.jsonl \
		--dataset $(DATASET) --out submissions --team $(TEAM)

validate:  ## Check the submission layout and images before handing in
	$(PYTHON) scripts/make_submission.py \
		--submissions submissions --team $(TEAM) --dataset $(DATASET)

app:  ## Launch the Gradio demo
	$(PYTHON) -m zir_image_enhancer.app

clean:
	rm -rf holdout val_out val_smoke val_hosted predictions submissions

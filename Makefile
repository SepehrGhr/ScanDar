PY ?= python

.DEFAULT_GOAL := help
.PHONY: help install data freeze preview sanity test train evaluate clean-pyc

CONFIG ?= configs/enhance.yaml

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install the package in editable mode, with dev extras
	$(PY) -m pip install -e ".[dev]"

data:  ## cache the scans and write data/splits.json
	$(PY) scripts/prepare_data.py

freeze:  ## generate the frozen synthetic evaluation sets, one per task
	$(PY) scripts/freeze_eval_sets.py

preview:  ## render generated samples into outputs/previews for inspection
	$(PY) scripts/preview_synth.py

sanity:  ## verify the environment, the data, the splits and the generator
	$(PY) scripts/sanity_checks.py

test:  ## run the unit tests
	$(PY) -m pytest

train:  ## train a model   (make train CONFIG=configs/enhance_residual.yaml)
	$(PY) train.py --config $(CONFIG)

evaluate:  ## score a trained model on the frozen synthetic buckets
	$(PY) evaluate.py --config $(CONFIG)

clean-pyc:  ## remove __pycache__ and stray bytecode
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
	find . -name '*.py[co]' -delete

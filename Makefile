PY ?= python

.DEFAULT_GOAL := help
.PHONY: help install data sanity test clean-pyc

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install the package in editable mode, with dev extras
	$(PY) -m pip install -e ".[dev]"

data:  ## cache the scans and write data/splits.json
	$(PY) scripts/prepare_data.py

sanity:  ## verify the environment, the data layout and the splits
	$(PY) scripts/sanity_checks.py

test:  ## run the unit tests
	$(PY) -m pytest

clean-pyc:  ## remove __pycache__ and stray bytecode
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
	find . -name '*.py[co]' -delete

SHELL := /bin/bash
ENV := srm
RUN := conda run --no-capture-output -n $(ENV)

.PHONY: setup lint test e2e

setup:
	conda env update -n $(ENV) -f environment.yml --prune || conda env create -n $(ENV) -f environment.yml

lint:
	$(RUN) ruff check srm tests

test:
	$(RUN) pytest -m "not e2e"

e2e:
	$(RUN) pytest -m e2e -v

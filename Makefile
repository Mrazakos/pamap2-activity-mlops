# Convenience targets. Heavy work always goes through Slurm (see README "Running on the cluster").
SHELL := /bin/bash
PY ?= python
MLFLOW_ENV := source slurm/mlflow_env.sh >/dev/null &&
MLFLOW_PORT ?= 5000

.PHONY: help setup setup-cluster lint test test-all repro repro-cpu docs mlflow-ui export-runs export-mlflow init-mlflow no-data-history

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## laptop: .venv with the pinned packages (CPU build of the same torch version) + git hooks
	$(PY) -m venv .venv
	sed -e 's/+cu126/+cpu/' -e 's#whl/cu126#whl/cpu#' requirements.txt > .venv/requirements-cpu.txt
	.venv/bin/python -m pip install -U pip && .venv/bin/python -m pip install -r .venv/requirements-cpu.txt
	.venv/bin/pre-commit install

setup-cluster:  ## deep21: conda env `mlops` from the exact lock file + git hooks
	source /shared/anaconda/anaconda/bin/activate && conda create -n mlops python=3.13 -y && \
	conda activate mlops && pip install -U pip && pip install -r requirements-lock.txt && pre-commit install

lint:  ## ruff + formatting + data/notebook guards
	ruff check .
	ruff format --check .
	$(PY) scripts/check_notebooks.py
	git ls-files -z | xargs -0 $(PY) scripts/check_no_data.py

test:  ## fast suite (model-behaviour tests skip without a trained model)
	$(PY) -m pytest -q

test-all:  ## full suite with coverage, as a CPU job (includes model tests if models/deep exists)
	mkdir -p slurm/logs && sbatch slurm/tests.sh

repro-cpu:  ## dvc repro up to the baseline in a CPU-only job
	mkdir -p slurm/logs && sbatch slurm/repro.sh baseline

repro:  ## full dvc repro (incl. train + evaluate) in a GPU job
	mkdir -p slurm/logs && sbatch --gres=gpu:rtx2080ti:1 --job-name=repro-gpu slurm/repro.sh

docs:  ## build the documentation site (strict)
	mkdocs build --strict

init-mlflow:  ## create the MLflow schema + experiment once (single process!)
	$(MLFLOW_ENV) $(PY) -m src.main init-mlflow

mlflow-ui:  ## UI of the interim SQLite store on deep21 (localhost only); tunnel: ssh -L 5000:127.0.0.1:5000 deep21
	source slurm/mlflow_env.sh >/dev/null && mlflow ui --backend-store-uri $$MLFLOW_INTERIM_URI --host 127.0.0.1 --port $(MLFLOW_PORT)

export-runs:  ## refresh the runs table in docs/results.md from MLflow
	$(MLFLOW_ENV) $(PY) scripts/export_runs.py

export-mlflow:  ## package mlflow.db + artifacts for the grader (outside the repo)
	bash scripts/export_mlflow.sh

no-data-history:  ## scan every commit on every ref for data/models/credentials
	$(PY) scripts/check_no_data.py --history

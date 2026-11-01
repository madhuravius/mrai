SHELL := /usr/bin/env bash
UV    ?= uv

.DEFAULT_GOAL := help

.PHONY: help sync build test lint fmt typecheck check run docker-build docker-run

%:
	@:

help:
	@printf 'Usage:\n\n'; grep -hE '^[A-Za-z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf " make %-15s #%s\n", $$1, $$2}'

sync: ## uv sync
	$(UV) sync

docker-build:
	docker build -t mrai .

docker-run:
	@mkdir -p ~/.mrai/renders
	@mkdir -p .mrai/renders
	docker run --rm -it \
		--user "$$(id -u):$$(id -g)" \
		--tmpfs /tmp \
		--tmpfs /home/nonroot/.cache \
		--cap-drop ALL \
		--security-opt=no-new-privileges \
		-v ~/.mrai:/tmp/mrai-data \
		-v "$$(pwd):$$(pwd):ro" \
		-v "$$(pwd)/.mrai:$$(pwd)/.mrai" \
		-w "$$(pwd)" \
		-e OPENROUTER_API_KEY \
		-e MRAI_DIR=/tmp/mrai-data \
		-e COLUMNS \
		mrai $(filter-out $@,$(MAKECMDGOALS))

test: ## uv run pytest
	$(UV) run pytest --cov=mrai --cov-report=term-missing -q

lint: ## ruff check + format --check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## ruff format + check --fix
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## pyright (venvPath .venv already configured)
	$(UV) run pyright

check: ## lint + typecheck + test
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

run: ## run agent (e.g. make run ARGS="where is slugify defined?")
	$(UV) run mrai $(ARGS)

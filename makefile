SHELL := /usr/bin/env bash
UV    ?= uv

.DEFAULT_GOAL := help

.PHONY: help sync build test lint fmt typecheck check run

help:
	@printf 'Usage:\n\n'; grep -hE '^[A-Za-z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf " make %-15s #%s\n", $$1, $$2}'

sync: ## uv sync
	$(UV) sync

build: ## hatchling wheel/sdist -> dist/
	$(UV) build

test: ## uv run pytest
	$(UV) run pytest

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

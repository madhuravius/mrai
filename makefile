SHELL := /usr/bin/env bash
UV    ?= uv

.DEFAULT_GOAL := help

.PHONY: help sync build test test-container lint fmt typecheck check run docker-build docker-run install

%:
	@:

help:
	@printf 'Usage:\n\n'; grep -hE '^[A-Za-z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf " make %-15s #%s\n", $$1, $$2}'

sync: ## uv sync
	$(UV) sync

install: docker-build ## build the image and install the ./ai wrapper to ~/.local/bin
	@mkdir -p ~/.local/bin
	@install -m 755 ai ~/.local/bin/ai
	@echo "ok  ~/.local/bin/ai"
	@[[ ":$${PATH}:" == *":$${HOME}/.local/bin:"* ]] || echo "⚠  add ~/.local/bin to your PATH"

docker-build: ## build a docker image that runs mrai
	docker build -t ghcr.io/madhuravius/mrai \
		--label org.opencontainers.image.version="$$(git describe --tags --always 2>/dev/null || echo dev)" .
	@rm -f "$${MRAI_HOME:-$$HOME/.mrai}/pinned-image"

docker-run: docker-build ## run the locally built image via ./ai (per-repo state under ~/.mrai/workspaces)
	./ai $(filter-out $@,$(MAKECMDGOALS))

test-container: docker-build ## check the container sandbox under ./ai's real flags
	@home=$$(mktemp -d); mkdir -p "$$home/workspaces/other-ws"; touch "$$home/mrai.yml"; \
	MRAI_HOME="$$home" MRAI_NO_UPDATE_CHECK=1 MRAI_PROBE_CANARY=leak MRAI_ENTRYPOINT=/bin/sh \
		./ai "$$PWD/tests/container/probe.sh"; rc=$$?; rm -rf "$$home"; exit $$rc

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

check: ## lint + typecheck + test + test-container
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) test-container

run: ## run agent (e.g. make run ARGS="where is slugify defined?")
	$(UV) run mrai $(ARGS)

.PHONY: help install test lint format test-integration test-state test-e2e test-e2e-resume test-cache test-alerts test-volumes test-training test-deployed-isolation

# A bare `make` lists the targets rather than running one.
.DEFAULT_GOAL := help

# Modal environment used by the integration targets. Override for your workspace:
#   make test-e2e MODAL_ENVIRONMENT=my-environment
MODAL_ENVIRONMENT ?= dev
export MODAL_ENVIRONMENT

##@ Setup

install: ## Install the package and all extras with uv
	uv sync --all-extras

##@ Development

test: install ## Run the unit tests
	uv run pytest tests

lint: install ## Check formatting and types (black, isort, flake8, mypy)
	uv run black --check src tests
	uv run isort --check-only src tests
	uv run flake8 src tests
	uv run mypy src

format: install ## Apply black and isort in place
	uv run black src tests
	uv run isort src tests

##@ Integration tests (deploy real Modal apps, need credentials, cost money)

test-integration: test-state test-e2e test-e2e-resume test-cache test-alerts test-volumes test-training test-deployed-isolation ## Run every integration target

test-e2e: install ## Run a workflow end to end
	uv run modal run tests/integration/test_e2e_workflow.py

test-e2e-resume: install ## Resume a workflow after a restart
	uv run modal run tests/integration/test_e2e_workflow_resume.py

test-cache: install ## Check step results are cached across restarts
	uv run modal run tests/integration/test_cache.py

test-alerts: install ## Send a real Slack alert
	uv run modal run tests/integration/test_e2e_alerts.py

test-training: install ## Exercise the @training checkpoint handoff
	uv run modal run tests/integration/test_training_decorator.py

test-state: install ## Round-trip workflow state through a volume
	uv run python -m tests.integration.test_state_roundtrip

test-volumes: install ## Create and tear down ephemeral volumes
	uv run python -m tests.integration.test_volumes_ephemeral

test-deployed-isolation: install ## Check state isolation between deployed app calls
	uv run python tests/integration/test_deployed_app_isolation.py

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} \
	     /^##@/ {printf "\n%s\n", substr($$0, 5); next} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

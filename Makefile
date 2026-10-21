.PHONY: help install test lint format testing-environment test-integration test-state test-e2e test-e2e-resume test-cache test-alerts test-training test-deployed-isolation test-class-methods test-map-restart test-map-exceptions

# A bare `make` lists the targets rather than running one.
.DEFAULT_GOAL := help

# The integration targets run in the Modal environment named by MODAL_TESTING_ENVIRONMENT,
# which must be set explicitly, e.g. `export MODAL_TESTING_ENVIRONMENT=my-scratch-env`.
INTEGRATION_RUN = MODAL_ENVIRONMENT=$(MODAL_TESTING_ENVIRONMENT) uv run

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

testing-environment:
	@test -n "$(MODAL_TESTING_ENVIRONMENT)" || { \
		echo "Set MODAL_TESTING_ENVIRONMENT to the Modal environment the integration tests should run in." >&2; \
		exit 1; }

test-integration: testing-environment test-state test-e2e test-e2e-resume test-cache test-alerts test-training test-deployed-isolation test-class-methods test-map-restart test-map-exceptions ## Run every integration target

test-e2e: testing-environment install ## Run a workflow end to end
	$(INTEGRATION_RUN) modal run tests/integration/test_e2e_workflow.py

test-e2e-resume: testing-environment install ## Resume a workflow after a restart
	$(INTEGRATION_RUN) modal run tests/integration/test_e2e_workflow_resume.py

test-cache: testing-environment install ## Check step results are cached across restarts
	$(INTEGRATION_RUN) modal run tests/integration/test_cache.py

test-alerts: testing-environment install ## Send a real Slack alert
	$(INTEGRATION_RUN) modal run tests/integration/test_e2e_alerts.py

test-training: testing-environment install ## Exercise the @training checkpoint handoff
	$(INTEGRATION_RUN) modal run tests/integration/test_training_decorator.py

test-state: testing-environment install ## Round-trip workflow state through a volume
	$(INTEGRATION_RUN) python -m tests.integration.test_state_roundtrip

test-deployed-isolation: testing-environment install ## Check state isolation between deployed app calls
	$(INTEGRATION_RUN) python tests/integration/test_deployed_app_isolation.py

test-class-methods: testing-environment install ## Use @app.cls methods as workflow steps
	$(INTEGRATION_RUN) modal run tests/integration/test_e2e_class_method_steps.py

test-map-restart: testing-environment install ## Restart a workflow while ctx.map items are in flight
	$(INTEGRATION_RUN) modal run tests/integration/test_e2e_map_restart.py

test-map-exceptions: testing-environment install ## Collect failed ctx.map items with return_exceptions
	$(INTEGRATION_RUN) modal run tests/integration/test_e2e_map_return_exceptions.py

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} \
	     /^##@/ {printf "\n%s\n", substr($$0, 5); next} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# mypy: ignore-errors
"""Integration test: alerting with real Modal functions.

Tests both functionalities:
  1. Explicit alert() calls from regular functions and workflows.
  2. Automatic alerting on exceptions via @alert_on_error decorator
     and workflow(alert_on_error=True).

Run:  uv run modal run tests/integration/test_e2e_alerts.py
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars, and a 'slack-alerts-test'
Modal secret holding SLACK_WEBHOOK_URL:

    modal secret create slack-alerts-test SLACK_WEBHOOK_URL=https://hooks.slack.com/...
"""

import modal

from modal_workflows.workflows import alert, alert_on_error, workflow

from tests.integration.utilities import build_workflows_image  # isort: skip

image = build_workflows_image()

app = modal.App(
    name="test-alerts",
    image=image,
    secrets=[modal.Secret.from_name("slack-alerts-test")],
)


# -- Regular function: explicit alert() -----------------------------------------


@app.function()
def task_with_explicit_alert():
    """Sends an explicit alert and returns normally."""
    alert("🧪 [test] explicit alert() from a regular function", app_id=app.app_id)
    return "ok"


# -- Regular function: @alert_on_error ------------------------------------------


@app.function()
@alert_on_error(app)
def task_that_fails():
    """Raises an exception; @alert_on_error should fire before the error propagates."""
    raise ValueError("simulated failure in task_that_fails")


# -- Workflow: explicit alert() + alert_on_error=True ----------------------------


@workflow(app, alert_on_error=True)
def workflow_that_alerts_and_fails(ctx):
    """Sends an explicit alert, then raises to trigger workflow-level alert_on_error."""
    alert("🧪 [test] explicit alert() from a workflow", app_id=app.app_id)
    raise RuntimeError("simulated failure in workflow_that_alerts_and_fails")


# -- Entrypoint ------------------------------------------------------------------


@app.local_entrypoint()
def main():
    print("--- Test 1: explicit alert() from a regular function ---")
    result = task_with_explicit_alert.remote()
    assert result == "ok", f"Expected 'ok', got {result}"
    print("PASS\n")

    print("--- Test 2: @alert_on_error on a regular function ---")
    try:
        task_that_fails.remote()
    except Exception as e:
        assert "simulated failure" in str(e).lower(), f"Unexpected error: {e}"
        print(f"PASS  (caught: {e})\n")
    else:
        raise AssertionError("Expected ValueError to propagate")

    print("--- Test 3: workflow(alert_on_error=True) + explicit alert() ---")
    try:
        workflow_that_alerts_and_fails.remote()
    except Exception as e:
        assert "simulated failure" in str(e).lower(), f"Unexpected error: {e}"
        print(f"PASS  (caught: {e})\n")
    else:
        raise AssertionError("Expected RuntimeError to propagate")

    print("All alert integration tests passed!")
    print("Verify Slack received 4 messages:")
    print("  1. explicit alert from regular function")
    print("  2. alert_on_error from regular function failure")
    print("  3. explicit alert from workflow")
    print("  4. alert_on_error from workflow failure")

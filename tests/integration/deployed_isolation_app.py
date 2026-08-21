# mypy: ignore-errors
"""Modal app deployed for the state-isolation integration test.

Do not run directly — lifecycle is managed by test_deployed_app_isolation.py.
"""

import modal

from modal_workflows.workflows.decorator import workflow, workflow_function

from tests.integration.utilities import build_workflows_image  # isort: skip

APP_NAME = "test-deployed-isolation"
image = build_workflows_image()
app = modal.App(APP_NAME)


@workflow_function(app, image=image)
def fast_step(label: str) -> str:
    return f"{label}:done"


@workflow(app, image=image)
def run_workflow(ctx, scenario: str = "normal") -> dict:
    """Runs one step and returns the state name so callers can verify isolation."""
    result = ctx.step(fast_step, "fast", "data")
    state_name = ctx.state_name
    ctx.finish()
    return {"state_name": state_name, "result": result}


@workflow(app, image=image, retries=1)
def retry_workflow(ctx) -> dict:
    """Proves that Modal retries share state.

    First attempt: runs step_a, writes a marker, then raises to force a retry.
    Second attempt (same function_call_id → same state_name):
        - step_a is found in cache and skipped,
        - marker exists so the raise is bypassed,
        - step_b runs and the result is returned.

    If the retry used a *different* state_name the marker would not exist,
    the workflow would raise again, exhaust retries, and the test would fail.
    """
    step_a = ctx.step(fast_step, "step_a", "data")

    marker = ctx.artifacts_run_path / "retry_done.marker"
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("done")
        ctx.commit_artifacts_volume()
        raise RuntimeError("Forced failure — Modal will retry with the same function_call_id.")

    step_b = ctx.step(fast_step, "step_b", step_a)

    # Cleanup so the test does not leak state/artifacts across CI runs.
    marker.unlink()
    ctx.commit_artifacts_volume()
    state_name = ctx.state_name
    ctx.finish()

    return {
        "state_name": state_name,
        "step_a": step_a,
        "step_b": step_b,
        "retried": True,
    }

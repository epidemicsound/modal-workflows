# mypy: ignore-errors
"""Modal app deployed for the state-isolation integration test.

Do not run directly — lifecycle is managed by test_deployed_app_isolation.py.
"""

import hashlib

import modal

from modal_workflows.workflows.decorator import workflow

from tests.integration.utilities import build_workflows_image  # isort: skip

APP_NAME = "test-deployed-isolation"
image = build_workflows_image()
app = modal.App(APP_NAME)


@app.function(image=image)
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
    """Raises once after step_a; the retry only succeeds if it shares the first attempt's state."""
    step_a = ctx.step(fast_step, "step_a", "data")

    marker_name = f"test-retry-marker-{hashlib.sha256(ctx.state_name.encode()).hexdigest()[:32]}"
    marker = modal.Dict.from_name(marker_name, create_if_missing=True)
    if "failed_once" not in marker:
        marker["failed_once"] = True
        raise RuntimeError("Forced failure — Modal will retry with the same function_call_id.")

    try:
        step_b = ctx.step(fast_step, "step_b", step_a)
        state_name = ctx.state_name
        ctx.finish()
    finally:
        modal.Dict.objects.delete(marker_name)

    return {
        "state_name": state_name,
        "step_a": step_a,
        "step_b": step_b,
        "retried": True,
    }

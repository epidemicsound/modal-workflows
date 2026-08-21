# mypy: ignore-errors
"""Integration test: end-to-end workflow with real Modal functions.

Run:  uv run modal run tests/integration/test_e2e_workflow.py
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

import modal

from modal_workflows.workflows import workflow, workflow_function

from tests.integration.utilities import build_workflows_image  # isort: skip

image = build_workflows_image()
app = modal.App("test-e2e", image=image)


@workflow_function(app)
def add(a: int, b: int) -> int:
    return a + b


@workflow_function(app)
def double(x: int) -> int:
    return x * 2


@workflow(app)
def integration_test(ctx):
    sum_result = ctx.step(add, "add", 3, 4)
    doubled = ctx.map(double, [1, 2, 3], step_id="double")
    ctx.finish()
    return {"sum": sum_result, "doubled": doubled}


@app.local_entrypoint()
def main():
    result = integration_test.remote()

    assert result["sum"] == 7, f"Expected sum=7, got {result['sum']}"
    assert result["doubled"] == [2, 4, 6], f"Expected [2,4,6], got {result['doubled']}"

    print("PASS: e2e workflow")

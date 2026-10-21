# mypy: ignore-errors
r"""Integration test: end-to-end workflow with real Modal functions.

Run:  make test-e2e
  or: MODAL_ENVIRONMENT=$MODAL_TESTING_ENVIRONMENT \
        uv run modal run tests/integration/test_e2e_workflow.py
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

import modal

from modal_workflows.workflows import workflow
from modal_workflows.workflows.state import STATE_DICT_NAME_PREFIX, _stable_short_name

from tests.integration.utilities import build_workflows_image  # isort: skip

image = build_workflows_image()
app = modal.App("test-e2e", image=image)


@app.function()
def add(a: int, b: int) -> int:
    return a + b


@app.function()
def double(x: int) -> int:
    return x * 2


@workflow(app)
def integration_test(ctx):
    sum_result = ctx.step(add, "add", 3, 4)
    doubled = ctx.map(double, [1, 2, 3], step_id="double")
    ctx.finish()
    return {"sum": sum_result, "doubled": doubled, "state_name": ctx.state_name}


def state_dict_exists(state_name: str) -> bool:
    name = f"{STATE_DICT_NAME_PREFIX}{_stable_short_name(state_name)}"
    try:
        modal.Dict.from_name(name).hydrate()
    except modal.exception.NotFoundError:
        return False
    return True


@app.local_entrypoint()
def main():
    result = integration_test.remote()

    assert result["sum"] == 7, f"Expected sum=7, got {result['sum']}"
    assert result["doubled"] == [2, 4, 6], f"Expected [2,4,6], got {result['doubled']}"
    assert not state_dict_exists(result["state_name"]), "finish() should delete the state Dict"

    print("PASS: e2e workflow")

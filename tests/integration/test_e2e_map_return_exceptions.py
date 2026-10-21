# mypy: ignore-errors
r"""
Integration test: ctx.map(return_exceptions=True) against real Modal.

One item in a batch genuinely fails - verifies the failure comes back as a
plain-dict failure snapshot instead of aborting the whole map, the other items
still succeed, and that a later, unrelated restart doesn't re-run the
already-cached failure.

Run:  make test-map-exceptions
  or: MODAL_ENVIRONMENT=$MODAL_TESTING_ENVIRONMENT \
        uv run modal run tests/integration/test_e2e_map_return_exceptions.py
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

from __future__ import annotations

import modal

from modal_workflows.workflows import WorkflowContext, is_workflow_item_failure, workflow

from tests.integration.utilities import (  # isort: skip
    build_workflows_image,
    resolve_workflow_restarts,
)

image = build_workflows_image()
app = modal.App("test-map-return-exceptions", image=image)

COUNTER_DICT_NAME = "test-map-return-exceptions-counts"


@app.function()
def maybe_fail(item_id: int) -> str:
    counter = modal.Dict.from_name(COUNTER_DICT_NAME, create_if_missing=True)
    key = str(item_id)
    counter[key] = (counter[key] if counter.contains(key) else 0) + 1
    if item_id == 2:
        raise ValueError(f"deliberate failure for item {item_id}")
    return f"item-{item_id}-ok"


@app.function()
def sleep_for_seconds(seconds: int) -> str:
    import time

    time.sleep(seconds)
    return f"slept-{seconds}"


@workflow(app, self_managed_timeout=15)
def map_with_failure_and_restart(ctx: WorkflowContext) -> list:
    results = ctx.map(maybe_fail, [0, 1, 2, 3], "maybe_fail", return_exceptions=True)
    # Force a restart *after* the map is fully cached, to check that a later,
    # unrelated restart doesn't cause the already-cached failure to be re-run.
    ctx.spawn(sleep_for_seconds, "sleep_to_force_restart", 20).get()
    ctx.finish()
    return results


@app.local_entrypoint()
def main() -> None:
    counter = modal.Dict.from_name(COUNTER_DICT_NAME, create_if_missing=True)
    counter.clear()

    call = map_with_failure_and_restart.spawn()
    result, retries = resolve_workflow_restarts(
        call,
        max_retries=10,
        retry_message="Workflow restarted, waiting for resumed run...",
        error_message="Workflow exceeded max restart retries.",
    )
    assert retries > 0, "Expected at least one workflow restart."

    assert len(result) == 4, f"Expected 4 results, got {len(result)}"
    for i, item_result in enumerate(result):
        if i == 2:
            assert is_workflow_item_failure(
                item_result
            ), f"Expected item 2 to be a failure snapshot, got {item_result!r}"
            assert "deliberate failure" in item_result["message"]
            assert item_result["exc_type"].endswith("ValueError")
        else:
            assert item_result == f"item-{i}-ok", f"Expected item-{i}-ok, got {item_result!r}"

    count_for_item_2 = counter["2"] if counter.contains("2") else 0
    assert count_for_item_2 == 1, (
        f"Expected maybe_fail(2) to run exactly once (its failure cached across the "
        f"restart, not re-run), but it ran {count_for_item_2} times"
    )

    print("PASS: map with one failing item degrades gracefully and isn't re-run after restart")

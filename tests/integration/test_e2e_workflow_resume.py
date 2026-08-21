# mypy: ignore-errors
"""
Runs a workflow with a short self-managed timeout to force at least one restart,
then validates the final output still contains every expected step result.
"""

from __future__ import annotations

import modal

from modal_workflows.workflows import WorkflowContext, workflow, workflow_function

from tests.integration.utilities import (  # isort: skip
    build_workflows_image,
    resolve_workflow_restarts,
)

image = build_workflows_image()
app = modal.App("test-workflow-resume", image=image)


@workflow_function(app)
def process_item(item_id: int) -> str:
    return f"item-{item_id}-done"


@workflow_function(app)
def sleep_for_seconds(seconds: int) -> str:
    import time

    time.sleep(seconds)
    return f"sleep-{seconds}-done"


@workflow(app, self_managed_timeout=15)
def resumable_workflow(ctx: WorkflowContext) -> dict[str, object]:
    results: list[str] = []

    for item_id in range(4):
        result = ctx.step(process_item, f"item_{item_id}", item_id)
        results.append(result)

    call = ctx.spawn(sleep_for_seconds, "sleep_spawn", 15)
    results.append(call.get())

    map_results = ctx.map(sleep_for_seconds, [1, 2, 1], step_id="sleep_map")
    results.extend(map_results)

    parallel_results = ctx.parallel(
        [
            (sleep_for_seconds, 2),
            (sleep_for_seconds, 1),
            (sleep_for_seconds, 2),
        ],
        step_id="sleep_parallel",
    )
    results.extend(parallel_results)

    ctx.finish()
    return {"count": len(results), "results": results}


@app.local_entrypoint()
def main() -> None:
    result = resumable_workflow.remote()
    result, retries = resolve_workflow_restarts(
        result,
        max_retries=10,
        retry_message="Workflow restarted, waiting for resumed run...",
        error_message="Workflow exceeded max restart retries.",
    )
    saw_restart = retries > 0

    assert saw_restart, "Expected at least one workflow restart."
    assert result["count"] == 11, f"Expected 11 results, got {result['count']}"

    expected = {
        "item-0-done",
        "item-1-done",
        "item-2-done",
        "item-3-done",
        "sleep-15-done",
        "sleep-1-done",
        "sleep-2-done",
    }

    assert expected.issubset(set(result["results"])), (
        "Missing expected results. "
        f"Expected at least {sorted(expected)}, got {sorted(set(result['results']))}"
    )

    print("PASS: e2e resumable workflow")

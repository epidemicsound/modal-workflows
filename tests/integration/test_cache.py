# mypy: ignore-errors
"""Integration test: exercises spawn/get, map, and parallel, then verifies cache contents."""

from __future__ import annotations

import modal

from modal_workflows.workflows import WorkflowContext, workflow, workflow_function

from tests.integration.utilities import build_workflows_image  # isort: skip

image = build_workflows_image()
app = modal.App("test-e2e-cache", image=image)


@workflow_function(app)
def add(a: int, b: int) -> int:
    return a + b


@workflow_function(app)
def double(x: int) -> int:
    return x * 2


@workflow_function(app)
def square(x: int) -> int:
    return x**2


@workflow(app)
def cache_test_workflow(ctx: WorkflowContext) -> dict:
    # step (sync)
    step_result = ctx.step(add, "step_add", 1, 2)

    # spawn + get (async)
    call = ctx.spawn(add, "spawn_add", 10, 20)
    spawn_result = call.get()

    # map
    map_results = ctx.map(double, [1, 2, 3], step_id="map_double")

    # parallel
    parallel_results = ctx.parallel(
        [
            (add, 5, 6),
            (square, 4),
        ],
        step_id="par_ops",
    )

    # verify cache BEFORE finish() clears it
    completed = ctx.state.completed_steps

    step_ids_to_check = {
        "step_add",
        "spawn_add",
        "map_double_0",
        "map_double_1",
        "map_double_2",
        "par_ops_0",
        "par_ops_1",
    }

    cache_checks = {}
    for step_id in step_ids_to_check:
        is_cached, result = ctx.state.get_cached_step_result(step_id)
        cache_checks[step_id] = {"is_cached": is_cached, "result": result}

    ctx.finish()

    return {
        "step_result": step_result,
        "spawn_result": spawn_result,
        "map_results": map_results,
        "parallel_results": parallel_results,
        "completed_steps": sorted(completed),
        "cache_checks": cache_checks,
    }


@app.local_entrypoint()
def main():
    result = cache_test_workflow.remote()

    # correctness
    assert result["step_result"] == 3, f"step: expected 3, got {result['step_result']}"
    assert result["spawn_result"] == 30, f"spawn: expected 30, got {result['spawn_result']}"
    assert result["map_results"] == [2, 4, 6], f"map: expected [2,4,6], got {result['map_results']}"
    assert result["parallel_results"] == [
        11,
        16,
    ], f"parallel: expected [11,16], got {result['parallel_results']}"

    # cache: step results
    expected_completed_steps = {
        "step_add",
        "spawn_add",
        "map_double_0",
        "map_double_1",
        "map_double_2",
        "par_ops_0",
        "par_ops_1",
    }

    actual_steps = set(result["completed_steps"])
    assert expected_completed_steps == actual_steps, (
        "cached steps mismatch: "
        f"expected {sorted(expected_completed_steps)}, got {sorted(actual_steps)}"
    )

    checks = result["cache_checks"]
    assert checks["step_add"] == {"is_cached": True, "result": 3}
    assert checks["spawn_add"] == {"is_cached": True, "result": 30}
    assert checks["map_double_0"] == {"is_cached": True, "result": 2}
    assert checks["map_double_1"] == {"is_cached": True, "result": 4}
    assert checks["map_double_2"] == {"is_cached": True, "result": 6}
    assert checks["par_ops_0"] == {"is_cached": True, "result": 11}
    assert checks["par_ops_1"] == {"is_cached": True, "result": 16}

    print("PASS: e2e cache verification (spawn, get, map, parallel)")

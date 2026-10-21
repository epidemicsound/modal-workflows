# mypy: ignore-errors
r"""
Checks that ``ctx.map`` executes every item exactly once across graceful workflow restarts,
with the final results complete and in order. Scenarios, each on its own workflow run:

- while waiting: the restart hits while slow items are in flight, which are re-attached by
  id and never spawned a second time.
- while spawning: the restart hits part way through spawning; spawns already in flight
  finish and have their call ids saved before the continuation is spawned, so it
  re-attaches them and spawns only the items never submitted.
- after finished maps: the restart follows each of several finished maps, whose call ids
  are cleared, so only the committed results stop the continuation from running them again.

A hard kill of the orchestrator is not covered: during spawning it can run up to 16 items
per crash twice by design (see ``WorkflowContext.map``).

All scenarios run, and every failing one is reported at the end.

Run:  make test-map-restart
  or: MODAL_ENVIRONMENT=$MODAL_TESTING_ENVIRONMENT \
        uv run modal run tests/integration/test_e2e_map_restart.py
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import modal

from modal_workflows.workflows import WorkflowContext, workflow

from tests.integration.utilities import (  # isort: skip
    build_workflows_image,
    resolve_workflow_restarts,
)

image = build_workflows_image()
app = modal.App("test-map-restart", image=image)

EXECUTION_KEY_PREFIX = "exec:"

WAIT_ITEM_COUNT = 18
WAIT_ITEM_SECONDS = 10

SPAWN_ITEM_COUNT = 120
SPAWN_RESTART_AFTER_SPAWNS = 20
SPAWN_ITEM_SECONDS = 5

AFTER_MAP_COUNT = 5
AFTER_MAP_ITEM_COUNT = 30


def record_execution(counter_name: str, item_id: int | str) -> None:
    key = f"{EXECUTION_KEY_PREFIX}{item_id}:{uuid.uuid4().hex}"
    modal.Dict.from_name(counter_name).put(key, time.time())


def executions_per_item(snapshot: dict) -> dict[str, int]:
    executed = [key.split(":")[1] for key in snapshot if key.startswith(EXECUTION_KEY_PREFIX)]
    return {item: executed.count(item) for item in set(executed)}


@contextmanager
def counter_dict(scenario: str) -> Iterator[str]:
    name = f"test-map-restart-{scenario}-{uuid.uuid4().hex[:8]}"
    modal.Dict.objects.create(name)
    try:
        yield name
    finally:
        modal.Dict.objects.delete(name)


def assert_each_item_executed_once(snapshot: dict, item_ids: list) -> None:
    actual = executions_per_item(snapshot)
    expected = {str(item_id): 1 for item_id in item_ids}
    assert actual == expected, f"Every item must execute exactly once, got {actual}"


@app.function(max_containers=2)
@modal.concurrent(max_inputs=WAIT_ITEM_COUNT)
def wait_item(item_id: int, counter_name: str) -> str:
    record_execution(counter_name, item_id)
    time.sleep(WAIT_ITEM_SECONDS)
    return f"item-{item_id}-done"


@workflow(app, self_managed_timeout=12, max_containers=2)
def restart_while_waiting_workflow(ctx: WorkflowContext, counter_name: str) -> dict[str, object]:
    results = ctx.map(wait_item, list(range(WAIT_ITEM_COUNT)), "slow_map", counter_name)
    ctx.finish()
    return {"results": results}


def check_restart_while_waiting() -> str:
    with counter_dict("waiting") as counter_name:
        call = restart_while_waiting_workflow.spawn(counter_name)
        result, retries = resolve_workflow_restarts(
            call,
            max_retries=10,
            retry_message="Workflow restarted mid-map, waiting for resumed run...",
        )
        snapshot = dict(modal.Dict.from_name(counter_name).items())

    assert retries > 0, "Expected at least one workflow restart during the map."
    expected = [f"item-{item_id}-done" for item_id in range(WAIT_ITEM_COUNT)]
    assert result["results"] == expected, f"Results incomplete or out of order: {result['results']}"
    assert_each_item_executed_once(snapshot, list(range(WAIT_ITEM_COUNT)))
    return f"{retries} restart(s)"


@app.function(max_containers=2, scaledown_window=5)
@modal.concurrent(max_inputs=SPAWN_ITEM_COUNT)
def spawn_item(item_id: int, counter_name: str) -> str:
    record_execution(counter_name, item_id)
    time.sleep(SPAWN_ITEM_SECONDS)
    return f"item-{item_id}-done"


@contextmanager
def restart_after_spawns(func, ctx: WorkflowContext, count: int) -> Iterator[None]:
    """Makes *ctx* due for a restart once *func* has been spawned *count* times."""
    original_spawn = func.spawn
    remaining = count
    lock = threading.Lock()

    def spawn(*args, **kwargs):
        nonlocal remaining
        with lock:
            remaining -= 1
            if remaining == 0:
                ctx.start_time = 0
        return original_spawn(*args, **kwargs)

    with patch.object(func, "spawn", spawn):
        yield


@workflow(app, self_managed_timeout=300, max_containers=2, scaledown_window=5)
def restart_while_spawning_workflow(ctx: WorkflowContext, counter_name: str) -> dict[str, object]:
    counter = modal.Dict.from_name(counter_name)
    attempt = counter.get("attempts", 0) + 1
    counter["attempts"] = attempt
    if attempt == 1:
        with restart_after_spawns(spawn_item, ctx, SPAWN_RESTART_AFTER_SPAWNS):
            results = ctx.map(spawn_item, list(range(SPAWN_ITEM_COUNT)), "slow_map", counter_name)
    else:
        counter[f"inherited_ids_{attempt}"] = len(ctx.state.cached_function_call_ids())
        results = ctx.map(spawn_item, list(range(SPAWN_ITEM_COUNT)), "slow_map", counter_name)
    ctx.finish()
    return {"results": results}


def check_restart_while_spawning() -> str:
    with counter_dict("spawning") as counter_name:
        call = restart_while_spawning_workflow.spawn(counter_name)
        result, retries = resolve_workflow_restarts(
            call,
            max_retries=3,
            retry_message="Workflow restarted while spawning, waiting for resumed run...",
        )
        snapshot = dict(modal.Dict.from_name(counter_name).items())

    assert retries == 1, f"Expected exactly one restart, got {retries}."
    inherited = snapshot["inherited_ids_2"]
    assert SPAWN_RESTART_AFTER_SPAWNS <= inherited < SPAWN_ITEM_COUNT, (
        f"The restart should land mid-spawn, but the continuation inherited {inherited} "
        f"of {SPAWN_ITEM_COUNT} call ids."
    )
    expected = [f"item-{item_id}-done" for item_id in range(SPAWN_ITEM_COUNT)]
    assert result["results"] == expected, f"Results incomplete or out of order: {result['results']}"
    assert_each_item_executed_once(snapshot, list(range(SPAWN_ITEM_COUNT)))
    return f"{inherited} call ids inherited"


@app.function(max_containers=3, scaledown_window=5)
@modal.concurrent(max_inputs=AFTER_MAP_ITEM_COUNT)
def record_item(item_id: str, counter_name: str) -> str:
    record_execution(counter_name, item_id)
    return item_id


@workflow(app, max_containers=2, scaledown_window=5)
def maps_with_restarts_workflow(ctx: WorkflowContext, counter_name: str) -> list[list[str]]:
    counter = modal.Dict.from_name(counter_name)
    all_results = []
    for map_index in range(AFTER_MAP_COUNT):
        item_ids = [f"{map_index}-{item}" for item in range(AFTER_MAP_ITEM_COUNT)]
        all_results.append(ctx.map(record_item, item_ids, f"map_{map_index}", counter_name))
        restart_key = f"restarted-{map_index}"
        if restart_key not in counter:
            counter.put(restart_key, True)
            ctx.restart()
    ctx.finish()
    return all_results


def check_restart_after_finished_maps() -> str:
    with counter_dict("after-map") as counter_name:
        call = maps_with_restarts_workflow.spawn(counter_name)
        result, retries = resolve_workflow_restarts(
            call,
            max_retries=AFTER_MAP_COUNT + 2,
            retry_message=(
                "Workflow restarted after a finished map, waiting for the continuation..."
            ),
        )
        snapshot = dict(modal.Dict.from_name(counter_name).items())

    assert retries == AFTER_MAP_COUNT, f"Expected {AFTER_MAP_COUNT} restarts, got {retries}."
    expected = [
        [f"{map_index}-{item}" for item in range(AFTER_MAP_ITEM_COUNT)]
        for map_index in range(AFTER_MAP_COUNT)
    ]
    assert result == expected, "Results incomplete or out of order."
    assert_each_item_executed_once(
        snapshot,
        [item for map_results in expected for item in map_results],
    )
    return f"{retries} restarts"


SCENARIOS: dict[str, Callable[[], str]] = {
    "restart while waiting": check_restart_while_waiting,
    "restart while spawning": check_restart_while_spawning,
    "restart after finished maps": check_restart_after_finished_maps,
}


@app.local_entrypoint()
def main() -> None:
    failures: dict[str, Exception] = {}
    for name, scenario in SCENARIOS.items():
        started = time.monotonic()
        try:
            detail = scenario()
        except Exception as error:
            failures[name] = error
            print(f"FAIL: {name}: {error!r}")
        else:
            print(f"PASS: {name} ({detail}, {time.monotonic() - started:.0f}s)")

    if failures:
        summary = "; ".join(f"{name}: {error!r}" for name, error in failures.items())
        raise AssertionError(f"{len(failures)} scenario(s) failed: {summary}")
    print("PASS: e2e map restart")

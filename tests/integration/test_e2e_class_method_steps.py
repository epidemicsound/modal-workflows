# mypy: ignore-errors
r"""
Integration test: Modal class methods used as workflow steps, against real Modal.

Scenarios, each with its own class and workflow so they stay independent:

- basic and restart: ``ctx.map`` (with a failing item), ``ctx.parallel`` with several
  positional arguments, ``ctx.step``, and a workflow restart while a ``ctx.map`` over the
  class method is in flight - every item must run exactly once.
- enter and concurrent: a class with ``@modal.enter`` and ``@modal.concurrent`` whose image
  differs from the workflow's; enter runs once per container and calls overlap.
- startup timeout: an enter longer than the per-call ``timeout`` works with an explicit
  ``startup_timeout``, and ``timeout`` still applies per call.

Every scenario runs; failures are collected and raised together at the end.

Run:  make test-class-methods
  or: MODAL_ENVIRONMENT=$MODAL_TESTING_ENVIRONMENT \
        uv run modal run tests/integration/test_e2e_class_method_steps.py
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

from __future__ import annotations

import collections
import time
import uuid

import modal

from modal_workflows.workflows import WorkflowContext, is_workflow_item_failure, workflow

from tests.integration.utilities import (  # isort: skip
    build_workflows_image,
    resolve_workflow_restarts,
)

image = build_workflows_image()
numpy_image = image.pip_install("numpy")
app = modal.App("test-class-method-steps", image=image)

with numpy_image.imports():
    import numpy as np

EXECUTIONS_DICT_PREFIX = "test-class-method-steps-executions"
RESTART_ITEM_COUNT = 16
RESTART_ITEM_SECONDS = 10


def _executions_dict_name(run_id: str) -> str:
    return f"{EXECUTIONS_DICT_PREFIX}-{run_id}"


def _record_execution(run_id: str, phase: str, item: int) -> None:
    executions = modal.Dict.from_name(_executions_dict_name(run_id), create_if_missing=True)
    executions[f"{phase}|{item}|{uuid.uuid4().hex}"] = True


@app.cls()
@modal.concurrent(max_inputs=16)
class BasicWorker:
    @modal.enter()
    def load(self) -> None:
        self.loaded = True

    @modal.method()
    def work(self, item: int, run_id: str, phase: str, seconds: float) -> str:
        import time

        assert self.loaded
        _record_execution(run_id, phase, item)
        time.sleep(seconds)
        if phase == "map" and item == 2:
            raise ValueError(f"deliberate failure for item {item}")
        return f"item-{item}-ok"

    @modal.method()
    def combine(self, number: int, text: str, scale: float) -> str:
        return f"{text}-{number * scale}"


@workflow(app)
def class_method_workflow(ctx: WorkflowContext, run_id: str) -> dict:
    mapped = ctx.map(
        BasicWorker().work, [0, 1, 2, 3], "map", run_id, "map", 1, return_exceptions=True
    )
    combine = BasicWorker().combine
    parallel = ctx.parallel(
        [(combine, 1, "a", 2.0), (combine, 2, "b", 2.5), (combine, 3, "c", 3.0)],
        "parallel",
    )
    single = ctx.step(combine, "single", 4, "d", 0.5)
    ctx.finish()
    return {"mapped": mapped, "parallel": parallel, "single": single}


@workflow(app, self_managed_timeout=12)
def class_method_restart_workflow(ctx: WorkflowContext, run_id: str) -> list:
    items = list(range(RESTART_ITEM_COUNT))
    results = ctx.map(
        BasicWorker().work, items, "restart_map", run_id, "restart", RESTART_ITEM_SECONDS
    )
    ctx.finish()
    return results


def _execution_counts(run_id: str, phase: str) -> collections.Counter:
    executions = modal.Dict.from_name(_executions_dict_name(run_id), create_if_missing=True)
    counts: collections.Counter = collections.Counter()
    for key in executions.keys():
        key_phase, item, _ = key.split("|")
        if key_phase == phase:
            counts[int(item)] += 1
    return counts


def _check_basic_steps(result: dict, run_id: str) -> None:
    mapped = result["mapped"]
    assert len(mapped) == 4, f"Expected 4 mapped results, got {mapped!r}"
    for index in (0, 1, 3):
        assert mapped[index] == f"item-{index}-ok", f"Unexpected result {mapped[index]!r}"
    assert is_workflow_item_failure(mapped[2]), f"Expected failure snapshot, got {mapped[2]!r}"
    assert "deliberate failure" in mapped[2]["message"]
    assert mapped[2]["exc_type"].endswith("ValueError")

    assert result["parallel"] == ["a-2.0", "b-5.0", "c-9.0"], result["parallel"]
    assert result["single"] == "d-2.0", result["single"]

    map_counts = _execution_counts(run_id, "map")
    assert dict(map_counts) == {0: 1, 1: 1, 2: 1, 3: 1}, f"map executions: {dict(map_counts)}"


def run_basic_and_restart() -> None:
    run_id = uuid.uuid4().hex[:12]
    try:
        basic_call = class_method_workflow.spawn(run_id)
        restart_call = class_method_restart_workflow.spawn(run_id)

        basic_result, _ = resolve_workflow_restarts(basic_call, max_retries=10)
        _check_basic_steps(basic_result, run_id)
        print("PASS: class method as map/parallel/step workflow steps")

        restart_result, retries = resolve_workflow_restarts(restart_call, max_retries=20)
        assert retries > 0, "Expected at least one workflow restart."
        expected = [f"item-{i}-ok" for i in range(RESTART_ITEM_COUNT)]
        assert restart_result == expected, f"Incomplete restart results: {restart_result!r}"

        restart_counts = _execution_counts(run_id, "restart")
        duplicates = {i: n for i, n in restart_counts.items() if n != 1}
        assert set(restart_counts) == set(range(RESTART_ITEM_COUNT)), restart_counts
        assert not duplicates, f"Items not executed exactly once: {duplicates}"
        print(
            f"PASS: class method map survived {retries} restart(s): "
            f"{RESTART_ITEM_COUNT} items, 0 duplicate executions"
        )
    finally:
        modal.Dict.objects.delete(_executions_dict_name(run_id), allow_missing=True)


CONCURRENT_ITEM_COUNT = 24
CONCURRENT_ITEM_SECONDS = 6
CONCURRENT_ENTER_SECONDS = 5
MAX_CONTAINERS = 4

enter_count = 0


@app.cls(
    image=numpy_image,
    max_containers=MAX_CONTAINERS,
    scaledown_window=5,
    timeout=300,
)
@modal.concurrent(max_inputs=16)
class ConcurrentWorker:
    @modal.enter()
    def setup(self) -> None:
        import os

        global enter_count
        enter_count += 1
        time.sleep(CONCURRENT_ENTER_SECONDS)
        self.container_id = os.environ["MODAL_TASK_ID"]
        self.enter_count = enter_count

    @modal.method()
    def work(self, item: int) -> dict:
        started = time.time()
        time.sleep(CONCURRENT_ITEM_SECONDS)
        return {
            "item": item,
            "container_id": self.container_id,
            "enter_count": self.enter_count,
            "started": started,
            "ended": time.time(),
            "triangular": int(np.arange(item + 1).sum()),
        }


@workflow(app, max_containers=1)
def enter_concurrent_workflow(ctx: WorkflowContext) -> dict:
    import importlib.util

    numpy_missing_in_workflow = importlib.util.find_spec("numpy") is None
    results = ctx.map(
        ConcurrentWorker().work,
        list(range(CONCURRENT_ITEM_COUNT)),
        "work",
        return_exceptions=True,
    )
    ctx.finish()
    return {"results": results, "numpy_missing_in_workflow": numpy_missing_in_workflow}


def max_overlap(intervals: list[tuple[float, float]]) -> int:
    events = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


def _check_enter_concurrent(output: dict) -> tuple[int, int]:
    assert output["numpy_missing_in_workflow"], "Workflow image unexpectedly has numpy."

    results = output["results"]
    assert (
        len(results) == CONCURRENT_ITEM_COUNT
    ), f"Expected {CONCURRENT_ITEM_COUNT} results, got {len(results)}"
    failures = [r for r in results if is_workflow_item_failure(r)]
    assert not failures, f"Unexpected item failures: {failures!r}"
    for index, result in enumerate(results):
        assert result["item"] == index, f"Result order broken at {index}: {result!r}"
        expected = index * (index + 1) // 2
        assert result["triangular"] == expected, f"Wrong numpy value for item {index}: {result!r}"

    by_container = collections.defaultdict(list)
    for result in results:
        by_container[result["container_id"]].append(result)
    assert len(by_container) <= MAX_CONTAINERS, f"Too many containers: {len(by_container)}"
    for container_id, calls in by_container.items():
        enters = {c["enter_count"] for c in calls}
        assert enters == {1}, f"Enter ran {enters} times in {container_id}"
    assert sum(len(c) for c in by_container.values()) == CONCURRENT_ITEM_COUNT

    overlap = max(
        max_overlap([(c["started"], c["ended"]) for c in calls]) for calls in by_container.values()
    )
    assert overlap >= 2, f"Calls never overlapped within a container (max overlap {overlap})"
    return len(by_container), overlap


def run_enter_concurrent() -> None:
    started = time.time()
    output, _ = resolve_workflow_restarts(enter_concurrent_workflow.spawn(), max_retries=10)
    containers, overlap = _check_enter_concurrent(output)
    print(
        f"PASS: class with enter + concurrent in a separate image: {CONCURRENT_ITEM_COUNT} items, "
        f"{containers} container(s), one enter each, max overlap {overlap}, "
        f"{time.time() - started:.0f}s"
    )


CALL_TIMEOUT = 20
STARTUP_TIMEOUT = 60
ENTER_SECONDS = 30
SHORT_CALL_SECONDS = 5
LONG_CALL_SECONDS = 25


@app.cls(
    max_containers=1,
    scaledown_window=5,
    timeout=CALL_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
)
class SlowStartWorker:
    @modal.enter()
    def setup(self) -> None:
        started = time.time()
        time.sleep(ENTER_SECONDS)
        self.enter_seconds = time.time() - started

    @modal.method()
    def work(self, seconds: float) -> dict:
        time.sleep(seconds)
        return {"seconds": seconds, "enter_seconds": self.enter_seconds}


@workflow(app, max_containers=1)
def startup_timeout_workflow(ctx: WorkflowContext) -> dict:
    first = ctx.step(SlowStartWorker().work, "short_call", SHORT_CALL_SECONDS)

    error_type = None
    error_message = None
    try:
        ctx.step(SlowStartWorker().work, "long_call", LONG_CALL_SECONDS)
    except Exception as error:  # noqa: BLE001
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        error_message = str(error)
    ctx.finish()
    return {"first": first, "error_type": error_type, "error_message": error_message}


def _check_startup_timeout(output: dict) -> None:
    first = output["first"]
    assert first["seconds"] == SHORT_CALL_SECONDS, first
    assert first["enter_seconds"] >= ENTER_SECONDS, first
    assert first["enter_seconds"] > CALL_TIMEOUT, "Enter must outlast the per-call timeout."

    assert output["error_type"] == "modal.exception.FunctionTimeoutError", output
    assert f"timeout of {CALL_TIMEOUT}s" in output["error_message"], output["error_message"]


def run_startup_timeout() -> None:
    started = time.time()
    output, _ = resolve_workflow_restarts(startup_timeout_workflow.spawn(), max_retries=10)
    _check_startup_timeout(output)
    print(
        f"PASS: {ENTER_SECONDS}s enter with timeout={CALL_TIMEOUT}s needs startup_timeout; "
        f"long call raised {output['error_type']} ({time.time() - started:.0f}s)"
    )


SCENARIOS = {
    "basic and restart": run_basic_and_restart,
    "enter and concurrent": run_enter_concurrent,
    "startup timeout": run_startup_timeout,
}


@app.local_entrypoint()
def main() -> None:
    failures = {}
    for name, scenario in SCENARIOS.items():
        try:
            scenario()
        except Exception as error:  # noqa: BLE001
            failures[name] = error
            print(f"FAIL: {name}: {type(error).__name__}: {error}")
    if failures:
        raise AssertionError(f"Failed scenarios: {', '.join(failures)}")

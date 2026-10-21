# mypy: ignore-errors
import logging
import pickle
import threading
import time
from unittest.mock import MagicMock, PropertyMock, call, patch

import modal
import pytest

import modal_workflows.workflows.context as context_module
from modal_workflows.workflows.context import WorkflowContext, WorkflowRestarted
from modal_workflows.workflows.failures import (
    WORKFLOW_ITEM_FAILURE_MARKER,
    is_workflow_item_failure,
)
from modal_workflows.workflows.state import WorkflowState


class _LockHoldingError(Exception):
    """Test double holding a genuinely unpicklable attribute."""

    def __init__(self, message):
        super().__init__(message)
        self.lock = threading.Lock()


def _spawn_by_item(items, calls):
    """Spawn double that hands each item its own call, whatever order spawns arrive in."""
    by_item = dict(zip(items, calls))
    return lambda item, *args, **kwargs: by_item[item]


def _wire_real_cache(mock_state):
    """Replaces a mocked state's cache methods with a real in-memory pickle round trip."""
    store: dict[str, bytes] = {}

    def _cache(step_id, result, commit=True):
        store[step_id] = WorkflowState._serialize_step_result(step_id, result)

    def _get(step_id):
        if step_id in store:
            return True, pickle.loads(store[step_id])
        return False, None

    mock_state.cache_step_result.side_effect = _cache
    mock_state.get_cached_step_result.side_effect = _get
    return store


@pytest.fixture
def mock_state():
    state = MagicMock()
    state.get_cached_step_result.return_value = (False, None)
    state.get_cached_function_call_id.return_value = (False, None)
    state.cached_function_call_ids.return_value = {}
    state.completed_steps = set()
    return state


@pytest.fixture
def workflow_context(mock_state):
    mock_app = MagicMock()
    mock_app.name = "test-app"
    mock_func = MagicMock(spec=modal.Function)
    with patch.object(context_module, "WorkflowState", return_value=mock_state):
        return WorkflowContext(
            app=mock_app,
            func=mock_func,
            func_name="test_workflow",
            state_name="test-run",
            args=(),
            kwargs={},
            timeout=3600,
        )


class TestStep:
    def test_calls_remote_and_returns_result(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = {"loss": 0.01}
        mock_func.spawn.return_value = mock_call

        result = workflow_context.step(mock_func, "train", "arg1")

        mock_func.spawn.assert_called_once_with("arg1")
        mock_call.get.assert_called_once_with(timeout=0)
        assert result == {"loss": 0.01}

    def test_returns_cached_result_without_calling_remote(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")
        mock_func = MagicMock(spec=modal.Function)

        result = workflow_context.step(mock_func, step_id="train")

        mock_func.spawn.assert_not_called()
        assert result == "cached"

    def test_forwards_kwargs(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = "done"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "train", "data", lr=0.001, epochs=10)

        mock_func.spawn.assert_called_once_with("data", lr=0.001, epochs=10)

    def test_different_step_ids_run_independently(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call_a = MagicMock()
        mock_call_b = MagicMock()
        mock_call_a.get.return_value = "result_a"
        mock_call_b.get.return_value = "result_b"
        mock_func.spawn.side_effect = [mock_call_a, mock_call_b]

        result_a = workflow_context.step(mock_func, step_id="step_a")
        result_b = workflow_context.step(mock_func, step_id="step_b")

        assert mock_func.spawn.call_count == 2
        assert result_a == "result_a"
        assert result_b == "result_b"


class TestSpawn:
    def test_calls_spawn_and_returns_handle(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.object_id = "fc-1"
        mock_func.spawn.return_value = mock_call

        result = workflow_context.spawn(mock_func, "notify", "message")

        mock_func.spawn.assert_called_once_with("message")
        assert result.object_id == "fc-1"

    def test_returns_cached_handle_without_respawning(self, workflow_context):
        workflow_context.state.get_cached_function_call_id.return_value = (True, "fc-123")
        inherited_call = MagicMock()
        inherited_call.object_id = "fc-123"
        workflow_context._restore_function_call = MagicMock(return_value=inherited_call)
        mock_func = MagicMock(spec=modal.Function)

        result = workflow_context.spawn(mock_func, step_id="notify")

        mock_func.spawn.assert_not_called()
        assert result.object_id == "fc-123"

    def test_spawned_handle_get_uses_workflow_waiter(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call
        workflow_context.wait_for_function_call = MagicMock(return_value="done")

        handle = workflow_context.spawn(mock_func, "notify", "message")
        result = handle.get()

        mock_call.get.assert_not_called()
        assert result == "done"

    def test_spawned_handle_get_forwards_arguments(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call
        workflow_context.wait_for_function_call = MagicMock(
            side_effect=lambda call, timeout=None: f"result-timeout-{timeout}"
        )

        handle = workflow_context.spawn(mock_func, step_id="notify")
        result = handle.get(wait_timeout=5)

        assert result == "result-timeout-5"

    def test_spawned_handle_get_failure_cancels_the_call_and_clears_its_id(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call
        workflow_context.wait_for_function_call = MagicMock(side_effect=RuntimeError("boom"))

        handle = workflow_context.spawn(mock_func, "notify")
        with pytest.raises(RuntimeError, match="boom"):
            handle.get()

        mock_call.cancel.assert_called_once()
        workflow_context.state.clear_cached_function_call_id.assert_called_with("notify")
        workflow_context.state.cache_step_result.assert_not_called()

    def test_spawned_handle_get_timeout_leaves_the_call_running(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call
        workflow_context.wait_for_function_call = MagicMock(side_effect=TimeoutError("slow"))

        handle = workflow_context.spawn(mock_func, "notify")
        with pytest.raises(TimeoutError):
            handle.get(wait_timeout=1)

        mock_call.cancel.assert_not_called()
        workflow_context.state.cache_step_result.assert_not_called()

    def test_spawned_handle_delegates_attributes(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.object_id = "fc-123"
        mock_func.spawn.return_value = mock_call

        handle = workflow_context.spawn(mock_func, step_id="notify")

        assert handle.object_id == "fc-123"


class TestMap:
    def test_distributes_items_and_returns_results(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        calls = [MagicMock(), MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.return_value = "r1"
        calls[2].get.return_value = "r2"
        mock_func.spawn.side_effect = _spawn_by_item([10, 20, 30], calls)

        results = workflow_context.map(mock_func, [10, 20, 30], "preprocess")

        mock_func.spawn.assert_has_calls([call(10), call(20), call(30)], any_order=True)
        assert results == ["r0", "r1", "r2"]

    def test_skips_cached_items(self, env):
        ctx = env.new_context()
        ctx.state.cache_step_result("preprocess_0", "cached_0")
        ctx.state.cache_step_result("preprocess_2", "cached_2")
        mock_func = MagicMock(spec=modal.Function)
        mock_func.spawn.return_value.get.return_value = "new_1"

        results = ctx.map(mock_func, ["a", "b", "c"], "preprocess")

        mock_func.spawn.assert_called_once_with("b")
        assert results == ["cached_0", "new_1", "cached_2"]

    def test_returns_all_cached_without_calling_map(self, env):
        ctx = env.new_context()
        for i in range(3):
            ctx.state.cache_step_result(f"batch_{i}", "cached")
        mock_func = MagicMock(spec=modal.Function)

        results = ctx.map(mock_func, [1, 2, 3], step_id="batch")

        mock_func.spawn.assert_not_called()
        assert results == ["cached", "cached", "cached"]

    def test_forwards_extra_args_and_kwargs(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        calls = [MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.return_value = "r1"
        mock_func.spawn.side_effect = _spawn_by_item([1, 2], calls)

        workflow_context.map(mock_func, [1, 2], "batch", "extra_arg", option=True)

        mock_func.spawn.assert_has_calls(
            [call(1, "extra_arg", option=True), call(2, "extra_arg", option=True)],
            any_order=True,
        )

    def test_default_raises_on_first_failure_and_aborts_remaining(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        calls = [MagicMock(), MagicMock()]
        calls[0].get.side_effect = RuntimeError("item 0 failed")
        calls[1].get.return_value = "r1"
        mock_func.spawn.side_effect = _spawn_by_item([10, 20], calls)

        with pytest.raises(RuntimeError, match="item 0 failed"):
            workflow_context.map(mock_func, [10, 20], "batch")

    def test_return_exceptions_true_returns_failure_snapshot_instead_of_raising(
        self, workflow_context
    ):
        mock_func = MagicMock(spec=modal.Function)
        calls = [MagicMock(), MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.side_effect = RuntimeError("item 1 failed")
        calls[2].get.return_value = "r2"
        mock_func.spawn.side_effect = _spawn_by_item([10, 20, 30], calls)

        results = workflow_context.map(mock_func, [10, 20, 30], "batch", return_exceptions=True)

        assert results[0] == "r0"
        assert is_workflow_item_failure(results[1])
        assert results[1]["exc_type"].endswith("RuntimeError")
        assert results[1]["message"] == "item 1 failed"
        assert results[2] == "r2"

    def test_return_exceptions_true_caches_the_failure_snapshot(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.side_effect = RuntimeError("boom")
        mock_func.spawn.return_value = mock_call

        workflow_context.map(mock_func, [10], "batch", return_exceptions=True)

        cached_args = workflow_context.state.cache_step_result.call_args
        assert cached_args.args[0] == "batch_0"
        assert is_workflow_item_failure(cached_args.args[1])
        assert cached_args.args[1]["message"] == "boom"

    def test_return_exceptions_true_still_propagates_workflow_restarted(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20
        workflow_context.func.spawn.return_value = MagicMock()

        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(WorkflowRestarted):
            workflow_context.map(mock_func, [10], "batch", return_exceptions=True)

        mock_func.spawn.assert_not_called()

    def test_return_exceptions_true_still_propagates_a_restart_mid_wait(self, workflow_context):
        # The restart comes from the wait loop's own check, after the item was spawned.
        workflow_context.func.spawn.return_value = MagicMock()
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call

        with patch.object(
            type(workflow_context), "should_restart", new_callable=PropertyMock
        ) as mock_should_restart:
            mock_should_restart.side_effect = [False, True]  # dispatch: no; first wait poll: yes

            with pytest.raises(WorkflowRestarted):
                workflow_context.map(mock_func, [10], "batch", return_exceptions=True)

        mock_func.spawn.assert_called_once()
        mock_call.get.assert_not_called()  # restart pre-empted the wait before polling for real

    def test_resumed_run_skips_a_previously_cached_failure(self, env):
        cached_failure = {
            WORKFLOW_ITEM_FAILURE_MARKER: True,
            "exc_type": "builtins.RuntimeError",
            "message": "cached failure",
            "traceback_str": (
                "Traceback (most recent call last):\n...\nRuntimeError: cached failure\n"
            ),
        }
        ctx = env.new_context()
        ctx.state.cache_step_result("batch_0", cached_failure)
        mock_func = MagicMock(spec=modal.Function)
        mock_func.spawn.return_value.get.return_value = "r1"

        results = ctx.map(mock_func, [10, 20], "batch", return_exceptions=True)

        mock_func.spawn.assert_called_once_with(20)
        assert results[0] == cached_failure
        assert results[1] == "r1"


class TestMapReturnExceptionsPortableFailure:
    def test_map_with_unpicklable_exception_attribute_does_not_abort(self, workflow_context):
        store = _wire_real_cache(workflow_context.state)

        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.side_effect = _LockHoldingError("lock stuck")
        mock_func.spawn.return_value = mock_call

        results = workflow_context.map(mock_func, [10], "batch", return_exceptions=True)

        assert is_workflow_item_failure(results[0])
        assert results[0]["message"] == "lock stuck"
        assert results[0]["exc_type"].endswith("_LockHoldingError")
        assert "batch_0" in store

        is_cached, reloaded = workflow_context.state.get_cached_step_result("batch_0")
        assert is_cached is True
        assert is_workflow_item_failure(reloaded)
        assert reloaded["message"] == "lock stuck"

    def test_map_degrades_gracefully_when_caching_the_failure_itself_fails(self, workflow_context):
        workflow_context.state.cache_step_result.side_effect = ValueError("volume unavailable")

        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.side_effect = RuntimeError("boom")
        mock_func.spawn.return_value = mock_call

        results = workflow_context.map(mock_func, [10], "batch", return_exceptions=True)

        assert is_workflow_item_failure(results[0])
        assert results[0]["message"] == "boom"


class TestParallel:
    def test_runs_different_functions_concurrently(self, workflow_context):
        mock_func_a = MagicMock(spec=modal.Function)
        mock_func_b = MagicMock(spec=modal.Function)
        mock_future_a = MagicMock()
        mock_future_b = MagicMock()
        mock_future_a.get.return_value = "metrics"
        mock_future_b.get.return_value = "registered"
        mock_func_a.spawn.return_value = mock_future_a
        mock_func_b.spawn.return_value = mock_future_b

        results = workflow_context.parallel(
            [(mock_func_a, "model.pt"), (mock_func_b, "model.pt")],
            step_id="eval",
        )

        mock_func_a.spawn.assert_called_once_with("model.pt")
        mock_func_b.spawn.assert_called_once_with("model.pt")
        assert results == ["metrics", "registered"]

    def test_skips_cached_steps(self, env):
        ctx = env.new_context()
        ctx.state.cache_step_result("eval_0", "cached_metrics")
        mock_func_a = MagicMock(spec=modal.Function)
        mock_func_b = MagicMock(spec=modal.Function)
        mock_func_b.spawn.return_value.get.return_value = "new_registration"

        results = ctx.parallel([(mock_func_a, "arg"), (mock_func_b, "arg")], step_id="eval")

        mock_func_a.spawn.assert_not_called()
        mock_func_b.spawn.assert_called_once_with("arg")
        assert results == ["cached_metrics", "new_registration"]

    def test_returns_all_cached_without_spawning(self, env):
        ctx = env.new_context()
        ctx.state.cache_step_result("eval_0", "cached")
        ctx.state.cache_step_result("eval_1", "cached")
        mock_func_a = MagicMock(spec=modal.Function)
        mock_func_b = MagicMock(spec=modal.Function)

        results = ctx.parallel([(mock_func_a, "arg"), (mock_func_b, "arg")], step_id="eval")

        mock_func_a.spawn.assert_not_called()
        mock_func_b.spawn.assert_not_called()
        assert results == ["cached", "cached"]

    def test_return_exceptions_true_returns_failure_snapshot_instead_of_raising(
        self, workflow_context
    ):
        mock_func_a = MagicMock(spec=modal.Function)
        mock_func_b = MagicMock(spec=modal.Function)
        mock_future_a = MagicMock()
        mock_future_a.get.return_value = "ok"
        mock_future_b = MagicMock()
        mock_future_b.get.side_effect = RuntimeError("step failed")
        mock_func_a.spawn.return_value = mock_future_a
        mock_func_b.spawn.return_value = mock_future_b

        results = workflow_context.parallel(
            [(mock_func_a, "arg"), (mock_func_b, "arg")],
            step_id="eval",
            return_exceptions=True,
        )

        assert results[0] == "ok"
        assert is_workflow_item_failure(results[1])
        assert results[1]["message"] == "step failed"


class TestRestart:
    def test_no_restart_before_timeout(self, workflow_context):
        assert workflow_context.should_restart is False

    def test_restart_detected_after_timeout(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20

        assert workflow_context.should_restart is True

    def test_restart_spawns_new_instance_and_raises(self, workflow_context):
        mock_new_call = MagicMock()
        workflow_context.func.spawn.return_value = mock_new_call

        with pytest.raises(WorkflowRestarted) as exc_info:
            workflow_context.restart()

        assert exc_info.value.function_call == mock_new_call

    def test_step_triggers_restart_when_timed_out(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20
        workflow_context.func.spawn.return_value = MagicMock()

        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(WorkflowRestarted):
            workflow_context.step(mock_func, step_id="late_step")

        mock_func.spawn.assert_not_called()

    def test_cached_step_bypasses_restart_even_when_timed_out(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (
            True,
            "early_result",
        )
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20

        mock_func = MagicMock(spec=modal.Function)
        result = workflow_context.step(mock_func, step_id="early_step")

        assert result == "early_result"
        mock_func.remote.assert_not_called()

    def test_map_triggers_restart_when_timed_out(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20
        workflow_context.func.spawn.return_value = MagicMock()

        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(WorkflowRestarted):
            workflow_context.map(mock_func, [1, 2], step_id="late_map")

        mock_func.spawn.assert_not_called()

    def test_parallel_triggers_restart_when_timed_out(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20
        workflow_context.func.spawn.return_value = MagicMock()

        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(WorkflowRestarted):
            workflow_context.parallel([(mock_func, "arg")], step_id="late_parallel")

        mock_func.spawn.assert_not_called()


class TestWaitForFunctionCall:
    def test_polls_with_short_timeout_until_success(self, workflow_context):
        mock_call = MagicMock()
        mock_call.get.side_effect = [TimeoutError("still running"), "done"]
        workflow_context.restart_if_needed = MagicMock()

        result = workflow_context.wait_for_function_call(mock_call)

        assert result == "done"
        mock_call.get.assert_any_call(timeout=0)

    def test_raises_non_timeout_errors(self, workflow_context):
        mock_call = MagicMock()
        mock_call.get.side_effect = RuntimeError("boom")
        workflow_context.restart_if_needed = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            workflow_context.wait_for_function_call(mock_call)

        mock_call.get.assert_called_once_with(timeout=0)

    def test_preserves_timeout_keyword_with_polling(self, workflow_context):
        mock_call = MagicMock()
        mock_call.get.return_value = "done"
        workflow_context.restart_if_needed = MagicMock()

        result = workflow_context.wait_for_function_call(mock_call, timeout=5)

        assert result == "done"
        mock_call.get.assert_called_once_with(timeout=0)


class TestDuplicateStepId:
    def test_step_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = "result"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "train")

        with pytest.raises(ValueError, match="train"):
            workflow_context.step(mock_func, "train")

    def test_spawn_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call

        workflow_context.spawn(mock_func, "notify")

        with pytest.raises(ValueError, match="notify"):
            workflow_context.spawn(mock_func, "notify")

    def test_map_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = "r"
        mock_func.spawn.return_value = mock_call

        workflow_context.map(mock_func, [1], "batch")

        with pytest.raises(ValueError, match="batch"):
            workflow_context.map(mock_func, [2], "batch")

    def test_parallel_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_future = MagicMock()
        mock_future.get.return_value = "r"
        mock_func.spawn.return_value = mock_future

        workflow_context.parallel([(mock_func, "arg")], step_id="eval")

        with pytest.raises(ValueError, match="eval"):
            workflow_context.parallel([(mock_func, "arg")], step_id="eval")

    def test_duplicate_across_methods_raises(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = "r"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "shared_id")

        with pytest.raises(ValueError, match="shared_id"):
            workflow_context.spawn(mock_func, "shared_id")

    def test_distinct_ids_do_not_raise(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = "r"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "step_a")
        workflow_context.step(mock_func, "step_b")
        workflow_context.step(mock_func, "step_c")

    def test_cached_step_still_registers_id(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")
        mock_func = MagicMock(spec=modal.Function)

        workflow_context.step(mock_func, "train")

        with pytest.raises(ValueError, match="train"):
            workflow_context.step(mock_func, "train")


_validation_app = modal.App("test-step-validation")


@_validation_app.function()
def _module_function(item):
    return item


@_validation_app.cls()
class _Worker:
    @modal.method()
    def run(self, item):
        return item


def _finished_call(result):
    call = MagicMock()
    call.object_id = "fc-validation"
    call.get.return_value = result
    return call


class TestWorkflowStepValidation:
    @pytest.mark.parametrize("target", [lambda item: item, _Worker, "not a function"])
    def test_all_entrypoints_reject_non_functions(self, workflow_context, target):
        calls = [
            lambda: workflow_context.step(target, "s"),
            lambda: workflow_context.spawn(target, "s"),
            lambda: workflow_context.map(target, [1], "s"),
            lambda: workflow_context.parallel([(target, 1)], "s"),
        ]
        for attempt in calls:
            with pytest.raises(TypeError, match="modal.Function"):
                attempt()

    @pytest.mark.parametrize("target", [_module_function, _Worker().run])
    def test_module_functions_and_bound_methods_are_accepted(self, workflow_context, target):
        with patch.object(target, "spawn", return_value=_finished_call("ok"), create=True):
            assert workflow_context.step(target, "s", 1) == "ok"
            assert workflow_context.spawn(target, "sp", 1).get() == "ok"
            assert workflow_context.map(target, [1], "m") == ["ok"]
            assert workflow_context.parallel([(target, 1)], "p") == ["ok"]


class TestLogEvent:
    @pytest.fixture(autouse=True)
    def capture_logs(self, caplog):
        caplog.set_level(logging.INFO, logger=context_module.logger.name)
        self.caplog = caplog

    def _events(self) -> list[str]:
        return [record.getMessage() for record in self.caplog.records]

    def test_log_event_emits_message_through_logger(self, workflow_context):
        workflow_context.log_event("test message")

        assert any("test message" in event for event in self._events())

    def test_log_event_includes_workflow_name(self, workflow_context):
        workflow_context.log_event("test message")

        assert any(workflow_context.func_name in event for event in self._events())

    def test_step_logs_event(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.get.return_value = "result"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "train", "arg")

        assert any("Running step train" in event for event in self._events())

    def test_cached_step_logs_nothing(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")

        workflow_context.step(MagicMock(spec=modal.Function), step_id="train")

        assert not any("Running step" in event for event in self._events())

    def test_spawn_logs_event(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_call = MagicMock()
        mock_call.object_id = "fc-1"
        mock_func.spawn.return_value = mock_call

        workflow_context.spawn(mock_func, "notify", "msg")

        assert any("Spawned notify" in event for event in self._events())

    def test_cached_spawn_logs_nothing(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")

        workflow_context.spawn(MagicMock(spec=modal.Function), step_id="notify")

        assert not any("Spawned" in event for event in self._events())

    def test_map_logs_event(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        calls = [MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.return_value = "r1"
        mock_func.spawn.side_effect = calls

        workflow_context.map(mock_func, [1, 2], "batch")

        assert any("Launched map batch (2 items)" in event for event in self._events())

    def test_parallel_logs_event(self, workflow_context):
        mock_func_a = MagicMock(spec=modal.Function)
        mock_func_b = MagicMock(spec=modal.Function)
        mock_future_a = MagicMock()
        mock_future_a.get.return_value = "a"
        mock_future_b = MagicMock()
        mock_future_b.get.return_value = "b"
        mock_func_a.spawn.return_value = mock_future_a
        mock_func_b.spawn.return_value = mock_future_b

        workflow_context.parallel(
            [(mock_func_a, "x"), (mock_func_b, "y")],
            step_id="eval",
        )

        assert any("Launched parallel eval (2 steps)" in event for event in self._events())

    def test_restart_logs_once(self, workflow_context):
        workflow_context.func.spawn.return_value = MagicMock()

        with pytest.raises(WorkflowRestarted):
            workflow_context.restart()

        assert len([e for e in self._events() if "Respawning" in e or "restart" in e.lower()]) == 1


class TestPathIdentifierValidation:
    def test_rejects_state_name_that_escapes_the_state_volume(self):
        with (
            patch.object(context_module, "WorkflowState", return_value=MagicMock()),
            pytest.raises(ValueError, match="Invalid state_name"),
        ):
            WorkflowContext(
                app=MagicMock(),
                func=MagicMock(spec=modal.Function),
                func_name="test_workflow",
                state_name="../../etc",
                args=(),
                kwargs={},
                timeout=3600,
            )

    def test_step_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.step(mock_func, "../../evil")

        mock_func.spawn.assert_not_called()

    def test_spawn_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.spawn(mock_func, "/absolute")

        mock_func.spawn.assert_not_called()

    def test_map_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.map(mock_func, [1, 2], "nested/step")

        mock_func.spawn.assert_not_called()

    def test_parallel_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.parallel([(mock_func,)], "..")

        mock_func.spawn.assert_not_called()

    def test_map_rejects_item_step_ids_over_the_length_limit_before_spawning(
        self, workflow_context
    ):
        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(ValueError, match="at most 128 characters"):
            workflow_context.map(mock_func, list(range(11)), "s" * 126)

        mock_func.spawn.assert_not_called()

    def test_parallel_rejects_step_ids_over_the_length_limit_before_spawning(
        self, workflow_context
    ):
        mock_func = MagicMock(spec=modal.Function)

        with pytest.raises(ValueError, match="at most 128 characters"):
            workflow_context.parallel([(mock_func,)] * 11, "s" * 126)

        mock_func.spawn.assert_not_called()

    def test_map_accepts_item_step_ids_at_the_length_limit(self, workflow_context):
        mock_func = MagicMock(spec=modal.Function)
        mock_func.spawn.side_effect = _finished_call

        assert workflow_context.map(mock_func, list(range(10)), "s" * 126) == list(range(10))

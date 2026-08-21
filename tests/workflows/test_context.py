# mypy: ignore-errors
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import modal_workflows.workflows.context as context_module
from modal_workflows.workflows.context import (
    WORKFLOW_STEP_MARKER,
    WorkflowContext,
    WorkflowRestarted,
)


@pytest.fixture
def mock_state():
    state = MagicMock()
    state.get_cached_step_result.return_value = (False, None)
    state.get_cached_function_call_id.return_value = (False, None)
    state.completed_steps = set()
    return state


@pytest.fixture
def workflow_context(mock_state):
    mock_app = MagicMock()
    mock_app.name = "test-app"
    mock_func = MagicMock()
    with (
        patch.object(
            context_module.workflow_volumes, "create_or_get_volume", return_value=MagicMock()
        ),
        patch.object(context_module, "WorkflowState", return_value=mock_state),
    ):
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
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.get.return_value = {"loss": 0.01}
        mock_func.spawn.return_value = mock_call

        result = workflow_context.step(mock_func, "train", "arg1")

        mock_func.spawn.assert_called_once_with("arg1")
        mock_call.get.assert_called_once_with(timeout=0)
        assert result == {"loss": 0.01}

    def test_returns_cached_result_without_calling_remote(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")
        mock_func = MagicMock()

        result = workflow_context.step(mock_func, step_id="train")

        mock_func.spawn.assert_not_called()
        assert result == "cached"

    def test_forwards_kwargs(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.get.return_value = "done"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "train", "data", lr=0.001, epochs=10)

        mock_func.spawn.assert_called_once_with("data", lr=0.001, epochs=10)

    def test_different_step_ids_run_independently(self, workflow_context):
        mock_func = MagicMock()
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
        mock_func = MagicMock()
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
        mock_func = MagicMock()

        result = workflow_context.spawn(mock_func, step_id="notify")

        mock_func.spawn.assert_not_called()
        assert result.object_id == "fc-123"

    def test_spawned_handle_get_uses_workflow_waiter(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call
        workflow_context.wait_for_function_call = MagicMock(return_value="done")

        handle = workflow_context.spawn(mock_func, "notify", "message")
        result = handle.get()

        mock_call.get.assert_not_called()
        assert result == "done"

    def test_spawned_handle_get_forwards_arguments(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call
        workflow_context.wait_for_function_call = MagicMock(
            side_effect=lambda call, timeout=None: f"result-timeout-{timeout}"
        )

        handle = workflow_context.spawn(mock_func, step_id="notify")
        result = handle.get(wait_timeout=5)

        assert result == "result-timeout-5"

    def test_spawned_handle_delegates_attributes(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.object_id = "fc-123"
        mock_func.spawn.return_value = mock_call

        handle = workflow_context.spawn(mock_func, step_id="notify")

        assert handle.object_id == "fc-123"


class TestMap:
    def test_distributes_items_and_returns_results(self, workflow_context):
        mock_func = MagicMock()
        calls = [MagicMock(), MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.return_value = "r1"
        calls[2].get.return_value = "r2"
        mock_func.spawn.side_effect = calls

        results = workflow_context.map(mock_func, [10, 20, 30], "preprocess")

        mock_func.spawn.assert_has_calls([call(10), call(20), call(30)])
        assert results == ["r0", "r1", "r2"]

    def test_skips_cached_items(self, workflow_context):
        def fake_cache_lookup(step_id):
            cached = {"preprocess_0": (True, "cached_0"), "preprocess_2": (True, "cached_2")}
            return cached.get(step_id, (False, None))

        workflow_context.state.get_cached_step_result.side_effect = fake_cache_lookup

        mock_func = MagicMock()
        pending_call = MagicMock()
        pending_call.get.return_value = "new_1"
        mock_func.spawn.return_value = pending_call

        results = workflow_context.map(mock_func, ["a", "b", "c"], "preprocess")

        mock_func.spawn.assert_called_once_with("b")
        assert results == ["cached_0", "new_1", "cached_2"]

    def test_returns_all_cached_without_calling_map(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")

        mock_func = MagicMock()
        results = workflow_context.map(mock_func, [1, 2, 3], step_id="batch")

        mock_func.spawn.assert_not_called()
        assert results == ["cached", "cached", "cached"]

    def test_forwards_extra_args_and_kwargs(self, workflow_context):
        mock_func = MagicMock()
        calls = [MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.return_value = "r1"
        mock_func.spawn.side_effect = calls

        workflow_context.map(mock_func, [1, 2], "batch", "extra_arg", option=True)

        mock_func.spawn.assert_has_calls(
            [call(1, "extra_arg", option=True), call(2, "extra_arg", option=True)],
        )


class TestParallel:
    def test_runs_different_functions_concurrently(self, workflow_context):
        mock_func_a = MagicMock()
        mock_func_b = MagicMock()
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

    def test_skips_cached_steps(self, workflow_context):
        def fake_cache_lookup(step_id):
            if step_id == "eval_0":
                return (True, "cached_metrics")
            return (False, None)

        workflow_context.state.get_cached_step_result.side_effect = fake_cache_lookup

        mock_func_a = MagicMock()
        mock_func_b = MagicMock()
        mock_future_b = MagicMock()
        mock_future_b.get.return_value = "new_registration"
        mock_func_b.spawn.return_value = mock_future_b

        results = workflow_context.parallel(
            [(mock_func_a, "arg"), (mock_func_b, "arg")],
            step_id="eval",
        )

        mock_func_a.spawn.assert_not_called()
        mock_func_b.spawn.assert_called_once_with("arg")
        assert results == ["cached_metrics", "new_registration"]

    def test_returns_all_cached_without_spawning(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")

        mock_func_a = MagicMock()
        mock_func_b = MagicMock()

        results = workflow_context.parallel(
            [(mock_func_a, "arg"), (mock_func_b, "arg")],
            step_id="eval",
        )

        mock_func_a.spawn.assert_not_called()
        mock_func_b.spawn.assert_not_called()
        assert results == ["cached", "cached"]


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

        mock_func = MagicMock()

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

        mock_func = MagicMock()
        result = workflow_context.step(mock_func, step_id="early_step")

        assert result == "early_result"
        mock_func.remote.assert_not_called()

    def test_map_triggers_restart_when_timed_out(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20
        workflow_context.func.spawn.return_value = MagicMock()

        mock_func = MagicMock()

        with pytest.raises(WorkflowRestarted):
            workflow_context.map(mock_func, [1, 2], step_id="late_map")

        mock_func.spawn.assert_not_called()

    def test_parallel_triggers_restart_when_timed_out(self, workflow_context):
        workflow_context.timeout = 10
        workflow_context.start_time = time.time() - 20
        workflow_context.func.spawn.return_value = MagicMock()

        mock_func = MagicMock()

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


def _unmarked_func(**kwargs):
    """Create a MagicMock that explicitly lacks the workflow-step marker."""
    mock = MagicMock(**kwargs)
    mock._workflow_step = False
    return mock


class TestDuplicateStepId:
    def test_step_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.get.return_value = "result"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "train")

        with pytest.raises(ValueError, match="train"):
            workflow_context.step(mock_func, "train")

    def test_spawn_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_func.spawn.return_value = mock_call

        workflow_context.spawn(mock_func, "notify")

        with pytest.raises(ValueError, match="notify"):
            workflow_context.spawn(mock_func, "notify")

    def test_map_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.get.return_value = "r"
        mock_func.spawn.return_value = mock_call

        workflow_context.map(mock_func, [1], "batch")

        with pytest.raises(ValueError, match="batch"):
            workflow_context.map(mock_func, [2], "batch")

    def test_parallel_raises_on_duplicate_id(self, workflow_context):
        mock_func = MagicMock()
        mock_future = MagicMock()
        mock_future.get.return_value = "r"
        mock_func.spawn.return_value = mock_future

        workflow_context.parallel([(mock_func, "arg")], step_id="eval")

        with pytest.raises(ValueError, match="eval"):
            workflow_context.parallel([(mock_func, "arg")], step_id="eval")

    def test_duplicate_across_methods_raises(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.get.return_value = "r"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "shared_id")

        with pytest.raises(ValueError, match="shared_id"):
            workflow_context.spawn(mock_func, "shared_id")

    def test_distinct_ids_do_not_raise(self, workflow_context):
        mock_func = MagicMock()
        mock_call = MagicMock()
        mock_call.get.return_value = "r"
        mock_func.spawn.return_value = mock_call

        workflow_context.step(mock_func, "step_a")
        workflow_context.step(mock_func, "step_b")
        workflow_context.step(mock_func, "step_c")

    def test_cached_step_still_registers_id(self, workflow_context):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")
        mock_func = MagicMock()

        workflow_context.step(mock_func, "train")

        with pytest.raises(ValueError, match="train"):
            workflow_context.step(mock_func, "train")


class TestWorkflowStepValidation:
    def test_step_rejects_unmarked_function(self, workflow_context):
        func = _unmarked_func()

        with pytest.raises(TypeError, match="@workflow_function"):
            workflow_context.step(func, "s")

    def test_spawn_rejects_unmarked_function(self, workflow_context):
        func = _unmarked_func()

        with pytest.raises(TypeError, match="@workflow_function"):
            workflow_context.spawn(func, "s")

    def test_map_rejects_unmarked_function(self, workflow_context):
        func = _unmarked_func()

        with pytest.raises(TypeError, match="@workflow_function"):
            workflow_context.map(func, [1], "s")

    def test_parallel_rejects_unmarked_function(self, workflow_context):
        func = _unmarked_func()

        with pytest.raises(TypeError, match="@workflow_function"):
            workflow_context.parallel([(func, "arg")], "s")

    def test_step_accepts_marked_function(self, workflow_context):
        func = MagicMock()
        setattr(func, WORKFLOW_STEP_MARKER, True)
        call = MagicMock()
        call.get.return_value = "ok"
        func.spawn.return_value = call

        result = workflow_context.step(func, "s")

        assert result == "ok"


class TestLogEvent:
    def test_log_event_appends_timestamp_and_message_to_file_and_commits(
        self, workflow_context, tmp_path: Path
    ):
        with patch.object(
            context_module.workflow_volumes,
            "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH",
            tmp_path,
        ):
            workflow_context.log_event("test message")

            run_path = workflow_context.artifacts_run_path
            log_files = list(run_path.glob("*.txt"))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "test message" in content
            assert " | " in content

    def test_log_event_appends_multiple_messages_each_on_new_line(
        self, workflow_context, tmp_path: Path
    ):
        with patch.object(
            context_module.workflow_volumes,
            "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH",
            tmp_path,
        ):
            workflow_context.log_event("first")
            workflow_context.log_event("second")
            workflow_context.log_event("third")

            run_path = workflow_context.artifacts_run_path
            log_files = list(run_path.glob("*.txt"))
            assert len(log_files) == 1
            lines = log_files[0].read_text().strip().split("\n")
            assert len(lines) == 3
            assert "first" in lines[0]
            assert "second" in lines[1]
            assert "third" in lines[2]

    def test_step_writes_event_to_log(self, workflow_context, tmp_path: Path):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)
        mock_call = MagicMock()
        mock_call.get.return_value = "result"
        mock_func.spawn.return_value = mock_call

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            workflow_context.step(mock_func, "train", "arg")

            log_files = list(workflow_context.artifacts_run_path.glob("*.txt"))
            assert len(log_files) == 1
            assert "Running step train" in log_files[0].read_text()

    def test_cached_step_writes_nothing_to_log(self, workflow_context, tmp_path: Path):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")
        mock_func = MagicMock()

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            workflow_context.step(mock_func, step_id="train")

        assert list(tmp_path.rglob("*.txt")) == []

    def test_spawn_writes_event_to_log(self, workflow_context, tmp_path: Path):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)
        mock_call = MagicMock()
        mock_call.object_id = "fc-1"
        mock_func.spawn.return_value = mock_call

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            workflow_context.spawn(mock_func, "notify", "msg")

            log_files = list(workflow_context.artifacts_run_path.glob("*.txt"))
            assert len(log_files) == 1
            assert "Spawned notify" in log_files[0].read_text()

    def test_cached_spawn_writes_nothing_to_log(self, workflow_context, tmp_path: Path):
        workflow_context.state.get_cached_step_result.return_value = (True, "cached")
        mock_func = MagicMock()

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            workflow_context.spawn(mock_func, step_id="notify")

        assert list(tmp_path.rglob("*.txt")) == []

    def test_map_writes_event_to_log(self, workflow_context, tmp_path: Path):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)
        calls = [MagicMock(), MagicMock()]
        calls[0].get.return_value = "r0"
        calls[1].get.return_value = "r1"
        mock_func.spawn.side_effect = calls

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            workflow_context.map(mock_func, [1, 2], "batch")

            log_files = list(workflow_context.artifacts_run_path.glob("*.txt"))
            assert len(log_files) == 1
            assert "Launched map batch (2 items)" in log_files[0].read_text()

    def test_parallel_writes_event_to_log(self, workflow_context, tmp_path: Path):
        mock_func_a = MagicMock()
        mock_func_b = MagicMock()
        setattr(mock_func_a, WORKFLOW_STEP_MARKER, True)
        setattr(mock_func_b, WORKFLOW_STEP_MARKER, True)
        mock_future_a = MagicMock()
        mock_future_a.get.return_value = "a"
        mock_future_b = MagicMock()
        mock_future_b.get.return_value = "b"
        mock_func_a.spawn.return_value = mock_future_a
        mock_func_b.spawn.return_value = mock_future_b

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            workflow_context.parallel(
                [(mock_func_a, "x"), (mock_func_b, "y")],
                step_id="eval",
            )

            log_files = list(workflow_context.artifacts_run_path.glob("*.txt"))
            assert len(log_files) == 1
            assert "Launched parallel eval (2 steps)" in log_files[0].read_text()

    def test_restart_writes_event_to_log(self, workflow_context, tmp_path: Path):
        workflow_context.func.spawn.return_value = MagicMock()

        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            with pytest.raises(WorkflowRestarted):
                workflow_context.restart()

            log_files = list(workflow_context.artifacts_run_path.glob("*.txt"))
            assert len(log_files) == 1
            assert "Workflow restart (timeout)" in log_files[0].read_text()


class TestWorkflowArtifactsPaths:
    def test_raises_when_artifacts_volume_is_not_mounted(self, workflow_context, tmp_path: Path):
        missing_mount_path = tmp_path / "missing_mount"
        with patch.object(
            context_module.workflow_volumes,
            "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH",
            missing_mount_path,
        ):
            with pytest.raises(RuntimeError, match="is not mounted"):
                workflow_context._artifacts_root_path

    def test_returns_run_artifacts_folder_by_default(self, workflow_context, tmp_path: Path):
        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            run_path = workflow_context.artifacts_run_path

        assert run_path == tmp_path / "test-run"
        assert run_path.exists() is True

    def test_returns_shared_artifacts_folder(self, workflow_context, tmp_path: Path):
        with patch.object(
            context_module.workflow_volumes, "WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH", tmp_path
        ):
            shared_path = workflow_context.artifacts_shared_path

        assert (
            shared_path
            == tmp_path / context_module.workflow_volumes.WORKFLOW_ARTIFACTS_SHARED_DIRECTORY
        )
        assert shared_path.exists() is True


class TestPathIdentifierValidation:
    def test_rejects_state_name_that_escapes_the_artifacts_volume(self):
        with (
            patch.object(
                context_module.workflow_volumes, "create_or_get_volume", return_value=MagicMock()
            ),
            patch.object(context_module, "WorkflowState", return_value=MagicMock()),
            pytest.raises(ValueError, match="Invalid state_name"),
        ):
            WorkflowContext(
                app=MagicMock(),
                func=MagicMock(),
                func_name="test_workflow",
                state_name="../../etc",
                args=(),
                kwargs={},
                timeout=3600,
            )

    def test_step_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.step(mock_func, "../../evil")

        mock_func.spawn.assert_not_called()

    def test_spawn_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.spawn(mock_func, "/absolute")

        mock_func.spawn.assert_not_called()

    def test_map_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.map(mock_func, [1, 2], "nested/step")

        mock_func.spawn.assert_not_called()

    def test_parallel_rejects_step_id_that_escapes_the_results_directory(self, workflow_context):
        mock_func = MagicMock()
        setattr(mock_func, WORKFLOW_STEP_MARKER, True)

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_context.parallel([(mock_func,)], "..")

        mock_func.spawn.assert_not_called()

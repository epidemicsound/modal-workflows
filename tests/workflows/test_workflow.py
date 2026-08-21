# mypy: ignore-errors
from unittest.mock import MagicMock, patch

import pytest

import modal_workflows.workflows.volumes as volumes_module
from modal_workflows.workflows.context import (
    WORKFLOW_STEP_MARKER,
    WorkflowContext,
    WorkflowRestarted,
)
from modal_workflows.workflows.decorator import (
    _STATE_NAME_KWARG,
    _resolve_state_name,
    workflow,
    workflow_function,
)


@pytest.fixture(autouse=True)
def mock_workflow_result_volume():
    with patch(
        "modal_workflows.workflows.volumes.create_or_get_volume",
    ) as mock_create_volume:
        mock_create_volume.return_value = MagicMock()
        yield


@pytest.fixture(autouse=True)
def mock_modal_function_call_id():
    # Workflows are only meant to run inside Modal containers, where
    # modal.current_function_call_id() returns a real ID. Tests run outside
    # Modal, so we stub it. Individual tests can override by patching again.
    with patch("modal.current_function_call_id", return_value="fc-test"):
        yield


def _make_step(return_value):
    """Return a mock Modal function that behaves like a @workflow_function step."""
    func = MagicMock()
    setattr(func, WORKFLOW_STEP_MARKER, True)
    call = MagicMock()
    call.get.return_value = return_value
    func.spawn.return_value = call
    return func


def _make_app():
    mock_app = MagicMock()
    mock_app.app_id = "ap-abc123"
    mock_app.function.return_value = lambda f: f
    return mock_app


def _uncached_state():
    """WorkflowState mock where nothing is cached yet."""
    state = MagicMock()
    state.get_cached_step_result.return_value = (False, None)
    state.get_cached_function_call_id.return_value = (False, None)
    state.completed_steps = set()
    return state


class TestWorkflowRunsBehavior:
    """End-to-end tests: workflow decorator + real WorkflowContext.
    These check what comes out of the workflow given what goes in,
    not which internal methods were called in what order.
    """

    def test_single_step_result_flows_through_to_caller(self):
        step = _make_step("trained_model.pt")

        with patch(
            "modal_workflows.workflows.context.WorkflowState", return_value=_uncached_state()
        ):

            @workflow(_make_app())
            def my_workflow(ctx, dataset):
                return ctx.step(step, "train", dataset)

            result = my_workflow("train_data")

        assert result == "trained_model.pt"

    def test_multiple_steps_results_are_combined_correctly(self):
        preprocess = _make_step("clean_data")
        train = _make_step("model.pt")
        evaluate = _make_step({"accuracy": 0.95})

        with patch(
            "modal_workflows.workflows.context.WorkflowState", return_value=_uncached_state()
        ):

            @workflow(_make_app())
            def my_workflow(ctx):
                data = ctx.step(preprocess, "preprocess")
                model = ctx.step(train, "train", data)
                metrics = ctx.step(evaluate, "evaluate", model)
                return {"model": model, "metrics": metrics}

            result = my_workflow()

        assert result == {"model": "model.pt", "metrics": {"accuracy": 0.95}}

    def test_cached_steps_are_skipped_and_cached_value_is_used(self):
        step = _make_step("new_result")
        state = _uncached_state()
        state.get_cached_step_result.return_value = (True, "cached_model.pt")

        with patch("modal_workflows.workflows.context.WorkflowState", return_value=state):

            @workflow(_make_app())
            def my_workflow(ctx):
                return ctx.step(step, "train")

            result = my_workflow()

        assert result == "cached_model.pt"
        step.spawn.assert_not_called()

    def test_partial_cache_first_step_cached_second_runs(self):
        step_a = _make_step("new_b")
        step_b = _make_step("new_b")
        state = _uncached_state()

        def cache_lookup(step_id):
            if step_id == "step_a":
                return (True, "cached_a")
            return (False, None)

        state.get_cached_step_result.side_effect = cache_lookup

        with patch("modal_workflows.workflows.context.WorkflowState", return_value=state):

            @workflow(_make_app())
            def my_workflow(ctx):
                a = ctx.step(step_a, "step_a")
                b = ctx.step(step_b, "step_b", a)
                return [a, b]

            result = my_workflow()

        assert result == ["cached_a", "new_b"]
        step_a.spawn.assert_not_called()
        step_b.spawn.assert_called_once_with("cached_a")

    def test_kwargs_forwarded_to_step(self):
        step = _make_step("done")

        with patch(
            "modal_workflows.workflows.context.WorkflowState", return_value=_uncached_state()
        ):

            @workflow(_make_app())
            def my_workflow(ctx):
                return ctx.step(step, "train", "data", lr=1e-4, epochs=10)

            my_workflow()

        step.spawn.assert_called_once_with("data", lr=1e-4, epochs=10)

    def test_state_name_kwarg_is_never_visible_to_user_function(self):
        received = {}

        with patch(
            "modal_workflows.workflows.context.WorkflowState", return_value=_uncached_state()
        ):

            @workflow(_make_app())
            def my_workflow(ctx, **kwargs):
                received.update(kwargs)
                return "done"

            my_workflow(_workflow_state_name="propagated-state")

        assert _STATE_NAME_KWARG not in received

    def test_explicit_state_name_pins_to_exact_state(self):
        # Second invocation with the same explicit state_name should find the
        # step cached from the first invocation and skip re-running it.
        step = _make_step("result")
        state = _uncached_state()
        invocation_count = {"n": 0}

        def track_and_cache(step_id):
            invocation_count["n"] += 1
            if invocation_count["n"] > 1:
                return (True, "cached_result")
            return (False, None)

        state.get_cached_step_result.side_effect = track_and_cache

        with patch("modal_workflows.workflows.context.WorkflowState", return_value=state):

            @workflow(_make_app(), state_name="my-crashed-run")
            def my_workflow(ctx):
                return ctx.step(step, "train")

            my_workflow()
            result = my_workflow()

        assert result == "cached_result"
        step.spawn.assert_called_once()  # only ran on the first invocation

    def test_workflow_kwargs_forwarded_to_user_function(self):
        received = {}

        with patch(
            "modal_workflows.workflows.context.WorkflowState", return_value=_uncached_state()
        ):

            @workflow(_make_app())
            def my_workflow(ctx, model_size, lr=0.01):
                received["model_size"] = model_size
                received["lr"] = lr
                return "done"

            my_workflow("large", lr=1e-5)

        assert received == {"model_size": "large", "lr": 1e-5}


class TestResolveStateName:
    """State name priority: explicit state_name, propagated kwarg, then app_id + call id."""

    def test_explicit_state_name_takes_priority_over_propagated_kwarg(self):
        app = MagicMock(app_id="ap-abc123")
        kwargs = {_STATE_NAME_KWARG: "propagated-state", "user_arg": "keep"}
        state_name = _resolve_state_name(app, "explicit-state", kwargs)
        assert state_name == "explicit-state"

    def test_kwarg_is_always_stripped_even_when_explicit_state_name_wins(self):
        app = MagicMock(app_id="ap-abc123")
        kwargs = {_STATE_NAME_KWARG: "propagated-state", "user_arg": "keep"}
        _resolve_state_name(app, "explicit-state", kwargs)
        assert kwargs == {"user_arg": "keep"}

    def test_propagated_kwarg_used_when_no_explicit_state_name(self):
        app = MagicMock(app_id="ap-abc123")
        kwargs = {_STATE_NAME_KWARG: "propagated-state", "user_arg": "keep"}
        state_name = _resolve_state_name(app, None, kwargs)
        assert state_name == "propagated-state"
        assert kwargs == {"user_arg": "keep"}

    def test_derives_from_app_id_and_function_call_id_when_nothing_provided(self):
        app = MagicMock(app_id="ap-abc123")
        with patch("modal.current_function_call_id", return_value="fc-from-modal"):
            state_name = _resolve_state_name(app, None, {})
        assert state_name == "ap-abc123_fc-from-modal"

    def test_raises_when_no_kwarg_no_state_name_and_not_in_modal_container(self):
        app = MagicMock(app_id="ap-abc123")
        with patch("modal.current_function_call_id", return_value=None):
            with pytest.raises(RuntimeError, match="Cannot resolve a state name"):
                _resolve_state_name(app, None, {})


class TestWorkflowDecorator:
    def test_forwards_kwargs_to_app_function(self):
        mock_app = MagicMock()
        mock_app.app_id = "ap-abc123"
        mock_app.function.return_value = lambda f: f

        with patch("modal_workflows.workflows.context.WorkflowState"):

            @workflow(mock_app, gpu="T4", secrets=[])
            def my_workflow(ctx):
                return "done"

        mock_app.function.assert_called_once()
        function_kwargs = mock_app.function.call_args.kwargs
        assert function_kwargs["gpu"] == "T4"
        assert function_kwargs["secrets"] == []
        assert function_kwargs["single_use_containers"] is True
        assert "volumes" in function_kwargs
        assert str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH) in function_kwargs["volumes"]
        assert (
            str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH) in function_kwargs["volumes"]
        )

    def test_injects_workflow_context_as_first_argument(self):
        mock_app = MagicMock()
        mock_app.app_id = "ap-abc123"
        mock_app.function.return_value = lambda f: f
        captured = {}

        with patch("modal_workflows.workflows.context.WorkflowState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.completed_steps.return_value = []
            mock_state_cls.return_value = mock_state

            @workflow(mock_app)
            def my_workflow(ctx):
                captured["ctx"] = ctx
                return "done"

            result = my_workflow()

        assert isinstance(captured["ctx"], WorkflowContext)
        assert result == "done"

    def test_catches_workflow_restarted_and_returns_restart_metadata(self):
        mock_app = MagicMock()
        mock_app.app_id = "ap-abc123"
        mock_app.function.return_value = lambda f: f
        mock_new_call = MagicMock()
        mock_new_call.object_id = "fc-123"

        with patch("modal_workflows.workflows.context.WorkflowState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.completed_steps.return_value = []
            mock_state_cls.return_value = mock_state

            @workflow(mock_app)
            def my_workflow(ctx):
                raise WorkflowRestarted(mock_new_call)

            result = my_workflow()

        mock_new_call.get.assert_not_called()
        assert result.function_name == "my_workflow"
        assert result.call_id == mock_new_call.object_id

    def test_raises_when_self_managed_timeout_exceeds_max_23_hours(self):
        mock_app = MagicMock()
        mock_app.app_id = "ap-abc123"
        mock_app.function.return_value = lambda f: f

        with pytest.raises(
            ValueError,
            match="self_managed_timeout cannot exceed 23 hours",
        ):
            workflow(mock_app, self_managed_timeout=30 * 3600)


class TestWorkflowFunctionDecorator:
    def test_mounts_artifacts_volume(self):
        mock_app = MagicMock()
        mock_app.function.return_value = lambda f: f

        with patch(
            "modal_workflows.workflows.volumes.create_or_get_volume",
        ) as mock_create_volume:
            mock_create_volume.return_value = MagicMock()

            @workflow_function(mock_app)
            def my_step(data):
                return data

        function_kwargs = mock_app.function.call_args.kwargs
        assert "volumes" in function_kwargs
        assert (
            str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH) in function_kwargs["volumes"]
        )

    def test_does_not_mount_state_volume(self):
        mock_app = MagicMock()
        mock_app.function.return_value = lambda f: f

        with patch(
            "modal_workflows.workflows.volumes.create_or_get_volume",
        ) as mock_create_volume:
            mock_create_volume.return_value = MagicMock()

            @workflow_function(mock_app)
            def my_step(data):
                return data

        function_kwargs = mock_app.function.call_args.kwargs
        assert (
            str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH) not in function_kwargs["volumes"]
        )

    def test_forwards_kwargs_to_app_function(self):
        mock_app = MagicMock()
        mock_app.function.return_value = lambda f: f

        with patch(
            "modal_workflows.workflows.volumes.create_or_get_volume",
        ) as mock_create_volume:
            mock_create_volume.return_value = MagicMock()

            @workflow_function(mock_app, gpu="T4", secrets=[])
            def my_step(data):
                return data

        function_kwargs = mock_app.function.call_args.kwargs
        assert function_kwargs["gpu"] == "T4"
        assert function_kwargs["secrets"] == []

    def test_sets_workflow_step_marker(self):
        mock_app = MagicMock()
        mock_app.function.return_value = lambda f: f

        with patch(
            "modal_workflows.workflows.volumes.create_or_get_volume",
        ) as mock_create_volume:
            mock_create_volume.return_value = MagicMock()

            @workflow_function(mock_app)
            def my_step(data):
                return data

        assert getattr(my_step, WORKFLOW_STEP_MARKER, False) is True

    def test_preserves_user_volumes(self):
        mock_app = MagicMock()
        mock_app.function.return_value = lambda f: f
        user_volume = MagicMock()

        with patch(
            "modal_workflows.workflows.volumes.create_or_get_volume",
        ) as mock_create_volume:
            mock_create_volume.return_value = MagicMock()

            @workflow_function(mock_app, volumes={"/mnt/data": user_volume})
            def my_step(data):
                return data

        function_kwargs = mock_app.function.call_args.kwargs
        assert function_kwargs["volumes"]["/mnt/data"] is user_volume
        assert (
            str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH) in function_kwargs["volumes"]
        )

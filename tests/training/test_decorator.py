import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import modal_workflows.training.decorator as training_module
from modal_workflows.training.decorator import training
from modal_workflows.workflows.context import WorkflowRestarted


@pytest.fixture(autouse=True)
def mock_volume_creation_and_mount_paths(tmp_path: Path):
    with (
        patch("modal_workflows.workflows.volumes.create_or_get_volume", return_value=MagicMock()),
        patch("modal_workflows.workflows.volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH", tmp_path),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_modal_function_call_id():
    # Training is only meant to run inside Modal containers, where
    # modal.current_function_call_id() returns a real ID. Tests run outside
    # Modal, so we stub it. Individual tests can override by patching again.
    with patch("modal.current_function_call_id", return_value="fc-test"):
        yield


def _build_mock_app() -> MagicMock:
    mock_app = MagicMock()
    mock_app.app_id = "ap-abc123"
    mock_app.function.return_value = lambda fn: fn
    return mock_app


class TestTrainingDecoratorClusterOptions:
    def test_rejects_cluster_size_below_one(self):
        app = _build_mock_app()

        with pytest.raises(ValueError, match="cluster_size must be >= 1"):
            training(app, cluster_size=0)

    def test_requires_gpu_when_cluster_size_above_one(self):
        app = _build_mock_app()

        with pytest.raises(ValueError, match="cluster_size > 1 requires a `gpu` function argument"):
            training(app, cluster_size=2)

    def test_applies_clustered_decorator_when_requested(self):
        app = _build_mock_app()

        def clustered_decorator(fn):
            def wrapped() -> str:
                return f"clustered:{fn()}"

            return wrapped

        experimental_api = SimpleNamespace(
            clustered=lambda **_: clustered_decorator,
            get_cluster_info=MagicMock(return_value=SimpleNamespace(rank=0)),
        )
        with patch.object(training_module.modal, "experimental", experimental_api, create=True):

            @training(app, cluster_size=2, rdma=True, gpu="H100:8")
            def train_fn() -> str:
                return "ok"

            assert train_fn() == "clustered:ok"

    def test_does_not_apply_clustered_decorator_for_single_node(self):
        app = _build_mock_app()

        def clustered_decorator(fn):
            def wrapped() -> str:
                return f"clustered:{fn()}"

            return wrapped

        experimental_api = SimpleNamespace(clustered=lambda **_: clustered_decorator)
        with patch.object(training_module.modal, "experimental", experimental_api, create=True):

            @training(app, cluster_size=1, gpu="H100:8")
            def train_fn() -> str:
                return "ok"

        assert train_fn() == "ok"


class TestTrainingDecoratorClusterRankSemantics:
    def test_non_leader_skips_restart_handoff_on_success(self):
        app = _build_mock_app()

        with (
            patch("modal_workflows.training.decorator._is_cluster_leader", return_value=False),
            patch(
                "modal_workflows.training.decorator._restart_from_metadata_path",
                side_effect=AssertionError("non-leader should not attempt handoff"),
            ),
        ):

            @training(app)
            def train_fn() -> str:
                return "done"

            result = train_fn()

        assert result == "done"

    def test_non_leader_exception_does_not_send_alert(self):
        app = _build_mock_app()

        with (
            patch("modal_workflows.training.decorator._is_cluster_leader", return_value=False),
            patch("modal_workflows.training.decorator.send_exception_alert") as mock_alert,
        ):

            @training(app, alert_on_error=True)
            def train_fn() -> str:
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                train_fn()

        mock_alert.assert_not_called()

    def test_leader_exception_sends_alert_when_enabled(self):
        app = _build_mock_app()

        with (
            patch("modal_workflows.training.decorator._is_cluster_leader", return_value=True),
            patch("modal_workflows.training.decorator.send_exception_alert") as mock_alert,
            patch(
                "modal_workflows.training.decorator._restart_from_metadata_path",
                return_value=None,
            ),
        ):

            @training(app, alert_on_error=True)
            def train_fn() -> str:
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                train_fn()

        mock_alert.assert_called_once()
        _, exc = mock_alert.call_args[0]
        assert isinstance(exc, RuntimeError)
        assert "boom" in str(exc)

    def test_non_leader_workflow_restarted_is_not_converted_to_handoff_result(self):
        app = _build_mock_app()
        function_call = MagicMock()

        with patch("modal_workflows.training.decorator._is_cluster_leader", return_value=False):

            @training(app)
            def train_fn() -> str:
                raise WorkflowRestarted(function_call)

            with pytest.raises(WorkflowRestarted):
                train_fn()


class TestTrainingStateNameIsolation:
    def test_separate_invocations_use_different_state_names(self):
        app = _build_mock_app()
        captured_state_names = []

        with (
            patch("modal_workflows.training.decorator._is_cluster_leader", return_value=True),
            patch(
                "modal_workflows.training.decorator._restart_from_metadata_path",
                return_value=None,
            ),
        ):
            call_ids = iter(["fc-call-1", "fc-call-2"])
            with patch(
                "modal.current_function_call_id",
                side_effect=lambda: next(call_ids),
            ):

                @training(app)
                def train_fn() -> str:
                    captured_state_names.append(
                        os.environ.get("MODAL_WORKFLOWS_TRAINING_STATE_NAME")
                    )
                    return "done"

                train_fn()
                train_fn()

        assert captured_state_names == ["ap-abc123_fc-call-1", "ap-abc123_fc-call-2"]

    def test_propagated_state_name_kwarg_pins_state_across_invocations(self):
        # Covers both Modal retry / preemption (same function_call_id → same state)
        # and graceful self-managed restart (state_name propagated via kwarg).
        app = _build_mock_app()
        captured_state_names = []

        with (
            patch("modal_workflows.training.decorator._is_cluster_leader", return_value=True),
            patch(
                "modal_workflows.training.decorator._restart_from_metadata_path",
                return_value=None,
            ),
        ):

            @training(app)
            def train_fn() -> str:
                captured_state_names.append(os.environ.get("MODAL_WORKFLOWS_TRAINING_STATE_NAME"))
                return "done"

            train_fn(_workflow_state_name="pinned-state")
            train_fn(_workflow_state_name="pinned-state")

        assert captured_state_names == ["pinned-state", "pinned-state"]

    def test_graceful_restart_propagates_state_name_to_handoff(self):
        app = _build_mock_app()
        mock_function_call = MagicMock()
        mock_function_call.object_id = "fc-continuation"

        with (
            patch("modal_workflows.training.decorator._is_cluster_leader", return_value=True),
            patch("modal.current_function_call_id", return_value="fc-original"),
            patch(
                "modal_workflows.training.decorator._restart_from_metadata_path",
                return_value=mock_function_call,
            ) as mock_restart,
        ):

            @training(app)
            def train_fn() -> str:
                return "done"

            train_fn()

        call_kwargs = mock_restart.call_args.kwargs
        assert call_kwargs["_workflow_state_name"] == "ap-abc123_fc-original"


class TestRestartMetadata:
    def test_write_then_read_round_trips(self, tmp_path: Path):
        path = tmp_path / "restart.json"
        training_module._write_restart_metadata_file(
            path, reason="timeout", payload={"checkpoint": "latest.ckpt"}
        )
        metadata = training_module._read_restart_metadata_file(path)
        assert metadata is not None
        assert metadata.restart_requested is True
        assert metadata.reason == "timeout"
        assert metadata.payload == {"checkpoint": "latest.ckpt"}
        assert isinstance(metadata.requested_at_unix_seconds, float)

    def test_read_missing_file_returns_none(self, tmp_path: Path):
        assert training_module._read_restart_metadata_file(tmp_path / "absent.json") is None

    def test_on_disk_schema_keys_are_stable(self, tmp_path: Path):
        path = tmp_path / "restart.json"
        training_module._write_restart_metadata_file(path, reason="timeout")
        import json

        data = json.loads(path.read_text())
        assert set(data) == {
            "restart_requested",
            "reason",
            "requested_at_unix_seconds",
            "payload",
        }

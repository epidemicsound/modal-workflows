# mypy: ignore-errors
from pathlib import Path
from unittest.mock import patch

import pytest

import modal_workflows.workflows.state as state_module
from modal_workflows.workflows.state import WorkflowState


class FakeModalDict:
    def __init__(self):
        self._data = {}

    def contains(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def pop(self, key, default=None):
        return self._data.pop(key, default)

    def clear(self):
        self._data.clear()


class FakeModalVolume:
    def __init__(self):
        self.commit_called = False

    def commit(self):
        self.commit_called = True

    def reload(self):
        return None


@pytest.fixture
def fake_dict():
    return FakeModalDict()


@pytest.fixture
def fake_volume():
    return FakeModalVolume()


@pytest.fixture
def workflow_state(fake_dict, fake_volume, tmp_path: Path):
    with patch.object(state_module, "modal") as mock_modal:
        mock_modal.Dict.from_name.return_value = fake_dict
        with (
            patch.object(
                state_module.workflow_volumes, "WORKFLOW_STATE_VOLUME_MOUNT_PATH", tmp_path
            ),
            patch.object(
                state_module.workflow_volumes, "create_or_get_volume", return_value=fake_volume
            ),
        ):
            state = WorkflowState("test-run")
            yield state


class TestWorkflowState:
    def test_fresh_state_has_no_completed_steps(self, workflow_state):
        assert workflow_state.completed_steps == set()

    def test_cache_and_retrieve_step_result(self, workflow_state, tmp_path: Path):
        workflow_state.cache_step_result("train", {"loss": 0.01})

        is_cached, result = workflow_state.get_cached_step_result("train")

        assert is_cached is True
        assert result == {"loss": 0.01}
        assert (tmp_path / "test-run" / "step_results" / "train.pkl").exists()

    def test_uncached_step_returns_false_and_none(self, workflow_state):
        is_cached, result = workflow_state.get_cached_step_result("nonexistent")

        assert is_cached is False
        assert result is None

    def test_non_picklable_result_raises_clear_error(self, workflow_state):
        with pytest.raises(
            ValueError,
            match="non-picklable result",
        ):
            workflow_state.cache_step_result("train", lambda x: x)

        is_cached, result = workflow_state.get_cached_step_result("train")
        assert is_cached is False
        assert result is None
        assert workflow_state.completed_steps == set()

    def test_missing_result_file_returns_uncached(self, workflow_state):
        is_cached, result = workflow_state.get_cached_step_result("nonexistent_step")

        assert is_cached is False
        assert result is None

    def test_completed_steps_contains_all_cached_steps(self, workflow_state):
        workflow_state.cache_step_result("generate_data", "data")
        workflow_state.cache_step_result("train_model", "model")
        workflow_state.cache_step_result("evaluate", "metrics")

        assert workflow_state.completed_steps == {
            "generate_data",
            "train_model",
            "evaluate",
        }

    def test_cache_and_clear_function_call_id(self, workflow_state):
        workflow_state.cache_function_call_id("train", "fc-123")

        is_cached, function_call_id = workflow_state.get_cached_function_call_id("train")
        assert is_cached is True
        assert function_call_id == "fc-123"

        workflow_state.clear_cached_function_call_id("train")

        is_cached_after_clear, function_call_id_after_clear = (
            workflow_state.get_cached_function_call_id("train")
        )
        assert is_cached_after_clear is False
        assert function_call_id_after_clear is None

    def test_clear_removes_all_entries(self, workflow_state, tmp_path: Path):
        workflow_state.cache_step_result("step_1", "result")
        workflow_state.cache_function_call_id("step_1", "fc-1")
        result_file = tmp_path / "test-run" / "step_results" / "step_1.pkl"
        assert result_file.exists() is True

        workflow_state.clear()

        assert workflow_state.completed_steps == set()
        is_cached, _ = workflow_state.get_cached_step_result("step_1")
        assert is_cached is False
        is_function_call_cached, _ = workflow_state.get_cached_function_call_id("step_1")
        assert is_function_call_cached is False
        assert result_file.exists() is False

    def test_rejects_state_name_that_escapes_the_volume(self):
        with pytest.raises(ValueError, match="Invalid state_name"):
            WorkflowState("../../etc")

    def test_rejects_step_id_that_escapes_the_results_directory(self, workflow_state):
        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_state.cache_step_result("../../evil", "payload")

        with pytest.raises(ValueError, match="Invalid step_id"):
            workflow_state.get_cached_step_result("../../evil")

    def test_commit_persists_state_volume(self, workflow_state, fake_volume):
        workflow_state.commit()
        assert fake_volume.commit_called is True

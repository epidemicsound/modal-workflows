# mypy: ignore-errors
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

import modal_workflows.workflows.state as state_module
from modal_workflows.workflows.identifiers import MAX_IDENTIFIER_LENGTH
from modal_workflows.workflows.state import (
    STATE_NAME_HASH_LENGTH,
    WorkflowState,
    _stable_short_name,
)

from tests.workflows import fakes  # isort: skip
from tests.workflows.fakes import FakeModalDict  # isort: skip


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
def durable_state(fake_dict, tmp_path: Path):
    """A state over a volume that loses whatever was written but not committed."""
    live = tmp_path / "live"
    live.mkdir()
    volume = fakes.FakeModalVolume(live, tmp_path / "snapshot")
    with patch.object(state_module, "modal") as mock_modal:
        mock_modal.Dict.from_name.return_value = fake_dict
        with (
            patch.object(state_module.workflow_volumes, "WORKFLOW_STATE_VOLUME_MOUNT_PATH", live),
            patch.object(
                state_module.workflow_volumes, "create_or_get_volume", return_value=volume
            ),
        ):
            yield WorkflowState("test-run"), volume


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
        assert (workflow_state.results_root / "step_results" / "train.pkl").exists()

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

    @pytest.mark.parametrize(
        "payload",
        [pickle.dumps({"loss": 0.01})[:-3], b"", b"\x00not a pickle"],
        ids=["truncated", "empty", "garbage"],
    )
    def test_unreadable_result_file_counts_as_not_cached(self, workflow_state, payload):
        workflow_state.cache_step_result("train", None)
        (workflow_state.step_results_directory / "train.pkl").write_bytes(payload)

        assert workflow_state.get_cached_step_result("train") == (False, None)

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

    def test_cached_function_call_ids_returns_every_id_by_step(self, workflow_state, fake_dict):
        workflow_state.cache_function_call_id("map_0", "fc-0")
        workflow_state.cache_function_call_id("map_1", "fc-1")
        fake_dict["unrelated"] = "x"

        assert workflow_state.cached_function_call_ids() == {"map_0": "fc-0", "map_1": "fc-1"}

    def test_clearing_a_function_call_id_that_was_never_cached_is_harmless(self, workflow_state):
        workflow_state.clear_cached_function_call_id("never-cached")

        assert workflow_state.get_cached_function_call_id("never-cached") == (False, None)

    def test_clear_removes_all_entries(self, workflow_state, tmp_path: Path):
        workflow_state.cache_step_result("step_1", "result")
        workflow_state.cache_function_call_id("step_1", "fc-1")
        result_file = workflow_state.results_root / "step_results" / "step_1.pkl"
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

    def test_results_cached_without_a_commit_become_durable_on_commit_pending(self, durable_state):
        state, volume = durable_state
        state.cache_step_result("a", 1, commit=False)
        state.cache_step_result("b", 2, commit=False)

        state.commit_pending()
        volume.crash()

        assert state.read_step_result("a") == (True, 1)
        assert state.read_step_result("b") == (True, 2)

    def test_reload_keeps_results_cached_without_a_commit(self, durable_state):
        state, _ = durable_state
        state.cache_step_result("a", 1, commit=False)
        assert state.read_step_result("a") == (True, 1)

        state.reload()

        assert state.read_step_result("a") == (True, 1)

    def test_results_committed_by_another_container_are_seen_after_reload(self, durable_state):
        state, volume = durable_state
        volume.on_reload = lambda: state.cache_step_result("other", "x")

        assert state.get_cached_step_result("other") == (True, "x")

    def test_commit_persists_state_volume(self, workflow_state, fake_volume):
        workflow_state.commit()
        assert fake_volume.commit_called is True


class TestStateNameHashing:
    def _make_state(self, name: str, fake_dict, fake_volume, tmp_path: Path) -> None:
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
                WorkflowState(name)
        return mock_modal

    def test_short_name_is_hashed_to_a_fixed_length(self, fake_dict, fake_volume, tmp_path: Path):
        name = "short"
        mock_modal = self._make_state(name, fake_dict, fake_volume, tmp_path)

        expected = f"state-{_stable_short_name(name)}"
        mock_modal.Dict.from_name.assert_called_once_with(expected, create_if_missing=True)
        assert len(expected) == len("state-") + STATE_NAME_HASH_LENGTH

    def test_longest_allowed_name_fits_modal_name_limit(
        self, fake_dict, fake_volume, tmp_path: Path
    ):
        name = "a" * MAX_IDENTIFIER_LENGTH
        assert len(name) > 64

        mock_modal = self._make_state(name, fake_dict, fake_volume, tmp_path)

        expected = f"state-{_stable_short_name(name)}"
        mock_modal.Dict.from_name.assert_called_once_with(expected, create_if_missing=True)
        assert len(expected) <= 64

    def test_same_name_produces_the_same_hash(self, fake_dict, fake_volume, tmp_path: Path):
        name = "example-run"

        mock_modal_first = self._make_state(name, fake_dict, FakeModalVolume(), tmp_path)
        mock_modal_second = self._make_state(name, FakeModalDict(), FakeModalVolume(), tmp_path)

        first_call = mock_modal_first.Dict.from_name.call_args
        second_call = mock_modal_second.Dict.from_name.call_args
        assert first_call == second_call

    def test_different_names_produce_different_hashes(self, fake_dict, fake_volume, tmp_path: Path):
        mock_modal_a = self._make_state("run-a", fake_dict, fake_volume, tmp_path)
        mock_modal_b = self._make_state("run-b", FakeModalDict(), FakeModalVolume(), tmp_path)

        name_a = mock_modal_a.Dict.from_name.call_args.args[0]
        name_b = mock_modal_b.Dict.from_name.call_args.args[0]
        assert name_a != name_b

# mypy: ignore-errors
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import modal

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.identifiers import validate_state_name, validate_step_id

logger = logging.getLogger(__name__)

FUNCTION_CALL_ID_KEY_PREFIX = "function_call_id_"
FUNCTION_CALL_ID_KEY_SUFFIX = "_cached"
WORKFLOW_RESULT_FILE_SUFFIX = ".pkl"


class WorkflowState:
    """Persistent workflow state.

    Step results are stored on a Modal volume as pickle payloads, and async
    function-call identifiers are stored in a Modal dict for restart inheritance.
    """

    def __init__(self, name: str) -> None:
        """Initializes the state, creating the backing stores if they do not exist.

        Args:
            name (str): Name of the workflow state. Used as a directory name on the
                state volume, so it must be a valid single path component.

        Raises:
            ValueError: If *name* is not a valid path identifier.
        """
        self._name = validate_state_name(name)

        # only used for ephemeral function-call ID bookkeeping
        self._dict = modal.Dict.from_name(f"state-{name}", create_if_missing=True)

        self._volume = workflow_volumes.create_or_get_volume(
            workflow_volumes.WORKFLOW_STATE_VOLUME_NAME,
        )
        self._results_root = (
            workflow_volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH / name / "step_results"
        )
        self._results_root.mkdir(parents=True, exist_ok=True)

    @property
    def results_root(self) -> Path:
        """
        Returns the path to the results root directory.
        Do not use this outside the modal_workflows package, because everything will be
        deleted after the run is done.
        Returns:
            Path: The path to the results root directory.
        """
        results_root_path = workflow_volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH / self._name
        if not results_root_path.exists():
            results_root_path.mkdir(parents=True, exist_ok=True)
        return results_root_path

    @property
    def step_results_directory(self) -> Path:
        """Returns the path to the step results directory.
        Returns:
            Path: The path to the results root directory.
        """
        step_results_root_path = self.results_root / "step_results"
        if not step_results_root_path.exists():
            step_results_root_path.mkdir(parents=True, exist_ok=True)

        return step_results_root_path

    def get_cached_step_result(self, step_id: str) -> tuple[bool, Any]:
        """Looks up a cached result for a given step.

        Args:
            step_id (str): The identifier of the step to look up.

        Returns:
            tuple[bool, Any]: A tuple of (is_cached, result). If not cached, result is None.
        """
        result_path = self._step_result_path(step_id)
        self._volume.reload()
        if not result_path.exists():
            return False, None

        with result_path.open("rb") as result_file:
            return True, pickle.load(result_file)

    def cache_step_result(self, step_id: str, result: Any) -> None:
        """Stores a step result on the volume.

        Args:
            step_id (str): The identifier of the step.
            result (Any): The result to cache. Must be pickle-serializable.
        """
        result_path = self._step_result_path(step_id)
        payload = self._serialize_step_result(step_id, result)
        with result_path.open("wb") as result_file:
            result_file.write(payload)
        self._volume.commit()

    def get_cached_function_call_id(self, step_id: str) -> tuple[bool, str | None]:
        """Looks up a cached function-call identifier for a given step.

        Args:
            step_id (str): The identifier of the step to look up.

        Returns:
            tuple[bool, str | None]: A tuple of (is_cached, function_call_id).
                If not cached, function_call_id is None.
        """
        key = self._function_call_id_key(step_id)
        if self._dict.contains(key):
            return True, self._dict[key]
        return False, None

    def cache_function_call_id(self, step_id: str, function_call_id: str) -> None:
        """Stores a function-call identifier for an async step.

        Args:
            step_id (str): The identifier of the step.
            function_call_id (str): The function-call identifier to cache.
        """
        key = self._function_call_id_key(step_id)
        self._dict[key] = function_call_id

    def clear_cached_function_call_id(self, step_id: str) -> None:
        """Removes a cached function-call identifier for a step, if present.

        Args:
            step_id (str): The identifier of the step.
        """
        key = self._function_call_id_key(step_id)
        if self._dict.contains(key):
            self._dict.pop(key, None)

    def commit(self) -> None:
        """Commits the state volume so that writes to results_root are persisted."""
        self._volume.commit()

    @property
    def completed_steps(self) -> set[str]:
        """Returns the set of step IDs that have been completed.

        Returns:
            set[str]: Set of completed step identifiers.
        """
        self._volume.reload()
        result_files = self.step_results_directory.glob(f"*{WORKFLOW_RESULT_FILE_SUFFIX}")
        return {p.stem for p in result_files}

    def _delete_dir(self, directory: Path) -> None:
        for file_or_dir in directory.iterdir():
            if file_or_dir.is_dir():
                self._delete_dir(file_or_dir)
            else:
                file_or_dir.unlink()

    def clear(self) -> None:
        """Removes all entries from the state."""
        for result_file_or_dir in self.results_root.iterdir():
            if result_file_or_dir.is_dir():
                self._delete_dir(result_file_or_dir)
                result_file_or_dir.rmdir()
            else:
                result_file_or_dir.unlink()
        self._volume.commit()
        self._dict.clear()

    def __str__(self) -> str:
        """Returns a human-readable summary of the current state.

        Returns:
            str: "empty" when no steps are cached, otherwise a count and list of cached step IDs.
        """
        steps = self.completed_steps
        if steps:
            return f"{len(steps)} cached step(s): {', '.join(sorted(steps))}"
        return "empty"

    @staticmethod
    def _function_call_id_key(step_id: str) -> str:
        return f"{FUNCTION_CALL_ID_KEY_PREFIX}{step_id}{FUNCTION_CALL_ID_KEY_SUFFIX}"

    def _step_result_path(self, step_id: str) -> Path:
        validate_step_id(step_id)
        return self.step_results_directory / f"{step_id}{WORKFLOW_RESULT_FILE_SUFFIX}"

    @staticmethod
    def _serialize_step_result(step_id: str, result: Any) -> bytes:
        try:
            return pickle.dumps(result)
        except Exception as exc:
            result_type = type(result).__name__
            raise ValueError(
                f"Step '{step_id}' returned a non-picklable result of type '{result_type}'. "
                "Return a pickle-serializable value.",
            ) from exc

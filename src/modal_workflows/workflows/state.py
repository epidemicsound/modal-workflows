# mypy: ignore-errors
from __future__ import annotations

import asyncio
import hashlib
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

STATE_DICT_NAME_PREFIX = "state-"
STATE_NAME_HASH_LENGTH = 32


def _finished_within(awaitable: Any, timeout: float) -> bool:
    """Runs *awaitable* to completion, or cancels it and returns False after *timeout* seconds."""
    try:
        asyncio.run(asyncio.wait_for(awaitable, timeout))
    except asyncio.TimeoutError:
        return False
    return True


def _stable_short_name(name: str) -> str:
    """Hashes to a fixed length, comfortably under Modal's 64-character object-name cap
    regardless of how long or oddly-shaped *name* is."""
    return hashlib.sha256(name.encode()).hexdigest()[:STATE_NAME_HASH_LENGTH]


class WorkflowState:
    """Persistent workflow state.

    Step results are stored on a Modal volume as pickle payloads, and async
    function-call identifiers are stored in a Modal dict for restart inheritance.
    """

    def __init__(self, name: str) -> None:
        """Initializes the state, creating the backing stores if they do not exist.

        Args:
            name (str): Name of the workflow state. Validated as a path identifier, then
                hashed to a fixed-length identifier before use, since Modal object names
                are capped at 64 characters. The mapping is logged so it stays traceable
                (e.g. when searching Modal's dashboard by state_name).

        Raises:
            ValueError: If *name* is not a valid path identifier.
        """
        self._name = _stable_short_name(validate_state_name(name))
        logger.info(f"Workflow state {name!r} stored under {STATE_DICT_NAME_PREFIX}{self._name}")

        self._dict_name = f"{STATE_DICT_NAME_PREFIX}{self._name}"
        self._dict = modal.Dict.from_name(self._dict_name, create_if_missing=True)

        self._volume = workflow_volumes.create_or_get_volume(
            workflow_volumes.WORKFLOW_STATE_VOLUME_NAME,
        )
        self._has_uncommitted_results = False

    @property
    def results_root(self) -> Path:
        """This state's results directory; package-internal, deleted when the run finishes."""
        results_root_path = workflow_volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH / self._name
        if not results_root_path.exists():
            results_root_path.mkdir(parents=True, exist_ok=True)
        return results_root_path

    @property
    def step_results_directory(self) -> Path:
        step_results_root_path = self.results_root / "step_results"
        if not step_results_root_path.exists():
            step_results_root_path.mkdir(parents=True, exist_ok=True)

        return step_results_root_path

    def get_cached_step_result(self, step_id: str) -> tuple[bool, Any]:
        """Looks up a cached result for a given step, after refreshing the volume.

        Args:
            step_id (str): The identifier of the step to look up.

        Returns:
            tuple[bool, Any]: A tuple of (is_cached, result). If not cached, result is None.
        """
        self.reload()
        return self.read_step_result(step_id)

    def read_step_result(self, step_id: str) -> tuple[bool, Any]:
        """Reads a cached step result from the volume as currently mounted, without reloading.

        Args:
            step_id (str): The identifier of the step to look up.

        Returns:
            tuple[bool, Any]: A tuple of (is_cached, result). If not cached, result is None.
                A truncated or corrupt result file counts as not cached.
        """
        try:
            with self._step_result_path(step_id).open("rb") as result_file:
                return True, pickle.load(result_file)
        except FileNotFoundError:
            return False, None
        except (EOFError, pickle.UnpicklingError) as exc:
            logger.warning(
                f"Cached result for step '{step_id}' is unreadable ({exc!r}); "
                "treating the step as not cached."
            )
            return False, None

    def reload(self) -> None:
        """Makes results committed by other containers visible to this one.

        Results written here but not yet committed are committed first, so a reload can
        never be the thing that drops them.
        """
        self.commit_pending()
        self._volume.reload()

    def cache_step_result(self, step_id: str, result: Any, commit: bool = True) -> None:
        """Stores a step result on the volume.

        Args:
            step_id (str): The identifier of the step.
            result (Any): The result to cache. Must be pickle-serializable.
            commit (bool): When False the result is written without a commit; the caller
                must call ``commit_pending()`` before it becomes visible to other containers.
        """
        result_path = self._step_result_path(step_id)
        payload = self._serialize_step_result(step_id, result)
        with result_path.open("wb") as result_file:
            result_file.write(payload)
        self._has_uncommitted_results = True
        if commit:
            self.commit()

    def get_cached_function_call_id(self, step_id: str) -> tuple[bool, str | None]:
        """Looks up a cached function-call identifier for a given step.

        Args:
            step_id (str): The identifier of the step to look up.

        Returns:
            tuple[bool, str | None]: A tuple of (is_cached, function_call_id).
                If not cached, function_call_id is None.
        """
        function_call_id = self._dict.get(self._function_call_id_key(step_id))
        if function_call_id is None:
            return False, None
        return True, function_call_id

    def cached_function_call_ids(self) -> dict[str, str]:
        """Returns every cached function-call identifier, keyed by step id, in one request."""
        call_ids = {}
        for key, function_call_id in self._dict.items():
            if (
                isinstance(key, str)
                and key.startswith(FUNCTION_CALL_ID_KEY_PREFIX)
                and key.endswith(FUNCTION_CALL_ID_KEY_SUFFIX)
            ):
                step_id = key[len(FUNCTION_CALL_ID_KEY_PREFIX) : -len(FUNCTION_CALL_ID_KEY_SUFFIX)]
                call_ids[step_id] = function_call_id
        return call_ids

    def cache_function_call_id(
        self, step_id: str, function_call_id: str, timeout: float | None = None
    ) -> bool:
        """Stores a function-call identifier for an async step.

        Args:
            step_id (str): The identifier of the step.
            function_call_id (str): The function-call identifier to cache.
            timeout (float | None): Optional max seconds to wait for the write. Must not be
                called from a running event loop when set.

        Returns:
            bool: False if *timeout* expired first, in which case the id may not be stored.
        """
        key = self._function_call_id_key(step_id)
        if timeout is None:
            self._dict[key] = function_call_id
            return True
        return _finished_within(self._dict.put.aio(key, function_call_id), timeout)

    def clear_cached_function_call_id(self, step_id: str) -> None:
        """Removes a cached function-call identifier for a step, if present."""
        self._dict.pop(self._function_call_id_key(step_id), None)

    def clear_all_cached_function_call_ids(self, timeout: float) -> bool:
        """Removes every cached function-call identifier in one request.

        Must not be called from a running event loop. Returns False if *timeout* expired
        first, in which case some or all ids may remain.
        """
        return _finished_within(self._dict.clear.aio(), timeout)

    def commit(self) -> None:
        """Commits the state volume so that writes to results_root are persisted."""
        self._volume.commit()
        self._has_uncommitted_results = False

    def commit_pending(self) -> None:
        """Commits the state volume only if results were written since the last commit."""
        if self._has_uncommitted_results:
            self.commit()

    @property
    def completed_steps(self) -> set[str]:
        """Step ids with a cached result, after reloading the volume."""
        self.reload()
        result_files = self.step_results_directory.glob(f"*{WORKFLOW_RESULT_FILE_SUFFIX}")
        return {p.stem for p in result_files}

    def _delete_dir(self, directory: Path) -> None:
        for file_or_dir in directory.iterdir():
            if file_or_dir.is_dir():
                self._delete_dir(file_or_dir)
            else:
                file_or_dir.unlink()

    def _clear_results(self) -> None:
        for result_file_or_dir in self.results_root.iterdir():
            if result_file_or_dir.is_dir():
                self._delete_dir(result_file_or_dir)
                result_file_or_dir.rmdir()
            else:
                result_file_or_dir.unlink()
        self.commit()

    def clear(self) -> None:
        """Removes all entries from the state."""
        self._clear_results()
        self._dict.clear()

    def delete(self) -> None:
        """Removes all results and deletes the function-call id Dict, recreated if used again."""
        self._clear_results()
        modal.Dict.objects.delete(self._dict_name, allow_missing=True)
        self._dict = modal.Dict.from_name(self._dict_name, create_if_missing=True)

    def __str__(self) -> str:
        """Returns "empty", or the count and ids of the cached steps."""
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

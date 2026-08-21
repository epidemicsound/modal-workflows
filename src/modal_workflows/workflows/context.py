# mypy: ignore-errors
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.alerts import modal_app_url
from modal_workflows.workflows.identifiers import validate_state_name, validate_step_id
from modal_workflows.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

WORKFLOW_STEP_MARKER = "_workflow_step"
_WORKFLOW_EVENT_LOG_FILENAME = "workflow_event_log.txt"


def _validate_workflow_step(modal_func: Any, method: str) -> None:
    """Raises TypeError if *modal_func* was not decorated with @workflow_function()."""
    if not getattr(modal_func, WORKFLOW_STEP_MARKER, False):
        raise TypeError(
            f"ctx.{method}() requires functions decorated with @workflow_function(). "
            f"Use @workflow_function(app) instead of @app.function() to ensure "
            f"the workflow artifacts volume is mounted."
        )


class WorkflowRestarted(Exception):
    """Raised to unwind the stack when a workflow triggers a graceful restart.

    The wrapper in the workflow decorator catches this and waits on the
    newly spawned instance.
    """

    def __init__(self, function_call: modal.functions.FunctionCall) -> None:
        """Wraps the FunctionCall handle for the newly spawned workflow continuation.

        Args:
            function_call (modal.functions.FunctionCall): Handle to the spawned continuation.
        """
        self.function_call = function_call


class WorkflowManagedFunctionCall:
    """Wraps a Modal FunctionCall so get() remains workflow-timeout aware.

    Calling ``modal.FunctionCall.get()`` directly can block until completion,
    which would prevent the workflow orchestrator from checking its own timeout.
    This wrapper routes waiting through ``WorkflowContext.wait_for_function_call()``
    so graceful restarts still happen when needed.
    """

    def __init__(
        self,
        context: WorkflowContext,
        function_call: modal.FunctionCall,
        step_id: str,
    ) -> None:
        self._context = context
        self._function_call = function_call
        self._step_id = step_id

    def get(self, *args: Any, wait_timeout: float | None = None, **kwargs: Any) -> Any:
        """Waits for completion while preserving workflow restart behavior."""
        is_cached, cached_result = self._context.state.get_cached_step_result(self._step_id)
        if is_cached:
            logger.debug(f"Skipping spawn '{self._step_id}': found cached result.")
            self._context.state.clear_cached_function_call_id(self._step_id)
            return cached_result

        try:
            if wait_timeout is not None:
                kwargs["timeout"] = wait_timeout
            result = self._context.wait_for_function_call(self._function_call, *args, **kwargs)
        except WorkflowRestarted:
            raise
        except Exception:
            self._context.state.clear_cached_function_call_id(self._step_id)
            raise

        self._context.state.cache_step_result(self._step_id, result)
        self._context.state.clear_cached_function_call_id(self._step_id)
        logger.debug(f"Spawn '{self._step_id}' completed and cached.")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._function_call, name)


class WorkflowContext:
    """Runtime context passed into functions decorated with ``@workflow``.

    Provides timeout-aware orchestration helpers (``step``, ``spawn``, ``map``,
    ``parallel``), access to persisted workflow state, and artifact paths.
    """

    def __init__(
        self,
        app: modal.App,
        func: modal.Function,
        func_name: str,
        state_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: float,
    ) -> None:
        """Creates a new context for a single workflow invocation.

        Args:
            app (modal.App): The Modal app this workflow belongs to.
            func (modal.Function): The wrapped Modal function, used for spawning restarts.
            func_name (str): Human-readable name of the workflow function.
            state_name (str): Unique state name for this workflow execution.
            args (tuple[Any, ...]): Positional arguments the workflow was called with.
            kwargs (dict[str, Any]): Keyword arguments the workflow was called with, including
                the propagated state name kwarg so graceful restarts continue on the same state.
            timeout (float): Maximum wall-clock seconds before a graceful restart is triggered.

        Raises:
            ValueError: If *state_name* is not a valid path identifier.
        """
        self.app = app
        self.func = func
        self.func_name = func_name
        self.state_name = validate_state_name(state_name)
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout
        self.start_time = time.time()

        self.state = WorkflowState(self.state_name)
        self._seen_step_ids: set[str] = set()

        self.artifacts_volume = workflow_volumes.create_or_get_volume(
            workflow_volumes.WORKFLOW_ARTIFACTS_VOLUME_NAME,
        )

        self._log_state_on_init()

    @property
    def _artifacts_root_path(self) -> Path:
        root_path = workflow_volumes.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH
        if not root_path.exists():
            raise RuntimeError(
                f"Workflow artifacts volume is not mounted at '{root_path}'.",
            )
        return root_path

    @property
    def artifacts_run_path(self) -> Path:
        run_path = self._artifacts_root_path / self.state_name
        if not run_path.exists():
            run_path.mkdir(parents=True, exist_ok=True)
            self.artifacts_volume.commit()
        return run_path

    @property
    def artifacts_shared_path(self) -> Path:
        shared_path = (
            self._artifacts_root_path / workflow_volumes.WORKFLOW_ARTIFACTS_SHARED_DIRECTORY
        )
        if not shared_path.exists():
            shared_path.mkdir(parents=True, exist_ok=True)
            self.artifacts_volume.commit()
        return shared_path

    @property
    def dashboard_url(self) -> str | None:
        """Returns the Modal dashboard URL for the running app, or None if unavailable."""
        app_id = self.app.app_id
        return modal_app_url(app_id)

    def commit_artifacts_volume(self) -> None:
        """Commits the artifacts volume to persist files."""
        self.artifacts_volume.commit()

    def _register_step_id(self, step_id: str) -> None:
        """Validates *step_id* and raises ValueError if it was already used in this run."""
        validate_step_id(step_id)
        if step_id in self._seen_step_ids:
            raise ValueError(
                f"Duplicate step id '{step_id}'. Each call to step(), spawn(), map(), or "
                "parallel() must use a unique step_id within the same workflow run."
            )
        self._seen_step_ids.add(step_id)

    def _log_state_on_init(self) -> None:
        """Logs the current workflow state at debug level on initialization."""
        logger.debug(f"Workflow '{self.func_name}' state: {self.state}")

    def log_event(self, message: str) -> None:
        """Appends a timestamped message to the workflow event log in the artifacts volume."""
        try:
            log_path = self.artifacts_run_path / _WORKFLOW_EVENT_LOG_FILENAME
            timestamp = datetime.now(timezone.utc).isoformat()
            with log_path.open("a") as log_file:
                log_file.write(f"{timestamp} | {message}\n")
            self.commit_artifacts_volume()
        except Exception as exc:
            logger.debug("Failed to write to workflow event log: %s", exc)

    def wait_for_function_call(
        self,
        function_call: modal.FunctionCall,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Workflow-timeout aware wait for a function call.

        Args:
            function_call (modal.FunctionCall): Modal function call handle to monitor.
            *args (Any): Positional arguments forwarded to FunctionCall.get().
            timeout (float | None): Optional max wait in seconds before raising TimeoutError.
            **kwargs (Any): Keyword arguments forwarded to FunctionCall.get().
        """

        poll_interval_seconds = 5
        deadline = None if timeout is None else time.time() + timeout

        while True:
            self.restart_if_needed()
            try:
                return function_call.get(*args, timeout=0, **kwargs)
            except TimeoutError as exc:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Function call did not complete before timeout.") from exc
                time.sleep(poll_interval_seconds)

    @staticmethod
    def _extract_function_call_id(function_call: modal.FunctionCall) -> str | None:
        """Extracts a stable function-call identifier from a Modal FunctionCall."""
        call_id = getattr(function_call, "object_id", None)
        if isinstance(call_id, str) and call_id:
            return call_id
        return None

    @staticmethod
    def _restore_function_call(function_call_id: str) -> modal.FunctionCall | None:
        """Best-effort restoration of a FunctionCall handle from id."""
        try:
            return modal.FunctionCall.from_id(function_call_id)
        except Exception:
            return None

    @staticmethod
    def _cancel_function_call(function_call: modal.FunctionCall, step_id: str) -> None:
        """Best-effort cancellation for stale inherited async jobs."""
        try:
            function_call.cancel()
            logger.warning(f"Cancelled stale inherited function call for step '{step_id}'.")
        except Exception as exc:
            logger.warning(
                f"Failed to cancel stale inherited function call for step '{step_id}': {exc}",
            )

    def _spawn_or_inherit_function_call(
        self,
        step_id: str,
        spawn_fn: Callable[[], modal.FunctionCall],
    ) -> modal.FunctionCall:
        """Returns an inherited async call when possible, else spawns a new call."""
        has_cached_id, cached_call_id = self.state.get_cached_function_call_id(step_id)
        if has_cached_id and cached_call_id:
            inherited_call = self._restore_function_call(cached_call_id)
            if inherited_call is not None:
                logger.debug(
                    f"Inheriting async call for step '{step_id}' from call id '{cached_call_id}'.",
                )
                return inherited_call

            logger.warning(
                f"Could not restore async call for step '{step_id}' from id '{cached_call_id}'. "
                "Spawning replacement.",
            )
            self.state.clear_cached_function_call_id(step_id)

        self.restart_if_needed()

        new_call = spawn_fn()
        call_id = self._extract_function_call_id(new_call)
        if call_id:
            self.state.cache_function_call_id(step_id, call_id)
            logger.debug(f"Cached function call id '{call_id}' for step '{step_id}'.")
        else:
            logger.warning(
                f"Spawned async call for step '{step_id}' has no stable id; "
                "inheritance may be unavailable on restart.",
            )
        return new_call

    def step(
        self,
        modal_func: modal.Function,
        step_id: str,
        *args: Any,
        wait_timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Executes one workflow step synchronously and caches its result.

        Args:
            modal_func (modal.Function): A function decorated with
                ``@workflow_function(app)``.
            step_id (str): Stable identifier used for cache lookup across restarts.
                Must be unique across all step(), spawn(), map(), and parallel() calls
                within the same workflow run.
            *args (Any): Positional arguments forwarded to the function.
            wait_timeout (float | None): Optional max wait in seconds for completion.
            **kwargs (Any): Keyword arguments forwarded to the function.

        Returns:
            Any: The return value of the Modal function call.

        Raises:
            ValueError: If *step_id* has already been used in this workflow run.
        """
        _validate_workflow_step(modal_func, "step")
        self._register_step_id(step_id)
        is_cached, cached_result = self.state.get_cached_step_result(step_id)
        if is_cached:
            logger.debug(f"Skipping step '{step_id}': found cached result.")
            self.state.clear_cached_function_call_id(step_id)
            return cached_result

        function_call = self._spawn_or_inherit_function_call(
            step_id,
            lambda: modal_func.spawn(*args, **kwargs),
        )
        self.log_event(f"Running step {step_id}")

        try:
            result = self.wait_for_function_call(function_call, timeout=wait_timeout)
        except WorkflowRestarted:
            raise
        except Exception:
            self._cancel_function_call(function_call, step_id)
            self.state.clear_cached_function_call_id(step_id)
            raise

        self.state.cache_step_result(step_id, result)
        self.state.clear_cached_function_call_id(step_id)
        logger.debug(f"Step '{step_id}' completed and cached.")
        return result

    def spawn(
        self,
        modal_func: modal.Function,
        step_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> WorkflowManagedFunctionCall:
        """Spawns one workflow step asynchronously and caches its function-call id.

        On restart, this step attempts to inherit the previous async call by id.
        If inheritance is not possible, a replacement call is spawned.

        Args:
            modal_func (modal.Function): A function decorated with
                ``@workflow_function(app)``.
            step_id (str): Stable identifier used for cache lookup across restarts.
                Must be unique across all step(), spawn(), map(), and parallel() calls
                within the same workflow run.
            *args (Any): Positional arguments forwarded to the function.
            **kwargs (Any): Keyword arguments forwarded to the function.

        Returns:
            WorkflowManagedFunctionCall: A handle to the spawned call whose .get()
                remains workflow-timeout aware.

        Raises:
            ValueError: If *step_id* has already been used in this workflow run.
        """
        _validate_workflow_step(modal_func, "spawn")
        self._register_step_id(step_id)
        is_cached, cached_result = self.state.get_cached_step_result(step_id)
        if is_cached:
            logger.debug(f"Skipping spawn '{step_id}': found cached result.")
            self.state.clear_cached_function_call_id(step_id)
            return WorkflowManagedFunctionCall(self, None, step_id)

        self.log_event(f"Spawned {step_id}")
        function_call = self._spawn_or_inherit_function_call(
            step_id,
            lambda: modal_func.spawn(*args, **kwargs),
        )
        return WorkflowManagedFunctionCall(self, function_call, step_id)

    def _run_batch(
        self,
        step_id: str,
        count: int,
        spawn_for_index: Callable[[int], Any],
        wait_timeout: float | None = None,
    ) -> list[Any]:
        """Shared batch execution logic for map and parallel.

        For each sub-step ({step_id}_{index}), this method:
        - uses cached step results when available,
        - otherwise inherits cached function-call ids when possible,
        - otherwise spawns replacement calls,
        - then resolves pending calls and caches final results.

        Args:
            step_id (str): Base identifier. Each item is keyed as {step_id}_{index}.
            count (int): Total number of items or steps.
            spawn_for_index (Callable[[int], Any]): Spawns one async call for a given index.
            wait_timeout (float | None): Optional max wait in seconds per pending item.

        Returns:
            list[Any]: Results in the original order, mixing cached and fresh results.

        Note:
            Results are collected in index order. If an earlier index is still running,
            later indices that have already finished remotely are not cached until their
            turn is reached.
        """
        results: list[Any] = [None] * count
        pending_calls: dict[int, Any] = {}

        for index in range(count):
            sub_step_id = f"{step_id}_{index}"
            is_cached, cached_result = self.state.get_cached_step_result(sub_step_id)
            if is_cached:
                logger.debug(f"Skipping step '{sub_step_id}': found cached result.")
                results[index] = cached_result
                self.state.clear_cached_function_call_id(sub_step_id)
                continue

            pending_calls[index] = self._spawn_or_inherit_function_call(
                sub_step_id,
                lambda i=index: spawn_for_index(i),
            )

        for index, function_call in pending_calls.items():
            sub_step_id = f"{step_id}_{index}"
            try:
                result = self.wait_for_function_call(function_call, timeout=wait_timeout)
            except WorkflowRestarted:
                raise
            except Exception:
                self._cancel_function_call(function_call, sub_step_id)
                self.state.clear_cached_function_call_id(sub_step_id)
                raise

            self.state.cache_step_result(sub_step_id, result)
            self.state.clear_cached_function_call_id(sub_step_id)
            logger.debug(f"Step '{sub_step_id}' completed and cached.")
            results[index] = result

        return results

    def map(
        self,
        modal_func: modal.Function,
        items: list[Any],
        step_id: str,
        *args: Any,
        wait_timeout: float | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Runs one workflow step over many items in parallel.

        Results are cached per item (``{step_id}_{index}``), and each item also
        tracks a function-call id so in-flight work can be inherited after restart.

        Args:
            modal_func (modal.Function): A function decorated with
                ``@workflow_function(app)``.
            items (list[Any]): List of items to process. Each item is the first positional argument.
            step_id (str): Base identifier. Each item is keyed as {step_id}_{index}.
                Must be unique across all step(), spawn(), map(), and parallel() calls
                within the same workflow run.
            *args (Any): Additional positional arguments forwarded to every call.
            wait_timeout (float | None): Optional max wait in seconds per pending item.
            **kwargs (Any): Keyword arguments forwarded to every call.

        Returns:
            list[Any]: Results in the same order as items.

        Raises:
            ValueError: If *step_id* has already been used in this workflow run.
        """
        _validate_workflow_step(modal_func, "map")
        self._register_step_id(step_id)
        self.log_event(f"Launched map {step_id} ({len(items)} items)")
        return self._run_batch(
            step_id=step_id,
            count=len(items),
            spawn_for_index=lambda i: modal_func.spawn(items[i], *args, **kwargs),
            wait_timeout=wait_timeout,
        )

    def parallel(
        self,
        steps: list[tuple[modal.Function, ...]],
        step_id: str,
        wait_timeout: float | None = None,
    ) -> list[Any]:
        """Runs multiple workflow steps concurrently, caching each individually.

        Each step is a tuple of (func, *args). All uncached steps are spawned
        concurrently, then their results are collected.

        Args:
            steps (list[tuple[modal.Function, ...]]): List of tuples, each containing
                (modal_func, arg1, arg2, ...), where each ``modal_func`` is
                decorated with ``@workflow_function(app)``.
            step_id (str): Base identifier. Each step is keyed as {step_id}_{index}.
                Must be unique across all step(), spawn(), map(), and parallel() calls
                within the same workflow run.
            wait_timeout (float | None): Optional max wait in seconds per pending step.

        Returns:
            list[Any]: Results in the same order as steps.

        Raises:
            ValueError: If *step_id* has already been used in this workflow run.
        """
        for step_tuple in steps:
            _validate_workflow_step(step_tuple[0], "parallel")
        self._register_step_id(step_id)
        self.log_event(f"Launched parallel {step_id} ({len(steps)} steps)")
        return self._run_batch(
            step_id=step_id,
            count=len(steps),
            spawn_for_index=lambda i: steps[i][0].spawn(*steps[i][1:]),
            wait_timeout=wait_timeout,
        )

    @property
    def should_restart(self) -> bool:
        """Checks whether the workflow has exceeded its timeout threshold.

        Returns:
            bool: True if elapsed time exceeds the timeout, False otherwise.
        """
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout

    def restart(self) -> None:
        """Spawns a new workflow instance and raises WorkflowRestarted.

        Raises:
            WorkflowRestarted: Always raised to unwind the stack and hand off to the wrapper.
        """
        logger.warning(f"Timeout threshold reached. Respawning workflow '{self.func_name}'...")
        self.log_event("Workflow restart (timeout)")
        self.commit_artifacts_volume()
        function_call = self.func.spawn(*self.args, **self.kwargs)
        raise WorkflowRestarted(function_call)

    def restart_if_needed(self) -> None:
        """Triggers a graceful restart if the timeout threshold has been exceeded."""
        if self.should_restart:
            self.restart()

    def finish(self) -> None:
        """Clears all persisted state, signaling that the workflow completed successfully."""
        logger.debug(f"Workflow '{self.func_name}' finished successfully. Clearing state.")
        self.state.clear()

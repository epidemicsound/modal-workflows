# mypy: ignore-errors
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

import modal

from modal_workflows.workflows.failures import failure_from_exception
from modal_workflows.workflows.identifiers import validate_state_name, validate_step_id
from modal_workflows.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

MAX_PARALLEL_REQUESTS = 16
POLL_LOOKAHEAD = 4 * MAX_PARALLEL_REQUESTS
POLL_INITIAL_INTERVAL_SECONDS = 0.25
POLL_MAX_INTERVAL_SECONDS = 10
MAX_POLL_DOUBLINGS = 10
CALL_ID_WRITE_TIMEOUT_SECONDS = 30
CALL_ID_CLEAR_TIMEOUT_SECONDS = 60


def _validate_workflow_step(modal_func: Any, method: str) -> None:
    """Raises TypeError unless *modal_func* is a ``modal.Function``."""
    if not isinstance(modal_func, modal.Function):
        raise TypeError(
            f"ctx.{method}() requires a modal.Function, got {type(modal_func).__name__}. "
            f"Pass a function decorated with @app.function() or a bound method of an "
            f"@app.cls class, e.g. MyCls().method."
        )


class WorkflowRestarted(Exception):
    """Raised to unwind the stack when a workflow triggers a graceful restart.

    The wrapper in the workflow decorator catches this and returns a plain dict
    naming the newly spawned instance's Modal call id, rather than waiting on it.
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
        is_cached, cached_result = self._context._take_cached_step_result(self._step_id)
        if is_cached:
            return cached_result
        if wait_timeout is not None:
            kwargs["timeout"] = wait_timeout
        return self._context._await_step(
            self._step_id, self._function_call, *args, cancel_on_timeout=False, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._function_call, name)


def _poll_delay(attempts: int) -> float:
    """Seconds to wait after *attempts* unfinished polls: 0.25, 0.5, 1, ... capped at 10."""
    doublings = min(attempts - 1, MAX_POLL_DOUBLINGS)
    return min(POLL_INITIAL_INTERVAL_SECONDS * 2**doublings, POLL_MAX_INTERVAL_SECONDS)


class _OutcomeBuffer:
    """Outcomes of finished calls fetched ahead of their turn, plus each call's poll schedule.

    The schedule runs on a virtual clock that advances only by the time the caller sleeps,
    so every unfinished call is polled with exponential backoff and no call is polled more
    often than the head of the line would be.
    """

    def __init__(self) -> None:
        self.outcomes: dict[int, tuple[bool, Any]] = {}
        self.clock = 0.0
        self._attempts: dict[int, int] = {}
        self._next_poll_at: dict[int, float] = {}

    def due(self, window: list[int]) -> list[int]:
        """Indices in *window* that are unfinished and due for a poll."""
        return [
            index
            for index in window
            if index not in self.outcomes and self._next_poll_at.get(index, 0.0) <= self.clock
        ]

    def record(self, index: int, outcome: tuple[bool, Any] | None) -> None:
        """Stores a finished outcome, or schedules the next poll of an unfinished call."""
        if outcome is not None:
            self.outcomes[index] = outcome
            return
        self._attempts[index] = self._attempts.get(index, 0) + 1
        self._next_poll_at[index] = self.clock + _poll_delay(self._attempts[index])

    def sleep_until_next_poll(self, window: list[int]) -> None:
        """Sleeps until the earliest unfinished call in *window* is due."""
        delay = min(
            max(self._next_poll_at.get(index, 0.0) - self.clock, 0.0)
            for index in window
            if index not in self.outcomes
        )
        time.sleep(delay)
        self.clock += delay


class WorkflowContext:
    """Runtime context passed into functions decorated with ``@workflow``.

    Provides timeout-aware orchestration helpers (``step``, ``spawn``, ``map``,
    ``parallel``), access to persisted workflow state and logging.
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
        self._unsaved_call_ids: dict[str, str] = {}

        self._log_state_on_init()

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
        logger.debug(f"Workflow '{self.func_name}' state: {self.state}")

    def log_event(self, message: str) -> None:
        """Logs a workflow event message."""
        logger.info("[%s] %s", self.func_name, message)

    def wait_for_function_call(
        self,
        function_call: modal.FunctionCall,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Workflow-timeout aware wait for a function call.

        Modal signals "not finished yet" with the built-in ``TimeoutError``, and a built-in
        ``TimeoutError`` raised by the called function itself arrives as the same type, so
        such a call is waited on until *timeout* expires, or forever without one.

        Args:
            function_call (modal.FunctionCall): Modal function call handle to monitor.
            *args (Any): Positional arguments forwarded to FunctionCall.get().
            timeout (float | None): Optional max wait in seconds before raising TimeoutError.
            **kwargs (Any): Keyword arguments forwarded to FunctionCall.get().
        """
        deadline = None if timeout is None else time.time() + timeout
        unfinished_polls = 0

        while True:
            self.restart_if_needed()
            try:
                return function_call.get(*args, timeout=0, **kwargs)
            except TimeoutError as exc:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Function call did not complete before timeout.") from exc
                unfinished_polls += 1
                time.sleep(_poll_delay(unfinished_polls))

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

    def _inherit_function_call(
        self, step_id: str, cached_call_id: str | None
    ) -> modal.FunctionCall | None:
        """Returns the call restored from *cached_call_id*, or None if there is none to inherit."""
        if not cached_call_id:
            return None
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
        return None

    def _spawn_and_cache_call_id(
        self,
        step_id: str,
        spawn_fn: Callable[[], modal.FunctionCall],
        write_timeout: float | None = None,
    ) -> modal.FunctionCall:
        """Spawns a call and persists its id so a restart can inherit it.

        An id whose write outlasts *write_timeout* is kept in memory and written again
        before a graceful restart.
        """
        new_call = spawn_fn()
        call_id = getattr(new_call, "object_id", None)
        if isinstance(call_id, str) and call_id:
            if self.state.cache_function_call_id(step_id, call_id, timeout=write_timeout):
                logger.debug(f"Cached function call id '{call_id}' for step '{step_id}'.")
            else:
                logger.warning(
                    f"Saving call id '{call_id}' for step '{step_id}' timed out; "
                    "it is saved again before any restart.",
                )
                self._unsaved_call_ids[step_id] = call_id
        else:
            logger.warning(
                f"Spawned async call for step '{step_id}' has no stable id; "
                "inheritance may be unavailable on restart.",
            )
        return new_call

    def _spawn_or_inherit_function_call(
        self,
        step_id: str,
        spawn_fn: Callable[[], modal.FunctionCall],
    ) -> modal.FunctionCall:
        """Returns an inherited async call when possible, else spawns a new call."""
        _, cached_call_id = self.state.get_cached_function_call_id(step_id)
        inherited_call = self._inherit_function_call(step_id, cached_call_id)
        if inherited_call is not None:
            return inherited_call

        self.restart_if_needed()
        return self._spawn_and_cache_call_id(step_id, spawn_fn)

    def _take_cached_step_result(self, step_id: str) -> tuple[bool, Any]:
        """Returns (found, result) from the step cache, dropping any stored call id on a hit."""
        is_cached, cached_result = self.state.get_cached_step_result(step_id)
        if is_cached:
            logger.debug(f"Step '{step_id}' found cached result.")
            self.state.clear_cached_function_call_id(step_id)
        return is_cached, cached_result

    def _await_step(
        self,
        step_id: str,
        function_call: modal.FunctionCall,
        *args: Any,
        cancel_on_timeout: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Waits for *function_call*, then caches its result and clears its stored call id.

        A restart propagates untouched so the call can be inherited later. Any other
        failure cancels the call and clears its stored id. With *cancel_on_timeout* False,
        a wait that merely times out leaves the call running.
        """
        try:
            result = self.wait_for_function_call(function_call, *args, **kwargs)
        except WorkflowRestarted:
            raise
        except Exception as exc:
            if cancel_on_timeout or not isinstance(exc, TimeoutError):
                self._cancel_function_call(function_call, step_id)
            self.state.clear_cached_function_call_id(step_id)
            raise

        self.state.cache_step_result(step_id, result)
        self.state.clear_cached_function_call_id(step_id)
        logger.debug(f"Step '{step_id}' completed and cached.")
        return result

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
            modal_func (modal.Function): A Modal function: a function decorated with
                ``@app.function()`` or a bound ``@app.cls`` method (``MyCls().method``).
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
        is_cached, cached_result = self._take_cached_step_result(step_id)
        if is_cached:
            return cached_result

        function_call = self._spawn_or_inherit_function_call(
            step_id,
            lambda: modal_func.spawn(*args, **kwargs),
        )
        self.log_event(f"Running step {step_id}")
        return self._await_step(step_id, function_call, timeout=wait_timeout)

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
            modal_func (modal.Function): A Modal function: a function decorated with
                ``@app.function()`` or a bound ``@app.cls`` method (``MyCls().method``).
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
        if self.state.get_cached_step_result(step_id)[0]:
            logger.debug(f"Skipping spawn '{step_id}': found cached result.")
            self.state.clear_cached_function_call_id(step_id)
            return WorkflowManagedFunctionCall(self, None, step_id)

        self.log_event(f"Spawned {step_id}")
        function_call = self._spawn_or_inherit_function_call(
            step_id,
            lambda: modal_func.spawn(*args, **kwargs),
        )
        return WorkflowManagedFunctionCall(self, function_call, step_id)

    @staticmethod
    def _poll_once(function_call: Any) -> tuple[bool, Any] | None:
        """Polls a call once. Returns (succeeded, result or exception), or None if unfinished."""
        try:
            return True, function_call.get(timeout=0)
        except TimeoutError:
            return None
        except Exception as exc:
            return False, exc

    def _await_outcome(
        self,
        pool: ThreadPoolExecutor,
        pending_calls: dict[int, Any],
        window: list[int],
        buffer: _OutcomeBuffer,
        wait_timeout: float | None,
    ) -> Any:
        """Returns the result of ``window[0]``, raising what its call raised.

        Every round also polls the other calls in *window* that are due, concurrently, and
        keeps what has finished in *buffer*, so later items are ready by the time their turn
        comes. Nothing is surfaced before its turn. The restart check runs before each round.
        A failure fetched ahead of its turn is dropped and fetched again at its turn, so a
        transient polling error on a lookahead poll never becomes the call's outcome.
        """
        head = window[0]
        if head in buffer.outcomes and not buffer.outcomes[head][0]:
            del buffer.outcomes[head]
        deadline = None if wait_timeout is None else time.time() + wait_timeout
        while head not in buffer.outcomes:
            self.restart_if_needed()
            polls = [
                (index, pool.submit(self._poll_once, pending_calls[index]))
                for index in buffer.due(window)
            ]
            for index, poll in polls:
                buffer.record(index, poll.result())
            if head in buffer.outcomes:
                break
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Function call did not complete before timeout.")
            buffer.sleep_until_next_poll(window)

        succeeded, value = buffer.outcomes.pop(head)
        if not succeeded:
            raise value
        return value

    def _run_batch(
        self,
        step_id: str,
        count: int,
        spawn_for_index: Callable[[int], Any],
        wait_timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Shared batch execution logic for map() and parallel(); see map() for the caveats.

        Calls are spawned from a thread pool, so not in strict index order. Results are
        collected in index order, so a later index that has already finished remotely is not
        cached until an earlier, still-running index is resolved, although finished calls are
        fetched ahead concurrently.

        Results and failures are committed once after the last one and only then are their
        call ids cleared, so a crash before that leaves the ids for a continuation to fetch
        the results from Modal again. A graceful restart commits before handing off and
        leaves the ids for the continuation.
        """
        # Sub-step ids are otherwise first validated when their results are cached, after
        # every call has run. The last index gives the longest id.
        if count:
            validate_step_id(f"{step_id}_{count - 1}")
        results, pending_calls, to_spawn, leftover_step_ids, only_batch_ids = self._plan_batch(
            step_id, count
        )

        pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS)
        try:
            pending_calls = self._spawn_batch(pool, pending_calls, to_spawn, spawn_for_index)
            collected_step_ids = self._collect_batch(
                pool,
                step_id,
                results,
                pending_calls,
                spawn_for_index,
                wait_timeout,
                return_exceptions,
            )
            self._commit_batch(pool, leftover_step_ids + collected_step_ids, only_batch_ids)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        return results

    def _plan_batch(
        self, step_id: str, count: int
    ) -> tuple[list[Any], dict[int, Any], list[tuple[int, str]], list[str], bool]:
        """Splits a batch into cached results, inherited calls and indices that need spawning.

        Returns (results, pending_calls, to_spawn, leftover_step_ids, only_batch_ids). The
        leftover ids belong to cached results whose call ids a crashed earlier run never
        cleared. *only_batch_ids* is True when every cached call id belongs to this batch.
        """
        results: list[Any] = [None] * count
        pending_calls: dict[int, Any] = {}
        to_spawn: list[tuple[int, str]] = []
        leftover_step_ids: list[str] = []

        completed_steps = self.state.completed_steps
        cached_call_ids = self.state.cached_function_call_ids()
        batch_step_ids = {f"{step_id}_{index}" for index in range(count)}
        only_batch_ids = cached_call_ids.keys() <= batch_step_ids

        for index in range(count):
            sub_step_id = f"{step_id}_{index}"
            is_cached, cached_result = (
                self.state.read_step_result(sub_step_id)
                if sub_step_id in completed_steps
                else (False, None)
            )
            if is_cached:
                logger.debug(f"Skipping step '{sub_step_id}': found cached result.")
                results[index] = cached_result
                if sub_step_id in cached_call_ids:
                    leftover_step_ids.append(sub_step_id)
                continue

            inherited_call = self._inherit_function_call(
                sub_step_id, cached_call_ids.get(sub_step_id)
            )
            if inherited_call is not None:
                pending_calls[index] = inherited_call
            else:
                to_spawn.append((index, sub_step_id))
        return results, pending_calls, to_spawn, leftover_step_ids, only_batch_ids

    def _spawn_batch(
        self,
        pool: ThreadPoolExecutor,
        pending_calls: dict[int, Any],
        to_spawn: list[tuple[int, str]],
        spawn_for_index: Callable[[int], Any],
    ) -> dict[int, Any]:
        """Spawns the missing calls on *pool* and returns all calls sorted by index.

        Each call id is cached right after its spawn, and at most twice the pool's size are
        in flight at once. The restart check runs on this thread between submissions. Before
        a restart, every submitted spawn is waited for, so each call that exists has its id
        persisted by the time the continuation is spawned.
        """
        in_flight: dict[Future[Any], int] = {}

        def settle(return_when: str) -> None:
            done, _ = wait(in_flight, return_when=return_when)
            for future in done:
                pending_calls[in_flight.pop(future)] = future.result()

        for index, sub_step_id in to_spawn:
            if len(in_flight) >= 2 * MAX_PARALLEL_REQUESTS:
                settle(FIRST_COMPLETED)
            if self.should_restart:
                settle(ALL_COMPLETED)
                self.restart()
            future = pool.submit(
                self._spawn_and_cache_call_id,
                sub_step_id,
                lambda i=index: spawn_for_index(i),
                CALL_ID_WRITE_TIMEOUT_SECONDS,
            )
            in_flight[future] = index
        settle(ALL_COMPLETED)
        return dict(sorted(pending_calls.items()))

    def _collect_batch(
        self,
        pool: ThreadPoolExecutor,
        step_id: str,
        results: list[Any],
        pending_calls: dict[int, Any],
        spawn_for_index: Callable[[int], Any],
        wait_timeout: float | None,
        return_exceptions: bool,
    ) -> list[str]:
        """Fills *results* in index order and returns the step ids cached without a commit."""
        buffer = _OutcomeBuffer()
        indices = list(pending_calls)
        collected_step_ids: list[str] = []
        respawned: set[int] = set()
        for position, index in enumerate(indices):
            sub_step_id = f"{step_id}_{index}"
            try:
                result = self._await_item(
                    pool,
                    sub_step_id,
                    index,
                    indices[position : position + POLL_LOOKAHEAD],
                    pending_calls,
                    buffer,
                    spawn_for_index,
                    wait_timeout,
                    respawned,
                )
            except WorkflowRestarted:
                raise
            except Exception as exc:
                self._cancel_function_call(pending_calls[index], sub_step_id)
                if not return_exceptions:
                    self.state.clear_cached_function_call_id(sub_step_id)
                    raise
                failure = failure_from_exception(exc)
                if self._cache_failure(sub_step_id, failure):
                    collected_step_ids.append(sub_step_id)
                results[index] = failure
                continue

            self.state.cache_step_result(sub_step_id, result, commit=False)
            collected_step_ids.append(sub_step_id)
            logger.debug(f"Step '{sub_step_id}' completed and cached.")
            results[index] = result
        return collected_step_ids

    def _await_item(
        self,
        pool: ThreadPoolExecutor,
        sub_step_id: str,
        index: int,
        window: list[int],
        pending_calls: dict[int, Any],
        buffer: _OutcomeBuffer,
        spawn_for_index: Callable[[int], Any],
        wait_timeout: float | None,
        respawned: set[int],
    ) -> Any:
        """Awaits one item, respawning its call once if Modal no longer has it."""
        while True:
            try:
                return self._await_outcome(pool, pending_calls, window, buffer, wait_timeout)
            except (modal.exception.OutputExpiredError, modal.exception.NotFoundError):
                if index in respawned:
                    raise
                respawned.add(index)
                logger.warning(f"Call of step '{sub_step_id}' is gone; respawning.")
                pending_calls[index] = self._spawn_and_cache_call_id(
                    sub_step_id, lambda i=index: spawn_for_index(i)
                )

    def _cache_failure(self, sub_step_id: str, failure: Any) -> bool:
        """Caches a failure without a commit; returns False if it could not be cached."""
        try:
            self.state.cache_step_result(sub_step_id, failure, commit=False)
        except Exception as cache_exc:
            logger.warning(
                f"Failed to cache failure for step '{sub_step_id}'; it will be "
                f"retried on resume ({cache_exc!r}).",
            )
            self.state.clear_cached_function_call_id(sub_step_id)
            return False
        logger.debug(f"Step '{sub_step_id}' failed; caching failure ({failure!r}).")
        return True

    def _commit_batch(
        self, pool: ThreadPoolExecutor, step_ids_to_clear: list[str], only_batch_ids: bool
    ) -> None:
        """Commits the batch's results once, then clears the call ids of finished steps.

        When every cached id belongs to this batch, all of them are finished once the commit
        is through, so the whole Dict is cleared in one bounded request. Ids left behind by a
        timeout only cost later batches time; ``finish()`` deletes them.
        """
        self.state.commit_pending()
        self._unsaved_call_ids.clear()
        if not step_ids_to_clear:
            return
        if only_batch_ids:
            cleared = pool.submit(
                self.state.clear_all_cached_function_call_ids, CALL_ID_CLEAR_TIMEOUT_SECONDS
            )
            if not cleared.result():
                logger.warning("Clearing the batch's call ids timed out; leftovers are harmless.")
        else:
            list(pool.map(self.state.clear_cached_function_call_id, step_ids_to_clear))

    def map(
        self,
        modal_func: modal.Function,
        items: list[Any],
        step_id: str,
        *args: Any,
        wait_timeout: float | None = None,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """Runs one workflow step over many items in parallel.

        Results are cached per item (``{step_id}_{index}``), and each item also
        tracks a function-call id so in-flight work can be inherited after restart.

        Because the cache key is the item's *position*, ``items`` must come back in the
        same order if the workflow restarts: the new orchestrator re-runs the workflow
        body from the top, rebuilding the list. A list whose order can vary between runs
        - a database query without ``ORDER BY``, a directory or bucket listing, a ``set``
        - will attribute a cached result to the wrong item, silently. Sort the list, or
        derive it from something whose order you control.

        A graceful restart re-attaches every spawned call, but a hard crash of the orchestrator
        (e.g. SIGKILL or an out-of-memory kill) while items are being spawned can leave up to
        16 calls per crash (the spawn thread pool's size) spawned without their call id saved,
        plus any whose id write timed out (after 30 s; such ids are written again before a
        graceful restart). A resumed run spawns those items again, so they execute twice;
        keep step functions idempotent.

        A call re-attached after a restart whose output has expired (Modal keeps outputs
        for 7 days) or which Modal no longer knows is spawned again, whatever
        ``return_exceptions`` is. A call that is gone is respawned once; a second failure
        of the same kind is an ordinary step failure.

        A built-in ``TimeoutError`` raised by the step function itself (including asyncio,
        ``concurrent.futures`` and socket timeouts) looks the same as "not finished yet", so
        that item is polled until ``wait_timeout`` expires, or forever without one. Catch it
        in the step function and raise a different exception type instead.

        With ``return_exceptions=True``, anything caught in the wait loop is cached as a
        failure and permanently skipped on resume - not just the item's own exception, but
        also a ``wait_timeout`` expiry or another orchestrator-side polling error. A
        transient failure of that kind won't get a second chance within the same run; it's
        retried only if a later, unrelated run starts fresh under a different state name.

        Args:
            modal_func (modal.Function): A Modal function: a function decorated with
                ``@app.function()`` or a bound ``@app.cls`` method (``MyCls().method``).
            items (list[Any]): List of items to process. Each item is the first positional argument.
            step_id (str): Base identifier. Each item is keyed as {step_id}_{index}.
                Must be unique across all step(), spawn(), map(), and parallel() calls
                within the same workflow run.
            *args (Any): Additional positional arguments forwarded to every call.
            wait_timeout (float | None): Optional max wait in seconds per pending item.
            return_exceptions (bool): When True, an item's exception is cached and a
                plain-dict snapshot of it (``is_workflow_item_failure(value)`` is True)
                is returned in its place instead of aborting the whole map, mirroring
                ``asyncio.gather(..., return_exceptions=True)`` - except the slot
                holds that snapshot rather than the original exception instance,
                since the original may not survive being pickled and cached across
                a restart, and may not be readable by a caller without ``modal_workflows``
                installed. A restart is never treated as an item failure - it always
                propagates regardless of this flag.
            **kwargs (Any): Keyword arguments forwarded to every call.

        Returns:
            list[Any]: Results in the same order as items. When *return_exceptions*
                is True, a failed item's slot holds a plain-dict failure snapshot
                (``exc_type``, ``message``, ``traceback_str`` - see
                ``is_workflow_item_failure()``) rather than a result or the original
                exception instance.

        Raises:
            ValueError: If *step_id* has already been used in this workflow run, or if
                ``{step_id}_{index}`` is not a valid identifier.
        """
        _validate_workflow_step(modal_func, "map")
        self._register_step_id(step_id)
        self.log_event(f"Launched map {step_id} ({len(items)} items)")
        return self._run_batch(
            step_id=step_id,
            count=len(items),
            spawn_for_index=lambda i: modal_func.spawn(items[i], *args, **kwargs),
            wait_timeout=wait_timeout,
            return_exceptions=return_exceptions,
        )

    def parallel(
        self,
        steps: list[tuple[modal.Function, ...]],
        step_id: str,
        wait_timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Runs multiple workflow steps concurrently, caching each individually.

        Each step is a tuple of (func, *args). All uncached steps are spawned
        concurrently, then their results are collected. As with ``map()``, a hard crash
        of the orchestrator while spawning can make up to 16 steps per crash execute twice.

        Args:
            steps (list[tuple[modal.Function, ...]]): List of tuples, each containing
                (modal_func, arg1, arg2, ...), where each ``modal_func`` is
                a Modal function (``@app.function()`` or a bound ``@app.cls`` method).
            step_id (str): Base identifier. Each step is keyed as {step_id}_{index}.
                Must be unique across all step(), spawn(), map(), and parallel() calls
                within the same workflow run.
            wait_timeout (float | None): Optional max wait in seconds per pending step.
            return_exceptions (bool): Same as ``map()``'s parameter of the same name -
                see its docstring for the full behavior and caveats.

        Returns:
            list[Any]: Results in the same order as steps.

        Raises:
            ValueError: If *step_id* has already been used in this workflow run, or if
                ``{step_id}_{index}`` is not a valid identifier.
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
            return_exceptions=return_exceptions,
        )

    @property
    def should_restart(self) -> bool:
        """True once elapsed time exceeds the workflow's timeout threshold."""
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout

    def restart(self) -> None:
        """Spawns a new workflow instance and raises WorkflowRestarted.

        Raises:
            WorkflowRestarted: Always raised to unwind the stack and hand off to the wrapper.
        """
        logger.warning(f"Timeout threshold reached. Respawning workflow '{self.func_name}'...")
        self.state.commit_pending()
        for step_id, call_id in list(self._unsaved_call_ids.items()):
            self.state.cache_function_call_id(step_id, call_id)
        function_call = self.func.spawn(*self.args, **self.kwargs)
        raise WorkflowRestarted(function_call)

    def restart_if_needed(self) -> None:
        """Triggers a graceful restart if the timeout threshold has been exceeded."""
        if self.should_restart:
            self.restart()

    def finish(self) -> None:
        """Deletes all persisted state, signaling that the workflow completed successfully."""
        logger.debug(f"Workflow '{self.func_name}' finished successfully. Deleting state.")
        self.state.delete()

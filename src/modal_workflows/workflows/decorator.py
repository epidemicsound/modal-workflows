# mypy: ignore-errors
from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import modal

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.alerts import send_exception_alert
from modal_workflows.workflows.context import (
    WORKFLOW_STEP_MARKER,
    WorkflowContext,
    WorkflowRestarted,
)
from modal_workflows.workflows.identifiers import validate_state_name

logger = logging.getLogger(__name__)
parameters = ParamSpec("parameters")
return_type = TypeVar("return_type")

MAX_SELF_MANAGED_TIMEOUT_SECONDS = 23 * 3600

_STATE_NAME_KWARG = "_workflow_state_name"


def _resolve_state_name(app: modal.App, state_name: str | None, kwargs: dict) -> str:
    """Pops and returns the stable state name for this invocation.

    Priority: user-provided ``state_name`` → propagated restart kwarg → derived from
    ``app.app_id`` and ``modal.current_function_call_id()``. The kwarg is always
    stripped so the user function never sees it, and is injected into restart spawn
    kwargs so the new orchestrator continues with the same state.

    The resolved name becomes a directory name on the state and artifacts volumes,
    so it is validated as a single path component.

    Raises:
        ValueError: If the resolved state name is not a valid path identifier.
    """
    propagated: str | None = kwargs.pop(_STATE_NAME_KWARG, None)
    if state_name:
        return validate_state_name(state_name)
    if propagated is not None:
        return validate_state_name(propagated)
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise RuntimeError(
            "Cannot resolve a state name: workflows must be invoked inside a Modal "
            f"function call, have an explicit `state_name`, or have "
            f"{_STATE_NAME_KWARG!r} propagated via kwargs."
        )
    return validate_state_name(f"{app.app_id}_{function_call_id}")


@dataclasses.dataclass
class WorkflowRestartedResult:
    """Returned when a workflow restarts due to a self-managed timeout.

    Attributes:
        function_name: Name of the workflow function that was restarted.
        call_id: Object ID of the newly spawned Modal FunctionCall, if available.
    """

    function_name: str
    call_id: str | None

    def to_modal_function(self) -> modal.FunctionCall:
        return modal.FunctionCall.from_id(self.call_id)


def workflow_function(
    app: modal.App,
    **function_kwargs: Any,
) -> Callable[[Callable[parameters, return_type]], modal.Function]:
    """Decorator that registers a Modal function with the workflow artifacts volume mounted.

    Required for functions that will be called as workflow steps via
    ``ctx.step()``, ``ctx.spawn()``, ``ctx.map()``, or ``ctx.parallel()``.
    Those methods validate that the function carries the workflow-step marker
    set by this decorator, raising ``TypeError`` if it is missing.

    The artifacts volume is automatically available at
    ``/mnt/workflow-artifacts`` without the caller having to configure it.

    Args:
        app (modal.App): The Modal app the function will be registered on.
        **function_kwargs (Any): Keyword arguments forwarded to ``app.function()``
            (e.g. ``image``, ``gpu``, ``secrets``).

    Returns:
        Callable[parameters, return_type]: A decorator that wraps the target function
            as a ``modal.Function``.
    """
    updated_kwargs = workflow_volumes.with_artifacts_volume(function_kwargs)
    app_decorator = app.function(**updated_kwargs)

    def decorator(function: Callable[parameters, return_type]) -> modal.Function:
        modal_function = app_decorator(function)
        setattr(modal_function, WORKFLOW_STEP_MARKER, True)
        return modal_function

    return decorator


def workflow(
    app: modal.App,
    self_managed_timeout: float = MAX_SELF_MANAGED_TIMEOUT_SECONDS,
    alert_on_error: bool = True,
    state_name: str | None = None,
    **function_kwargs: Any,
) -> Callable[..., Any]:
    """Decorator that turns a function into a resumable, stateful Modal workflow.

    The decorated function receives a WorkflowContext as its first argument.
    Step results are cached in a modal.Dict so the workflow can resume after
    a timeout or crash without re-running completed steps.

    Args:
        app (modal.App): The Modal app the workflow function will be registered on.
        self_managed_timeout (float): Maximum wall-clock seconds before this decorator
            triggers a graceful restart. This is separate from Modal-managed function
            timeouts. Modal enforces a hard 24-hour cap on function wall-clock time;
            this parameter defaults to 23 hours to leave headroom for the restart handoff.
            Must be less than or equal to 23 hours.
        alert_on_error (bool): When True, sends a Slack alert if the workflow raises
            an unhandled exception. Requires the ``SLACK_WEBHOOK_URL`` environment
            variable to be set (typically via a Modal Secret) and ``slack_sdk`` to be
            installed in the container image. Defaults to True.
        state_name (str | None): Exact state key used to persist step results.
            When provided, the key is used as-is, letting you target a specific existing
            state — useful for resuming a crashed run. When omitted, the key is derived
            as ``{app.app_id}_{function_call_id}``, giving each distinct invocation its
            own isolated state while retries and preemptions of the same call share state
            because they share the same function call ID.
        **function_kwargs (Any): Additional keyword arguments forwarded to app.function()
            (e.g. image, retries, secrets). ``single_use_containers`` is always forced to
            ``True`` so containers are never reused across restarts.

    Returns:
        Callable[..., Any]: A decorator that wraps the target function as a modal.Function.
    """
    if self_managed_timeout > MAX_SELF_MANAGED_TIMEOUT_SECONDS:
        raise ValueError("self_managed_timeout cannot exceed 23 hours")

    def decorator(fn: Callable[..., Any]) -> modal.Function:
        workflow_function_kwargs = workflow_volumes.with_workflow_volumes(function_kwargs)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_state_name = _resolve_state_name(app, state_name, kwargs)
            kwargs_with_state_name = {**kwargs, _STATE_NAME_KWARG: effective_state_name}

            ctx = WorkflowContext(
                app=app,
                func=modal_func,
                func_name=fn.__name__,
                state_name=effective_state_name,
                args=args,
                kwargs=kwargs_with_state_name,
                timeout=self_managed_timeout,
            )
            try:
                return fn(ctx, *args, **kwargs)
            except WorkflowRestarted as e:
                call_id = getattr(e.function_call, "object_id", None)
                logger.info(
                    "Workflow '%s' restarted in a new orchestrator (call id: '%s'). "
                    "This orchestrator is now exiting.",
                    fn.__name__,
                    call_id,
                )
                return WorkflowRestartedResult(
                    function_name=fn.__name__,
                    call_id=call_id,
                )
            except Exception as e:
                if alert_on_error:
                    send_exception_alert(fn, e, app_id=app.app_id)
                raise

        workflow_function_kwargs["single_use_containers"] = True
        if "timeout" in workflow_function_kwargs:
            raise ValueError(
                "Custom `timeout` is not allowed in @workflow. "
                "Use the `self_managed_timeout` parameter instead."
            )
        workflow_function_kwargs["timeout"] = 24 * 3600

        modal_func = app.function(**workflow_function_kwargs)(wrapper)

        return modal_func

    return decorator

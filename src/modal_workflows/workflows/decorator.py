# mypy: ignore-errors
from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import modal

from modal_workflows.workflows._common import (
    MAX_SELF_MANAGED_TIMEOUT_SECONDS,
    build_function_kwargs,
    resolve_state_name,
    restart_kwargs,
    validate_self_managed_timeout,
)
from modal_workflows.workflows.alerts import send_exception_alert
from modal_workflows.workflows.context import WorkflowContext, WorkflowRestarted

logger = logging.getLogger(__name__)
parameters = ParamSpec("parameters")
return_type = TypeVar("return_type")


@dataclasses.dataclass
class WorkflowRestartedResult:
    """Returned by ``@training`` when a training run restarts due to a self-managed timeout.

    Attributes:
        function_name: Name of the training function that was restarted.
        call_id: Object ID of the newly spawned Modal FunctionCall, if available.
    """

    function_name: str
    call_id: str | None

    def to_modal_function(self) -> modal.FunctionCall:
        return modal.FunctionCall.from_id(self.call_id)


def workflow(
    app: modal.App,
    self_managed_timeout: float = MAX_SELF_MANAGED_TIMEOUT_SECONDS,
    alert_on_error: bool = True,
    state_name: str | None = None,
    **function_kwargs: Any,
) -> Callable[..., Any]:
    """Decorator that turns a function into a resumable, stateful Modal workflow.

    The decorated function receives a WorkflowContext as its first argument.
    Step results are cached on the workflow state volume so the workflow can resume
    after a timeout or crash without re-running completed steps.

    If ``self_managed_timeout`` is reached, the wrapped function returns
    ``{"_workflow_restarted": True, "call_id": "<id>"}`` instead of its normal
    result - a plain dict, not a custom class, so any caller can read it without
    ``modal_workflows`` installed. A continuation is already running independently on
    the same ``state_name`` by the time this returns; attach to it with
    ``modal.FunctionCall.from_id(call_id).get()``. Calling the workflow function
    again yourself does not resume it - see the ``state_name`` note below.

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
            own isolated state, while retries and preemptions of the same call share
            state because they share the same function call ID. Re-invoking the workflow
            yourself (rather than attaching to a restart's continuation) gets a fresh
            call ID and therefore a new, unrelated state - unless you pass this same
            explicit ``state_name``, in which case you race the continuation on it
            instead, since ``WorkflowState`` has no locking.
        **function_kwargs (Any): Additional keyword arguments forwarded to app.function()
            (e.g. image, secrets, retries). ``single_use_containers`` is always forced to
            ``True`` so containers are never reused across restarts. ``retries`` is safe
            to combine with ``self_managed_timeout``: a restart returns a value rather
            than raising, so Modal never treats it as a failure to retry.

    Returns:
        Callable[..., Any]: A decorator that wraps the target function as a modal.Function.
    """
    validate_self_managed_timeout(self_managed_timeout)

    def decorator(fn: Callable[..., Any]) -> modal.Function:
        modal_kwargs = build_function_kwargs(function_kwargs, "workflow")

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_state_name = resolve_state_name(app, state_name, kwargs)
            kwargs_with_state_name = restart_kwargs(kwargs, effective_state_name)

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
                logger.warning(
                    "Workflow '%s' (state %r) hit its self-managed timeout and restarted; "
                    "continuation running as call %r. This is not a failure.",
                    fn.__name__,
                    effective_state_name,
                    call_id,
                )
                return {"_workflow_restarted": True, "call_id": call_id}
            except Exception as e:
                if alert_on_error:
                    send_exception_alert(fn, e, app_id=app.app_id)
                raise

        modal_func = app.function(**modal_kwargs)(wrapper)

        return modal_func

    return decorator

# mypy: ignore-errors
"""Plumbing shared by the ``@workflow`` and ``@training`` decorators."""

from __future__ import annotations

from typing import Any

import modal

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.identifiers import validate_state_name

MAX_SELF_MANAGED_TIMEOUT_SECONDS = 23 * 3600
MODAL_MAX_TIMEOUT_SECONDS = 24 * 3600

STATE_NAME_KWARG = "_workflow_state_name"


def validate_self_managed_timeout(self_managed_timeout: float) -> None:
    """Raises ValueError if *self_managed_timeout* leaves no headroom for the restart handoff."""
    if self_managed_timeout > MAX_SELF_MANAGED_TIMEOUT_SECONDS:
        raise ValueError("self_managed_timeout cannot exceed 23 hours")


def build_function_kwargs(function_kwargs: dict[str, Any], decorator_name: str) -> dict[str, Any]:
    """Returns ``app.function()`` kwargs with the state volume, single-use containers and timeout.

    Raises:
        ValueError: If *function_kwargs* sets a custom ``timeout``.
    """
    merged_kwargs = workflow_volumes.with_state_volume(function_kwargs)
    merged_kwargs["single_use_containers"] = True
    if "timeout" in merged_kwargs:
        raise ValueError(
            f"Custom `timeout` is not allowed in @{decorator_name}. "
            "Use the `self_managed_timeout` parameter instead."
        )
    merged_kwargs["timeout"] = MODAL_MAX_TIMEOUT_SECONDS
    return merged_kwargs


def resolve_state_name(app: modal.App, state_name: str | None, kwargs: dict) -> str:
    """Pops and returns the stable state name for this invocation.

    Priority: user-provided ``state_name`` → propagated restart kwarg → derived from
    ``app.app_id`` and ``modal.current_function_call_id()``. The kwarg is always
    stripped so the user function never sees it.

    The resolved name becomes a directory name on the state volume, so it is
    validated as a single path component.

    Raises:
        ValueError: If the resolved state name is not a valid path identifier.
    """
    propagated: str | None = kwargs.pop(STATE_NAME_KWARG, None)
    if state_name:
        return validate_state_name(state_name)
    if propagated is not None:
        return validate_state_name(propagated)
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise RuntimeError(
            "Cannot resolve a state name: workflows must be invoked inside a Modal "
            f"function call, have an explicit `state_name`, or have "
            f"{STATE_NAME_KWARG!r} propagated via kwargs."
        )
    return validate_state_name(f"{app.app_id}_{function_call_id}")


def restart_kwargs(kwargs: dict[str, Any], state_name: str) -> dict[str, Any]:
    """Returns *kwargs* plus the state name, so a restart continues with the same state."""
    return {**kwargs, STATE_NAME_KWARG: state_name}

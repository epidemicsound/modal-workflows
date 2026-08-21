# mypy: ignore-errors
"""Modal ``@training`` decorator and the restart-handoff plumbing it builds on."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import modal
import modal.experimental

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.alerts import send_exception_alert
from modal_workflows.workflows.context import WorkflowRestarted
from modal_workflows.workflows.decorator import (
    _STATE_NAME_KWARG,
    MAX_SELF_MANAGED_TIMEOUT_SECONDS,
    WorkflowRestartedResult,
    _resolve_state_name,
)
from modal_workflows.workflows.identifiers import validate_state_name

logger = logging.getLogger(__name__)

_RESTART_METADATA_FILENAME = "training_restart_metadata.json"
_CHECKPOINT_FILENAME = "latest_checkpoint.ckpt"
_STATE_NAME_ENV_VAR = "MODAL_WORKFLOWS_TRAINING_STATE_NAME"
TRAINING_TIMEOUT_SECONDS_ENV_VAR = "MODAL_WORKFLOWS_TRAINING_TIMEOUT_SECONDS"


@dataclasses.dataclass
class _TrainingRunPaths:
    """Filesystem layout for a single training run on the workflow state volume."""

    state_name: str
    results_root: Path
    checkpoint_path: Path
    restart_metadata_path: Path


def _training_run_paths(state_name: str) -> _TrainingRunPaths:
    """Ensures checkpoint and restart metadata directories exist and returns their paths.

    Raises:
        ValueError: If *state_name* is not a valid path identifier.
    """
    validate_state_name(state_name)
    results_root = workflow_volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH / state_name
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root = results_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    return _TrainingRunPaths(
        state_name=state_name,
        results_root=results_root,
        checkpoint_path=checkpoints_root / _CHECKPOINT_FILENAME,
        restart_metadata_path=results_root / _RESTART_METADATA_FILENAME,
    )


def _training_run_paths_from_env() -> _TrainingRunPaths:
    """Like :func:`_training_run_paths`, reading the state name from the environment.

    Uses ``MODAL_WORKFLOWS_TRAINING_STATE_NAME``, which ``@training`` sets.
    """
    state_name = os.environ.get(_STATE_NAME_ENV_VAR)
    if state_name is None:
        raise RuntimeError(
            f"{_STATE_NAME_ENV_VAR} is not set. " "This function must run under @training."
        )
    return _training_run_paths(state_name)


@dataclasses.dataclass(frozen=True)
class _RestartMetadata:
    """Restart request written by a training run and consumed by the handoff logic.

    Serialized to JSON on the workflow state volume. The dataclass is the single
    source of truth for the on-disk schema.
    """

    reason: str
    requested_at_unix_seconds: float
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    restart_requested: bool = True

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> _RestartMetadata:
        data = json.loads(text)
        return cls(
            reason=data["reason"],
            requested_at_unix_seconds=data["requested_at_unix_seconds"],
            payload=data.get("payload") or {},
            restart_requested=data.get("restart_requested", True),
        )


def _write_restart_metadata_file(
    metadata_path: Path,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Writes JSON restart request consumed by the cluster leader after a graceful training stop."""
    metadata = _RestartMetadata(
        reason=reason,
        requested_at_unix_seconds=time.time(),
        payload=payload or {},
    )
    metadata_path.write_text(metadata.to_json(), encoding="utf-8")


def _read_restart_metadata_file(metadata_path: Path) -> _RestartMetadata | None:
    """Returns parsed restart metadata, or ``None`` if the file does not exist."""
    if not metadata_path.exists():
        return None
    return _RestartMetadata.from_json(metadata_path.read_text(encoding="utf-8"))


def _clear_restart_metadata_file(metadata_path: Path) -> None:
    """Removes restart metadata after it has been acted on so it is not applied twice."""
    if metadata_path.exists():
        metadata_path.unlink()


def _restart_from_metadata_path(
    metadata_path: Path,
    modal_func: modal.Function,
    *args: Any,
    **kwargs: Any,
) -> modal.FunctionCall | None:
    """Clears the metadata file and spawns ``modal_func`` if a restart was requested."""
    metadata = _read_restart_metadata_file(metadata_path)
    if metadata is None or not metadata.restart_requested:
        return None
    _clear_restart_metadata_file(metadata_path)
    return modal_func.spawn(*args, **kwargs)


def _handoff_result(
    function_name: str,
    function_call: modal.FunctionCall,
) -> WorkflowRestartedResult:
    """Logs the handoff, then returns a ``WorkflowRestartedResult`` without raising."""
    call_id = getattr(function_call, "object_id", None)
    logger.info(
        "Modal training handoff after graceful stop; continuation spawned (call_id=%s). "
        "This container exits successfully.",
        call_id,
    )
    return WorkflowRestartedResult(
        function_name=function_name,
        call_id=call_id,
    )


def _handoff_from_metadata_if_requested(
    function_name: str,
    metadata_path: Path,
    modal_func: modal.Function,
    *args: Any,
    **kwargs: Any,
) -> WorkflowRestartedResult | None:
    """Spawns a continuation when restart metadata is present; otherwise returns ``None``."""
    function_call = _restart_from_metadata_path(
        metadata_path,
        modal_func,
        *args,
        **kwargs,
    )
    if function_call is None:
        return None
    return _handoff_result(function_name, function_call)


def _is_cluster_leader(cluster_size: int | None) -> bool:
    """Whether this process may hand off a restart (true unless multi-node non-rank-0)."""
    if cluster_size is None or cluster_size <= 1:
        return True
    return modal.experimental.get_cluster_info().rank == 0


def training(
    app: modal.App,
    self_managed_timeout: float = MAX_SELF_MANAGED_TIMEOUT_SECONDS,
    alert_on_error: bool = True,
    state_name: str | None = None,
    cluster_size: int | None = None,
    rdma: bool = False,
    **function_kwargs: Any,
) -> Any:
    """Registers a Modal function with workflow volumes and training env for checkpoint restarts.

    Sets ``MODAL_WORKFLOWS_TRAINING_STATE_NAME`` and ``MODAL_WORKFLOWS_TRAINING_TIMEOUT_SECONDS`` so
    :class:`~modal_workflows.training.callbacks.TrainingCheckpointRestartCallback` can locate
    the correct checkpoint directory.
    After the wrapped function returns normally, restart metadata written by the callback
    is consumed: the continuation is spawned and
    :class:`~modal_workflows.workflows.decorator.WorkflowRestartedResult` is returned without
    raising, so Modal records a successful handoff. Uncaught exceptions skip the handoff
    entirely.

    Args:
        app (modal.App): The Modal app the training function will be registered on.
        self_managed_timeout (float): Seconds before
            :class:`~modal_workflows.training.callbacks.TrainingCheckpointRestartCallback`
            triggers a checkpoint and graceful restart. Must be <= 23 hours. Defaults to 23 hours.
        alert_on_error (bool): When True, sends a Slack alert on uncaught exceptions (cluster
            leader only). Defaults to True.
        state_name (str | None): Exact checkpoint directory name used to persist training
            state. When provided, the name is used as-is, letting you target a specific
            existing run — useful for resuming a crashed job. When omitted, the name is
            derived as ``{app.app_id}_{function_call_id}``, giving each distinct
            invocation its own isolated directory while Modal retries and preemptions of
            the same call share the directory because they share the function call ID.
        cluster_size (int | None): Number of nodes for multi-node training via
            ``modal.experimental.clustered``. ``None`` or ``1`` for single-node.
            Values > 1 require a ``gpu`` argument. Defaults to None.
        rdma (bool): Whether to enable RDMA networking for multi-node clusters.
            Ignored when ``cluster_size <= 1``. Defaults to False.
        **function_kwargs (Any): Additional keyword arguments forwarded to ``app.function()``
            (e.g. ``image``, ``gpu``, ``secrets``).
    """

    if self_managed_timeout > MAX_SELF_MANAGED_TIMEOUT_SECONDS:
        raise ValueError("self_managed_timeout cannot exceed 23 hours")
    if cluster_size is not None and cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if cluster_size is not None and cluster_size > 1 and "gpu" not in function_kwargs:
        raise ValueError("cluster_size > 1 requires a `gpu` function argument")

    def decorator(fn: Any) -> modal.Function:
        """Registers the callable as a Modal function with workflow volumes and timeouts."""
        merged_kwargs = workflow_volumes.with_workflow_volumes(function_kwargs)
        merged_kwargs["single_use_containers"] = True
        if "timeout" in merged_kwargs:
            raise ValueError(
                "Custom `timeout` is not allowed in @training. "
                "Use the `self_managed_timeout` parameter instead."
            )
        merged_kwargs["timeout"] = 24 * 3600

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Runs training, then on the leader applies the restart handoff."""
            resolved_state_name = _resolve_state_name(app, state_name, kwargs)
            kwargs_with_state_name = {**kwargs, _STATE_NAME_KWARG: resolved_state_name}

            os.environ[_STATE_NAME_ENV_VAR] = resolved_state_name
            os.environ[TRAINING_TIMEOUT_SECONDS_ENV_VAR] = str(self_managed_timeout)
            paths = _training_run_paths(resolved_state_name)

            if not _is_cluster_leader(cluster_size):
                return fn(*args, **kwargs)

            try:
                result = fn(*args, **kwargs)
            except WorkflowRestarted as restarted:
                return _handoff_result(fn.__name__, restarted.function_call)
            except Exception as error:
                if alert_on_error:
                    send_exception_alert(fn, error, app_id=app.app_id)
                raise error

            handoff = _handoff_from_metadata_if_requested(
                fn.__name__,
                paths.restart_metadata_path,
                modal_func,
                *args,
                **kwargs_with_state_name,
            )
            return handoff if handoff is not None else result

        decorated_wrapper = wrapper
        if cluster_size is not None and cluster_size > 1:
            decorated_wrapper = modal.experimental.clustered(size=cluster_size, rdma=rdma)(
                decorated_wrapper
            )

        modal_func = app.function(**merged_kwargs)(decorated_wrapper)
        return modal_func

    return decorator

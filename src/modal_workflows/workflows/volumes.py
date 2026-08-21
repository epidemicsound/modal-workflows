# mypy: ignore-errors

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

WORKFLOW_STATE_VOLUME_NAME = "workflow-state"
WORKFLOW_STATE_VOLUME_MOUNT_PATH = Path("/mnt/workflow-state")
WORKFLOW_ARTIFACTS_VOLUME_NAME = "workflow-artifacts"
WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH = Path("/mnt/workflow-artifacts")
WORKFLOW_ARTIFACTS_SHARED_DIRECTORY = "shared"


def create_or_get_volume(volume_name: str) -> Any:
    return modal.Volume.from_name(volume_name, create_if_missing=True, version=2)


def with_artifacts_volume(function_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Adds the workflow artifacts volume to Modal function kwargs if not already present."""
    updated_function_kwargs = dict(function_kwargs)
    configured_volumes = dict(updated_function_kwargs.get("volumes", {}))

    if str(WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH) not in configured_volumes:
        configured_volumes[str(WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH)] = create_or_get_volume(
            WORKFLOW_ARTIFACTS_VOLUME_NAME
        )

    updated_function_kwargs["volumes"] = configured_volumes
    return updated_function_kwargs


def with_workflow_volumes(function_kwargs: dict[str, Any]) -> dict[str, Any]:
    updated_function_kwargs = dict(function_kwargs)
    configured_volumes = dict(updated_function_kwargs.get("volumes", {}))

    if str(WORKFLOW_STATE_VOLUME_MOUNT_PATH) not in configured_volumes:
        configured_volumes[str(WORKFLOW_STATE_VOLUME_MOUNT_PATH)] = create_or_get_volume(
            WORKFLOW_STATE_VOLUME_NAME
        )

    if str(WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH) not in configured_volumes:
        configured_volumes[str(WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH)] = create_or_get_volume(
            WORKFLOW_ARTIFACTS_VOLUME_NAME
        )

    updated_function_kwargs["volumes"] = configured_volumes

    return updated_function_kwargs

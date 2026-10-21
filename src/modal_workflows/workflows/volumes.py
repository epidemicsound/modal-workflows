# mypy: ignore-errors

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

WORKFLOW_STATE_VOLUME_NAME = "workflow-state"
WORKFLOW_STATE_VOLUME_MOUNT_PATH = Path("/mnt/workflow-state")


def create_or_get_volume(volume_name: str) -> Any:
    return modal.Volume.from_name(volume_name, create_if_missing=True, version=2)


def with_state_volume(function_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Returns *function_kwargs* with the workflow state volume mounted, unless already mounted."""
    volumes = dict(function_kwargs.get("volumes", {}))
    mount_path = str(WORKFLOW_STATE_VOLUME_MOUNT_PATH)
    if mount_path not in volumes:
        volumes[mount_path] = create_or_get_volume(WORKFLOW_STATE_VOLUME_NAME)
    return {**function_kwargs, "volumes": volumes}

# mypy: ignore-errors
"""Integration test: WorkflowState round-trip against a real modal.Dict.

Run:  uv run python -m tests.integration.test_state_roundtrip
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import modal

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.state import WorkflowState

from tests.integration.utilities import build_workflows_image  # isort: skip

integration_image = build_workflows_image()


def _exercise_state_roundtrip(
    run_name: str,
    state_volume: modal.Volume,
) -> str:
    with patch.object(workflow_volumes, "create_or_get_volume", return_value=state_volume):
        state = WorkflowState(run_name)

    try:
        assert state.completed_steps == set()

        state.cache_step_result("step_a", {"loss": 0.01})
        is_cached, result = state.get_cached_step_result("step_a")
        assert is_cached is True
        assert result == {"loss": 0.01}

        state.cache_step_result("step_b", 42)
        assert state.completed_steps == {"step_a", "step_b"}

        state.clear()
        assert state.completed_steps == set()
        is_cached, _ = state.get_cached_step_result("step_a")
        assert is_cached is False

        return "ok"
    finally:
        state.clear()


def main() -> None:
    run_name = f"test-state-{uuid.uuid4().hex[:8]}"
    app = modal.App(f"test-state-roundtrip-{uuid.uuid4().hex[:8]}")

    with modal.Volume.ephemeral() as state_volume:
        function_kwargs = workflow_volumes.with_workflow_volumes(
            {
                "image": integration_image,
                "volumes": {
                    str(workflow_volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH): state_volume,
                },
            }
        )
        exercise_function = app.function(**function_kwargs)(_exercise_state_roundtrip)
        with app.run():
            result = exercise_function.remote(run_name, state_volume)

        assert result == "ok"

    print("PASS: state round-trip")


if __name__ == "__main__":
    main()

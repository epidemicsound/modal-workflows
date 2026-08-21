# mypy: ignore-errors
"""Integration test: WorkflowState and WorkflowContext with ephemeral volumes.

Run:  uv run python -m tests.integration.test_volumes_ephemeral
Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import modal

import modal_workflows.workflows.volumes as workflow_volumes
from modal_workflows.workflows.context import WorkflowContext
from modal_workflows.workflows.state import WorkflowState

from tests.integration.utilities import build_workflows_image  # isort: skip

integration_image = build_workflows_image()


class IntegrationApp:
    def __init__(self, name: str) -> None:
        self.name = name


class IntegrationFunction:
    def spawn(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("This integration test does not exercise restart behavior.")


def _exercise_state_and_context(
    run_name: str,
    state_volume: modal.Volume,
    artifacts_volume: modal.Volume,
) -> str:
    volume_map = {
        workflow_volumes.WORKFLOW_STATE_VOLUME_NAME: state_volume,
        workflow_volumes.WORKFLOW_ARTIFACTS_VOLUME_NAME: artifacts_volume,
    }
    with patch.object(
        workflow_volumes,
        "create_or_get_volume",
        side_effect=lambda name: volume_map[name],
    ):
        state = WorkflowState(run_name)
        state.cache_step_result("train", {"loss": 0.01})

        is_cached, cached_result = state.get_cached_step_result("train")
        assert is_cached is True
        assert cached_result == {"loss": 0.01}

        state_volume.reload()
        state_after_reload = WorkflowState(run_name)
        is_cached_after_reload, cached_result_after_reload = (
            state_after_reload.get_cached_step_result("train")
        )
        assert is_cached_after_reload is True
        assert cached_result_after_reload == {"loss": 0.01}

        context = WorkflowContext(
            app=IntegrationApp(name=f"integration-{run_name}"),
            func=IntegrationFunction(),
            func_name="integration_workflow",
            state_name=run_name,
            args=(),
            kwargs={},
            timeout=300.0,
        )

    run_scoped_file = context.artifacts_run_path / "artifacts/model.txt"
    run_scoped_file.parent.mkdir(parents=True, exist_ok=True)
    run_scoped_file.write_text("model-ready", encoding="utf-8")
    shared_file = context.artifacts_shared_path / "datasets/index.txt"
    shared_file.parent.mkdir(parents=True, exist_ok=True)
    shared_file.write_text("shared-ready", encoding="utf-8")
    context.commit_artifacts_volume()

    return "ok"


def _read_persisted_artifacts(
    run_name: str,
    state_volume: modal.Volume,
    artifacts_volume: modal.Volume,
) -> dict[str, str]:
    volume_map = {
        workflow_volumes.WORKFLOW_STATE_VOLUME_NAME: state_volume,
        workflow_volumes.WORKFLOW_ARTIFACTS_VOLUME_NAME: artifacts_volume,
    }
    with patch.object(
        workflow_volumes,
        "create_or_get_volume",
        side_effect=lambda name: volume_map[name],
    ):
        state_after_reload = WorkflowState(run_name)
        is_cached, cached_result = state_after_reload.get_cached_step_result("train")
        assert is_cached is True
        assert cached_result == {"loss": 0.01}

        context = WorkflowContext(
            app=IntegrationApp(name=f"integration-{run_name}"),
            func=IntegrationFunction(),
            func_name="integration_workflow",
            state_name=run_name,
            args=(),
            kwargs={},
            timeout=300.0,
        )
    run_scoped_path = context.artifacts_run_path / "artifacts/model.txt"
    shared_path = context.artifacts_shared_path / "datasets/index.txt"

    return {
        "run_scoped_content": run_scoped_path.read_text(encoding="utf-8"),
        "shared_content": shared_path.read_text(encoding="utf-8"),
    }


def main() -> None:
    run_name = f"test-volumes-{uuid.uuid4().hex[:8]}"
    app = modal.App(f"test-volumes-ephemeral-{uuid.uuid4().hex[:8]}")

    with modal.Volume.ephemeral() as state_volume:
        with modal.Volume.ephemeral() as artifacts_volume:
            assert state_volume.listdir("/") == []
            assert artifacts_volume.listdir("/") == []

            function_kwargs = workflow_volumes.with_workflow_volumes(
                {
                    "image": integration_image,
                    "volumes": {
                        str(workflow_volumes.WORKFLOW_STATE_VOLUME_MOUNT_PATH): state_volume,
                        str(
                            workflow_volumes.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH
                        ): artifacts_volume,
                    },
                }
            )
            exercise_function = app.function(**function_kwargs)(_exercise_state_and_context)
            read_function = app.function(**function_kwargs)(_read_persisted_artifacts)
            with app.run():
                exercise_result = exercise_function.remote(run_name, state_volume, artifacts_volume)
                persisted_results = read_function.remote(run_name, state_volume, artifacts_volume)

            assert exercise_result == "ok"
            assert persisted_results["run_scoped_content"] == "model-ready"
            assert persisted_results["shared_content"] == "shared-ready"

    print("PASS: state + context with ephemeral volumes")


if __name__ == "__main__":
    main()

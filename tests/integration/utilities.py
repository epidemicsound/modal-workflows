# mypy: ignore-errors
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONTAINER_PROJECT_ROOT = "/modal-workflows"


def build_workflows_image(extra_install_arguments: str = ""):
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("setuptools", "wheel")
        .add_local_dir(
            str(PROJECT_ROOT),
            remote_path=CONTAINER_PROJECT_ROOT,
            copy=True,
            ignore=["*.venv", ".idea", "__pycache__", ".pytest_cache", "*.egg-info"],
        )
        .run_commands(f"cd {CONTAINER_PROJECT_ROOT} && pip install -e .{extra_install_arguments}")
        .env({"PYTHONPATH": CONTAINER_PROJECT_ROOT})
    )
    return image


def resolve_workflow_restarts(
    result,
    *,
    max_retries: int = 10,
    retry_message: str = "Workflow restarted, waiting for results...",
    error_message: str = "Workflow exceeded max restart retries.",
):
    from modal_workflows.workflows.decorator import WorkflowRestartedResult

    retries = 0
    while isinstance(result, WorkflowRestartedResult):
        retries += 1
        if retries > max_retries:
            raise RuntimeError(error_message)

        print(retry_message)
        result = result.to_modal_function().get()

    return result, retries

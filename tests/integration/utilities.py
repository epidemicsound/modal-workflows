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
    call: modal.functions.FunctionCall,
    *,
    max_retries: int = 10,
    retry_message: str = "Workflow restarted, waiting for results...",
    error_message: str = "Workflow exceeded max restart retries.",
):
    """Follows a workflow's restarts to its final result.

    Handles both restart contracts: ``@workflow`` returns a plain
    ``{"_workflow_restarted": True, "call_id": ...}`` dict, while ``@training``
    returns a ``WorkflowRestartedResult`` sentinel. *call* is the initial
    ``FunctionCall`` (e.g. from ``.spawn()``), not an already-resolved result, so a
    restart on the very first attempt is handled the same way as a later one.
    """
    from modal_workflows.workflows.decorator import WorkflowRestartedResult

    retries = 0
    while True:
        result = call.get()

        if isinstance(result, dict) and result.get("_workflow_restarted"):
            next_call = modal.FunctionCall.from_id(result["call_id"])
        elif isinstance(result, WorkflowRestartedResult):
            next_call = result.to_modal_function()
        else:
            return result, retries

        retries += 1
        if retries > max_retries:
            raise RuntimeError(error_message)

        print(retry_message)
        call = next_call

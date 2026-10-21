# mypy: ignore-errors
import signal
from pathlib import Path
from unittest.mock import patch

import pytest

import modal_workflows.workflows.state as state_module

from tests.workflows.fakes import Env  # isort: skip

TEST_TIME_LIMIT_SECONDS = 10


@pytest.fixture(autouse=True)
def fail_instead_of_hanging():
    """Fails a test that outlives the time limit, e.g. a wait loop that polls forever."""

    def on_alarm(signum, frame):
        pytest.fail(f"Test did not finish within {TEST_TIME_LIMIT_SECONDS} s.")

    previous_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(TEST_TIME_LIMIT_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@pytest.fixture
def env(tmp_path: Path):
    """A fake Modal backend (Dict and state Volume) behind every ``WorkflowState``."""
    environment = Env(tmp_path)
    with (
        patch.object(state_module, "modal") as mock_modal,
        patch.object(
            state_module.workflow_volumes, "WORKFLOW_STATE_VOLUME_MOUNT_PATH", environment.live
        ),
        patch.object(
            state_module.workflow_volumes,
            "create_or_get_volume",
            side_effect=environment._create_or_get_volume,
        ),
    ):
        mock_modal.Dict.from_name.return_value = environment.dict
        mock_modal.Dict.objects.delete.side_effect = environment._delete_dict
        yield environment

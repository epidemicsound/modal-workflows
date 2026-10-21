from unittest.mock import patch

import pytest

from modal_workflows.workflows.alerts import modal_app_url


@pytest.fixture
def modal_environment():
    """Patches the Modal environment that ``modal_app_url`` reads from the config."""
    config = {"environment": ""}
    with patch("modal_workflows.workflows.alerts.modal.config.config", config):
        yield config


class TestModalAppUrl:
    def test_includes_workspace_and_environment_when_both_are_set(
        self, modal_environment, monkeypatch
    ):
        monkeypatch.setenv("MODAL_WORKSPACE", "my-workspace")
        modal_environment["environment"] = "testing"

        assert modal_app_url("ap-123") == "https://modal.com/apps/my-workspace/testing/ap-123"

    def test_omits_workspace_when_unset(self, modal_environment, monkeypatch):
        monkeypatch.delenv("MODAL_WORKSPACE", raising=False)
        modal_environment["environment"] = "testing"

        assert modal_app_url("ap-123") == "https://modal.com/apps/testing/ap-123"

    def test_omits_environment_when_unset(self, modal_environment, monkeypatch):
        monkeypatch.setenv("MODAL_WORKSPACE", "my-workspace")

        assert modal_app_url("ap-123") == "https://modal.com/apps/my-workspace/ap-123"

# mypy: ignore-errors
from unittest.mock import patch

import modal_workflows.workflows.volumes as volumes_module


class TestWorkflowVolumes:
    def test_create_volume_uses_modal_volume(self):
        with patch.object(volumes_module, "modal") as mock_modal:
            expected_volume = object()
            mock_modal.Volume.from_name.return_value = expected_volume

            result = volumes_module.create_or_get_volume("my-volume")

        assert result is expected_volume
        mock_modal.Volume.from_name.assert_called_once_with(
            "my-volume",
            create_if_missing=True,
            version=2,
        )

    def test_with_state_volume_adds_missing_mount(self):
        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            state_volume = object()
            mock_create_volume.return_value = state_volume

            result = volumes_module.with_state_volume({})

        assert (
            result["volumes"][str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH)] is state_volume
        )
        mock_create_volume.assert_called_once_with(volumes_module.WORKFLOW_STATE_VOLUME_NAME)

    def test_with_state_volume_preserves_existing_mount(self):
        existing_state_volume = object()
        existing_volumes = {
            str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH): existing_state_volume,
        }

        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            result = volumes_module.with_state_volume({"volumes": existing_volumes})

        assert (
            result["volumes"][str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH)]
            is existing_state_volume
        )
        mock_create_volume.assert_not_called()

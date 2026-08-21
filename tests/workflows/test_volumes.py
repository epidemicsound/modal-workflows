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

    def test_with_workflow_volumes_adds_missing_mounts(self):
        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            state_volume = object()
            artifacts_volume = object()
            mock_create_volume.side_effect = [state_volume, artifacts_volume]

            result = volumes_module.with_workflow_volumes({})

        assert (
            result["volumes"][str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH)] is state_volume
        )
        assert (
            result["volumes"][str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH)]
            is artifacts_volume
        )
        assert mock_create_volume.call_count == 2
        mock_create_volume.assert_any_call(volumes_module.WORKFLOW_STATE_VOLUME_NAME)
        mock_create_volume.assert_any_call(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_NAME)

    def test_with_artifacts_volume_adds_artifacts_mount(self):
        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            artifacts_volume = object()
            mock_create_volume.return_value = artifacts_volume

            result = volumes_module.with_artifacts_volume({})

        assert (
            result["volumes"][str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH)]
            is artifacts_volume
        )
        mock_create_volume.assert_called_once_with(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_NAME)

    def test_with_artifacts_volume_does_not_add_state_mount(self):
        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            mock_create_volume.return_value = object()

            result = volumes_module.with_artifacts_volume({})

        assert str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH) not in result["volumes"]

    def test_with_artifacts_volume_preserves_existing_artifacts_mount(self):
        existing_artifacts_volume = object()
        existing_volumes = {
            str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH): existing_artifacts_volume,
        }

        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            result = volumes_module.with_artifacts_volume({"volumes": existing_volumes})

        assert (
            result["volumes"][str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH)]
            is existing_artifacts_volume
        )
        mock_create_volume.assert_not_called()

    def test_with_artifacts_volume_preserves_user_volumes(self):
        user_volume = object()
        existing_volumes = {"/mnt/data": user_volume}

        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            mock_create_volume.return_value = object()

            result = volumes_module.with_artifacts_volume({"volumes": existing_volumes})

        assert result["volumes"]["/mnt/data"] is user_volume
        assert str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH) in result["volumes"]

    def test_with_workflow_volumes_preserves_existing_mounts(self):
        existing_state_volume = object()
        existing_artifacts_volume = object()
        existing_volumes = {
            str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH): existing_state_volume,
            str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH): existing_artifacts_volume,
        }

        with patch.object(volumes_module, "create_or_get_volume") as mock_create_volume:
            result = volumes_module.with_workflow_volumes({"volumes": existing_volumes})

        assert (
            result["volumes"][str(volumes_module.WORKFLOW_STATE_VOLUME_MOUNT_PATH)]
            is existing_state_volume
        )
        assert (
            result["volumes"][str(volumes_module.WORKFLOW_ARTIFACTS_VOLUME_MOUNT_PATH)]
            is existing_artifacts_volume
        )
        mock_create_volume.assert_not_called()

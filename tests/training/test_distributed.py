import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import modal_workflows.training.distributed as distributed_module
from modal_workflows.training.distributed import (
    DistributedLaunchContext,
    build_modal_multi_node_torchrun_command,
)


class TestDistributedLaunchContext:
    def test_modal_multi_node_rejects_cluster_size_below_two(self):
        with pytest.raises(ValueError, match="requires cluster_size > 1"):
            DistributedLaunchContext.modal_multi_node(
                cluster_size=1,
                gpu_per_node=4,
            )

    def test_resolve_clustered_context_uses_modal_cluster_info(self):
        cluster_info = SimpleNamespace(rank=1, container_ips=["fd00::1", "fd00::2"])
        experimental_api = SimpleNamespace(get_cluster_info=MagicMock(return_value=cluster_info))
        with patch.object(
            distributed_module.modal,
            "experimental",
            experimental_api,
            create=True,
        ):
            context = DistributedLaunchContext.modal_multi_node(
                cluster_size=2,
                gpu_per_node=8,
            )

        assert context.num_nodes == 2
        assert context.node_rank == 1
        assert context.master_addr == "fd00::1"
        assert context.world_size == 16

    def test_resolve_clustered_context_rejects_size_mismatch(self):
        cluster_info = SimpleNamespace(rank=0, container_ips=["fd00::1"])
        experimental_api = SimpleNamespace(get_cluster_info=MagicMock(return_value=cluster_info))
        with patch.object(
            distributed_module.modal,
            "experimental",
            experimental_api,
            create=True,
        ):
            with pytest.raises(ValueError, match="Cluster size mismatch"):
                DistributedLaunchContext.modal_multi_node(
                    cluster_size=2,
                    gpu_per_node=8,
                )

    def test_resolve_clustered_context_rejects_empty_container_ips(self):
        cluster_info = SimpleNamespace(rank=0, container_ips=[])
        experimental_api = SimpleNamespace(get_cluster_info=MagicMock(return_value=cluster_info))
        with patch.object(
            distributed_module.modal,
            "experimental",
            experimental_api,
            create=True,
        ):
            with pytest.raises(ValueError, match="did not include any container IPs"):
                DistributedLaunchContext.modal_multi_node(
                    cluster_size=2,
                    gpu_per_node=8,
                )

    def test_validate_rejects_invalid_rank_or_node_count(self):
        with pytest.raises(ValueError, match="node_rank must satisfy 0 <= node_rank < num_nodes"):
            DistributedLaunchContext(
                num_nodes=2,
                gpu_per_node=8,
                node_rank=2,
                master_addr="fd00::1",
            )
        with pytest.raises(ValueError, match="num_nodes must be >= 2"):
            DistributedLaunchContext(
                num_nodes=1,
                gpu_per_node=1,
                node_rank=0,
                master_addr="127.0.0.1",
            )

    def test_build_modal_multi_node_torchrun_command_returns_expected_argv(self):
        cluster_info = SimpleNamespace(rank=1, container_ips=["fd00::1", "fd00::2", "fd00::3"])
        experimental_api = SimpleNamespace(get_cluster_info=MagicMock(return_value=cluster_info))
        with patch.object(
            distributed_module.modal,
            "experimental",
            experimental_api,
            create=True,
        ):
            command = build_modal_multi_node_torchrun_command(
                cluster_size=3,
                gpu_per_node=8,
                training_target=["-m", "training.entrypoint", "--config", "cfg.toml"],
            )

        assert command == [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=3",
            "--node-rank=1",
            "--nproc-per-node=8",
            "--master-addr=fd00::1",
            "-m",
            "training.entrypoint",
            "--config",
            "cfg.toml",
        ]

    def test_build_modal_multi_node_torchrun_command_logs_startup(self, caplog):
        cluster_info = SimpleNamespace(rank=0, container_ips=["fd00::1", "fd00::2"])
        experimental_api = SimpleNamespace(get_cluster_info=MagicMock(return_value=cluster_info))
        with patch.object(
            distributed_module.modal,
            "experimental",
            experimental_api,
            create=True,
        ):
            with caplog.at_level("INFO"):
                build_modal_multi_node_torchrun_command(
                    cluster_size=2,
                    gpu_per_node=8,
                    training_target=["-m", "training.entrypoint"],
                )

        assert "Distributed launch mode=clustered" in caplog.text
        assert "rank=0" in caplog.text
        assert "num_nodes=2" in caplog.text
        assert "master=fd00::1" in caplog.text

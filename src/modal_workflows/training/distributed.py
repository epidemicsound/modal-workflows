"""Modal multi-node distributed launch helpers.

Call :func:`build_modal_multi_node_torchrun_command` from training entrypoints; it
resolves cluster topology, logs, and returns a ``torchrun`` argv list.
"""

from __future__ import annotations

import dataclasses
import logging
import sys

import modal.experimental

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DistributedLaunchContext:
    """Resolved launch topology for ``torch.run`` / ``torch.distributed.run`` on Modal clusters.

    ``world_size`` is derived as ``num_nodes * gpu_per_node``. Only multi-node
    configurations (``num_nodes >= 2``) are supported.
    """

    num_nodes: int
    gpu_per_node: int
    node_rank: int
    master_addr: str
    world_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Sets ``world_size`` and runs field validation."""
        self.world_size = self.num_nodes * self.gpu_per_node
        self._validate()

    def _validate(self) -> None:
        """Raises :class:`ValueError` if topology fields are inconsistent or invalid."""
        if self.num_nodes < 2:
            raise ValueError("num_nodes must be >= 2 for DistributedLaunchContext")
        if self.gpu_per_node < 1:
            raise ValueError("gpu_per_node must be >= 1")
        if not 0 <= self.node_rank < self.num_nodes:
            raise ValueError("node_rank must satisfy 0 <= node_rank < num_nodes")
        if not self.master_addr:
            raise ValueError("master_addr must be a non-empty string")

    @classmethod
    def modal_multi_node(
        cls,
        cluster_size: int,
        gpu_per_node: int,
    ) -> DistributedLaunchContext:
        """Builds a context from :func:`modal.experimental.get_cluster_info`.

        The cluster leader (first container IP) becomes ``master_addr``. ``node_rank`` matches
        Modal's rank. Raises :class:`ValueError` if ``cluster_size`` does not match Modal's
        reported node list or if IPs are missing.
        """
        if cluster_size <= 1:
            raise ValueError("modal_multi_node requires cluster_size > 1")

        cluster_info = modal.experimental.get_cluster_info()
        if not cluster_info.container_ips:
            raise ValueError("Cluster info did not include any container IPs")
        if len(cluster_info.container_ips) != cluster_size:
            raise ValueError(
                f"Cluster size mismatch: requested {cluster_size} nodes, "
                f"but cluster info reports {len(cluster_info.container_ips)} nodes"
            )
        return cls(
            num_nodes=cluster_size,
            gpu_per_node=gpu_per_node,
            node_rank=cluster_info.rank,
            master_addr=cluster_info.container_ips[0],
        )

    def torchrun_argv(self, training_target: list[str]) -> list[str]:
        """Builds an argv list: ``python -m torch.distributed.run ...`` plus ``training_target``."""
        if not training_target:
            raise ValueError("training_target cannot be empty")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={self.num_nodes}",
            f"--node-rank={self.node_rank}",
            f"--nproc-per-node={self.gpu_per_node}",
            f"--master-addr={self.master_addr}",
        ]
        command.extend(training_target)
        return command

    def log_launch_context(self) -> None:
        """Emits a single INFO line with rank, node count, GPUs per node, world size, and master."""
        logger.info(
            "Distributed launch mode=clustered rank=%s num_nodes=%s gpu_per_node=%s "
            "world_size=%s master=%s",
            self.node_rank,
            self.num_nodes,
            self.gpu_per_node,
            self.world_size,
            self.master_addr,
        )


def build_modal_multi_node_torchrun_command(
    *,
    cluster_size: int,
    gpu_per_node: int,
    training_target: list[str],
) -> list[str]:
    """Builds a full ``torchrun`` argv for Modal multi-node training.

    Resolves topology from Modal cluster metadata, logs one startup line, and returns
    the command list.
    """
    context = DistributedLaunchContext.modal_multi_node(
        cluster_size=cluster_size,
        gpu_per_node=gpu_per_node,
    )
    context.log_launch_context()
    return context.torchrun_argv(training_target)

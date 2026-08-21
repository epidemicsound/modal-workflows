"""Training helpers for Modal.

Framework-agnostic helpers (``@training``, distributed launch) are importable without
any extras. The Lightning callback requires the ``training`` extra (pytorch-lightning,
torch) and lives at :mod:`modal_workflows.training.callbacks`.
"""

from modal_workflows.training.decorator import training
from modal_workflows.training.distributed import (
    DistributedLaunchContext,
    build_modal_multi_node_torchrun_command,
)

__all__ = [
    "training",
    "DistributedLaunchContext",
    "build_modal_multi_node_torchrun_command",
]

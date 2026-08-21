# mypy: ignore-errors
"""PyTorch Lightning callback for time-based checkpoints and Modal timeout restarts."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning.callbacks import Callback as LightningCallback

from modal_workflows.training.decorator import (
    _training_run_paths_from_env,
    _write_restart_metadata_file,
)

logger = logging.getLogger(__name__)


class TrainingCheckpointRestartCallback(LightningCallback):
    """Lightning callback for time-based checkpoints and Modal-orchestrated timeout restarts.

    Works for single-GPU and multi-GPU (single node). Interval saves follow the same
    wall-clock semantics as PyTorch Lightning's ``ModelCheckpoint`` with
    ``train_time_interval``: global rank 0 decides, then the decision is broadcast so
    every process calls ``Trainer.save_checkpoint`` when required by the strategy.
    When the self-managed timeout elapses, restart metadata is written and training
    stops via ``should_stop`` so the wrapped function returns normally; ``@training``
    then spawns the continuation. This callback does not raise
    :class:`~modal_workflows.workflows.context.WorkflowRestarted`.
    """

    def __init__(
        self,
        save_interval_seconds: float,
        self_managed_timeout_seconds: float,
    ) -> None:
        """Loads paths from env (set by ``@training``).

        ``save_interval_seconds`` controls periodic checkpoints; ``self_managed_timeout_seconds``
        must align with the decorator's ``self_managed_timeout`` so timeout and restart line up.
        """
        super().__init__()
        if save_interval_seconds <= 0:
            raise ValueError("save_interval_seconds must be > 0")
        if self_managed_timeout_seconds <= 0:
            raise ValueError("self_managed_timeout_seconds must be > 0")
        paths = _training_run_paths_from_env()
        self._checkpoint_path = paths.checkpoint_path
        self._restart_metadata_path = paths.restart_metadata_path
        self._state_name = paths.state_name
        self._results_root = paths.results_root
        self._save_interval_seconds = save_interval_seconds
        self._self_managed_timeout_seconds = self_managed_timeout_seconds
        self._training_start_monotonic: float | None = None
        self._last_interval_save_monotonic: float | None = None

    @staticmethod
    def _broadcast_bool_from_global_zero(trainer: Any, value: bool) -> bool:
        """Broadcasts a boolean from global rank 0 so every process shares the same decision.

        PyTorch Lightning uses the same pattern for ``train_time_interval`` in
        ``ModelCheckpoint``: all ranks must agree whether to call
        ``Trainer.save_checkpoint``, which ends with a collective barrier.
        """
        return trainer.strategy.broadcast(value)

    @staticmethod
    def _sync_bool_from_global_zero_across_processes(
        trainer: Any, value_on_rank_zero: bool
    ) -> bool:
        """Aligns a boolean across ranks when multiple processes are active (e.g. DDP).

        Global rank 0 supplies the real value; other ranks participate in the broadcast
        so every process stops and checkpoints together for a coordinated restart.
        """
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return value_on_rank_zero
        device = trainer.strategy.root_device
        encoded = 1 if (trainer.is_global_zero and value_on_rank_zero) else 0
        signal = torch.tensor([encoded], dtype=torch.int64, device=device)
        torch.distributed.broadcast(signal, src=0)
        return signal.item() == 1

    def _save_lightning_checkpoint(self, trainer: Any, *, weights_only: bool = False) -> None:
        """Persists a checkpoint (safe when multiple processes train in parallel).

        Every rank must invoke ``Trainer.save_checkpoint`` when the strategy requires it;
        only global rank 0 atomically promotes the temp file to the final path.
        """
        checkpoint_path = self._checkpoint_path
        tmp_path = checkpoint_path.with_suffix(".ckpt.tmp")
        trainer.save_checkpoint(str(tmp_path), weights_only=weights_only)
        if trainer.is_global_zero:
            os.replace(tmp_path, checkpoint_path)

    def _should_save_interval_checkpoint(self, now_monotonic: float) -> bool:
        """Returns whether periodic checkpointing should run on this step.

        ``False`` when the last-save time is unknown; otherwise compares monotonic
        elapsed time since the *completed* previous save (see ``on_train_batch_end``)
        to ``self._save_interval_seconds`` so slow writes do not starve training steps.
        """
        last_save = self._last_interval_save_monotonic
        if last_save is None:
            return False
        return (now_monotonic - last_save) >= self._save_interval_seconds

    def _restart_timeout_due(self, now_monotonic: float) -> bool:
        """Returns whether elapsed training time exceeds the Modal self-managed timeout.

        Intended to be evaluated only on global rank 0 before
        :meth:`_sync_bool_from_global_zero_across_processes` distributes the result.
        """
        start = self._training_start_monotonic
        if start is None:
            return False
        return (now_monotonic - start) > self._self_managed_timeout_seconds

    @property
    def state_name(self) -> str:
        """Workflow state id (same as ``MODAL_WORKFLOWS_TRAINING_STATE_NAME``).

        Useful as a run identifier for loggers such as W&B.
        """
        return self._state_name

    @property
    def results_root(self) -> Path:
        """Root directory on the state volume for this run (sibling of the checkpoints)."""
        return self._results_root

    def resume_checkpoint_path(self) -> str | None:
        """Absolute path for ``Trainer.fit(ckpt_path=...)``, or ``None`` if no checkpoint exists."""
        if not self._checkpoint_path.exists():
            return None
        return str(self._checkpoint_path)

    def on_train_start(self, trainer: Any, _pl_module: Any) -> None:
        """Initializes monotonic clocks for timeout and interval checkpointing."""
        started = time.monotonic()
        self._training_start_monotonic = started
        self._last_interval_save_monotonic = started

    def _handle_timeout_restart(self, trainer: Any, now: float) -> bool:
        """If the self-managed timeout has elapsed, checkpoint, write restart metadata, and stop.

        Returns ``True`` when training is being stopped so the caller should not run
        interval checkpoint logic. Global rank 0 decides timeout;
        :meth:`_sync_bool_from_global_zero_across_processes` aligns all ranks before
        the save and ``should_stop``.
        """
        if trainer.is_global_zero:
            restart_due = self._restart_timeout_due(now)
        else:
            restart_due = False
        restart_due = self._sync_bool_from_global_zero_across_processes(trainer, restart_due)
        if not restart_due:
            return False

        self._save_lightning_checkpoint(trainer, weights_only=False)
        if trainer.is_global_zero:
            _write_restart_metadata_file(
                metadata_path=self._restart_metadata_path,
                reason="runtime_timeout",
            )
        trainer.limit_val_batches = 0
        trainer.should_stop = True
        if trainer.is_global_zero:
            logger.info(
                "Training timeout handoff completed: checkpoint saved at %s, restart "
                "metadata written, validation disabled, and should_stop=True.",
                self._checkpoint_path,
            )
        return True

    def _handle_interval_checkpoint(self, trainer: Any, now: float) -> None:
        """When the interval timer says so (broadcast from rank 0), saves and advances the timer.

        Updates :attr:`_last_interval_save_monotonic` only after :meth:`_save_lightning_checkpoint`
        so long writes do not compress the interval between saves.
        """
        if trainer.is_global_zero:
            should_save = self._should_save_interval_checkpoint(now)
        else:
            should_save = False
        should_save = self._broadcast_bool_from_global_zero(trainer, should_save)
        if not should_save:
            return
        self._save_lightning_checkpoint(trainer, weights_only=False)
        if trainer.is_global_zero:
            logger.info(
                "Interval checkpoint saved at %s (interval=%ss).",
                self._checkpoint_path,
                self._save_interval_seconds,
            )
        self._last_interval_save_monotonic = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: Any,
        _pl_module: Any,
        _outputs: Any,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        """Handles timeout restart (save, metadata, stop) before periodic interval saves."""
        now = time.monotonic()
        if self._training_start_monotonic is None:
            return
        if self._handle_timeout_restart(trainer, now):
            return
        self._handle_interval_checkpoint(trainer, now)

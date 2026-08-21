# mypy: ignore-errors
"""Integration test: ``@training`` and ``TrainingCheckpointRestartCallback`` Modal handoff.

Exercises self-managed timeout, checkpoint resume across Modal spawns, and the local
entrypoint loop over :class:`~modal_workflows.workflows.decorator.WorkflowRestartedResult`.

Run:  MODAL_ENVIRONMENT=dev uv run modal run tests/integration/test_training_decorator.py

Requires ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``. Install Lightning for local
imports: ``uv sync --extra training``.
"""

from __future__ import annotations

from typing import Any

import modal

from modal_workflows.training import training
from modal_workflows.training.callbacks import TrainingCheckpointRestartCallback

from tests.integration.utilities import (  # isort: skip
    build_workflows_image,
    resolve_workflow_restarts,
)

SELF_MANAGED_TIMEOUT_SECONDS = 1.25
SAVE_INTERVAL_SECONDS = 0.35
SLEEP_PER_TRAINING_STEP_SECONDS = 0.018
TARGET_MAX_STEPS = 120
MAX_RESTARTS = 25

image = build_workflows_image(extra_install_arguments="[training]")
app = modal.App("test-training-decorator", image=image)


@training(
    app,
    self_managed_timeout=SELF_MANAGED_TIMEOUT_SECONDS,
    alert_on_error=False,
    cpu=2,
)
def run_training_with_handoff() -> dict[str, Any]:
    import time

    import torch
    from pytorch_lightning import LightningDataModule, LightningModule, Trainer
    from torch.utils.data import DataLoader, TensorDataset

    class SlowTinyModule(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            time.sleep(SLEEP_PER_TRAINING_STEP_SECONDS)
            inputs, targets = batch
            return self.linear(inputs).squeeze(-1).sub(targets).pow(2).mean()

        def configure_optimizers(self) -> Any:
            return torch.optim.SGD(self.parameters(), lr=0.01)

    class TinyDataModule(LightningDataModule):
        def __init__(self) -> None:
            super().__init__()
            self.batch_size = 8
            self.num_workers = 0

        def setup(self, stage: str | None = None) -> None:
            data = torch.randn(512, 2)
            targets = torch.randn(512)
            self._dataset = TensorDataset(data, targets)

        def train_dataloader(self) -> DataLoader[Any]:
            return DataLoader(
                self._dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

    model = SlowTinyModule()
    data_module = TinyDataModule()
    restart_callback = TrainingCheckpointRestartCallback(
        save_interval_seconds=SAVE_INTERVAL_SECONDS,
        self_managed_timeout_seconds=SELF_MANAGED_TIMEOUT_SECONDS,
    )
    ckpt_path = restart_callback.resume_checkpoint_path()
    trainer = Trainer(
        max_steps=TARGET_MAX_STEPS,
        accelerator="cpu",
        devices=1,
        callbacks=[restart_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    return {
        "status": "completed",
        "global_step": int(trainer.global_step),
        "max_steps": TARGET_MAX_STEPS,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_training_with_handoff.remote()
    result, restart_count = resolve_workflow_restarts(
        result,
        max_retries=MAX_RESTARTS,
        retry_message="Training handoff (timeout); waiting for continuation...",
        error_message="Training exceeded max Modal handoff retries.",
    )

    assert (
        restart_count >= 1
    ), f"Expected at least one WorkflowRestartedResult handoff, got {restart_count}"
    assert result["status"] == "completed", result
    assert result["global_step"] == TARGET_MAX_STEPS, result
    print(
        f"PASS: @training handoff ({restart_count} restart(s)), global_step={result['global_step']}"
    )

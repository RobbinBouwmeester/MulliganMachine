"""Training loop for the DeckTransformer using PyTorch Lightning."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from mulligan_machine.data.dataset import DeckDataset
from mulligan_machine.model.config import ModelConfig
from mulligan_machine.model.transformer import DeckTransformer

logger = logging.getLogger(__name__)

# Enable TF32 for Tensor Cores (RTX 30xx/40xx) — negligible precision impact
torch.set_float32_matmul_precision("high")


class DeckTransformerLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for DeckTransformer training."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = DeckTransformer(config)

    def forward(
        self, input_ids: torch.Tensor, card_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(input_ids, card_features)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        card_features = batch.get("card_features")  # (B, T, n_features) or None
        loss = self.model.compute_loss(input_ids, card_features=card_features, ignore_index=0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log perplexity
        ppl = torch.exp(loss).item()
        self.log("train_ppl", ppl, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        card_features = batch.get("card_features")
        loss = self.model.compute_loss(input_ids, card_features=card_features, ignore_index=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        ppl = torch.exp(loss).item()
        self.log("val_ppl", ppl, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):  # type: ignore[override]
        # Separate weight decay groups: no decay for biases and layernorms
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "ln" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )

        # Warmup + cosine annealing schedule
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        # Estimate total steps for cosine annealing
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 10_000
        cosine_steps = int(max(total_steps - self.config.warmup_steps, 1))
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def train(
    config: ModelConfig,
    train_dataset: DeckDataset,
    val_dataset: DeckDataset,
    checkpoint_dir: Path = Path("checkpoints"),
    resume_from: str | None = None,
    num_workers: int = 4,
    accelerator: str = "auto",
    early_stop_patience: int = 100,
) -> DeckTransformerLightning:
    """
    Train the DeckTransformer model.

    Args:
        config: Model/training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        checkpoint_dir: Where to save checkpoints.
        resume_from: Path to a checkpoint to resume from.
        num_workers: DataLoader workers.
        accelerator: 'auto', 'gpu', 'cpu'.

    Returns:
        The trained Lightning module.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Model
    model = DeckTransformerLightning(config)

    # Callbacks
    callbacks_list: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="deck-transformer-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    if early_stop_patience > 0:
        callbacks_list.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                mode="min",
            )
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks_list,
        log_every_n_steps=10,
        val_check_interval=0.25,  # validate 4 times per epoch
        default_root_dir=str(checkpoint_dir),
    )

    logger.info("Starting training...")
    logger.info("  Train samples: %d", len(train_dataset))
    logger.info("  Val samples:   %d", len(val_dataset))
    logger.info("  Batch size:    %d", config.batch_size)
    logger.info("  Max epochs:    %d", config.max_epochs)
    logger.info("  Model params:  ~%.1fM", config.total_params_estimate() / 1e6)

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    return model

"""Training loop for Chess Master 9001."""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from chess_master.config import Config
from chess_master.model import ChessMaster9001
from chess_master.training.callbacks import (
    Checkpointer,
    NaNDetector,
    TensorBoardLogger,
)
from chess_master.training.losses import ChessLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with logging, checkpointing, and diagnostics."""

    def __init__(self, config: Config, model: ChessMaster9001):
        self.config = config
        self.model = model
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss
        self.criterion = ChessLoss(config.training.loss_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Setup run directory
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save frozen config
        OmegaConf.save(OmegaConf.structured(config), self.run_dir / "config.yaml")

        # Callbacks
        self.tb_logger = TensorBoardLogger(str(self.run_dir / "tensorboard"))
        self.nan_detector = NaNDetector(str(self.run_dir / "emergency"))
        self.checkpointer = Checkpointer(str(self.run_dir / "checkpoints"))

        self.global_step = 0

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        """Run training loop."""
        self.model.train()
        tc = self.config.training

        logger.info(f"Starting training for {tc.max_steps} steps")
        logger.info(f"Device: {self.device}")

        start_time = time.time()

        for batch in self._infinite_loader(train_loader):
            if self.global_step >= tc.max_steps:
                break

            loss_dict = self._train_step(batch)

            # Logging
            if self.global_step % 100 == 0:
                self.tb_logger.log_losses(loss_dict, self.global_step)
                self.tb_logger.log_gradient_norms(self.model, self.global_step)

                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Step {self.global_step}/{tc.max_steps} | "
                    f"loss={loss_dict['total'].item():.4f} | "
                    f"policy={loss_dict['policy'].item():.4f} | "
                    f"value={loss_dict['value'].item():.4f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            # Evaluation
            if val_loader is not None and self.global_step % tc.eval_every == 0 and self.global_step > 0:
                self._evaluate(val_loader)

            # Checkpointing
            if self.global_step % tc.checkpoint_every == 0 and self.global_step > 0:
                self.checkpointer.save(
                    self.model, self.optimizer, self.global_step
                )

            self.global_step += 1

        # Final checkpoint
        self.checkpointer.save(self.model, self.optimizer, self.global_step)
        self.tb_logger.close()
        logger.info(f"Training complete at step {self.global_step}")

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Execute a single training step."""
        # Move batch to device
        board_tensor = batch["board_tensor"].to(self.device)
        target_move = batch["target_move"].to(self.device)
        target_value = batch["target_value"].to(self.device)
        legal_mask = batch.get("legal_mask")
        if legal_mask is not None:
            legal_mask = legal_mask.to(self.device)

        # Forward
        output = self.model(board_tensor, legal_mask=legal_mask)

        # Loss
        loss_dict = self.criterion(
            output.policy_logits, output.value, target_move, target_value
        )

        # Backward
        self.optimizer.zero_grad()
        loss_dict["total"].backward()

        # NaN detection
        if self.nan_detector.check(self.model, self.global_step):
            logger.error("NaN detected, skipping step")
            self.optimizer.zero_grad()
            return loss_dict

        # Gradient clipping
        if self.config.training.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip
            )

        self.optimizer.step()

        # Log policy accuracy
        if self.global_step % 100 == 0:
            self.tb_logger.log_policy_accuracy(
                output.policy_logits.detach(), target_move, self.global_step
            )

        return loss_dict

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_top1 = 0
        total = 0

        for batch in val_loader:
            board_tensor = batch["board_tensor"].to(self.device)
            target_move = batch["target_move"].to(self.device)
            target_value = batch["target_value"].to(self.device)
            legal_mask = batch.get("legal_mask")
            if legal_mask is not None:
                legal_mask = legal_mask.to(self.device)

            output = self.model(board_tensor, legal_mask=legal_mask)
            loss_dict = self.criterion(
                output.policy_logits, output.value, target_move, target_value
            )

            B = board_tensor.shape[0]
            total_loss += loss_dict["total"].item() * B
            total_policy_loss += loss_dict["policy"].item() * B
            total_value_loss += loss_dict["value"].item() * B
            correct_top1 += (output.policy_logits.argmax(-1) == target_move).sum().item()
            total += B

        metrics = {
            "val/loss": total_loss / total,
            "val/policy_loss": total_policy_loss / total,
            "val/value_loss": total_value_loss / total,
            "val/top1_accuracy": correct_top1 / total,
        }

        for key, val in metrics.items():
            self.tb_logger.log_scalar(key, val, self.global_step)

        logger.info(
            f"Eval step {self.global_step} | "
            f"loss={metrics['val/loss']:.4f} | "
            f"top1={metrics['val/top1_accuracy']:.4f}"
        )

        self.model.train()
        return metrics

    @staticmethod
    def _infinite_loader(loader: DataLoader):
        """Iterate over a DataLoader infinitely."""
        while True:
            yield from loader

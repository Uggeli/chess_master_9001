"""Training callbacks for logging, checkpointing, and diagnostics."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Structured TensorBoard logging for training metrics."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)

    def log_losses(self, losses: dict[str, Tensor], step: int) -> None:
        for name, value in losses.items():
            self.writer.add_scalar(f"loss/{name}", value.item(), step)

    def log_gradient_norms(self, model: nn.Module, step: int) -> None:
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                # Group by top-level module
                module = name.split(".")[0]
                self.writer.add_scalar(f"grad_norm/{module}/{name}", norm, step)

    def log_retrieval_stats(
        self, similarity_scores: Tensor, step: int,
    ) -> None:
        if similarity_scores is None:
            return
        self.writer.add_scalar(
            "retrieval/mean_similarity", similarity_scores.mean().item(), step
        )
        self.writer.add_scalar(
            "retrieval/max_similarity", similarity_scores.max().item(), step
        )
        # Retrieval entropy
        probs = torch.softmax(similarity_scores, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        self.writer.add_scalar("retrieval/entropy", entropy.item(), step)

    def log_policy_accuracy(
        self, logits: Tensor, targets: Tensor, step: int,
    ) -> None:
        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == targets).float().mean()
            top5 = _topk_accuracy(logits, targets, k=5)
            self.writer.add_scalar("accuracy/top1", top1.item(), step)
            self.writer.add_scalar("accuracy/top5", top5.item(), step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()


def _topk_accuracy(logits: Tensor, targets: Tensor, k: int) -> Tensor:
    topk_preds = logits.topk(k, dim=-1).indices
    return (topk_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean()


class NaNDetector:
    """Detects NaN/Inf in gradients and saves emergency checkpoint."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def check(self, model: nn.Module, step: int) -> bool:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"NaN/Inf gradient detected in {name} at step {step}")
                    self._save_emergency(model, step)
                    return True
        return False

    def _save_emergency(self, model: nn.Module, step: int) -> None:
        path = self.save_dir / f"emergency_step_{step}.pt"
        torch.save(model.state_dict(), path)
        logger.error(f"Emergency checkpoint saved to {path}")


class Checkpointer:
    """Saves model checkpoints periodically."""

    def __init__(self, save_dir: str, keep_last: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._saved: list[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        config: dict | None = None,
    ) -> Path:
        path = self.save_dir / f"checkpoint_step_{step}.pt"
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        if config is not None:
            state["config"] = config
        torch.save(state, path)

        self._saved.append(path)
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

        logger.info(f"Checkpoint saved to {path}")
        return path

    @staticmethod
    def load(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
        state = torch.load(path, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return state.get("step", 0)

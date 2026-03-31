"""Integration test: one training step reduces loss."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from chess_master.config import Config, ModelConfig, LossWeights, TrainingConfig
from chess_master.model import ChessMaster9001
from chess_master.training.losses import ChessLoss
from chess_master.chess_master_types import NUM_ACTIONS, Phase


class TestTrainingStep:
    def test_one_step_reduces_loss(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        model.train()

        criterion = ChessLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create a small batch
        board_tensor = torch.randn(8, 18, 8, 8)
        target_move = torch.randint(0, NUM_ACTIONS, (8,))
        target_value = torch.randn(8).clamp(-1, 1)

        # First forward pass
        output1 = model(board_tensor)
        losses1 = criterion(output1.policy_logits, output1.value, target_move, target_value)

        # Backward + step
        optimizer.zero_grad()
        losses1["total"].backward()
        optimizer.step()

        # Second forward pass
        output2 = model(board_tensor)
        losses2 = criterion(output2.policy_logits, output2.value, target_move, target_value)

        # Loss should decrease (with high probability for random init)
        assert losses2["total"].item() < losses1["total"].item()

    def test_gradient_clipping(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        model.train()

        board_tensor = torch.randn(4, 18, 8, 8)
        output = model(board_tensor)
        (output.policy_logits.sum() * 1000).backward()  # Large gradient

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        total_norm = sum(
            p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm <= 1.0 + 1e-6

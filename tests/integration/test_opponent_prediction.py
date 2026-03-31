"""Integration tests for opponent reply prediction (Phase B)."""

import pytest
import torch

from chess_master.config import ModelConfig, LossWeights
from chess_master.model import ChessMaster9001
from chess_master.training.losses import ChessLoss
from chess_master.chess_master_types import NUM_ACTIONS, Phase


class TestOpponentPredictionModel:
    def test_opponent_head_created_in_phase_b(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.B)
        assert hasattr(model, "opponent_head")

    def test_opponent_head_not_created_in_phase_a(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        assert not hasattr(model, "opponent_head")

    def test_predict_opponent_reply_output_shape(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.B)
        afterstate = torch.randn(2, 18, 8, 8)
        logits = model.predict_opponent_reply(afterstate)
        assert logits.shape == (2, NUM_ACTIONS)

    def test_predict_opponent_reply_with_mask(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.B)
        afterstate = torch.randn(2, 18, 8, 8)
        mask = torch.zeros(2, NUM_ACTIONS)
        mask[:, :10] = 1.0
        logits = model.predict_opponent_reply(afterstate, opponent_legal_mask=mask)
        assert (logits[:, 10:] == float("-inf")).all()
        assert (logits[:, :10] != float("-inf")).all()

    def test_predict_opponent_reply_raises_in_phase_a(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        afterstate = torch.randn(2, 18, 8, 8)
        with pytest.raises(RuntimeError, match="Opponent head not available"):
            model.predict_opponent_reply(afterstate)

    def test_forward_pass_unchanged_in_phase_b(self, tiny_config):
        """The main forward pass should still work normally in Phase B."""
        model = ChessMaster9001(tiny_config, active_phase=Phase.B)
        board_tensor = torch.randn(2, 18, 8, 8)
        output = model(board_tensor)
        assert output.policy_logits.shape == (2, NUM_ACTIONS)
        assert output.value.shape == (2, 1)

    def test_opponent_gradient_flow(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.B)
        afterstate = torch.randn(2, 18, 8, 8)
        logits = model.predict_opponent_reply(afterstate)
        logits.sum().backward()
        # Opponent head parameters should have gradients
        for name, param in model.opponent_head.named_parameters():
            assert param.grad is not None, f"No gradient for opponent_head.{name}"
        # Backbone parameters should also have gradients (shared encoder)
        for name, param in model.backbone.named_parameters():
            assert param.grad is not None, f"No gradient for backbone.{name}"


class TestOpponentLoss:
    def test_opponent_loss_computed_when_targets_provided(self):
        loss_fn = ChessLoss()
        logits = torch.randn(4, NUM_ACTIONS)
        value = torch.randn(4, 1)
        target_move = torch.randint(0, NUM_ACTIONS, (4,))
        target_value = torch.randn(4)
        opponent_logits = torch.randn(4, NUM_ACTIONS)
        target_opponent = torch.randint(0, NUM_ACTIONS, (4,))

        losses = loss_fn(
            logits, value, target_move, target_value,
            opponent_logits=opponent_logits,
            target_opponent_move=target_opponent,
        )
        assert "opponent" in losses
        assert losses["opponent"].item() > 0

    def test_opponent_loss_not_computed_without_targets(self):
        loss_fn = ChessLoss()
        logits = torch.randn(4, NUM_ACTIONS)
        value = torch.randn(4, 1)
        target_move = torch.randint(0, NUM_ACTIONS, (4,))
        target_value = torch.randn(4)

        losses = loss_fn(logits, value, target_move, target_value)
        assert "opponent" not in losses

    def test_opponent_loss_affects_total(self):
        weights = LossWeights(policy=1.0, value=1.0, opponent=0.5)
        loss_fn = ChessLoss(weights)
        logits = torch.randn(4, NUM_ACTIONS)
        value = torch.randn(4, 1)
        target_move = torch.randint(0, NUM_ACTIONS, (4,))
        target_value = torch.randn(4)
        opponent_logits = torch.randn(4, NUM_ACTIONS)
        target_opponent = torch.randint(0, NUM_ACTIONS, (4,))

        losses = loss_fn(
            logits, value, target_move, target_value,
            opponent_logits=opponent_logits,
            target_opponent_move=target_opponent,
        )
        expected_total = (
            weights.policy * losses["policy"]
            + weights.value * losses["value"]
            + weights.opponent * losses["opponent"]
        )
        torch.testing.assert_close(losses["total"], expected_total)

    def test_opponent_loss_gradient_flow(self):
        loss_fn = ChessLoss()
        opponent_logits = torch.randn(4, NUM_ACTIONS, requires_grad=True)
        target_opponent = torch.randint(0, NUM_ACTIONS, (4,))
        policy_logits = torch.randn(4, NUM_ACTIONS)
        value = torch.randn(4, 1)
        target_move = torch.randint(0, NUM_ACTIONS, (4,))
        target_value = torch.randn(4)

        losses = loss_fn(
            policy_logits, value, target_move, target_value,
            opponent_logits=opponent_logits,
            target_opponent_move=target_opponent,
        )
        losses["total"].backward()
        assert opponent_logits.grad is not None

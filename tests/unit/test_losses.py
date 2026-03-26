"""Tests for loss functions."""

import torch

from chess_master.training.losses import ChessLoss
from chess_master.config import LossWeights


class TestChessLoss:
    def test_policy_loss(self):
        loss_fn = ChessLoss()
        logits = torch.randn(4, 4672)
        targets = torch.randint(0, 4672, (4,))
        loss = loss_fn.policy_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_value_loss(self):
        loss_fn = ChessLoss()
        pred = torch.randn(4, 1)
        target = torch.randn(4)
        loss = loss_fn.value_loss(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_combined_loss(self):
        loss_fn = ChessLoss()
        logits = torch.randn(4, 4672)
        value = torch.randn(4, 1)
        target_move = torch.randint(0, 4672, (4,))
        target_value = torch.randn(4)
        losses = loss_fn(logits, value, target_move, target_value)
        assert "total" in losses
        assert "policy" in losses
        assert "value" in losses

    def test_loss_weights(self):
        weights = LossWeights(policy=2.0, value=0.5, retrieval=0.0)
        loss_fn = ChessLoss(weights)
        logits = torch.randn(4, 4672)
        value = torch.randn(4, 1)
        target_move = torch.randint(0, 4672, (4,))
        target_value = torch.randn(4)
        losses = loss_fn(logits, value, target_move, target_value)
        # Total should be 2.0 * policy + 0.5 * value
        expected = 2.0 * losses["policy"] + 0.5 * losses["value"]
        torch.testing.assert_close(losses["total"], expected)

    def test_retrieval_loss(self):
        loss_fn = ChessLoss()
        query = torch.randn(4, 16)
        positive = query + torch.randn(4, 16) * 0.1  # similar
        loss = loss_fn.retrieval_loss(query, positive)
        assert loss.item() < 1.0  # Should be small for similar embeddings

    def test_gradients_flow(self):
        loss_fn = ChessLoss()
        logits = torch.randn(4, 4672, requires_grad=True)
        value = torch.randn(4, 1, requires_grad=True)
        target_move = torch.randint(0, 4672, (4,))
        target_value = torch.randn(4)
        losses = loss_fn(logits, value, target_move, target_value)
        losses["total"].backward()
        assert logits.grad is not None
        assert value.grad is not None

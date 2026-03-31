"""Tests for opponent reply prediction head."""

import torch

from chess_master.heads.opponent import OpponentHead
from chess_master.chess_master_types import NUM_ACTIONS


class TestOpponentHead:
    def test_output_shape(self):
        head = OpponentHead(d_model=32)
        spatial = torch.randn(2, 64, 32)
        logits = head(spatial)
        assert logits.shape == (2, NUM_ACTIONS)

    def test_output_shape_single(self):
        head = OpponentHead(d_model=32)
        spatial = torch.randn(1, 64, 32)
        logits = head(spatial)
        assert logits.shape == (1, NUM_ACTIONS)

    def test_legal_masking(self):
        head = OpponentHead(d_model=32)
        spatial = torch.randn(1, 64, 32)
        mask = torch.zeros(1, NUM_ACTIONS)
        mask[0, :5] = 1.0  # Only 5 legal moves
        logits = head(spatial, opponent_legal_mask=mask)
        assert (logits[0, 5:] == float("-inf")).all()
        assert (logits[0, :5] != float("-inf")).all()

    def test_no_mask(self):
        head = OpponentHead(d_model=32)
        spatial = torch.randn(1, 64, 32)
        logits = head(spatial, opponent_legal_mask=None)
        # No -inf values when no mask is applied
        assert torch.isfinite(logits).all()

    def test_gradient_flow(self):
        head = OpponentHead(d_model=32)
        spatial = torch.randn(2, 64, 32, requires_grad=True)
        logits = head(spatial)
        logits.sum().backward()
        assert spatial.grad is not None
        assert spatial.grad.shape == (2, 64, 32)

    def test_different_d_model(self):
        for d_model in [64, 128, 256]:
            head = OpponentHead(d_model=d_model)
            spatial = torch.randn(2, 64, d_model)
            logits = head(spatial)
            assert logits.shape == (2, NUM_ACTIONS)

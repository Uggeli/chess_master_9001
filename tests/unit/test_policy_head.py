"""Tests for policy head."""

import torch

from chess_master.heads.policy import PolicyHead
from chess_master.types import NUM_ACTIONS


class TestPolicyHead:
    def test_output_shape(self):
        head = PolicyHead(d_model=32)
        spatial = torch.randn(2, 64, 32)
        logits = head(spatial)
        assert logits.shape == (2, NUM_ACTIONS)

    def test_legal_masking(self):
        head = PolicyHead(d_model=32)
        spatial = torch.randn(1, 64, 32)
        mask = torch.zeros(1, NUM_ACTIONS)
        mask[0, :5] = 1.0  # Only 5 legal moves
        logits = head(spatial, legal_mask=mask)
        assert (logits[0, 5:] == float("-inf")).all()
        assert (logits[0, :5] != float("-inf")).all()

    def test_gradient_flow(self):
        head = PolicyHead(d_model=32)
        spatial = torch.randn(2, 64, 32, requires_grad=True)
        logits = head(spatial)
        logits.sum().backward()
        assert spatial.grad is not None

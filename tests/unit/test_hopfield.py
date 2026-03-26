"""Tests for Hopfield memory layer."""

import torch

from chess_master.memory.hopfield import HopfieldLayer


class TestHopfieldLayer:
    def test_output_shapes(self):
        layer = HopfieldLayer(dim=16, n_heads=4)
        query = torch.randn(2, 16)       # [B, d]
        keys = torch.randn(2, 10, 16)    # [B, N, d]
        values = torch.randn(2, 10, 16)  # [B, N, d]
        retrieved, attn, sims = layer(query, keys, values)
        assert retrieved.shape == (2, 16)
        assert attn.shape == (2, 4, 1, 10)  # [B, H, Q, N]
        assert sims.shape == (2, 10)        # [B, N]

    def test_batched_query(self):
        layer = HopfieldLayer(dim=16, n_heads=4)
        query = torch.randn(2, 3, 16)     # [B, Q, d]
        keys = torch.randn(2, 10, 16)
        values = torch.randn(2, 10, 16)
        retrieved, attn, sims = layer(query, keys, values)
        assert retrieved.shape == (2, 3, 16)
        assert attn.shape == (2, 4, 3, 10)
        assert sims.shape == (2, 3, 10)

    def test_attention_sums_to_one(self):
        layer = HopfieldLayer(dim=16, n_heads=4)
        query = torch.randn(2, 16)
        keys = torch.randn(2, 10, 16)
        values = torch.randn(2, 10, 16)
        _, attn, _ = layer(query, keys, values)
        sums = attn.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        layer = HopfieldLayer(dim=16, n_heads=4)
        query = torch.randn(2, 16, requires_grad=True)
        keys = torch.randn(2, 10, 16)
        values = torch.randn(2, 10, 16)
        retrieved, _, _ = layer(query, keys, values)
        retrieved.sum().backward()
        assert query.grad is not None

    def test_mask(self):
        layer = HopfieldLayer(dim=16, n_heads=4)
        query = torch.randn(1, 16)
        keys = torch.randn(1, 5, 16)
        values = torch.randn(1, 5, 16)
        mask = torch.tensor([[True, True, False, False, False]])
        _, attn, _ = layer(query, keys, values, mask=mask)
        # Masked positions should have ~0 attention
        assert attn[0, :, :, 2:].sum().item() < 1e-6

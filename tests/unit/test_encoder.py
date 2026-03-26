"""Tests for encoder components."""

import torch

from chess_master.encoder.backbone import EncoderBackbone
from chess_master.encoder.projections import PolicyValueProjection, RetrievalProjection
from chess_master.encoder.short_term import ShortTermEncoder


class TestEncoderBackbone:
    def test_output_shapes(self, tiny_config):
        enc = EncoderBackbone(
            d_model=tiny_config.d_model,
            n_layers=tiny_config.n_layers,
            n_heads=tiny_config.n_heads,
            d_ff=tiny_config.d_ff,
            dropout=tiny_config.dropout,
        )
        x = torch.randn(2, 18, 8, 8)
        z_global, spatial = enc(x)
        assert z_global.shape == (2, tiny_config.d_model)
        assert spatial.shape == (2, 64, tiny_config.d_model)

    def test_with_short_term_context(self, tiny_config):
        enc = EncoderBackbone(
            d_model=tiny_config.d_model,
            n_layers=tiny_config.n_layers,
            n_heads=tiny_config.n_heads,
            d_ff=tiny_config.d_ff,
        )
        x = torch.randn(2, 18, 8, 8)
        ctx = torch.randn(2, 4, tiny_config.d_model)
        z_global, spatial = enc(x, short_term_ctx=ctx)
        assert z_global.shape == (2, tiny_config.d_model)
        assert spatial.shape == (2, 64, tiny_config.d_model)

    def test_gradients_flow(self, tiny_config):
        enc = EncoderBackbone(
            d_model=tiny_config.d_model,
            n_layers=tiny_config.n_layers,
            n_heads=tiny_config.n_heads,
            d_ff=tiny_config.d_ff,
        )
        x = torch.randn(2, 18, 8, 8)
        z_global, _ = enc(x)
        z_global.sum().backward()
        assert enc.input_proj.weight.grad is not None


class TestProjections:
    def test_pv_projection_shape(self, tiny_config):
        proj = PolicyValueProjection(tiny_config.d_model)
        z = torch.randn(2, tiny_config.d_model)
        out = proj(z)
        assert out.shape == (2, tiny_config.d_model)

    def test_retrieval_projection_shape(self, tiny_config):
        proj = RetrievalProjection(tiny_config.d_model, tiny_config.retrieval_dim)
        z = torch.randn(2, tiny_config.d_model)
        out = proj(z)
        assert out.shape == (2, tiny_config.retrieval_dim)

    def test_projections_differ(self, tiny_config):
        """The two projections should produce different outputs."""
        pv = PolicyValueProjection(tiny_config.d_model)
        ret = RetrievalProjection(tiny_config.d_model, tiny_config.retrieval_dim)
        z = torch.randn(2, tiny_config.d_model)
        # They have different output dims, so they must differ
        assert pv(z).shape != ret(z).shape


class TestShortTermEncoder:
    def test_output_shape(self, tiny_config):
        enc = ShortTermEncoder(
            d_model=tiny_config.d_model,
            max_window=tiny_config.short_term_window,
        )
        B, S = 2, 4
        moves = torch.randint(0, 100, (B, S))
        values = torch.randn(B, S)
        deltas = torch.randn(B, S)
        side = torch.ones(B, S)
        out = enc(moves, values, deltas, side)
        assert out.shape == (B, S, tiny_config.d_model)

    def test_masking(self, tiny_config):
        enc = ShortTermEncoder(
            d_model=tiny_config.d_model,
            max_window=tiny_config.short_term_window,
        )
        B, S = 2, 4
        moves = torch.randint(0, 100, (B, S))
        values = torch.randn(B, S)
        deltas = torch.randn(B, S)
        side = torch.ones(B, S)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        out = enc(moves, values, deltas, side, mask=mask)
        # Masked positions should be zeroed
        assert (out[0, 2:] == 0).all()
        assert (out[1, 3:] == 0).all()

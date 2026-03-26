"""Tests for the confidence head."""

import pytest
import torch

from chess_master.heads.confidence import ConfidenceHead


@pytest.fixture
def confidence_head():
    return ConfidenceHead(d_model=256, retrieval_dim=128)


@pytest.fixture
def sample_inputs():
    B = 4
    N = 8
    z_global = torch.randn(B, 256)
    retrieved_context = torch.randn(B, 128)
    similarity_scores = torch.randn(B, N)
    return z_global, retrieved_context, similarity_scores


class TestConfidenceHead:
    def test_output_shape(self, confidence_head, sample_inputs):
        """Output should be [B, 1]."""
        z_global, retrieved_context, similarity_scores = sample_inputs
        out = confidence_head(z_global, retrieved_context, similarity_scores)
        assert out.shape == (4, 1)

    def test_output_range(self, confidence_head, sample_inputs):
        """Output should be in [0, 1] due to sigmoid."""
        z_global, retrieved_context, similarity_scores = sample_inputs
        out = confidence_head(z_global, retrieved_context, similarity_scores)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_gradient_flow(self, confidence_head, sample_inputs):
        """Gradients should flow back through all inputs."""
        z_global, retrieved_context, similarity_scores = sample_inputs
        z_global.requires_grad_(True)
        retrieved_context.requires_grad_(True)

        out = confidence_head(z_global, retrieved_context, similarity_scores)
        loss = out.sum()
        loss.backward()

        assert z_global.grad is not None
        assert z_global.grad.abs().sum() > 0
        assert retrieved_context.grad is not None
        assert retrieved_context.grad.abs().sum() > 0

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        head = ConfidenceHead(d_model=128, retrieval_dim=64)
        for B in [1, 2, 8, 16]:
            z = torch.randn(B, 128)
            r = torch.randn(B, 64)
            s = torch.randn(B, 4)
            out = head(z, r, s)
            assert out.shape == (B, 1)
            assert (out >= 0.0).all()
            assert (out <= 1.0).all()

    def test_different_memory_sizes(self, confidence_head):
        """Should work with different numbers of memory entries."""
        B = 4
        z = torch.randn(B, 256)
        r = torch.randn(B, 128)
        for N in [1, 4, 16, 64]:
            s = torch.randn(B, N)
            out = confidence_head(z, r, s)
            assert out.shape == (B, 1)

    def test_different_model_dims(self):
        """Should work with various d_model and retrieval_dim."""
        for d_model, retrieval_dim in [(64, 32), (128, 64), (512, 256)]:
            head = ConfidenceHead(d_model=d_model, retrieval_dim=retrieval_dim)
            B = 2
            out = head(
                torch.randn(B, d_model),
                torch.randn(B, retrieval_dim),
                torch.randn(B, 8),
            )
            assert out.shape == (B, 1)

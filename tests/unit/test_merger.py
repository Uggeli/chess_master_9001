"""Tests for heuristic and learned mergers."""

import pytest
import torch

from chess_master.heads.confidence import ConfidenceHead
from chess_master.merger.heuristic import HeuristicMerger
from chess_master.merger.learned import LearnedMerger


class TestHeuristicMerger:
    def test_alpha_high_similarity(self):
        """High similarity should produce low alpha (trust memory)."""
        merger = HeuristicMerger(similarity_threshold=0.7, alpha_high=0.95, alpha_low=0.5)
        # High similarity scores
        sim = torch.tensor([[0.9, 0.8, 0.7]])
        alpha = merger.compute_alpha(sim)
        # Max sim is 0.9, so alpha should be closer to alpha_low
        assert alpha.item() < 0.7

    def test_alpha_low_similarity(self):
        """Low similarity should produce high alpha (trust model)."""
        merger = HeuristicMerger(similarity_threshold=0.7, alpha_high=0.95, alpha_low=0.5)
        # Low similarity scores
        sim = torch.tensor([[0.1, 0.05, 0.2]])
        alpha = merger.compute_alpha(sim)
        # Max sim is 0.2, so alpha should be closer to alpha_high
        assert alpha.item() > 0.8

    def test_alpha_shape(self):
        """Alpha should be [B, 1]."""
        merger = HeuristicMerger()
        sim = torch.randn(8, 16)
        alpha = merger.compute_alpha(sim)
        assert alpha.shape == (8, 1)

    def test_alpha_range(self):
        """Alpha should be in [alpha_low, alpha_high]."""
        merger = HeuristicMerger(alpha_high=0.95, alpha_low=0.5)
        sim = torch.rand(32, 8)
        alpha = merger.compute_alpha(sim)
        assert (alpha >= 0.5 - 1e-6).all()
        assert (alpha <= 0.95 + 1e-6).all()

    def test_merge_output_shape(self):
        """Merged output should match input logit shape."""
        merger = HeuristicMerger()
        B, A, N = 4, 4672, 8
        model_logits = torch.randn(B, A)
        memory_logits = torch.randn(B, A)
        sim = torch.rand(B, N)
        out = merger.merge(model_logits, memory_logits, sim)
        assert out.shape == (B, A)

    def test_merge_interpolation(self):
        """Merge should be an interpolation of model and memory logits."""
        merger = HeuristicMerger()
        B, A = 2, 100
        model_logits = torch.ones(B, A) * 10
        memory_logits = torch.ones(B, A) * 0
        sim = torch.rand(B, 8)
        out = merger.merge(model_logits, memory_logits, sim)
        # Should be between 0 and 10
        assert (out >= 0.0 - 1e-6).all()
        assert (out <= 10.0 + 1e-6).all()


class TestLearnedMerger:
    @pytest.fixture
    def merger(self):
        confidence_head = ConfidenceHead(d_model=256, retrieval_dim=128)
        return LearnedMerger(confidence_head)

    def test_output_shape(self, merger):
        """Output logits should match input shape."""
        B, A, N = 4, 4672, 8
        model_logits = torch.randn(B, A)
        memory_logits = torch.randn(B, A)
        z_global = torch.randn(B, 256)
        retrieved = torch.randn(B, 128)
        sim = torch.randn(B, N)
        blended, confidence = merger(
            model_logits, memory_logits, z_global, retrieved, sim,
        )
        assert blended.shape == (B, A)
        assert confidence.shape == (B, 1)

    def test_confidence_range(self, merger):
        """Confidence should be in [0, 1]."""
        B = 8
        _, confidence = merger(
            torch.randn(B, 4672),
            torch.randn(B, 4672),
            torch.randn(B, 256),
            torch.randn(B, 128),
            torch.randn(B, 8),
        )
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0).all()

    def test_gradient_flow(self, merger):
        """Gradients should flow through the learned merger."""
        B = 4
        model_logits = torch.randn(B, 4672, requires_grad=True)
        memory_logits = torch.randn(B, 4672, requires_grad=True)
        z_global = torch.randn(B, 256, requires_grad=True)
        retrieved = torch.randn(B, 128, requires_grad=True)
        sim = torch.randn(B, 8)

        blended, confidence = merger(
            model_logits, memory_logits, z_global, retrieved, sim,
        )
        loss = blended.sum() + confidence.sum()
        loss.backward()

        assert model_logits.grad is not None
        assert z_global.grad is not None
        assert retrieved.grad is not None
        # Check confidence head parameters have gradients
        for p in merger.confidence_head.parameters():
            assert p.grad is not None

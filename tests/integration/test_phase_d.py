"""Integration tests for Phase D (learned memory confidence)."""

import pytest
import torch

from chess_master.config import ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.types import Phase


@pytest.fixture
def model_config():
    return ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        retrieval_dim=32,
    )


@pytest.fixture
def phase_d_model(model_config):
    return ChessMaster9001(model_config, active_phase=Phase.D)


@pytest.fixture
def phase_a_model(model_config):
    return ChessMaster9001(model_config, active_phase=Phase.A)


@pytest.fixture
def sample_inputs(model_config):
    B = 2
    N = 4
    board = torch.randn(B, 18, 8, 8)
    legal_mask = torch.ones(B, 4672)
    mem_keys = torch.randn(B, N, model_config.retrieval_dim)
    mem_vals = torch.randn(B, N, model_config.retrieval_dim)
    mem_mask = torch.ones(B, N, dtype=torch.bool)
    return board, legal_mask, mem_keys, mem_vals, mem_mask


class TestPhaseDModel:
    def test_phase_d_produces_confidence(self, phase_d_model, sample_inputs):
        """Phase D model should produce confidence scores in retrieval_info."""
        board, legal_mask, mem_keys, mem_vals, mem_mask = sample_inputs
        output = phase_d_model(
            board, legal_mask,
            memory_keys=mem_keys,
            memory_values=mem_vals,
            memory_mask=mem_mask,
        )
        assert output.retrieval_info is not None
        assert "confidence" in output.retrieval_info
        confidence = output.retrieval_info["confidence"]
        assert confidence.shape == (2, 1)
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0).all()

    def test_phase_d_has_memory_logits(self, phase_d_model, sample_inputs):
        """Phase D should produce memory_logits in retrieval_info."""
        board, legal_mask, mem_keys, mem_vals, mem_mask = sample_inputs
        output = phase_d_model(
            board, legal_mask,
            memory_keys=mem_keys,
            memory_values=mem_vals,
            memory_mask=mem_mask,
        )
        assert "memory_logits" in output.retrieval_info
        assert output.retrieval_info["memory_logits"].shape == (2, 4672)

    def test_phase_gating_no_confidence_in_phase_a(self, phase_a_model, sample_inputs):
        """Phase A model should not have confidence head."""
        assert not hasattr(phase_a_model, "confidence_head")
        assert not hasattr(phase_a_model, "learned_merger")

    def test_phase_gating_confidence_in_phase_d(self, phase_d_model):
        """Phase D model should have confidence head and learned merger."""
        assert hasattr(phase_d_model, "confidence_head")
        assert hasattr(phase_d_model, "learned_merger")
        assert hasattr(phase_d_model, "memory_policy_proj")

    def test_phase_d_no_memory_no_confidence(self, phase_d_model):
        """Without memory, Phase D should not produce confidence."""
        board = torch.randn(2, 18, 8, 8)
        output = phase_d_model(board)
        assert output.retrieval_info is None

    def test_full_forward_pass_with_memory(self, phase_d_model, sample_inputs):
        """Full forward pass with memory should produce valid outputs."""
        board, legal_mask, mem_keys, mem_vals, mem_mask = sample_inputs
        output = phase_d_model(
            board, legal_mask,
            memory_keys=mem_keys,
            memory_values=mem_vals,
            memory_mask=mem_mask,
        )
        # Standard outputs
        assert output.policy_logits.shape == (2, 4672)
        assert output.value.shape == (2, 1)
        # Confidence outputs
        assert output.retrieval_info["confidence"].shape == (2, 1)

    def test_gradient_flow_through_confidence(self, phase_d_model, sample_inputs):
        """Gradients should flow through the confidence head during training."""
        board, legal_mask, mem_keys, mem_vals, mem_mask = sample_inputs
        output = phase_d_model(
            board, legal_mask,
            memory_keys=mem_keys,
            memory_values=mem_vals,
            memory_mask=mem_mask,
        )
        loss = output.policy_logits.sum()
        loss.backward()

        # Check confidence head has gradients
        for p in phase_d_model.confidence_head.parameters():
            assert p.grad is not None

    def test_phase_c_no_confidence(self, model_config):
        """Phase C model should not have confidence head."""
        model = ChessMaster9001(model_config, active_phase=Phase.C)
        assert not hasattr(model, "confidence_head")

"""Integration test: full model forward and backward pass."""

import torch

from chess_master.config import ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.types import NUM_ACTIONS, Phase


class TestForwardPass:
    def test_basic_forward(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        board_tensor = torch.randn(2, 18, 8, 8)
        output = model(board_tensor)
        assert output.policy_logits.shape == (2, NUM_ACTIONS)
        assert output.value.shape == (2, 1)

    def test_with_legal_mask(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        board_tensor = torch.randn(2, 18, 8, 8)
        legal_mask = torch.zeros(2, NUM_ACTIONS)
        legal_mask[:, :20] = 1.0
        output = model(board_tensor, legal_mask=legal_mask)
        assert (output.policy_logits[:, 20:] == float("-inf")).all()

    def test_with_memory(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        board_tensor = torch.randn(2, 18, 8, 8)
        mem_keys = torch.randn(2, 10, tiny_config.retrieval_dim)
        mem_vals = torch.randn(2, 10, tiny_config.retrieval_dim)
        output = model(
            board_tensor,
            memory_keys=mem_keys,
            memory_values=mem_vals,
        )
        assert output.retrieval_info is not None
        assert "similarity_scores" in output.retrieval_info

    def test_backward_pass(self, tiny_config):
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        board_tensor = torch.randn(2, 18, 8, 8)
        mem_keys = torch.randn(2, 5, tiny_config.retrieval_dim)
        mem_vals = torch.randn(2, 5, tiny_config.retrieval_dim)
        output = model(board_tensor, memory_keys=mem_keys, memory_values=mem_vals)
        loss = output.policy_logits.sum() + output.value.sum()
        loss.backward()
        # Check gradients exist for components used in forward pass
        # (short_term_encoder is not used when no short-term context is provided)
        skip_prefixes = ("short_term_encoder.",)
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith(skip_prefixes):
                assert param.grad is not None, f"No gradient for {name}"

    def test_single_sample(self, tiny_config):
        """Model should work with batch size 1."""
        model = ChessMaster9001(tiny_config, active_phase=Phase.A)
        board_tensor = torch.randn(1, 18, 8, 8)
        output = model(board_tensor)
        assert output.policy_logits.shape == (1, NUM_ACTIONS)

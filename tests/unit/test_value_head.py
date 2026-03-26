"""Tests for value head."""

import torch

from chess_master.heads.value import ValueHead


class TestValueHead:
    def test_output_shape(self):
        head = ValueHead(d_model=32, use_memory=True)
        z = torch.randn(2, 32)
        value = head(z)
        assert value.shape == (2, 1)

    def test_output_range(self):
        head = ValueHead(d_model=32, use_memory=False)
        z = torch.randn(100, 32)
        value = head(z)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_with_memory_context(self):
        head = ValueHead(d_model=32, use_memory=True)
        z = torch.randn(2, 32)
        mem = torch.randn(2, 32)
        value = head(z, memory_context=mem)
        assert value.shape == (2, 1)

    def test_without_memory_flag(self):
        head = ValueHead(d_model=32, use_memory=False)
        z = torch.randn(2, 32)
        value = head(z)
        assert value.shape == (2, 1)

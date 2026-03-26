"""Shared test fixtures."""

import chess
import pytest
import torch

from chess_master.config import ModelConfig, Config
from chess_master.types import MemoryEntry


@pytest.fixture
def start_board():
    """Starting position board."""
    return chess.Board()


@pytest.fixture
def midgame_board():
    """A typical middlegame position."""
    return chess.Board("r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")


@pytest.fixture
def endgame_board():
    """A simple endgame position."""
    return chess.Board("8/5k2/8/8/8/8/4K3/4R3 w - - 0 1")


@pytest.fixture
def tiny_config():
    """Small model config for fast testing."""
    return ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        memory_size=100,
        retrieval_k=4,
        retrieval_dim=16,
        short_term_window=4,
    )


@pytest.fixture
def full_config(tiny_config):
    """Full Config with tiny model for testing."""
    return Config(
        model=tiny_config,
        device="cpu",
        run_dir="/tmp/chess_master_test_run",
    )


@pytest.fixture
def sample_memory_entries():
    """Sample memory entries for testing."""
    entries = []
    for i in range(20):
        entries.append(MemoryEntry(
            state_emb=torch.randn(16),  # retrieval_dim=16 for tiny config
            move_idx=i % 100,
            value=0.5 - (i % 3) * 0.5,
            outcome=1.0 if i % 3 == 0 else (-1.0 if i % 3 == 1 else 0.0),
            source="stockfish",
            importance=1.0 / (i + 1),
        ))
    return entries

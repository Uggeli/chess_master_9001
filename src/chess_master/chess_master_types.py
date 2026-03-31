"""Shared type aliases and constants."""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor


class Phase(IntEnum):
    A = 1  # One-step predictor
    B = 2  # Opponent reply prediction
    C = 3  # Two-ply expected-value scoring
    D = 4  # Learned memory confidence


# AlphaZero-style action space: 73 planes × 8 × 8
NUM_ACTIONS = 4672
BOARD_SIZE = 8

# Board tensor: 18 planes (6 piece types × 2 colors + extras)
NUM_PIECE_PLANES = 12  # 6 piece types × 2 colors
NUM_EXTRA_PLANES = 6   # castling rights (4) + en passant (1) + side to move (1)
NUM_INPUT_PLANES = NUM_PIECE_PLANES + NUM_EXTRA_PLANES


@dataclass
class ModelOutput:
    """Output from the full model forward pass."""
    policy_logits: Tensor       # [B, NUM_ACTIONS]
    value: Tensor               # [B, 1]
    retrieval_info: dict | None = None  # similarity scores, attention weights, etc.
    opponent_logits: Tensor | None = None  # [B, NUM_ACTIONS] opponent reply prediction


class MemoryEntry(NamedTuple):
    """A single entry in the long-term memory store."""
    state_emb: Tensor    # [d_model]
    move_idx: int
    value: float
    outcome: float       # game outcome: 1.0 win, 0.0 draw, -1.0 loss
    source: str          # 'stockfish', 'self_play', 'correction'
    importance: float

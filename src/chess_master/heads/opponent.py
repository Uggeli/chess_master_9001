"""Opponent reply prediction head: P(b | s, a).

Predicts the distribution over the opponent's response moves given
the afterstate (position after our move is applied).
"""

import torch.nn as nn
from torch import Tensor

from chess_master.chess_master_types import NUM_ACTIONS


class OpponentHead(nn.Module):
    """Predict opponent's reply distribution P(b | s, a).

    Takes the afterstate representation (after our move is applied)
    and predicts what the opponent is likely to play.

    Version A: operates on re-encoded afterstate spatial tokens.
    Same architecture as PolicyHead but conceptually separate — this
    head learns the opponent's move distribution rather than our own.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Process spatial tokens
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Map to action space: 73 move types per source square
        self.action_proj = nn.Linear(d_model, 73)

    def forward(
        self,
        afterstate_spatial: Tensor,
        opponent_legal_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute opponent reply logits.

        Args:
            afterstate_spatial: [B, 64, d_model] spatial tokens of position
                after our move, encoded from the opponent's perspective.
            opponent_legal_mask: [B, 4672] legal moves for opponent
                (1 = legal, 0 = illegal).

        Returns:
            logits: [B, 4672] opponent move logits.
        """
        B = afterstate_spatial.shape[0]

        # Project spatial tokens
        x = self.spatial_proj(afterstate_spatial)  # [B, 64, d_model]

        # Compute per-square move type logits
        per_square = self.action_proj(x)  # [B, 64, 73]

        # Reshape: plane * 64 + square -> [B, 4672]
        logits = per_square.transpose(1, 2).reshape(B, -1)  # [B, 4672]

        # Apply legal move mask
        if opponent_legal_mask is not None:
            logits = logits.masked_fill(opponent_legal_mask == 0, float("-inf"))

        return logits

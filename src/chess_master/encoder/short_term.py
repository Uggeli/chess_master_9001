"""Short-term game buffer encoder.

Encodes recent game events (moves, values, evaluation deltas) into
context tokens that are fed into the transformer backbone alongside
the board spatial tokens.
"""

import torch
import torch.nn as nn
from torch import Tensor

from chess_master.chess_master_types import NUM_ACTIONS


class ShortTermEncoder(nn.Module):
    """Encode short-term game history into context tokens.

    Each game event consists of:
    - move_idx: index into the 4672 action space
    - value: position evaluation from the acting side's perspective
    - value_delta: change in evaluation from previous position
    - side_to_move: 0 or 1

    These are embedded and combined into per-event tokens.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_window: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_window = max_window

        # Embed move index
        self.move_embedding = nn.Embedding(NUM_ACTIONS + 1, d_model, padding_idx=NUM_ACTIONS)

        # Project scalar features (value, value_delta, side_to_move)
        self.scalar_proj = nn.Linear(3, d_model)

        # Combine move embedding and scalar features
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Temporal position encoding
        self.temporal_pos = nn.Embedding(max_window, d_model)

    def forward(
        self,
        move_indices: Tensor,
        values: Tensor,
        value_deltas: Tensor,
        side_to_move: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode short-term game events.

        Args:
            move_indices: [B, S] move indices (padded with NUM_ACTIONS).
            values: [B, S] position values.
            value_deltas: [B, S] evaluation deltas.
            side_to_move: [B, S] side to move (0 or 1).
            mask: [B, S] boolean mask (True = valid event, False = padding).

        Returns:
            [B, S, d_model] context tokens for the transformer backbone.
        """
        B, S = move_indices.shape

        # Embed moves
        move_emb = self.move_embedding(move_indices)  # [B, S, d_model]

        # Stack and project scalar features
        scalars = torch.stack([values, value_deltas, side_to_move], dim=-1)  # [B, S, 3]
        scalar_emb = self.scalar_proj(scalars)  # [B, S, d_model]

        # Combine
        combined = torch.cat([move_emb, scalar_emb], dim=-1)  # [B, S, 2*d_model]
        tokens = self.combine(combined)  # [B, S, d_model]

        # Add temporal position encoding
        positions = torch.arange(S, device=tokens.device)
        tokens = tokens + self.temporal_pos(positions).unsqueeze(0)

        # Zero out padding positions
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        return tokens

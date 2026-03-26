"""Policy head: predicts move distribution over the action space."""

import torch
import torch.nn as nn
from torch import Tensor

from chess_master.types import NUM_ACTIONS, BOARD_SIZE


class PolicyHead(nn.Module):
    """Policy head producing logits over the 4672 action space.

    Uses spatial tokens from the encoder to produce per-square features,
    then maps to the full action space.

    Architecture:
    - 1x1 conv over spatial tokens to reduce channels
    - Flatten and project to action space
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Process spatial tokens
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Map to action space
        # 73 move types per source square, 64 source squares = 4672
        self.action_proj = nn.Linear(d_model, 73)  # per-square: 73 move types

    def forward(
        self,
        spatial_tokens: Tensor,
        legal_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute policy logits.

        Args:
            spatial_tokens: [B, 64, d_model] per-square representations.
            legal_mask: [B, 4672] binary mask (1 = legal, 0 = illegal).
                If provided, illegal moves get -inf logits.

        Returns:
            logits: [B, 4672] move logits.
        """
        B = spatial_tokens.shape[0]

        # Project spatial tokens
        x = self.spatial_proj(spatial_tokens)  # [B, 64, d_model]

        # Compute per-square move type logits
        per_square = self.action_proj(x)  # [B, 64, 73]

        # Reshape: each source square has 73 move types
        # Action index = plane * 64 + square, so we need to transpose
        # per_square is [B, 64_squares, 73_planes]
        # We want [B, 73_planes * 64_squares] = [B, 4672]
        logits = per_square.transpose(1, 2).reshape(B, -1)  # [B, 4672]

        # Apply legal move mask
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))

        return logits

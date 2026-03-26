"""Transformer backbone for board state encoding."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from chess_master.types import NUM_INPUT_PLANES, BOARD_SIZE


class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for 8x8 board."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, d_model, BOARD_SIZE, BOARD_SIZE) * 0.02
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embedding


class EncoderBackbone(nn.Module):
    """Shared transformer backbone.

    Takes a board tensor [B, 18, 8, 8] and produces:
    - z_global: [B, d_model] global board representation
    - spatial_tokens: [B, 64, d_model] per-square token representations
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input planes to d_model channels
        self.input_proj = nn.Conv2d(NUM_INPUT_PLANES, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(d_model)

        # Transformer encoder over 64 spatial tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Global pooling token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, board_tensor: Tensor, short_term_ctx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode a board tensor.

        Args:
            board_tensor: [B, 18, 8, 8] board planes.
            short_term_ctx: [B, S, d_model] optional short-term context tokens.

        Returns:
            z_global: [B, d_model] global representation (from CLS token).
            spatial_tokens: [B, 64, d_model] per-square representations.
        """
        B = board_tensor.shape[0]

        # Project and add positional encoding: [B, d_model, 8, 8]
        x = self.input_proj(board_tensor)
        x = self.pos_encoding(x)

        # Reshape to sequence: [B, 64, d_model]
        spatial = x.flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, spatial], dim=1)  # [B, 65, d_model]

        # Optionally append short-term context tokens
        if short_term_ctx is not None:
            tokens = torch.cat([tokens, short_term_ctx], dim=1)

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        z_global = tokens[:, 0]        # [B, d_model] CLS token
        spatial_tokens = tokens[:, 1:65]  # [B, 64, d_model] board squares

        return z_global, spatial_tokens

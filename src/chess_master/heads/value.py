"""Value head: predicts position evaluation."""

import torch
import torch.nn as nn
from torch import Tensor


class ValueHead(nn.Module):
    """Value head producing a scalar position evaluation.

    Takes the global representation from the encoder and optionally
    the retrieved memory context, and predicts a value in [-1, 1].

    Architecture:
    - MLP with optional memory context concatenation
    - tanh output for bounded [-1, 1] range
    """

    def __init__(self, d_model: int = 256, use_memory: bool = True):
        super().__init__()
        input_dim = d_model * 2 if use_memory else d_model
        self.use_memory = use_memory

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        z_global: Tensor,
        memory_context: Tensor | None = None,
    ) -> Tensor:
        """Compute value prediction.

        Args:
            z_global: [B, d_model] global board representation.
            memory_context: [B, d_model] retrieved memory context (optional).

        Returns:
            value: [B, 1] position evaluation in [-1, 1].
        """
        if self.use_memory and memory_context is not None:
            x = torch.cat([z_global, memory_context], dim=-1)
        else:
            # If memory not available, zero-pad to match expected input dim
            if self.use_memory:
                zeros = torch.zeros_like(z_global)
                x = torch.cat([z_global, zeros], dim=-1)
            else:
                x = z_global

        return self.mlp(x)

"""Projection heads for policy/value and memory retrieval."""

import torch.nn as nn
from torch import Tensor


class PolicyValueProjection(nn.Module):
    """Project backbone output for policy/value computation."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Project global representation for policy/value heads.

        Args:
            z: [B, d_model] from backbone.
        Returns:
            [B, d_model] projected representation.
        """
        return self.proj(z)


class RetrievalProjection(nn.Module):
    """Project backbone output for memory retrieval queries.

    Uses a separate embedding space to prevent policy gradients
    from collapsing the retrieval geometry.
    """

    def __init__(self, d_model: int = 256, retrieval_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, retrieval_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Project global representation for memory retrieval.

        Args:
            z: [B, d_model] from backbone.
        Returns:
            [B, retrieval_dim] retrieval query embedding.
        """
        return self.proj(z)

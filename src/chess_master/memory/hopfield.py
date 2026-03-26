"""Modern Hopfield Network for episodic memory retrieval.

Implements a continuous Hopfield retrieval mechanism where stored patterns
are retrieved via an attention-like operation with exponential interaction.

The key idea: given a query (current state embedding), retrieve a weighted
combination of stored memory values using softmax attention over stored keys.

This is differentiable and can be trained end-to-end.

Reference: Ramsauer et al., "Hopfield Networks is All You Need" (2020)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HopfieldLayer(nn.Module):
    """Hopfield associative memory retrieval.

    Given a query embedding, retrieves relevant information from a set
    of stored key-value pairs using scaled softmax attention.

    The separation temperature β controls retrieval sharpness:
    - high β → sharp retrieval (nearest neighbor)
    - low β → soft retrieval (blended average)
    """

    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        beta_init: float = 1.0,
        learn_beta: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable inverse temperature
        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta_init))
        else:
            self.register_buffer("beta", torch.tensor(beta_init))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Retrieve from memory using Hopfield attention.

        Args:
            query: [B, d] or [B, Q, d] query embeddings.
            keys: [B, N, d] stored memory key embeddings.
            values: [B, N, d] stored memory value embeddings.
            mask: [B, N] optional boolean mask (True = valid, False = ignore).

        Returns:
            retrieved: [B, d] or [B, Q, d] retrieved memory context.
            attention_weights: [B, H, Q, N] attention weights per head.
            similarity_scores: [B, Q, N] raw similarity scores (pre-softmax).
        """
        # Handle unbatched queries
        squeeze_q = False
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, d]
            squeeze_q = True

        B, Q, _ = query.shape
        N = keys.shape[1]
        H = self.n_heads
        D = self.head_dim

        # Project
        q = self.q_proj(query).reshape(B, Q, H, D).transpose(1, 2)  # [B, H, Q, D]
        k = self.k_proj(keys).reshape(B, N, H, D).transpose(1, 2)   # [B, H, N, D]
        v = self.v_proj(values).reshape(B, N, H, D).transpose(1, 2) # [B, H, N, D]

        # Scaled dot-product attention with Hopfield temperature
        scale = self.beta / math.sqrt(D)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Q, N]

        # Store raw similarities (averaged over heads) before masking
        similarity_scores = attn_logits.mean(dim=1).detach()  # [B, Q, N]

        # Apply mask
        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            attn_logits = attn_logits.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, Q, N]
        attn_weights = self.dropout(attn_weights)

        # Retrieve
        retrieved = torch.matmul(attn_weights, v)  # [B, H, Q, D]
        retrieved = retrieved.transpose(1, 2).reshape(B, Q, -1)  # [B, Q, d]
        retrieved = self.out_proj(retrieved)

        if squeeze_q:
            retrieved = retrieved.squeeze(1)       # [B, d]
            similarity_scores = similarity_scores.squeeze(1)  # [B, N]

        return retrieved, attn_weights, similarity_scores

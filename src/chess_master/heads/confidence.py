"""Confidence head: predicts how much to trust retrieved memory."""

import torch
import torch.nn as nn
from torch import Tensor


class ConfidenceHead(nn.Module):
    """Learned confidence head for memory trust estimation.

    Takes retrieval information (similarity scores, query embedding,
    retrieved context) and predicts a confidence score alpha in [0, 1].

    alpha near 1.0 -> trust model predictions
    alpha near 0.0 -> trust memory prior

    Training targets can come from:
    - Whether memory-conditioned predictions were more accurate
    - Retrieval quality metrics
    - Prediction error when trusting memory vs not
    """

    def __init__(self, d_model: int = 256, retrieval_dim: int = 128):
        super().__init__()
        # Input: z_global (d_model) + retrieved_context (retrieval_dim) + similarity features
        # We summarize similarity scores into a fixed-size feature vector
        self.sim_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
        )

        input_dim = d_model + retrieval_dim + 16

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_global: Tensor,
        retrieved_context: Tensor,
        similarity_scores: Tensor,
    ) -> Tensor:
        """Predict confidence alpha in [0, 1].

        Args:
            z_global: [B, d_model] global board representation.
            retrieved_context: [B, retrieval_dim] retrieved memory context.
            similarity_scores: [B, N] similarity scores from retrieval.

        Returns:
            confidence: [B, 1] confidence scores.
        """
        # Summarize similarity scores: use max similarity as a single scalar
        max_sim = similarity_scores.max(dim=-1, keepdim=True).values  # [B, 1]
        sim_features = self.sim_proj(max_sim)  # [B, 16]

        # Concatenate all inputs
        x = torch.cat([z_global, retrieved_context, sim_features], dim=-1)

        return self.mlp(x)

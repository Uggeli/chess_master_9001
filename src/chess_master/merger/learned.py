"""Learned confidence-weighted merger for Phase D."""

import torch.nn as nn
from torch import Tensor

from chess_master.heads.confidence import ConfidenceHead


class LearnedMerger(nn.Module):
    """Learned confidence-weighted merger.

    Uses the confidence head to dynamically blend model predictions
    with memory-based prior based on retrieval quality.

    When confidence is high (near 1.0), trusts the model.
    When confidence is low (near 0.0), trusts memory.
    """

    def __init__(self, confidence_head: ConfidenceHead):
        super().__init__()
        self.confidence_head = confidence_head

    def forward(
        self,
        model_logits: Tensor,
        memory_logits: Tensor,
        z_global: Tensor,
        retrieved_context: Tensor,
        similarity_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Blend predictions using learned confidence.

        Args:
            model_logits: [B, A] model policy logits.
            memory_logits: [B, A] memory-derived policy logits.
            z_global: [B, d_model] global board representation.
            retrieved_context: [B, retrieval_dim] retrieved memory context.
            similarity_scores: [B, N] similarity scores from retrieval.

        Returns:
            blended_logits: [B, A] combined logits.
            confidence: [B, 1] confidence scores used for blending.
        """
        confidence = self.confidence_head(
            z_global, retrieved_context, similarity_scores,
        )  # [B, 1]

        blended = confidence * model_logits + (1.0 - confidence) * memory_logits
        return blended, confidence

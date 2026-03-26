"""Fixed-alpha merger based on retrieval similarity thresholds.

Used in Phases A-C before the learned confidence head (Phase D) replaces it.
"""

import torch
from torch import Tensor


class HeuristicMerger:
    """Fixed-alpha merger based on retrieval similarity thresholds.

    Blends model predictions with memory-based prior:
    p_combined proportional to p_model^alpha * p_memory^(1-alpha)

    In log-space this becomes:
    logits_combined = alpha * model_logits + (1 - alpha) * memory_logits

    Alpha is derived from retrieval quality:
    - High similarity -> lower alpha (trust memory more)
    - Low similarity -> higher alpha (trust model more)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        alpha_high: float = 0.95,
        alpha_low: float = 0.5,
    ):
        self.similarity_threshold = similarity_threshold
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low

    def compute_alpha(self, similarity_scores: Tensor) -> Tensor:
        """Compute per-sample alpha from retrieval similarity.

        Uses the max similarity across memory entries per sample.
        When max similarity > threshold, alpha is low (trust memory).
        When max similarity <= threshold, alpha is high (trust model).

        Args:
            similarity_scores: [B, N] similarity scores from retrieval.

        Returns:
            alpha: [B, 1] blending weights in [alpha_low, alpha_high].
        """
        # Max similarity across memory entries
        max_sim = similarity_scores.max(dim=-1, keepdim=True).values  # [B, 1]

        # Linear interpolation: high similarity -> alpha_low, low similarity -> alpha_high
        # Clamp similarity to [0, 1] for safety
        normalized = torch.clamp(max_sim, 0.0, 1.0)

        # Interpolate: when normalized=1.0 -> alpha_low, when normalized=0.0 -> alpha_high
        alpha = self.alpha_high + (self.alpha_low - self.alpha_high) * normalized

        return alpha

    def merge(
        self,
        model_logits: Tensor,
        memory_logits: Tensor,
        similarity_scores: Tensor,
    ) -> Tensor:
        """Blend model and memory-derived logits.

        Args:
            model_logits: [B, A] model policy logits.
            memory_logits: [B, A] memory-derived policy logits.
            similarity_scores: [B, N] similarity scores from retrieval.

        Returns:
            blended_logits: [B, A] combined logits.
        """
        alpha = self.compute_alpha(similarity_scores)  # [B, 1]
        return alpha * model_logits + (1.0 - alpha) * memory_logits

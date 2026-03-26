"""Loss functions for Chess Master 9001."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chess_master.config import LossWeights


class ChessLoss(nn.Module):
    """Combined loss for policy, value, and retrieval alignment.

    Components:
    - Policy: cross-entropy against Stockfish best move
    - Value: MSE against Stockfish evaluation (normalized to [-1, 1])
    - Retrieval alignment: cosine similarity loss encouraging the retrieval
      projection to produce useful queries (only when memory targets available)
    """

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()

    def policy_loss(self, logits: Tensor, target_move: Tensor) -> Tensor:
        """Cross-entropy loss for policy prediction.

        Args:
            logits: [B, 4672] raw policy logits.
            target_move: [B] target move indices.

        Returns:
            Scalar loss.
        """
        return F.cross_entropy(logits, target_move)

    def value_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        """MSE loss for value prediction.

        Args:
            predicted: [B, 1] predicted values in [-1, 1].
            target: [B] or [B, 1] target values in [-1, 1].

        Returns:
            Scalar loss.
        """
        if target.dim() == 1:
            target = target.unsqueeze(1)
        return F.mse_loss(predicted, target)

    def retrieval_loss(
        self, query_emb: Tensor, positive_emb: Tensor, negative_embs: Tensor | None = None,
    ) -> Tensor:
        """Contrastive retrieval alignment loss.

        Encourages query embeddings to be similar to embeddings of
        positions that share the same best move or similar evaluation.

        Args:
            query_emb: [B, d] query embeddings from retrieval projection.
            positive_emb: [B, d] embeddings of similar positions.
            negative_embs: [B, K, d] embeddings of dissimilar positions (optional).

        Returns:
            Scalar loss.
        """
        # Simple cosine similarity loss (maximize similarity with positive)
        pos_sim = F.cosine_similarity(query_emb, positive_emb, dim=-1)
        loss = 1.0 - pos_sim.mean()

        if negative_embs is not None:
            # Push away from negatives
            neg_sim = F.cosine_similarity(
                query_emb.unsqueeze(1), negative_embs, dim=-1
            )  # [B, K]
            loss = loss + neg_sim.mean()

        return loss

    def confidence_loss(
        self,
        confidence: Tensor,
        model_logits: Tensor,
        memory_logits: Tensor,
        target_move: Tensor,
    ) -> Tensor:
        """Confidence calibration loss.

        Trains the confidence head to predict whether trusting the model
        or trusting memory produces more accurate predictions.

        If memory-conditioned prediction was more accurate, confidence
        should be LOW (trust memory). If model-only prediction was more
        accurate, confidence should be HIGH (trust model).

        Args:
            confidence: [B, 1] predicted confidence scores.
            model_logits: [B, A] model-only policy logits.
            memory_logits: [B, A] memory-derived policy logits.
            target_move: [B] target move indices.

        Returns:
            Scalar loss.
        """
        # Compute per-sample cross-entropy for model and memory predictions
        model_ce = F.cross_entropy(model_logits, target_move, reduction="none")  # [B]
        memory_ce = F.cross_entropy(memory_logits, target_move, reduction="none")  # [B]

        # Target: 1.0 if model was better (lower CE), 0.0 if memory was better
        target_confidence = (model_ce < memory_ce).float().unsqueeze(1)  # [B, 1]

        return F.binary_cross_entropy(confidence, target_confidence)

    def opponent_loss(self, logits: Tensor, target_move: Tensor) -> Tensor:
        """Cross-entropy loss for opponent reply prediction.

        Args:
            logits: [B, 4672] opponent reply logits.
            target_move: [B] target opponent move indices.

        Returns:
            Scalar loss.
        """
        return F.cross_entropy(logits, target_move)

    def forward(
        self,
        policy_logits: Tensor,
        value_pred: Tensor,
        target_move: Tensor,
        target_value: Tensor,
        query_emb: Tensor | None = None,
        positive_emb: Tensor | None = None,
        negative_embs: Tensor | None = None,
        opponent_logits: Tensor | None = None,
        target_opponent_move: Tensor | None = None,
        confidence: Tensor | None = None,
        memory_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Returns:
            Dict with 'total', 'policy', 'value', and optionally
            'retrieval' and 'opponent' losses.
        """
        p_loss = self.policy_loss(policy_logits, target_move)
        v_loss = self.value_loss(value_pred, target_value)

        total = self.weights.policy * p_loss + self.weights.value * v_loss

        losses = {
            "policy": p_loss,
            "value": v_loss,
        }

        if query_emb is not None and positive_emb is not None:
            r_loss = self.retrieval_loss(query_emb, positive_emb, negative_embs)
            total = total + self.weights.retrieval * r_loss
            losses["retrieval"] = r_loss

        if opponent_logits is not None and target_opponent_move is not None:
            o_loss = self.opponent_loss(opponent_logits, target_opponent_move)
            total = total + self.weights.opponent * o_loss
            losses["opponent"] = o_loss

        if confidence is not None and memory_logits is not None:
            c_loss = self.confidence_loss(
                confidence, policy_logits, memory_logits, target_move,
            )
            total = total + self.weights.confidence * c_loss
            losses["confidence"] = c_loss

        losses["total"] = total
        return losses

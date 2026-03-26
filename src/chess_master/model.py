"""Top-level Chess Master 9001 model.

Composes encoder, memory, and head modules into a single nn.Module.
Phase-gated: inactive components are not instantiated or computed.
"""

import torch
import torch.nn as nn
from torch import Tensor

from chess_master.config import ModelConfig
from chess_master.encoder.backbone import EncoderBackbone
from chess_master.encoder.projections import PolicyValueProjection, RetrievalProjection
from chess_master.encoder.short_term import ShortTermEncoder
from chess_master.heads.confidence import ConfidenceHead
from chess_master.heads.opponent import OpponentHead
from chess_master.heads.policy import PolicyHead
from chess_master.heads.value import ValueHead
from chess_master.memory.hopfield import HopfieldLayer
from chess_master.merger.learned import LearnedMerger
from chess_master.types import ModelOutput, Phase


class ChessMaster9001(nn.Module):
    """Memory-assisted Bayesian chess model.

    Composes:
    - Transformer encoder backbone
    - Dual projection heads (policy/value vs retrieval)
    - Short-term game context encoder
    - Hopfield episodic memory retrieval
    - Policy and value heads

    Higher-phase components (opponent model, planner, learned merger)
    are only instantiated when active_phase >= their phase.
    """

    def __init__(self, config: ModelConfig, active_phase: int = Phase.A):
        super().__init__()
        self.config = config
        self.active_phase = active_phase

        # Core encoder
        self.backbone = EncoderBackbone(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Dual projections
        self.pv_projection = PolicyValueProjection(config.d_model)
        self.retrieval_projection = RetrievalProjection(
            config.d_model, config.retrieval_dim
        )

        # Short-term context encoder
        self.short_term_encoder = ShortTermEncoder(
            d_model=config.d_model,
            max_window=config.short_term_window,
        )

        # Hopfield memory retrieval
        self.hopfield = HopfieldLayer(
            dim=config.retrieval_dim,
            n_heads=4,
            dropout=config.dropout,
        )

        # Project retrieved memory back to d_model for value head
        self.memory_proj = nn.Linear(config.retrieval_dim, config.d_model)

        # Heads
        self.policy_head = PolicyHead(config.d_model)
        self.value_head = ValueHead(config.d_model, use_memory=True)

        # Phase B+: opponent reply head
        if active_phase >= Phase.B:
            self.opponent_head = OpponentHead(config.d_model)

        # Phase D: confidence head and learned merger
        if active_phase >= Phase.D:
            self.confidence_head = ConfidenceHead(
                d_model=config.d_model,
                retrieval_dim=config.retrieval_dim,
            )
            self.learned_merger = LearnedMerger(self.confidence_head)
            # Project retrieved memory to policy logit space for merging
            self.memory_policy_proj = nn.Sequential(
                nn.Linear(config.retrieval_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 4672),
            )

    def forward(
        self,
        board_tensor: Tensor,
        legal_mask: Tensor | None = None,
        # Short-term context (optional)
        st_move_indices: Tensor | None = None,
        st_values: Tensor | None = None,
        st_value_deltas: Tensor | None = None,
        st_side_to_move: Tensor | None = None,
        st_mask: Tensor | None = None,
        # Memory keys/values from the store (optional)
        memory_keys: Tensor | None = None,
        memory_values: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> ModelOutput:
        """Full forward pass.

        Args:
            board_tensor: [B, 18, 8, 8] encoded board state.
            legal_mask: [B, 4672] legal move mask.
            st_*: Short-term game context tensors (all [B, S]).
            memory_keys: [B, N, retrieval_dim] stored memory embeddings.
            memory_values: [B, N, retrieval_dim] stored memory values.
            memory_mask: [B, N] mask for valid memory entries.

        Returns:
            ModelOutput with policy_logits, value, and retrieval_info.
        """
        # 1. Encode short-term context if provided
        short_term_ctx = None
        if st_move_indices is not None:
            short_term_ctx = self.short_term_encoder(
                st_move_indices, st_values, st_value_deltas,
                st_side_to_move, st_mask,
            )

        # 2. Run backbone
        z_global, spatial_tokens = self.backbone(board_tensor, short_term_ctx)

        # 3. Dual projections
        z_pv = self.pv_projection(z_global)
        z_ret = self.retrieval_projection(z_global)

        # 4. Memory retrieval (if memory is provided)
        retrieval_info = None
        memory_context = None
        if memory_keys is not None and memory_values is not None:
            retrieved, attn_weights, sim_scores = self.hopfield(
                z_ret, memory_keys, memory_values, memory_mask
            )
            memory_context = self.memory_proj(retrieved)
            retrieval_info = {
                "attention_weights": attn_weights,
                "similarity_scores": sim_scores,
                "query_embedding": z_ret.detach(),
                "retrieved": retrieved.detach(),
            }

        # 5. Policy head (uses spatial tokens)
        policy_logits = self.policy_head(spatial_tokens, legal_mask)

        # 6. Phase D: learned confidence merger
        if (
            self.active_phase >= Phase.D
            and memory_keys is not None
            and memory_values is not None
            and retrieval_info is not None
        ):
            # Generate memory-derived policy logits
            memory_logits = self.memory_policy_proj(retrieved)
            if legal_mask is not None:
                memory_logits = memory_logits.masked_fill(legal_mask == 0, float("-inf"))

            # Use learned merger to blend
            policy_logits, confidence = self.learned_merger(
                policy_logits, memory_logits,
                z_global, retrieved, sim_scores,
            )

            retrieval_info["confidence"] = confidence
            retrieval_info["memory_logits"] = memory_logits.detach()

        # 7. Value head (uses global repr + memory context)
        value = self.value_head(z_pv, memory_context)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            retrieval_info=retrieval_info,
        )

    def predict_opponent_reply(
        self,
        afterstate_tensor: Tensor,
        opponent_legal_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict opponent's reply distribution for a given afterstate.

        Re-encodes the afterstate through the backbone and runs the opponent
        head on the resulting spatial tokens (Version A: shared encoder).

        Args:
            afterstate_tensor: [B, 18, 8, 8] board state after our move,
                encoded from the opponent's perspective.
            opponent_legal_mask: [B, 4672] legal moves for opponent.

        Returns:
            logits: [B, 4672] opponent move logits.

        Raises:
            RuntimeError: If opponent head is not available (phase < B).
        """
        if not hasattr(self, "opponent_head"):
            raise RuntimeError(
                "Opponent head not available. Set active_phase >= Phase.B."
            )

        # Re-encode the afterstate through the shared backbone
        _z_global, spatial_tokens = self.backbone(afterstate_tensor)

        # Predict opponent reply
        return self.opponent_head(spatial_tokens, opponent_legal_mask)

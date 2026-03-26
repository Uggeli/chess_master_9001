"""Two-ply expected-value move selection.

For each candidate move, simulates opponent replies and scores
the resulting positions to compute Q(s,a).
"""

import chess
import torch
import torch.nn.functional as F
from torch import Tensor

from chess_master.board.moves import index_to_move, legal_move_mask, move_to_index
from chess_master.board.tensor import board_to_tensor
from chess_master.types import NUM_ACTIONS


class TwoPlyPlanner:
    """Two-ply expected value move selection.

    For each of top-k candidate moves:
    1. Apply move to get afterstate
    2. Predict opponent replies P(b|s,a)
    3. For top-r replies, evaluate resulting position
    4. Score: Q(s,a) = sum_b P(b|s,a) * V(s_ab)
    """

    def __init__(self, model, k: int = 5, r: int = 3, device: str = "cpu"):
        """Initialize the two-ply planner.

        Args:
            model: A ChessMaster9001 model (must have active_phase >= Phase.B).
            k: Number of top candidate moves to evaluate.
            r: Number of top opponent replies to consider per candidate.
            device: Device for tensor computations.
        """
        self.model = model
        self.k = k
        self.r = r
        self.device = torch.device(device)

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """Select best move using two-ply lookahead.

        Returns (best_move, info_dict) where info_dict has:
        - candidates: list of {move, q_value, opponent_replies: [{move, prob, value}]}
        - selected_move: the chosen move
        - one_ply_move: what one-step policy would have chosen (for comparison)
        """
        # Get policy distribution for our moves
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        mask = legal_move_mask(board).unsqueeze(0).to(self.device)
        output = self.model(board_tensor, legal_mask=mask)

        policy_logits = output.policy_logits[0]  # [NUM_ACTIONS]
        policy_probs = F.softmax(policy_logits, dim=-1)

        # One-ply baseline: simple argmax
        one_ply_idx = policy_logits.argmax().item()
        one_ply_move = index_to_move(one_ply_idx, board)

        # Get top-k candidate moves
        num_legal = int((mask[0] > 0).sum().item())
        actual_k = min(self.k, num_legal)
        topk_vals, topk_indices = torch.topk(policy_probs, actual_k)

        candidates = []
        for i in range(actual_k):
            move_idx = topk_indices[i].item()
            try:
                move = index_to_move(move_idx, board)
            except ValueError:
                continue

            # Apply move to get afterstate
            afterstate_board = board.copy()
            afterstate_board.push(move)

            # Check if game is over after our move
            if afterstate_board.is_game_over():
                result = afterstate_board.result()
                if result == "1-0":
                    q_value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    q_value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    q_value = 0.0
                candidates.append({
                    "move": move.uci(),
                    "q_value": q_value,
                    "opponent_replies": [],
                })
                continue

            # Encode afterstate from opponent's perspective
            afterstate_tensor = board_to_tensor(afterstate_board).unsqueeze(0).to(self.device)
            opp_mask = legal_move_mask(afterstate_board).unsqueeze(0).to(self.device)

            # Get opponent reply distribution
            opp_logits = self.model.predict_opponent_reply(afterstate_tensor, opp_mask)
            opp_probs = F.softmax(opp_logits[0], dim=-1)

            # Get top-r opponent replies
            num_opp_legal = int((opp_mask[0] > 0).sum().item())
            actual_r = min(self.r, num_opp_legal)
            opp_topk_vals, opp_topk_indices = torch.topk(opp_probs, actual_r)

            # Renormalize over the top-r replies
            opp_topk_normalized = opp_topk_vals / opp_topk_vals.sum()

            # Evaluate each resulting position
            opponent_replies = []
            q_value = 0.0
            for j in range(actual_r):
                opp_move_idx = opp_topk_indices[j].item()
                opp_prob = opp_topk_normalized[j].item()

                try:
                    opp_move = index_to_move(opp_move_idx, afterstate_board)
                except ValueError:
                    continue

                # Apply opponent's reply
                result_board = afterstate_board.copy()
                result_board.push(opp_move)

                if result_board.is_game_over():
                    result = result_board.result()
                    if result == "1-0":
                        v = 1.0 if board.turn == chess.WHITE else -1.0
                    elif result == "0-1":
                        v = -1.0 if board.turn == chess.WHITE else 1.0
                    else:
                        v = 0.0
                else:
                    # Evaluate resulting position
                    result_tensor = board_to_tensor(result_board).unsqueeze(0).to(self.device)
                    result_mask = legal_move_mask(result_board).unsqueeze(0).to(self.device)
                    result_output = self.model(result_tensor, legal_mask=result_mask)
                    # Value is from the side-to-move's perspective (which is us again)
                    v = result_output.value[0, 0].item()

                q_value += opp_prob * v
                opponent_replies.append({
                    "move": opp_move.uci(),
                    "prob": opp_prob,
                    "value": v,
                })

            candidates.append({
                "move": move.uci(),
                "q_value": q_value,
                "opponent_replies": opponent_replies,
            })

        # Select move with highest Q-value
        if not candidates:
            # Fallback to one-ply
            return one_ply_move, {
                "candidates": [],
                "selected_move": one_ply_move.uci(),
                "one_ply_move": one_ply_move.uci(),
            }

        best_candidate = max(candidates, key=lambda c: c["q_value"])
        best_move = chess.Move.from_uci(best_candidate["move"])

        # Validate the best move is legal (should always be true)
        if best_move not in board.legal_moves:
            best_move = one_ply_move

        info = {
            "candidates": candidates,
            "selected_move": best_move.uci(),
            "one_ply_move": one_ply_move.uci(),
        }

        return best_move, info

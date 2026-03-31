"""Recursive single-line trajectory projection.

Projects continuation lines forward by alternating our best move
and the opponent's most likely reply, then scores each trajectory.
"""

import chess
import torch
import torch.nn.functional as F
from torch import Tensor

from chess_master.board.moves import index_to_move, legal_move_mask, move_to_index
from chess_master.board.tensor import board_to_tensor
from chess_master.chess_master_types import NUM_ACTIONS


class ProjectionPlanner:
    """Recursive single-line trajectory projection.

    For each root candidate:
    - Force that move
    - Recursively alternate: our best move, opponent's best reply
    - Continue for a fixed horizon
    - Score: endpoint value + value trend
    """

    def __init__(self, model, k: int = 5, horizon: int = 10, device: str = "cpu"):
        """Initialize the projection planner.

        Args:
            model: A ChessMaster9001 model (must have active_phase >= Phase.B).
            k: Number of top candidate moves to evaluate at the root.
            horizon: Number of half-moves (plies) to project forward.
            device: Device for tensor computations.
        """
        self.model = model
        self.k = k
        self.horizon = horizon
        self.device = torch.device(device)

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """Select best move using projected trajectory comparison.

        Returns (best_move, info_dict) with projected lines for each candidate.
        """
        # Get policy distribution for our moves
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        mask = legal_move_mask(board).unsqueeze(0).to(self.device)
        output = self.model(board_tensor, legal_mask=mask)

        policy_logits = output.policy_logits[0]
        policy_probs = F.softmax(policy_logits, dim=-1)

        # One-ply baseline
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

            # Project a line starting with this move
            proj_board = board.copy()
            proj_board.push(move)

            line_info = self._project_line(proj_board, self.horizon - 1)
            line_info["root_move"] = move.uci()
            line_info["moves"].insert(0, move.uci())

            candidates.append(line_info)

        if not candidates:
            return one_ply_move, {
                "candidates": [],
                "selected_move": one_ply_move.uci(),
                "one_ply_move": one_ply_move.uci(),
            }

        # Score each candidate: endpoint_value + value_trend_bonus
        best_candidate = max(candidates, key=lambda c: c["score"])
        best_move = chess.Move.from_uci(best_candidate["root_move"])

        if best_move not in board.legal_moves:
            best_move = one_ply_move

        info = {
            "candidates": candidates,
            "selected_move": best_move.uci(),
            "one_ply_move": one_ply_move.uci(),
        }

        return best_move, info

    @torch.no_grad()
    def _project_line(self, board: chess.Board, horizon: int) -> dict:
        """Project a single continuation line.

        Alternates between our best move and opponent's best reply
        for the given horizon (in half-moves).

        Returns dict with:
        - moves: list of UCI move strings in the projected line
        - values: list of position values at each ply
        - endpoint_value: value at the end of the line
        - value_trend: slope of value over the line
        - score: combined score (endpoint_value + 0.1 * value_trend)
        """
        moves = []
        values = []
        current_board = board.copy()
        # Track whose perspective we started from (the root player)
        root_turn = not board.turn  # The player who made the root move

        for ply in range(horizon):
            if current_board.is_game_over():
                # Score the terminal position
                result = current_board.result()
                if result == "1-0":
                    terminal_v = 1.0 if root_turn == chess.WHITE else -1.0
                elif result == "0-1":
                    terminal_v = -1.0 if root_turn == chess.WHITE else 1.0
                else:
                    terminal_v = 0.0
                values.append(terminal_v)
                break

            bt = board_to_tensor(current_board).unsqueeze(0).to(self.device)
            m = legal_move_mask(current_board).unsqueeze(0).to(self.device)

            is_our_turn = (current_board.turn == root_turn)

            if is_our_turn:
                # Our turn: pick the best policy move
                out = self.model(bt, legal_mask=m)
                v = out.value[0, 0].item()
                values.append(v)

                best_idx = out.policy_logits[0].argmax().item()
                try:
                    best_move = index_to_move(best_idx, current_board)
                except ValueError:
                    break
                moves.append(best_move.uci())
                current_board.push(best_move)
            else:
                # Opponent's turn: use opponent head for reply prediction
                opp_logits = self.model.predict_opponent_reply(bt, m)
                # Also get value from our perspective
                out = self.model(bt, legal_mask=m)
                # Value from model is from side-to-move (opponent), negate for our perspective
                v = -out.value[0, 0].item()
                values.append(v)

                best_opp_idx = opp_logits[0].argmax().item()
                try:
                    opp_move = index_to_move(best_opp_idx, current_board)
                except ValueError:
                    break
                moves.append(opp_move.uci())
                current_board.push(opp_move)

        # Compute score
        if not values:
            return {
                "moves": moves,
                "values": values,
                "endpoint_value": 0.0,
                "value_trend": 0.0,
                "score": 0.0,
            }

        endpoint_value = values[-1]

        # Value trend: simple linear slope
        if len(values) >= 2:
            # Least squares slope over the values
            n = len(values)
            x_mean = (n - 1) / 2.0
            y_mean = sum(values) / n
            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            value_trend = numerator / denominator if denominator > 0 else 0.0
        else:
            value_trend = 0.0

        score = endpoint_value + 0.1 * value_trend

        return {
            "moves": moves,
            "values": values,
            "endpoint_value": endpoint_value,
            "value_trend": value_trend,
            "score": score,
        }

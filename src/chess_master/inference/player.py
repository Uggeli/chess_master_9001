"""Full inference pipeline for playing chess."""

import logging

import chess
import torch
import torch.nn.functional as F
from torch import Tensor

from chess_master.board.tensor import board_to_tensor
from chess_master.board.moves import move_to_index, index_to_move, legal_move_mask
from chess_master.config import ModelConfig
from chess_master.inference.session import GameSession
from chess_master.memory.store import MemoryStore
from chess_master.model import ChessMaster9001
from chess_master.planner.two_ply import TwoPlyPlanner
from chess_master.planner.projection import ProjectionPlanner
from chess_master.chess_master_types import NUM_ACTIONS

logger = logging.getLogger(__name__)


class ChessPlayer:
    """Plays chess using the Chess Master 9001 model.

    Handles the full inference pipeline:
    1. Encode the current board
    2. Retrieve relevant memories
    3. Include short-term game context
    4. Select the best move
    """

    def __init__(
        self,
        model: ChessMaster9001,
        memory_store: MemoryStore | None = None,
        device: torch.device | str = "cpu",
        temperature: float = 0.0,
        top_k: int = 1,
        planner: str | None = None,
        planner_k: int = 5,
        planner_r: int = 3,
        projection_horizon: int = 10,
    ):
        self.model = model
        self.model.eval()
        self.memory_store = memory_store
        self.device = torch.device(device)
        self.model.to(self.device)
        self.temperature = temperature
        self.top_k = top_k
        self.session = GameSession(max_window=model.config.short_term_window)

        # Set up planner
        self._planner = None
        if planner == "two_ply":
            self._planner = TwoPlyPlanner(
                model=self.model, k=planner_k, r=planner_r, device=str(self.device),
            )
        elif planner == "projection":
            self._planner = ProjectionPlanner(
                model=self.model, k=planner_k, horizon=projection_horizon,
                device=str(self.device),
            )

    def new_game(self, fen: str | None = None) -> None:
        """Start a new game, resetting session state."""
        self.session.reset(fen)

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """Select the best move for the current position.

        Args:
            board: Current board position.

        Returns:
            Tuple of (selected_move, info_dict) where info_dict contains:
            - value: position evaluation
            - policy: top-k moves with probabilities
            - retrieval: memory retrieval info (if available)
        """
        # Delegate to planner if one is configured
        if self._planner is not None:
            move, planner_info = self._planner.select_move(board)
            # Record in session with a basic value estimate
            board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            mask = legal_move_mask(board).unsqueeze(0).to(self.device)
            output = self.model(board_tensor, legal_mask=mask)
            value = output.value[0, 0].item()
            move_idx = move_to_index(move, board.turn)
            self.session.record_move(move, move_idx, value)
            planner_info["value"] = value
            return move, planner_info

        # 1. Encode board
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        mask = legal_move_mask(board).unsqueeze(0).to(self.device)

        # 2. Prepare short-term context
        st_kwargs = {}
        ctx = self.session.get_context_tensors(device=self.device)
        if ctx is not None:
            st_kwargs = ctx

        # 3. Prepare memory
        mem_kwargs = {}
        if self.memory_store is not None and not self.memory_store.is_empty:
            # We need a retrieval query, but we need to run the model first
            # to get the retrieval embedding. For now, do a two-pass approach:
            # First pass without memory to get the retrieval embedding,
            # then query the store and do a second pass with memory.
            first_output = self.model(
                board_tensor, legal_mask=mask, **st_kwargs,
            )

            if first_output.retrieval_info and "query_embedding" in first_output.retrieval_info:
                query_emb = first_output.retrieval_info["query_embedding"][0]  # [retrieval_dim]
                results = self.memory_store.query(query_emb, k=self.model.config.retrieval_k)

                if results:
                    keys = torch.stack([r[0].state_emb for r in results]).unsqueeze(0).to(self.device)
                    vals = keys.clone()  # Use same embeddings as values
                    mem_kwargs = {"memory_keys": keys, "memory_values": vals}

        # 4. Full forward pass
        output = self.model(board_tensor, legal_mask=mask, **st_kwargs, **mem_kwargs)

        # 5. Select move
        logits = output.policy_logits[0]  # [4672]
        value = output.value[0, 0].item()

        if self.temperature > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            # Sample from top-k
            topk = torch.topk(probs, min(self.top_k, (mask[0] > 0).sum().item()))
            sample_probs = F.softmax(topk.values, dim=-1)
            choice = torch.multinomial(sample_probs, 1).item()
            move_idx = topk.indices[choice].item()
        else:
            move_idx = logits.argmax().item()

        move = index_to_move(move_idx, board)

        # Record in session
        self.session.record_move(move, move_idx, value)

        # Build info dict
        top_probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(top_probs, min(5, (mask[0] > 0).sum().item()))
        top_moves = []
        for prob, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            try:
                m = index_to_move(idx, board)
                top_moves.append({"move": m.uci(), "probability": prob})
            except ValueError:
                continue

        info = {
            "value": value,
            "policy": top_moves,
            "move_index": move_idx,
        }

        if output.retrieval_info is not None:
            info["retrieval"] = {
                "num_memories_used": len(mem_kwargs.get("memory_keys", [])),
            }
            if "similarity_scores" in output.retrieval_info:
                scores = output.retrieval_info["similarity_scores"]
                info["retrieval"]["mean_similarity"] = scores.mean().item()
                info["retrieval"]["max_similarity"] = scores.max().item()

        return move, info

    def play_game(
        self,
        opponent=None,
        max_moves: int = 200,
        starting_fen: str | None = None,
    ) -> list[dict]:
        """Play a complete game.

        Args:
            opponent: Another ChessPlayer, or None for self-play.
            max_moves: Maximum number of full moves.
            starting_fen: Starting position FEN.

        Returns:
            List of move records with move, value, and info.
        """
        self.new_game(starting_fen)
        if opponent is not None:
            opponent.new_game(starting_fen)

        board = chess.Board(starting_fen) if starting_fen else chess.Board()
        game_record = []

        for move_num in range(max_moves * 2):  # half-moves
            if board.is_game_over():
                break

            if opponent is not None and board.turn == chess.BLACK:
                move, info = opponent.select_move(board)
            else:
                move, info = self.select_move(board)

            game_record.append({
                "move_number": board.fullmove_number,
                "side": "white" if board.turn == chess.WHITE else "black",
                "move": move.uci(),
                "value": info["value"],
                "top_moves": info["policy"][:3],
            })

            board.push(move)

        # Record result
        result = board.result() if board.is_game_over() else "*"
        for record in game_record:
            record["game_result"] = result

        return game_record

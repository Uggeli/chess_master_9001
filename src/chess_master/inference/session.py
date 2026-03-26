"""Game session managing short-term memory buffer."""

from dataclasses import dataclass, field

import chess
import torch
from torch import Tensor

from chess_master.types import NUM_ACTIONS


@dataclass
class GameEvent:
    """A single event in the game history."""
    move_idx: int
    value: float
    value_delta: float
    side_to_move: int  # 0=black, 1=white


class GameSession:
    """Manages per-game state including the short-term history buffer.

    Tracks recent moves, evaluations, and evaluation deltas to provide
    short-term context to the model.
    """

    def __init__(self, max_window: int = 10):
        self.max_window = max_window
        self.history: list[GameEvent] = []
        self.board = chess.Board()
        self._last_value: float | None = None

    def reset(self, fen: str | None = None) -> None:
        """Reset the session for a new game."""
        self.history.clear()
        self.board = chess.Board(fen) if fen else chess.Board()
        self._last_value = None

    def record_move(self, move: chess.Move, move_idx: int, value: float) -> None:
        """Record a move and its evaluation in the game history.

        Args:
            move: The chess move played.
            move_idx: The action index of the move.
            value: Position evaluation after this move (from side-to-move perspective).
        """
        if self._last_value is not None:
            value_delta = value - self._last_value
        else:
            value_delta = 0.0

        event = GameEvent(
            move_idx=move_idx,
            value=value,
            value_delta=value_delta,
            side_to_move=int(self.board.turn),
        )

        self.history.append(event)
        if len(self.history) > self.max_window:
            self.history.pop(0)

        self.board.push(move)
        self._last_value = value

    def get_context_tensors(self, device: torch.device | None = None) -> dict[str, Tensor] | None:
        """Get short-term context tensors for the model.

        Returns:
            Dict with st_move_indices, st_values, st_value_deltas, st_side_to_move, st_mask
            as [1, S] tensors, or None if no history.
        """
        if not self.history:
            return None

        S = self.max_window
        move_indices = torch.full((1, S), NUM_ACTIONS, dtype=torch.long)  # padding idx
        values = torch.zeros(1, S)
        value_deltas = torch.zeros(1, S)
        side_to_move = torch.zeros(1, S)
        mask = torch.zeros(1, S, dtype=torch.bool)

        for i, event in enumerate(self.history):
            move_indices[0, i] = event.move_idx
            values[0, i] = event.value
            value_deltas[0, i] = event.value_delta
            side_to_move[0, i] = event.side_to_move
            mask[0, i] = True

        if device is not None:
            move_indices = move_indices.to(device)
            values = values.to(device)
            value_deltas = value_deltas.to(device)
            side_to_move = side_to_move.to(device)
            mask = mask.to(device)

        return {
            "st_move_indices": move_indices,
            "st_values": values,
            "st_value_deltas": value_deltas,
            "st_side_to_move": side_to_move,
            "st_mask": mask,
        }

    @property
    def move_count(self) -> int:
        """Number of moves recorded in this session."""
        return len(self.history)

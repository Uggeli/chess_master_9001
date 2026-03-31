"""PyTorch datasets for chess training data."""

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from chess_master.board.tensor import board_to_tensor
from chess_master.board.moves import move_to_index, legal_move_mask
from chess_master.chess_master_types import NUM_ACTIONS


class ChessPositionDataset(Dataset):
    """Dataset of chess positions with Stockfish labels.

    Expects preprocessed data as a directory containing:
    - positions.jsonl: one JSON object per line with fields:
        - fen: FEN string
        - best_move: UCI string (e.g. "e2e4")
        - evaluation: float in [-1, 1]

    Or a single .jsonl file path.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: list[dict] = []

        if self.path.is_dir():
            data_file = self.path / "positions.jsonl"
        else:
            data_file = self.path

        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        entry = self.entries[idx]

        board = chess.Board(entry["fen"])
        move = chess.Move.from_uci(entry["best_move"])

        board_tensor = board_to_tensor(board)
        move_idx = move_to_index(move, board.turn)
        mask = legal_move_mask(board)
        evaluation = entry["evaluation"]

        result = {
            "board_tensor": board_tensor,
            "target_move": torch.tensor(move_idx, dtype=torch.long),
            "target_value": torch.tensor(evaluation, dtype=torch.float32),
            "legal_mask": mask,
        }

        # Phase B: opponent reply data
        if "opponent_reply" in entry:
            opponent_move = chess.Move.from_uci(entry["opponent_reply"])
            # Apply our move to get the afterstate
            afterstate_board = board.copy()
            afterstate_board.push(move)
            # Encode afterstate from opponent's perspective
            afterstate_tensor = board_to_tensor(afterstate_board)
            opponent_move_idx = move_to_index(opponent_move, afterstate_board.turn)
            opponent_mask = legal_move_mask(afterstate_board)
            result["afterstate_tensor"] = afterstate_tensor
            result["target_opponent_move"] = torch.tensor(
                opponent_move_idx, dtype=torch.long
            )
            result["opponent_legal_mask"] = opponent_mask

        return result


class ChessPositionNpzDataset(Dataset):
    """Dataset loading from preprocessed numpy arrays for speed.

    Expects an .npz file with arrays:
    - board_tensors: [N, 18, 8, 8] float32
    - target_moves: [N] int64
    - target_values: [N] float32
    - legal_masks: [N, 4672] float32
    """

    def __init__(self, path: str | Path):
        data = np.load(path)
        self.board_tensors = torch.from_numpy(data["board_tensors"])
        self.target_moves = torch.from_numpy(data["target_moves"]).long()
        self.target_values = torch.from_numpy(data["target_values"]).float()
        self.legal_masks = torch.from_numpy(data["legal_masks"]).float()

    def __len__(self) -> int:
        return len(self.board_tensors)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "board_tensor": self.board_tensors[idx],
            "target_move": self.target_moves[idx],
            "target_value": self.target_values[idx],
            "legal_mask": self.legal_masks[idx],
        }

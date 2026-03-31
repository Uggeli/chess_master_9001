"""Board state to tensor encoding."""

import chess
import numpy as np
import torch
from torch import Tensor

from chess_master.chess_master_types import NUM_INPUT_PLANES, BOARD_SIZE

# Piece type to plane index mapping (white pieces 0-5, black pieces 6-11)
PIECE_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> Tensor:
    """Encode a chess board as a tensor.

    Always encodes from the perspective of the side to move:
    - Planes 0-5: current player's pieces (P, N, B, R, Q, K)
    - Planes 6-11: opponent's pieces (P, N, B, R, Q, K)
    - Plane 12: current player kingside castling rights
    - Plane 13: current player queenside castling rights
    - Plane 14: opponent kingside castling rights
    - Plane 15: opponent queenside castling rights
    - Plane 16: en passant square (1 where en passant capture is possible)
    - Plane 17: side to move (all 1s if white, all 0s if black)

    Args:
        board: A python-chess Board object.

    Returns:
        Tensor of shape [18, 8, 8].
    """
    planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    flip = board.turn == chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # Get row/col, flip board if black to move
        row = chess.square_rank(square)
        col = chess.square_file(square)
        if flip:
            row = 7 - row

        # Determine plane: current player's pieces go in 0-5, opponent in 6-11
        piece_type = piece.piece_type  # 1-6
        is_current_player = (piece.color == board.turn)

        if is_current_player:
            plane_idx = piece_type - 1  # 0-5
        else:
            plane_idx = piece_type - 1 + 6  # 6-11

        planes[plane_idx, row, col] = 1.0

    # Castling rights (from current player's perspective)
    if board.turn == chess.WHITE:
        planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    else:
        planes[12, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[13, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
        planes[14, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[15, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))

    # En passant
    if board.ep_square is not None:
        ep_row = chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        if flip:
            ep_row = 7 - ep_row
        planes[16, ep_row, ep_col] = 1.0

    # Side to move
    planes[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return torch.from_numpy(planes)


def boards_to_tensor(boards: list[chess.Board]) -> Tensor:
    """Encode multiple boards as a batched tensor.

    Args:
        boards: List of python-chess Board objects.

    Returns:
        Tensor of shape [B, 18, 8, 8].
    """
    return torch.stack([board_to_tensor(b) for b in boards])

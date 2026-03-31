"""Move index encoding/decoding (AlphaZero 4672 action space)."""

import chess
import numpy as np
import torch
from torch import Tensor

from chess_master.chess_master_types import NUM_ACTIONS

# Direction vectors for queen-like moves (N, NE, E, SE, S, SW, W, NW)
QUEEN_DIRS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
]

# Knight move offsets
KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

# Underpromotion pieces (queen promotion is handled by queen moves)
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Underpromotion capture directions: left, forward, right (from current player's perspective)
UNDERPROMO_DIRS = [(-1, 1), (0, 1), (1, 1)]  # (file_delta, rank_delta)


def _move_to_plane(move: chess.Move, turn: chess.Color) -> int | None:
    """Convert a move to its plane index (0-72).

    Coordinates are from the perspective of the side to move.
    """
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    # Flip if black to move (so we always think from "bottom up")
    if turn == chess.BLACK:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    dr = to_rank - from_rank
    df = to_file - from_file

    # Check underpromotion first
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_idx = UNDERPROMO_PIECES.index(move.promotion)
        for dir_idx, (fd, rd) in enumerate(UNDERPROMO_DIRS):
            if df == fd and dr == rd:
                return 64 + promo_idx * 3 + dir_idx
        return None

    # Knight moves
    if (dr, df) in [(m[1], m[0]) for m in KNIGHT_MOVES]:
        # Map to knight index
        for i, (kf, kr) in enumerate(KNIGHT_MOVES):
            if df == kf and dr == kr:
                return 56 + i
        return None

    # Queen-like moves (includes queen promotions which use same direction)
    if dr == 0 and df == 0:
        return None

    # Determine direction
    if dr != 0:
        dr_sign = dr // abs(dr)
    else:
        dr_sign = 0
    if df != 0:
        df_sign = df // abs(df)
    else:
        df_sign = 0

    distance = max(abs(dr), abs(df))

    for dir_idx, (d_r, d_f) in enumerate(QUEEN_DIRS):
        if d_r == dr_sign and d_f == df_sign:
            return dir_idx * 7 + (distance - 1)

    return None


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Convert a chess move to an action index (0-4671).

    Index = plane * 64 + from_square_transformed.
    The from_square is flipped for black so the encoding is always
    from the current player's perspective.

    Args:
        move: A python-chess Move.
        turn: The side to move (chess.WHITE or chess.BLACK).

    Returns:
        Action index in [0, 4671].
    """
    plane = _move_to_plane(move, turn)
    if plane is None:
        raise ValueError(f"Cannot encode move {move}")

    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)

    if turn == chess.BLACK:
        from_rank = 7 - from_rank

    square_idx = from_rank * 8 + from_file
    return plane * 64 + square_idx


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert an action index back to a chess move.

    Args:
        index: Action index in [0, 4671].
        board: Current board state (needed for legality check and perspective).

    Returns:
        A python-chess Move.

    Raises:
        ValueError: If the index doesn't correspond to a legal move.
    """
    plane = index // 64
    square_idx = index % 64

    from_rank = square_idx // 8
    from_file = square_idx % 8

    if board.turn == chess.BLACK:
        from_rank = 7 - from_rank

    from_square = chess.square(from_file, from_rank)

    # Decode the plane to get the move direction/type
    if plane < 56:
        # Queen move
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        dr, df = QUEEN_DIRS[dir_idx]
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        if board.turn == chess.BLACK:
            # Undo the perspective flip for from_rank, compute to in original coords
            orig_from_rank = 7 - (square_idx // 8)  # un-flipped
            # dr was in flipped space, so flip it
            to_rank_orig = orig_from_rank + dr * distance  # but dr is in flipped perspective
            # Actually let's recompute properly
            pass
        # Simpler: compute in flipped space, then unflip
        flipped_from_rank = square_idx // 8
        flipped_to_rank = flipped_from_rank + dr * distance
        flipped_to_file = from_file + df * distance

        to_file = flipped_to_file
        to_rank = flipped_to_rank
        if board.turn == chess.BLACK:
            to_rank = 7 - to_rank
            from_rank_actual = 7 - flipped_from_rank
        else:
            from_rank_actual = flipped_from_rank

        from_square = chess.square(from_file, from_rank_actual)

        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Index {index} produces out-of-bounds move")

        to_square = chess.square(to_file, to_rank)

        # Check if this is a pawn reaching promotion rank -> queen promotion
        piece = board.piece_at(from_square)
        promotion = None
        if piece and piece.piece_type == chess.PAWN:
            if (board.turn == chess.WHITE and to_rank == 7) or \
               (board.turn == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN

        return chess.Move(from_square, to_square, promotion=promotion)

    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        kf, kr = KNIGHT_MOVES[knight_idx]

        flipped_from_rank = square_idx // 8
        flipped_to_rank = flipped_from_rank + kr
        to_file = from_file + kf

        if board.turn == chess.BLACK:
            to_rank = 7 - flipped_to_rank
            from_rank_actual = 7 - flipped_from_rank
        else:
            to_rank = flipped_to_rank
            from_rank_actual = flipped_from_rank

        from_square = chess.square(from_file, from_rank_actual)

        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Index {index} produces out-of-bounds move")

        to_square = chess.square(to_file, to_rank)
        return chess.Move(from_square, to_square)

    else:
        # Underpromotion
        promo_plane = plane - 64
        promo_piece = UNDERPROMO_PIECES[promo_plane // 3]
        dir_idx = promo_plane % 3
        fd, rd = UNDERPROMO_DIRS[dir_idx]

        flipped_from_rank = square_idx // 8
        flipped_to_rank = flipped_from_rank + rd
        to_file = from_file + fd

        if board.turn == chess.BLACK:
            to_rank = 7 - flipped_to_rank
            from_rank_actual = 7 - flipped_from_rank
        else:
            to_rank = flipped_to_rank
            from_rank_actual = flipped_from_rank

        from_square = chess.square(from_file, from_rank_actual)

        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Index {index} produces out-of-bounds move")

        to_square = chess.square(to_file, to_rank)
        return chess.Move(from_square, to_square, promotion=promo_piece)


def legal_move_mask(board: chess.Board) -> Tensor:
    """Create a binary mask over the action space for legal moves.

    Args:
        board: Current board state.

    Returns:
        Tensor of shape [4672] with 1.0 for legal moves, 0.0 otherwise.
    """
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board.turn)
            mask[idx] = 1.0
        except ValueError:
            continue
    return mask

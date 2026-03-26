"""Board utility functions."""

import chess


def flip_board(board: chess.Board) -> chess.Board:
    """Create a color-flipped copy of the board.

    Swaps piece colors and mirrors vertically. Useful for
    perspective consistency testing.
    """
    flipped = board.mirror()
    return flipped


def fen_to_board(fen: str) -> chess.Board:
    """Create a board from a FEN string."""
    return chess.Board(fen)


def game_phase(board: chess.Board) -> str:
    """Estimate the game phase based on material.

    Returns:
        One of 'opening', 'middlegame', 'endgame'.
    """
    # Count non-pawn, non-king material
    material = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        material += len(board.pieces(piece_type, chess.WHITE))
        material += len(board.pieces(piece_type, chess.BLACK))

    move_number = board.fullmove_number

    if move_number <= 10 and material >= 12:
        return "opening"
    elif material <= 6:
        return "endgame"
    else:
        return "middlegame"

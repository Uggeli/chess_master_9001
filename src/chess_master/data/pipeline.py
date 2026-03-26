"""Data generation and preprocessing pipeline.

Reads chess games (PGN format), analyzes positions with Stockfish,
and saves preprocessed training data.
"""

import json
import logging
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chess_master.board.tensor import board_to_tensor
from chess_master.board.moves import move_to_index, legal_move_mask
from chess_master.data.stockfish import StockfishLabeler

logger = logging.getLogger(__name__)


def generate_from_pgn(
    pgn_path: str | Path,
    output_path: str | Path,
    stockfish_path: str = "",
    stockfish_depth: int = 16,
    max_games: int | None = None,
    max_positions: int | None = None,
    skip_opening_moves: int = 6,
) -> int:
    """Generate labeled training data from PGN games.

    Args:
        pgn_path: Path to PGN file.
        output_path: Path for output .jsonl file.
        stockfish_path: Path to Stockfish binary (empty for pip default).
        stockfish_depth: Stockfish search depth.
        max_games: Maximum number of games to process.
        max_positions: Maximum total positions to generate.
        skip_opening_moves: Skip this many opening moves per game.

    Returns:
        Number of positions generated.
    """
    labeler = StockfishLabeler(path=stockfish_path, depth=stockfish_depth)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    games_processed = 0

    with open(pgn_path) as pgn_file, open(output_path, "w") as out_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            if max_games is not None and games_processed >= max_games:
                break

            board = game.board()
            move_num = 0

            for move in game.mainline_moves():
                board.push(move)
                move_num += 1

                if move_num <= skip_opening_moves:
                    continue

                if board.is_game_over():
                    continue

                label = labeler.label(board)
                if label is None:
                    continue

                entry = {
                    "fen": board.fen(),
                    "best_move": label.best_move.uci(),
                    "evaluation": label.evaluation,
                }
                out_file.write(json.dumps(entry) + "\n")
                count += 1

                if count % 1000 == 0:
                    logger.info(f"Generated {count} positions from {games_processed + 1} games")

                if max_positions is not None and count >= max_positions:
                    break

            games_processed += 1

            if max_positions is not None and count >= max_positions:
                break

    labeler.close()
    logger.info(f"Done: {count} positions from {games_processed} games -> {output_path}")
    return count


def jsonl_to_npz(
    jsonl_path: str | Path,
    npz_path: str | Path,
) -> int:
    """Convert JSONL training data to NPZ format for fast loading.

    Args:
        jsonl_path: Path to input .jsonl file.
        npz_path: Path for output .npz file.

    Returns:
        Number of positions converted.
    """
    board_tensors = []
    target_moves = []
    target_values = []
    legal_masks = []

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            board = chess.Board(entry["fen"])
            move = chess.Move.from_uci(entry["best_move"])

            board_tensors.append(board_to_tensor(board).numpy())
            target_moves.append(move_to_index(move, board.turn))
            target_values.append(entry["evaluation"])
            legal_masks.append(legal_move_mask(board).numpy())

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        npz_path,
        board_tensors=np.array(board_tensors, dtype=np.float32),
        target_moves=np.array(target_moves, dtype=np.int64),
        target_values=np.array(target_values, dtype=np.float32),
        legal_masks=np.array(legal_masks, dtype=np.float32),
    )

    logger.info(f"Converted {len(board_tensors)} positions -> {npz_path}")
    return len(board_tensors)

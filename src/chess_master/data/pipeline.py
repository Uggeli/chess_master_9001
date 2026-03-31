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


def _extract_positions_from_game(game, skip_opening_moves: int) -> list[str]:
    """Extract FEN positions from a game, skipping openings and terminal positions."""
    board = game.board()
    move_num = 0
    fens = []
    for move in game.mainline_moves():
        board.push(move)
        move_num += 1
        if move_num <= skip_opening_moves:
            continue
        if board.is_game_over():
            continue
        fens.append(board.fen())
    return fens


def _worker_label_positions(args):
    """Worker function for parallel labeling."""
    fens, stockfish_path, stockfish_depth, threads, hash_mb = args
    labeler = StockfishLabeler(
        path=stockfish_path, depth=stockfish_depth,
        threads=threads, hash_mb=hash_mb,
    )
    results = []
    for fen in fens:
        board = chess.Board(fen)
        label = labeler.label(board)
        if label is not None:
            results.append({
                "fen": fen,
                "best_move": label.best_move.uci(),
                "evaluation": label.evaluation,
            })
    labeler.close()
    return results


def generate_from_pgn(
    pgn_path: str | Path,
    output_path: str | Path,
    stockfish_path: str = "",
    stockfish_depth: int = 16,
    max_games: int | None = None,
    max_positions: int | None = None,
    skip_opening_moves: int = 6,
    num_workers: int = 1,
    threads_per_worker: int = 1,
    hash_per_worker: int = 64,
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
        num_workers: Number of parallel Stockfish workers.
        threads_per_worker: Stockfish threads per worker.
        hash_per_worker: Stockfish hash MB per worker.

    Returns:
        Number of positions generated.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract all positions from games (fast, no Stockfish needed)
    logger.info("Extracting positions from PGN...")
    all_fens = []
    games_processed = 0

    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            if max_games is not None and games_processed >= max_games:
                break
            fens = _extract_positions_from_game(game, skip_opening_moves)
            all_fens.extend(fens)
            games_processed += 1
            if games_processed % 1000 == 0:
                logger.info(f"Read {games_processed} games, {len(all_fens)} positions")
            if max_positions is not None and len(all_fens) >= max_positions:
                all_fens = all_fens[:max_positions]
                break

    logger.info(f"Extracted {len(all_fens)} positions from {games_processed} games")

    # Phase 2: Label positions with Stockfish (parallel)
    if num_workers > 1:
        import multiprocessing as mp

        chunk_size = (len(all_fens) + num_workers - 1) // num_workers
        chunks = [all_fens[i:i + chunk_size] for i in range(0, len(all_fens), chunk_size)]
        worker_args = [
            (chunk, stockfish_path, stockfish_depth, threads_per_worker, hash_per_worker)
            for chunk in chunks
        ]

        logger.info(f"Labeling with {num_workers} workers x {threads_per_worker} threads, depth {stockfish_depth}")
        with mp.Pool(num_workers) as pool:
            results = pool.map(_worker_label_positions, worker_args)

        all_entries = []
        for chunk_results in results:
            all_entries.extend(chunk_results)
    else:
        logger.info(f"Labeling with 1 worker x {threads_per_worker} threads, depth {stockfish_depth}")
        all_entries = _worker_label_positions(
            (all_fens, stockfish_path, stockfish_depth, threads_per_worker, hash_per_worker)
        )

    # Phase 3: Write output
    with open(output_path, "w") as out_file:
        for entry in all_entries:
            out_file.write(json.dumps(entry) + "\n")

    logger.info(f"Done: {len(all_entries)} positions from {games_processed} games -> {output_path}")
    return len(all_entries)


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

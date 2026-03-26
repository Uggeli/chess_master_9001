"""Self-play evaluation for relative strength estimation."""

import logging
import math

import chess
import torch

from chess_master.inference.player import ChessPlayer
from chess_master.model import ChessMaster9001

logger = logging.getLogger(__name__)


def play_match(
    player_a: ChessPlayer,
    player_b: ChessPlayer,
    num_games: int = 100,
    max_moves: int = 200,
) -> dict:
    """Play a match between two players.

    Each pair of games is played with colors swapped for fairness.

    Args:
        player_a: First player.
        player_b: Second player.
        num_games: Total number of games (should be even).
        max_moves: Max moves per game.

    Returns:
        Match results dict.
    """
    num_games = num_games - (num_games % 2)  # ensure even

    results = {"a_wins": 0, "b_wins": 0, "draws": 0}

    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            white, black = player_a, player_b
            white_is_a = True
        else:
            white, black = player_b, player_a
            white_is_a = False

        white.new_game()
        black.new_game()
        board = chess.Board()

        for _ in range(max_moves * 2):
            if board.is_game_over():
                break

            if board.turn == chess.WHITE:
                move, _ = white.select_move(board)
            else:
                move, _ = black.select_move(board)
            board.push(move)

        result = board.result()
        if result == "1-0":
            if white_is_a:
                results["a_wins"] += 1
            else:
                results["b_wins"] += 1
        elif result == "0-1":
            if white_is_a:
                results["b_wins"] += 1
            else:
                results["a_wins"] += 1
        else:
            results["draws"] += 1

        if (i + 1) % 10 == 0:
            logger.info(
                f"Game {i + 1}/{num_games}: "
                f"A={results['a_wins']} B={results['b_wins']} D={results['draws']}"
            )

    total = results["a_wins"] + results["b_wins"] + results["draws"]
    score_a = (results["a_wins"] + 0.5 * results["draws"]) / total

    results["total_games"] = total
    results["score_a"] = score_a
    results["score_b"] = 1.0 - score_a
    results["elo_diff"] = score_to_elo_diff(score_a)

    return results


def score_to_elo_diff(score: float) -> float:
    """Convert a match score to an Elo difference estimate.

    Args:
        score: Win rate of player A (0.0 to 1.0).

    Returns:
        Estimated Elo difference (positive = A is stronger).
    """
    if score <= 0.0 or score >= 1.0:
        return float("inf") if score >= 1.0 else float("-inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def print_match_report(results: dict, name_a: str = "A", name_b: str = "B") -> None:
    """Print a formatted match report."""
    print(f"\nMatch: {name_a} vs {name_b}")
    print("=" * 40)
    print(f"Games played:  {results['total_games']}")
    print(f"{name_a} wins:       {results['a_wins']}")
    print(f"{name_b} wins:       {results['b_wins']}")
    print(f"Draws:         {results['draws']}")
    print(f"{name_a} score:      {results['score_a']:.1%}")
    print(f"Elo diff:      {results['elo_diff']:+.0f} ({name_a} vs {name_b})")

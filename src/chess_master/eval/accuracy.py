"""Move accuracy evaluation against Stockfish."""

import logging
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from chess_master.board.tensor import board_to_tensor
from chess_master.board.moves import move_to_index, legal_move_mask
from chess_master.board.utils import game_phase
from chess_master.model import ChessMaster9001

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_accuracy(
    model: ChessMaster9001,
    positions: list[dict],
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> dict:
    """Evaluate move prediction accuracy against Stockfish labels.

    Args:
        model: The chess model to evaluate.
        positions: List of dicts with 'fen', 'best_move' keys.
        device: Device to run on.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with accuracy metrics segmented by game phase.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    results = {
        "overall": {"correct_top1": 0, "correct_top5": 0, "total": 0},
        "opening": {"correct_top1": 0, "correct_top5": 0, "total": 0},
        "middlegame": {"correct_top1": 0, "correct_top5": 0, "total": 0},
        "endgame": {"correct_top1": 0, "correct_top5": 0, "total": 0},
    }

    for i in range(0, len(positions), batch_size):
        batch_positions = positions[i:i + batch_size]

        boards = [chess.Board(p["fen"]) for p in batch_positions]
        target_moves = [chess.Move.from_uci(p["best_move"]) for p in batch_positions]
        phases = [game_phase(b) for b in boards]

        board_tensors = torch.stack([board_to_tensor(b) for b in boards]).to(device)
        masks = torch.stack([legal_move_mask(b) for b in boards]).to(device)
        target_indices = torch.tensor(
            [move_to_index(m, b.turn) for m, b in zip(target_moves, boards)],
            dtype=torch.long,
        ).to(device)

        output = model(board_tensors, legal_mask=masks)
        logits = output.policy_logits

        # Top-1 accuracy
        top1_preds = logits.argmax(dim=-1)
        top1_correct = (top1_preds == target_indices)

        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1).indices
        top5_correct = (top5_preds == target_indices.unsqueeze(-1)).any(dim=-1)

        for j, phase in enumerate(phases):
            results[phase]["total"] += 1
            results["overall"]["total"] += 1
            if top1_correct[j]:
                results[phase]["correct_top1"] += 1
                results["overall"]["correct_top1"] += 1
            if top5_correct[j]:
                results[phase]["correct_top5"] += 1
                results["overall"]["correct_top5"] += 1

    # Compute percentages
    summary = {}
    for phase, counts in results.items():
        if counts["total"] > 0:
            summary[phase] = {
                "top1_accuracy": counts["correct_top1"] / counts["total"],
                "top5_accuracy": counts["correct_top5"] / counts["total"],
                "count": counts["total"],
            }
        else:
            summary[phase] = {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "count": 0}

    return summary


def print_accuracy_report(summary: dict) -> None:
    """Print a formatted accuracy report."""
    print(f"\n{'Phase':<15} {'Top-1':>8} {'Top-5':>8} {'Count':>8}")
    print("-" * 42)
    for phase in ["opening", "middlegame", "endgame", "overall"]:
        if phase in summary:
            s = summary[phase]
            print(f"{phase:<15} {s['top1_accuracy']:>7.1%} {s['top5_accuracy']:>7.1%} {s['count']:>8}")

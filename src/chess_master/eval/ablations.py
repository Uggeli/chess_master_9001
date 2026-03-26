"""Ablation studies to measure component contributions.

Compares model behavior with and without memory / short-term context
to verify these components are contributing meaningfully.
"""

import logging

import chess
import torch

from chess_master.board.tensor import board_to_tensor
from chess_master.board.moves import index_to_move, legal_move_mask
from chess_master.model import ChessMaster9001

logger = logging.getLogger(__name__)


@torch.no_grad()
def memory_ablation(
    model: ChessMaster9001,
    positions: list[dict],
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    device: torch.device | str = "cpu",
) -> dict:
    """Compare model predictions with and without memory.

    Args:
        model: The model to test.
        positions: List of dicts with 'fen' key.
        memory_keys: [N, retrieval_dim] memory embeddings.
        memory_values: [N, retrieval_dim] memory values.
        device: Device to run on.

    Returns:
        Dict with ablation metrics.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    move_changes = 0
    value_diffs = []
    total = 0

    for pos in positions:
        board = chess.Board(pos["fen"])
        bt = board_to_tensor(board).unsqueeze(0).to(device)
        mask = legal_move_mask(board).unsqueeze(0).to(device)

        # Without memory
        out_no_mem = model(bt, legal_mask=mask)
        move_no_mem = out_no_mem.policy_logits.argmax(dim=-1).item()
        value_no_mem = out_no_mem.value[0, 0].item()

        # With memory
        mk = memory_keys.unsqueeze(0).to(device)
        mv = memory_values.unsqueeze(0).to(device)
        out_with_mem = model(bt, legal_mask=mask, memory_keys=mk, memory_values=mv)
        move_with_mem = out_with_mem.policy_logits.argmax(dim=-1).item()
        value_with_mem = out_with_mem.value[0, 0].item()

        if move_no_mem != move_with_mem:
            move_changes += 1
        value_diffs.append(abs(value_with_mem - value_no_mem))
        total += 1

    return {
        "total_positions": total,
        "move_change_rate": move_changes / total if total > 0 else 0,
        "mean_value_diff": sum(value_diffs) / total if total > 0 else 0,
        "max_value_diff": max(value_diffs) if value_diffs else 0,
    }


def print_ablation_report(results: dict, label: str = "Memory") -> None:
    """Print a formatted ablation report."""
    print(f"\n{label} Ablation Report")
    print("=" * 40)
    print(f"Positions tested:   {results['total_positions']}")
    print(f"Move change rate:   {results['move_change_rate']:.1%}")
    print(f"Mean value diff:    {results['mean_value_diff']:.4f}")
    print(f"Max value diff:     {results['max_value_diff']:.4f}")

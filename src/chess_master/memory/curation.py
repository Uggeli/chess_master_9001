"""Memory curation: importance scoring and eviction policies."""

import torch
from torch import Tensor

from chess_master.types import MemoryEntry


def score_importance(
    entry: MemoryEntry,
    retrieval_count: int = 0,
    prediction_error: float = 0.0,
) -> float:
    """Compute importance score for a memory entry.

    Signals that increase importance:
    - High retrieval frequency (the entry is useful)
    - High prediction error (the entry is surprising/informative)
    - Extreme outcome (decisive game result)
    - Non-stockfish source (harder to regenerate)

    Args:
        entry: The memory entry to score.
        retrieval_count: How many times this entry has been retrieved.
        prediction_error: How much the model's prediction differed from stored value.

    Returns:
        Importance score (higher = more important to keep).
    """
    score = entry.importance  # Start with existing importance

    # Retrieval frequency bonus
    score += 0.1 * retrieval_count

    # Surprise bonus
    score += 0.5 * abs(prediction_error)

    # Decisive outcome bonus
    score += 0.2 * abs(entry.outcome)

    # Source rarity bonus
    if entry.source != "stockfish":
        score += 0.1

    return score

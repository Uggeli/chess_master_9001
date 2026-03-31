"""Long-term episodic memory store.

Manages a bounded collection of MemoryEntry items with:
- Add/query operations using cosine similarity
- Disk persistence (save/load)
- Importance-weighted eviction when at capacity
- Time-based decay of importance scores
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from chess_master.chess_master_types import MemoryEntry


class MemoryStore:
    """Bounded episodic memory with similarity-based retrieval."""

    def __init__(self, capacity: int = 10_000, decay_rate: float = 0.999):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._embeddings: list[Tensor] = []  # each [d_model]
        self._entries: list[MemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def add(self, entries: list[MemoryEntry]) -> None:
        """Add entries to the store, evicting low-importance items if at capacity."""
        for entry in entries:
            if len(self._entries) >= self.capacity:
                self._evict_one()
            self._embeddings.append(entry.state_emb.detach().cpu())
            self._entries.append(entry)

    def query(self, embedding: Tensor, k: int = 8) -> list[tuple[MemoryEntry, float]]:
        """Retrieve the k most similar entries by cosine similarity.

        Args:
            embedding: Query embedding [d_model].
            k: Number of entries to retrieve.

        Returns:
            List of (entry, similarity_score) tuples, sorted by descending similarity.
        """
        if self.is_empty:
            return []

        k = min(k, len(self._entries))

        # Stack all embeddings and compute cosine similarity
        emb_matrix = torch.stack(self._embeddings)  # [N, d_model]
        query = embedding.detach().cpu()

        similarities = F.cosine_similarity(
            query.unsqueeze(0), emb_matrix, dim=1
        )  # [N]

        top_k = torch.topk(similarities, k)

        results = []
        for idx, sim in zip(top_k.indices.tolist(), top_k.values.tolist()):
            results.append((self._entries[idx], sim))

        return results

    def decay_importance(self) -> None:
        """Apply time-based decay to all importance scores."""
        decayed = []
        for entry in self._entries:
            new_importance = entry.importance * self.decay_rate
            decayed.append(entry._replace(importance=new_importance))
        self._entries = decayed

    def _evict_one(self) -> None:
        """Remove the entry with the lowest importance score."""
        if self.is_empty:
            return
        min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].importance)
        self._entries.pop(min_idx)
        self._embeddings.pop(min_idx)

    def save(self, path: str | Path) -> None:
        """Save the memory store to disk.

        Saves embeddings as a .pt tensor file and metadata as JSON.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._embeddings:
            emb_tensor = torch.stack(self._embeddings)
            torch.save(emb_tensor, path / "embeddings.pt")

        metadata = []
        for entry in self._entries:
            metadata.append({
                "move_idx": entry.move_idx,
                "value": entry.value,
                "outcome": entry.outcome,
                "source": entry.source,
                "importance": entry.importance,
            })

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    def load(self, path: str | Path) -> None:
        """Load a memory store from disk."""
        path = Path(path)

        emb_path = path / "embeddings.pt"
        meta_path = path / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata found at {meta_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        if emb_path.exists():
            emb_tensor = torch.load(emb_path, weights_only=True)
            embeddings = list(emb_tensor.unbind(0))
        else:
            embeddings = []

        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != metadata count ({len(metadata)})"
            )

        self._embeddings = embeddings
        self._entries = []
        for emb, meta in zip(embeddings, metadata):
            self._entries.append(MemoryEntry(
                state_emb=emb,
                move_idx=meta["move_idx"],
                value=meta["value"],
                outcome=meta["outcome"],
                source=meta["source"],
                importance=meta["importance"],
            ))

    def stats(self) -> dict:
        """Return summary statistics about the store."""
        if self.is_empty:
            return {"size": 0}

        importances = [e.importance for e in self._entries]
        values = [e.value for e in self._entries]
        sources = {}
        for e in self._entries:
            sources[e.source] = sources.get(e.source, 0) + 1

        return {
            "size": len(self._entries),
            "capacity": self.capacity,
            "importance_mean": sum(importances) / len(importances),
            "importance_min": min(importances),
            "importance_max": max(importances),
            "value_mean": sum(values) / len(values),
            "sources": sources,
        }

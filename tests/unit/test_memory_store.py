"""Tests for memory store."""

import tempfile

import torch

from chess_master.memory.store import MemoryStore
from chess_master.chess_master_types import MemoryEntry


class TestMemoryStore:
    def test_add_and_len(self, sample_memory_entries):
        store = MemoryStore(capacity=100)
        store.add(sample_memory_entries)
        assert len(store) == 20

    def test_query_returns_similar(self):
        store = MemoryStore(capacity=100)
        # Add a known entry
        target_emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
        store.add([MemoryEntry(target_emb, 0, 0.5, 1.0, "stockfish", 1.0)])
        # Add some random entries
        for i in range(10):
            store.add([MemoryEntry(torch.randn(4), i + 1, 0.0, 0.0, "stockfish", 0.5)])

        # Query with something similar to target
        query = torch.tensor([0.9, 0.1, 0.0, 0.0])
        results = store.query(query, k=1)
        assert len(results) == 1
        assert results[0][0].move_idx == 0  # Should retrieve the target

    def test_capacity_eviction(self):
        store = MemoryStore(capacity=5)
        entries = [
            MemoryEntry(torch.randn(4), i, 0.0, 0.0, "stockfish", float(i))
            for i in range(10)
        ]
        store.add(entries)
        assert len(store) == 5

    def test_save_and_load(self, sample_memory_entries):
        store = MemoryStore(capacity=100)
        store.add(sample_memory_entries)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            loaded = MemoryStore(capacity=100)
            loaded.load(tmpdir)

            assert len(loaded) == len(store)

    def test_decay_importance(self, sample_memory_entries):
        store = MemoryStore(capacity=100, decay_rate=0.5)
        store.add(sample_memory_entries)

        orig_importance = [e.importance for e in store._entries]
        store.decay_importance()

        for orig, entry in zip(orig_importance, store._entries):
            assert abs(entry.importance - orig * 0.5) < 1e-6

    def test_stats(self, sample_memory_entries):
        store = MemoryStore(capacity=100)
        store.add(sample_memory_entries)
        stats = store.stats()
        assert stats["size"] == 20
        assert "importance_mean" in stats
        assert "sources" in stats

    def test_empty_store(self):
        store = MemoryStore()
        assert store.is_empty
        assert store.query(torch.randn(4), k=5) == []
        assert store.stats() == {"size": 0}

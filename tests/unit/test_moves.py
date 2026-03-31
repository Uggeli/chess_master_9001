"""Tests for move encoding/decoding."""

import chess
import torch

from chess_master.board.moves import move_to_index, index_to_move, legal_move_mask
from chess_master.chess_master_types import NUM_ACTIONS


class TestMoveEncoding:
    def test_roundtrip_starting_position(self, start_board):
        """Every legal move should survive encode->decode."""
        for move in start_board.legal_moves:
            idx = move_to_index(move, start_board.turn)
            assert 0 <= idx < NUM_ACTIONS
            decoded = index_to_move(idx, start_board)
            assert decoded == move, f"Roundtrip failed: {move} -> {idx} -> {decoded}"

    def test_roundtrip_midgame(self, midgame_board):
        for move in midgame_board.legal_moves:
            idx = move_to_index(move, midgame_board.turn)
            decoded = index_to_move(idx, midgame_board)
            assert decoded == move

    def test_unique_indices(self, start_board):
        """Each legal move should map to a unique index."""
        indices = set()
        for move in start_board.legal_moves:
            idx = move_to_index(move, start_board.turn)
            assert idx not in indices, f"Duplicate index {idx} for move {move}"
            indices.add(idx)

    def test_promotion_queen(self):
        """Queen promotion should encode as a queen-direction move."""
        board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        move = chess.Move.from_uci("e7e8q")
        idx = move_to_index(move, board.turn)
        decoded = index_to_move(idx, board)
        assert decoded.promotion == chess.QUEEN

    def test_promotion_knight(self):
        """Knight underpromotion should encode correctly."""
        board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        move = chess.Move.from_uci("e7e8n")
        idx = move_to_index(move, board.turn)
        decoded = index_to_move(idx, board)
        assert decoded.promotion == chess.KNIGHT

    def test_black_moves(self):
        """Moves from black's perspective should encode correctly."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))  # black to move
        for move in board.legal_moves:
            idx = move_to_index(move, board.turn)
            decoded = index_to_move(idx, board)
            assert decoded == move


class TestLegalMoveMask:
    def test_shape(self, start_board):
        mask = legal_move_mask(start_board)
        assert mask.shape == (NUM_ACTIONS,)

    def test_count(self, start_board):
        mask = legal_move_mask(start_board)
        # Starting position has 20 legal moves
        assert mask.sum().item() == 20

    def test_legal_moves_are_one(self, start_board):
        mask = legal_move_mask(start_board)
        for move in start_board.legal_moves:
            idx = move_to_index(move, start_board.turn)
            assert mask[idx] == 1.0

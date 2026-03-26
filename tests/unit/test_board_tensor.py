"""Tests for board tensor encoding."""

import chess
import torch

from chess_master.board.tensor import board_to_tensor, boards_to_tensor
from chess_master.types import NUM_INPUT_PLANES, BOARD_SIZE


class TestBoardToTensor:
    def test_output_shape(self, start_board):
        tensor = board_to_tensor(start_board)
        assert tensor.shape == (NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)

    def test_dtype(self, start_board):
        tensor = board_to_tensor(start_board)
        assert tensor.dtype == torch.float32

    def test_starting_position_has_pieces(self, start_board):
        tensor = board_to_tensor(start_board)
        # In starting position (white to move), white pieces are on planes 0-5
        # White pawns on rank 1 (row 1)
        assert tensor[0, 1, :].sum() == 8  # 8 white pawns

    def test_empty_board(self):
        board = chess.Board(fen=None)  # empty board
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        tensor = board_to_tensor(board)
        # Only 2 pieces on the board
        assert tensor[:12].sum() == 2

    def test_perspective_flip(self):
        """Black to move should flip the board perspective."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))  # Now black to move
        tensor = board_to_tensor(board)
        # Black pieces should be in planes 0-5 (current player)
        # Black pawns should appear on rank 1 from black's perspective (row 1 after flip)
        assert tensor[0, 1, :].sum() == 8  # 8 current-player (black) pawns

    def test_side_to_move_plane(self, start_board):
        tensor = board_to_tensor(start_board)
        # Plane 17: all 1s for white to move
        assert tensor[17].sum() == 64

    def test_side_to_move_black(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        tensor = board_to_tensor(board)
        # Plane 17: all 0s for black to move
        assert tensor[17].sum() == 0

    def test_castling_planes(self, start_board):
        tensor = board_to_tensor(start_board)
        # Starting position: both sides have both castling rights
        assert tensor[12].sum() == 64  # current player kingside
        assert tensor[13].sum() == 64  # current player queenside
        assert tensor[14].sum() == 64  # opponent kingside
        assert tensor[15].sum() == 64  # opponent queenside

    def test_batch_encoding(self, start_board, midgame_board):
        batch = boards_to_tensor([start_board, midgame_board])
        assert batch.shape == (2, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)


class TestEnPassant:
    def test_en_passant_square(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        # After e2e4, en passant square is e3
        tensor = board_to_tensor(board)
        # From black's perspective (flipped), e3 becomes e6 (row 5)
        # Plane 16 should have exactly one 1
        assert tensor[16].sum() == 1

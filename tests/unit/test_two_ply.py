"""Unit tests for TwoPlyPlanner."""

import chess
import pytest
import torch

from chess_master.config import ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.planner.two_ply import TwoPlyPlanner
from chess_master.chess_master_types import Phase


@pytest.fixture
def phase_b_model(tiny_config):
    """Create a model with Phase B (opponent head enabled)."""
    model = ChessMaster9001(tiny_config, active_phase=Phase.B)
    model.eval()
    return model


class TestTwoPlyPlanner:
    def test_returns_valid_move(self, phase_b_model, start_board):
        """Planner should return a legal move."""
        planner = TwoPlyPlanner(phase_b_model, k=3, r=2, device="cpu")
        move, info = planner.select_move(start_board)
        assert move in start_board.legal_moves

    def test_returns_valid_move_midgame(self, phase_b_model, midgame_board):
        """Planner should work on a middlegame position."""
        planner = TwoPlyPlanner(phase_b_model, k=3, r=2, device="cpu")
        move, info = planner.select_move(midgame_board)
        assert move in midgame_board.legal_moves

    def test_info_dict_structure(self, phase_b_model, start_board):
        """Info dict should have the expected keys."""
        planner = TwoPlyPlanner(phase_b_model, k=3, r=2, device="cpu")
        move, info = planner.select_move(start_board)

        assert "candidates" in info
        assert "selected_move" in info
        assert "one_ply_move" in info
        assert isinstance(info["candidates"], list)
        assert len(info["candidates"]) > 0

    def test_candidate_count(self, phase_b_model, start_board):
        """Should evaluate the requested number of candidates."""
        k = 4
        planner = TwoPlyPlanner(phase_b_model, k=k, r=2, device="cpu")
        move, info = planner.select_move(start_board)
        # Starting position has 20 legal moves, so we should get k candidates
        assert len(info["candidates"]) == k

    def test_opponent_replies_count(self, phase_b_model, start_board):
        """Each candidate should have up to r opponent replies."""
        r = 2
        planner = TwoPlyPlanner(phase_b_model, k=3, r=r, device="cpu")
        move, info = planner.select_move(start_board)

        for candidate in info["candidates"]:
            # Game shouldn't end after one move from starting position
            assert len(candidate["opponent_replies"]) <= r
            assert len(candidate["opponent_replies"]) > 0

    def test_q_values_are_weighted_sums(self, phase_b_model, start_board):
        """Q-value should equal sum of prob * value for opponent replies."""
        planner = TwoPlyPlanner(phase_b_model, k=3, r=3, device="cpu")
        move, info = planner.select_move(start_board)

        for candidate in info["candidates"]:
            if candidate["opponent_replies"]:
                expected_q = sum(
                    reply["prob"] * reply["value"]
                    for reply in candidate["opponent_replies"]
                )
                assert abs(candidate["q_value"] - expected_q) < 1e-5, (
                    f"Q-value {candidate['q_value']} != expected {expected_q}"
                )

    def test_different_k_values(self, phase_b_model, start_board):
        """Changing k should change the number of candidates evaluated."""
        planner_k2 = TwoPlyPlanner(phase_b_model, k=2, r=2, device="cpu")
        planner_k5 = TwoPlyPlanner(phase_b_model, k=5, r=2, device="cpu")

        _, info_k2 = planner_k2.select_move(start_board)
        _, info_k5 = planner_k5.select_move(start_board)

        assert len(info_k2["candidates"]) == 2
        assert len(info_k5["candidates"]) == 5

    def test_k_exceeds_legal_moves(self, phase_b_model, endgame_board):
        """When k > number of legal moves, should evaluate all legal moves."""
        planner = TwoPlyPlanner(phase_b_model, k=100, r=2, device="cpu")
        move, info = planner.select_move(endgame_board)
        num_legal = len(list(endgame_board.legal_moves))
        # candidates may be fewer if some moves can't be encoded
        assert len(info["candidates"]) <= num_legal
        assert len(info["candidates"]) > 0

    def test_selected_move_matches_best_q(self, phase_b_model, start_board):
        """Selected move should be the one with highest Q-value."""
        planner = TwoPlyPlanner(phase_b_model, k=3, r=2, device="cpu")
        move, info = planner.select_move(start_board)

        best_candidate = max(info["candidates"], key=lambda c: c["q_value"])
        assert info["selected_move"] == best_candidate["move"]

"""Unit tests for ProjectionPlanner."""

import chess
import pytest
import torch

from chess_master.config import ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.planner.projection import ProjectionPlanner
from chess_master.chess_master_types import Phase


@pytest.fixture
def phase_b_model(tiny_config):
    """Create a model with Phase B (opponent head enabled)."""
    model = ChessMaster9001(tiny_config, active_phase=Phase.B)
    model.eval()
    return model


class TestProjectionPlanner:
    def test_returns_valid_move(self, phase_b_model, start_board):
        """Planner should return a legal move."""
        planner = ProjectionPlanner(phase_b_model, k=3, horizon=4, device="cpu")
        move, info = planner.select_move(start_board)
        assert move in start_board.legal_moves

    def test_returns_valid_move_midgame(self, phase_b_model, midgame_board):
        """Planner should work on a middlegame position."""
        planner = ProjectionPlanner(phase_b_model, k=3, horizon=4, device="cpu")
        move, info = planner.select_move(midgame_board)
        assert move in midgame_board.legal_moves

    def test_info_dict_structure(self, phase_b_model, start_board):
        """Info dict should have the expected keys."""
        planner = ProjectionPlanner(phase_b_model, k=3, horizon=4, device="cpu")
        move, info = planner.select_move(start_board)

        assert "candidates" in info
        assert "selected_move" in info
        assert "one_ply_move" in info
        assert isinstance(info["candidates"], list)
        assert len(info["candidates"]) > 0

    def test_projected_line_length(self, phase_b_model, start_board):
        """Projected lines should not exceed the horizon."""
        horizon = 6
        planner = ProjectionPlanner(phase_b_model, k=2, horizon=horizon, device="cpu")
        move, info = planner.select_move(start_board)

        for candidate in info["candidates"]:
            # moves includes the root move plus up to (horizon-1) projected moves
            assert len(candidate["moves"]) <= horizon + 1
            assert len(candidate["values"]) <= horizon

    def test_projected_line_has_values(self, phase_b_model, start_board):
        """Each projected line should have values at each ply."""
        planner = ProjectionPlanner(phase_b_model, k=2, horizon=4, device="cpu")
        move, info = planner.select_move(start_board)

        for candidate in info["candidates"]:
            assert len(candidate["values"]) > 0
            assert "endpoint_value" in candidate
            assert "value_trend" in candidate
            assert "score" in candidate

    def test_handles_game_over_within_horizon(self, phase_b_model):
        """Should handle positions where the game ends within the horizon."""
        # Fool's mate setup: game can end quickly
        # Use a position close to checkmate
        board = chess.Board("rnbqkbnr/ppppp2p/5p2/6p1/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
        planner = ProjectionPlanner(phase_b_model, k=3, horizon=10, device="cpu")
        move, info = planner.select_move(board)
        assert move in board.legal_moves

    def test_candidate_count(self, phase_b_model, start_board):
        """Should evaluate the requested number of candidates."""
        k = 4
        planner = ProjectionPlanner(phase_b_model, k=k, horizon=4, device="cpu")
        move, info = planner.select_move(start_board)
        assert len(info["candidates"]) == k

    def test_score_includes_trend(self, phase_b_model, start_board):
        """Score should combine endpoint value and trend."""
        planner = ProjectionPlanner(phase_b_model, k=2, horizon=4, device="cpu")
        move, info = planner.select_move(start_board)

        for candidate in info["candidates"]:
            expected_score = candidate["endpoint_value"] + 0.1 * candidate["value_trend"]
            assert abs(candidate["score"] - expected_score) < 1e-5

    def test_different_horizons(self, phase_b_model, start_board):
        """Longer horizon should produce longer projected lines."""
        planner_short = ProjectionPlanner(phase_b_model, k=2, horizon=2, device="cpu")
        planner_long = ProjectionPlanner(phase_b_model, k=2, horizon=6, device="cpu")

        _, info_short = planner_short.select_move(start_board)
        _, info_long = planner_long.select_move(start_board)

        # Longer horizon should generally produce longer lines
        max_short = max(len(c["moves"]) for c in info_short["candidates"])
        max_long = max(len(c["moves"]) for c in info_long["candidates"])
        assert max_long >= max_short

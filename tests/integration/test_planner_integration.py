"""Integration tests for planner with the full inference pipeline."""

import chess
import pytest
import torch

from chess_master.config import ModelConfig, PlannerConfig
from chess_master.inference.player import ChessPlayer
from chess_master.model import ChessMaster9001
from chess_master.planner.two_ply import TwoPlyPlanner
from chess_master.planner.projection import ProjectionPlanner
from chess_master.types import Phase


@pytest.fixture
def phase_b_model(tiny_config):
    """Create a Phase B model."""
    model = ChessMaster9001(tiny_config, active_phase=Phase.B)
    model.eval()
    return model


class TestPlannerIntegration:
    def test_player_with_two_ply_planner(self, phase_b_model, start_board):
        """ChessPlayer with two_ply planner should select a valid move."""
        player = ChessPlayer(
            model=phase_b_model,
            device="cpu",
            planner="two_ply",
            planner_k=3,
            planner_r=2,
        )
        move, info = player.select_move(start_board)
        assert move in start_board.legal_moves
        assert "candidates" in info
        assert "value" in info

    def test_player_with_projection_planner(self, phase_b_model, start_board):
        """ChessPlayer with projection planner should select a valid move."""
        player = ChessPlayer(
            model=phase_b_model,
            device="cpu",
            planner="projection",
            planner_k=3,
            projection_horizon=4,
        )
        move, info = player.select_move(start_board)
        assert move in start_board.legal_moves
        assert "candidates" in info
        assert "value" in info

    def test_player_without_planner(self, phase_b_model, start_board):
        """ChessPlayer without planner should use one-step policy."""
        player = ChessPlayer(
            model=phase_b_model,
            device="cpu",
        )
        move, info = player.select_move(start_board)
        assert move in start_board.legal_moves
        assert "policy" in info

    def test_two_ply_may_differ_from_one_step(self, phase_b_model, midgame_board):
        """Two-ply and one-step may select different moves (not guaranteed)."""
        fen = midgame_board.fen()
        player_one_step = ChessPlayer(model=phase_b_model, device="cpu")
        player_two_ply = ChessPlayer(
            model=phase_b_model, device="cpu",
            planner="two_ply", planner_k=5, planner_r=3,
        )

        player_one_step.new_game(fen)
        player_two_ply.new_game(fen)

        move_one, _ = player_one_step.select_move(midgame_board)
        move_two, info_two = player_two_ply.select_move(midgame_board)

        # Both should be legal, but may differ
        assert move_one in midgame_board.legal_moves
        assert move_two in midgame_board.legal_moves

        # The info should record what one-ply would have chosen
        assert "one_ply_move" in info_two

    def test_projection_may_differ_from_one_step(self, phase_b_model, midgame_board):
        """Projection and one-step may select different moves."""
        fen = midgame_board.fen()
        player_one_step = ChessPlayer(model=phase_b_model, device="cpu")
        player_proj = ChessPlayer(
            model=phase_b_model, device="cpu",
            planner="projection", planner_k=5, projection_horizon=6,
        )

        player_one_step.new_game(fen)
        player_proj.new_game(fen)

        move_one, _ = player_one_step.select_move(midgame_board)
        move_proj, info_proj = player_proj.select_move(midgame_board)

        assert move_one in midgame_board.legal_moves
        assert move_proj in midgame_board.legal_moves
        assert "one_ply_move" in info_proj

    def test_session_records_planner_moves(self, phase_b_model, start_board):
        """Planner moves should be recorded in the game session."""
        player = ChessPlayer(
            model=phase_b_model, device="cpu",
            planner="two_ply", planner_k=3, planner_r=2,
        )
        move, info = player.select_move(start_board)
        assert player.session.move_count == 1

    def test_planner_config_dataclass(self):
        """PlannerConfig should have correct defaults."""
        config = PlannerConfig()
        assert config.mode == "none"
        assert config.top_k == 5
        assert config.top_r == 3
        assert config.projection_horizon == 10

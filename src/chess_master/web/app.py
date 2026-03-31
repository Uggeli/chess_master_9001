"""Flask web app for interactive chess against Chess Master 9001."""

import argparse
import json
import logging
import os

import chess
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

from chess_master.board.moves import index_to_move, legal_move_mask, move_to_index
from chess_master.board.tensor import board_to_tensor
from chess_master.board.utils import game_phase
from chess_master.config import ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.planner.projection import ProjectionPlanner
from chess_master.planner.two_ply import TwoPlyPlanner
from chess_master.training.callbacks import Checkpointer
from chess_master.chess_master_types import Phase

logger = logging.getLogger(__name__)

# Global state
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

model = None
board = None
game_history = []
two_ply_planner = None
projection_planner = None


def load_model(checkpoint_path, config_path=None):
    """Load the model from a checkpoint."""
    global model, two_ply_planner, projection_planner

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    if config_path:
        from omegaconf import OmegaConf
        config = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ModelConfig),
                OmegaConf.load(config_path).get("model", {}),
            )
        )
        model_config = config
    elif "config" in checkpoint and hasattr(checkpoint["config"], "model"):
        model_config = checkpoint["config"].model
    else:
        model_config = ModelConfig()

    model = ChessMaster9001(model_config, active_phase=Phase.B)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    two_ply_planner = TwoPlyPlanner(model, k=5, r=3)
    projection_planner = ProjectionPlanner(model, k=5, horizon=8)

    logger.info("Model loaded successfully")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global board, game_history
    data = request.get_json(silent=True) or {}
    fen = data.get("fen")

    board = chess.Board(fen) if fen else chess.Board()
    game_history = []

    return jsonify({
        "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
    })


@app.route("/api/move", methods=["POST"])
def make_move():
    global board, game_history
    data = request.get_json()
    human_move_uci = data.get("move")

    if board is None:
        board = chess.Board()

    # Validate and apply human move
    try:
        human_move = chess.Move.from_uci(human_move_uci)
        if human_move not in board.legal_moves:
            return jsonify({"error": f"Illegal move: {human_move_uci}"}), 400
    except (ValueError, chess.InvalidMoveError):
        return jsonify({"error": f"Invalid move format: {human_move_uci}"}), 400

    # Get value before human move
    human_value = _get_value(board)
    board.push(human_move)

    # Record human move
    game_history.append({
        "move": human_move_uci,
        "san": board.peek().uci(),  # already pushed
        "value": human_value,
        "delta": human_value - (game_history[-1]["value"] if game_history else 0),
        "side": "white",
        "move_number": board.fullmove_number,
    })

    # Check if game is over after human move
    if board.is_game_over():
        return jsonify({
            "human_move": human_move_uci,
            "bot_move": None,
            "fen": board.fen(),
            "analysis": None,
            "game_history": game_history,
            "game_over": True,
            "result": board.result(),
        })

    # Bot's turn — run analysis
    analysis = _analyze_position(board)

    # Pick the bot's move (use one-step policy for speed)
    bot_move_uci = analysis["policy"][0]["move"]
    bot_move = chess.Move.from_uci(bot_move_uci)
    bot_value = analysis["value"]

    board.push(bot_move)

    # Record bot move
    game_history.append({
        "move": bot_move_uci,
        "value": bot_value,
        "delta": bot_value - game_history[-1]["value"] if game_history else 0,
        "side": "black",
        "move_number": board.fullmove_number,
    })

    return jsonify({
        "human_move": human_move_uci,
        "bot_move": bot_move_uci,
        "fen": board.fen(),
        "analysis": analysis,
        "game_history": game_history,
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    fen = data.get("fen")

    if fen:
        analyze_board = chess.Board(fen)
    elif board is not None:
        analyze_board = board
    else:
        analyze_board = chess.Board()

    analysis = _analyze_position(analyze_board)
    return jsonify(analysis)


@torch.no_grad()
def _get_value(b):
    """Quick value estimate for a position."""
    bt = board_to_tensor(b).unsqueeze(0)
    mask = legal_move_mask(b).unsqueeze(0)
    output = model(bt, legal_mask=mask)
    return round(output.value[0, 0].item(), 4)


@torch.no_grad()
def _analyze_position(b):
    """Run full analysis on a position."""
    bt = board_to_tensor(b).unsqueeze(0)
    mask = legal_move_mask(b).unsqueeze(0)
    output = model(bt, legal_mask=mask)

    logits = output.policy_logits[0]
    probs = F.softmax(logits, dim=-1)
    value = output.value[0, 0].item()

    # Top moves with probabilities
    num_legal = int((mask[0] > 0).sum().item())
    topk_vals, topk_idx = torch.topk(probs, min(10, num_legal))

    policy = []
    for prob, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
        try:
            move = index_to_move(idx, b)
            policy.append({"move": move.uci(), "prob": round(prob, 4)})
        except ValueError:
            continue

    analysis = {
        "value": round(value, 4),
        "phase": game_phase(b),
        "policy": policy,
    }

    # Two-ply analysis
    try:
        _, two_ply_info = two_ply_planner.select_move(b)
        analysis["two_ply"] = {
            "selected": two_ply_info["selected_move"],
            "one_step_choice": two_ply_info["one_ply_move"],
            "agrees": two_ply_info["selected_move"] == two_ply_info["one_ply_move"],
            "candidates": [
                {
                    "move": c["move"],
                    "q_value": round(c["q_value"], 4),
                    "replies": [
                        {
                            "move": r["move"],
                            "prob": round(r["prob"], 4),
                            "value": round(r["value"], 4),
                        }
                        for r in c.get("opponent_replies", [])
                    ],
                }
                for c in two_ply_info.get("candidates", [])
            ],
        }
    except Exception as e:
        logger.warning(f"Two-ply analysis failed: {e}")
        analysis["two_ply"] = None

    # Projection analysis
    try:
        _, proj_info = projection_planner.select_move(b)
        analysis["projection"] = {
            "selected": proj_info["selected_move"],
            "one_step_choice": proj_info["one_ply_move"],
            "lines": [
                {
                    "root": c.get("root_move", c["moves"][0] if c["moves"] else ""),
                    "moves": c["moves"],
                    "values": [round(v, 4) for v in c.get("values", [])],
                    "score": round(c.get("score", 0), 4),
                    "endpoint_value": round(c.get("endpoint_value", 0), 4),
                    "value_trend": round(c.get("value_trend", 0), 4),
                }
                for c in proj_info.get("candidates", [])
            ],
        }
    except Exception as e:
        logger.warning(f"Projection analysis failed: {e}")
        analysis["projection"] = None

    return analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Chess Master 9001 Web UI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    load_model(args.checkpoint, args.config)

    # Initialize board
    global board
    board = chess.Board()

    logger.info(f"Starting Chess Master 9001 Web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

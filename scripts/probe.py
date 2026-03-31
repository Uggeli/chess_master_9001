"""Interactive probe tool for exploring model thinking.

Lets you set up positions and see:
- What moves the model thinks are best (one-step policy)
- Two-ply analysis: "if I play X, opponent plays Y, resulting position is worth Z"
- Long projection: full projected game lines for each candidate move
- Compare one-step vs two-ply vs projection move choices
"""

import argparse
import logging

import chess
import torch
from omegaconf import OmegaConf

from chess_master.board.moves import index_to_move, legal_move_mask
from chess_master.board.tensor import board_to_tensor
from chess_master.board.utils import game_phase
from chess_master.config import Config, ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.planner.projection import ProjectionPlanner
from chess_master.planner.two_ply import TwoPlyPlanner
from chess_master.training.callbacks import Checkpointer
from chess_master.chess_master_types import Phase


def load_model(checkpoint_path, config_path=None):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    if config_path:
        config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(config_path))
        )
        model_config = config.model
    elif "config" in checkpoint:
        config = checkpoint["config"]
        model_config = config.model if hasattr(config, "model") else ModelConfig()
    else:
        model_config = ModelConfig()

    # Use Phase B so opponent head is available for planning
    model = ChessMaster9001(model_config, active_phase=Phase.B)
    # Load with strict=False since Phase B head won't be in Phase A checkpoint
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, model_config


def show_policy(model, board, top_k=10):
    """Show one-step policy distribution."""
    bt = board_to_tensor(board).unsqueeze(0)
    mask = legal_move_mask(board).unsqueeze(0)

    with torch.no_grad():
        output = model(bt, legal_mask=mask)

    logits = output.policy_logits[0]
    probs = torch.softmax(logits, dim=-1)
    value = output.value[0, 0].item()

    entropy = -(probs * (probs + 1e-8).log()).sum().item()
    num_legal = int((mask[0] > 0).sum().item())
    actual_k = min(top_k, num_legal)
    topk_vals, topk_idx = torch.topk(probs, actual_k)

    print(f"\n  One-Step Policy (entropy: {entropy:.2f}, value: {value:+.4f})")
    print(f"  {'#':<4} {'Move':<8} {'Prob':>7} {'Logit':>7}")
    print(f"  {'─' * 30}")
    for rank, (prob, idx) in enumerate(zip(topk_vals.tolist(), topk_idx.tolist()), 1):
        try:
            move = index_to_move(idx, board)
            logit = logits[idx].item()
            marker = " <--" if rank == 1 else ""
            print(f"  {rank:<4} {move.uci():<8} {prob:>6.1%} {logit:>7.2f}{marker}")
        except ValueError:
            continue


def show_two_ply(model, board, k=5, r=3):
    """Show two-ply analysis."""
    planner = TwoPlyPlanner(model, k=k, r=r)
    move, info = planner.select_move(board)

    one_ply = info["one_ply_move"]
    selected = info["selected_move"]
    agrees = one_ply == selected

    print(f"\n  Two-Ply Analysis (k={k} candidates, r={r} replies)")
    print(f"  One-step choice: {one_ply} | Two-ply choice: {selected}", end="")
    print(f" {'(agree)' if agrees else '(DISAGREE!)'}")
    print()

    for c in sorted(info["candidates"], key=lambda x: x["q_value"], reverse=True):
        marker = " ***" if c["move"] == selected else ""
        print(f"  {c['move']:<8} Q={c['q_value']:+.4f}{marker}")
        if c["opponent_replies"]:
            for rep in c["opponent_replies"]:
                print(f"    └─ opp {rep['move']:<8} p={rep['prob']:.1%}  V={rep['value']:+.4f}")


def show_projection(model, board, k=5, horizon=10):
    """Show projected game lines."""
    planner = ProjectionPlanner(model, k=k, horizon=horizon)
    move, info = planner.select_move(board)

    one_ply = info["one_ply_move"]
    selected = info["selected_move"]

    print(f"\n  Projection Analysis (k={k}, horizon={horizon} plies)")
    print(f"  One-step: {one_ply} | Projection: {selected}")
    print()

    for c in sorted(info["candidates"], key=lambda x: x["score"], reverse=True):
        marker = " ***" if c.get("root_move") == selected else ""
        root = c.get("root_move", c["moves"][0] if c["moves"] else "?")
        ep = c["endpoint_value"]
        trend = c["value_trend"]
        score = c["score"]

        print(f"  {root:<8} score={score:+.4f} (endpoint={ep:+.4f}, trend={trend:+.4f}){marker}")

        # Show the projected line
        line_str = " ".join(c["moves"])
        if len(line_str) > 70:
            line_str = line_str[:67] + "..."
        print(f"    line: {line_str}")

        # Show value curve
        if c["values"]:
            vals = c["values"]
            min_v, max_v = min(vals), max(vals)
            print(f"    values: [{', '.join(f'{v:+.2f}' for v in vals[:8])}", end="")
            if len(vals) > 8:
                print(f", ... {vals[-1]:+.2f}]")
            else:
                print("]")
        print()


def show_full_analysis(model, board, k=5, r=3, horizon=10):
    """Run all analyses on a position."""
    print(f"\n{'═' * 60}")
    print(f"  Position: {board.fen()}")
    print(f"  Side: {'White' if board.turn else 'Black'} | Phase: {game_phase(board)} | Legal moves: {board.legal_moves.count()}")
    print(f"{'═' * 60}")

    show_policy(model, board)
    show_two_ply(model, board, k=k, r=r)
    show_projection(model, board, k=k, horizon=horizon)


def interactive_loop(model, k=5, r=3, horizon=10):
    """Interactive REPL for probing the model."""
    board = chess.Board()

    print("\nChess Master 9001 — Interactive Probe")
    print("Commands:")
    print("  fen <FEN>       Set position from FEN")
    print("  move <uci>      Play a move (e.g. 'move e2e4')")
    print("  undo            Undo last move")
    print("  reset           Reset to starting position")
    print("  board           Show current board")
    print("  policy [N]      Show top-N policy moves (default 10)")
    print("  2ply            Two-ply analysis")
    print("  project [H]     Projection analysis (H=horizon, default 10)")
    print("  all             Full analysis (policy + 2ply + projection)")
    print("  play [N]        Auto-play N moves from current position")
    print("  quit            Exit")
    print()
    print(board)
    print()

    while True:
        try:
            cmd = input("probe> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        try:
            if action == "quit":
                break

            elif action == "fen":
                fen = " ".join(parts[1:])
                board = chess.Board(fen)
                print(board)
                print()

            elif action == "move":
                uci = parts[1]
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    board.push(move)
                    print(board)
                    print()
                else:
                    print(f"  Illegal move: {uci}")

            elif action == "undo":
                if board.move_stack:
                    board.pop()
                    print(board)
                    print()
                else:
                    print("  No moves to undo")

            elif action == "reset":
                board = chess.Board()
                print(board)
                print()

            elif action == "board":
                print(board)
                print(f"  FEN: {board.fen()}")
                print()

            elif action == "policy":
                n = int(parts[1]) if len(parts) > 1 else 10
                show_policy(model, board, top_k=n)

            elif action == "2ply":
                show_two_ply(model, board, k=k, r=r)

            elif action == "project":
                h = int(parts[1]) if len(parts) > 1 else horizon
                show_projection(model, board, k=k, horizon=h)

            elif action == "all":
                show_full_analysis(model, board, k=k, r=r, horizon=horizon)

            elif action == "play":
                n = int(parts[1]) if len(parts) > 1 else 5
                for i in range(n * 2):
                    if board.is_game_over():
                        print(f"  Game over: {board.result()}")
                        break
                    bt = board_to_tensor(board).unsqueeze(0)
                    mask = legal_move_mask(board).unsqueeze(0)
                    with torch.no_grad():
                        out = model(bt, legal_mask=mask)
                    move_idx = out.policy_logits[0].argmax().item()
                    move = index_to_move(move_idx, board)
                    value = out.value[0, 0].item()
                    side = "W" if board.turn == chess.WHITE else "B"
                    print(f"  {board.fullmove_number}.{'  ' if board.turn == chess.BLACK else ''}{side} {move.uci()} (v={value:+.3f})")
                    board.push(move)
                print()
                print(board)
                print()

            else:
                print(f"  Unknown command: {action}")
                print("  Type 'quit' to exit")

        except Exception as e:
            print(f"  Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe model thinking interactively")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fen", type=str, default=None, help="Analyze this FEN and exit (non-interactive)")
    parser.add_argument("--k", type=int, default=5, help="Top-k candidates")
    parser.add_argument("--r", type=int, default=3, help="Top-r opponent replies")
    parser.add_argument("--horizon", type=int, default=10, help="Projection horizon (plies)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)

    model, _ = load_model(args.checkpoint, args.config)

    if args.fen:
        board = chess.Board(args.fen)
        show_full_analysis(model, board, k=args.k, r=args.r, horizon=args.horizon)
    else:
        interactive_loop(model, k=args.k, r=args.r, horizon=args.horizon)


if __name__ == "__main__":
    main()

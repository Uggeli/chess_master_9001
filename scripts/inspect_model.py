"""Inspect a trained model on a given position."""

import argparse
import logging

import chess
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from chess_master.board.tensor import board_to_tensor
from chess_master.board.moves import index_to_move, legal_move_mask
from chess_master.board.utils import game_phase
from chess_master.config import Config, ModelConfig
from chess_master.model import ChessMaster9001
from chess_master.types import NUM_ACTIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect model on a position")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (uses checkpoint config if omitted)")
    parser.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="FEN string to analyze")
    parser.add_argument("--show-policy", action="store_true", help="Show top moves with probabilities")
    parser.add_argument("--show-retrieval", action="store_true", help="Show retrieval information")
    parser.add_argument("--show-value", action="store_true", help="Show value prediction")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top moves to show")
    parser.add_argument("--all", action="store_true", help="Show all information")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")

    # Load config
    if args.config:
        config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(args.config))
        )
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        logging.warning("No config found, using defaults")
        config = Config()

    model_config = config.model if hasattr(config, "model") else ModelConfig()

    # Build and load model
    model = ChessMaster9001(model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Analyze position
    board = chess.Board(args.fen)
    board_tensor = board_to_tensor(board).unsqueeze(0)
    mask = legal_move_mask(board).unsqueeze(0)

    with torch.no_grad():
        output = model(board_tensor, legal_mask=mask)

    show_all = args.all

    # Header
    print(f"\n{'=' * 60}")
    print(f"Position: {args.fen}")
    print(f"Side to move: {'White' if board.turn else 'Black'}")
    print(f"Game phase: {game_phase(board)}")
    print(f"Legal moves: {board.legal_moves.count()}")
    print(f"{'=' * 60}")

    # Value
    if args.show_value or show_all:
        value = output.value[0, 0].item()
        print(f"\nValue prediction: {value:+.4f}")
        print(f"  (1.0 = winning, -1.0 = losing, 0.0 = equal)")

    # Policy
    if args.show_policy or show_all:
        logits = output.policy_logits[0]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()

        topk_vals, topk_idx = torch.topk(probs, min(args.top_k, (mask[0] > 0).sum().item()))

        print(f"\nTop {args.top_k} moves (entropy: {entropy:.2f}):")
        print(f"  {'Move':<10} {'Prob':>8} {'Logit':>8}")
        print(f"  {'-' * 28}")
        for prob, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            try:
                move = index_to_move(idx, board)
                logit = logits[idx].item()
                print(f"  {move.uci():<10} {prob:>8.4f} {logit:>8.2f}")
            except ValueError:
                continue

    # Retrieval
    if args.show_retrieval or show_all:
        if output.retrieval_info is not None:
            ri = output.retrieval_info
            if "similarity_scores" in ri:
                scores = ri["similarity_scores"]
                print(f"\nRetrieval info:")
                print(f"  Mean similarity: {scores.mean().item():.4f}")
                print(f"  Max similarity:  {scores.max().item():.4f}")
                print(f"  Min similarity:  {scores.min().item():.4f}")
        else:
            print(f"\nNo retrieval info (no memory provided)")

    print()


if __name__ == "__main__":
    main()

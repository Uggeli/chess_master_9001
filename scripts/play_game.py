"""Play a game interactively or run self-play."""

import argparse
import logging

import chess
import torch
from omegaconf import OmegaConf

from chess_master.config import Config, ModelConfig
from chess_master.inference.player import ChessPlayer
from chess_master.model import ChessMaster9001


def parse_args():
    parser = argparse.ArgumentParser(description="Play chess with Chess Master 9001")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--mode", choices=["interactive", "self-play"], default="interactive")
    parser.add_argument("--fen", type=str, default=None, help="Starting FEN")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves for self-play")
    return parser.parse_args()


def load_model(args):
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")

    if args.config:
        config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(args.config))
        )
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = Config()

    model_config = config.model if hasattr(config, "model") else ModelConfig()
    model = ChessMaster9001(model_config)
    model.load_state_dict(checkpoint["model"])
    return model


def interactive_mode(player: ChessPlayer, starting_fen: str | None):
    """Play interactively: human vs model."""
    player.new_game(starting_fen)
    board = chess.Board(starting_fen) if starting_fen else chess.Board()

    print("Chess Master 9001 - Interactive Mode")
    print("Enter moves in UCI format (e.g., e2e4)")
    print("Type 'quit' to exit, 'board' to show position\n")
    print(board)
    print()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Human plays white
            while True:
                user_input = input("Your move: ").strip()
                if user_input == "quit":
                    return
                if user_input == "board":
                    print(board)
                    continue
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except (ValueError, chess.InvalidMoveError):
                    print("Invalid format. Use UCI notation (e.g., e2e4)")
        else:
            # Model plays black
            move, info = player.select_move(board)
            board.push(move)
            print(f"\nModel plays: {move.uci()} (value: {info['value']:+.3f})")
            if info["policy"]:
                top3 = info["policy"][:3]
                alternatives = ", ".join(f"{m['move']}({m['probability']:.2f})" for m in top3)
                print(f"  Top moves: {alternatives}")

        print(f"\n{board}\n")

    print(f"Game over: {board.result()}")


def self_play_mode(player: ChessPlayer, starting_fen: str | None, max_moves: int):
    """Run self-play and print the game."""
    records = player.play_game(max_moves=max_moves, starting_fen=starting_fen)

    board = chess.Board(starting_fen) if starting_fen else chess.Board()

    print("Chess Master 9001 - Self-Play")
    print(f"Starting position: {board.fen()}\n")

    for record in records:
        side = record["side"]
        move = record["move"]
        value = record["value"]
        move_num = record["move_number"]

        if side == "white":
            print(f"{move_num}. {move} (v={value:+.3f})", end="  ")
        else:
            print(f"{move} (v={value:+.3f})")

        board.push(chess.Move.from_uci(move))

    print(f"\n\nResult: {records[-1].get('game_result', '*') if records else '*'}")
    print(f"\nFinal position:\n{board}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)

    model = load_model(args)
    player = ChessPlayer(model, temperature=args.temperature)

    if args.mode == "interactive":
        interactive_mode(player, args.fen)
    else:
        self_play_mode(player, args.fen, args.max_moves)


if __name__ == "__main__":
    main()

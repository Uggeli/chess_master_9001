"""Generate Stockfish-labeled training data from PGN files."""

import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training data from PGN files")
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--output", type=str, required=True, help="Output path (.jsonl)")
    parser.add_argument("--stockfish-path", type=str, default="", help="Path to Stockfish binary")
    parser.add_argument("--depth", type=int, default=16, help="Stockfish search depth")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process")
    parser.add_argument("--max-positions", type=int, default=None, help="Max positions to generate")
    parser.add_argument("--skip-opening", type=int, default=6, help="Skip first N moves per game")
    parser.add_argument("--to-npz", type=str, default=None, help="Also convert to NPZ at this path")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel Stockfish workers")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Stockfish threads per worker")
    parser.add_argument("--hash-per-worker", type=int, default=64, help="Stockfish hash MB per worker")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from chess_master.data.pipeline import generate_from_pgn, jsonl_to_npz

    count = generate_from_pgn(
        pgn_path=args.pgn,
        output_path=args.output,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.depth,
        max_games=args.max_games,
        max_positions=args.max_positions,
        skip_opening_moves=args.skip_opening,
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
        hash_per_worker=args.hash_per_worker,
    )

    logging.info(f"Generated {count} positions")

    if args.to_npz:
        jsonl_to_npz(args.output, args.to_npz)
        logging.info(f"Converted to NPZ: {args.to_npz}")


if __name__ == "__main__":
    main()

"""Split NPZ dataset into train and validation sets."""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Split NPZ into train/val")
    parser.add_argument("--input", type=str, required=True, help="Input NPZ path")
    parser.add_argument("--train", type=str, required=True, help="Output train NPZ path")
    parser.add_argument("--val", type=str, required=True, help="Output val NPZ path")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data = np.load(args.input)
    n = len(data["board_tensors"])

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n)

    val_size = int(n * args.val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    for path, idx in [(args.train, train_idx), (args.val, val_idx)]:
        np.savez(
            path,
            board_tensors=data["board_tensors"][idx],
            target_moves=data["target_moves"][idx],
            target_values=data["target_values"][idx],
            legal_masks=data["legal_masks"][idx],
        )
        print(f"{path}: {len(idx)} positions")


if __name__ == "__main__":
    main()

"""Train Chess Master 9001."""

import argparse
import logging
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from chess_master.config import Config
from chess_master.model import ChessMaster9001
from chess_master.training.trainer import Trainer
from chess_master.types import Phase


def parse_args():
    parser = argparse.ArgumentParser(description="Train Chess Master 9001")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides in key=value format (e.g. training.lr=0.001)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load config
    schema = OmegaConf.structured(Config)
    file_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(args.override)
    config = OmegaConf.merge(schema, file_conf, cli_conf)
    config = OmegaConf.to_object(config)

    # Set seed
    torch.manual_seed(config.seed)

    logging.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(config))}")

    # Build model
    model = ChessMaster9001(config.model, active_phase=config.active_phase)
    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {param_count:,}")

    # Load data
    # Support both JSONL and NPZ formats
    train_path = config.data.train_path
    if train_path.endswith(".npz"):
        from chess_master.data.dataset import ChessPositionNpzDataset
        train_dataset = ChessPositionNpzDataset(train_path)
    else:
        from chess_master.data.dataset import ChessPositionDataset
        train_dataset = ChessPositionDataset(train_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if config.data.val_path:
        val_path = config.data.val_path
        if val_path.endswith(".npz"):
            from chess_master.data.dataset import ChessPositionNpzDataset
            val_dataset = ChessPositionNpzDataset(val_path)
        else:
            from chess_master.data.dataset import ChessPositionDataset
            val_dataset = ChessPositionDataset(val_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

    # Build trainer
    trainer = Trainer(config, model)

    # Resume from checkpoint
    if args.resume:
        from chess_master.training.callbacks import Checkpointer
        step = Checkpointer.load(args.resume, model, trainer.optimizer)
        trainer.global_step = step
        logging.info(f"Resumed from step {step}")

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

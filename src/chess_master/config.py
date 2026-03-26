"""Structured configuration for Chess Master 9001."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from chess_master.types import Phase


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    # Memory
    memory_size: int = 10_000
    retrieval_k: int = 8
    retrieval_dim: int = 128
    # Short-term
    short_term_window: int = 10


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    retrieval: float = 0.1
    opponent: float = 0.5
    confidence: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_steps: int = 100_000
    eval_every: int = 1000
    checkpoint_every: int = 5000
    phase: str = "A"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    gradient_clip: float = 1.0


@dataclass
class DataConfig:
    train_path: str = MISSING
    val_path: str = MISSING
    num_workers: int = 4
    stockfish_depth: int = 16
    stockfish_path: str = ""  # empty = use pip-installed default


@dataclass
class EvalConfig:
    sentinel_fens_path: str = ""
    puzzle_path: str = ""
    self_play_games: int = 100
    stockfish_eval_positions: int = 10_000


@dataclass
class PlannerConfig:
    mode: str = "none"  # "none", "two_ply", "projection"
    top_k: int = 5
    top_r: int = 3
    projection_horizon: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    seed: int = 42
    active_phase: int = 1  # Phase.A = 1
    run_dir: str = "runs/default"
    device: str = "cuda"

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chess Master 9001 is a memory-assisted Bayesian chess model combining a transformer encoder, Hopfield episodic memory, and short-horizon planning. Full design: `Docs/model_proposition.md`.

## Environment & Setup

- Python 3.14 (free-threaded build), venv at `.venv`
- Activate: `source .venv/bin/activate`
- Install: `pip install -e ".[dev]"` (add `stockfish` extra for data generation)

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run a single test file
pytest tests/unit/test_board_tensor.py -v

# Run integration tests
pytest tests/integration/ -v

# Train (requires data)
python scripts/train.py --config configs/phase_a.yaml

# Generate training data from PGN
python scripts/generate_data.py --pgn data/raw/games.pgn --output data/processed/train.jsonl --depth 16

# Inspect model on a position
python scripts/inspect_model.py --checkpoint runs/default/checkpoints/checkpoint.pt --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" --all

# Play interactively or self-play
python scripts/play_game.py --checkpoint <path> --mode interactive
python scripts/play_game.py --checkpoint <path> --mode self-play
```

## Architecture

Phase A is implemented. Phases B-D are scaffolded (empty packages ready).

### Data flow

```
Board (python-chess) → board/tensor.py [B,18,8,8]
  → encoder/backbone.py (transformer) → z_global [B,d_model] + spatial [B,64,d_model]
    ├→ projections.py (PolicyValue) → heads/policy.py → logits [B,4672]
    │                               → heads/value.py  → value [B,1]
    └→ projections.py (Retrieval)   → memory/hopfield.py → retrieved context
```

### Key design constraints

- **Dual projection heads**: policy/value and retrieval projections are separate MLPs on the shared backbone. This prevents policy gradients from collapsing the retrieval embedding geometry.
- **Perspective convention**: board tensors and evaluation values are always encoded from the side-to-move's perspective. Never mix white-centric and side-to-move conventions.
- **Action space**: AlphaZero 4672 (73 move types x 64 source squares). Legal moves masked with `-inf` logits.
- **Memory acts as prior**: trust is controlled by retrieval similarity thresholds, not learned confidence (until Phase D).

### Module map

- `src/chess_master/model.py` — top-level `ChessMaster9001` nn.Module composing all components
- `src/chess_master/board/` — tensor encoding, move index encoding (4672 action space), utilities
- `src/chess_master/encoder/` — transformer backbone, dual projection heads, short-term context encoder
- `src/chess_master/memory/` — Hopfield retrieval layer, memory store (CRUD + persistence), curation
- `src/chess_master/heads/` — policy head, value head (Phase B: opponent.py, Phase D: confidence.py)
- `src/chess_master/training/` — losses (policy CE + value MSE + retrieval contrastive), trainer, callbacks (TensorBoard, NaN detection, checkpointing)
- `src/chess_master/data/` — Stockfish labeler, JSONL/NPZ datasets, PGN pipeline
- `src/chess_master/inference/` — game session (short-term buffer), player (full inference pipeline)
- `src/chess_master/eval/` — accuracy by game phase, memory ablation, self-play Elo

### Config

OmegaConf dataclass schemas in `config.py`. YAML configs in `configs/`. Hierarchy: dataclass defaults <- base.yaml <- phase_X.yaml <- CLI overrides. Every training run freezes config to `runs/<run_id>/config.yaml`.

## Phased Implementation

- **Phase A** (implemented): One-step predictor — policy/value heads, Hopfield retrieval, supervised by Stockfish
- **Phase B** (scaffolded): Opponent reply prediction — `heads/opponent.py`
- **Phase C** (scaffolded): Two-ply expected-value planning — `planner/two_ply.py`, `planner/projection.py`
- **Phase D** (scaffolded): Learned memory confidence — `heads/confidence.py`, `merger/learned.py`

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chess Master 9001 is a memory-assisted Bayesian chess model. The full design is in `Docs/model_proposition.md`. No code exists yet — the project is in the design/planning phase.

## Environment

- Python 3.14 (free-threaded build) with venv at `.venv`
- Activate: `source .venv/bin/activate`

## Design Reference

See `Docs/model_proposition.md` for the complete architecture and phased implementation plan. Key points:

- Combines transformer encoder, Hopfield episodic memory, opponent modeling, and short-horizon planning
- Dual memory system: long-term (persistent across games) and short-term (per-game trajectory buffer)
- Phased build: one-step predictor (A) -> opponent reply model (B) -> two-ply planning (C) -> learned confidence (D)
- Encoder needs separate projection heads for policy/value vs memory retrieval
- All evaluation values must use a consistent perspective convention

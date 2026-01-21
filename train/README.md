# Training Guide

## Overview

This folder contains PPO training scripts for the Ludo environment with dice banking and explicit die selection.

## Files

- **`single.py`**: PPO training script for single-player mode (self-play with shared weights)
- **`eval.py`**: Load checkpoint and render evaluation games

## Quick Start

```bash
# Train
python train/single.py

# Evaluate (requires checkpoint)
python train/eval.py
```

## Environment Details

- **Action space**: `Discrete(65)` = `(4 pieces × 16 dice slots) + 1 PASS`
  - Actions encode `(piece_index, dice_index)` pairs
  - Agents explicitly choose which die from their bank to use
- **Observation space**: `Box(0.0, 1.0, shape=(86,))`
  - Indices 0-69: Core game state
  - Indices 70-85: Full dice bank (up to 16 dice, normalized)
- **Action mask**: Available in `info["action_mask"]` with length 65

## Key Features

- **Dice banking**: Dice accumulate in a bank; agents choose which to use
- **Three-sixes penalty**: Rolling three consecutive 6s cancels that roll attempt
- **Extra turns**: Granted on finishing pieces, capturing enemies, or rolling 6
- **Checkpoints**: Saved to `train/checkpoints/single.pt`

## Training Configuration

See `single.py` for hyperparameters (learning rate, PPO epochs, entropy coefficient, etc.). The `ActorCritic` network auto-infers action/observation dimensions from the environment.

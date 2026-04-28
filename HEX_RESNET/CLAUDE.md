# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hex 11×11 AI playground (Paris 8 L3). Six pure-Python players: Alpha-Beta, AlphaZero MCTS, light MCTS (UCT, no net), pure Monte Carlo, BFS heuristic, random. No build step — `pip install torch numpy numba`.

Code, comments, and CLI output are written in **French** — match that style when editing.

## Common commands

All scripts must be run from `alphazero/`.

```bash
# Tournament between two players (keywords: alphabeta, random, alphazero, mc_pure, mcts_light, heuristic)
python tournament.py alphabeta random 20
python tournament.py alphabeta alphazero 10 -v -t 2.0     # -v verbose, -t seconds/move
python tournament.py alphabeta ./external_binary 20       # any non-keyword is treated as an external subprocess

# Unified ranking (incremental — only missing matchups are replayed; writes rank/ranking.{csv,html})
python ranking.py --games 20
python ranking.py --mode classic --games 20
python ranking.py --mode alphazero --games 20

# AlphaZero training
python train/trainer.py --iterations 1 --games 10 --simulations 100                          # smoke test (CPU)
python train/trainer.py --iterations 100 --games 100 --simulations 800 --steps 500 --device cuda
python train/trainer.py --iterations 20 --games 50 --simulations 200 --no-eval

# Targeted checks
python train/evaluate.py                                                     # win rate vs random
python train/network.py                                                      # mini training-loss sanity check
python -c "from train.hex_env import HexEnv; print(HexEnv().get_state_tensor().shape)"  # → (3, 11, 11)

# Housekeeping
make clean                                                                   # purge __pycache__ / *.pyc
```

`train_colab.ipynb` (repo root) is the Colab counterpart of `trainer.py` with FP16 + multi-process self-play.

## Per-AI CLI protocol

Every player in `ia/` is also a standalone CLI: `python <ia>.py BOARD PLAYER [time_s]`.
- `BOARD`: 121 row-major chars, `.` empty, `O` Blue (N→S), `@` Red (W→E)
- stdout: move in `A1`..`K11` notation; stderr: search stats

Example (empty board): `python ia/alphabeta.py "$(python -c 'print("."*121)')" O`

## Architecture

### Game engine — `train/hex_env.py` (single source of truth)
Two `bool` 11×11 numpy arrays (`blue`, `red`). Win check via BFS flood-fill. `get_state_tensor()` returns a `(3, 11, 11)` float32 tensor (Blue plane, Red plane, current-player plane). `mirror()` provides the diagonal-reflection symmetry used for data augmentation. `apply_move`/`undo_move` enable in-place search (Alpha-Beta uses undo, MCTS uses copy).

### Player interface — `ia/`
Each Python player exposes `select_move(env: HexEnv, time_s: float) -> int` and writes per-move diagnostics to `self.last_stats`. `tournament.py` and `ranking.py` consume that contract.

- **`alphabeta.py`** — Numba-JITted BFS 0-1 heuristic (`Red_path − Blue_path`). Adaptive depth based on legal-move count; Zobrist transposition table; killer moves + history heuristic; root-level move ordering. Exports `AlphaBetaPlayer` and `eval_heuristic()`.
- **`mcts_az.py`** — UCB-PUCT (`c_puct=1.0`), Dirichlet root noise (α=0.03, ε=0.25) in self-play, τ=1 for first 20 plies then argmax, virtual-loss batched GPU inference (~16–32 leaves/batch), tree reuse between moves, FP16 autocast.
- **`mcts_light.py`** — UCT, no network. **`monte_carlo_pure.py`** — random rollouts, no tree. **`heuristic_player.py`** — greedy via shortest virtual path. **`random_player.py`** — uniform.
- **`play.py`** (alphazero root) — CLI wrapper that loads `checkpoints/best_model.pt` and scales simulation count to the time budget.

### AlphaZero pipeline — `train/`
- **`network.py`** — ResNet (~6 blocks × 128 filters in current code; double-check before claiming a count) with policy + value heads, `predict()` / `batch_predict()`, FP16 via `torch.amp.autocast` on CUDA.
- **`self_play.py`** — `GameSlot` per game; `run_self_play` runs ~8 games concurrently and cross-batches positions (~64) into a single GPU forward. Circular replay buffer (~150k positions). See `train/self_play.txt` for design notes.
- **`trainer.py`** — self-play → train → evaluate loop. Loss = `MSE(v,z) + CE(p,π) + λ·L2` (λ=1e-4). Adam lr=1e-3 with cosine annealing to 1e-5. A new model replaces `best_model.pt` only at ≥55% win rate over the eval games. **When evaluation rejects a model, the optimizer + scheduler are reset** alongside the revert to `best_model.pt` — preserve that behavior when refactoring.
- **`evaluate.py`** — pairwise model comparison + vs-random baseline.

### Tournament & ranking
- `tournament.py` — head-to-head; alternates colors each game; accepts external subprocess opponents.
- `ranking.py` — round-robin across all classic + AZ variants. The CSV in `rank/ranking.csv` is **incremental**: re-runs only fill missing matchups, so don't blow it away unless you intend a fresh leaderboard. HTML report regenerated each run.
- `model_naming.py` — names accepted models `model_<num>_<parent>_<DD>_<MM>.pt` and copies them into `model/`. `model_rank/` is a read-only archive.

## Operational notes

- **Architecture changes invalidate checkpoints.** If you change `network.py` (block count, filters, head shape), delete `checkpoints/best_model.pt`, `checkpoints/model_iter_*.pt`, and `checkpoints/replay_buffer.npz` before training, otherwise loading will fail or silently mis-shape.
- ROCm users on RX 6600 / gfx1032: `export HSA_OVERRIDE_GFX_VERSION=10.3.0`. Pass `--device cuda` (PyTorch ROCm uses the CUDA backend names).
- Color convention: Blue = `O` plays N→S, Red = `@` plays W→E. Never swap silently — many heuristics encode the orientation.

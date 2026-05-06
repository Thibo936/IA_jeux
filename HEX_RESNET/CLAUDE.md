# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AlphaZero-style trainer + benchmarking harness for Hex 11×11 (Paris 8 L3 project). Pure Python (PyTorch, numpy, numba). Six native AI families plus AlphaZero variants and several "LLM" players (one Python file per LLM-generated agent).

Board convention: Blue (`O`) connects North→South; Red (`@`) connects West→East. Cells are indexed row-major `0..120`, alphanumeric notation `A1..K11`.

## Important: scripts run from `alphazero/`

All Python entry points (`tournament.py`, `ranking.py`, `versus.py`, `play.py`, `train/trainer.py`) bootstrap their `sys.path` by computing `__file__`'s directory and inserting `alphazero/`, `alphazero/ia/`, and `alphazero/train/`. **Always `cd alphazero/` before running them** — relative checkpoint paths (`checkpoints/best_model.pt`, `model/`) are resolved against CWD.

## Common commands

```bash
cd alphazero

# Self-play training (AlphaZero loop)
python train/trainer.py --iterations 100 --games 100 --simulations 800 --device cuda
python train/trainer.py --iterations 1 --games 10 --simulations 100        # smoke test
python train/trainer.py --iterations 20 --games 50 --no-eval               # skip eval

# Two-player tournament (CLI mnemonics resolved in tournament._resolve_ai)
python tournament.py alphabeta random 20
python tournament.py alphabeta alphazero 10 -v -t 2.0
python tournament.py alphabeta ./external_binary 20 -t 1.5                 # external = subprocess

# Round-robin ranking → rank/ranking.html (CSV is incremental)
python ranking.py --games 50
python ranking.py --mode classic           # only non-AZ
python ranking.py --mode alphazero         # only AZ models
python ranking.py --add path/to/model.pt   # add an extra checkpoint

# Two-AZ duel (just for picking between checkpoints)
python versus.py best checkpoints/model_iter_10.pt 20 -s 400

# Single-move CLI used by tournament for external/Python AIs (BOARD/PLAYER protocol)
python play.py "<121-char board>" O 1.5
python ia/alphabeta.py "<121-char board>" @ 1.0

# Engine smoke checks
python -c "from train.hex_env import HexEnv; print(HexEnv().get_state_tensor().shape)"   # → (3,11,11)
python train/network.py    # PASS if loss decreases
python train/evaluate.py   # win rate vs random

# Cleanup pyc / __pycache__
make clean
```

There is no test suite or linter configured — `network.py`/`evaluate.py` `__main__` blocks are the closest thing to smoke tests.

## Architecture

### Engine, network, search

- **`alphazero/train/hex_env.py`** — `HexEnv` is the single source of truth for game state. Two `bool (11,11)` numpy arrays (`blue`, `red`) plus `blue_to_play`. Win detection is BFS flood-fill. `mirror()` exposes the diagonal symmetry used for data augmentation. **Mutating API (`apply_move`, `undo_move`)** is used everywhere for speed — never assume a player's `select_move` left `env` untouched without inspecting it; the convention is "do not mutate" but it is enforced only by code review.
- **`alphazero/train/network.py`** — `HexNet`: 3-channel input (Blue plane, Red plane, side-to-move plane) → conv stem → `NUM_RES_BLOCKS` residual blocks → policy head (`log_softmax` over 121 cells) + value head (`tanh`).
- **`alphazero/ia/mcts_az.py`** — `MCTSAgent` / `MCTSNode`: PUCT search with batched GPU inference (16 leaves per forward), root Dirichlet noise (only in self-play, gated by `add_dirichlet`), FP16 autocast on CUDA. `get_policy(env, move_count, return_root=True)` is the canonical entry point used by `play.py`, `versus.py`, and the `AlphaZeroPlayer` wrapper in `tournament.py`.
- **`alphazero/train/self_play.py`** — Parallel self-play (`N_PARALLEL_GAMES` games stepped together, `LEAVES_PER_GAME` leaves per GPU round). Produces `(state, policy_target, value_target)` examples; `ReplayBuffer` is a circular `deque`-based buffer.
- **`alphazero/train/trainer.py`** — Outer loop. Per iteration: self-play → train N steps (Adam + cosine schedule) → eval new vs current best → swap if win rate ≥ `WIN_RATE_THRESHOLD` (0.55, see `config.py`) over `EVAL_GAMES`. On accept it copies `checkpoints/best_model.pt` to `model/` via `model_naming.copy_best_to_model`.
- **`alphazero/train/config.py`** — All hyperparameters live here (board size, MCTS sims, training steps, win-rate threshold, paths). Treat as the central tuning surface.

### Player protocol (every `ia/*.py`)

Every player file in `ia/` exposes both:
1. A Python class with `select_move(env: HexEnv, time_s: float) -> int` and a `last_stats: dict` attribute (used by `tournament.py` for verbose stats).
2. A `__main__` CLI implementing `python <file>.py BOARD PLAYER [time_s]` where `BOARD` is 121 chars (`.`/`O`/`@`), with the chosen move on stdout (`A1`..`K11`) and stats on stderr (`SCORE:.. NODES:.. DEPTH:..` for alpha-beta-style, `ITERS:.. VISITS:.. WINRATE:.. TIME:..` for MCTS-style). The CLI form is what lets `tournament.py` drive arbitrary external programs through `subprocess`.

A full how-to-add-a-player guide lives at `alphazero/ia/non_utiliser/add_model.md`. The non-obvious part: every new player must be registered in **two places** — `tournament._resolve_ai` (mnemonic → instance + display name) and **five sections** of `ranking.py` (`DEFAULT_CLASSICS`, `CLASSIC_DISPLAY`, `CLASSIC_TYPE`, `CLASSIC_COLORS`, `_make_classic`).

### Model storage and naming

Two parallel locations:
- **`alphazero/checkpoints/`** — live training output: `best_model.pt`, `model_iter_*.pt`, `replay_buffer.npz`. Overwritten each iteration.
- **`alphazero/model/`** — accepted-models archive. When the trainer accepts a new best, it copies `best_model.pt` here as `model_<NN>_<PP>_<DD>_<MM>.pt` (sequential num, parent num, day, month). `model_naming.py` is the canonical parser/builder/scanner — never hand-roll filenames; use `model_naming.next_model_number`, `build_model_name`, `copy_best_to_model`.

`ranking.py` auto-discovers everything in `model/` plus `checkpoints/best_model.pt` and labels them `AZ-NN` / `AZ-best` in the leaderboard.

### Ranking is incremental

`alphazero/rank/ranking.csv` is the source of truth; `ranking.py` only plays missing matchups, then regenerates a timestamped HTML report. To force replay, edit/clear the relevant rows in `rank/ranking.csv` (do not delete checkpoints just to retrigger games).

### LLM-authored players (`alphazero/ia/*_V*.py`)

Files like `claude_opus_4_7_xhigh_V2.py`, `deepseek_v4_pro_max_V3.py`, `gemini_3_1_pro_preview_V2.py`, etc. are fully self-contained Python AIs generated by various LLMs as a benchmark exercise. They follow the same `select_move`/CLI protocol as native players. The unversioned originals live in `alphazero/model_LLM/` (do not edit those — they are the "as-submitted" snapshots). The numbered `_V2`/`_V3` variants under `ia/` are the ones registered for tournaments.

## Conventions worth respecting

- **No `cd` inside Python.** Use the `_dir = os.path.dirname(os.path.abspath(__file__))` + `sys.path.insert` pattern that every existing entry point uses.
- **Always check the immediate-winning move** before running any expensive search (every native player does — see the pattern in `add_model.md` §5).
- **Mutate-and-undo over copy** is the performance idiom for tree search (`env.apply_move(m)` / `env.undo_move(m, was_blue)`).
- The codebase, comments, and CLI help are in **French**. Match the surrounding language when editing.
- ROCm AMD users: `export HSA_OVERRIDE_GFX_VERSION=10.3.0` for RX 6600 (gfx1032); still pass `--device cuda` to PyTorch.

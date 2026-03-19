# IA_jeux - Model for Breakthrough

A C++11 project implementing and benchmarking multiple algorithms to play the board game **Breakthrough** on an 8×8 board. The project also integrates with the **Ludii** game platform via Java wrappers.

---

## The Game: Breakthrough

Breakthrough is a two-player abstract strategy game played on an 8×8 board:

- **White** (`O`) starts on rows 7–8 and moves downward
- **Black** (`@`) starts on rows 1–2 and moves upward
- Each piece can move **forward**, **forward-left** (capture), or **forward-right** (capture)
- A player wins by **reaching the opponent's back row** or **capturing all opponent pieces**

---

## Project Structure

```
IA_jeux/
├── bkbb64.h                  # Core bitboard engine
├── random_bk.cpp             # Random player (baseline)
├── alphabeta_bk.cpp          # Alpha-Beta minimax (depth 6)
├── montecarlo_bk.cpp         # Flat parallel Monte Carlo
├── mcts_bk.cpp               # MCTS with shared tree + virtual loss (annotated)
├── mcts_bk_clean.cpp         # MCTS clean version (without comments)
├── uct_bk.cpp                # UCT with root parallelization
├── mast_bk.cpp               # MAST (Move Average Sampling Technique)
├── rave_bk.cpp               # RAVE / AMAF parallel
├── tournament.cpp            # 2-player match runner
├── big_tournament.cpp        # Full round-robin tournament
├── benchmark_mcts.cpp        # MCTS vs AlphaBeta benchmark sweep
├── Makefile                  # Build system
└── Ludii/
    ├── TC_MG_alphabeta_player.java   # Ludii plugin: AlphaBeta
    ├── TC_MG_mcts.java               # Ludii plugin: MCTS
    ├── TC_MG_montecarlo.java         # Ludii plugin: Monte Carlo
    ├── makeJar.sh                    # Builds Java .jar plugins
    └── makeRun.sh                    # Launches the Ludii GUI
```

---

## Algorithms

### Random (`random_bk.cpp`)
Picks a uniformly random legal move. Used as a baseline.

### Alpha-Beta (`alphabeta_bk.cpp`)
Depth-6 negamax with alpha-beta pruning. Evaluation function based on:
- Material count (number of pieces)
- Piece advancement (how far pieces have advanced)
- Protected pieces bonus

### Flat Monte Carlo (`montecarlo_bk.cpp`)
Root-parallel flat Monte Carlo search:
- Each thread runs independent random playouts
- Epsilon-greedy simulation: 80% chance to prefer captures
- Immediate win detection and unsafe move filtering

### MCTS with Virtual Loss (`mcts_bk.cpp`)
Full UCT tree search with shared tree and thread synchronization:
- **UCB1** selection formula for tree traversal
- **Virtual loss** to encourage thread exploration diversity
- **Shallow alpha-beta** (depth 3) at leaf nodes instead of random rollouts
- Sigmoid win-probability conversion from evaluation scores
- Final move: most-visited child (robustness criterion)

### UCT Root Parallelization (`uct_bk.cpp`)
Each thread builds its own independent UCT tree (no shared state, no mutexes). Trees are merged at the end by aggregating move statistics.

### MAST (`mast_bk.cpp`)
Move Average Sampling Technique (Finnsson & Björnsson, AAAI 2008):
- Each thread maintains a hash map of move → historical win rate
- Rollouts use **Gibbs sampling** to prefer historically good moves
- Zero contention between threads

### RAVE / AMAF (`rave_bk.cpp`)
RAVE with All Moves As First statistics (Gelly & Silver, ICML 2007):
- Each node stores AMAF statistics for all 4096 possible moves
- Move scoring blends UCT exploitation with AMAF:
  `score = (1 - β) * Q_uct + β * Q_amaf + UCT_C * sqrt(log(N)/n)`
- Root parallelization with statistics merged at the end

---

## Board Representation

All model share `bkbb64.h`, a 64-bit **bitboard engine**:

- The board state is stored as two `uint64_t` bitmasks (one per player)
- Move generation uses bitwise shifts and masks , extremely fast
- Thread-local **XorShift RNG** (lock-free, seedable per thread)
- Board coordinates are in chess-style notation (`A1`–`H8`)

---

## Build

Requires `g++` with C++11 support and `pthreads`.

```bash
make
make clean
```

This produces the following executables:
- `TC_MG_random`, `TC_MG_alphabeta_player`, `TC_MG_montecarlo`
- `TC_MG_mcts`, `TC_MG_uct`, `TC_MG_mast`, `TC_MG_rave`
- `tournament`, `big_tournament`, `benchmark_mcts`

---

## Usage

All player executables share the same CLI interface:

```
./TC_MG_<algo> <board_string> <player> [time_seconds]
```

- `board_string`: 64-character string with `@` (black), `O` (white), `.` (empty), row 8 first
- `player`: `O` for white, `@` for black
- `time_seconds`: optional time limit in seconds (default varies per agent)

The algorithm prints the chosen move to stdout as a coordinate string (e.g., `D5`).

### Example

```bash
./TC_MG_mcts "OOOOOOOOOOOOOOOO................................@@@@@@@@@@@@@@@@" O 1
```

---

## Running Tournaments

### 2-player match

```bash
./tournament <algo1> <algo2> <num_games> [verbose]
```

Example: 20 games between MCTS and AlphaBeta, with board display:

```bash
./tournament TC_MG_mcts TC_MG_alphabeta_player 20 1
```

### Round-robin tournament

```bash
./big_tournament
```

Auto-detects all `TC_MG_*` executables in the current directory, runs all pairs with adaptive match lengths, and displays a live colored leaderboard.

### MCTS benchmark

```bash
./benchmark_mcts
```

Sweeps MCTS iteration counts (50k → 1.5M) against AlphaBeta and writes results to `benchmark_results.txt`.

---

## Ludii Integration

The `Ludii/` folder contains Java plugins to run the algorithms inside the **Ludii game platform** GUI.

### Requirements

- Java 11+
- Download [Ludii-1.3.14.jar](https://ludii.games/download.php) and place it in `Ludii/`

### Build Java plugins

```bash
cd Ludii
chmod +x makeJar.sh
./makeJar.sh
```

### Launch Ludii GUI

```bash
cd Ludii
./makeRun.sh
```

The algorithms will appear as selectable algorithms in the Ludii interface for the Breakthrough game.

---

## References

- Finnsson, H. & Björnsson, Y. (2008). *Simulation-Based Approach to General Game Playing.* AAAI 2008. - **MAST**
- Gelly, S. & Silver, D. (2007). *Combining Online and Offline Knowledge in UCT.* ICML 2007. - **RAVE**
- Kocsis, L. & Szepesvári, C. (2006). *Bandit Based Monte-Carlo Planning.* ECML 2006. - **UCT / MCTS**

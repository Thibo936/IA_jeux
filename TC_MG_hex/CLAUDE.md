# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
make hex          # builds TC_MG_alphabeta_hex, TC_MG_mcts_hex, tournament_hex
make clean        # removes all binaries
```

Compiler: `g++ -std=c++11 -Wall -O3`. MCTS binary links `-lpthread`.

## Running

AI executables follow a common CLI protocol:
```
./TC_MG_<ai>_hex BOARD PLAYER [time_s]
```
- **BOARD**: 121-char string (row-major, left-to-right, top-to-bottom): `.` empty, `O` Blue, `@` Red
- **PLAYER**: `O` (Blue, North-South) or `@` (Red, West-East)
- **time_s**: time limit per move in seconds (MCTS uses it; alphabeta ignores it)
- Output on **stdout**: move in `"A1".."K11"` notation (column letter + row number)
- Stats on **stderr**: `SCORE:%d NODES:%d DEPTH:%d` (alphabeta) or `ITERS:%d VISITS:%d WINRATE:%f TIME:%f` (MCTS)

Tournament runner:
```
./tournament_hex ./TC_MG_alphabeta_hex ./TC_MG_mcts_hex 20 -v -t 1.5
```
Flags: `-v` verbose board display, `-t <seconds>` time per move (default 1.5s). Colors alternate each game.

## Architecture

- **hexbb.h**: Complete Hex 11x11 engine. Bitboard representation using 2×uint64_t (bits 0-63 in `_lo`, 64-120 in `_hi`). Contains `HexBoard` (state, move gen, BFS win detection, random playout via Fisher-Yates fill, 0-1 BFS shortest-path eval). Cell index = `row*11 + col`.
- **alphabeta_hex.cpp**: Alpha-beta player, fixed depth 4, uses `HexBoard::eval()` (shortest-path difference heuristic).
- **mcts_hex.cpp**: Multi-threaded MCTS with UCB1 selection, virtual loss, and tactical move filtering (avoids giving immediate wins to opponent). Leaf evaluation = 4 random playouts. Configurable time limit.
- **tournament.cpp**: Orchestrates games between any two AI executables via `popen`, parses their stderr stats for display.
- **mcts_bk.cpp**: Reference MCTS for Breakthrough (different game, `bkbb64.h`). Uses shallow alpha-beta (depth 3) + sigmoid for leaf evaluation instead of random playouts — significantly stronger approach.

## Key Design Patterns

- All AI players are standalone executables communicating via argv/stdout/stderr — no shared library or IPC.
- `HexBoard::random_playout()` exploits Hex's fill property: shuffle all empty cells, assign alternately to players, then check winner. This is valid because Hex has no draws and the board fills completely.
- The Breakthrough MCTS (`mcts_bk.cpp`) is notably stronger than the Hex MCTS because it replaces random rollouts with a shallow alpha-beta evaluation + sigmoid conversion — this is the key technique to port to Hex.

## Language

Comments and variable names are in French. User-facing messages (tournament output) are also French.

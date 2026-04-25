# heuristic_player.py — Joueur heuristique pour Hex 11×11
# Heuristique : différence de plus court chemin virtuel (BFS 0-1)
# Interface CLI : python heuristic_player.py BOARD PLAYER [time_s]

import sys
import os
from collections import deque
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import BOARD_SIZE, NUM_CELLS

_HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
_INF = 10 ** 9


def _shortest_virtual_path(board_player: np.ndarray,
                           board_blocker: np.ndarray,
                           start_row: bool) -> int:
    """
    Plus court chemin virtuel via BFS 0-1.
    Coût 0 sur une case déjà occupée par le joueur, coût 1 sur case vide,
    case bloquée (adversaire) interdite.
    """
    dist = np.full((BOARD_SIZE, BOARD_SIZE), _INF, dtype=np.int32)
    dq = deque()

    if start_row:
        for c in range(BOARD_SIZE):
            if board_blocker[0, c]:
                continue
            cost = 0 if board_player[0, c] else 1
            if cost < dist[0, c]:
                dist[0, c] = cost
                if cost == 0:
                    dq.appendleft((0, c))
                else:
                    dq.append((0, c))
    else:
        for r in range(BOARD_SIZE):
            if board_blocker[r, 0]:
                continue
            cost = 0 if board_player[r, 0] else 1
            if cost < dist[r, 0]:
                dist[r, 0] = cost
                if cost == 0:
                    dq.appendleft((r, 0))
                else:
                    dq.append((r, 0))

    while dq:
        r, c = dq.popleft()
        d_cur = dist[r, c]

        if start_row and r == BOARD_SIZE - 1:
            return int(d_cur)
        if (not start_row) and c == BOARD_SIZE - 1:
            return int(d_cur)

        for dr, dc in _HEX_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                continue
            if board_blocker[nr, nc]:
                continue

            step = 0 if board_player[nr, nc] else 1
            nd = d_cur + step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                if step == 0:
                    dq.appendleft((nr, nc))
                else:
                    dq.append((nr, nc))

    return _INF


def _eval_position(env: HexEnv, blue_to_play: bool) -> int:
    """Score > 0 si favorable au joueur courant."""
    pb = _shortest_virtual_path(env.blue, env.red, start_row=True)
    pr = _shortest_virtual_path(env.red, env.blue, start_row=False)

    if pb >= _INF and pr >= _INF:
        score = 0
    elif pb >= _INF:
        score = -100000
    elif pr >= _INF:
        score = 100000
    else:
        score = pr - pb  # >0 avantage Blue

    return score if blue_to_play else -score


class HeuristicPlayer:
    """
    Joueur glouton heuristique :
    - teste tous les coups légaux,
    - choisit celui maximisant l'évaluation statique.

    Interface : select_move(env, time_s) -> int (index 0..120)
    """

    def __init__(self):
        self.last_stats: dict = {}

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play

        # Coup gagnant immédiat prioritaire
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, root_blue)
            if (root_blue and w == 'blue') or ((not root_blue) and w == 'red'):
                self.last_stats = {'score': 100000, 'nodes': 1, 'depth': 1}
                print("SCORE:100000 NODES:1 DEPTH:1", file=sys.stderr)
                return m

        best_move = int(moves[0])
        best_score = -10 ** 9
        nodes = 0

        for move in moves:
            m = int(move)
            env.apply_move(m)
            score = -_eval_position(env, not root_blue)
            env.undo_move(m, root_blue)
            nodes += 1

            if score > best_score:
                best_score = score
                best_move = m

        self.last_stats = {'score': int(best_score), 'nodes': nodes, 'depth': 1}
        print(f"SCORE:{int(best_score)} NODES:{nodes} DEPTH:1", file=sys.stderr)
        return best_move


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python heuristic_player.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = HeuristicPlayer()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

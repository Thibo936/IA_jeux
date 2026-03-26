# alphabeta.py — Joueur Alpha-Beta profondeur 4 pour Hex 11×11
# Heuristique : différence de plus court chemin virtuel (BFS 0-1)
# Interface CLI : python alphabeta.py BOARD PLAYER [time_s]

import sys
import os
from collections import deque
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from config import BOARD_SIZE, NUM_CELLS

# ─── Constantes ───────────────────────────────────────────────────────────────

# Profondeur de base. En Python, la vitesse limite la profondeur utile :
# - ≥ 60 coups légaux (début de partie) : profondeur 1 (greedy ~40ms)
# - 30–60 coups : profondeur 2 (~200ms avec ordonnancement)
# - < 30 coups (fin de partie) : profondeur 3
# L'ordonnancement des coups à la racine améliore drastiquement les coupures.
MAX_DEPTH  = 6
SCORE_WIN  =  100_000
SCORE_LOSE = -100_000

_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


# ─── BFS 0-1 : plus court chemin virtuel ──────────────────────────────────────

def _shortest_path(board_player: np.ndarray, board_blocker: np.ndarray,
                   start_row: bool) -> int:
    """
    BFS 0-1 depuis un bord vers l'autre.
    Coût 0 = case du joueur, coût 1 = case vide, bloqué = case adverse.
    start_row=True  → Blue : ligne 0 → ligne 10
    start_row=False → Red  : col  0 → col  10
    """
    N = BOARD_SIZE
    INF = 10 ** 9
    dist = np.full(N * N, INF, dtype=np.int32)
    dq = deque()

    # Cases de départ : ligne 0 pour Blue, colonne 0 pour Red
    starts = range(N) if start_row else range(0, N * N, N)

    for idx in starts:
        r, c = divmod(idx, N)
        if board_blocker[r, c]:
            continue
        cost = 0 if board_player[r, c] else 1
        if cost < dist[idx]:
            dist[idx] = cost
            dq.appendleft(idx) if cost == 0 else dq.append(idx)

    while dq:
        cur = dq.popleft()
        r, c = divmod(cur, N)
        d_cur = dist[cur]
        # Vérifier si le bord opposé est atteint
        if (start_row and r == N - 1) or (not start_row and c == N - 1):
            return d_cur
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            nidx = nr * N + nc
            if board_blocker[nr, nc]:
                continue
            cost = 0 if board_player[nr, nc] else 1
            nd = d_cur + cost
            if nd < dist[nidx]:
                dist[nidx] = nd
                dq.appendleft(nidx) if cost == 0 else dq.append(nidx)

    return INF


def eval_heuristic(env: HexEnv, blue_to_play: bool) -> int:
    """
    Score = Red_path − Blue_path, du point de vue du joueur courant.
    Positif = avantage pour le joueur courant.
    """
    pb = _shortest_path(env.blue, env.red,  start_row=True)
    pr = _shortest_path(env.red,  env.blue, start_row=False)
    INF = 10 ** 9

    if pb == INF and pr == INF:
        score = 0
    elif pb == INF:
        score = -SCORE_WIN
    elif pr == INF:
        score = SCORE_WIN
    else:
        score = pr - pb  # positif = Blue plus proche de gagner

    return score if blue_to_play else -score


# ─── Récursion Alpha-Beta (négamax) ───────────────────────────────────────────

def _alphabeta(env: HexEnv, depth: int, alpha: int, beta: int,
               blue_to_play: bool, nodes: list) -> int:
    nodes[0] += 1

    w = env.winner()
    if w == 'blue':
        return (SCORE_WIN  + depth) if blue_to_play else (SCORE_LOSE - depth)
    if w == 'red':
        return (SCORE_LOSE - depth) if blue_to_play else (SCORE_WIN  + depth)
    if depth == 0:
        return eval_heuristic(env, blue_to_play)

    moves = env.get_legal_moves()
    if len(moves) == 0:
        return eval_heuristic(env, blue_to_play)

    best = SCORE_LOSE
    for move in moves:
        child = env.copy()
        child.apply_move(int(move))
        val = -_alphabeta(child, depth - 1, -beta, -alpha, not blue_to_play, nodes)
        if val > best:
            best = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break  # coupure bêta
    return best


# ─── Joueur Alpha-Beta ────────────────────────────────────────────────────────

class AlphaBetaPlayer:
    """
    Joueur Alpha-Beta profondeur fixe pour Hex 11×11.
    Interface : select_move(env, time_s) -> int (index 0..120)
    Stats disponibles dans self.last_stats après chaque coup.
    """

    def __init__(self, depth: int = MAX_DEPTH):
        self.depth = depth
        self.last_stats: dict = {}

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        """
        Retourne l'index (0..120) du meilleur coup.

        Stratégie :
        - Profondeur adaptative : depth+1 si ≤ 40 coups légaux, depth+2 si ≤ 15.
        - Ordonnancement des coups à la racine (tri par eval_heuristic) pour
          améliorer drastiquement les coupures alpha-beta.
        """
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        blue_to_play = env.blue_to_play

        # ── Coup gagnant immédiat ─────────────────────────────────────────────
        for move in moves:
            child = env.copy()
            child.apply_move(int(move))
            w = child.winner()
            if (blue_to_play and w == 'blue') or (not blue_to_play and w == 'red'):
                self.last_stats = {'score': SCORE_WIN, 'nodes': 1, 'depth': self.depth}
                print(f"SCORE:{SCORE_WIN} NODES:1 DEPTH:{self.depth}", file=sys.stderr)
                return int(move)

        # ── Profondeur adaptative ─────────────────────────────────────────────
        # En début de partie, la BFS donne ~0 pour tous les coups → pas de pruning.
        # Limiter la profondeur selon le nombre de coups pour respecter le budget temps.
        n = len(moves)
        effective_depth = self.depth + (1 if n <= 60 else 0) + (1 if n <= 30 else 0)

        # ── Ordonnancement à la racine par eval_heuristic (améliore la coupure)
        scored = []
        for move in moves:
            child = env.copy()
            child.apply_move(int(move))
            s = -eval_heuristic(child, not blue_to_play)
            scored.append((s, int(move)))
        scored.sort(reverse=True)

        nodes = [0]
        best_move = scored[0][1]
        best_val  = SCORE_LOSE

        for _, move in scored:
            child = env.copy()
            child.apply_move(move)
            val = -_alphabeta(child, effective_depth - 1, SCORE_LOSE, SCORE_WIN,
                              not blue_to_play, nodes)
            if val > best_val:
                best_val = val
                best_move = move

        self.last_stats = {
            'score': best_val, 'nodes': nodes[0], 'depth': effective_depth
        }
        print(f"SCORE:{best_val} NODES:{nodes[0]} DEPTH:{effective_depth}",
              file=sys.stderr)
        return best_move


# ─── Interface CLI (compatible protocole BOARD/PLAYER) ───────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python alphabeta.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = AlphaBetaPlayer()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

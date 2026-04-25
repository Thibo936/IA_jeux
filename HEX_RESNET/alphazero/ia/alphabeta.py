# alphabeta.py — Joueur Alpha-Beta pour Hex 11×11
# Heuristique : différence de plus court chemin virtuel (BFS 0-1)
# Optimisations : table de transposition Zobrist, killer moves, history heuristic
# Interface CLI : python alphabeta.py BOARD PLAYER [time_s]

import sys
import os
import random as _rng
from collections import deque
import numpy as np
import numba

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import BOARD_SIZE, NUM_CELLS

# ─── Constantes ───────────────────────────────────────────────────────────────

MAX_DEPTH  = 4 # (4 rapide, 5 10sec, 6 trop long)
SCORE_WIN  =  100_000
SCORE_LOSE = -100_000

_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

# ─── Zobrist hashing ─────────────────────────────────────────────────────────

_rng.seed(42)
# _ZOBRIST[pos][0] = clé Blue, _ZOBRIST[pos][1] = clé Red
_ZOBRIST = [[_rng.getrandbits(64) for _ in range(2)] for _ in range(NUM_CELLS)]
_ZOBRIST_TURN = _rng.getrandbits(64)

# ─── Ordre statique : préférer le centre du plateau ──────────────────────────

_CENTER = BOARD_SIZE // 2
_STATIC_ORDER = sorted(
    range(NUM_CELLS),
    key=lambda i: abs(i // BOARD_SIZE - _CENTER) + abs(i % BOARD_SIZE - _CENTER)
)

# Flags pour la table de transposition
_TT_EXACT = 0
_TT_LOWER = 1   # alpha (borne inférieure)
_TT_UPPER = 2   # beta  (borne supérieure)


def _compute_hash(env: HexEnv) -> int:
    """Calcule le hash Zobrist complet depuis l'état du plateau."""
    h = 0
    for i in range(NUM_CELLS):
        r, c = divmod(i, BOARD_SIZE)
        if env.blue[r, c]:
            h ^= _ZOBRIST[i][0]
        elif env.red[r, c]:
            h ^= _ZOBRIST[i][1]
    if not env.blue_to_play:
        h ^= _ZOBRIST_TURN
    return h


def _hash_apply(h: int, pos: int, is_blue: bool) -> int:
    """Met à jour le hash après un apply_move."""
    h ^= _ZOBRIST[pos][0 if is_blue else 1]
    h ^= _ZOBRIST_TURN  # changement de tour
    return h


def _hash_undo(h: int, pos: int, was_blue: bool) -> int:
    """Met à jour le hash après un undo_move (identique à apply, XOR est involutif)."""
    h ^= _ZOBRIST[pos][0 if was_blue else 1]
    h ^= _ZOBRIST_TURN
    return h


# ─── BFS 0-1 : plus court chemin virtuel (Numba JIT) ─────────────────────────

# Voisins hexagonaux sous forme de tableaux pour Numba
_NB_DR = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int32)
_NB_DC = np.array([0, 1, -1, 1, -1, 0], dtype=np.int32)

@numba.njit(cache=True)
def _shortest_path_jit(player_flat, blocker_flat, start_row, N, nb_dr, nb_dc):
    """
    BFS 0-1 Numba. player_flat/blocker_flat sont des bool[N*N].
    start_row=True → Blue (ligne 0→10), False → Red (col 0→10).
    Retourne la distance minimale ou 1_000_000_000 si pas de chemin.
    """
    INF = 1_000_000_000
    NN = N * N
    dist = np.full(NN, INF, dtype=np.int32)

    # Deque manuelle : buffer circulaire
    buf = np.empty(NN * 2, dtype=np.int32)
    head = NN  # milieu du buffer, on pousse à gauche et à droite
    tail = NN

    # Cases de départ
    if start_row:
        for c in range(N):
            idx = c  # ligne 0
            if blocker_flat[idx]:
                continue
            cost = 0 if player_flat[idx] else 1
            if cost < dist[idx]:
                dist[idx] = cost
                if cost == 0:
                    head -= 1
                    buf[head] = idx
                else:
                    buf[tail] = idx
                    tail += 1
    else:
        for r in range(N):
            idx = r * N  # colonne 0
            if blocker_flat[idx]:
                continue
            cost = 0 if player_flat[idx] else 1
            if cost < dist[idx]:
                dist[idx] = cost
                if cost == 0:
                    head -= 1
                    buf[head] = idx
                else:
                    buf[tail] = idx
                    tail += 1

    while head < tail:
        cur = buf[head]
        head += 1
        r = cur // N
        c = cur % N
        d_cur = dist[cur]

        # Vérifier bord opposé
        if start_row and r == N - 1:
            return d_cur
        if not start_row and c == N - 1:
            return d_cur

        for k in range(6):
            nr = r + nb_dr[k]
            nc = c + nb_dc[k]
            if nr < 0 or nr >= N or nc < 0 or nc >= N:
                continue
            nidx = nr * N + nc
            if blocker_flat[nidx]:
                continue
            cost = 0 if player_flat[nidx] else 1
            nd = d_cur + cost
            if nd < dist[nidx]:
                dist[nidx] = nd
                if cost == 0:
                    head -= 1
                    buf[head] = nidx
                else:
                    buf[tail] = nidx
                    tail += 1

    return INF


def _shortest_path(board_player: np.ndarray, board_blocker: np.ndarray,
                   start_row: bool) -> int:
    """Wrapper pour la BFS 0-1 Numba."""
    return int(_shortest_path_jit(
        board_player.ravel(), board_blocker.ravel(),
        start_row, BOARD_SIZE, _NB_DR, _NB_DC
    ))


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


# ─── Récursion Alpha-Beta (négamax) avec TT + ordonnancement ─────────────────

def _order_moves(moves, tt_best, killers, history, blue_to_play):
    """Ordonne les coups : TT best → killers → history score → statique."""
    color_idx = 0 if blue_to_play else 1
    ordered = []
    rest = []
    killer_set = set(killers)

    for move in moves:
        m = int(move)
        if m == tt_best:
            continue  # sera inséré en tête
        if m in killer_set:
            ordered.append((1, -history[color_idx][m], m))
        else:
            ordered.append((2, -history[color_idx][m], m))

    ordered.sort()
    result = []
    if tt_best >= 0:
        result.append(tt_best)
    result.extend(m for _, _, m in ordered)
    return result


def _alphabeta(env: HexEnv, depth: int, alpha: int, beta: int,
               blue_to_play: bool, nodes: list, zhash: int,
               tt: dict, killers: list, history: list) -> int:
    nodes[0] += 1
    alpha_orig = alpha

    # ── Lookup table de transposition ─────────────────────────────────────
    tt_entry = tt.get(zhash)
    tt_best_move = -1
    if tt_entry is not None:
        tt_depth, tt_flag, tt_score, tt_bm = tt_entry
        if tt_depth >= depth:
            if tt_flag == _TT_EXACT:
                return tt_score
            elif tt_flag == _TT_LOWER:
                alpha = max(alpha, tt_score)
            elif tt_flag == _TT_UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score
        tt_best_move = tt_bm

    # ── Terminal / feuille ────────────────────────────────────────────────
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

    # ── Ordonnancement des coups ──────────────────────────────────────────
    depth_killers = killers[depth] if depth < len(killers) else []
    ordered = _order_moves(moves, tt_best_move, depth_killers, history, blue_to_play)

    best = SCORE_LOSE
    best_move = ordered[0] if ordered else -1

    for m in ordered:
        zhash_new = _hash_apply(zhash, m, blue_to_play)
        env.apply_move(m)
        val = -_alphabeta(env, depth - 1, -beta, -alpha, not blue_to_play,
                          nodes, zhash_new, tt, killers, history)
        env.undo_move(m, blue_to_play)

        if val > best:
            best = val
            best_move = m
        if val > alpha:
            alpha = val
        if alpha >= beta:
            # Mise à jour killer moves et history heuristic
            if depth < len(killers):
                k = killers[depth]
                if m not in k:
                    if len(k) >= 2:
                        k.pop(0)
                    k.append(m)
            color_idx = 0 if blue_to_play else 1
            history[color_idx][m] += depth * depth
            break

    # ── Stocker dans la TT ────────────────────────────────────────────────
    if best <= alpha_orig:
        flag = _TT_UPPER
    elif best >= beta:
        flag = _TT_LOWER
    else:
        flag = _TT_EXACT
    tt[zhash] = (depth, flag, best, best_move)

    return best


# ─── Joueur Alpha-Beta ────────────────────────────────────────────────────────

class AlphaBetaPlayer:
    """
    Joueur Alpha-Beta pour Hex 11×11 avec table de transposition Zobrist,
    killer moves, et history heuristic.
    Interface : select_move(env, time_s) -> int (index 0..120)
    Stats disponibles dans self.last_stats après chaque coup.
    """

    def __init__(self, depth: int = MAX_DEPTH):
        self.depth = depth
        self.last_stats: dict = {}
        self._tt: dict = {}  # table de transposition (persistante entre coups)

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        """
        Retourne l'index (0..120) du meilleur coup.

        Stratégie :
        - Profondeur adaptative selon le nombre de coups restants.
        - Ordonnancement à la racine par heuristique, puis TT/killer/history en récursion.
        """
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        blue_to_play = env.blue_to_play
        zhash = _compute_hash(env)

        # ── Coup gagnant immédiat ─────────────────────────────────────────────
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, blue_to_play)
            if (blue_to_play and w == 'blue') or (not blue_to_play and w == 'red'):
                self.last_stats = {'score': SCORE_WIN, 'nodes': 1, 'depth': self.depth}
                print(f"SCORE:{SCORE_WIN} NODES:1 DEPTH:{self.depth}", file=sys.stderr)
                return m

        # ── Profondeur adaptative ─────────────────────────────────────────────
        n = len(moves)
        effective_depth = self.depth + (1 if n <= 60 else 0) + (1 if n <= 30 else 0)

        # ── Structures de recherche ───────────────────────────────────────────
        # Killer moves : 2 par niveau de profondeur
        killers = [[] for _ in range(effective_depth + 1)]
        # History heuristic : [blue][move], [red][move]
        history = [[0] * NUM_CELLS, [0] * NUM_CELLS]

        # ── Ordonnancement à la racine par eval_heuristic ─────────────────────
        scored = []
        for move in moves:
            m = int(move)
            env.apply_move(m)
            s = -eval_heuristic(env, not blue_to_play)
            env.undo_move(m, blue_to_play)
            scored.append((s, m))
        scored.sort(reverse=True)

        nodes = [0]
        best_move = scored[0][1]
        best_val  = SCORE_LOSE
        alpha = SCORE_LOSE

        for _, move in scored:
            zhash_new = _hash_apply(zhash, move, blue_to_play)
            env.apply_move(move)
            val = -_alphabeta(env, effective_depth - 1, -SCORE_WIN, -alpha,
                              not blue_to_play, nodes, zhash_new,
                              self._tt, killers, history)
            env.undo_move(move, blue_to_play)
            if val > best_val:
                best_val = val
                best_move = move
            if val > alpha:
                alpha = val

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

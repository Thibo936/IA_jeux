# qwen36plus.py — MCTS avec rollouts heuristiques pour Hex 11×11
# Interface CLI : python qwen36plus.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random
import numpy as np
from collections import deque

# ─── Bootstrap des imports train/ ─────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE

# ─── Constantes ───────────────────────────────────────────────────────────────
N = BOARD_SIZE
TOTAL = NUM_CELLS

# Voisins hexagonaux (grille rhombique)
HEX_NEIGHBORS = []
for r in range(N):
    for c in range(N):
        neighbors = []
        for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                neighbors.append(nr * N + nc)
        HEX_NEIGHBORS.append(neighbors)


def _is_connected(board: np.ndarray, is_blue: bool) -> bool:
    """Vérifie si le joueur a un chemin connecté via BFS."""
    visited = np.zeros((N, N), dtype=bool)
    queue = deque()

    if is_blue:
        for c in range(N):
            if board[0, c]:
                visited[0, c] = True
                queue.append((0, c))
    else:
        for r in range(N):
            if board[r, 0]:
                visited[r, 0] = True
                queue.append((r, 0))

    while queue:
        r, c = queue.popleft()
        if is_blue and r == N - 1:
            return True
        if not is_blue and c == N - 1:
            return True
        for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and not visited[nr, nc] and board[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False


def _check_win_move(env, move, root_blue):
    """Applique move, vérifie si le joueur qui vient de jouer gagne, undo."""
    env.apply_move(move)
    w = env.winner()
    env.undo_move(move, root_blue)
    return w


class MCTSNode:
    __slots__ = ['N', 'W', 'children', 'legal_moves', 'parent']

    def __init__(self, legal_moves=None, parent=None):
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.legal_moves = legal_moves
        self.parent = parent


class Qwen36Plus:
    """MCTS avec rollouts heuristiques pour Hex 11×11."""

    def __init__(self, c_uct: float = 1.41, seed: int | None = None):
        self.c_uct = c_uct
        self.last_stats: dict = {}
        if seed is not None:
            random.seed(seed)

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            self.last_stats = {'iters': 0, 'visits': 0, 'winrate': 0.5, 'time': 0.0}
            return -1
        if len(moves) == 1:
            self.last_stats = {'iters': 0, 'visits': 0, 'winrate': 0.5, 'time': 0.0}
            return int(moves[0])

        root_blue = env.blue_to_play
        legal = [int(m) for m in moves]

        # 1) Coup gagnant immédiat
        for m in legal:
            w = _check_win_move(env, m, root_blue)
            if w is not None:
                if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
                    self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
                    return m

        # 2) Bloquer menace adverse (si l'adversaire gagne en jouant ici)
        opp_blue = not root_blue
        for m in legal:
            env.blue_to_play = opp_blue
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, opp_blue)
            env.blue_to_play = root_blue
            if w is not None:
                if (opp_blue and w == 'blue') or (not opp_blue and w == 'red'):
                    self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 0.9, 'time': 0.0}
                    return m

        # 3) MCTS
        t0 = time.time()
        deadline = t0 + time_s * 0.95

        root = MCTSNode(legal_moves=legal)
        sims = 0

        while time.time() < deadline:
            sims += 1
            node, path_moves = self._select(root)

            # Expansion : ajouter TOUS les coups non explorés
            if node.legal_moves is not None:
                untried = [m for m in node.legal_moves if m not in node.children]
                if untried:
                    child_env = env.copy()
                    cur_blue = root_blue
                    for m in path_moves:
                        child_env.apply_move(m)
                        cur_blue = not cur_blue

                    for child_move in untried:
                        child_env.apply_move(child_move)
                        child_legal = [int(m) for m in child_env.get_legal_moves()]
                        child_env.undo_move(child_move, cur_blue)
                        child = MCTSNode(legal_moves=child_legal, parent=node)
                        node.children[child_move] = child

                    child_move = random.choice(untried)
                    path_moves.append(child_move)
                    node = node.children[child_move]

            result = self._simulate(env, path_moves, root_blue)
            self._backpropagate(node, result)

        best_move = max(root.children.keys(), key=lambda m: root.children[m].N)
        visits = sum(c.N for c in root.children.values())
        winrate = root.children[best_move].W / root.children[best_move].N if root.children[best_move].N > 0 else 0.5
        elapsed = time.time() - t0

        self.last_stats = {
            'iters': sims,
            'visits': visits,
            'winrate': winrate,
            'time': elapsed,
        }
        return best_move

    def _select(self, root):
        """Descente UCB1. Retourne (feuille, liste_des_coups_depuis_racine)."""
        node = root
        path_moves = []
        while node.children:
            best_child = None
            best_val = -float('inf')
            total_n = node.N

            for move, child in node.children.items():
                if child.N == 0:
                    return child, path_moves + [move]
                q = child.W / child.N
                ucb = q + self.c_uct * math.sqrt(math.log(total_n + 1) / child.N)
                if ucb > best_val:
                    best_val = ucb
                    best_child = move

            path_moves.append(best_child)
            node = node.children[best_child]
        return node, path_moves

    def _simulate(self, env, path_moves, root_blue):
        """
        Applique path_moves sur une copie de env, puis rollout heuristique.
        Retourne 1.0 si Blue gagne, 0.0 si Red gagne.
        """
        sim_env = env.copy()
        for m in path_moves:
            sim_env.apply_move(m)

        if sim_env.is_terminal():
            w = sim_env.winner()
            return 1.0 if w == 'blue' else 0.0

        moves = list(sim_env.get_legal_moves())
        occupied = set()
        for r in range(N):
            for c in range(N):
                if sim_env.blue[r, c] or sim_env.red[r, c]:
                    occupied.add(r * N + c)

        moves_played = len(occupied)
        cur_blue = sim_env.blue_to_play

        while moves:
            mi_list = [int(m) for m in moves]

            # 1) Bloquer menace immédiate de l'adversaire
            blocking = None
            for mi in mi_list:
                sim_env.blue_to_play = not cur_blue
                sim_env.apply_move(mi)
                w = sim_env.winner()
                sim_env.undo_move(mi, not cur_blue)
                sim_env.blue_to_play = cur_blue
                if w is not None:
                    if (not cur_blue and w == 'blue') or (cur_blue and w == 'red'):
                        blocking = mi
                        break
            if blocking is not None:
                sim_env.apply_move(blocking)
                occupied.add(blocking)
                if sim_env.is_terminal():
                    w = sim_env.winner()
                    return 1.0 if w == 'blue' else 0.0
                moves = list(sim_env.get_legal_moves())
                cur_blue = sim_env.blue_to_play
                continue

            # 2) Score heuristique pour chaque coup
            scores = {}
            for mi in mi_list:
                r, c = divmod(mi, N)
                score = 0.0

                # Connectivité : voisins du joueur courant
                my_board = sim_env.blue if cur_blue else sim_env.red
                opp_board = sim_env.red if cur_blue else sim_env.blue
                my_neighbors = 0
                opp_neighbors = 0
                for nb in HEX_NEIGHBORS[mi]:
                    if my_board[nb // N, nb % N]:
                        my_neighbors += 1
                    if opp_board[nb // N, nb % N]:
                        opp_neighbors += 1
                score += my_neighbors * 2.0
                score += opp_neighbors * 1.5

                # Centre : bonus en début de partie
                if moves_played < 20:
                    dist = abs(r - N // 2) + abs(c - N // 2)
                    score += (N - dist) * 0.3

                scores[mi] = score

            # 3) Sélection softmax avec 15% d'exploration
            if random.random() < 0.15:
                move = random.choice(mi_list)
            else:
                max_score = max(scores.values())
                weighted = [(mi, math.exp(min(s - max_score, 0)) + 0.01) for mi, s in scores.items()]
                total = sum(w for _, w in weighted)
                r_val = random.random() * total
                cum = 0.0
                move = mi_list[0]
                for mi, w in weighted:
                    cum += w
                    if r_val <= cum:
                        move = mi
                        break

            sim_env.apply_move(move)
            occupied.add(move)
            moves_played += 1

            if sim_env.is_terminal():
                w = sim_env.winner()
                return 1.0 if w == 'blue' else 0.0

            moves = list(sim_env.get_legal_moves())
            cur_blue = sim_env.blue_to_play

        w = sim_env.winner()
        if w is None:
            return 1.0 if _is_connected(sim_env.blue, True) else 0.0
        return 1.0 if w == 'blue' else 0.0

    def _backpropagate(self, node, result):
        """
        result: 1.0 si Blue gagne, 0.0 si Red gagne.
        On accumule W du point de vue de Blue.
        """
        while node is not None:
            node.N += 1
            node.W += result
            node = node.parent


# ─── Interface CLI (protocole BOARD/PLAYER) ───────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python qwen36plus.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = Qwen36Plus()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

    s = _player.last_stats
    print(f"ITERS:{s['iters']} VISITS:{s['visits']} WINRATE:{s['winrate']:.4f} TIME:{s['time']:.3f}", file=sys.stderr)

import sys
import os
import time
import math
import random
from collections import deque

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE

# --------------------------------------------------------------------------- #
#  Constantes hexagonales
# --------------------------------------------------------------------------- #
_DR = (-1, -1, 0, 0, 1, 1)
_DC = (0, 1, -1, 1, -1, 0)

# (dr,dc) de l'autre pierre du pont  →  les deux cases intermédiaires
_BRIDGE_PATTERNS = (
    ((-2, 1), ((-1, 0), (-1, 1))),
    ((-1, 2), ((-1, 1), (0, 1))),
    ((1, 1), ((0, 1), (1, 0))),
    ((2, -1), ((1, 0), (1, -1))),
    ((1, -2), ((1, -1), (0, -1))),
    ((-1, -1), ((0, -1), (-1, 0))),
)

# Bytearray réutilisable pour les BFS partiels (évite les allocations)
_visit_buf_global = bytearray(NUM_CELLS)
_visit_tok_global = 0


def _quick_connected(board: np.ndarray, sr: int, sc: int, target) -> bool:
    """
    BFS partiel sur les pierres de `board` depuis (sr,sc).
    Retourne True si on atteint une case satisfaisant `target(r,c)`.
    """
    global _visit_tok_global, _visit_buf_global
    _visit_tok_global += 1
    if _visit_tok_global > 250:
        _visit_buf_global = bytearray(NUM_CELLS)
        _visit_tok_global = 1
    tok = _visit_tok_global

    if target(sr, sc):
        return True

    q = deque()
    idx = sr * BOARD_SIZE + sc
    _visit_buf_global[idx] = tok
    q.append((sr, sc))

    while q:
        r, c = q.popleft()
        for k in range(6):
            nr, nc = r + _DR[k], c + _DC[k]
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc]:
                    nidx = nr * BOARD_SIZE + nc
                    if _visit_buf_global[nidx] != tok:
                        if target(nr, nc):
                            return True
                        _visit_buf_global[nidx] = tok
                        q.append((nr, nc))
    return False


def _quick_win(board: np.ndarray, sr: int, sc: int, target1, target2) -> bool:
    """
    BFS partiel unique : retourne True si le cluster de (sr,sc) touche
    à la fois target1 ET target2.
    """
    global _visit_tok_global, _visit_buf_global
    _visit_tok_global += 1
    if _visit_tok_global > 250:
        _visit_buf_global = bytearray(NUM_CELLS)
        _visit_tok_global = 1
    tok = _visit_tok_global

    t1 = target1(sr, sc)
    t2 = target2(sr, sc)
    if t1 and t2:
        return True

    q = deque()
    idx = sr * BOARD_SIZE + sc
    _visit_buf_global[idx] = tok
    q.append((sr, sc))

    while q:
        r, c = q.popleft()
        for k in range(6):
            nr, nc = r + _DR[k], c + _DC[k]
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc]:
                    nidx = nr * BOARD_SIZE + nc
                    if _visit_buf_global[nidx] != tok:
                        if target1(nr, nc):
                            t1 = True
                        if target2(nr, nc):
                            t2 = True
                        if t1 and t2:
                            return True
                        _visit_buf_global[nidx] = tok
                        q.append((nr, nc))
    return False


# --------------------------------------------------------------------------- #
#  Pool de coups O(1)  (supprime les list.remove coûteux)
# --------------------------------------------------------------------------- #
class FastPool:
    __slots__ = ('arr', 'pos', 'n')

    def __init__(self, legal_moves):
        self.n = len(legal_moves)
        self.arr = np.empty(NUM_CELLS, dtype=np.int16)
        self.pos = np.empty(NUM_CELLS, dtype=np.int16)
        self.arr[:self.n] = legal_moves
        for i in range(self.n):
            self.pos[int(legal_moves[i])] = i

    def remove(self, move: int):
        idx = self.pos[move]
        last = int(self.arr[self.n - 1])
        self.arr[idx] = last
        self.pos[last] = idx
        self.n -= 1

    def add(self, move: int):
        self.arr[self.n] = move
        self.pos[move] = self.n
        self.n += 1

    def random_pop(self, rng) -> int:
        idx = rng.randint(0, self.n - 1)
        move = int(self.arr[idx])
        self.remove(move)
        return move

    def copy(self):
        p = object.__new__(FastPool)
        p.arr = self.arr.copy()
        p.pos = self.pos.copy()
        p.n = self.n
        return p

    def tolist(self):
        return [int(self.arr[i]) for i in range(self.n)]

    def __len__(self):
        return self.n

    def is_empty(self):
        return self.n == 0


# --------------------------------------------------------------------------- #
#  Nœud MCTS
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = (
        'parent', 'move', 'children', 'wins', 'visits',
        'untried', 'is_blue', 'solved', 'prior'
    )

    def __init__(self, parent, move, untried, is_blue, prior=0.0):
        self.parent = parent
        self.move = move
        self.children = {}
        self.wins = 0.0
        self.visits = 0
        self.untried = untried          # FastPool
        self.is_blue = is_blue
        self.solved = None          # None / 0.0(loss for player to move) / 1.0(win)
        self.prior = prior


# --------------------------------------------------------------------------- #
#  IA
# --------------------------------------------------------------------------- #
class KimiK26:
    NAME = "KimiK26"

    # Hyper-paramètres MCTS
    CPUCT = 2.2
    MAX_ROLLOUT = 14
    CHECK_EVERY = 32
    ROLLOUT_SAMPLES = 10          # nombre de candidats évalués par coup de rollout

    def __init__(self, seed=None):
        self.last_stats = {}
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

        # Buffers réutilisables pour la BFS 0-1 (distance de connexion)
        self._dist_buf = np.full(NUM_CELLS, 1000, dtype=np.int16)
        self._visit_buf_local = bytearray(NUM_CELLS)
        self._visit_tok_local = 0

    # ------------------------------------------------------------------ #
    #  API publique
    # ------------------------------------------------------------------ #
    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play
        root_player = 'blue' if root_blue else 'red'
        opp = 'red' if root_blue else 'blue'

        # 1) Coup gagnant immédiat
        for m in moves:
            m_int = int(m)
            env.apply_move(m_int)
            w = env.winner()
            env.undo_move(m_int, root_blue)
            if w == root_player:
                self.last_stats = {'iters': 1, 'visits': 1,
                                   'winrate': 1.0, 'time': 0.0}
                return m_int

        # 2) Coup bloquant immédiat (on simule le tour adverse avec une copie)
        opp_winning = []
        for om in moves:
            om_int = int(om)
            tmp = env.copy()
            tmp.blue_to_play = not root_blue
            tmp.apply_move(om_int)
            if tmp.winner() == opp:
                opp_winning.append(om_int)
        if len(opp_winning) == 1:
            self.last_stats = {'iters': 1, 'visits': 1,
                               'winrate': 0.5, 'time': 0.0}
            return opp_winning[0]

        # 3) MCTS
        best_move, iters, visits, winrate, elapsed = self._mcts(env, time_s)
        self.last_stats = {
            'iters': iters,
            'visits': visits,
            'winrate': winrate,
            'time': elapsed,
        }
        return best_move

    # ------------------------------------------------------------------ #
    #  Heuristiques rapides
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bridge_score(board: np.ndarray, opp_board: np.ndarray, r: int, c: int) -> float:
        """Détecte les ponts et connexions tactiques."""
        bonus = 0.0
        for (dr, dc), ((dr1, dc1), (dr2, dc2)) in _BRIDGE_PATTERNS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc]:
                    p1r, p1c = r + dr1, c + dc1
                    p2r, p2c = r + dr2, c + dc2
                    if (0 <= p1r < BOARD_SIZE and 0 <= p1c < BOARD_SIZE and
                            0 <= p2r < BOARD_SIZE and 0 <= p2c < BOARD_SIZE):
                        blocked = (opp_board[p1r, p1c] and opp_board[p2r, p2c])
                        if not blocked:
                            if board[p1r, p1c] or board[p2r, p2c]:
                                bonus += 150.0
                            else:
                                bonus += 40.0
        return bonus

    def _move_score_fast(self, env: HexEnv, move_int: int) -> float:
        """Heuristique rapide pour les priors PUCT (expansion uniquement)."""
        r, c = divmod(move_int, BOARD_SIZE)
        is_blue = env.blue_to_play
        board = env.blue if is_blue else env.red
        opp_board = env.red if is_blue else env.blue

        score = 1.0
        score += self._bridge_score(board, opp_board, r, c)

        own = opp = 0
        for k in range(6):
            nr, nc = r + _DR[k], c + _DC[k]
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc]:
                    own += 1
                elif opp_board[nr, nc]:
                    opp += 1
        score += own * 6.0 + opp * 3.0
        score += (6 - abs(r - 5)) * 0.6 + (6 - abs(c - 5)) * 0.6

        if is_blue:
            if r == 0 or r == 10:
                score += 5.0
        else:
            if c == 0 or c == 10:
                score += 5.0
        return score

    def _rollout_move_score(self, env: HexEnv, move_int: int) -> float:
        """Heuristique encore plus rapide pour choisir un coup pendant le rollout."""
        r, c = divmod(move_int, BOARD_SIZE)
        is_blue = env.blue_to_play
        board = env.blue if is_blue else env.red
        opp_board = env.red if is_blue else env.blue

        score = 5.0

        # Ponts (même patterns que _bridge_score mais pondérés différemment)
        for (dr, dc), ((dr1, dc1), (dr2, dc2)) in _BRIDGE_PATTERNS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc]:
                    p1r, p1c = r + dr1, c + dc1
                    p2r, p2c = r + dr2, c + dc2
                    if (0 <= p1r < BOARD_SIZE and 0 <= p1c < BOARD_SIZE and
                            0 <= p2r < BOARD_SIZE and 0 <= p2c < BOARD_SIZE):
                        if opp_board[p1r, p1c] and opp_board[p2r, p2c]:
                            continue
                        if board[p1r, p1c] or board[p2r, p2c]:
                            score += 80.0
                        else:
                            score += 25.0

        # Voisinage
        own = opp = 0
        for k in range(6):
            nr, nc = r + _DR[k], c + _DC[k]
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc]:
                    own += 1
                elif opp_board[nr, nc]:
                    opp += 1
        score += own * 12.0 + opp * 5.0

        # Bords objectifs (très important en rollout pour guider la connexion)
        if is_blue:
            score += (10 - r) * 1.5
            if r == 0:
                score += 20.0
            if r == 10:
                score += 20.0
        else:
            score += (10 - c) * 1.5
            if c == 0:
                score += 20.0
            if c == 10:
                score += 20.0

        # Centre
        score += (5 - abs(r - 5)) * 0.8 + (5 - abs(c - 5)) * 0.8
        return score

    # ------------------------------------------------------------------ #
    #  BFS 0-1  →  distance minimale de connexion
    # ------------------------------------------------------------------ #
    def _dist_conn(self, env: HexEnv, is_blue: bool) -> int:
        """
        Retourne le nombre minimum de pierres vides nécessaires pour connecter
        les deux bords objectifs du joueur `is_blue`.
        0 = déjà connecté, 1 = un coup manque, etc.
        """
        board = env.blue if is_blue else env.red
        opp_board = env.red if is_blue else env.blue

        dist = self._dist_buf
        dist.fill(1000)

        self._visit_tok_local += 1
        if self._visit_tok_local > 250:
            self._visit_buf_local = bytearray(NUM_CELLS)
            self._visit_tok_local = 1
        tok = self._visit_tok_local
        vb = self._visit_buf_local

        dq = deque()

        if is_blue:
            for c in range(BOARD_SIZE):
                idx = c
                if board[0, c]:
                    dist[idx] = 0
                    vb[idx] = tok
                    dq.appendleft(idx)
                elif not opp_board[0, c]:
                    dist[idx] = 1
                    vb[idx] = tok
                    dq.append(idx)
        else:
            for r in range(BOARD_SIZE):
                idx = r * BOARD_SIZE
                if board[r, 0]:
                    dist[idx] = 0
                    vb[idx] = tok
                    dq.appendleft(idx)
                elif not opp_board[r, 0]:
                    dist[idx] = 1
                    vb[idx] = tok
                    dq.append(idx)

        while dq:
            idx = dq.popleft()
            r, c = divmod(idx, BOARD_SIZE)
            d = dist[idx]
            for k in range(6):
                nr, nc = r + _DR[k], c + _DC[k]
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    nidx = nr * BOARD_SIZE + nc
                    if opp_board[nr, nc]:
                        continue
                    cost = 0 if board[nr, nc] else 1
                    nd = d + cost
                    if nd < dist[nidx]:
                        dist[nidx] = nd
                        if cost == 0:
                            dq.appendleft(nidx)
                        else:
                            dq.append(nidx)

        if is_blue:
            return int(dist[BOARD_SIZE * (BOARD_SIZE - 1):].min())
        else:
            best = 1000
            for r in range(BOARD_SIZE):
                v = dist[r * BOARD_SIZE + (BOARD_SIZE - 1)]
                if v < best:
                    best = v
            return best

    def _eval_state(self, env: HexEnv) -> float:
        """
        Évalue la position pour Blue en fin de rollout (pas de winner).
        Retourne un float dans [0, 1].
        """
        d_blue = self._dist_conn(env, True)
        d_red = self._dist_conn(env, False)

        if d_blue == 0:
            return 1.0
        if d_red == 0:
            return 0.0

        # Si blue est beaucoup plus proche de la victoire que red,
        # la valeur tend vers 1.0 (et inversement).
        return 1.0 / (1.0 + math.exp(-(d_red - d_blue) * 0.45))

    # ------------------------------------------------------------------ #
    #  MCTS cœur
    # ------------------------------------------------------------------ #
    def _mcts(self, env: HexEnv, time_s: float):
        t0 = time.perf_counter()
        deadline = t0 + time_s * 0.85

        root_is_blue = env.blue_to_play
        root_moves = env.get_legal_moves()
        root_pool = FastPool(root_moves)
        root = _Node(None, -1, root_pool.copy(), root_is_blue)

        iters = 0
        while True:
            iters += 1
            if (iters & (self.CHECK_EVERY - 1)) == 0:
                if time.perf_counter() > deadline:
                    break

            node = root
            pool = root_pool.copy()
            sel_path = []

            # ---- Sélection ----
            while not node.untried.is_empty() and node.children:
                best_score = -float('inf')
                best_move = None
                best_child = None
                log_visits = math.log(node.visits)

                for move, child in node.children.items():
                    if child.solved is not None:
                        if child.solved == 1.0:
                            score = -1e12
                        else:
                            score = 1e12
                    elif child.visits == 0:
                        score = float('inf')
                    else:
                        if node.is_blue:
                            q = child.wins / child.visits
                        else:
                            q = 1.0 - child.wins / child.visits
                        p = child.prior
                        u = self.CPUCT * p * math.sqrt(log_visits) / (1.0 + child.visits)
                        score = q + u

                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child

                node = best_child
                pool.remove(best_move)
                env.apply_move(best_move)
                sel_path.append(best_move)

            # ---- Expansion ----
            if not node.untried.is_empty():
                n_untried = len(node.untried)
                if n_untried <= 6:
                    candidates = node.untried.tolist()
                else:
                    # échantillonne 6 coups sans réallocation lourde
                    idxs = self.rng.sample(range(n_untried), 6)
                    candidates = [int(node.untried.arr[i]) for i in idxs]

                scores = [self._move_score_fast(env, m) for m in candidates]
                scores = [max(0.0, s) for s in scores]
                total = sum(scores)
                if total <= 0:
                    move = self.rng.choice(candidates)
                    prior = 1.0 / n_untried
                else:
                    probs = [s / total for s in scores]
                    idx = int(np.random.choice(len(candidates), p=probs))
                    move = candidates[idx]
                    prior = probs[idx]

                node.untried.remove(move)
                pool.remove(move)
                env.apply_move(move)
                sel_path.append(move)

                child = _Node(node, move, pool.copy(), env.blue_to_play, prior)
                node.children[move] = child
                node = child

            # ---- Simulation (heavy rollout) ----
            sim_path = []
            winner = None
            while not pool.is_empty():
                if pool.n > 15 and len(sim_path) >= self.MAX_ROLLOUT:
                    break

                # Choix heuristique pondéré parmi un échantillon
                move = self._select_rollout_move(env, pool)
                pool.remove(move)
                env.apply_move(move)
                was_blue = not env.blue_to_play
                sim_path.append((move, was_blue))

                # Vérification rapide de victoire (cluster touche les 2 bords)
                r, c = divmod(move, BOARD_SIZE)
                if was_blue:
                    if _quick_win(env.blue, r, c, lambda rr, _: rr == 0, lambda rr, _: rr == 10):
                        winner = 'blue'
                        break
                else:
                    if _quick_win(env.red, r, c, lambda _, cc: cc == 0, lambda _, cc: cc == 10):
                        winner = 'red'
                        break

            # ---- Évaluation ----
            if winner is None:
                result_blue = self._eval_state(env)
            else:
                result_blue = 1.0 if winner == 'blue' else 0.0

            # ---- Rétropropagation + Solver ----
            solved_val = None
            if winner is not None:
                if winner == 'blue':
                    solved_val = 1.0 if node.is_blue else 0.0
                else:
                    solved_val = 0.0 if node.is_blue else 1.0

            n = node
            while n is not None:
                n.visits += 1
                if n.parent is not None:
                    if not n.is_blue:
                        n.wins += result_blue
                    else:
                        n.wins += (1.0 - result_blue)

                if solved_val is not None and n.solved is None:
                    n.solved = solved_val
                    parent = n.parent
                    if parent is not None:
                        if solved_val == 0.0:
                            parent.solved = 1.0
                            solved_val = 1.0
                            n = parent
                            continue
                        else:
                            if parent.untried.is_empty():
                                all_loss = True
                                for ch in parent.children.values():
                                    if ch.solved != 1.0:
                                        all_loss = False
                                        break
                                if all_loss:
                                    parent.solved = 0.0
                                    solved_val = 0.0
                                    n = parent
                                    continue
                solved_val = None
                n = n.parent

            # ---- Undo simulation ----
            for move, was_blue in reversed(sim_path):
                pool.add(move)
                env.undo_move(move, was_blue)

            # ---- Undo sélection ----
            for move in reversed(sel_path):
                pool.add(move)
                env.undo_move(move, not env.blue_to_play)

        # ---- Choix du coup final ----
        best_move = None
        best_visits = -1
        best_wr = 0.0
        for move, child in root.children.items():
            if child.solved == 0.0:
                continue
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                best_wr = child.wins / child.visits if child.visits else 0.0

        if best_move is None and root.children:
            # Tous les coups sont marqués solved==0.0 (perdus) → on joue le moins mauvais
            for move, child in root.children.items():
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_move = move
                    best_wr = child.wins / child.visits if child.visits else 0.0

        if root_is_blue:
            actual_wr = best_wr
        else:
            actual_wr = 1.0 - best_wr

        return best_move, iters, root.visits, actual_wr, time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    #  Helpers rollout
    # ------------------------------------------------------------------ #
    def _select_rollout_move(self, env: HexEnv, pool: FastPool) -> int:
        """Choix heuristique pondéré parmi un petit échantillon."""
        n = pool.n
        if n <= self.ROLLOUT_SAMPLES:
            candidates = [int(pool.arr[i]) for i in range(n)]
        else:
            idxs = self.rng.sample(range(n), self.ROLLOUT_SAMPLES)
            candidates = [int(pool.arr[i]) for i in idxs]

        scores = [self._rollout_move_score(env, m) for m in candidates]
        scores = [max(1.0, s) for s in scores]

        # Tirage pondéré (roulette) — version Python rapide
        total = sum(scores)
        pick = self.rng.random() * total
        cum = 0.0
        for m, s in zip(candidates, scores):
            cum += s
            if cum >= pick:
                return m
        return candidates[-1]


# ---------------------------------------------------------------------- #
#  Interface CLI (protocole BOARD / PLAYER)
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python kimi_k26.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = KimiK26()
    _move = _player.select_move(_env, _time_s)
    _stats = _player.last_stats
    if _stats:
        print(
            f"ITERS:{_stats.get('iters', 0)} "
            f"VISITS:{_stats.get('visits', 0)} "
            f"WINRATE:{_stats.get('winrate', 0.5):.4f} "
            f"TIME:{_stats.get('time', 0.0):.3f}",
            file=sys.stderr,
        )
    print(_env.pos_to_str(_move))

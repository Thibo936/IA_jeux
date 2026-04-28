import sys
import os
import time
import math
import random

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE


class _Node:
    """Nœud léger pour MCTS."""
    __slots__ = ('parent', 'move', 'children', 'wins', 'visits', 'untried', 'is_blue')

    def __init__(self, parent, move, untried, is_blue):
        self.parent = parent
        self.move = move
        self.children = {}
        self.wins = 0.0
        self.visits = 0
        self.untried = untried
        self.is_blue = is_blue


class KimiK26:
    """
    IA Hex 11×11 basée sur MCTS avec rollout heuristique.
    Détecte les coups gagnants/bloquants immédiats et utilise une
    politique de simulation biaisée par proximité et contrôle du centre.
    """

    NAME = "KimiK26"

    def __init__(self, seed=None):
        self.last_stats: dict = {}
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------ #
    #  API publique
    # ------------------------------------------------------------------ #
    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play
        root_player = 'blue' if root_blue else 'red'

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

        # 2) Coup bloquant immédiat (seulement si peu de coups)
        if len(moves) <= 60:
            opp = 'red' if root_blue else 'blue'
            opp_winning = []
            for om in moves:
                om_int = int(om)
                env.apply_move(om_int)
                w = env.winner()
                env.undo_move(om_int, not root_blue)
                if w == opp:
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
    #  Heuristique locale
    # ------------------------------------------------------------------ #
    def _move_score(self, env: HexEnv, move_int: int) -> float:
        """Évalue rapidement un coup (plus = meilleur)."""
        r, c = divmod(move_int, 11)
        blue = env.blue
        red = env.red
        is_blue = env.blue_to_play

        score = 1.0

        # Bonus centre
        score += (6 - abs(r - 5)) * 0.5 + (6 - abs(c - 5)) * 0.5

        # Voisinage hexagonal (6 directions)
        neighbors = ((r - 1, c), (r - 1, c + 1), (r, c - 1),
                     (r, c + 1), (r + 1, c - 1), (r + 1, c))
        own = opp = 0
        for nr, nc in neighbors:
            if 0 <= nr < 11 and 0 <= nc < 11:
                if is_blue:
                    if blue[nr, nc]:
                        own += 1
                    elif red[nr, nc]:
                        opp += 1
                else:
                    if red[nr, nc]:
                        own += 1
                    elif blue[nr, nc]:
                        opp += 1

        score += own * 6.0 + opp * 4.0

        # Bords stratégiques
        if is_blue:
            if r == 0 or r == 10:
                score += 3.0
        else:
            if c == 0 or c == 10:
                score += 3.0

        return score

    def _heuristic_move(self, env: HexEnv, legal: np.ndarray) -> int:
        """Choisit un coup selon une heuristique rapide (échantillonnée)."""
        n = len(legal)
        if n == 1:
            return int(legal[0])

        # On n'évalue pas tout le plateau : échantillon de 12 coups
        sample = 12 if n > 12 else n
        if sample == n:
            candidates = [int(x) for x in legal]
        else:
            indices = random.sample(range(n), sample)
            candidates = [int(legal[i]) for i in indices]

        best_score = -1.0
        best_move = candidates[0]
        for m in candidates:
            s = self._move_score(env, m)
            if s > best_score:
                best_score = s
                best_move = m
        return best_move

    # ------------------------------------------------------------------ #
    #  MCTS cœur
    # ------------------------------------------------------------------ #
    def _mcts(self, env: HexEnv, time_s: float):
        t0 = time.time()
        deadline = t0 + time_s * 0.95

        root_moves = [int(x) for x in env.get_legal_moves()]
        root = _Node(None, -1, root_moves, env.blue_to_play)

        iters = 0
        while time.time() < deadline:
            iters += 1
            node = root
            sim_env = env.copy()

            # ---- Sélection ----
            while not node.untried and node.children:
                best_score = -float('inf')
                best_move = None
                best_child = None
                log_visits = math.log(node.visits)
                for move, child in node.children.items():
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        q = child.wins / child.visits
                        u = math.sqrt(2.0 * log_visits / child.visits)
                        score = q + u
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                node = best_child
                sim_env.apply_move(best_move)

            # ---- Expansion ----
            if node.untried:
                if len(node.untried) == 1:
                    move = node.untried[0]
                else:
                    scores = [self._move_score(sim_env, m) for m in node.untried]
                    total = sum(scores)
                    if total <= 0:
                        move = random.choice(node.untried)
                    else:
                        pick = random.random() * total
                        s = 0.0
                        move = node.untried[-1]
                        for i, sc in enumerate(scores):
                            s += sc
                            if pick <= s:
                                move = node.untried[i]
                                break

                node.untried.remove(move)
                sim_env.apply_move(move)
                new_untried = [int(x) for x in sim_env.get_legal_moves()]
                child = _Node(node, move, new_untried, sim_env.blue_to_play)
                node.children[move] = child
                node = child

            # ---- Simulation ----
            while not sim_env.is_terminal():
                legal = sim_env.get_legal_moves()
                if len(legal) == 0:
                    break
                move = self._heuristic_move(sim_env, legal)
                sim_env.apply_move(move)

            # ---- Rétropropagation ----
            winner = sim_env.winner()
            result_blue = 1.0 if winner == 'blue' else 0.0

            n = node
            while n is not None:
                n.visits += 1
                if n.parent is not None:
                    # Le coup menant à ce nœud a été joué par `not n.is_blue`
                    if not n.is_blue:          # Blue a joué
                        n.wins += result_blue
                    else:                      # Red a joué
                        n.wins += (1.0 - result_blue)
                n = n.parent

        # ---- Choix du coup ----
        best_move = None
        best_visits = -1
        best_wr = 0.0
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                best_wr = child.wins / child.visits if child.visits > 0 else 0.0

        # Winrate du point de vue du joueur dont c'est le tour à la racine
        if env.blue_to_play:
            actual_wr = best_wr
        else:
            actual_wr = 1.0 - best_wr

        return best_move, iters, root.visits, actual_wr, time.time() - t0


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

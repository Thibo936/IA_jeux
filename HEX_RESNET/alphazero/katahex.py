# katahex.py — KataHex : MCTS avec prior par patterns et estimation d'ownership
# Inspiré de KataGo/KataHex : PUCT + prior heuristique + ownership map
# Interface CLI : python katahex.py BOARD PLAYER [time_s]

import sys
import os
import math
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from config import BOARD_SIZE, NUM_CELLS

_HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

# 6 directions de pont
_BRIDGE_PATTERNS = [
    ((-2, +1), (-1,  0), (-1, +1)),
    ((-1, +2), (-1, +1), ( 0, +1)),
    ((+1, +1), ( 0, +1), (+1,  0)),
    ((+2, -1), (+1, -1), (+1,  0)),
    ((+1, -2), ( 0, -1), (+1, -1)),
    ((-1, -1), (-1,  0), ( 0, -1)),
]

BS = BOARD_SIZE
CENTER = BS // 2


# ─── Calcul du prior par patterns ────────────────────────────────────────────

def _compute_prior(env: HexEnv) -> dict[int, float]:
    """
    Calcule une distribution prior sur les coups légaux (remplace le réseau).
    Features : centre, voisins amis, bord propre, complétion de pont.
    """
    legal = env.get_legal_moves()
    n = len(legal)
    if n == 0:
        return {}

    own = env.blue if env.blue_to_play else env.red
    opp = env.red if env.blue_to_play else env.blue
    scores = np.ones(n, dtype=np.float64) * 0.1

    # Détecter les cases de pont (intermédiaires de ponts amis vivants)
    bridge_cells: set[int] = set()
    for r in range(BS):
        for c in range(BS):
            if not own[r, c]:
                continue
            for (tdr, tdc), (i1r, i1c), (i2r, i2c) in _BRIDGE_PATTERNS:
                nr, nc = r + tdr, c + tdc
                if not (0 <= nr < BS and 0 <= nc < BS):
                    continue
                if not own[nr, nc]:
                    continue
                mr1, mc1 = r + i1r, c + i1c
                mr2, mc2 = r + i2r, c + i2c
                if not (0 <= mr1 < BS and 0 <= mc1 < BS):
                    continue
                if not (0 <= mr2 < BS and 0 <= mc2 < BS):
                    continue
                if opp[mr1, mc1] or opp[mr2, mc2]:
                    continue
                if not own[mr1, mc1]:
                    bridge_cells.add(mr1 * BS + mc1)
                if not own[mr2, mc2]:
                    bridge_cells.add(mr2 * BS + mc2)

    for i, m in enumerate(legal):
        m_int = int(m)
        r, c = divmod(m_int, BS)

        # Bonus centre (distance Manhattan)
        dist_c = abs(r - CENTER) + abs(c - CENTER)
        scores[i] += max(0, 6 - dist_c) * 0.5

        # Bonus voisin ami
        for dr, dc in _HEX_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BS and 0 <= nc < BS and own[nr, nc]:
                scores[i] += 1.5

        # Bonus bord propre (Blue: lignes 0/10, Red: colonnes 0/10)
        if env.blue_to_play:
            if r == 0 or r == BS - 1:
                scores[i] += 1.0
        else:
            if c == 0 or c == BS - 1:
                scores[i] += 1.0

        # Bonus complétion/défense de pont
        if m_int in bridge_cells:
            scores[i] += 3.0

    # Softmax avec température
    scores /= 2.0
    scores -= scores.max()
    probs = np.exp(scores)
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs[:] = 1.0 / n

    return {int(legal[i]): float(probs[i]) for i in range(n)}


# ─── Nœud PUCT ──────────────────────────────────────────────────────────────

class _KataNode:
    """Nœud MCTS avec prior (style PUCT / AlphaZero)."""

    __slots__ = ('env', 'parent', 'move', 'children', 'untried',
                 'visits', 'value_sum', 'prior')

    def __init__(self, env: HexEnv, prior_map: dict[int, float],
                 parent=None, move: int = -1):
        self.env = env
        self.parent = parent
        self.move = move
        self.children: dict[int, _KataNode] = {}
        # Trier les coups non explorés par prior décroissant
        self.untried = sorted(
            [int(m) for m in env.get_legal_moves()],
            key=lambda m: prior_map.get(m, 0.0),
            reverse=True,
        )
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior_map

    def is_terminal(self) -> bool:
        return self.env.is_terminal()


# ─── Joueur KataHex ──────────────────────────────────────────────────────────

class KataHexPlayer:
    """
    KataHex : MCTS avec PUCT et prior calculé par patterns.
    - Le prior guide l'exploration vers les coups structurellement forts
    - Formule PUCT (même que AlphaZero) avec prior heuristique au lieu du réseau
    - Ownership map : suivi de la possession des cases à travers les simulations
    Interface : select_move(env, time_s) -> int
    """

    def __init__(self, c_puct: float = 1.4, min_simulations: int = 100):
        self.c_puct = c_puct
        self.min_simulations = min_simulations
        self.last_stats: dict = {}
        self.ownership: np.ndarray | None = None  # carte d'ownership (pour analyse)

    @staticmethod
    def _winner_for_player(winner: str, is_blue: bool) -> float:
        if winner is None:
            return 0.0
        if is_blue:
            return 1.0 if winner == 'blue' else 0.0
        return 1.0 if winner == 'red' else 0.0

    def _puct_select(self, node: _KataNode) -> int:
        """Retourne le meilleur coup (enfant existant ou non exploré) par PUCT."""
        sqrt_n = math.sqrt(max(node.visits, 1))
        best_move = -1
        best_score = -1e18

        # Enfants déjà explorés
        for move, child in node.children.items():
            q = 1.0 - (child.value_sum / max(child.visits, 1))
            p = node.prior.get(move, 1e-6)
            score = q + self.c_puct * p * sqrt_n / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_move = move

        # Coups non explorés (FPU = valeur du parent ou 0.5)
        if node.visits > 0:
            fpu = 1.0 - (node.value_sum / node.visits)
        else:
            fpu = 0.5
        for move in node.untried:
            p = node.prior.get(move, 1e-6)
            score = fpu + self.c_puct * p * sqrt_n
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    @staticmethod
    def _rollout(env: HexEnv) -> str:
        """Rollout aléatoire jusqu'à la fin."""
        while not env.is_terminal():
            legal = env.get_legal_moves()
            move = int(np.random.choice(legal))
            env.apply_move(move)
        return env.winner()

    def _simulate_once(self, root: _KataNode,
                       own_map: np.ndarray) -> None:
        node = root

        # 1) Sélection + Expansion intégrées (style PUCT)
        while not node.is_terminal():
            move = self._puct_select(node)
            if move < 0:
                break

            if move in node.children:
                node = node.children[move]
            else:
                # Expansion du meilleur coup non exploré
                if move in node.untried:
                    node.untried.remove(move)
                child_env = node.env.copy()
                child_env.apply_move(move)
                child_prior = _compute_prior(child_env)
                child = _KataNode(child_env, child_prior,
                                  parent=node, move=move)
                node.children[move] = child
                node = child
                break  # nouveau nœud → rollout

        # 2) Simulation
        sim_env = node.env.copy()
        if not sim_env.is_terminal():
            winner = self._rollout(sim_env)
        else:
            winner = node.env.winner()

        # 3) Ownership : enregistrer la possession finale
        for pos in range(NUM_CELLS):
            r, c = divmod(pos, BS)
            if sim_env.blue[r, c]:
                own_map[pos] += 1.0
            elif sim_env.red[r, c]:
                own_map[pos] -= 1.0

        # 4) Backpropagation
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += self._winner_for_player(
                winner, cur.env.blue_to_play)
            cur = cur.parent

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        legal = env.get_legal_moves()
        if len(legal) == 0:
            return -1

        root_color = env.blue_to_play

        # Coup gagnant immédiat
        for move in legal:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, root_color)
            if (root_color and w == 'blue') or (not root_color and w == 'red'):
                self.last_stats = {
                    'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0,
                }
                print("ITERS:1 VISITS:1 WINRATE:1.0000 TIME:0.000",
                      file=sys.stderr)
                return m

        root_prior = _compute_prior(env)
        root = _KataNode(env.copy(), root_prior)

        own_map = np.zeros(NUM_CELLS, dtype=np.float64)

        t0 = time.time()
        deadline = t0 + max(time_s, 0.01)
        sims = 0

        while sims < self.min_simulations or time.time() < deadline:
            self._simulate_once(root, own_map)
            sims += 1

        # Normaliser l'ownership map
        if sims > 0:
            self.ownership = own_map / sims

        best_move = int(legal[0])
        best_visits = -1
        best_wr = 0.0
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                best_wr = 1.0 - (child.value_sum / max(child.visits, 1))

        elapsed = time.time() - t0
        self.last_stats = {
            'iters': sims, 'visits': root.visits,
            'winrate': best_wr, 'time': elapsed,
        }
        print(f"ITERS:{sims} VISITS:{root.visits} "
              f"WINRATE:{best_wr:.4f} TIME:{elapsed:.3f}",
              file=sys.stderr)
        return best_move


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python katahex.py BOARD PLAYER [time_s]",
              file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = KataHexPlayer()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

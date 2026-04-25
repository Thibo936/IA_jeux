# mcts_light.py — MCTS léger (UCT) pour Hex 11×11, sans réseau
# Politique de simulation : rollouts aléatoires
# Interface CLI : python mcts_light.py BOARD PLAYER [time_s]

import sys
import os
import math
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv


class _Node:
    """Nœud MCTS classique avec liste des coups non encore développés."""

    __slots__ = ('env', 'parent', 'move', 'children', 'untried', 'visits', 'wins')

    def __init__(self, env: HexEnv, parent=None, move: int = -1):
        self.env = env
        self.parent = parent
        self.move = move
        self.children: dict[int, _Node] = {}
        self.untried = [int(m) for m in env.get_legal_moves()]
        self.visits = 0
        self.wins = 0.0  # probabilité de victoire du joueur au trait dans ce nœud

    def is_terminal(self) -> bool:
        return self.env.is_terminal()


class LightMCTSPlayer:
    """
    MCTS léger UCT (sans politique/réseau) avec simulations aléatoires.
    Interface : select_move(env, time_s) -> int (index 0..120)
    """

    def __init__(self, c_uct: float = 1.4, min_simulations: int = 128):
        self.c_uct = c_uct
        self.min_simulations = min_simulations
        self.last_stats: dict = {}

    @staticmethod
    def _winner_for_player(winner: str, is_blue: bool) -> float:
        if winner is None:
            return 0.0
        if is_blue:
            return 1.0 if winner == 'blue' else 0.0
        return 1.0 if winner == 'red' else 0.0

    def _uct_select(self, node: _Node) -> _Node:
        """Sélectionne l'enfant max UCT du point de vue du joueur au trait de node."""
        log_n = math.log(max(node.visits, 1))
        best_child = None
        best_score = -1e18

        for child in node.children.values():
            if child.visits == 0:
                score = 1e9
            else:
                # child.wins/visits est du point de vue du joueur au trait dans child,
                # donc du point de vue de node c'est l'opposé.
                exploit = 1.0 - (child.wins / child.visits)
                explore = self.c_uct * math.sqrt(log_n / child.visits)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    @staticmethod
    def _rollout(env: HexEnv) -> str:
        """Rollout aléatoire jusqu'à la fin."""
        while not env.is_terminal():
            legal = env.get_legal_moves()
            move = int(np.random.choice(legal))
            env.apply_move(move)
        return env.winner()

    def _simulate_once(self, root: _Node) -> None:
        node = root

        # 1) Sélection
        while not node.is_terminal() and len(node.untried) == 0 and node.children:
            node = self._uct_select(node)

        # 2) Expansion
        if not node.is_terminal() and node.untried:
            move_idx = np.random.randint(len(node.untried))
            move = node.untried.pop(move_idx)
            child_env = node.env.copy()
            child_env.apply_move(move)
            child = _Node(child_env, parent=node, move=move)
            node.children[move] = child
            node = child

        # 3) Simulation
        sim_env = node.env.copy()
        winner = self._rollout(sim_env) if not sim_env.is_terminal() else sim_env.winner()

        # 4) Backpropagation
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.wins += self._winner_for_player(winner, cur.env.blue_to_play)
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
            if (root_color and w == 'blue') or ((not root_color) and w == 'red'):
                self.last_stats = {
                    'iters': 1,
                    'visits': 1,
                    'winrate': 1.0,
                    'time': 0.0,
                }
                print("ITERS:1 VISITS:1 WINRATE:1.0000 TIME:0.000", file=sys.stderr)
                return m

        root = _Node(env.copy())

        t0 = time.time()
        deadline = t0 + max(time_s, 0.01)
        sims = 0

        while sims < self.min_simulations or time.time() < deadline:
            self._simulate_once(root)
            sims += 1

        best_move = int(legal[0])
        best_visits = -1
        best_root_winrate = 0.0

        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                best_root_winrate = 1.0 - (child.wins / max(child.visits, 1))

        elapsed = time.time() - t0
        self.last_stats = {
            'iters': sims,
            'visits': root.visits,
            'winrate': best_root_winrate,
            'time': elapsed,
        }
        print(
            f"ITERS:{sims} VISITS:{root.visits} WINRATE:{best_root_winrate:.4f} TIME:{elapsed:.3f}",
            file=sys.stderr,
        )
        return best_move


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python mcts_light.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = LightMCTSPlayer()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

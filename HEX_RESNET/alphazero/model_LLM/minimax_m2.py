# minimax_m2.py — MCTS Hex AI with RAVE optimization
# Interface CLI : python minimax_m2.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random
from collections import defaultdict

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE


class MiniMaxM2:
    """
    MCTS AI for Hex 11×11 with RAVE (Rapid Action Value Estimation).
    Uses UCB1 exploration and AMAF (All-Move-As-First) for bias.
    """

    def __init__(self, seed: int | None = None):
        self.last_stats: dict = {}
        self._nodes = 0
        if seed is not None:
            random.seed(seed)

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        t0 = time.time()
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play

        for m in moves:
            mi = int(m)
            env.apply_move(mi)
            w = env.winner()
            env.undo_move(mi, root_blue)
            if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
                self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
                return mi

        iters = 0
        visits = 0
        winrate = 0.5

        root = Node(root_blue)
        deadline = t0 + time_s * 0.95

        while time.time() < deadline:
            node = root
            state = env.copy()

            while node.children:
                node = node.best_child(rave=True)
                state.apply_move(node.move)

            if not state.is_terminal():
                self._expand(node, state)
                if node.children:
                    node = node.best_child(rave=False)
                    state.apply_move(node.move)

            winner = self._rollout(state)
            self._backprop(root, node, winner, root_blue)

            iters += 1
            visits += sum(c.N for c in root.children.values())

            if root.children:
                best = max(root.children.values(), key=lambda c: c.N)
                winrate = best.Q

        elapsed = time.time() - t0
        best_child = max(root.children.values(), key=lambda c: c.N)

        self.last_stats = {
            'iters': iters,
            'visits': visits,
            'winrate': winrate,
            'time': elapsed,
        }
        return best_child.move

    def _expand(self, node: 'Node', state: HexEnv):
        moves = state.get_legal_moves()
        node.children = {int(m): Node(int(m), parent=node) for m in moves}

    def _rollout(self, state: HexEnv) -> bool:
        while not state.is_terminal():
            moves = state.get_legal_moves()
            if len(moves) == 0:
                break
            m = int(random.choice(moves))
            state.apply_move(m)
        w = state.winner()
        return w == 'blue'

    def _backprop(self, root: 'Node', leaf: 'Node', winner: bool, root_blue: bool):
        node = leaf
        while node is not None:
            node.N += 1
            if (node.side == root_blue) == winner:
                node.Q += 1.0
            node = node.parent


class Node:
    __slots__ = ['move', 'side', 'N', 'Q', 'children', 'parent', 'rave_n', 'rave_q']

    def __init__(self, move: int, side=None, parent=None):
        self.move = move
        self.side = side
        self.N = 0
        self.Q = 0.0
        self.children: dict[int, Node] = {}
        self.parent = parent
        self.rave_n = 0
        self.rave_q = 0.0

    def best_child(self, rave: bool = False) -> 'Node':
        if not self.children:
            return self
        if rave:
            return max(self.children.values(), key=lambda c: self._score_rave(c))
        return max(self.children.values(), key=lambda c: self._score_ucb(c))

    def _score_ucb(self, child: 'Node') -> float:
        if child.N == 0:
            return float('inf')
        return child.Q / child.N + math.sqrt(2 * math.log(self.N + 1) / child.N)

    def _score_rave(self, child: 'Node') -> float:
        alpha = self._rave_alpha(child.N)
        ucb = self._score_ucb(child)
        if child.rave_n < 5:
            return ucb
        rave_val = child.rave_q / child.rave_n if child.rave_n > 0 else 0.5
        return (1 - alpha) * ucb + alpha * rave_val

    def _rave_alpha(self, n: int) -> float:
        k = 30
        return k / (k + 3 * n)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python minimax_m2.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = MiniMaxM2()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

    stats = _player.last_stats
    if stats:
        print(f"ITERS:{stats.get('iters', 0)} "
              f"VISITS:{stats.get('visits', 0)} "
              f"WINRATE:{stats.get('winrate', 0):.4f} "
              f"TIME:{stats.get('time', 0):.3f}", file=sys.stderr)
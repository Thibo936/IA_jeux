# minimax_m2.py — Enhanced MCTS Hex AI with RAVE + Heuristic Rollouts
# Interface CLI : python minimax_m2.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random
import hashlib
from collections import defaultdict

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE

_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
_CENTER = BOARD_SIZE // 2


# ─── BFS 0-1 Path Evaluation (Numba-accelerated) ───────────────────────────────

try:
    import numba
    import numpy as np

    _NB_DR = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int32)
    _NB_DC = np.array([0, 1, -1, 1, -1, 0], dtype=np.int32)

    @numba.njit(cache=True)
    def _bfs_01_jit(player_flat, blocker_flat, start_row, N, nb_dr, nb_dc):
        INF = 1_000_000_000
        NN = N * N
        dist = np.full(NN, INF, dtype=np.int32)
        buf = np.empty(NN * 2, dtype=np.int32)
        head = NN
        tail = NN

        if start_row:
            for c in range(N):
                idx = c
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
                idx = r * N
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

    def _shortest_path(board_player, board_blocker, start_row):
        return int(_bfs_01_jit(
            board_player.ravel(), board_blocker.ravel(),
            start_row, BOARD_SIZE, _NB_DR, _NB_DC
        ))

    _HAS_NUMBA = True

except Exception:
    _HAS_NUMBA = False

    def _shortest_path(board_player, board_blocker, start_row):
        from collections import deque
        INF = 10 ** 9
        N = BOARD_SIZE
        dist = np.full((N, N), INF, dtype=np.int32)
        dq = deque()

        if start_row:
            for c in range(N):
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
            for r in range(N):
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

            if start_row and r == N - 1:
                return int(d_cur)
            if not start_row and c == N - 1:
                return int(d_cur)

            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= N or nc < 0 or nc >= N:
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

        return INF


# ─── Zobrist Hashing for Transposition Detection ───────────────────────────────

_rng = random.Random(42)
_ZOBRIST = [[_rng.getrandbits(64) for _ in range(2)] for _ in range(NUM_CELLS)]
_ZOBRIST_TURN = _rng.getrandbits(64)


def _compute_hash(env: HexEnv) -> int:
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


# ─── Node Class ────────────────────────────────────────────────────────────────

class Node:
    __slots__ = ['move', 'side', 'N', 'Q', 'children', 'parent',
                 'rave_n', 'rave_q', 'zhash']

    def __init__(self, move: int = -1, side=None, parent=None, zhash: int = 0):
        self.move = move
        self.side = side
        self.N = 0
        self.Q = 0.0
        self.children: dict[int, Node] = {}
        self.parent = parent
        self.rave_n = 0
        self.rave_q = 0.0
        self.zhash = zhash


# ─── RAVE Utilities ────────────────────────────────────────────────────────────

def _rave_alpha(n: int, k: int = 30) -> float:
    return k / (k + 3 * n)


# ─── Main MCTS Class ────────────────────────────────────────────────────────────

class MiniMaxM2:
    """
    Enhanced MCTS for Hex 11×11 with:
    - RAVE (Rapid Action Value Estimation) with proper AMAF backpropagation
    - Heuristic rollouts using BFS 0-1 path evaluation
    - Transposition-friendly tree structure
    - History heuristic for move ordering
    - Progressive widening for node expansion
    """

    def __init__(self, seed: int | None = None):
        self.last_stats: dict = {}
        self._nodes = 0
        self._history = [defaultdict(float), defaultdict(float)]
        if seed is not None:
            random.seed(seed)

    # ─── Heuristic: evaluate position from current player's perspective ──────────

    def _eval_position(self, env: HexEnv, blue_to_play: bool) -> float:
        pb = _shortest_path(env.blue, env.red, start_row=True)
        pr = _shortest_path(env.red, env.blue, start_row=False)
        INF = 10 ** 9

        if pb >= INF and pr >= INF:
            return 0.0
        elif pb >= INF:
            return -10.0
        elif pr >= INF:
            return 10.0
        else:
            score = (pr - pb) / 11.0
            return score if blue_to_play else -score

    # ─── Heuristic rollout: prefer moves that improve path ───────────────────

    def _heuristic_move(self, env: HexEnv, epsilon: float = 0.25) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1
        if len(moves) == 1:
            return int(moves[0])

        if random.random() < epsilon:
            return int(random.choice(moves))

        blue_to_play = env.blue_to_play
        current_score = self._eval_position(env, blue_to_play)

        best_move = int(moves[0])
        best_score = -1e9

        for m in moves:
            mi = int(m)
            env.apply_move(mi)
            eval_score = self._eval_position(env, not blue_to_play)
            env.undo_move(mi, blue_to_play)

            if eval_score > best_score:
                best_score = eval_score
                best_move = mi

        improved = abs(best_score) > abs(current_score) + 0.3
        if random.random() < (0.8 if improved else 0.2):
            return best_move

        return int(random.choice(moves))

    # ─── Rollout with heuristic ───────────────────────────────────────────────

    def _rollout(self, state: HexEnv, max_depth: int = 150) -> bool:
        depth = 0

        while not state.is_terminal() and depth < max_depth:
            moves = state.get_legal_moves()
            if len(moves) == 0:
                break

            m = self._heuristic_move(state)
            state.apply_move(m)
            depth += 1

        w = state.winner()
        return w == 'blue'

    # ─── AMAF Backpropagation (proper RAVE) ──────────────────────────────────

    def _backprop(self, root: Node, leaf: Node, winner: bool, root_blue: bool,
                   played_moves: list[int]):
        node = leaf
        while node is not None:
            node.N += 1
            if (node.side == root_blue) == winner:
                node.Q += 1.0
            node = node.parent

        for amaf_move in played_moves:
            an = root
            while an is not None:
                an.rave_n += 1
                if (an.side == root_blue) == winner:
                    an.rave_q += 1.0
                if amaf_move in an.children:
                    child = an.children[amaf_move]
                    child.rave_n += 1
                    if (child.side == root_blue) == winner:
                        child.rave_q += 1.0
                an = an.parent

    # ─── Select best child (UCB1 or RAVE) ────────────────────────────────────

    def _score_ucb(self, parent: Node, child: Node) -> float:
        if child.N == 0:
            return float('inf')
        return child.Q / child.N + math.sqrt(2 * math.log(parent.N + 1) / child.N)

    def _score_rave(self, parent: Node, child: Node) -> float:
        if child.rave_n < 5:
            return self._score_ucb(parent, child)
        alpha = _rave_alpha(child.N)
        ucb = self._score_ucb(parent, child)
        rave_val = child.rave_q / child.rave_n
        return (1 - alpha) * ucb + alpha * rave_val

    def _best_child(self, node: Node, rave: bool = False) -> Node:
        if not node.children:
            return node

        if rave:
            return max(node.children.values(), key=lambda c: self._score_rave(node, c))
        return max(node.children.values(), key=lambda c: self._score_ucb(node, c))

    # ─── Expand node with move ordering ──────────────────────────────────────

    def _expand(self, node: Node, state: HexEnv):
        moves = state.get_legal_moves()
        node_zhash = node.zhash
        if node_zhash == 0:
            node_zhash = _compute_hash(state)

        history_score = self._history[0 if state.blue_to_play else 1]

        scored_moves = []
        for m in moves:
            mi = int(m)
            hist = history_score.get(mi, 0.0)
            is_center = abs(mi // BOARD_SIZE - _CENTER) + abs(mi % BOARD_SIZE - _CENTER)
            scored_moves.append((hist, -is_center, mi))

        scored_moves.sort(reverse=True)

        for _, _, mi in scored_moves:
            child_zhash = node_zhash ^ _ZOBRIST[mi][0 if state.blue_to_play else 1] ^ _ZOBRIST_TURN
            node.children[mi] = Node(
                move=mi,
                side=not state.blue_to_play,
                parent=node,
                zhash=child_zhash
            )

    # ─── Progressive widening ────────────────────────────────────────────────

    def _should_expand(self, node: Node) -> bool:
        n = len(node.children)
        if n < 3:
            return True
        return random.random() < (3.0 / node.N) if node.N > 0 else True

    # ─── Single simulation ────────────────────────────────────────────────────

    def _simulate(self, root: Node, env: HexEnv) -> tuple[Node, list[int], bool]:
        node = root
        state = env.copy()
        played_moves = []

        while node.children:
            node = self._best_child(node, rave=True)
            state.apply_move(node.move)
            played_moves.append(node.move)

        if not state.is_terminal():
            if node.children or self._should_expand(node):
                if not node.children:
                    self._expand(node, state)
                if node.children:
                    node = self._best_child(node, rave=False)
                    state.apply_move(node.move)
                    played_moves.append(node.move)

        winner = self._rollout(state)
        self._backprop(root, node, winner, root.side == True if root.side is not None else True, played_moves)

        if root.children:
            best = max(root.children.values(), key=lambda c: c.N)
            if best.N > 0:
                color_idx = 0 if root.side == True else 1
                self._history[color_idx][best.move] += 1.0

        return node, played_moves, winner

    # ─── Main select_move ───────────────────────────────────────────────────

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

        root_zhash = _compute_hash(env)
        root = Node(side=root_blue, zhash=root_zhash)
        self._expand(root, env)

        deadline = t0 + time_s * 0.92

        while time.time() < deadline:
            local_env = env.copy()
            node, played, winner = self._simulate(root, local_env)
            iters += 1
            visits += root.N if root.N > 0 else 0

            if root.children:
                best = max(root.children.values(), key=lambda c: c.N)
                winrate = best.Q / best.N if best.N > 0 else 0.5

        elapsed = time.time() - t0

        if root.children:
            best_child = max(root.children.values(), key=lambda c: c.N)
            winrate = best_child.Q / best_child.N if best_child.N > 0 else 0.5
        else:
            best_child = None

        self.last_stats = {
            'iters': iters,
            'visits': visits,
            'winrate': winrate,
            'time': elapsed,
        }

        return best_child.move if best_child else int(moves[0])


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
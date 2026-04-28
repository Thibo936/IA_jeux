# mimo_v25_pro.py — MiMo-V2.5-Pro high: MCTS Hex AI with heuristic rollouts
# Interface CLI : python mimo_v25_pro.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE


class UnionFind:
    __slots__ = ('parent', 'rank')

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)


class MCTSNode:
    __slots__ = ('visits', 'wins', 'children', 'untried', 'parent', 'move')

    def __init__(self, parent=None, move=-1, untried=None):
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0.0
        self.children = {}
        self.untried = untried if untried is not None else []


class MimoV25Pro:
    def __init__(self, exploration=1.4, max_time=1.5):
        self.exploration = exploration
        self.max_time = max_time
        self.last_stats = {}
        self.simulations = 0

    def _count_connections(self, blue, red, is_blue):
        board = [[0] * 11 for _ in range(11)]
        for r in range(11):
            for c in range(11):
                if blue[r][c]:
                    board[r][c] = 1
                elif red[r][c]:
                    board[r][c] = 2

        uf = UnionFind(123)
        north, south, east, west = 121, 122, 119, 120

        for r in range(11):
            for c in range(11):
                idx = r * 11 + c
                if board[r][c] == 1:
                    if r == 0:
                        uf.union(idx, north)
                    if r == 10:
                        uf.union(idx, south)
                    for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                        nr, nc2 = r + dr, c + dc
                        if 0 <= nr < 11 and 0 <= nc2 < 11 and board[nr][nc2] == 1:
                            uf.union(idx, nr * 11 + nc2)
                elif board[r][c] == 2:
                    if c == 0:
                        uf.union(idx, west)
                    if c == 10:
                        uf.union(idx, east)
                    for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                        nr, nc2 = r + dr, c + dc
                        if 0 <= nr < 11 and 0 <= nc2 < 11 and board[nr][nc2] == 2:
                            uf.union(idx, nr * 11 + nc2)

        blue_connected = uf.connected(north, south)
        red_connected = uf.connected(east, west)

        if is_blue:
            return 1.0 if blue_connected else 0.0
        else:
            return 1.0 if red_connected else 0.0

    def _rollout(self, env, is_blue):
        sim_env = env.copy()
        rollout_moves = 0
        max_rollout = 121

        while not sim_env.is_terminal() and rollout_moves < max_rollout:
            legal = sim_env.get_legal_moves()
            if len(legal) == 0:
                break

            cur_blue = sim_env.blue_to_play
            move = self._rollout_move(sim_env, legal, cur_blue)
            sim_env.apply_move(move)
            rollout_moves += 1

        winner = sim_env.winner()
        if winner == 'blue':
            return 1.0 if is_blue else 0.0
        elif winner == 'red':
            return 0.0 if is_blue else 1.0
        return 0.5

    def _rollout_move(self, env, legal, is_blue):
        if len(legal) <= 2:
            return int(legal[random.randint(0, len(legal) - 1)])

        blue = env.blue
        red = env.red

        best_score = -1
        best_moves = []

        for move in legal:
            m = int(move)
            r, c = divmod(m, 11)
            score = random.random() * 0.3

            if is_blue:
                if r == 0 or r == 10:
                    score += 0.4
                if c >= 3 and c <= 7:
                    score += 0.15
                if r >= 1 and r <= 9:
                    score += 0.1
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 11 and 0 <= nc < 11 and blue[nr][nc]:
                        score += 0.25
            else:
                if c == 0 or c == 10:
                    score += 0.4
                if r >= 3 and r <= 7:
                    score += 0.15
                if c >= 1 and c <= 9:
                    score += 0.1
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 11 and 0 <= nc < 11 and red[nr][nc]:
                        score += 0.25

            if score > best_score:
                best_score = score
                best_moves = [m]
            elif score >= best_score - 0.05:
                best_moves.append(m)

        return best_moves[random.randint(0, len(best_moves) - 1)]

    def _get_ordered_moves(self, env, legal, is_blue):
        blue = env.blue
        red = env.red
        scored = []

        for move in legal:
            m = int(move)
            r, c = divmod(m, 11)
            score = 0

            if is_blue:
                if r == 0 or r == 10:
                    score += 5
                if 2 <= r <= 8 and 2 <= c <= 8:
                    score += 3
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 11 and 0 <= nc < 11 and blue[nr][nc]:
                        score += 4
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 11 and 0 <= nc < 11 and red[nr][nc]:
                        score += 1
            else:
                if c == 0 or c == 10:
                    score += 5
                if 2 <= r <= 8 and 2 <= c <= 8:
                    score += 3
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 11 and 0 <= nc < 11 and red[nr][nc]:
                        score += 4
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 11 and 0 <= nc < 11 and blue[nr][nc]:
                        score += 1

            scored.append((score + random.random() * 0.5, m))

        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def _check_win_or_block(self, env, moves, is_blue):
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, is_blue)
            if (is_blue and w == 'blue') or (not is_blue and w == 'red'):
                return m

        opponent_can_win = []
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, is_blue)
            if (is_blue and w == 'red') or (not is_blue and w == 'blue'):
                opponent_can_win.append(m)

        if len(opponent_can_win) == 1:
            return opponent_can_win[0]

        return None

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        self.simulations = 0
        t_start = time.time()
        deadline = t_start + time_s * 0.95

        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        is_blue = env.blue_to_play

        if len(moves) == 1:
            self.last_stats = {
                'iters': 0, 'visits': 0, 'winrate': 1.0,
                'time': time.time() - t_start
            }
            return int(moves[0])

        win_block = self._check_win_or_block(env, moves, is_blue)
        if win_block is not None:
            self.last_stats = {
                'iters': 1, 'visits': 1, 'winrate': 1.0,
                'time': time.time() - t_start
            }
            return win_block

        ordered = self._get_ordered_moves(env, moves, is_blue)
        root = MCTSNode(untried=ordered)

        total_visits = 0
        iters = 0

        while time.time() < deadline:
            node = root
            sim_env = env.copy()
            cur_blue = is_blue

            while not node.untried and node.children and not sim_env.is_terminal():
                best_score = -float('inf')
                best_child = None
                sqrt_parent = math.sqrt(max(1, node.visits))
                ln_parent = math.log(max(1, node.visits))

                for child in node.children.values():
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        exploit = child.wins / child.visits
                        if cur_blue:
                            explore = self.exploration * math.sqrt(ln_parent / child.visits)
                        else:
                            explore = self.exploration * math.sqrt(ln_parent / child.visits)
                        score = exploit + explore

                    if score > best_score:
                        best_score = score
                        best_child = child

                if best_child is None:
                    break

                sim_env.apply_move(best_child.move)
                cur_blue = not cur_blue
                node = best_child

            if sim_env.is_terminal():
                winner = sim_env.winner()
                result = 1.0 if (winner == 'blue' and is_blue) or (winner == 'red' and not is_blue) else 0.0
            elif node.untried:
                move = node.untried.pop()
                sim_env.apply_move(move)
                cur_blue_exp = not cur_blue

                legal = sim_env.get_legal_moves()
                ordered_children = self._get_ordered_moves(sim_env, legal, cur_blue_exp) if not sim_env.is_terminal() else []

                child = MCTSNode(parent=node, move=move, untried=ordered_children)
                node.children[move] = child

                result = self._rollout(sim_env, is_blue)
                node = child
                cur_blue = cur_blue_exp
            else:
                result = self._rollout(sim_env, is_blue)

            back_node = node
            back_blue = cur_blue
            while back_node is not None:
                back_node.visits += 1
                if back_blue == is_blue:
                    back_node.wins += result
                else:
                    back_node.wins += (1.0 - result)
                back_node = back_node.parent
                back_blue = not back_blue

            iters += 1
            total_visits += 1

        best_move = -1
        best_visits = -1
        best_wr = 0.0

        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                best_wr = child.wins / child.visits if child.visits > 0 else 0.0

        elapsed = time.time() - t_start
        self.last_stats = {
            'iters': iters,
            'visits': total_visits,
            'winrate': best_wr,
            'time': elapsed
        }

        return best_move


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python mimo_v25_pro.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = MimoV25Pro()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))
    s = _player.last_stats
    print(f"ITERS:{s.get('iters', 0)} VISITS:{s.get('visits', 0)} WINRATE:{s.get('winrate', 0):.4f} TIME:{s.get('time', 0):.3f}", file=sys.stderr)

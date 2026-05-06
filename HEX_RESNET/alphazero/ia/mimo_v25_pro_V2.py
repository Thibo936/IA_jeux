# mimo_v25_pro.py — MiMo-V2.5-Pro high: MCTS Hex AI with heuristic rollouts
# Improved: incremental state, bridge-save rollouts, virtual distance eval, PUCT
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

BOARD = 11
N = BOARD * BOARD

B_N = 121
B_S = 122
R_W = 123
R_E = 124
UF_SIZE = 125

_NEIGHBORS = []
for _r in range(BOARD):
    for _c in range(BOARD):
        nb = []
        for _dr, _dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < BOARD and 0 <= _nc < BOARD:
                nb.append(_nr * BOARD + _nc)
        _NEIGHBORS.append(tuple(nb))

_BRIDGE_DELTAS = (
    (-2, +1, -1, 0, -1, +1),
    (-1, +2, -1, +1, 0, +1),
    (+1, +1, 0, +1, +1, 0),
    (+2, -1, +1, 0, +1, -1),
    (+1, -2, 0, -1, +1, -1),
    (-1, -1, -1, 0, 0, -1),
)
_BRIDGES = []
for _r in range(BOARD):
    for _c in range(BOARD):
        bs = []
        for dr, dc, c1r, c1c, c2r, c2c in _BRIDGE_DELTAS:
            pr, pc = _r + dr, _c + dc
            cr1, cc1 = _r + c1r, _c + c1c
            cr2, cc2 = _r + c2r, _c + c2c
            if (0 <= pr < BOARD and 0 <= pc < BOARD and
                    0 <= cr1 < BOARD and 0 <= cc1 < BOARD and
                    0 <= cr2 < BOARD and 0 <= cc2 < BOARD):
                bs.append((pr * BOARD + pc, cr1 * BOARD + cc1, cr2 * BOARD + cc2))
        _BRIDGES.append(tuple(bs))

_CARRIER_OF = [[] for _ in range(N)]
for _p in range(N):
    for _q, _c1, _c2 in _BRIDGES[_p]:
        if _p < _q:
            _CARRIER_OF[_c1].append((_p, _q, _c2))
            _CARRIER_OF[_c2].append((_p, _q, _c1))
for _i in range(N):
    _CARRIER_OF[_i] = tuple(_CARRIER_OF[_i])

_CENTER = (BOARD - 1) / 2.0
_CENTER_BONUS = tuple(
    (BOARD - (abs(_r - _CENTER) + abs(_c - _CENTER)))
    for _r in range(BOARD) for _c in range(BOARD)
)


class State:
    __slots__ = ('blue', 'red', 'btp', 'par', 'rk', 'last_move')

    def __init__(self):
        self.blue = bytearray(N)
        self.red = bytearray(N)
        self.btp = True
        self.par = list(range(UF_SIZE))
        self.rk = [0] * UF_SIZE
        self.last_move = -1

    @classmethod
    def from_env(cls, env):
        s = cls()
        bb = env.blue
        rr = env.red
        for r in range(BOARD):
            for c in range(BOARD):
                p = r * BOARD + c
                if bb[r, c]:
                    s.blue[p] = 1
                    s._connect(p, True)
                elif rr[r, c]:
                    s.red[p] = 1
                    s._connect(p, False)
        s.btp = bool(env.blue_to_play)
        return s

    def _find(self, x):
        par = self.par
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def _union(self, a, b):
        par = self.par
        rk = self.rk
        ra = a
        while par[ra] != ra:
            par[ra] = par[par[ra]]
            ra = par[ra]
        rb = b
        while par[rb] != rb:
            par[rb] = par[par[rb]]
            rb = par[rb]
        if ra == rb:
            return
        if rk[ra] < rk[rb]:
            ra, rb = rb, ra
        par[rb] = ra
        if rk[ra] == rk[rb]:
            rk[ra] += 1

    def _connect(self, p, is_blue):
        if is_blue:
            r = p // BOARD
            if r == 0:
                self._union(p, B_N)
            if r == BOARD - 1:
                self._union(p, B_S)
            blue = self.blue
            for nb in _NEIGHBORS[p]:
                if blue[nb]:
                    self._union(p, nb)
        else:
            c = p - (p // BOARD) * BOARD
            if c == 0:
                self._union(p, R_W)
            if c == BOARD - 1:
                self._union(p, R_E)
            red = self.red
            for nb in _NEIGHBORS[p]:
                if red[nb]:
                    self._union(p, nb)

    def play(self, p):
        if self.btp:
            self.blue[p] = 1
            self._connect(p, True)
        else:
            self.red[p] = 1
            self._connect(p, False)
        self.last_move = p
        self.btp = not self.btp

    def winner(self):
        if self._find(B_N) == self._find(B_S):
            return 1
        if self._find(R_W) == self._find(R_E):
            return -1
        return 0

    def legal_moves(self):
        blue = self.blue
        red = self.red
        return [i for i in range(N) if not blue[i] and not red[i]]

    def copy(self):
        s = State.__new__(State)
        s.blue = bytearray(self.blue)
        s.red = bytearray(self.red)
        s.btp = self.btp
        s.par = self.par[:]
        s.rk = self.rk[:]
        s.last_move = self.last_move
        return s


def _shortest_path_dijkstra(board, player_val, blocker_val):
    INF = 10 ** 9
    dist = [INF] * N
    done = [False] * N
    if player_val == 1:
        for c in range(BOARD):
            idx = c
            if board[idx] == blocker_val:
                continue
            dist[idx] = 0 if board[idx] == player_val else 1
    else:
        for r in range(BOARD):
            idx = r * BOARD
            if board[idx] == blocker_val:
                continue
            dist[idx] = 0 if board[idx] == player_val else 1

    for _ in range(N):
        best = INF
        u = -1
        for i in range(N):
            if not done[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u < 0:
            break
        done[u] = True
        r_u = u // BOARD
        c_u = u % BOARD
        if player_val == 1 and r_u == BOARD - 1:
            return best
        if player_val == -1 and c_u == BOARD - 1:
            return best
        du = best
        for nb in _NEIGHBORS[u]:
            if done[nb]:
                continue
            cell = board[nb]
            if cell == player_val:
                step = 0
            elif cell == 0:
                step = 1
            else:
                step = 4
            nd = du + step
            if nd < dist[nb]:
                dist[nb] = nd

    goal = INF
    if player_val == 1:
        for c in range(BOARD):
            g = (BOARD - 1) * BOARD + c
            if dist[g] < goal:
                goal = dist[g]
    else:
        for r in range(BOARD):
            g = r * BOARD + (BOARD - 1)
            if dist[g] < goal:
                goal = dist[g]
    return goal


def _rollout(state):
    blue = state.blue
    red = state.red
    empties = []
    pos_of = [-1] * N
    for i in range(N):
        if not blue[i] and not red[i]:
            pos_of[i] = len(empties)
            empties.append(i)

    last_move = state.last_move
    rand = random.random

    while empties:
        my_arr = blue if state.btp else red

        chosen = -1
        if last_move >= 0:
            for p, q, other in _CARRIER_OF[last_move]:
                if my_arr[p] and my_arr[q] and pos_of[other] >= 0:
                    chosen = other
                    break

        if chosen < 0:
            idx = int(rand() * len(empties))
            chosen = empties[idx]
        else:
            idx = pos_of[chosen]

        last = empties[-1]
        empties[idx] = last
        pos_of[last] = idx
        empties.pop()
        pos_of[chosen] = -1

        state.play(chosen)
        last_move = chosen

        w = state.winner()
        if w != 0:
            return w

    return state.winner()


class _Node:
    __slots__ = ('children', 'untried', 'wins', 'visits', 'to_play',
                 'terminal_value', 'priors')

    def __init__(self, untried, to_play, terminal_value=0, priors=None):
        self.children = {}
        self.untried = untried
        self.wins = 0.0
        self.visits = 0
        self.to_play = to_play
        self.terminal_value = terminal_value
        self.priors = priors if priors is not None else {}


class MimoV25Pro:
    def __init__(self, exploration=1.4, max_time=1.5):
        self.exploration = exploration
        self.max_time = max_time
        self.last_stats = {}
        self.simulations = 0
        self.c_puct = 1.35

    def _compute_priors(self, board, moves, is_blue):
        if not moves:
            return {}
        priors = {}
        total = 0.0
        player_val = 1 if is_blue else -1
        for m in moves:
            score = 1.0
            row = m // BOARD
            col = m % BOARD

            adj_own = 0
            adj_opp = 0
            adj_empty = 0
            for nb in _NEIGHBORS[m]:
                v = board[nb]
                if v == player_val:
                    adj_own += 1
                elif v == -player_val:
                    adj_opp += 1
                else:
                    adj_empty += 1

            score += 2.4 * adj_own + 1.7 * adj_opp + 0.45 * adj_empty
            score += 0.35 * _CENTER_BONUS[m]

            edge_bonus = 0.0
            if is_blue:
                if row == 0 or row == BOARD - 1:
                    edge_bonus += 1.0
                if col == 0 or col == BOARD - 1:
                    edge_bonus += 0.25
            else:
                if col == 0 or col == BOARD - 1:
                    edge_bonus += 1.0
                if row == 0 or row == BOARD - 1:
                    edge_bonus += 0.25
            score += 0.8 * edge_bonus

            if score < 0.01:
                score = 0.01
            priors[m] = score
            total += score

        if total <= 0.0:
            u = 1.0 / len(moves)
            for m in moves:
                priors[m] = u
        else:
            inv = 1.0 / total
            for m in moves:
                priors[m] *= inv
        return priors

    def _evaluate_value(self, board, root_player, to_play):
        blue_cost = _shortest_path_dijkstra(board, 1, -1)
        red_cost = _shortest_path_dijkstra(board, -1, 1)
        if root_player == 1:
            own_cost = blue_cost
            opp_cost = red_cost
        else:
            own_cost = red_cost
            opp_cost = blue_cost

        delta = float(opp_cost - own_cost)
        x = max(-12.0, min(12.0, 0.85 * delta))
        p = 1.0 / (1.0 + math.exp(-x))
        if to_play == root_player:
            p += 0.015
        else:
            p -= 0.015
        if p < 0.001:
            p = 0.001
        elif p > 0.999:
            p = 0.999
        return p

    def _check_win(self, board, player):
        if player == 1:
            seen = [False] * N
            stack = []
            for p in range(BOARD):
                if board[p] == 1:
                    seen[p] = True
                    stack.append(p)
            while stack:
                cur = stack.pop()
                if cur // BOARD == BOARD - 1:
                    return True
                for nb in _NEIGHBORS[cur]:
                    if not seen[nb] and board[nb] == 1:
                        seen[nb] = True
                        stack.append(nb)
            return False
        else:
            seen = [False] * N
            stack = []
            for r in range(BOARD):
                p = r * BOARD
                if board[p] == -1:
                    seen[p] = True
                    stack.append(p)
            while stack:
                cur = stack.pop()
                if cur % BOARD == BOARD - 1:
                    return True
                for nb in _NEIGHBORS[cur]:
                    if not seen[nb] and board[nb] == -1:
                        seen[nb] = True
                        stack.append(nb)
            return False

    def _find_immediate_wins(self, board, player, legal_moves):
        wins = []
        for m in legal_moves:
            board[m] = player
            if self._check_win(board, player):
                wins.append(m)
            board[m] = 0
        return wins

    def _best_block_move(self, board, player, threats):
        if len(threats) == 1:
            return threats[0]

        best_move = threats[0]
        best_val = -1.0
        for m in threats:
            board[m] = player
            if self._check_win(board, player):
                board[m] = 0
                return m
            val = self._evaluate_value(board, player, -player)
            board[m] = 0
            if val > best_val:
                best_val = val
                best_move = m
        return best_move

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        self.simulations = 0
        t_start = time.time()
        deadline = t_start + max(0.05, time_s - 0.05)

        moves_arr = env.get_legal_moves()
        n_legal = len(moves_arr)
        if n_legal == 0:
            self.last_stats = {'iters': 0, 'visits': 0, 'winrate': 0.0, 'time': 0.0}
            return -1

        legal = [int(m) for m in moves_arr]

        board = [0] * N
        for r in range(BOARD):
            for c in range(BOARD):
                p = r * BOARD + c
                if env.blue[r, c]:
                    board[p] = 1
                elif env.red[r, c]:
                    board[p] = -1

        root_player = 1 if env.blue_to_play else -1

        own_wins = self._find_immediate_wins(board, root_player, legal)
        if own_wins:
            best_immediate = own_wins[0]
            best_s = -1.0
            for m in own_wins:
                s = _CENTER_BONUS[m]
                if s > best_s:
                    best_s = s
                    best_immediate = m
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': time.time() - t_start}
            return best_immediate

        opp_wins = self._find_immediate_wins(board, -root_player, legal)
        if opp_wins:
            block_move = self._best_block_move(board, root_player, opp_wins)
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 0.6, 'time': time.time() - t_start}
            return block_move

        if n_legal == 1:
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 0.5, 'time': time.time() - t_start}
            return legal[0]

        center = (BOARD - 1) // 2
        ordered = sorted(legal, key=lambda m: abs(m // BOARD - center) + abs(m % BOARD - center))
        root_priors = self._compute_priors(board, ordered, root_player == 1)

        state = State.from_env(env)
        root = _Node(untried=ordered, to_play=state.btp, priors=root_priors)

        iters = 0
        check_every = 64
        while True:
            if iters % check_every == 0 and time.time() >= deadline:
                break
            iters += 1
            self._iterate(root, state, root_player)

        if not root.children:
            return ordered[0]

        best_mv = -1
        best_visits = -1
        best_q = -1.0
        for mv, ch in root.children.items():
            q = ch.wins / ch.visits if ch.visits > 0 else 0.0
            if ch.visits > best_visits or (ch.visits == best_visits and q > best_q):
                best_visits = ch.visits
                best_q = q
                best_mv = mv

        if best_mv < 0:
            best_mv = ordered[0]

        elapsed = time.time() - t_start
        total_visits = sum(c.visits for c in root.children.values())
        winrate = best_q if best_visits > 0 else 0.5

        self.last_stats = {
            'iters': iters,
            'visits': total_visits,
            'winrate': float(winrate),
            'time': elapsed,
        }
        return best_mv

    def _iterate(self, root, root_state, root_player):
        state = root_state.copy()
        node = root
        path = [node]

        while (node.terminal_value == 0
               and not node.untried
               and node.children):
            mv, ch = self._select_child(node, root_player)
            if mv < 0:
                break
            state.play(mv)
            node = ch
            path.append(node)

        if node.terminal_value != 0:
            self._backprop(path, node.terminal_value, root_player)
            return

        if node.untried:
            mv = node.untried.pop()
            state.play(mv)
            w = state.winner()
            if w != 0:
                child = _Node(untried=[], to_play=state.btp, terminal_value=w)
            else:
                legal = state.legal_moves()
                board_list = [state.blue[i] - state.red[i] for i in range(N)]
                priors = self._compute_priors(board_list, legal, state.btp)
                child = _Node(untried=legal, to_play=state.btp, priors=priors)
            node.children[mv] = child
            path.append(child)
            if w != 0:
                self._backprop(path, w, root_player)
                return
            node = child

        winner_color = _rollout(state)
        self._backprop(path, winner_color, root_player)

    def _select_child(self, node, root_player):
        sqrt_n = math.sqrt(node.visits + 1.0)
        default_prior = 1.0 / max(1, len(node.children))

        best_move = -1
        best_child = None
        best_score = -1e18

        root_turn = (node.to_play == root_player)
        for m, child in node.children.items():
            if child.visits > 0:
                q = child.wins / child.visits
            else:
                q = 0.5
            if not root_turn:
                q = 1.0 - q
            prior = node.priors.get(m, default_prior)
            u = self.c_puct * prior * sqrt_n / (1.0 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = m
                best_child = child
        return best_move, best_child

    @staticmethod
    def _backprop(path, winner_color, root_player):
        for i, node in enumerate(path):
            node.visits += 1
            if i == 0:
                continue
            mover_color = 1 if path[i - 1].to_play else -1
            if mover_color == winner_color:
                node.wins += 1.0


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

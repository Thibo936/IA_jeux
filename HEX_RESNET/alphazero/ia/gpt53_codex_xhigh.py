# gpt53_codex_xhigh.py - GPT 5.3 Codex Xhigh Hex 11x11 player
# CLI interface: python gpt53_codex_xhigh.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random
from collections import deque


_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from hex_env import HexEnv
except Exception:
    HexEnv = None


BOARD_SIZE = 11
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
MODEL_NAME = "GPT 5.3 Codex Xhigh"

_LAST = BOARD_SIZE - 1
_BLUE_START = tuple(range(BOARD_SIZE))
_RED_START = tuple(r * BOARD_SIZE for r in range(BOARD_SIZE))
_BLUE_GOAL = tuple((_LAST * BOARD_SIZE) + c for c in range(BOARD_SIZE))
_RED_GOAL = tuple(r * BOARD_SIZE + _LAST for r in range(BOARD_SIZE))

_ROW = tuple(i // BOARD_SIZE for i in range(NUM_CELLS))
_COL = tuple(i % BOARD_SIZE for i in range(NUM_CELLS))

_CENTER = (BOARD_SIZE - 1) / 2.0
_CENTER_BONUS = tuple(
    (BOARD_SIZE - (abs(_ROW[i] - _CENTER) + abs(_COL[i] - _CENTER)))
    for i in range(NUM_CELLS)
)

_NEIGHBORS = []
for _r in range(BOARD_SIZE):
    for _c in range(BOARD_SIZE):
        _ns = []
        for _rr, _cc in (
            (_r - 1, _c),
            (_r + 1, _c),
            (_r, _c - 1),
            (_r, _c + 1),
            (_r - 1, _c + 1),
            (_r + 1, _c - 1),
        ):
            if 0 <= _rr < BOARD_SIZE and 0 <= _cc < BOARD_SIZE:
                _ns.append(_rr * BOARD_SIZE + _cc)
        _NEIGHBORS.append(tuple(_ns))
_NEIGHBORS = tuple(_NEIGHBORS)


class _Node:
    __slots__ = (
        'parent', 'move', 'to_play', 'depth', 'wins', 'visits',
        'children', 'untried', 'priors', 'terminal', 'winner'
    )

    def __init__(self, parent, move, to_play, depth, terminal, winner, untried, priors):
        self.parent = parent
        self.move = move
        self.to_play = to_play
        self.depth = depth
        self.terminal = terminal
        self.winner = winner
        self.wins = 0.0
        self.visits = 0
        self.children = {}
        self.untried = untried
        self.priors = priors


class GPT53CodexXhigh:
    def __init__(self, seed=None, c_puct=1.35):
        self.model_name = MODEL_NAME
        self.last_stats = {}
        self.c_puct = float(c_puct)
        self.rng = random.Random(seed)
        self.max_iterations = 50000
        self.opponent_cost = 4

    def _extract_board_player(self, env):
        if hasattr(env, 'to_string'):
            board_str = env.to_string()
            board = [0] * NUM_CELLS
            lim = min(NUM_CELLS, len(board_str))
            for i in range(lim):
                ch = board_str[i]
                if ch == 'O':
                    board[i] = 1
                elif ch == '@':
                    board[i] = -1
            player = 1 if bool(getattr(env, 'blue_to_play', True)) else -1
            return board, player

        if hasattr(env, 'blue') and hasattr(env, 'red'):
            board = [0] * NUM_CELLS
            idx = 0
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if env.blue[r, c]:
                        board[idx] = 1
                    elif env.red[r, c]:
                        board[idx] = -1
                    idx += 1
            player = 1 if bool(getattr(env, 'blue_to_play', True)) else -1
            return board, player

        raise ValueError('Unsupported env format: missing to_string()/blue_to_play.')

    def _legal_moves(self, board):
        return [i for i, v in enumerate(board) if v == 0]

    def _check_win(self, board, player):
        if player == 1:
            seen = [False] * NUM_CELLS
            stack = deque()
            for p in _BLUE_START:
                if board[p] == 1:
                    seen[p] = True
                    stack.append(p)
            while stack:
                cur = stack.pop()
                if _ROW[cur] == _LAST:
                    return True
                for nb in _NEIGHBORS[cur]:
                    if (not seen[nb]) and board[nb] == 1:
                        seen[nb] = True
                        stack.append(nb)
            return False

        seen = [False] * NUM_CELLS
        stack = deque()
        for p in _RED_START:
            if board[p] == -1:
                seen[p] = True
                stack.append(p)
        while stack:
            cur = stack.pop()
            if _COL[cur] == _LAST:
                return True
            for nb in _NEIGHBORS[cur]:
                if (not seen[nb]) and board[nb] == -1:
                    seen[nb] = True
                    stack.append(nb)
        return False

    def _shortest_path_cost(self, board, player):
        inf = 10 ** 9
        dist = [inf] * NUM_CELLS
        done = [False] * NUM_CELLS
        opp_cost = self.opponent_cost

        if player == 1:
            starts = _BLUE_START
            goals = _BLUE_GOAL
            for s in starts:
                v = board[s]
                if v == -1:
                    continue
                dist[s] = 0 if v == 1 else 1

            for _ in range(NUM_CELLS):
                best = inf
                u = -1
                for i in range(NUM_CELLS):
                    if (not done[i]) and dist[i] < best:
                        best = dist[i]
                        u = i
                if u < 0:
                    break
                done[u] = True
                if _ROW[u] == _LAST:
                    return best
                du = best
                for nb in _NEIGHBORS[u]:
                    if done[nb]:
                        continue
                    cell = board[nb]
                    if cell == 1:
                        step = 0
                    elif cell == 0:
                        step = 1
                    else:
                        step = opp_cost
                    nd = du + step
                    if nd < dist[nb]:
                        dist[nb] = nd
        else:
            starts = _RED_START
            goals = _RED_GOAL
            for s in starts:
                v = board[s]
                if v == 1:
                    continue
                dist[s] = 0 if v == -1 else 1

            for _ in range(NUM_CELLS):
                best = inf
                u = -1
                for i in range(NUM_CELLS):
                    if (not done[i]) and dist[i] < best:
                        best = dist[i]
                        u = i
                if u < 0:
                    break
                done[u] = True
                if _COL[u] == _LAST:
                    return best
                du = best
                for nb in _NEIGHBORS[u]:
                    if done[nb]:
                        continue
                    cell = board[nb]
                    if cell == -1:
                        step = 0
                    elif cell == 0:
                        step = 1
                    else:
                        step = opp_cost
                    nd = du + step
                    if nd < dist[nb]:
                        dist[nb] = nd

        best_goal = inf
        for g in goals:
            if dist[g] < best_goal:
                best_goal = dist[g]
        return best_goal

    def _evaluate_value(self, board, root_player, to_play):
        blue_cost = self._shortest_path_cost(board, 1)
        red_cost = self._shortest_path_cost(board, -1)
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

    def _local_move_score(self, board, move, player):
        own_adj = 0
        opp_adj = 0
        empty_adj = 0
        for nb in _NEIGHBORS[move]:
            v = board[nb]
            if v == player:
                own_adj += 1
            elif v == -player:
                opp_adj += 1
            else:
                empty_adj += 1

        row = _ROW[move]
        col = _COL[move]
        edge_bonus = 0.0
        if player == 1:
            if row == 0 or row == _LAST:
                edge_bonus += 1.0
            if col == 0 or col == _LAST:
                edge_bonus += 0.25
        else:
            if col == 0 or col == _LAST:
                edge_bonus += 1.0
            if row == 0 or row == _LAST:
                edge_bonus += 0.25

        score = (
            1.0
            + 2.4 * own_adj
            + 1.7 * opp_adj
            + 0.45 * empty_adj
            + 0.35 * _CENTER_BONUS[move]
            + 0.8 * edge_bonus
        )
        return score

    def _branch_cap(self, depth, n_moves):
        if depth == 0:
            if n_moves > 90:
                return 20
            if n_moves > 70:
                return 24
            if n_moves > 45:
                return 28
            return min(34, n_moves)
        if depth == 1:
            return min(24, n_moves)
        if depth == 2:
            return min(16, n_moves)
        if depth == 3:
            return min(12, n_moves)
        return min(8, n_moves)

    def _candidate_moves(self, board, legal_moves, player, depth):
        n = len(legal_moves)
        if n <= 1:
            return legal_moves[:]
        cap = self._branch_cap(depth, n)
        if n <= cap:
            return legal_moves[:]

        scored = []
        for m in legal_moves:
            scored.append((self._local_move_score(board, m, player), m))
        scored.sort(reverse=True)
        return [m for _, m in scored[:cap]]

    def _compute_priors(self, board, moves, player):
        if not moves:
            return {}
        priors = {}
        total = 0.0
        for m in moves:
            s = self._local_move_score(board, m, player)
            if s < 0.01:
                s = 0.01
            priors[m] = s
            total += s
        if total <= 0.0:
            u = 1.0 / len(moves)
            for m in moves:
                priors[m] = u
            return priors
        inv = 1.0 / total
        for m in moves:
            priors[m] *= inv
        return priors

    def _pick_untried(self, node):
        best = node.untried[0]
        best_s = -1.0
        for m in node.untried:
            s = node.priors.get(m, 0.0) + self.rng.random() * 1e-6
            if s > best_s:
                best_s = s
                best = m
        node.untried.remove(best)
        return best

    def _select_child(self, node, root_player):
        sqrt_n = math.sqrt(node.visits + 1.0)
        default_prior = 1.0 / max(1, len(node.children))

        best_move = None
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

    def _create_node(self, parent, move, to_play, depth, board, winner):
        terminal = (winner != 0)
        untried = []
        priors = {}

        if not terminal:
            legal = self._legal_moves(board)
            if not legal:
                terminal = True
                if self._check_win(board, 1):
                    winner = 1
                elif self._check_win(board, -1):
                    winner = -1
                else:
                    winner = 0
            else:
                untried = self._candidate_moves(board, legal, to_play, depth)
                priors = self._compute_priors(board, untried, to_play)

        return _Node(
            parent=parent,
            move=move,
            to_play=to_play,
            depth=depth,
            terminal=terminal,
            winner=winner,
            untried=untried,
            priors=priors,
        )

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
            return threats[0], 0.60

        best_move = threats[0]
        best_val = -1.0
        for m in threats:
            board[m] = player
            if self._check_win(board, player):
                val = 1.0
            else:
                val = self._evaluate_value(board, player, -player)
            board[m] = 0
            if val > best_val:
                best_val = val
                best_move = m
        return best_move, best_val

    def _fallback_best_move(self, board, legal_moves, player):
        if not legal_moves:
            return -1, 0.0

        if len(legal_moves) > 40:
            scored = []
            for m in legal_moves:
                scored.append((self._local_move_score(board, m, player), m))
            scored.sort(reverse=True)
            candidates = [m for _, m in scored[:24]]
        else:
            candidates = legal_moves

        best_move = candidates[0]
        best_score = -1e18
        best_wr = 0.5
        for m in candidates:
            board[m] = player
            if self._check_win(board, player):
                board[m] = 0
                return m, 1.0
            v = self._evaluate_value(board, player, -player)
            s = v + 0.02 * self._local_move_score(board, m, player)
            board[m] = 0
            if s > best_score:
                best_score = s
                best_move = m
                best_wr = v
        return best_move, best_wr

    def select_move(self, env, time_s=1.5):
        t0 = time.time()
        board, root_player = self._extract_board_player(env)
        legal_moves = self._legal_moves(board)
        if not legal_moves:
            self.last_stats = {'iters': 0, 'visits': 0, 'winrate': 0.0, 'time': 0.0}
            return -1

        own_wins = self._find_immediate_wins(board, root_player, legal_moves)
        if own_wins:
            best_immediate = own_wins[0]
            best_score = -1.0
            for m in own_wins:
                s = self._local_move_score(board, m, root_player)
                if s > best_score:
                    best_score = s
                    best_immediate = m
            elapsed = time.time() - t0
            self.last_stats = {
                'iters': 1,
                'visits': 1,
                'winrate': 1.0,
                'time': elapsed,
            }
            return int(best_immediate)

        opp_wins = self._find_immediate_wins(board, -root_player, legal_moves)
        if opp_wins:
            block_move, block_wr = self._best_block_move(board, root_player, opp_wins)
            elapsed = time.time() - t0
            self.last_stats = {
                'iters': 1,
                'visits': 1,
                'winrate': float(block_wr),
                'time': elapsed,
            }
            return int(block_move)

        if len(legal_moves) == 1:
            elapsed = time.time() - t0
            self.last_stats = {
                'iters': 1,
                'visits': 1,
                'winrate': 0.5,
                'time': elapsed,
            }
            return int(legal_moves[0])

        root_candidates = self._candidate_moves(board, legal_moves, root_player, depth=0)
        root_priors = self._compute_priors(board, root_candidates, root_player)
        root = _Node(
            parent=None,
            move=None,
            to_play=root_player,
            depth=0,
            terminal=False,
            winner=0,
            untried=root_candidates,
            priors=root_priors,
        )

        budget = max(0.02, float(time_s) * 0.94)
        deadline = t0 + budget

        iters = 0
        check_every = 8
        while iters < self.max_iterations:
            if (iters % check_every == 0) and (time.time() >= deadline):
                break

            node = root
            sim_board = board[:]
            player = root_player

            while True:
                if node.terminal:
                    break

                if node.untried:
                    move = self._pick_untried(node)
                    sim_board[move] = player
                    winner = player if self._check_win(sim_board, player) else 0
                    child = self._create_node(
                        parent=node,
                        move=move,
                        to_play=-player,
                        depth=node.depth + 1,
                        board=sim_board,
                        winner=winner,
                    )
                    node.children[move] = child
                    node = child
                    player = -player
                    break

                if not node.children:
                    break

                move, child = self._select_child(node, root_player)
                sim_board[move] = player
                node = child
                player = -player

            if node.terminal:
                if node.winner == 0:
                    value = 0.5
                else:
                    value = 1.0 if node.winner == root_player else 0.0
            else:
                value = self._evaluate_value(sim_board, root_player, node.to_play)

            cur = node
            while cur is not None:
                cur.visits += 1
                cur.wins += value
                cur = cur.parent

            iters += 1

        best_move = None
        best_child = None
        if root.children:
            best_visits = -1
            best_q = -1.0
            for m, ch in root.children.items():
                q = (ch.wins / ch.visits) if ch.visits > 0 else 0.0
                if (ch.visits > best_visits) or (ch.visits == best_visits and q > best_q):
                    best_visits = ch.visits
                    best_q = q
                    best_move = m
                    best_child = ch

        if best_move is None:
            best_move, wr = self._fallback_best_move(board, legal_moves, root_player)
        else:
            wr = (best_child.wins / best_child.visits) if best_child and best_child.visits > 0 else 0.5

        elapsed = time.time() - t0
        visits = sum(ch.visits for ch in root.children.values())
        self.last_stats = {
            'iters': int(iters),
            'visits': int(visits),
            'winrate': float(wr),
            'time': elapsed,
        }
        return int(best_move)


def _pos_to_str(pos):
    r, c = divmod(int(pos), BOARD_SIZE)
    return f"{chr(ord('A') + c)}{r + 1}"


class _CLIEnv:
    def __init__(self, board, player_char):
        self._board = board
        self.blue_to_play = (player_char == 'O')

    def to_string(self):
        return self._board

    def pos_to_str(self, pos):
        return _pos_to_str(pos)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python gpt53_codex_xhigh.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) or '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _board_arg = sys.argv[1]
    _player_arg = sys.argv[2]
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5

    if len(_board_arg) != NUM_CELLS:
        print("invalid BOARD length, expected 121 chars", file=sys.stderr)
        sys.exit(2)
    if _player_arg not in ('O', '@'):
        print("invalid PLAYER, expected 'O' or '@'", file=sys.stderr)
        sys.exit(2)

    if HexEnv is not None:
        _env = HexEnv.from_string(_board_arg, _player_arg)
    else:
        _env = _CLIEnv(_board_arg, _player_arg)

    _player = GPT53CodexXhigh()
    _move = _player.select_move(_env, _time_s)

    if hasattr(_env, 'pos_to_str'):
        print(_env.pos_to_str(_move))
    else:
        print(_pos_to_str(_move))

    _s = _player.last_stats if isinstance(_player.last_stats, dict) else {}
    print(
        f"ITERS:{int(_s.get('iters', 0))} "
        f"VISITS:{int(_s.get('visits', 0))} "
        f"WINRATE:{float(_s.get('winrate', 0.0)):.4f} "
        f"TIME:{float(_s.get('time', 0.0)):.3f}",
        file=sys.stderr,
    )

#!/usr/bin/env python3
# deepseek_v4_pro_max.py — DeepSeek V4 Pro MAX MCTS Hex 11×11
# CLI : python deepseek_v4_pro_max.py BOARD PLAYER [time_s]

import sys, os, time, math, random
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hex_env import HexEnv

SIZE = 11; NC = SIZE * SIZE; C = math.sqrt(2.0)

# ─── Précalculs ───────────────────────────────────────────────────────────────
NB_RC = [None] * NC; NB_IDX = [None] * NC
CD = np.zeros(NC, np.float32); EDG = np.zeros((4, NC), np.float32)
for r in range(SIZE):
    for c in range(SIZE):
        i = r * SIZE + c
        nb = []; nbr = []
        for nr, nc in [(r-1,c),(r-1,c+1),(r,c-1),(r,c+1),(r+1,c-1),(r+1,c)]:
            if 0 <= nr < SIZE and 0 <= nc < SIZE:
                nb.append((nr, nc)); nbr.append(nr * SIZE + nc)
        NB_RC[i] = nb; NB_IDX[i] = nbr
        CD[i] = math.hypot(r - 5.0, c - 5.0); EDG[0,i] = r; EDG[1,i] = SIZE-1-r
        EDG[2,i] = c; EDG[3,i] = SIZE-1-c


def _win_bfs(stones, ns):  # ns=True → Nord-Sud(Blue), ns=False → Ouest-Est(Red)
    if not stones.any(): return False
    vis = np.zeros((SIZE, SIZE), bool); q = []
    if ns:  # row 0
        for cc in range(SIZE):
            if stones[0, cc]: q.append((0, cc)); vis[0, cc] = True
    else:  # col 0
        for rr in range(SIZE):
            if stones[rr, 0]: q.append((rr, 0)); vis[rr, 0] = True
    head = 0
    while head < len(q):
        r, c = q[head]; head += 1
        if (ns and r == SIZE - 1) or (not ns and c == SIZE - 1): return True
        for nr, nc in NB_RC[r * SIZE + c]:
            if stones[nr, nc] and not vis[nr, nc]:
                vis[nr, nc] = True; q.append((nr, nc))
    return False


# ─── FastBoard pour rollouts ──────────────────────────────────────────────────
class _FB:
    __slots__ = ('b', 'r', 'btp', 'emp', 'last')
    def __init__(self):
        self.b = np.zeros((SIZE, SIZE), bool); self.r = np.zeros((SIZE, SIZE), bool)
        self.btp = True; self.emp = NC; self.last = None
    def from_env(self, env):
        np.copyto(self.b, env.blue); np.copyto(self.r, env.red)
        self.btp = env.blue_to_play; self.emp = NC - int(self.b.sum()) - int(self.r.sum())
        self.last = None
    def copy(self):
        f = _FB.__new__(_FB); f.b = self.b.copy(); f.r = self.r.copy()
        f.btp = self.btp; f.emp = self.emp; f.last = self.last; return f
    def apply(self, pos):
        r, c = divmod(pos, SIZE)
        if self.btp: self.b[r, c] = True
        else: self.r[r, c] = True
        self.btp = not self.btp; self.emp -= 1; self.last = (r, c)
    def winner(self):
        if _win_bfs(self.b, True): return 'blue'
        if _win_bfs(self.r, False): return 'red'
        return None
    def is_terminal(self):
        return _win_bfs(self.b, True) or _win_bfs(self.r, False)
    def get_legal_moves(self):
        occ = self.b | self.r; free = np.where(~occ.ravel())[0]; return free


# ─── Node MCTS ────────────────────────────────────────────────────────────────
class _N:
    __slots__ = ('mv', 'p', 'q', 'n', 'ch', 'ut', 'pl')
    def __init__(self, mv=None, p=None, pl=None):
        self.mv = mv; self.p = p; self.q = 0.0; self.n = 0
        self.ch = {}; self.ut = None; self.pl = pl


# ─── DeepSeekV4ProMax ─────────────────────────────────────────────────────────
class DeepSeekV4ProMax:
    def __init__(self):
        self.last_stats = {}

    def select_move(self, env, time_s=1.5):
        moves = env.get_legal_moves()
        if len(moves) == 0: return -1
        root_blue = env.blue_to_play

        # 1) Gagnant immédiat
        for m in moves:
            mv = int(m)
            env.apply_move(mv); w = env.winner(); env.undo_move(mv, root_blue)
            if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
                self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
                return mv

        # 2) Bloquer menace adverse unique
        opp_blue = not root_blue
        board = env.blue if opp_blue else env.red
        threats = []
        for m in moves:
            mv = int(m); r, c = divmod(mv, SIZE)
            board[r, c] = True
            if _win_bfs(board, opp_blue): threats.append(mv)
            board[r, c] = False
        if len(threats) == 1:
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
            return threats[0]

        # 3) MCTS
        return self._mcts(env, moves, root_blue, time_s)

    def _mcts(self, env, moves, root_blue, time_s):
        t0 = time.time()
        deadline = t0 + time_s * 0.90
        root = _N(pl='blue' if root_blue else 'red')
        root.ut = self._order(moves, env)
        stack = []; its = 0

        # FastBoard pour les rollouts (réutilisé)
        fb = _FB()

        while True:
            if its >= 300 and time.time() >= deadline: break
            if its >= 200000: break

            node = root
            while not node.ut and node.ch:
                node = self._ucb(node)
                wb = env.blue_to_play; env.apply_move(node.mv); stack.append((node.mv, wb))

            if env.is_terminal():
                res = env.winner(); leaf = node
            else:
                if node.ut:
                    mv = node.ut.pop(); wb = env.blue_to_play
                    ch = _N(mv=mv, p=node, pl='blue' if wb else 'red')
                    node.ch[mv] = ch; env.apply_move(mv); stack.append((mv, wb))
                    leaf = ch
                else:
                    leaf = node
                if env.is_terminal():
                    res = env.winner()
                else:
                    fb.from_env(env)
                    res = self._rollout_fast(fb)

            # Backprop
            bw = (res == 'blue'); cur = leaf
            while cur:
                cur.n += 1
                if cur is not root and cur.pl:
                    if (cur.pl == 'blue' and bw) or (cur.pl == 'red' and not bw):
                        cur.q += 1.0
                cur = cur.p

            # Undo
            for mv, wb in reversed(stack): env.undo_move(mv, wb)
            stack.clear(); its += 1

        elapsed = time.time() - t0
        if root.ch:
            bc = max(root.ch.values(), key=lambda c: c.n)
            best = bc.mv; wr = bc.q / max(bc.n, 1) if bc.n > 0 else 0.0; tv = sum(c.n for c in root.ch.values())
        else:
            best = int(moves[0]); wr = 0.0; tv = 0

        self.last_stats = {'iters': its, 'visits': tv, 'winrate': float(wr), 'time': elapsed}
        return best

    def _rollout_fast(self, fb):
        for _ in range(NC):
            if fb.is_terminal(): return fb.winner()
            moves = fb.get_legal_moves()
            if len(moves) == 0: return 'blue'

            # 75% adjacent au dernier coup, 25% aléatoire
            if fb.last is not None and random.random() < 0.75:
                r, c = fb.last; adj = []
                for nr, nc in NB_RC[r * SIZE + c]:
                    if not fb.b[nr, nc] and not fb.r[nr, nc]: adj.append(nr * SIZE + nc)
                if adj: mv = random.choice(adj)
                else: mv = int(random.choice(moves))
            else:
                mv = int(random.choice(moves))
            fb.apply(mv)
        return fb.winner() or 'blue'

    def _ucb(self, node):
        ln = math.log(max(node.n, 1)); best, bs = None, -1e9
        for c in node.ch.values():
            if c.n == 0: return c
            sc = c.q / c.n + C * math.sqrt(ln / c.n)
            if sc > bs: bs = sc; best = c
        return best

    def _order(self, moves, env):
        own = env.blue if env.blue_to_play else env.red
        opp = env.red if env.blue_to_play else env.blue
        btp = env.blue_to_play
        sc = []
        for m in moves:
            mv = int(m); s = (1.0 - CD[mv] / CD.max()) * 5.0
            for nr, nc in NB_RC[mv]:
                if own[nr, nc]: s += 12.0
                elif opp[nr, nc]: s += 6.0
            if btp:
                s += max(0.0, (SIZE - EDG[0, mv]) / SIZE) * 2.0
                s += max(0.0, (SIZE - EDG[1, mv]) / SIZE) * 2.0
            else:
                s += max(0.0, (SIZE - EDG[2, mv]) / SIZE) * 2.0
                s += max(0.0, (SIZE - EDG[3, mv]) / SIZE) * 2.0
            sc.append((s, mv))
        sc.sort(key=lambda x: -x[0])
        return [mv for _, mv in sc]


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python deepseek_v4_pro_max.py BOARD PLAYER [time_s]", file=sys.stderr)
        sys.exit(1)
    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = DeepSeekV4ProMax()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))
    s = _player.last_stats
    if s:
        print(f"ITERS:{s.get('iters',0)} VISITS:{s.get('visits',0)} "
              f"WINRATE:{s.get('winrate',0.0):.4f} TIME:{s.get('time',0.0):.3f}", file=sys.stderr)

# claude_opus_4_7_xhigh.py — Claude Opus 4.7 Xhigh
# IA Hex 11x11 : MCTS + UCB1 + rollouts heuristiques (bridge-save)
#                + détection coup gagnant / menace + union-find incrémental
# Interface CLI : python claude_opus_4_7_xhigh.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random

# ─── Bootstrap des imports train/ ─────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv


# ─── Constantes du plateau ───────────────────────────────────────────────────

MODEL_NAME = "Claude Opus 4.7 Xhigh"

BOARD = 11
N = BOARD * BOARD  # 121

# Noeuds virtuels d'union-find pour les bords
B_N = 121  # bord nord (Blue)
B_S = 122  # bord sud  (Blue)
R_W = 123  # bord ouest (Red)
R_E = 124  # bord est   (Red)
UF_SIZE = 125

# Voisins hexagonaux (6 directions)
_NEIGHBORS = []
for r in range(BOARD):
    for c in range(BOARD):
        nb = []
        for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD and 0 <= nc < BOARD:
                nb.append(nr * BOARD + nc)
        _NEIGHBORS.append(tuple(nb))

# Two-bridges : pour chaque cellule p, liste de (q, c1, c2)
# où (p,q) forment un pont virtuel à 2 cellules-porteurs c1, c2.
_BRIDGE_DELTAS = (
    (-2, +1, -1,  0, -1, +1),
    (-1, +2, -1, +1,  0, +1),
    (+1, +1,  0, +1, +1,  0),
    (+2, -1, +1,  0, +1, -1),
    (+1, -2,  0, -1, +1, -1),
    (-1, -1, -1,  0,  0, -1),
)
_BRIDGES = []
for r in range(BOARD):
    for c in range(BOARD):
        bs = []
        for (dr, dc, c1r, c1c, c2r, c2c) in _BRIDGE_DELTAS:
            pr, pc = r + dr, c + dc
            cr1, cc1 = r + c1r, c + c1c
            cr2, cc2 = r + c2r, c + c2c
            if (0 <= pr  < BOARD and 0 <= pc  < BOARD and
                0 <= cr1 < BOARD and 0 <= cc1 < BOARD and
                0 <= cr2 < BOARD and 0 <= cc2 < BOARD):
                bs.append((pr*BOARD + pc, cr1*BOARD + cc1, cr2*BOARD + cc2))
        _BRIDGES.append(tuple(bs))

# Reverse-map : pour chaque carrier c, liste de (p, q, other_carrier)
# tel que (p,q) est un pont avec c et other_carrier comme porteurs.
_CARRIER_OF = [[] for _ in range(N)]
for p in range(N):
    for q, c1, c2 in _BRIDGES[p]:
        if p < q:  # éviter les doublons (un pont par paire)
            _CARRIER_OF[c1].append((p, q, c2))
            _CARRIER_OF[c2].append((p, q, c1))
for i in range(N):
    _CARRIER_OF[i] = tuple(_CARRIER_OF[i])


# ─── État de jeu interne avec union-find incrémental ─────────────────────────

class State:
    """État Hex compact. Utilise des bytearray + union-find avec bords virtuels."""
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
        """+1 si Blue a connecté N-S, -1 si Red a connecté W-E, 0 sinon."""
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


# ─── Rollout heuristique avec bridge-save ────────────────────────────────────

def _rollout(state):
    """
    Joue un rollout aléatoire avec heuristique :
    - Si l'adversaire vient de prendre un porteur d'un pont entre 2 de mes pierres,
      je sauve en jouant l'autre porteur.
    - Sinon, choix uniforme parmi les cases vides.
    Retourne +1 (Blue gagne) ou -1 (Red gagne).
    """
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

        # swap-pop pour retirer chosen de empties en O(1)
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


# ─── MCTS ────────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ('children', 'untried', 'wins', 'visits', 'to_play', 'terminal_value')

    def __init__(self, untried, to_play, terminal_value=0):
        self.children = {}
        self.untried = untried
        self.wins = 0.0   # depuis POV du joueur qui a joué pour atteindre ce noeud
        self.visits = 0
        self.to_play = to_play         # qui joue à ce noeud
        self.terminal_value = terminal_value  # 0, +1 ou -1


_C_UCB = 1.4


def _ucb_select(node):
    log_n = math.log(node.visits)
    best_score = -1e18
    best_mv = -1
    best_ch = None
    c = _C_UCB
    for mv, ch in node.children.items():
        # Si l'enfant est terminal et gagnant pour le joueur du noeud courant,
        # on le sélectionne immédiatement.
        if ch.terminal_value != 0:
            mover_color = 1 if node.to_play else -1
            if ch.terminal_value == mover_color:
                return mv, ch
        score = (ch.wins / ch.visits) + c * math.sqrt(log_n / ch.visits)
        if score > best_score:
            best_score = score
            best_mv = mv
            best_ch = ch
    return best_mv, best_ch


# ─── Player ──────────────────────────────────────────────────────────────────

class ClaudeOpus47Xhigh:
    """
    Hex 11x11 — Claude Opus 4.7 Xhigh.
    MCTS UCB1 + rollouts heuristiques + détection tactique 1-coup.
    """

    NAME = MODEL_NAME

    def __init__(self, ucb_c: float = 1.4):
        global _C_UCB
        _C_UCB = ucb_c
        self.last_stats: dict = {}

    # ── API obligatoire ──────────────────────────────────────────────────────

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves_arr = env.get_legal_moves()
        n_legal = len(moves_arr)
        if n_legal == 0:
            return -1

        legal = [int(m) for m in moves_arr]

        # 1) Plateau vide : centre
        n_blue = int(env.blue.sum())
        n_red = int(env.red.sum())
        if n_blue + n_red == 0:
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 0.5, 'time': 0.0}
            return 60  # (5,5)

        # 2) État interne
        state = State.from_env(env)
        cur_is_blue = state.btp
        cur_color = 1 if cur_is_blue else -1
        opp_color = -cur_color

        # 3) Coup gagnant immédiat
        for m in legal:
            s2 = state.copy()
            s2.play(m)
            if s2.winner() == cur_color:
                self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
                return m

        # 4) Détection des menaces : si l'adversaire peut gagner en un coup,
        #    on prépare la liste pour bloquer en priorité.
        threats = []
        for m in legal:
            s2 = state.copy()
            s2.btp = not s2.btp  # tour adverse
            s2.play(m)
            if s2.winner() == opp_color:
                threats.append(m)
        if len(threats) == 1:
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 0.4, 'time': 0.0}
            return threats[0]
        # Si 2+ menaces : on est probablement perdu, on lance le MCTS quand même.

        # 5) MCTS
        t0 = time.time()
        deadline = t0 + max(0.05, time_s - 0.05)

        # Ordre d'expansion : menaces > centralité (réduit la variance MCTS au début)
        center = (BOARD - 1) // 2
        threat_set = set(threats)
        ordered = sorted(
            legal,
            key=lambda m: (
                0 if m in threat_set else 1,
                abs(m // BOARD - center) + abs(m % BOARD - center)
            )
        )

        root = _Node(untried=list(ordered), to_play=cur_is_blue)

        iters = 0
        check_every = 64
        while True:
            if iters % check_every == 0 and time.time() >= deadline:
                break
            iters += 1
            self._iterate(root, state)

        # 6) Sélection finale : enfant le plus visité (robuste)
        if not root.children:
            return ordered[0]
        best_mv, best_ch = max(root.children.items(),
                               key=lambda kv: (kv[1].visits, kv[1].wins))
        winrate = best_ch.wins / best_ch.visits if best_ch.visits else 0.5
        elapsed = time.time() - t0
        total_visits = sum(c.visits for c in root.children.values())

        self.last_stats = {
            'iters':   iters,
            'visits':  total_visits,
            'winrate': float(winrate),
            'time':    elapsed,
        }
        return best_mv

    # ── Itération MCTS ───────────────────────────────────────────────────────

    def _iterate(self, root, root_state):
        state = root_state.copy()
        node = root
        path = [node]

        # Sélection : descendre par UCB tant qu'il n'y a rien à étendre
        while (node.terminal_value == 0
               and not node.untried
               and node.children):
            mv, ch = _ucb_select(node)
            state.play(mv)
            node = ch
            path.append(node)

        # Si terminal : backprop direct
        if node.terminal_value != 0:
            self._backprop(path, node.terminal_value)
            return

        # Expansion
        if node.untried:
            mv = node.untried.pop()
            state.play(mv)
            w = state.winner()
            if w != 0:
                child = _Node(untried=[], to_play=state.btp, terminal_value=w)
            else:
                child = _Node(untried=state.legal_moves(), to_play=state.btp)
            node.children[mv] = child
            path.append(child)
            if w != 0:
                self._backprop(path, w)
                return
            node = child

        # Simulation
        winner_color = _rollout(state)
        self._backprop(path, winner_color)

    @staticmethod
    def _backprop(path, winner_color):
        # winner_color = +1 (Blue) ou -1 (Red)
        # path[i].wins est dans la POV du joueur qui a joué le coup pour y arriver
        # = path[i-1].to_play
        for i, node in enumerate(path):
            node.visits += 1
            if i == 0:
                continue
            mover_color = 1 if path[i - 1].to_play else -1
            if mover_color == winner_color:
                node.wins += 1.0


# ─── CLI (protocole BOARD/PLAYER) ─────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python claude_opus_4_7_xhigh.py BOARD PLAYER [time_s]",
              file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = ClaudeOpus47Xhigh()
    _move = _player.select_move(_env, _time_s)

    s = _player.last_stats
    if s and 'winrate' in s:
        print(
            f"ITERS:{s['iters']} VISITS:{s['visits']} "
            f"WINRATE:{s['winrate']:.4f} TIME:{s['time']:.3f}",
            file=sys.stderr,
        )
    print(_env.pos_to_str(_move))

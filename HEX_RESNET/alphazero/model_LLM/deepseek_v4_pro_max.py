#!/usr/bin/env python3
# deepseek_v4_pro_max.py — DeepSeek V4 Pro MAX MCTS Hex 11×11
# v2: RAVE/AMAF, prior heuristique Numba, rollouts Numba, détection de ponts,
#     filtrage cases mortes, tree reuse, PUCT, shortest-path early termination.
# CLI : python deepseek_v4_pro_max.py BOARD PLAYER [time_s]

import sys, os, time, math, random
import numpy as np
import numba

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from hex_env import HexEnv
from alphabeta import _shortest_path_jit, _NB_DR, _NB_DC

SIZE = 11; NC = SIZE * SIZE

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
        CD[i] = math.hypot(r - 5.0, c - 5.0)
        EDG[0,i] = r; EDG[1,i] = SIZE-1-r
        EDG[2,i] = c; EDG[3,i] = SIZE-1-c

# ─── Patterns de pont (6 directions, Numba-compatible) ────────────────────────
_BRIDGE_ARR = np.array([
    [-2, +1, -1,  0, -1, +1],
    [-1, +2, -1, +1,  0, +1],
    [+1, +1,  0, +1, +1,  0],
    [+2, -1, +1, -1, +1,  0],
    [+1, -2,  0, -1, +1, -1],
    [-1, -1, -1,  0,  0, -1],
], dtype=np.int32)


# ─── Win BFS JIT ──────────────────────────────────────────────────────────────
@numba.njit(cache=True)
def _win_bfs_jit(stones_flat, ns, N, nb_dr, nb_dc):
    """ns=True → Blue (ligne 0→N-1), ns=False → Red (col 0→N-1)"""
    NN = N * N
    vis = np.zeros(NN, dtype=np.bool_)
    q = np.empty(NN, dtype=np.int32)
    head = 0; tail = 0
    if ns:
        for cc in range(N):
            idx = cc
            if stones_flat[idx]:
                vis[idx] = True; q[tail] = idx; tail += 1
    else:
        for rr in range(N):
            idx = rr * N
            if stones_flat[idx]:
                vis[idx] = True; q[tail] = idx; tail += 1
    while head < tail:
        cur = q[head]; head += 1
        r = cur // N; c = cur - r * N
        if (ns and r == N - 1) or (not ns and c == N - 1):
            return True
        for k in range(6):
            nr = r + nb_dr[k]; nc = c + nb_dc[k]
            if nr < 0 or nr >= N or nc < 0 or nc >= N:
                continue
            nidx = nr * N + nc
            if stones_flat[nidx] and not vis[nidx]:
                vis[nidx] = True; q[tail] = nidx; tail += 1
    return False


# ─── Prior heuristique JIT ────────────────────────────────────────────────────
@numba.njit(cache=True)
def _compute_prior_jit(blue_flat, red_flat, blue_to_play, N, nb_dr, nb_dc, bridges):
    """
    Calcule une distribution de prior softmax (T=2) sur toutes les cases.
    Features : centre, voisins amis, bord propre, cases de pont.
    """
    NN = N * N
    center = N // 2
    scores = np.zeros(NN, dtype=np.float64)

    # Détection des cases de pont (intermédiaires libres de ponts amis vivants)
    bridge_bonus = np.zeros(NN, dtype=np.bool_)
    for a_idx in range(NN):
        if blue_to_play:
            if not blue_flat[a_idx]: continue
        else:
            if not red_flat[a_idx]: continue
        r = a_idx // N; c = a_idx - r * N
        for k in range(6):
            tdr, tdc = bridges[k,0], bridges[k,1]
            i1r, i1c = bridges[k,2], bridges[k,3]
            i2r, i2c = bridges[k,4], bridges[k,5]
            nr = r + tdr; nc = c + tdc
            if not (0 <= nr < N and 0 <= nc < N): continue
            b_idx = nr * N + nc
            if blue_to_play:
                if not blue_flat[b_idx]: continue
            else:
                if not red_flat[b_idx]: continue
            mr1 = r + i1r; mc1 = c + i1c
            mr2 = r + i2r; mc2 = c + i2c
            if not (0 <= mr1 < N and 0 <= mc1 < N): continue
            if not (0 <= mr2 < N and 0 <= mc2 < N): continue
            m1idx = mr1 * N + mc1
            m2idx = mr2 * N + mc2
            if blue_to_play:
                opp1 = red_flat[m1idx]; opp2 = red_flat[m2idx]
                own1 = blue_flat[m1idx]; own2 = blue_flat[m2idx]
            else:
                opp1 = blue_flat[m1idx]; opp2 = blue_flat[m2idx]
                own1 = red_flat[m1idx]; own2 = red_flat[m2idx]
            if opp1 or opp2: continue
            if not own1: bridge_bonus[m1idx] = True
            if not own2: bridge_bonus[m2idx] = True

    for idx in range(NN):
        if blue_flat[idx] or red_flat[idx]:
            scores[idx] = -1e18
            continue
        r = idx // N; c = idx - r * N
        s = 0.1  # score de base

        # Centre (distance Manhattan)
        dist_c = abs(r - center) + abs(c - center)
        if dist_c < 6:
            s += (6 - dist_c) * 0.5

        # Voisins amis
        for k in range(6):
            nr = r + nb_dr[k]; nc = c + nb_dc[k]
            if 0 <= nr < N and 0 <= nc < N:
                nidx = nr * N + nc
                if blue_to_play:
                    if blue_flat[nidx]: s += 1.5
                else:
                    if red_flat[nidx]: s += 1.5

        # Bord propre (Blue: haut/bas, Red: gauche/droite)
        if blue_to_play:
            if r == 0 or r == N - 1: s += 1.0
        else:
            if c == 0 or c == N - 1: s += 1.0

        # Case de pont
        if bridge_bonus[idx]:
            s += 3.0

        scores[idx] = s

    # Softmax T=2
    max_s = -1e18
    for idx in range(NN):
        if blue_flat[idx] or red_flat[idx]: continue
        if scores[idx] > max_s: max_s = scores[idx]

    prior = np.zeros(NN, dtype=np.float32)
    total = 0.0
    for idx in range(NN):
        if blue_flat[idx] or red_flat[idx]:
            prior[idx] = 0.0; continue
        v = math.exp((scores[idx] - max_s) / 2.0)
        prior[idx] = v; total += v

    if total > 0.0:
        inv_t = 1.0 / total
        for idx in range(NN):
            if prior[idx] > 0.0: prior[idx] *= inv_t
    else:
        n_legal = 0
        for idx in range(NN):
            if not (blue_flat[idx] or red_flat[idx]): n_legal += 1
        if n_legal > 0:
            inv = 1.0 / n_legal
            for idx in range(NN):
                if not (blue_flat[idx] or red_flat[idx]): prior[idx] = inv
    return prior


# ─── Détection cases mortes JIT ───────────────────────────────────────────────
@numba.njit(cache=True)
def _dead_cells_jit(blue_flat, red_flat, N, nb_dr, nb_dc):
    """
    Case = complètement entourée d'une seule couleur ET ≥4 voisins → morte.
    """
    NN = N * N
    dead = np.zeros(NN, dtype=np.bool_)
    for idx in range(NN):
        if blue_flat[idx] or red_flat[idx]: continue
        r = idx // N; c = idx - r * N
        blue_n = 0; red_n = 0; empty_n = 0; in_board = 0
        for k in range(6):
            nr = r + nb_dr[k]; nc = c + nb_dc[k]
            if 0 <= nr < N and 0 <= nc < N:
                in_board += 1
                nidx = nr * N + nc
                if blue_flat[nidx]: blue_n += 1
                elif red_flat[nidx]: red_n += 1
                else: empty_n += 1
        if in_board >= 4 and empty_n == 0:
            if blue_n > 0 and red_n == 0: dead[idx] = True
            elif red_n > 0 and blue_n == 0: dead[idx] = True
    return dead


# ─── Rollout Numba JIT (avec early termination) ──────────────────────────────
@numba.njit(cache=True)
def _rollout_jit(blue_flat, red_flat, empty_arr, n_empty, blue_to_play,
                 moves_out, colors_out, N, nb_dr, nb_dc, bridges, p_complete):
    """
    Rollout rapide : bridge-complete (p) → adjacent au dernier coup → aléatoire.
    Early termination via _shortest_path_jit (d=0 → victoire).
    Retourne (winner_int, n_played) avec winner_int 0=blue, 1=red, -1=draw/nul.
    """
    pos_to_idx = np.full(N * N, -1, dtype=np.int32)
    for i in range(n_empty):
        pos_to_idx[empty_arr[i]] = i
    remaining = n_empty
    n_played = 0
    last_move = -1
    cur_blue = blue_to_play

    while remaining > 0:
        move = -1

        # 1) Complétion de pont (prob p_complete)
        if move < 0 and p_complete > 0.0 and np.random.random() < p_complete:
            start_offset = np.random.randint(N * N)
            for step in range(N * N):
                own_idx = (start_offset + step) % (N * N)
                if cur_blue:
                    own_here = blue_flat[own_idx]
                else:
                    own_here = red_flat[own_idx]
                if not own_here: continue
                r = own_idx // N; c = own_idx - r * N
                found = False
                for k in range(6):
                    tdr, tdc = bridges[k,0], bridges[k,1]
                    i1r, i1c = bridges[k,2], bridges[k,3]
                    i2r, i2c = bridges[k,4], bridges[k,5]
                    br = r + tdr; bc = c + tdc
                    if not (0 <= br < N and 0 <= bc < N): continue
                    bidx = br * N + bc
                    if blue_flat[bidx] or red_flat[bidx]: continue
                    mr1 = r + i1r; mc1 = c + i1c
                    mr2 = r + i2r; mc2 = c + i2c
                    if not (0 <= mr1 < N and 0 <= mc1 < N): continue
                    if not (0 <= mr2 < N and 0 <= mc2 < N): continue
                    if cur_blue:
                        opp1 = red_flat[mr1*N+mc1]; opp2 = red_flat[mr2*N+mc2]
                    else:
                        opp1 = blue_flat[mr1*N+mc1]; opp2 = blue_flat[mr2*N+mc2]
                    if opp1 or opp2: continue
                    if pos_to_idx[bidx] >= 0:
                        move = bidx; found = True; break
                if found: break

        # 2) Adjacent au dernier coup (prob 0.7)
        if move < 0 and last_move >= 0 and np.random.random() < 0.7:
            r = last_move // N; c = last_move % N
            adj = np.empty(6, dtype=np.int32); na = 0
            for k in range(6):
                nr = r + nb_dr[k]; nc = c + nb_dc[k]
                if 0 <= nr < N and 0 <= nc < N:
                    nidx = nr * N + nc
                    if not blue_flat[nidx] and not red_flat[nidx]:
                        if pos_to_idx[nidx] >= 0:
                            adj[na] = nidx; na += 1
            if na > 0: move = adj[np.random.randint(na)]

        # 3) Aléatoire uniforme
        if move < 0:
            idx = np.random.randint(remaining)
            move = empty_arr[idx]

        # Swap-pop O(1)
        idx = pos_to_idx[move]
        last_pos = empty_arr[remaining - 1]
        empty_arr[idx] = last_pos
        pos_to_idx[last_pos] = idx
        pos_to_idx[move] = -1
        remaining -= 1

        # Appliquer le coup
        if cur_blue: blue_flat[move] = True
        else: red_flat[move] = True
        moves_out[n_played] = move
        colors_out[n_played] = 1 if cur_blue else 0
        n_played += 1
        last_move = move

        # Early termination : vérifier si un joueur a gagné
        if n_played >= N:
            if cur_blue:
                d = _shortest_path_jit(blue_flat, red_flat, True, N, nb_dr, nb_dc)
                if d == 0: return 0, n_played
            else:
                d = _shortest_path_jit(red_flat, blue_flat, False, N, nb_dr, nb_dc)
                if d == 0: return 1, n_played

        cur_blue = not cur_blue

    return -1, n_played


# ─── Nœud MCTS avec RAVE ──────────────────────────────────────────────────────
class _N:
    """Nœud MCTS avec RAVE/AMAF et prior heuristique."""
    __slots__ = ('mv', 'p', 'q', 'n', 'ch', 'ut', 'pl', 'pr', 'av', 'aw', 'env')

    def __init__(self, mv=None, p=None, pl=None, prior=None, env=None):
        self.mv = mv          # coup qui a mené à ce nœud
        self.p = p            # nœud parent
        self.q = 0.0          # somme des victoires (du pdv de ce nœud)
        self.n = 0            # visites
        self.ch = {}          # enfants {move: _N}
        self.ut = []          # coups non essayés (triés par prior décroissant)
        self.pl = pl          # 'blue' ou 'red' (joueur qui joue depuis ce nœud)
        self.pr = prior       # prior np.float32[NC] ou None
        self.av = np.zeros(NC, dtype=np.int32)   # AMAF visites
        self.aw = np.zeros(NC, dtype=np.float32)  # AMAF wins
        self.env = env        # HexEnv (stocké uniquement pour tree reuse + terminal)


# ─── DeepSeekV4ProMax ─────────────────────────────────────────────────────────
class DeepSeekV4ProMax:
    """
    MCTS + RAVE/AMAF + prior heuristique + rollouts Numba + tree reuse.
    Interface : select_move(env, time_s) -> int
    """

    def __init__(self, c_puct=1.2, rave_k=300.0, c_bias=0.0,
                 p_complete=0.5, min_sims=256):
        self.c_puct = c_puct
        self.rave_k = rave_k
        self.c_bias = c_bias
        self.p_complete = p_complete
        self.min_sims = min_sims
        self.last_stats: dict = {}
        self._root: _N | None = None

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _compute_prior(self, env):
        return _compute_prior_jit(
            env.blue.ravel(), env.red.ravel(),
            env.blue_to_play, SIZE, _NB_DR, _NB_DC, _BRIDGE_ARR,
        )

    def _create_node(self, env, parent=None, mv=None):
        prior = self._compute_prior(env)
        blue_flat = env.blue.ravel()
        red_flat = env.red.ravel()
        dead = _dead_cells_jit(blue_flat, red_flat, SIZE, _NB_DR, _NB_DC)
        legal = env.get_legal_moves()
        filtered = [int(m) for m in legal if not dead[m]]
        if not filtered and len(legal) > 0:
            filtered = [int(m) for m in legal]
        filtered.sort(key=lambda m: -prior[m])
        node = _N(mv=mv, p=parent,
                  pl='blue' if env.blue_to_play else 'red',
                  prior=prior, env=env)
        node.ut = filtered
        return node

    def _try_reuse_tree(self, env):
        """
        Si l'état actuel correspond à un enfant de l'ancienne racine
        (l'adversaire a joué exactement 1 coup qu'on avait exploré),
        on réutilise le sous-arbre.
        """
        if self._root is None or self._root.env is None:
            return None
        stored = self._root.env
        # Pierres retirées → partie différente
        if (stored.blue & ~env.blue).any() or (stored.red & ~env.red).any():
            return None
        blue_added = env.blue & ~stored.blue
        red_added = env.red & ~stored.red
        n_added = int(blue_added.sum()) + int(red_added.sum())

        if n_added == 0:
            if stored.blue_to_play == env.blue_to_play:
                return self._root
            return None

        if n_added != 1:
            return None

        if blue_added.any():
            rr, cc = np.argwhere(blue_added)[0]
            is_blue_move = True
        else:
            rr, cc = np.argwhere(red_added)[0]
            is_blue_move = False
        m = int(rr) * SIZE + int(cc)

        if is_blue_move != stored.blue_to_play:
            return None

        child = self._root.ch.get(m)
        if child is None or child.env is None:
            return None
        if child.env.blue_to_play != env.blue_to_play:
            return None
        child.p = None  # détacher du parent
        return child

    # ─── Sélection PUCT + RAVE ─────────────────────────────────────────────

    def _select(self, node):
        """Sélectionne le meilleur enfant avec PUCT + RAVE blending + progressive bias."""
        log_n = math.log(max(node.n, 1))
        sqrt_n = math.sqrt(max(node.n, 1))
        beta_den = 3.0 * node.n + self.rave_k
        beta = math.sqrt(self.rave_k / beta_den) if beta_den > 0 else 0.0

        best_c = None
        best_s = -1e18

        for c in node.ch.values():
            if c.n == 0:
                return c

            # Q du point de vue du parent (c.pl est le joueur de l'enfant)
            q = 1.0 - (c.q / c.n)  # du pdv du joueur qui a joué pour arriver à c

            # RAVE blending
            amaf_n = node.av[c.mv] if c.mv is not None and c.mv >= 0 else 0
            if amaf_n > 0:
                amaf_val = node.aw[c.mv] / amaf_n
            else:
                amaf_val = 0.5

            combined = (1.0 - beta) * q + beta * amaf_val

            # PUCT exploration
            puct = 0.0
            if node.pr is not None and c.mv is not None and c.mv >= 0:
                puct = self.c_puct * float(node.pr[c.mv]) * sqrt_n / (1.0 + c.n)

            # Progressive bias
            bias = 0.0
            if self.c_bias > 0 and node.pr is not None and c.mv is not None and c.mv >= 0:
                bias = self.c_bias * float(node.pr[c.mv]) * sqrt_n / (1.0 + c.n)

            sc = combined + puct + bias
            if sc > best_s:
                best_s = sc
                best_c = c

        return best_c

    # ─── Simulation MCTS ───────────────────────────────────────────────────

    def _simulate_once(self, root):
        node = root

        # 1) Sélection : descendre tant que tous les enfants sont connus
        while node.ut is not None and len(node.ut) == 0 and node.ch:
            nxt = self._select(node)
            if nxt is None:
                break
            node = nxt

        # 2) Si terminal
        if node.env and node.env.is_terminal():
            w = node.env.winner()
            self._backprop(node, 0 if w == 'blue' else 1, None, 0, None)
            return

        # 3) Expansion
        if node.ut:
            mv = node.ut.pop(0)
            child_env = node.env.copy()
            child_env.apply_move(mv)
            child = self._create_node(child_env, parent=node, mv=mv)
            node.ch[mv] = child
            node = child

        # 4) Terminal après expansion
        if node.env and node.env.is_terminal():
            w = node.env.winner()
            self._backprop(node, 0 if w == 'blue' else 1, None, 0, None)
            return

        # 5) Rollout Numba
        blue_flat = node.env.blue.ravel().copy()
        red_flat = node.env.red.ravel().copy()
        empty_arr = np.where(~(blue_flat | red_flat))[0].astype(np.int32)
        n_empty = empty_arr.shape[0]
        moves_arr = np.zeros(n_empty, dtype=np.int32)
        colors_arr = np.zeros(n_empty, dtype=np.int8)

        winner_int, n_played = _rollout_jit(
            blue_flat, red_flat, empty_arr, n_empty,
            node.env.blue_to_play,
            moves_arr, colors_arr,
            SIZE, _NB_DR, _NB_DC, _BRIDGE_ARR,
            self.p_complete,
        )

        played_moves = moves_arr[:n_played] if n_played > 0 else None
        played_colors = colors_arr[:n_played] if n_played > 0 else None

        self._backprop(node, winner_int, played_moves, n_played, played_colors)

    def _backprop(self, node, winner_int, played_moves, n_played, played_colors=None):
        """
        Backpropagation : Q + RAVE/AMAF vectorisé.
        winner_int: 0=blue, 1=red, -1=draw
        """
        if winner_int == -1:  # draw (ne devrait pas arriver en Hex)
            cur = node
            while cur is not None:
                cur.n += 1
                cur = cur.p
            return

        is_blue_win = (winner_int == 0)

        cur = node
        while cur is not None:
            cur.n += 1
            if cur.pl == 'blue':
                win_val = 1.0 if is_blue_win else 0.0
            elif cur.pl == 'red':
                win_val = 0.0 if is_blue_win else 1.0
            else:
                win_val = 0.0
            cur.q += win_val

            # RAVE/AMAF : les coups joués par la couleur de cur.pl dans le rollout
            if n_played > 0 and played_moves is not None and played_colors is not None:
                node_blue_int = 1 if cur.pl == 'blue' else 0
                mask = played_colors == node_blue_int
                matched = played_moves[mask]
                cur.av[matched] += 1
                if win_val != 0.0:
                    cur.aw[matched] += win_val

            cur = cur.p

    # ─── Point d'entrée principal ──────────────────────────────────────────

    def select_move(self, env, time_s=1.5):
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play

        # 1) Coup gagnant immédiat
        for m in moves:
            mv = int(m)
            env.apply_move(mv); w = env.winner(); env.undo_move(mv, root_blue)
            if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
                self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
                self._root = None
                return mv

        # 2) Bloquer menace adverse unique
        opp_blue = not root_blue
        board = env.blue if opp_blue else env.red
        threats = []
        for m in moves:
            mv = int(m); r, c = divmod(mv, SIZE)
            board[r, c] = True
            if _win_bfs_jit(board.ravel(), opp_blue, SIZE, _NB_DR, _NB_DC):
                threats.append(mv)
                if len(threats) > 1:
                    break
            board[r, c] = False
        if len(threats) == 1:
            self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 0.5, 'time': 0.0}
            self._root = None
            return threats[0]

        # 3) Tree reuse ou nouvelle racine
        reused = self._try_reuse_tree(env)
        if reused is not None:
            root = reused
        else:
            root = self._create_node(env.copy())

        # 4) Boucle MCTS
        t0 = time.time()
        deadline = t0 + max(time_s, 0.01) * 0.95
        sims = 0
        while sims < self.min_sims or time.time() < deadline:
            self._simulate_once(root)
            sims += 1

        # 5) Meilleur coup par visites
        if root.ch:
            bc = max(root.ch.values(), key=lambda c: c.n)
            best = bc.mv
            wr = 1.0 - (bc.q / max(bc.n, 1)) if bc.n > 0 else 0.0
            tv = root.n
        else:
            best = int(moves[0])
            wr = 0.0
            tv = 0

        # 6) Tree reuse : descendre dans le coup choisi pour le prochain appel
        chosen = root.ch.get(best)
        if chosen is not None and chosen.env is not None:
            chosen.p = None
            self._root = chosen
        else:
            self._root = None

        elapsed = time.time() - t0
        self.last_stats = {'iters': sims, 'visits': tv,
                           'winrate': float(wr), 'time': elapsed}
        return best


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

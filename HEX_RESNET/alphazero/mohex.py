# mohex.py — MoHex : MCTS + RAVE/AMAF + rollouts Numba pour Hex 11×11
# Améliorations v2 : tree reuse, blocage forcé, prior heuristique (progressive bias),
#                   bridge-completion, fill-in dead cells, patterns étendus.
# Inspiré de MoHex (Université d'Alberta).
# Interface CLI : python mohex.py BOARD PLAYER [time_s]

import sys
import os
import math
import time
import random
import numpy as np
import numba

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from config import BOARD_SIZE, NUM_CELLS
from alphabeta import _shortest_path_jit, _NB_DR, _NB_DC

BS = BOARD_SIZE
NN = NUM_CELLS

# 6 patterns de pont : (target_dr, target_dc, i1_dr, i1_dc, i2_dr, i2_dc)
_BRIDGE_ARR = np.array([
    [-2, +1, -1,  0, -1, +1],
    [-1, +2, -1, +1,  0, +1],
    [+1, +1,  0, +1, +1,  0],
    [+2, -1, +1, -1, +1,  0],
    [+1, -2,  0, -1, +1, -1],
    [-1, -1, -1,  0,  0, -1],
], dtype=np.int32)


# ─── Sauvegarde de pont (JIT) ─────────────────────────────────────────────────

@numba.njit(cache=True)
def _bridge_save_jit(blue_flat, red_flat, cur_blue, last_move, N, bridges):
    """
    Si last_move (joué par l'adversaire) menace un pont du joueur courant,
    retourne l'index de la case de sauvegarde, sinon -1.
    """
    r0 = last_move // N
    c0 = last_move - r0 * N
    for k in range(6):
        tdr = bridges[k, 0]
        tdc = bridges[k, 1]
        i1r = bridges[k, 2]
        i1c = bridges[k, 3]
        i2r = bridges[k, 4]
        i2c = bridges[k, 5]

        # Cas 1 : last_move = intermédiaire i1
        ar = r0 - i1r
        ac = c0 - i1c
        br = ar + tdr
        bc = ac + tdc
        if 0 <= ar < N and 0 <= ac < N and 0 <= br < N and 0 <= bc < N:
            aidx = ar * N + ac
            bidx = br * N + bc
            if cur_blue:
                own_a = blue_flat[aidx]
                own_b = blue_flat[bidx]
            else:
                own_a = red_flat[aidx]
                own_b = red_flat[bidx]
            if own_a and own_b:
                sr = ar + i2r
                sc = ac + i2c
                if 0 <= sr < N and 0 <= sc < N:
                    sidx = sr * N + sc
                    if not blue_flat[sidx] and not red_flat[sidx]:
                        return sidx

        # Cas 2 : last_move = intermédiaire i2
        ar = r0 - i2r
        ac = c0 - i2c
        br = ar + tdr
        bc = ac + tdc
        if 0 <= ar < N and 0 <= ac < N and 0 <= br < N and 0 <= bc < N:
            aidx = ar * N + ac
            bidx = br * N + bc
            if cur_blue:
                own_a = blue_flat[aidx]
                own_b = blue_flat[bidx]
            else:
                own_a = red_flat[aidx]
                own_b = red_flat[bidx]
            if own_a and own_b:
                sr = ar + i1r
                sc = ac + i1c
                if 0 <= sr < N and 0 <= sc < N:
                    sidx = sr * N + sc
                    if not blue_flat[sidx] and not red_flat[sidx]:
                        return sidx
    return -1


# ─── Complétion de pont (JIT) ────────────────────────────────────────────────

@numba.njit(cache=True)
def _bridge_complete_jit(blue_flat, red_flat, cur_blue, N, bridges, start_offset):
    """
    Cherche une case vide qui, posée, crée un pont avec une pierre amie existante.
    start_offset ∈ [0, N*N) sert à désynchroniser l'ordre de scan entre appels.
    Retourne l'index de la case, ou -1 si aucune.
    """
    NN_ = N * N
    for step in range(NN_):
        own_idx = (start_offset + step) % NN_
        if cur_blue:
            own_here = blue_flat[own_idx]
        else:
            own_here = red_flat[own_idx]
        if not own_here:
            continue
        r = own_idx // N
        c = own_idx - r * N
        for k in range(6):
            tdr = bridges[k, 0]
            tdc = bridges[k, 1]
            i1r = bridges[k, 2]
            i1c = bridges[k, 3]
            i2r = bridges[k, 4]
            i2c = bridges[k, 5]
            br = r + tdr
            bc = c + tdc
            if not (0 <= br < N and 0 <= bc < N):
                continue
            bidx = br * N + bc
            # bidx doit être vide
            if blue_flat[bidx] or red_flat[bidx]:
                continue
            # Les 2 intermédiaires ne doivent pas être adverses
            mr1 = r + i1r
            mc1 = c + i1c
            mr2 = r + i2r
            mc2 = c + i2c
            if not (0 <= mr1 < N and 0 <= mc1 < N):
                continue
            if not (0 <= mr2 < N and 0 <= mc2 < N):
                continue
            m1idx = mr1 * N + mc1
            m2idx = mr2 * N + mc2
            if cur_blue:
                opp1 = red_flat[m1idx]
                opp2 = red_flat[m2idx]
            else:
                opp1 = blue_flat[m1idx]
                opp2 = blue_flat[m2idx]
            if opp1 or opp2:
                continue
            return bidx
    return -1


# ─── Prior heuristique (JIT) ─────────────────────────────────────────────────

@numba.njit(cache=True)
def _compute_prior_jit(blue_flat, red_flat, blue_to_play, N, nb_dr, nb_dc, bridges):
    """
    Prior par patterns (centre, voisins amis, bord propre, case de pont).
    Retourne un float32[N*N] normalisé par softmax (T=2), zero pour cases occupées.
    """
    NN_ = N * N
    center = N // 2
    scores = np.zeros(NN_, dtype=np.float64)

    # Bridge cells : intermédiaires libres de ponts amis vivants
    bridge_bonus = np.zeros(NN_, dtype=np.bool_)
    for a_idx in range(NN_):
        if blue_to_play:
            if not blue_flat[a_idx]:
                continue
        else:
            if not red_flat[a_idx]:
                continue
        r = a_idx // N
        c = a_idx - r * N
        for k in range(6):
            tdr = bridges[k, 0]; tdc = bridges[k, 1]
            i1r = bridges[k, 2]; i1c = bridges[k, 3]
            i2r = bridges[k, 4]; i2c = bridges[k, 5]
            nr = r + tdr; nc = c + tdc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            b_idx = nr * N + nc
            if blue_to_play:
                own_b = blue_flat[b_idx]
            else:
                own_b = red_flat[b_idx]
            if not own_b:
                continue
            mr1 = r + i1r; mc1 = c + i1c
            mr2 = r + i2r; mc2 = c + i2c
            if not (0 <= mr1 < N and 0 <= mc1 < N):
                continue
            if not (0 <= mr2 < N and 0 <= mc2 < N):
                continue
            m1idx = mr1 * N + mc1
            m2idx = mr2 * N + mc2
            if blue_to_play:
                opp1 = red_flat[m1idx]; opp2 = red_flat[m2idx]
                own1 = blue_flat[m1idx]; own2 = blue_flat[m2idx]
            else:
                opp1 = blue_flat[m1idx]; opp2 = blue_flat[m2idx]
                own1 = red_flat[m1idx]; own2 = red_flat[m2idx]
            if opp1 or opp2:
                continue
            if not own1:
                bridge_bonus[m1idx] = True
            if not own2:
                bridge_bonus[m2idx] = True

    # Scores par case
    for idx in range(NN_):
        if blue_flat[idx] or red_flat[idx]:
            scores[idx] = -1e18
            continue
        r = idx // N
        c = idx - r * N
        s = 0.1

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
                    if blue_flat[nidx]:
                        s += 1.5
                else:
                    if red_flat[nidx]:
                        s += 1.5

        # Bord propre
        if blue_to_play:
            if r == 0 or r == N - 1:
                s += 1.0
        else:
            if c == 0 or c == N - 1:
                s += 1.0

        # Case de pont
        if bridge_bonus[idx]:
            s += 3.0

        scores[idx] = s

    # Softmax T=2 sur les cases légales
    max_s = -1e18
    for idx in range(NN_):
        if scores[idx] > max_s:
            max_s = scores[idx]

    prior = np.zeros(NN_, dtype=np.float32)
    total = 0.0
    for idx in range(NN_):
        if blue_flat[idx] or red_flat[idx]:
            prior[idx] = 0.0
            continue
        v = math.exp((scores[idx] - max_s) / 2.0)
        prior[idx] = v
        total += v

    if total > 0.0:
        for idx in range(NN_):
            prior[idx] /= total
    else:
        # fallback : uniforme sur les légales
        n_legal = 0
        for idx in range(NN_):
            if not (blue_flat[idx] or red_flat[idx]):
                n_legal += 1
        if n_legal > 0:
            inv = 1.0 / n_legal
            for idx in range(NN_):
                if not (blue_flat[idx] or red_flat[idx]):
                    prior[idx] = inv
    return prior


# ─── Détection de cases mortes (JIT, règle des 6 voisins) ────────────────────

@numba.njit(cache=True)
def _dead_cells_6n_jit(blue_flat, red_flat, N, nb_dr, nb_dc):
    """
    Case 'morte' = case vide dont tous les voisins in-board sont de la même
    couleur et il y a ≥ 4 voisins in-board. Retire les cases dominées par une
    couleur (fill-in trivial).
    """
    NN_ = N * N
    dead = np.zeros(NN_, dtype=np.bool_)
    for idx in range(NN_):
        if blue_flat[idx] or red_flat[idx]:
            continue
        r = idx // N
        c = idx - r * N
        blue_n = 0
        red_n = 0
        empty_n = 0
        in_board = 0
        for k in range(6):
            nr = r + nb_dr[k]; nc = c + nb_dc[k]
            if 0 <= nr < N and 0 <= nc < N:
                in_board += 1
                nidx = nr * N + nc
                if blue_flat[nidx]:
                    blue_n += 1
                elif red_flat[nidx]:
                    red_n += 1
                else:
                    empty_n += 1
        if in_board >= 4 and empty_n == 0:
            if blue_n > 0 and red_n == 0:
                dead[idx] = True
            elif red_n > 0 and blue_n == 0:
                dead[idx] = True
    return dead


# ─── Rollout Numba (save + complete + uniform) ───────────────────────────────

@numba.njit(cache=True)
def _rollout_jit(blue_flat, red_flat, empty_arr, n_empty, blue_to_play,
                 moves_out, colors_out, N, nb_dr, nb_dc, bridges,
                 p_complete):
    """
    Rollout en place : remplit les cases vides alternativement.
    Priorité : save-bridge > complete-bridge (prob p_complete) > aléatoire.
    Retourne (winner, n_played) avec winner 0=blue, 1=red.
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

        # 1) Sauvegarde d'un pont menacé
        if last_move >= 0:
            save = _bridge_save_jit(blue_flat, red_flat, cur_blue,
                                    last_move, N, bridges)
            if save >= 0 and pos_to_idx[save] >= 0:
                move = save

        # 2) Complétion d'un pont (avec probabilité p_complete)
        if move < 0 and p_complete > 0.0 and np.random.random() < p_complete:
            start_offset = np.random.randint(N * N)
            comp = _bridge_complete_jit(blue_flat, red_flat, cur_blue,
                                        N, bridges, start_offset)
            if comp >= 0 and pos_to_idx[comp] >= 0:
                move = comp

        # 3) Aléatoire uniforme
        if move < 0:
            idx = np.random.randint(remaining)
            move = empty_arr[idx]

        # Swap-pop O(1) depuis empty_arr
        idx = pos_to_idx[move]
        last_pos = empty_arr[remaining - 1]
        empty_arr[idx] = last_pos
        pos_to_idx[last_pos] = idx
        pos_to_idx[move] = -1
        remaining -= 1

        if cur_blue:
            blue_flat[move] = True
        else:
            red_flat[move] = True
        moves_out[n_played] = move
        colors_out[n_played] = 1 if cur_blue else 0
        n_played += 1
        last_move = move

        if n_played >= N:
            if cur_blue:
                d = _shortest_path_jit(blue_flat, red_flat, True, N, nb_dr, nb_dc)
                if d == 0:
                    return 0, n_played
            else:
                d = _shortest_path_jit(red_flat, blue_flat, False, N, nb_dr, nb_dc)
                if d == 0:
                    return 1, n_played

        cur_blue = not cur_blue

    return -1, n_played


# ─── Nœud RAVE avec prior et filtrage des cases mortes ───────────────────────

class _RAVENode:
    """Nœud MCTS avec AMAF, prior heuristique, et filtrage des cases mortes."""

    __slots__ = ('env', 'parent', 'move', 'children', 'untried',
                 'visits', 'wins', 'amaf_visits', 'amaf_wins',
                 '_terminal', 'prior')

    def __init__(self, env: HexEnv, prior: np.ndarray,
                 parent=None, move: int = -1):
        self.env = env
        self.parent = parent
        self.move = move
        self.children: dict[int, _RAVENode] = {}
        self.prior = prior

        # Filtrage dead cells + tri par prior décroissant
        blue_flat = env.blue.ravel()
        red_flat = env.red.ravel()
        dead = _dead_cells_6n_jit(blue_flat, red_flat, BS, _NB_DR, _NB_DC)
        legal = env.get_legal_moves()
        filtered = [int(m) for m in legal if not dead[m]]
        if not filtered and len(legal) > 0:
            filtered = [int(m) for m in legal]
        filtered.sort(key=lambda m: -prior[m])
        self.untried = filtered

        self.visits = 0
        self.wins = 0.0
        self.amaf_visits = np.zeros(NN, dtype=np.int32)
        self.amaf_wins = np.zeros(NN, dtype=np.float32)
        self._terminal = None

    def is_terminal(self) -> bool:
        if self._terminal is None:
            self._terminal = self.env.is_terminal()
        return self._terminal


# ─── Joueur MoHex ─────────────────────────────────────────────────────────────

class MoHexPlayer:
    """
    MoHex v2 : MCTS + RAVE/AMAF + prior heuristique + tree reuse + rollouts Numba.
    Interface : select_move(env, time_s) -> int
    """

    def __init__(self, c_uct: float = 0.7, rave_k: float = 300.0,
                 c_bias: float = 2.0, p_complete: float = 0.5,
                 min_simulations: int = 256):
        self.c_uct = c_uct
        self.rave_k = rave_k
        self.c_bias = c_bias
        self.p_complete = p_complete
        self.min_simulations = min_simulations
        self.last_stats: dict = {}
        self._root: _RAVENode | None = None

    def reset(self) -> None:
        """Vide l'arbre (à appeler en début de nouvelle partie)."""
        self._root = None

    def _compute_prior(self, env: HexEnv) -> np.ndarray:
        return _compute_prior_jit(
            env.blue.ravel(), env.red.ravel(),
            env.blue_to_play, BS, _NB_DR, _NB_DC, _BRIDGE_ARR,
        )

    def _create_node(self, env: HexEnv, parent=None, move: int = -1) -> _RAVENode:
        prior = self._compute_prior(env)
        return _RAVENode(env, prior, parent, move)

    def _try_reuse_tree(self, env: HexEnv) -> _RAVENode | None:
        """
        Si l'état env correspond à un descendant proche (0 ou 1 coup) de
        self._root, on y descend et on retourne le sous-arbre. Sinon None.
        """
        if self._root is None:
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
        m = int(rr) * BS + int(cc)

        # La pierre ajoutée doit correspondre à la couleur qui devait jouer dans stored
        if is_blue_move != stored.blue_to_play:
            return None

        child = self._root.children.get(m)
        if child is None:
            return None
        if child.env.blue_to_play != env.blue_to_play:
            return None
        child.parent = None
        return child

    def _find_forced_block(self, env: HexEnv, legal, root_color: bool) -> int:
        """
        Si l'adversaire a UN coup gagnant immédiat (menace à 1 coup), on le bloque.
        Retourne l'index de la case à jouer, ou -1 si aucune menace.
        """
        threats = []
        for move in legal:
            m = int(move)
            r, c = divmod(m, BS)
            if root_color:
                env.red[r, c] = True
            else:
                env.blue[r, c] = True
            env._winner = None
            w = env.winner()
            if root_color:
                env.red[r, c] = False
            else:
                env.blue[r, c] = False
            env._winner = None
            if (root_color and w == 'red') or (not root_color and w == 'blue'):
                threats.append(m)
                if len(threats) > 1:
                    break
        if len(threats) == 1:
            return threats[0]
        return -1

    def _rave_select(self, node: _RAVENode) -> _RAVENode:
        """Sélectionne l'enfant maximisant RAVE-UCT + progressive bias."""
        log_n = math.log(max(node.visits, 1))
        sqrt_n = math.sqrt(max(node.visits, 1))
        beta_den = 3.0 * node.visits + self.rave_k
        beta = math.sqrt(self.rave_k / beta_den) if beta_den > 0 else 0.0
        best_child = None
        best_score = -1e18

        for move, child in node.children.items():
            if child.visits == 0:
                return child

            exploit = 1.0 - (child.wins / child.visits)

            amaf_n = node.amaf_visits[move]
            if amaf_n > 0:
                amaf_val = node.amaf_wins[move] / amaf_n
            else:
                amaf_val = 0.5

            combined = (1.0 - beta) * exploit + beta * amaf_val
            explore = self.c_uct * math.sqrt(log_n / child.visits)
            bias = self.c_bias * float(node.prior[move]) * sqrt_n / (1 + child.visits)
            score = combined + explore + bias

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _simulate_once(self, root: _RAVENode) -> None:
        node = root

        # 1) Sélection
        while (not node.is_terminal()
               and len(node.untried) == 0
               and node.children):
            node = self._rave_select(node)

        # 2) Expansion (coups untried déjà triés par prior)
        if not node.is_terminal() and node.untried:
            move = node.untried.pop(0)
            child_env = node.env.copy()
            child_env.apply_move(move)
            child = self._create_node(child_env, parent=node, move=move)
            node.children[move] = child
            node = child

        # 3) Rollout Numba ou terminal
        if node.is_terminal():
            w = node.env.winner()
            winner_int = 0 if w == 'blue' else 1
            n_played = 0
            played_moves = None
            played_colors = None
        else:
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
                BS, _NB_DR, _NB_DC, _BRIDGE_ARR,
                self.p_complete,
            )
            played_moves = moves_arr[:n_played] if n_played > 0 else None
            played_colors = colors_arr[:n_played] if n_played > 0 else None

        # 4) Backpropagation + RAVE vectorisée
        cur = node
        while cur is not None:
            cur.visits += 1
            node_blue = cur.env.blue_to_play
            if winner_int == 0:
                win_val = 1.0 if node_blue else 0.0
            else:
                win_val = 0.0 if node_blue else 1.0
            cur.wins += win_val

            if n_played > 0:
                node_blue_int = 1 if node_blue else 0
                mask = played_colors == node_blue_int
                moves_matched = played_moves[mask]
                cur.amaf_visits[moves_matched] += 1
                if win_val != 0.0:
                    cur.amaf_wins[moves_matched] += win_val

            cur = cur.parent

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        legal = env.get_legal_moves()
        if len(legal) == 0:
            return -1

        root_color = env.blue_to_play

        # 1) Coup gagnant immédiat
        for move in legal:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, root_color)
            if (root_color and w == 'blue') or (not root_color and w == 'red'):
                self._root = None
                self.last_stats = {
                    'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0,
                }
                print("ITERS:1 VISITS:1 WINRATE:1.0000 TIME:0.000",
                      file=sys.stderr)
                return m

        # 2) Blocage forcé si l'adversaire a une menace unique à 1 coup
        forced = self._find_forced_block(env, legal, root_color)
        if forced >= 0:
            self._root = None
            self.last_stats = {
                'iters': 1, 'visits': 1, 'winrate': 0.5, 'time': 0.0,
            }
            print("ITERS:1 VISITS:1 WINRATE:0.5000 TIME:0.000 FORCED_BLOCK",
                  file=sys.stderr)
            return forced

        # 3) Tree reuse si possible, sinon nouvelle racine
        reused = self._try_reuse_tree(env)
        if reused is not None:
            root = reused
        else:
            root = self._create_node(env.copy())

        # 4) Boucle MCTS
        t0 = time.time()
        deadline = t0 + max(time_s, 0.01)
        sims = 0
        while sims < self.min_simulations or time.time() < deadline:
            self._simulate_once(root)
            sims += 1

        # 5) Meilleur coup par nombre de visites
        best_move = int(legal[0])
        best_visits = -1
        best_wr = 0.0
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                best_wr = 1.0 - (child.wins / max(child.visits, 1))

        # 6) Avance la racine dans le coup choisi pour le prochain appel
        chosen_child = root.children.get(best_move)
        if chosen_child is not None:
            chosen_child.parent = None
            self._root = chosen_child
        else:
            self._root = None

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
        print("usage: python mohex.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = MoHexPlayer()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

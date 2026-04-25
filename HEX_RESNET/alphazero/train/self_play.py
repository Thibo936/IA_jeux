# self_play.py — Génération de parties en self-play pour AlphaZero Hex

import os
import sys
import time
import numpy as np
from collections import deque

_dir = os.path.dirname(os.path.abspath(__file__))
_ia = os.path.join(os.path.dirname(_dir), 'ia')
for _p in [_dir, _ia]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from mcts_az import MCTSAgent, MCTSNode
from config import (
    REPLAY_BUFFER_SIZE, GAMES_PER_ITER, MCTS_SIMULATIONS, NUM_CELLS,
    DIRICHLET_ALPHA, DIRICHLET_EPS, TEMPERATURE_MOVES,
    N_PARALLEL_GAMES, LEAVES_PER_GAME,
)


class ReplayBuffer:
    """Buffer circulaire stockant les exemples d'entraînement."""

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self._states    : deque = deque(maxlen=max_size)
        self._policies  : deque = deque(maxlen=max_size)
        self._values    : deque = deque(maxlen=max_size)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        self._states.append(state)
        self._policies.append(policy)
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._states)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne un batch aléatoire (states, policies, values)."""
        idx = np.random.choice(len(self._states), size=batch_size, replace=False)
        states   = np.stack([self._states[i]   for i in idx])
        policies = np.stack([self._policies[i] for i in idx])
        values   = np.array([self._values[i]   for i in idx], dtype=np.float32)
        return states, policies, values

    def save(self, path: str) -> None:
        """Sauvegarde le buffer sur disque."""
        if len(self) == 0:
            return
        np.savez_compressed(
            path,
            states=np.stack(list(self._states)),
            policies=np.stack(list(self._policies)),
            values=np.array(list(self._values), dtype=np.float32),
        )

    def load(self, path: str) -> int:
        """Charge le buffer depuis le disque. Retourne le nombre d'exemples chargés."""
        import os
        if not os.path.isfile(path):
            return 0
        data = np.load(path)
        states, policies, values = data['states'], data['policies'], data['values']
        count = 0
        for i in range(len(states)):
            self.add(states[i], policies[i], float(values[i]))
            count += 1
        return count


def _augment(state: np.ndarray, policy: np.ndarray, value: float):
    """
    Augmentation de données : réflexion diagonale (transpose) du plateau.

    Pour le Hex, la réflexion diagonale échange Blue et Red, donc :
    - on transpose chaque plan du tenseur d'état
    - on échange les plans Blue (0) et Red (1)
    - on remet le plan joueur courant à jour (inversé)
    - on retourne la politique (121 coups) réindexée
    - la valeur est CONSERVÉE (la symétrie échange totalement les rôles :
      si la position était gagnante pour le joueur courant, elle l'est
      toujours après transformation pour le nouveau joueur courant)

    Retourne (state_aug, policy_aug, value).
    """
    # state : (3, 11, 11)
    state_aug = np.zeros_like(state)
    state_aug[0] = state[1].T          # Red transposé → nouveau Blue
    state_aug[1] = state[0].T          # Blue transposé → nouveau Red
    state_aug[2] = 1.0 - state[2]      # joueur inversé

    # Politique : réindexer les 121 coups (row*11+col) → (col*11+row) après transpose
    policy_aug = np.zeros(NUM_CELLS, dtype=np.float32)
    for i in range(NUM_CELLS):
        r, c = divmod(i, 11)
        j = c * 11 + r                 # indice transposé
        policy_aug[j] = policy[i]

    return state_aug, policy_aug, value  # valeur conservée (symétrie totale des rôles)


def play_one_game(
    agent: MCTSAgent,
    buffer: ReplayBuffer,
    augment: bool = True,
    verbose: bool = False,
) -> str:
    """
    Joue une partie complète en self-play.
    Enregistre tous les exemples dans `buffer`.
    Retourne 'blue' ou 'red'.
    """
    env = HexEnv()
    history = []   # liste de (state_tensor, pi, blue_to_play)
    move_count = 0
    reuse_root = None  # sous-arbre pour tree reuse

    while not env.is_terminal():
        pi, root = agent.get_policy(env, move_count=move_count,
                                     return_root=True, reuse_root=reuse_root)

        state = env.get_state_tensor()
        history.append((state.copy(), pi.copy(), env.blue_to_play))

        # Sélection du coup
        move = int(np.random.choice(NUM_CELLS, p=pi))

        # Tree reuse : récupérer le sous-arbre du coup joué
        if move in root.children:
            reuse_root = root.children[move]
        else:
            reuse_root = None
        if verbose:
            r, c = divmod(move, 11)
            print(f"  Coup {move_count+1}: {chr(ord('A')+c)}{r+1} "
                  f"({'Blue' if env.blue_to_play else 'Red'})")
        env.apply_move(move)
        move_count += 1

    gagnant = env.winner()  # 'blue' ou 'red'

    # Conversion vainqueur → valeur (+1 pour le gagnant, -1 pour le perdant)
    # valeur depuis le point de vue du joueur qui joue à cet état
    for state, pi, was_blue_to_play in history:
        if gagnant == 'blue':
            z = 1.0 if was_blue_to_play else -1.0
        else:
            z = -1.0 if was_blue_to_play else 1.0

        buffer.add(state, pi, float(z))
        if augment:
            s_aug, p_aug, z_aug = _augment(state, pi, float(z))
            buffer.add(s_aug, p_aug, z_aug)

    if verbose:
        print(f"  → Vainqueur : {gagnant} en {move_count} coups")

    return gagnant


# ─── Self-play parallèle avec batching cross-games ───────────────────────────

class GameSlot:
    """Encapsule une partie en cours pour le self-play parallèle."""

    def __init__(self, agent: MCTSAgent):
        self.env = HexEnv()
        self.history: list = []   # [(state_tensor, pi, blue_to_play)]
        self.move_count = 0
        self.sims_remaining = 0
        self.reuse_node = None
        self._init_root(agent)

    def _init_root(self, agent: MCTSAgent) -> None:
        """Initialise le nœud racine MCTS pour le coup courant."""
        if self.reuse_node is not None and self.reuse_node.is_expanded:
            self.root = self.reuse_node
            self.root.parent = None
        else:
            self.root = MCTSNode(self.env.copy())
            agent._expand(self.root)

        # Bruit Dirichlet à la racine
        if agent.add_dirichlet and self.root.children:
            moves = list(self.root.children.keys())
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
            for m, n in zip(moves, noise):
                child = self.root.children[m]
                child.P = (1 - DIRICHLET_EPS) * child.P + DIRICHLET_EPS * n

        self.sims_remaining = agent.sims - 1  # -1 car racine déjà évaluée

    def needs_sims(self) -> bool:
        return self.sims_remaining > 0

    def collect_leaves(self, agent: MCTSAgent, max_leaves: int) -> list:
        """
        Sélectionne des feuilles pour évaluation GPU.
        Gère les terminaux/collisions immédiatement.
        Retourne la liste des feuilles nécessitant une inférence réseau.
        """
        k = min(max_leaves, self.sims_remaining)
        leaves = []
        sims_consumed = 0

        for _ in range(k):
            leaf = agent._select_leaf(self.root)
            if leaf.is_terminal or leaf.is_expanded:
                if leaf.is_terminal:
                    agent._backprop(leaf, -1.0)
                    sims_consumed += 1
                continue
            if leaf.env is None:
                continue
            if leaf.env.is_terminal():
                leaf.is_terminal = True
                agent._backprop(leaf, -1.0)
                sims_consumed += 1
                continue
            agent._apply_virtual_loss(leaf)
            leaves.append(leaf)
            sims_consumed += 1

        self.sims_remaining -= max(sims_consumed, 1)  # au moins 1 pour éviter stalling
        return leaves

    def dispatch_results(
        self, agent: MCTSAgent,
        leaves: list, policies: np.ndarray, values: np.ndarray,
    ) -> None:
        """Expand et backprop les feuilles évaluées par le GPU."""
        for i, leaf in enumerate(leaves):
            agent._undo_virtual_loss(leaf)
            agent._expand_with_policy(leaf, policies[i])
            agent._backprop(leaf, float(values[i]))

    def advance_move(self, agent: MCTSAgent) -> bool:
        """Calcule pi, joue un coup, tree reuse. Retourne True si fin de partie."""
        # Distribution de visites
        pi = np.zeros(NUM_CELLS, dtype=np.float32)
        for move, child in self.root.children.items():
            pi[move] = child.N

        # Température
        if self.move_count < TEMPERATURE_MOVES:
            s = pi.sum()
            if s > 0:
                pi /= s
        else:
            best = pi.argmax()
            pi[:] = 0.0
            pi[best] = 1.0

        # Enregistrer l'état avant le coup
        state = self.env.get_state_tensor()
        self.history.append((state.copy(), pi.copy(), self.env.blue_to_play))

        # Sélection du coup
        if self.move_count < TEMPERATURE_MOVES:
            move = int(np.random.choice(NUM_CELLS, p=pi))
        else:
            move = int(pi.argmax())

        # Tree reuse
        if move in self.root.children:
            self.reuse_node = self.root.children[move]
        else:
            self.reuse_node = None

        self.env.apply_move(move)
        self.move_count += 1

        if self.env.is_terminal():
            return True

        self._init_root(agent)
        return False

    def finalize(self, buffer: ReplayBuffer, augment: bool = True) -> str:
        """Écrit les positions dans le buffer avec z-values. Retourne le vainqueur."""
        winner = self.env.winner()
        for state, pi, was_blue in self.history:
            if winner == 'blue':
                z = 1.0 if was_blue else -1.0
            else:
                z = -1.0 if was_blue else 1.0
            buffer.add(state, pi, float(z))
            if augment:
                s_aug, p_aug, z_aug = _augment(state, pi, float(z))
                buffer.add(s_aug, p_aug, z_aug)
        return winner


def run_self_play(
    agent: MCTSAgent,
    buffer: ReplayBuffer,
    num_games: int = GAMES_PER_ITER,
    verbose: bool = False,
) -> dict:
    """
    Lance `num_games` parties de self-play en parallèle (N slots simultanés).
    Les évaluations réseau de tous les slots sont batchées ensemble.
    Retourne les statistiques : {'blue_wins': int, 'red_wins': int, 'total': int}.
    """
    stats = {'blue_wins': 0, 'red_wins': 0, 'total': num_games}
    t_start = time.time()
    width = 25

    n_slots = min(N_PARALLEL_GAMES, num_games)
    slots: list[GameSlot | None] = [GameSlot(agent) for _ in range(n_slots)]
    games_started = n_slots
    games_completed = 0

    while games_completed < num_games:
        # ── Collecter les feuilles de tous les slots actifs ───────────────
        all_leaves: list = []
        leaf_ranges: list = []  # (slot_idx, start, count)

        for idx in range(len(slots)):
            slot = slots[idx]
            if slot is None or not slot.needs_sims():
                continue
            leaves = slot.collect_leaves(agent, LEAVES_PER_GAME)
            if leaves:
                leaf_ranges.append((idx, len(all_leaves), len(leaves)))
                all_leaves.extend(leaves)

        # ── Inférence GPU batchée ─────────────────────────────────────────
        if all_leaves:
            states = np.stack([l.env.get_state_tensor() for l in all_leaves])
            masks  = np.stack([l.env.legal_mask()       for l in all_leaves])

            if agent.net is not None:
                policies, values = agent.net.batch_predict(
                    states, masks, agent.device
                )
            else:
                # Sans réseau : politique uniforme
                policies = masks.astype(np.float32)
                sums = policies.sum(axis=1, keepdims=True)
                policies /= np.maximum(sums, 1e-8)
                values = np.zeros(len(all_leaves), dtype=np.float32)

            for slot_idx, start, count in leaf_ranges:
                slots[slot_idx].dispatch_results(
                    agent,
                    all_leaves[start:start + count],
                    policies[start:start + count],
                    values[start:start + count],
                )

        # ── Avancer les slots qui ont fini leurs simulations ──────────────
        for idx in range(len(slots)):
            slot = slots[idx]
            if slot is None or slot.needs_sims():
                continue
            terminal = slot.advance_move(agent)
            if terminal:
                winner = slot.finalize(buffer)
                games_completed += 1
                if winner == 'blue':
                    stats['blue_wins'] += 1
                else:
                    stats['red_wins'] += 1

                # Barre de progression
                done = games_completed
                elapsed = time.time() - t_start
                vitesse = done / elapsed if elapsed > 0 else 0
                eta = (num_games - done) / vitesse if vitesse > 0 else 0
                filled = int(width * done / num_games)
                bar = '█' * filled + '░' * (width - filled)
                sys.stdout.write(
                    f"\r  [{bar}] {done}/{num_games} | "
                    f"Blue:{stats['blue_wins']} Red:{stats['red_wins']} | "
                    f"{vitesse:.2f}p/s | écoulé:{elapsed:.0f}s ETA:{eta:.0f}s | "
                    f"buf:{len(buffer)}"
                )
                sys.stdout.flush()
                if verbose:
                    sys.stdout.write('\n')

                # Remplir le slot avec une nouvelle partie
                if games_started < num_games:
                    slots[idx] = GameSlot(agent)
                    games_started += 1
                else:
                    slots[idx] = None

    sys.stdout.write('\n')
    return stats

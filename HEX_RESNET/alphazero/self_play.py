# self_play.py — Génération de parties en self-play pour AlphaZero Hex

import numpy as np
from collections import deque

from hex_env import HexEnv
from mcts_az import MCTSAgent
from config import (
    REPLAY_BUFFER_SIZE, GAMES_PER_ITER, MCTS_SIMULATIONS, NUM_CELLS
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

    while not env.is_terminal():
        pi = agent.get_policy(env, move_count=move_count)

        state = env.get_state_tensor()
        history.append((state.copy(), pi.copy(), env.blue_to_play))

        # Sélection du coup
        move = int(np.random.choice(NUM_CELLS, p=pi))
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


def run_self_play(
    agent: MCTSAgent,
    buffer: ReplayBuffer,
    num_games: int = GAMES_PER_ITER,
    verbose: bool = False,
) -> dict:
    """
    Lance `num_games` parties de self-play.
    Retourne les statistiques : {'blue_wins': int, 'red_wins': int, 'total': int}.
    """
    stats = {'blue_wins': 0, 'red_wins': 0, 'total': num_games}
    for i in range(num_games):
        winner = play_one_game(agent, buffer, augment=True, verbose=verbose)
        if winner == 'blue':
            stats['blue_wins'] += 1
        else:
            stats['red_wins'] += 1
        if verbose or (i + 1) % 10 == 0:
            print(f"  Partie {i+1}/{num_games} terminée | buffer={len(buffer)}")

    return stats

# mcts_az.py — MCTS guidé par réseau de neurones (UCB-PUCT style AlphaZero)

import math
import numpy as np
import torch

from hex_env import HexEnv
from config import (
    C_PUCT, MCTS_SIMULATIONS, DIRICHLET_ALPHA, DIRICHLET_EPS,
    TEMPERATURE_MOVES, NUM_CELLS
)


class MCTSNode:
    """Nœud de l'arbre MCTS."""

    __slots__ = ('env', 'parent', 'move', 'children',
                 'N', 'W', 'Q', 'P', 'is_expanded', 'is_terminal')

    def __init__(self, env: HexEnv, parent: "MCTSNode | None" = None, move: int = -1, prior: float = 0.0):
        self.env         = env
        self.parent      = parent
        self.move        = move        # coup qui a mené à ce nœud
        self.children    : dict[int, "MCTSNode"] = {}

        self.N  = 0        # nombre de visites
        self.W  = 0.0      # somme des valeurs
        self.Q  = 0.0      # valeur moyenne W/N
        self.P  = prior    # probabilité a priori (issue du réseau)

        self.is_expanded = False
        self.is_terminal = False


class MCTSAgent:
    """
    Agent MCTS guidé par un réseau de neurones (AlphaZero UCB-PUCT).

    Si `net` est None, utilise une politique uniforme (utile pour les tests).
    """

    def __init__(
        self,
        net,                        # HexNet ou None
        device: torch.device | None = None,
        sims: int = MCTS_SIMULATIONS,
        c_puct: float = C_PUCT,
        add_dirichlet: bool = True,
    ):
        self.net           = net
        self.device        = device or torch.device("cpu")
        self.sims          = sims
        self.c_puct        = c_puct
        self.add_dirichlet = add_dirichlet

    # ─── Évaluation réseau ────────────────────────────────────────────────────

    def _evaluate(self, env: HexEnv) -> tuple[np.ndarray, float]:
        """
        Appelle le réseau sur l'état `env`.
        Retourne (policy, value) depuis le point de vue du joueur courant.
        Si net=None, retourne politique uniforme et valeur 0.
        """
        legal_mask = env.legal_mask()

        if self.net is None:
            # Politique uniforme sur les coups légaux
            n_legal = legal_mask.sum()
            policy = legal_mask.astype(np.float32) / max(n_legal, 1)
            return policy, 0.0

        state = env.get_state_tensor()
        policy, value = self.net.predict(state, legal_mask, self.device)
        return policy, value

    # ─── Sélection (UCB-PUCT) ─────────────────────────────────────────────────

    def _select_child(self, node: MCTSNode) -> int:
        """Retourne le coup (move) de l'enfant avec le score UCB-PUCT le plus élevé."""
        sqrt_N = math.sqrt(max(node.N, 1))
        best_score = -float('inf')
        best_move  = -1

        for move, child in node.children.items():
            # Q depuis le point de vue du parent = -Q_enfant
            q = -child.Q if child.N > 0 else 0.0
            u = self.c_puct * child.P * sqrt_N / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_move  = move

        return best_move

    # ─── Expansion ────────────────────────────────────────────────────────────

    def _expand(self, node: MCTSNode) -> float:
        """
        Expanse le nœud : appelle le réseau et crée les enfants.
        Retourne la valeur (depuis le point de vue du joueur AU nœud).
        """
        if node.env.is_terminal():
            node.is_terminal = True
            # Le joueur qui vient de jouer a gagné → valeur +1 pour l'adversaire
            # (c'est l'adversaire qui est maintenant au trait)
            return -1.0

        policy, value = self._evaluate(node.env)
        node.is_expanded = True

        for move in range(NUM_CELLS):
            if policy[move] > 0:
                child_env = node.env.copy()
                child_env.apply_move(move)
                node.children[move] = MCTSNode(
                    env=child_env, parent=node, move=move, prior=float(policy[move])
                )

        return value  # valeur du joueur courant au nœud

    # ─── Backpropagation ──────────────────────────────────────────────────────

    def _backprop(self, node: MCTSNode, value: float) -> None:
        """
        Remonte la valeur dans l'arbre.
        La valeur alterne de signe à chaque niveau (point de vue du joueur courant).
        """
        cur = node
        v   = value
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q  = cur.W / cur.N
            v   = -v           # changement de point de vue
            cur = cur.parent

    # ─── Simulation complète ──────────────────────────────────────────────────

    def _simulate(self, root: MCTSNode) -> None:
        """Effectue une simulation MCTS depuis la racine."""
        node = root

        # Descente
        while node.is_expanded and not node.is_terminal:
            move = self._select_child(node)
            if move == -1:
                break
            node = node.children[move]

        # Expansion + évaluation
        value = self._expand(node)

        # Backpropagation
        self._backprop(node, value)

    # ─── Politique MCTS ───────────────────────────────────────────────────────

    def get_policy(
        self,
        env: HexEnv,
        move_count: int = 0,
        return_root: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, MCTSNode]:
        """
        Lance `self.sims` simulations depuis l'état `env`.
        Retourne la distribution de visites π(a) ∈ ℝ^121.

        move_count : numéro du coup (pour le paramètre de température τ)
        return_root : si True, retourne aussi le nœud racine
        """
        root_env = env.copy()
        root = MCTSNode(root_env)

        # Premier expand pour obtenir les a priori
        self._expand(root)

        # Bruit Dirichlet à la racine (self-play uniquement)
        if self.add_dirichlet and root.children:
            moves  = list(root.children.keys())
            noise  = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
            for m, n in zip(moves, noise):
                child = root.children[m]
                child.P = (1 - DIRICHLET_EPS) * child.P + DIRICHLET_EPS * n

        # Simulations
        for _ in range(self.sims - 1):   # -1 car on a déjà évalué la racine
            self._simulate(root)

        # Construction de la distribution π
        pi = np.zeros(NUM_CELLS, dtype=np.float32)
        for move, child in root.children.items():
            pi[move] = child.N

        # Température
        if move_count < TEMPERATURE_MOVES:
            # τ = 1 : proportionnel aux visites
            s = pi.sum()
            if s > 0:
                pi /= s
        else:
            # τ → 0 : coup argmax
            best = pi.argmax()
            pi[:] = 0.0
            pi[best] = 1.0

        if return_root:
            return pi, root
        return pi

    # ─── Sélection du meilleur coup ───────────────────────────────────────────

    def select_move(self, env: HexEnv, move_count: int = 0) -> int:
        """Retourne le coup choisi (index 0..120) selon la politique MCTS."""
        pi = self.get_policy(env, move_count=move_count)
        # Échantillonnage si τ=1, argmax si τ→0
        if move_count < TEMPERATURE_MOVES:
            return int(np.random.choice(NUM_CELLS, p=pi))
        else:
            return int(pi.argmax())

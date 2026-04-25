# mcts_az.py — MCTS guidé par réseau de neurones (UCB-PUCT style AlphaZero)
# Optimisations : expansion paresseuse, inférence batchée, virtual loss

import math
import os
import sys
import numpy as np
import torch

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import (
    C_PUCT, MCTS_SIMULATIONS, DIRICHLET_ALPHA, DIRICHLET_EPS,
    TEMPERATURE_MOVES, NUM_CELLS
)

# Virtual loss appliquée lors de la sélection batchée
_VIRTUAL_LOSS = 3
# Nombre de feuilles à collecter par batch
_BATCH_SIZE = 32


class MCTSNode:
    """Nœud de l'arbre MCTS."""

    __slots__ = ('env', 'parent', 'move', 'children',
                 'N', 'W', 'Q', 'P', 'is_expanded', 'is_terminal')

    def __init__(self, env: HexEnv | None, parent: "MCTSNode | None" = None, move: int = -1, prior: float = 0.0):
        self.env         = env         # None si expansion paresseuse (créé à la visite)
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
    Utilise l'inférence batchée avec virtual loss pour maximiser l'utilisation GPU.

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

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray) -> None:
        """Crée les enfants à partir d'une politique déjà calculée."""
        node.is_expanded = True
        for move in range(NUM_CELLS):
            if policy[move] > 0:
                node.children[move] = MCTSNode(
                    env=None, parent=node, move=move, prior=float(policy[move])
                )

    def _expand(self, node: MCTSNode) -> float:
        """
        Expanse le nœud : appelle le réseau et crée les enfants (paresseusement).
        Retourne la valeur (depuis le point de vue du joueur AU nœud).
        """
        if node.env.is_terminal():
            node.is_terminal = True
            return -1.0

        policy, value = self._evaluate(node.env)
        self._expand_with_policy(node, policy)
        return value

    # ─── Backpropagation ──────────────────────────────────────────────────────

    def _backprop(self, node: MCTSNode, value: float) -> None:
        """Remonte la valeur dans l'arbre (alterne le signe à chaque niveau)."""
        cur = node
        v   = value
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q  = cur.W / cur.N
            v   = -v
            cur = cur.parent

    # ─── Virtual loss ─────────────────────────────────────────────────────────

    def _apply_virtual_loss(self, node: MCTSNode) -> None:
        """Applique une virtual loss sur le chemin de la racine à la feuille."""
        cur = node
        while cur is not None:
            cur.N += _VIRTUAL_LOSS
            cur.W -= _VIRTUAL_LOSS
            cur.Q = cur.W / cur.N if cur.N > 0 else 0.0
            cur = cur.parent

    def _undo_virtual_loss(self, node: MCTSNode) -> None:
        """Annule la virtual loss sur le chemin de la racine à la feuille."""
        cur = node
        while cur is not None:
            cur.N -= _VIRTUAL_LOSS
            cur.W += _VIRTUAL_LOSS
            cur.Q = cur.W / cur.N if cur.N > 0 else 0.0
            cur = cur.parent

    # ─── Sélection d'une feuille (descente + materialisation env) ─────────────

    def _select_leaf(self, root: MCTSNode) -> MCTSNode:
        """Descend de la racine jusqu'à une feuille non-expandée."""
        node = root
        while node.is_expanded and not node.is_terminal:
            move = self._select_child(node)
            if move == -1:
                break
            child = node.children[move]
            if child.env is None:
                child.env = node.env.copy()
                child.env.apply_move(move)
            node = child
        return node

    # ─── Simulation séquentielle (fallback) ───────────────────────────────────

    def _simulate(self, root: MCTSNode) -> None:
        """Effectue une simulation MCTS depuis la racine."""
        node = self._select_leaf(root)
        value = self._expand(node)
        self._backprop(node, value)

    # ─── Simulation batchée ───────────────────────────────────────────────────

    def _simulate_batch(self, root: MCTSNode, batch_size: int) -> int:
        """
        Sélectionne plusieurs feuilles avec virtual loss, les évalue en batch,
        puis expand et backprop.
        Retourne le nombre de simulations effectivement réalisées.
        """
        leaves = []
        handled = 0  # terminaux et collisions traités

        for _ in range(batch_size):
            leaf = self._select_leaf(root)

            if leaf.is_terminal or leaf.is_expanded:
                # Terminal ou déjà expandé (collision) : traitement immédiat
                if leaf.is_terminal:
                    self._backprop(leaf, -1.0)
                    handled += 1
                continue

            # Matérialiser l'env si nécessaire (devrait déjà être fait par _select_leaf)
            if leaf.env is None:
                continue

            if leaf.env.is_terminal():
                leaf.is_terminal = True
                self._backprop(leaf, -1.0)
                handled += 1
                continue

            # Appliquer virtual loss pour diversifier les sélections suivantes
            self._apply_virtual_loss(leaf)
            leaves.append(leaf)

        if not leaves:
            return max(handled, 1)  # au moins 1 pour éviter boucle infinie

        if self.net is None:
            # Sans réseau : expansion séquentielle uniforme
            for leaf in leaves:
                self._undo_virtual_loss(leaf)
                value = self._expand(leaf)
                self._backprop(leaf, value)
            return len(leaves) + handled

        # ── Batch inference GPU ───────────────────────────────────────────────
        states = np.stack([leaf.env.get_state_tensor() for leaf in leaves])
        masks = np.stack([leaf.env.legal_mask() for leaf in leaves])

        policies, values = self.net.batch_predict(states, masks, self.device)

        # ── Expand + backprop ─────────────────────────────────────────────────
        for i, leaf in enumerate(leaves):
            self._undo_virtual_loss(leaf)
            self._expand_with_policy(leaf, policies[i])
            self._backprop(leaf, float(values[i]))

        return len(leaves) + handled

    # ─── Politique MCTS ───────────────────────────────────────────────────────

    def get_policy(
        self,
        env: HexEnv,
        move_count: int = 0,
        return_root: bool = False,
        reuse_root: MCTSNode | None = None,
    ) -> np.ndarray | tuple[np.ndarray, MCTSNode]:
        """
        Lance `self.sims` simulations depuis l'état `env`.
        Retourne la distribution de visites π(a) ∈ ℝ^121.

        move_count : numéro du coup (pour le paramètre de température τ)
        return_root : si True, retourne aussi le nœud racine
        reuse_root : sous-arbre à réutiliser (tree reuse)
        """
        if reuse_root is not None and reuse_root.is_expanded:
            root = reuse_root
            root.parent = None  # détacher du parent pour le GC
        else:
            root_env = env.copy()
            root = MCTSNode(root_env)
            self._expand(root)

        # Bruit Dirichlet à la racine (self-play uniquement)
        if self.add_dirichlet and root.children:
            moves  = list(root.children.keys())
            noise  = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
            for m, n in zip(moves, noise):
                child = root.children[m]
                child.P = (1 - DIRICHLET_EPS) * child.P + DIRICHLET_EPS * n

        # Simulations batchées (si réseau disponible)
        remaining = self.sims - 1  # -1 car on a déjà évalué la racine
        use_batch = self.net is not None
        batch_sz = _BATCH_SIZE if use_batch else 1

        while remaining > 0:
            if use_batch and remaining >= batch_sz:
                done = self._simulate_batch(root, batch_sz)
                remaining -= done
            else:
                self._simulate(root)
                remaining -= 1

        # Construction de la distribution π
        pi = np.zeros(NUM_CELLS, dtype=np.float32)
        for move, child in root.children.items():
            pi[move] = child.N

        # Température
        if move_count < TEMPERATURE_MOVES:
            s = pi.sum()
            if s > 0:
                pi /= s
        else:
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
        if move_count < TEMPERATURE_MOVES:
            return int(np.random.choice(NUM_CELLS, p=pi))
        else:
            return int(pi.argmax())

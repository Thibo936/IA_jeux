# hex_env.py — Moteur Hex 11×11 en Python/numpy
# Blue (O) connecte Nord (ligne 0) → Sud (ligne 10)
# Red  (@) connecte Ouest (col 0)  → Est  (col 10)

import numpy as np
from collections import deque
from config import BOARD_SIZE, NUM_CELLS

# Voisins hexagonaux : (dr, dc)
_HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


class HexEnv:
    """
    État d'une partie de Hex 11×11.
    Toutes les opérations sont sur des tableaux numpy bool 11×11.
    """

    def __init__(self):
        self.blue  = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)  # pions Blue
        self.red   = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)  # pions Red
        self.blue_to_play = True   # True = Blue joue, False = Red joue
        self._winner = None        # cache: None / 'blue' / 'red'

    # ─── Construction depuis une chaîne 121 chars ─────────────────────────────

    @classmethod
    def from_string(cls, board_str: str, player_char: str) -> "HexEnv":
        """
        board_str : 121 chars '.' / 'O' / '@' (row-major)
        player_char : 'O' (Blue) ou '@' (Red)
        """
        env = cls()
        for i, ch in enumerate(board_str[:NUM_CELLS]):
            r, c = divmod(i, BOARD_SIZE)
            if ch == 'O':
                env.blue[r, c] = True
            elif ch == '@':
                env.red[r, c] = True
        env.blue_to_play = (player_char == 'O')
        return env

    # ─── Coups légaux ─────────────────────────────────────────────────────────

    def get_legal_moves(self) -> np.ndarray:
        """Retourne un tableau 1-D d'indices (0..120) des cases vides."""
        occupied = self.blue | self.red
        return np.where(~occupied.ravel())[0]

    def legal_mask(self) -> np.ndarray:
        """Retourne un masque bool (121,) : True = coup légal."""
        occupied = self.blue | self.red
        return ~occupied.ravel()

    # ─── Application d'un coup ────────────────────────────────────────────────

    def apply_move(self, pos: int) -> None:
        """
        Joue le coup `pos` (0..120) pour le joueur courant.
        Met à jour blue_to_play et invalide le cache vainqueur.
        """
        r, c = divmod(pos, BOARD_SIZE)
        if self.blue_to_play:
            self.blue[r, c] = True
        else:
            self.red[r, c] = True
        self._winner = None
        self.blue_to_play = not self.blue_to_play

    # ─── Détection de victoire (BFS) ──────────────────────────────────────────

    def _blue_wins(self) -> bool:
        """Blue gagne si elle connecte ligne 0 → ligne 10."""
        visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        queue = deque()
        for c in range(BOARD_SIZE):
            if self.blue[0, c] and not visited[0, c]:
                visited[0, c] = True
                queue.append((0, c))
        while queue:
            r, c = queue.popleft()
            if r == BOARD_SIZE - 1:
                return True
            for dr, dc in _HEX_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.blue[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return False

    def _red_wins(self) -> bool:
        """Red gagne si elle connecte col 0 → col 10."""
        visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        queue = deque()
        for r in range(BOARD_SIZE):
            if self.red[r, 0] and not visited[r, 0]:
                visited[r, 0] = True
                queue.append((r, 0))
        while queue:
            r, c = queue.popleft()
            if c == BOARD_SIZE - 1:
                return True
            for dr, dc in _HEX_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.red[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return False

    def is_terminal(self) -> bool:
        """Retourne True si la partie est terminée (un joueur a gagné)."""
        if self._winner is not None:
            return True
        if self._blue_wins():
            self._winner = 'blue'
            return True
        if self._red_wins():
            self._winner = 'red'
            return True
        return False

    def winner(self) -> str | None:
        """Retourne 'blue', 'red', ou None si la partie n'est pas terminée."""
        self.is_terminal()
        return self._winner

    # ─── Représentation tensorielle pour le réseau ────────────────────────────

    def get_state_tensor(self) -> np.ndarray:
        """
        Retourne un array numpy float32 de forme (3, 11, 11) :
          Plan 0 : pièces Blue
          Plan 1 : pièces Red
          Plan 2 : joueur courant (1.0 = Blue, 0.0 = Red) — broadcast
        """
        tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        tensor[0] = self.blue.astype(np.float32)
        tensor[1] = self.red.astype(np.float32)
        tensor[2] = 1.0 if self.blue_to_play else 0.0
        return tensor

    # ─── Copie ────────────────────────────────────────────────────────────────

    def copy(self) -> "HexEnv":
        """Retourne une copie indépendante de l'état courant."""
        env = HexEnv()
        env.blue = self.blue.copy()
        env.red  = self.red.copy()
        env.blue_to_play = self.blue_to_play
        env._winner = self._winner
        return env

    # ─── Augmentation de données ──────────────────────────────────────────────

    def mirror(self) -> "HexEnv":
        """
        Symétrie miroir du plateau Hex : réflexion diagonale principale.
        Échange aussi les rôles Blue/Red et réajuste la direction de victoire :
        Blue devient la connexion Ouest-Est et Red Nord-Sud → on transpose
        les tableaux ET on échange les joueurs pour maintenir la sémantique.

        Note : la réflexion diagonale (transpose) est LA symétrie naturelle du Hex
        car elle échange exactement les deux directions de connexion.
        """
        env = HexEnv()
        # Transposition = réflexion diagonale
        env.blue = self.red.T.copy()   # ancien Red devient nouveau Blue
        env.red  = self.blue.T.copy()  # ancien Blue devient nouveau Red
        env.blue_to_play = not self.blue_to_play
        env._winner = None
        return env

    # ─── Conversion vers chaîne ───────────────────────────────────────────────

    def to_string(self) -> str:
        """Retourne la chaîne 121 chars compatible avec les binaires C++."""
        chars = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.blue[r, c]:
                    chars.append('O')
                elif self.red[r, c]:
                    chars.append('@')
                else:
                    chars.append('.')
        return ''.join(chars)

    def pos_to_str(self, pos: int) -> str:
        """Convertit un index 0..120 en notation 'A1'..'K11'."""
        r, c = divmod(pos, BOARD_SIZE)
        return f"{chr(ord('A') + c)}{r + 1}"

    @staticmethod
    def str_to_pos(s: str) -> int:
        """Convertit 'A1'..'K11' en index 0..120."""
        col = ord(s[0].upper()) - ord('A')
        row = int(s[1:]) - 1
        return row * BOARD_SIZE + col

    # ─── Affichage ────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        lines = []
        header = "  " + " ".join(chr(ord('A') + c) for c in range(BOARD_SIZE))
        lines.append(header)
        for r in range(BOARD_SIZE):
            prefix = " " * r
            row_str = " ".join(
                'O' if self.blue[r, c] else ('@' if self.red[r, c] else '.')
                for c in range(BOARD_SIZE)
            )
            lines.append(f"{prefix}{r+1:2d} {row_str}")
        return "\n".join(lines)

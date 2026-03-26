# random_player.py — Joueur aléatoire pour Hex 11×11
# Choisit un coup légal uniformément au hasard.
# Interface CLI : python random_player.py BOARD PLAYER [time_s]

import sys
import os
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv


class RandomPlayer:
    """
    Joueur aléatoire : sélectionne un coup légal uniformément au hasard.
    Interface : select_move(env, time_s) -> int (index 0..120)
    """

    def __init__(self):
        self.last_stats: dict = {}

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1
        self.last_stats = {}
        return int(np.random.choice(moves))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python random_player.py BOARD PLAYER [time_s]", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _player = RandomPlayer()
    _move = _player.select_move(_env)
    print(_env.pos_to_str(_move))

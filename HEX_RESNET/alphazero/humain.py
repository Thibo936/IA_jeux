# humain.py — Joueur humain pour Hex 11×11
# Demande le coup au clavier via stdin.
# Interface CLI : python humain.py BOARD PLAYER

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv


class HumanPlayer:
    """
    Joueur humain : affiche le plateau et demande un coup au clavier.
    Interface : select_move(env, time_s) -> int (index 0..120)
    """

    def __init__(self):
        self.last_stats: dict = {}

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        couleur = "Blue (O, Nord→Sud)" if env.blue_to_play else "Red (@, Ouest→Est)"
        print(f"\n{env}", file=sys.stderr)
        print(f"\nÀ vous de jouer [{couleur}]", file=sys.stderr)

        while True:
            try:
                txt = input("Votre coup (ex: A1, K11) : ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nPartie abandonnée.", file=sys.stderr)
                sys.exit(0)

            if not txt:
                continue
            try:
                pos = HexEnv.str_to_pos(txt)
            except (ValueError, IndexError):
                print(f"  Format invalide : '{txt}'. Utilisez A1..K11.", file=sys.stderr)
                continue

            legal = env.get_legal_moves()
            if pos not in legal:
                print(f"  Case {txt.upper()} déjà occupée.", file=sys.stderr)
                continue

            self.last_stats = {"coup": txt.upper()}
            return pos


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python humain.py BOARD PLAYER", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _player = HumanPlayer()
    _move = _player.select_move(_env)
    print(_env.pos_to_str(_move))

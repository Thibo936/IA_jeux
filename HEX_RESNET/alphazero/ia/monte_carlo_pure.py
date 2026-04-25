# monte_carlo_pure.py — Joueur Monte Carlo pur (flat MC) pour Hex 11×11
# Heuristique : aucune. Évalue les coups par rollouts aléatoires complets.
# Interface CLI : python monte_carlo_pure.py BOARD PLAYER [time_s]

import sys
import os
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv


class PureMonteCarloPlayer:
    """
    Monte Carlo pur (sans arbre) :
    - choisit un coup candidat,
    - simule une partie aléatoire jusqu'au terminal,
    - estime le taux de victoire de chaque coup.

    Interface : select_move(env, time_s) -> int (index 0..120)
    """

    def __init__(self, min_rollouts: int = 64):
        self.min_rollouts = min_rollouts
        self.last_stats: dict = {}

    @staticmethod
    def _random_rollout(env: HexEnv) -> str:
        """Joue aléatoirement jusqu'à la fin et retourne le vainqueur."""
        while not env.is_terminal():
            legal = env.get_legal_moves()
            move = int(np.random.choice(legal))
            env.apply_move(move)
        return env.winner()

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play

        # Coup gagnant immédiat
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, root_blue)
            if (root_blue and w == 'blue') or ((not root_blue) and w == 'red'):
                self.last_stats = {
                    'iters': 1,
                    'visits': 1,
                    'winrate': 1.0,
                    'time': 0.0,
                }
                print("ITERS:1 VISITS:1 WINRATE:1.0000 TIME:0.000", file=sys.stderr)
                return m

        wins = {int(m): 0 for m in moves}
        visits = {int(m): 0 for m in moves}

        t0 = time.time()
        deadline = t0 + max(time_s, 0.01)
        idx = 0
        iters = 0

        # Assure un minimum de rollouts pour éviter des choix trop bruités.
        while iters < self.min_rollouts or time.time() < deadline:
            m = int(moves[idx])
            idx = (idx + 1) % len(moves)

            sim_env = env.copy()
            sim_env.apply_move(m)

            if sim_env.is_terminal():
                winner = sim_env.winner()
            else:
                winner = self._random_rollout(sim_env)

            iters += 1
            visits[m] += 1
            if (root_blue and winner == 'blue') or ((not root_blue) and winner == 'red'):
                wins[m] += 1

        best_move = int(moves[0])
        best_rate = -1.0
        for move in moves:
            m = int(move)
            rate = wins[m] / max(visits[m], 1)
            if rate > best_rate:
                best_rate = rate
                best_move = m

        elapsed = time.time() - t0
        total_visits = sum(visits.values())

        self.last_stats = {
            'iters': iters,
            'visits': total_visits,
            'winrate': best_rate,
            'time': elapsed,
        }
        print(
            f"ITERS:{iters} VISITS:{total_visits} WINRATE:{best_rate:.4f} TIME:{elapsed:.3f}",
            file=sys.stderr,
        )
        return best_move


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python monte_carlo_pure.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = PureMonteCarloPlayer()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))

import sys
import os
import time
import random

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv

class Gemini31ProPreviewV2:
    def __init__(self, seed: int | None = None):
        self.last_stats: dict = {}
        if seed is not None:
            random.seed(seed)

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play

        # 1) Coup gagnant immédiat
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, root_blue)
            if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
                self.last_stats = {'iters': 1, 'visits': 1, 'winrate': 1.0, 'time': 0.0}
                return m

        # 2) Algorithme - Simulation Monte Carlo basique
        t0 = time.time()
        best_move = int(moves[0])
        best_winrate = -1.0
        
        # Budget temps avec marge de sécurité
        time_budget = time_s * 0.9 
        sims_per_move = 20
        total_sims = 0
        
        for move in moves:
            m = int(move)
            wins = 0
            for _ in range(sims_per_move):
                if time.time() - t0 > time_budget:
                    break
                
                env_copy = env.copy()
                env_copy.apply_move(m)
                
                # Random rollout rapide
                while not env_copy.is_terminal():
                    rollout_moves = env_copy.get_legal_moves()
                    env_copy.apply_move(int(random.choice(rollout_moves)))
                    
                winner = env_copy.winner()
                if (root_blue and winner == 'blue') or (not root_blue and winner == 'red'):
                    wins += 1
                total_sims += 1
                
            winrate = wins / max(1, sims_per_move)
            if winrate > best_winrate:
                best_winrate = winrate
                best_move = m
                
            if time.time() - t0 > time_budget:
                break
                
        elapsed = time.time() - t0

        self.last_stats = {
            'iters': total_sims,
            'visits': total_sims,
            'winrate': best_winrate,
            'time': elapsed,
        }
        return best_move

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python gemini_3_1_pro_preview_V2.py BOARD PLAYER [time_s]", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = Gemini31ProPreviewV2()
    _move = _player.select_move(_env, _time_s)
    
    stats = _player.last_stats
    print(f"ITERS:{stats.get('iters',0)} VISITS:{stats.get('visits',0)} WINRATE:{stats.get('winrate',0):.4f} TIME:{stats.get('time',0):.3f}", file=sys.stderr)
    print(_env.pos_to_str(_move))

# gemini_3_1_pro_preview.py — Gemini3.1 Pro Preview (MCTS)
# IA de compétition pour le tournoi Hex 11x11 - Modèle: Gemini3.1 Pro Preview
# Interface CLI : python gemini_3_1_pro_preview.py BOARD PLAYER [time_s]

import sys
import os
import time
import math
import random

# ─── Bootstrap des imports train/ ─────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from hex_env import HexEnv
except ImportError:
    pass # Will be handled by the environment

class Node:
    __slots__ = ['move', 'parent', 'children', 'visits', 'wins', 'untried_moves', 'is_terminal', 'winner', 'blue_to_play']
    
    def __init__(self, move, parent, untried_moves, is_terminal, winner, blue_to_play):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = untried_moves
        self.is_terminal = is_terminal
        self.winner = winner
        self.blue_to_play = blue_to_play
        
    def ucb1(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

class Gemini3_1ProPreviewIA:
    def __init__(self, param: int = 42):
        self.param = param
        self.last_stats: dict = {}

    def select_move(self, env, time_s: float = 1.5) -> int:
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

        t0 = time.time()
        deadline = t0 + max(0.1, time_s - 0.05)
        
        root_node = Node(None, None, [int(m) for m in moves], env.is_terminal(), env.winner(), root_blue)
        iters = 0
        
        while time.time() < deadline:
            iters += 1
            node = root_node
            sim_env = env.copy()
            
            # Selection
            while node.untried_moves == [] and not node.is_terminal:
                node = max(node.children, key=lambda c: c.ucb1())
                sim_env.apply_move(node.move)
                
            # Expansion
            if not node.is_terminal and len(node.untried_moves) > 0:
                m = random.choice(node.untried_moves)
                node.untried_moves.remove(m)
                
                was_blue = sim_env.blue_to_play
                sim_env.apply_move(m)
                
                is_terminal = sim_env.is_terminal()
                winner = sim_env.winner()
                
                child = Node(m, node, [int(x) for x in sim_env.get_legal_moves()], is_terminal, winner, sim_env.blue_to_play)
                node.children.append(child)
                node = child
                
            # Simulation (Rollout)
            while not sim_env.is_terminal():
                m = int(random.choice(sim_env.get_legal_moves()))
                sim_env.apply_move(m)
                
            winner = sim_env.winner()
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                # If the parent's turn matches the winner, the move leading to this node is good for the parent
                if node.parent is not None:
                    parent_blue = node.parent.blue_to_play
                    if (parent_blue and winner == 'blue') or (not parent_blue and winner == 'red'):
                        node.wins += 1.0
                node = node.parent

        elapsed = time.time() - t0
        
        if root_node.children:
            best_child = max(root_node.children, key=lambda c: c.visits)
            best_move = best_child.move
            winrate = best_child.wins / best_child.visits if best_child.visits > 0 else 0.0
        else:
            best_move = int(random.choice(moves))
            winrate = 0.5
            
        self.last_stats = {
            'iters': iters,
            'visits': root_node.visits,
            'winrate': winrate,
            'time': elapsed,
        }
        
        return best_move

# ─── Interface CLI ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python gemini_3_1_pro_preview.py BOARD PLAYER [time_s]", file=sys.stderr)
        sys.exit(1)

    try:
        from hex_env import HexEnv
        _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
        _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
        _player = Gemini3_1ProPreviewIA()
        _move = _player.select_move(_env, _time_s)
        
        sims = _player.last_stats.get('iters', 1)
        visits = _player.last_stats.get('visits', 1)
        winrate = _player.last_stats.get('winrate', 0.0)
        elapsed = _player.last_stats.get('time', 0.0)
        print(f"ITERS:{sims} VISITS:{visits} WINRATE:{winrate:.4f} TIME:{elapsed:.3f}", file=sys.stderr)
        
        print(_env.pos_to_str(_move))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

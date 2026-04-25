# evaluate.py — Comparaison de deux modèles HexNet par tournoi interne

import os
import sys
import time
import numpy as np
import torch

_dir = os.path.dirname(os.path.abspath(__file__))
_ia = os.path.join(os.path.dirname(_dir), 'ia')
for _p in [_dir, _ia]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from mcts_az import MCTSAgent
from network import HexNet
from config import MCTS_SIMULATIONS_EVAL, EVAL_GAMES, NUM_CELLS


def _play_game(
    agent_blue: MCTSAgent,
    agent_red: MCTSAgent,
) -> str:
    """
    Joue une partie entre deux agents.
    agent_blue joue Blue (Nord-Sud), agent_red joue Red (Ouest-Est).
    Retourne 'blue' ou 'red'.
    """
    env = HexEnv()
    move_count = 0
    while not env.is_terminal():
        if env.blue_to_play:
            pi = agent_blue.get_policy(env, move_count=999)  # τ→0 (argmax)
        else:
            pi = agent_red.get_policy(env, move_count=999)
        move = int(pi.argmax())
        env.apply_move(move)
        move_count += 1
    return env.winner()


def evaluate_models(
    new_net: HexNet,
    best_net: HexNet,
    device: torch.device,
    num_games: int = EVAL_GAMES,
    sims: int = MCTS_SIMULATIONS_EVAL,
) -> float:
    """
    Fait jouer `new_net` contre `best_net` pendant `num_games` parties.
    num_games/2 parties avec new_net=Blue, num_games/2 avec new_net=Red.
    Retourne le win rate de new_net ∈ [0, 1].
    """
    new_net.eval()
    best_net.eval()

    agent_new  = MCTSAgent(new_net,  device=device, sims=sims, add_dirichlet=False)
    agent_best = MCTSAgent(best_net, device=device, sims=sims, add_dirichlet=False)

    wins = 0
    half = num_games // 2
    width = 25
    t_start = time.time()

    def _afficher(done: int, phase: str) -> None:
        elapsed = time.time() - t_start
        vitesse = done / elapsed if elapsed > 0 else 0
        eta = (num_games - done) / vitesse if vitesse > 0 else 0
        wr = wins / done if done > 0 else 0.0
        filled = int(width * done / num_games)
        bar = '█' * filled + '░' * (width - filled)
        sys.stdout.write(
            f"\r  [{bar}] {done}/{num_games} ({phase}) | "
            f"wins={wins} wr={wr:.0%} | "
            f"écoulé:{elapsed:.0f}s ETA:{eta:.0f}s"
        )
        sys.stdout.flush()

    # Moitié des parties : new_net joue Blue
    for i in range(half):
        winner = _play_game(agent_new, agent_best)
        if winner == 'blue':
            wins += 1
        _afficher(i + 1, "Blue")

    # Moitié des parties : new_net joue Red
    for i in range(num_games - half):
        winner = _play_game(agent_best, agent_new)
        if winner == 'red':
            wins += 1
        _afficher(half + i + 1, "Red ")

    sys.stdout.write('\n')
    return wins / num_games


def evaluate_vs_random(
    net: HexNet,
    device: torch.device,
    num_games: int = 20,
    sims: int = 50,
) -> float:
    """
    Fait jouer `net` contre un joueur aléatoire.
    Retourne le win rate de net.
    """
    net.eval()
    agent = MCTSAgent(net, device=device, sims=sims, add_dirichlet=False)
    wins = 0

    for i in range(num_games):
        env = HexEnv()
        net_is_blue = (i % 2 == 0)
        move_count = 0

        while not env.is_terminal():
            if env.blue_to_play == net_is_blue:
                # Tour du réseau
                pi = agent.get_policy(env, move_count=999)
                move = int(pi.argmax())
            else:
                # Tour aléatoire
                legal = env.get_legal_moves()
                move = int(np.random.choice(legal))
            env.apply_move(move)
            move_count += 1

        winner = env.winner()
        if (net_is_blue and winner == 'blue') or (not net_is_blue and winner == 'red'):
            wins += 1

    return wins / num_games


if __name__ == "__main__":
    import sys
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    best_path = "checkpoints/best_model.pt"
    if not os.path.isfile(best_path):
        print(f"Aucun modèle trouvé à {best_path}. Lancez d'abord trainer.py.")
        sys.exit(1)

    net = HexNet().to(device)
    try:
        net.load_state_dict(torch.load(best_path, map_location=device))
    except RuntimeError:
        print(f"Checkpoint incompatible ({best_path}), architecture changée.")
        sys.exit(1)
    print(f"Modèle chargé : {best_path}")

    print("\nTest vs joueur aléatoire (20 parties)...")
    wr = evaluate_vs_random(net, device, num_games=20)
    print(f"Win rate vs random : {wr:.1%}")

#!/usr/bin/env python3
"""
ranking.py — Tournoi round-robin entre tous les checkpoints AlphaZero.

Chaque paire de modèles joue N parties (moitié Blue, moitié Red).
Résultats sauvés dans ranking_results.txt.

Usage :
  python ranking.py                          # défaut : 4 parties/matchup, 100 sims
  python ranking.py --games 10 --sims 200    # plus précis mais plus long
  python ranking.py --output classement.txt  # fichier de sortie personnalisé
"""

import os
import sys
import glob
import argparse
import time
from itertools import combinations

import torch
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from network import HexNet
from mcts_az import MCTSAgent
from config import NUM_CELLS


def load_model(path: str, device: torch.device) -> HexNet:
    """Charge un modèle HexNet depuis un fichier .pt."""
    net = HexNet().to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net


def play_game(agent_blue: MCTSAgent, agent_red: MCTSAgent) -> str:
    """Joue une partie. Retourne 'blue' ou 'red'."""
    env = HexEnv()
    while not env.is_terminal():
        if env.blue_to_play:
            pi = agent_blue.get_policy(env, move_count=999)
        else:
            pi = agent_red.get_policy(env, move_count=999)
        move = int(pi.argmax())
        env.apply_move(move)
    return env.winner()


def match(net_a: HexNet, net_b: HexNet, device: torch.device,
          num_games: int, sims: int) -> tuple[int, int]:
    """
    Fait jouer net_a vs net_b sur num_games parties (couleurs alternées).
    Retourne (wins_a, wins_b).
    """
    agent_a = MCTSAgent(net_a, device=device, sims=sims, add_dirichlet=False)
    agent_b = MCTSAgent(net_b, device=device, sims=sims, add_dirichlet=False)

    wins_a = 0
    wins_b = 0

    for i in range(num_games):
        a_is_blue = (i % 2 == 0)
        if a_is_blue:
            winner = play_game(agent_a, agent_b)
            if winner == 'blue':
                wins_a += 1
            else:
                wins_b += 1
        else:
            winner = play_game(agent_b, agent_a)
            if winner == 'red':
                wins_a += 1
            else:
                wins_b += 1

    return wins_a, wins_b


def extract_iter(path: str) -> str:
    """Extrait un nom lisible depuis le chemin du checkpoint."""
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    return name


def compute_elo(names: list[str], results: dict, k: float = 32.0,
                initial: float = 1000.0) -> dict[str, float]:
    """
    Calcule un classement Elo à partir des résultats des matchs.
    results[(a, b)] = (wins_a, wins_b)
    """
    elo = {name: initial for name in names}

    # Plusieurs passes pour convergence
    for _ in range(10):
        for (a, b), (wa, wb) in results.items():
            total = wa + wb
            if total == 0:
                continue
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400))
            eb = 1.0 - ea
            score_a = wa / total
            score_b = wb / total
            elo[a] += k * (score_a - ea)
            elo[b] += k * (score_b - eb)

    return elo


def main():
    parser = argparse.ArgumentParser(
        description="Tournoi round-robin entre checkpoints AlphaZero")
    parser.add_argument('--games', type=int, default=4,
                        help="Parties par matchup (défaut: 4)")
    parser.add_argument('--sims', type=int, default=100,
                        help="Simulations MCTS par coup (défaut: 100)")
    parser.add_argument('--output', type=str, default='ranking_results.txt',
                        help="Fichier de sortie (défaut: ranking_results.txt)")
    parser.add_argument('--checkpoints-dir', type=str,
                        default=os.path.join(_dir, 'checkpoints'),
                        help="Dossier des checkpoints")
    parser.add_argument('--device', type=str, default=None,
                        help="Device (cuda/cpu, défaut: auto)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Trouver tous les checkpoints
    pattern = os.path.join(args.checkpoints_dir, "*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Aucun checkpoint trouvé dans {args.checkpoints_dir}")
        sys.exit(1)

    names = [extract_iter(f) for f in files]
    n = len(files)
    total_matchups = n * (n - 1) // 2

    print(f"\n{n} modèles trouvés :")
    for name in names:
        print(f"  - {name}")
    print(f"\n{total_matchups} matchups à jouer ({args.games} parties chacun, "
          f"{args.sims} sims/coup)")
    print("=" * 60)

    # Charger tous les modèles
    print("\nChargement des modèles...")
    models = {}
    for f, name in zip(files, names):
        models[name] = load_model(f, device)
        print(f"  ✓ {name}")

    # Tournoi round-robin
    results = {}   # (name_a, name_b) -> (wins_a, wins_b)
    wins_total = {name: 0 for name in names}
    games_total = {name: 0 for name in names}
    matchup_count = 0

    print(f"\n{'='*60}")
    print("Début du tournoi")
    print(f"{'='*60}\n")
    t_start = time.time()

    for name_a, name_b in combinations(names, 2):
        matchup_count += 1
        print(f"[{matchup_count}/{total_matchups}] {name_a} vs {name_b} ... ",
              end="", flush=True)

        t0 = time.time()
        wa, wb = match(models[name_a], models[name_b], device,
                       args.games, args.sims)
        elapsed = time.time() - t0

        results[(name_a, name_b)] = (wa, wb)
        wins_total[name_a] += wa
        wins_total[name_b] += wb
        games_total[name_a] += wa + wb
        games_total[name_b] += wa + wb

        print(f"{wa}-{wb}  ({elapsed:.1f}s)")

    total_time = time.time() - t_start

    # Calcul Elo
    elo = compute_elo(names, results)

    # Classement par Elo
    ranking = sorted(names, key=lambda n: elo[n], reverse=True)

    # Affichage et sauvegarde
    output_lines = []
    output_lines.append(f"Classement AlphaZero — Tournoi round-robin")
    output_lines.append(f"  {n} modèles, {args.games} parties/matchup, "
                        f"{args.sims} sims/coup")
    output_lines.append(f"  Temps total : {total_time:.0f}s")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append(f"{'Rang':<6} {'Modèle':<25} {'Elo':>6} "
                        f"{'Victoires':>10} {'Parties':>8} {'Win%':>6}")
    output_lines.append("-" * 70)

    for rank, name in enumerate(ranking, 1):
        w = wins_total[name]
        g = games_total[name]
        pct = 100.0 * w / g if g > 0 else 0
        output_lines.append(
            f"{rank:<6} {name:<25} {elo[name]:>6.0f} "
            f"{w:>10} {g:>8} {pct:>5.1f}%"
        )

    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("Détail des matchups :")
    output_lines.append(f"{'Modèle A':<25} {'Modèle B':<25} {'Score':>10}")
    output_lines.append("-" * 70)

    for (a, b), (wa, wb) in sorted(results.items()):
        output_lines.append(f"{a:<25} {b:<25} {wa:>4} - {wb:<4}")

    report = "\n".join(output_lines)

    print(f"\n{report}")

    output_path = os.path.join(_dir, args.output)
    with open(output_path, "w") as f:
        f.write(report + "\n")
    print(f"\nRésultats sauvés dans {output_path}")


if __name__ == "__main__":
    main()

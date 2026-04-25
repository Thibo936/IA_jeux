#!/usr/bin/env python3
# play.py — Wrapper CLI AlphaZero (compatible protocole BOARD/PLAYER argv C++)
# Usage : python play.py BOARD PLAYER [time_s]
#   BOARD  : chaîne 121 chars ('.' / 'O' / '@')
#   PLAYER : 'O' (Blue) ou '@' (Red)
#   time_s : temps alloué en secondes (utilisé pour adapter les simulations)
# Sortie stdout : coup en notation "A1".."K11"
# Sortie stderr : ITERS:N WINRATE:f TIME:f

import sys
import os
import time

_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_dir, 'ia'), os.path.join(_dir, 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from hex_env import HexEnv
from network import HexNet
from mcts_az import MCTSAgent
from config import MCTS_SIMULATIONS, BEST_MODEL_FILE, NUM_CELLS


def compute_sims(time_limit: float) -> int:
    """Adapte le nombre de simulations au temps alloué."""
    if time_limit <= 0.5:
        return 200
    elif time_limit <= 1.0:
        return 400
    elif time_limit <= 2.0:
        return 800
    else:
        return 1600


def load_best_model(device: torch.device) -> HexNet | None:
    """Charge le meilleur modèle sauvegardé. Retourne None si absent."""
    model_path = os.path.join(_dir, BEST_MODEL_FILE)
    if not os.path.isfile(model_path):
        # Essai relatif au répertoire courant
        model_path = BEST_MODEL_FILE
    if not os.path.isfile(model_path):
        print(f"WARN: aucun modèle trouvé ({BEST_MODEL_FILE}), utilisation politique uniforme.",
              file=sys.stderr)
        return None
    net = HexNet().to(device)
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        print(f"WARN: checkpoint incompatible ({model_path}), utilisation politique uniforme.",
              file=sys.stderr)
        return None
    net.eval()
    return net


def main():
    if len(sys.argv) < 3:
        print("Usage: python play.py BOARD PLAYER [time_s]", file=sys.stderr)
        sys.exit(1)

    board_str  = sys.argv[1]
    player_str = sys.argv[2]
    time_limit = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.5

    if len(board_str) != NUM_CELLS:
        print(f"Erreur : BOARD doit faire {NUM_CELLS} chars, reçu {len(board_str)}", file=sys.stderr)
        sys.exit(1)
    if player_str not in ('O', '@'):
        print("Erreur : PLAYER doit être 'O' ou '@'", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    # ─── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Chargement du modèle ─────────────────────────────────────────────────
    net = load_best_model(device)

    # ─── Construction de l'état ───────────────────────────────────────────────
    env = HexEnv.from_string(board_str, player_str)

    # ─── MCTS ─────────────────────────────────────────────────────────────────
    sims = compute_sims(time_limit)
    agent = MCTSAgent(net, device=device, sims=sims, add_dirichlet=False)

    # move_count=999 → τ→0 (choix déterministe, argmax)
    pi, root = agent.get_policy(env, move_count=999, return_root=True)
    move = int(pi.argmax())

    t_elapsed = time.time() - t_start

    # ─── Statistiques (stderr) ────────────────────────────────────────────────
    # Winrate du nœud enfant sélectionné depuis le point de vue du joueur courant
    child = root.children.get(move)
    winrate = (-child.Q) if (child and child.N > 0) else 0.0  # -Q car Q est du côté enfant
    total_visits = sum(c.N for c in root.children.values())

    print(
        f"ITERS:{sims} VISITS:{total_visits} WINRATE:{winrate:.4f} TIME:{t_elapsed:.3f}",
        file=sys.stderr,
    )

    # ─── Sortie du coup (stdout) ──────────────────────────────────────────────
    print(env.pos_to_str(move))


if __name__ == "__main__":
    main()

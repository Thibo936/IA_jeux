#!/usr/bin/env python3
"""
versus.py — Duel entre deux modèles AlphaZero Hex 11×11

Utilisation :
  python alphazero/versus.py <modele1> <modele2> [nb_parties] [-v] [-t <s>] [-s <sims>]

  modele : chemin vers un fichier .pt, ou 'best' pour checkpoints/best_model.pt

Exemples :
  python alphazero/versus.py best checkpoints/model_iter_10.pt 20
  python alphazero/versus.py checkpoints/model_iter_20.pt checkpoints/model_iter_10.pt 10 -v
  python alphazero/versus.py best checkpoints/model_iter_5.pt 40 -t 2.0 -s 400
"""

import sys
import os
import argparse
import time

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from config import NUM_CELLS, BEST_MODEL_FILE, CHECKPOINT_DIR


# ─── Chargement d'un modèle ───────────────────────────────────────────────────

def _load_player(path: str, device, sims: int):
    """Charge un AlphaZeroPlayer depuis un chemin .pt."""
    import torch
    from network import HexNet
    from mcts_az import MCTSAgent

    if not os.path.isfile(path):
        # Essai relatif au répertoire alphazero/
        alt = os.path.join(_dir, path)
        if os.path.isfile(alt):
            path = alt
        else:
            print(f"ERREUR : modèle introuvable : {path}", file=sys.stderr)
            sys.exit(1)

    net = HexNet().to(device)
    try:
        net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        print(f"ERREUR : impossible de charger '{path}' : {e}", file=sys.stderr)
        sys.exit(1)
    net.eval()

    agent = MCTSAgent(net, device=device, sims=sims, add_dirichlet=False)
    return agent


def _resolve_path(name: str) -> str:
    """Résout 'best' en chemin absolu, sinon retourne tel quel."""
    if name.lower() in ('best', 'best_model', 'best_model.pt'):
        p = os.path.join(_dir, BEST_MODEL_FILE)
        if os.path.isfile(p):
            return p
        if os.path.isfile(BEST_MODEL_FILE):
            return BEST_MODEL_FILE
        print(f"ERREUR : best_model.pt introuvable ({BEST_MODEL_FILE})", file=sys.stderr)
        sys.exit(1)
    return name


def _short_name(path: str) -> str:
    """Nom court pour l'affichage : juste le basename sans extension."""
    return os.path.splitext(os.path.basename(path))[0]


# ─── Stats ────────────────────────────────────────────────────────────────────

def _stats_bar(wr_pct: float) -> str:
    filled = min(10, int(wr_pct / 10))
    bar    = '#' * filled + '.' * (10 - filled)
    label  = ("CRITIQUE"  if wr_pct < 10 else
              "DIFFICILE" if wr_pct < 40 else
              "EQUILIBRE" if wr_pct < 75 else "DOMINANT")
    return f"[{bar}] {label}"


# ─── Partie ───────────────────────────────────────────────────────────────────

def _play_game(agent1, agent2, ai1_is_blue: bool,
               sims: int, verbose: bool,
               name1: str, name2: str) -> int:
    """
    Joue une partie entre agent1 et agent2.
    Retourne 0 si agent1 gagne, 1 si agent2 gagne.
    """
    env  = HexEnv()
    turn = 0

    while not env.is_terminal():
        turn += 1
        cur_is_blue = env.blue_to_play
        cur_agent = agent1 if (cur_is_blue == ai1_is_blue) else agent2
        cur_name  = name1  if (cur_is_blue == ai1_is_blue) else name2
        player_label = "Blue(O)" if cur_is_blue else "Red(@)"

        t0 = time.time()
        pi, root = cur_agent.get_policy(env, move_count=999, return_root=True)
        move = int(pi.argmax())
        elapsed = time.time() - t0

        # Valider
        if move < 0 or move >= NUM_CELLS:
            print(f"  [{player_label}] {cur_name} → coup invalide ({move}), abandon.")
            return 1 if (cur_is_blue == ai1_is_blue) else 0
        r, c = divmod(move, 11)
        if env.blue[r, c] or env.red[r, c]:
            print(f"  [{player_label}] {cur_name} → case déjà occupée, abandon.")
            return 1 if (cur_is_blue == ai1_is_blue) else 0

        env.apply_move(move)

        if verbose:
            child  = root.children.get(move)
            wr_raw = (-child.Q) if (child and child.N > 0) else 0.0
            wr_pct = wr_raw * 100
            visits = sum(c.N for c in root.children.values())
            move_str = env.pos_to_str(move)
            print(f"\n  Coup #{turn} [{player_label}] {cur_name} joue: {move_str}")
            print(str(env))
            print(f"  O(Blue/Nord-Sud):{int(env.blue.sum())}  "
                  f"@(Red/Ouest-Est):{int(env.red.sum())}")
            print(f"  MCTS: {sims} sims, {visits} visites, "
                  f"winrate {wr_pct:.1f}% {_stats_bar(wr_pct)}, {elapsed:.2f}s")

    winner = env.winner()
    if (winner == 'blue' and ai1_is_blue) or (winner == 'red' and not ai1_is_blue):
        return 0
    return 1


# ─── Tournoi ──────────────────────────────────────────────────────────────────

def run_versus(path1: str, path2: str,
               n_games: int = 20,
               verbose: bool = False,
               time_s: float = 1.5,
               sims: int = 400):
    """Lance un duel entre deux modèles AlphaZero."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name1 = _short_name(path1)
    name2 = _short_name(path2)

    print(f"Chargement des modèles sur {device}...")
    agent1 = _load_player(path1, device, sims)
    agent2 = _load_player(path2, device, sims)
    print(f"  Modèle 1 : {name1}  ({path1})")
    print(f"  Modèle 2 : {name2}  ({path2})")
    print()
    print(f"Duel AlphaZero : {name1} vs {name2}")
    print(f"  {n_games} parties, {sims} simulations/coup, device={device}")
    print("  Blue(O) : connecte Nord → Sud")
    print("  Red (@) : connecte Ouest → Est")
    print("  Les couleurs alternent à chaque partie.")
    print("─" * 60)

    wins1 = wins2 = 0
    wins1_blue = wins1_red = wins2_blue = wins2_red = 0

    for g in range(n_games):
        ai1_is_blue = (g % 2 == 0)
        c1 = "Blue" if ai1_is_blue else "Red"
        c2 = "Red"  if ai1_is_blue else "Blue"

        if verbose:
            print(f"Partie {g + 1} ({name1}={c1}, {name2}={c2}):")

        result = _play_game(agent1, agent2, ai1_is_blue,
                            sims, verbose, name1, name2)

        if result == 0:
            wins1 += 1
            if ai1_is_blue: wins1_blue += 1
            else:           wins1_red  += 1
            if not verbose:
                print(f"Partie {g+1:3d} [{name1}={c1} {name2}={c2}]: {name1} gagne")
        else:
            wins2 += 1
            if ai1_is_blue: wins2_red  += 1
            else:           wins2_blue += 1
            if not verbose:
                print(f"Partie {g+1:3d} [{name1}={c1} {name2}={c2}]: {name2} gagne")

        sys.stdout.flush()

    pct1 = 100 * wins1 / n_games if n_games else 0
    pct2 = 100 * wins2 / n_games if n_games else 0
    print("═" * 60)
    print(f"Résultats ({n_games} parties, {sims} sims/coup) :")
    print(f"  {name1:<35s}: {wins1:3d} victoires ({pct1:.1f}%)"
          f"  [Blue:{wins1_blue}  Red:{wins1_red}]")
    print(f"  {name2:<35s}: {wins2:3d} victoires ({pct2:.1f}%)"
          f"  [Blue:{wins2_blue}  Red:{wins2_red}]")

    if pct1 >= 55:
        print(f"\n  → {name1} est MEILLEUR (≥ 55 %)")
    elif pct2 >= 55:
        print(f"\n  → {name2} est MEILLEUR (≥ 55 %)")
    else:
        print(f"\n  → Résultat NON CONCLUSIF (< 55 % d'écart)")

    return wins1, wins2


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    # Liste les checkpoints disponibles pour aider l'utilisateur
    ckpt_dir = os.path.join(_dir, CHECKPOINT_DIR)
    available = ""
    if os.path.isdir(ckpt_dir):
        pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
        if pts:
            available = "\nCheckpoints disponibles dans " + CHECKPOINT_DIR + " :\n"
            available += "\n".join(f"  {f}" for f in pts)

    parser = argparse.ArgumentParser(
        description="Duel entre deux modèles AlphaZero Hex 11×11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__ + available,
    )
    parser.add_argument('model1',
        help="Modèle 1 : chemin .pt ou 'best' pour best_model.pt")
    parser.add_argument('model2',
        help="Modèle 2 : chemin .pt ou 'best' pour best_model.pt")
    parser.add_argument('games', nargs='?', type=int, default=20,
        help="Nombre de parties (défaut: 20)")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Affichage du plateau après chaque coup")
    parser.add_argument('-t', '--time', type=float, default=1.5, dest='time_s',
        help="Temps par coup indicatif en secondes (non utilisé, sims contrôle la durée)")
    parser.add_argument('-s', '--sims', type=int, default=None,
        help="Simulations MCTS par coup (défaut: 400 si t<2s, 800 sinon)")
    args = parser.parse_args()

    path1 = _resolve_path(args.model1)
    path2 = _resolve_path(args.model2)

    if args.sims is not None:
        sims = args.sims
    else:
        sims = 800 if args.time_s >= 2.0 else 400

    run_versus(path1, path2,
               n_games=args.games,
               verbose=args.verbose,
               time_s=args.time_s,
               sims=sims)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
tournament.py — Tournoi Python entre deux IA Hex 11×11

Chaque IA peut être :
  - Un objet Python avec select_move(env: HexEnv, time_s: float) -> int
  - Une chaîne de commande externe (ex: "./TC_MG_mcts_hex")

Utilisation en ligne de commande (depuis la racine du projet) :
  python alphazero/tournament.py <ai1> <ai2> [nb_parties] [-v] [-t <s>]

  ai : 'alphabeta', 'random', 'alphazero', ou chemin d'un exécutable/commande

Exemples :
  python alphazero/tournament.py alphabeta random 20
  python alphazero/tournament.py alphabeta alphazero 10 -v -t 2.0
  python alphazero/tournament.py alphabeta ./TC_MG_mcts_hex 20 -t 1.5
"""

import sys
import os
import argparse
import subprocess
import shlex
import re
import time

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from config import NUM_CELLS, BEST_MODEL_FILE


# ─── Wrapper AlphaZero ────────────────────────────────────────────────────────

class AlphaZeroPlayer:
    """
    Wrapper MCTSAgent + HexNet pour utilisation dans le tournoi.
    Charge automatiquement checkpoints/best_model.pt.
    Interface : select_move(env, time_s) -> int
    """

    def __init__(self, model_path: str | None = None, device=None, sims: int = 200):
        import torch
        from network import HexNet
        from mcts_az import MCTSAgent

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        path = model_path or os.path.join(_dir, BEST_MODEL_FILE)
        if not os.path.isfile(path):
            path = BEST_MODEL_FILE
        if not os.path.isfile(path):
            print(f"WARN: modèle non trouvé ({path}), politique uniforme.", file=sys.stderr)
            net = None
        else:
            net = HexNet().to(device)
            try:
                net.load_state_dict(torch.load(path, map_location=device))
            except RuntimeError:
                print(f"WARN: checkpoint incompatible ({path}), architecture changée → politique uniforme.", file=sys.stderr)
                net = None
            if net is not None:
                net.eval()

        self._agent = MCTSAgent(net, device=device, sims=sims, add_dirichlet=False)
        self._sims = sims
        self.last_stats: dict = {}

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        t0 = time.time()
        pi, root = self._agent.get_policy(env, move_count=999, return_root=True)
        move = int(pi.argmax())
        elapsed = time.time() - t0

        child = root.children.get(move)
        winrate = (-child.Q) if (child and child.N > 0) else 0.0
        total_visits = sum(c.N for c in root.children.values())
        self.last_stats = {
            'iters':   self._sims,
            'visits':  total_visits,
            'winrate': winrate,
            'time':    elapsed,
        }
        return move


# ─── Appel d'une IA externe via subprocess ────────────────────────────────────

def _call_external(cmd: str, env: HexEnv, time_s: float):
    """
    Appelle une commande externe selon le protocole BOARD PLAYER [time_s].
    Retourne (move_index, stats_stderr_str) ou (-1, "") en cas d'erreur.
    """
    board_str   = env.to_string()
    player_char = 'O' if env.blue_to_play else '@'
    argv = shlex.split(cmd) + [board_str, player_char, f"{time_s:.2f}"]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=time_s + 10
        )
        move_str  = result.stdout.strip()
        stats_str = result.stderr.strip()
        if not move_str:
            return -1, ""
        return HexEnv.str_to_pos(move_str), stats_str
    except Exception as e:
        print(f"Erreur appel IA externe '{cmd}': {e}", file=sys.stderr)
        return -1, ""


# ─── Formatage des stats ──────────────────────────────────────────────────────

def _stats_bar(wr_pct: float) -> str:
    filled = min(10, int(wr_pct / 10))
    bar    = '#' * filled + '.' * (10 - filled)
    label  = ("CRITIQUE"  if wr_pct < 10 else
              "DIFFICILE" if wr_pct < 40 else
              "EQUILIBRE" if wr_pct < 75 else "DOMINANT")
    return f"[{bar}] {label}"


def _format_stats(ai) -> str:
    """Retourne une ligne de stats pour une IA Python, ou '' si absentes."""
    if not (hasattr(ai, 'last_stats') and ai.last_stats):
        return ""
    s = ai.last_stats
    if 'score' in s:
        return (f"AlphaBeta: score={s['score']:+d}, "
                f"{s['nodes']//1000}k nœuds, prof.{s['depth']}")
    if 'winrate' in s:
        wr = s['winrate'] * 100
        return (f"MCTS: {s['iters']} sims, {s['visits']} visites, "
                f"winrate {wr:.1f}% {_stats_bar(wr)}, {s['time']:.2f}s")
    return ""


def _format_external_stats(stats_str: str) -> str:
    """Parse et formate les stats issues du stderr d'une IA externe."""
    if not stats_str:
        return ""
    m = re.match(r'SCORE:(-?\d+)\s+NODES:(\d+)\s+DEPTH:(\d+)', stats_str)
    if m:
        score, nodes, depth = int(m[1]), int(m[2]), int(m[3])
        return f"AlphaBeta: score={score:+d}, {nodes//1000}k nœuds, prof.{depth}"
    m = re.match(r'ITERS:(\d+)\s+VISITS:(\d+)\s+WINRATE:([0-9.]+)\s+TIME:([0-9.]+)', stats_str)
    if m:
        iters, visits = int(m[1]), int(m[2])
        wr = float(m[3]) * 100
        return (f"MCTS: {iters} sims, {visits} visites, "
                f"winrate {wr:.1f}% {_stats_bar(wr)}, {float(m[4]):.2f}s")
    return stats_str


# ─── Affichage verbose ────────────────────────────────────────────────────────

def _print_verbose(env: HexEnv, move: int, turn: int,
                   player_label: str, ai_name: str, stats: str) -> None:
    move_str = env.pos_to_str(move)
    print(f"\n  Coup #{turn} [{player_label}] {ai_name} joue: {move_str}")
    print(str(env))
    blue_count = int(env.blue.sum())
    red_count  = int(env.red.sum())
    print(f"  O(Blue/Nord-Sud):{blue_count}  @(Red/Ouest-Est):{red_count}")
    if stats:
        print(f"  {stats}")
    print()


# ─── Partie ───────────────────────────────────────────────────────────────────

def _play_game(ai1, ai1_is_blue: bool, ai2,
               time_s: float, verbose: bool,
               ai1_name: str, ai2_name: str) -> int:
    """
    Joue une partie entre ai1 et ai2.
    ai1_is_blue=True → ai1 joue Blue (Nord-Sud).
    Retourne 0 si ai1 gagne, 1 si ai2 gagne, -1 si erreur.
    """
    env  = HexEnv()
    turn = 0

    while not env.is_terminal():
        turn += 1
        cur_is_blue = env.blue_to_play
        cur_ai   = ai1   if (cur_is_blue == ai1_is_blue) else ai2
        cur_name = ai1_name if (cur_is_blue == ai1_is_blue) else ai2_name
        player_label = "Blue(O)" if cur_is_blue else "Red(@)"

        # ── Obtenir le coup ───────────────────────────────────────────────────
        ext_stats = ""
        if isinstance(cur_ai, str):
            move, ext_stats = _call_external(cur_ai, env, time_s)
        else:
            move = cur_ai.select_move(env, time_s)

        # ── Valider ───────────────────────────────────────────────────────────
        if move < 0 or move >= NUM_CELLS:
            print(f"  [{player_label}] {cur_name} → coup invalide ({move}), abandon.")
            return 1 if (cur_is_blue == ai1_is_blue) else 0
        r, c = divmod(move, 11)
        if env.blue[r, c] or env.red[r, c]:
            print(f"  [{player_label}] {cur_name} → case déjà occupée, abandon.")
            return 1 if (cur_is_blue == ai1_is_blue) else 0

        # ── Appliquer ─────────────────────────────────────────────────────────
        env.apply_move(move)

        # ── Affichage verbose ─────────────────────────────────────────────────
        if verbose:
            stats = (_format_external_stats(ext_stats) if isinstance(cur_ai, str)
                     else _format_stats(cur_ai))
            _print_verbose(env, move, turn, player_label, cur_name, stats)

    # ── Résultat ──────────────────────────────────────────────────────────────
    winner = env.winner()
    if (winner == 'blue' and ai1_is_blue) or (winner == 'red' and not ai1_is_blue):
        return 0
    return 1


# ─── Tournoi ──────────────────────────────────────────────────────────────────

def run_tournament(ai1, ai2,
                   n_games: int = 20,
                   verbose: bool = False,
                   time_s: float = 1.5,
                   ai1_name: str | None = None,
                   ai2_name: str | None = None):
    """
    Lance un tournoi entre ai1 et ai2.
    Les couleurs alternent à chaque partie (partie paire → ai1=Blue).
    """
    if ai1_name is None:
        ai1_name = ai1 if isinstance(ai1, str) else type(ai1).__name__
    if ai2_name is None:
        ai2_name = ai2 if isinstance(ai2, str) else type(ai2).__name__

    print(f"Tournoi Hex 11×11 : {ai1_name} vs {ai2_name} "
          f"({n_games} parties, {time_s:.1f}s/coup)")
    print("Blue(O) : connecte Nord → Sud")
    print("Red (@) : connecte Ouest → Est")
    print("Les couleurs alternent à chaque partie.")
    print("─" * 60)

    wins1 = wins2 = errors = 0
    wins1_blue = wins1_red = wins2_blue = wins2_red = 0

    for g in range(n_games):
        ai1_is_blue = (g % 2 == 0)
        c1 = "Blue" if ai1_is_blue else "Red"
        c2 = "Red"  if ai1_is_blue else "Blue"

        if verbose:
            print(f"Partie {g + 1} ({ai1_name}={c1}, {ai2_name}={c2}):")

        result = _play_game(ai1, ai1_is_blue, ai2,
                            time_s, verbose, ai1_name, ai2_name)

        if result == 0:
            wins1 += 1
            if ai1_is_blue:
                wins1_blue += 1
            else:
                wins1_red += 1
            if not verbose:
                print(f"Partie {g+1:3d} [{ai1_name}={c1} {ai2_name}={c2}]: "
                      f"{ai1_name} gagne")
        elif result == 1:
            wins2 += 1
            if ai1_is_blue:
                wins2_red += 1
            else:
                wins2_blue += 1
            if not verbose:
                print(f"Partie {g+1:3d} [{ai1_name}={c1} {ai2_name}={c2}]: "
                      f"{ai2_name} gagne")
        else:
            errors += 1
            if not verbose:
                print(f"Partie {g+1:3d} [{ai1_name}={c1} {ai2_name}={c2}]: erreur")

        sys.stdout.flush()

    pct1 = 100 * wins1 / n_games if n_games else 0
    pct2 = 100 * wins2 / n_games if n_games else 0
    print("═" * 60)
    print(f"Résultats ({n_games} parties) :")
    print(f"  {ai1_name:<30s}: {wins1} victoires ({pct1:.1f}%)"
          f"  [Blue:{wins1_blue}  Red:{wins1_red}]")
    print(f"  {ai2_name:<30s}: {wins2} victoires ({pct2:.1f}%)"
          f"  [Blue:{wins2_blue}  Red:{wins2_red}]")
    if errors:
        print(f"  Erreurs : {errors}")

    return wins1, wins2, errors


# ─── Résolution des noms d'IA ─────────────────────────────────────────────────

def _resolve_ai(name: str, time_s: float):
    """
    'alphabeta' → AlphaBetaPlayer()
    'random'    → RandomPlayer()
    'alphazero' → AlphaZeroPlayer()
    autre       → traité comme commande externe (chaîne brute)
    """
    n = name.lower()
    if n == 'alphabeta':
        from alphabeta import AlphaBetaPlayer
        return AlphaBetaPlayer(), 'AlphaBeta'
    if n == 'random':
        from random_player import RandomPlayer
        return RandomPlayer(), 'Random'
    if n == 'alphazero':
        sims = 800 if time_s >= 2.0 else 400 if time_s >= 1.0 else 200
        return AlphaZeroPlayer(sims=sims), 'AlphaZero'
    return name, name


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tournoi Hex 11×11 en Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('ai1',
        help="IA 1 : 'alphabeta', 'random', 'alphazero', ou commande externe")
    parser.add_argument('ai2',
        help="IA 2 : 'alphabeta', 'random', 'alphazero', ou commande externe")
    parser.add_argument('games', nargs='?', type=int, default=20,
        help="Nombre de parties (défaut: 20)")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Affichage du plateau après chaque coup")
    parser.add_argument('-t', '--time', type=float, default=1.5, dest='time_s',
        help="Temps par coup en secondes (défaut: 1.5)")
    args = parser.parse_args()

    ai1, name1 = _resolve_ai(args.ai1, args.time_s)
    ai2, name2 = _resolve_ai(args.ai2, args.time_s)

    run_tournament(ai1, ai2,
                   n_games=args.games,
                   verbose=args.verbose,
                   time_s=args.time_s,
                   ai1_name=name1,
                   ai2_name=name2)


if __name__ == '__main__':
    main()

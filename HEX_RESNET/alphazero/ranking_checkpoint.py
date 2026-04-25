#!/usr/bin/env python3
"""
ranking_checkpoint.py — Tournoi round-robin entre tous les checkpoints AlphaZero.

Scanne automatiquement le dossier `checkpoints/` et fait jouer chaque modèle
contre tous les autres pour mesurer la progression de l'entraînement.

Usage :
  python ranking_checkpoint.py                    # défaut : 100 parties, 400 sims
  python ranking_checkpoint.py --games 20         # test rapide
  python ranking_checkpoint.py --sims 200         # moins de calcul
  python ranking_checkpoint.py --model-dir checkpoints
  python ranking_checkpoint.py --no-html
"""

import os
import sys
import re
import argparse
import csv
import time
import json
import io
import contextlib
from itertools import combinations
from datetime import datetime

import torch
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_dir, 'ia'), os.path.join(_dir, 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, CHECKPOINT_DIR


# ─── Chargement des modèles ───────────────────────────────────────────────────

def _load_agent(path: str, device: torch.device, sims: int):
    from network import HexNet
    from mcts_az import MCTSAgent

    net = HexNet().to(device)
    try:
        net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        raise ValueError(f"architecture incompatible : {e}") from e
    net.eval()
    return MCTSAgent(net, device=device, sims=sims, add_dirichlet=False)


def _sort_key(filename: str) -> tuple:
    """Trie model_iter_NNNN avant best_model, dans l'ordre numérique."""
    m = re.match(r'model_iter_(\d+)', filename)
    if m:
        return (0, int(m.group(1)))
    return (1, filename)


def discover_models(model_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(model_dir):
        print(f"ERREUR : dossier introuvable : {model_dir}", file=sys.stderr)
        sys.exit(1)
    pts = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not pts:
        print(f"ERREUR : aucun fichier .pt trouvé dans {model_dir}", file=sys.stderr)
        sys.exit(1)
    pts.sort(key=_sort_key)
    return [(os.path.splitext(f)[0], os.path.join(model_dir, f)) for f in pts]


# ─── Partie ───────────────────────────────────────────────────────────────────

def play_game(agent_blue, agent_red) -> dict:
    env = HexEnv()
    times_blue = []
    times_red  = []

    while not env.is_terminal():
        cur_agent = agent_blue if env.blue_to_play else agent_red
        t0 = time.time()
        with contextlib.redirect_stderr(io.StringIO()):
            pi, _ = cur_agent.get_policy(env, move_count=999, return_root=True)
        move = int(pi.argmax())
        elapsed = time.time() - t0

        if env.blue_to_play:
            times_blue.append(elapsed)
        else:
            times_red.append(elapsed)

        if move < 0 or move >= NUM_CELLS:
            winner = 'red' if env.blue_to_play else 'blue'
            return {'winner': winner, 'total_moves': 0,
                    'avg_time_blue': 0.0, 'avg_time_red': 0.0}
        r, c = divmod(move, 11)
        if env.blue[r, c] or env.red[r, c]:
            winner = 'red' if env.blue_to_play else 'blue'
            return {'winner': winner, 'total_moves': 0,
                    'avg_time_blue': 0.0, 'avg_time_red': 0.0}

        env.apply_move(move)

    total_moves = int(env.blue.sum()) + int(env.red.sum())
    return {
        'winner':        env.winner(),
        'total_moves':   total_moves,
        'avg_time_blue': float(np.mean(times_blue)) if times_blue else 0.0,
        'avg_time_red':  float(np.mean(times_red))  if times_red  else 0.0,
    }


def match(agent_a, agent_b, name_a: str, name_b: str, num_games: int) -> dict:
    wins_a = wins_b = 0
    times_a_all = []
    times_b_all = []
    moves_all   = []
    games       = []

    for i in range(num_games):
        a_is_blue  = (i % 2 == 0)
        blue_agent = agent_a if a_is_blue else agent_b
        red_agent  = agent_b if a_is_blue else agent_a

        g = play_game(blue_agent, red_agent)
        g['a_is_blue']   = a_is_blue
        g['blue_name']   = name_a if a_is_blue else name_b
        g['red_name']    = name_b if a_is_blue else name_a
        g['winner_name'] = g['blue_name'] if g['winner'] == 'blue' else g['red_name']

        if a_is_blue:
            if g['winner'] == 'blue': wins_a += 1
            else:                     wins_b += 1
            times_a_all.append(g['avg_time_blue'])
            times_b_all.append(g['avg_time_red'])
        else:
            if g['winner'] == 'red':  wins_a += 1
            else:                     wins_b += 1
            times_a_all.append(g['avg_time_red'])
            times_b_all.append(g['avg_time_blue'])

        moves_all.append(g['total_moves'])
        games.append(g)

    return {
        'name_a':     name_a,
        'name_b':     name_b,
        'wins_a':     wins_a,
        'wins_b':     wins_b,
        'games':      games,
        'avg_time_a': float(np.mean(times_a_all)) if times_a_all else 0.0,
        'avg_time_b': float(np.mean(times_b_all)) if times_b_all else 0.0,
        'avg_moves':  float(np.mean(moves_all))   if moves_all   else 0.0,
    }


# ─── Calcul Elo ───────────────────────────────────────────────────────────────

def compute_elo(names: list[str], results: dict,
                k: float = 32.0, initial: float = 1000.0) -> dict[str, float]:
    elo = {name: initial for name in names}
    for _ in range(10):
        for (a, b), (wa, wb) in results.items():
            total = wa + wb
            if total == 0:
                continue
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400))
            elo[a] += k * (wa / total - ea)
            elo[b] += k * (wb / total - (1.0 - ea))
    return elo


# ─── Génération HTML ──────────────────────────────────────────────────────────

def generate_html_report(all_stats: dict, output_path: str):
    names       = all_stats['names']
    elo         = all_stats['elo']
    wins_total  = all_stats['wins_total']
    games_total = all_stats['games_total']
    total_time  = all_stats['total_time']
    n           = len(names)
    total_matchups = n * (n - 1) // 2

    ranking  = sorted(names, key=lambda x: elo[x], reverse=True)
    elo_data = [elo[name] for name in ranking]

    # Couleur dégradée vert→rouge selon le rang
    def rank_color(i, total):
        t = i / max(total - 1, 1)
        r = int(231 * t + 46 * (1 - t))
        g = int(76  * t + 204 * (1 - t))
        b = int(60  * t + 113 * (1 - t))
        return f'rgb({r},{g},{b})'

    bar_colors = [rank_color(i, len(ranking)) for i in range(len(ranking))]

    rows_html = ''
    for rank, name in enumerate(ranking, 1):
        rank_class = f"rank-{rank}" if rank <= 3 else ""
        w   = wins_total[name]
        g   = games_total[name]
        pct = 100.0 * w / g if g > 0 else 0
        rows_html += f"""<tr>
            <td class="{rank_class}">#{rank}</td>
            <td>{name}</td>
            <td>{elo[name]:.0f}</td>
            <td>{w}/{g}</td>
            <td>{pct:.1f}%</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Progression AlphaZero — Checkpoints Hex 11×11</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; max-width: 960px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #e74c3c; margin-bottom: 0.4rem; font-size: 1.8rem; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.9rem; }}
        .stats-row {{ display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }}
        .stat-box {{ background: #1a2634; border-radius: 10px; padding: 0.8rem 1.5rem; text-align: center; }}
        .stat-box .value {{ font-size: 1.6rem; font-weight: bold; color: #e74c3c; }}
        .stat-box .label {{ font-size: 0.75rem; color: #888; margin-top: 0.2rem; }}
        .card {{ background: #1a2634; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 1.5rem; }}
        .card h2 {{ color: #e74c3c; margin-bottom: 1rem; font-size: 1.1rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.5rem; }}
        canvas {{ max-height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
        th, td {{ padding: 0.55rem 0.8rem; text-align: left; border-bottom: 1px solid #2a3a4a; font-size: 0.85rem; }}
        th {{ color: #e74c3c; font-weight: 600; }}
        tr:hover {{ background: #243447; }}
        .rank-1 {{ color: #ffd700; font-weight: bold; }}
        .rank-2 {{ color: #c0c0c0; font-weight: bold; }}
        .rank-3 {{ color: #cd7f32; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Progression AlphaZero — Checkpoints</h1>
    <p class="subtitle">
        Tournoi round-robin · {n} modèles · {total_matchups} matchups ·
        {all_stats['games_per_matchup']} parties/matchup · {all_stats['sims']} sims/coup ·
        Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}
    </p>

    <div class="stats-row">
        <div class="stat-box"><div class="value">{n}</div><div class="label">Checkpoints</div></div>
        <div class="stat-box"><div class="value">{sum(games_total.values()) // 2}</div><div class="label">Parties totales</div></div>
        <div class="stat-box"><div class="value">{total_time:.0f}s</div><div class="label">Temps total</div></div>
        <div class="stat-box"><div class="value">{all_stats['sims']}</div><div class="label">Sims/coup</div></div>
    </div>

    <div class="card">
        <h2>Score Elo par checkpoint (du plus fort au plus faible)</h2>
        <canvas id="eloChart"></canvas>
    </div>

    <div class="card">
        <h2>Classement détaillé</h2>
        <table>
            <thead>
                <tr><th>Rang</th><th>Modèle</th><th>Elo</th><th>Victoires/Parties</th><th>Win%</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <script>
        const names = {json.dumps(ranking)};
        const eloData = {json.dumps(elo_data)};
        const barColors = {json.dumps(bar_colors)};

        new Chart(document.getElementById('eloChart'), {{
            type: 'bar',
            data: {{
                labels: names,
                datasets: [{{
                    data: eloData,
                    backgroundColor: barColors,
                    borderRadius: 6,
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#2a3a4a' }}, ticks: {{ color: '#aaa' }} }},
                    y: {{ grid: {{ display: false }}, ticks: {{ color: '#e0e0e0', font: {{ size: 11 }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Rapport HTML : {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tournoi round-robin entre tous les checkpoints AlphaZero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model-dir', type=str,
                        default=os.path.join(_dir, CHECKPOINT_DIR),
                        help=f"Dossier de checkpoints (défaut: {CHECKPOINT_DIR}/)")
    parser.add_argument('--games', type=int, default=100,
                        help="Parties par matchup (défaut: 100)")
    parser.add_argument('--sims', type=int, default=400,
                        help="Simulations MCTS par coup (défaut: 400)")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_dir, 'rank'),
                        help="Dossier de sortie (défaut: rank/)")
    parser.add_argument('--output', type=str, default=None,
                        help="Fichier HTML explicite")
    parser.add_argument('--no-html', action='store_true',
                        help="Sortie texte uniquement")
    parser.add_argument('--no-csv', action='store_true',
                        help="Ne pas écrire le CSV")
    parser.add_argument('--device', type=str, default=None,
                        help="Device (cuda/cpu, défaut: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = discover_models(args.model_dir)
    if len(entries) < 2:
        print("ERREUR : il faut au moins 2 modèles pour un tournoi.", file=sys.stderr)
        sys.exit(1)

    n = len(entries)
    print(f"Chargement de {n} modèles sur {device}...")
    agents = {}
    skipped = []
    for name, path in entries:
        try:
            agents[name] = _load_agent(path, device, args.sims)
            print(f"  OK   {name}  ({os.path.basename(path)})")
        except ValueError as e:
            print(f"  SKIP {name} : {e}", file=sys.stderr)
            skipped.append(name)

    valid_names = [name for name, _ in entries if name not in skipped]
    if len(valid_names) < 2:
        print("ERREUR : pas assez de modèles compatibles.", file=sys.stderr)
        sys.exit(1)

    n = len(valid_names)
    total_matchups = n * (n - 1) // 2
    print(f"\nTournoi : {n} modèles, {total_matchups} matchups, "
          f"{args.games} parties/matchup, {args.sims} sims/coup, device={device}")
    print("─" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    run_stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    if args.output:
        html_path = args.output
        base = os.path.splitext(html_path)[0]
    else:
        base = os.path.join(args.output_dir, f'ranking_checkpoint_{run_stamp}')
        html_path = base + '.html'
    csv_path = base + '.csv'

    csv_file   = None
    csv_writer = None
    if not args.no_csv:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'run_timestamp', 'name_a', 'name_b', 'game_index',
            'a_is_blue', 'blue_name', 'red_name', 'winner',
            'winner_name', 'total_moves', 'avg_time_blue', 'avg_time_red',
        ])

    results     = {}
    wins_total  = {name: 0   for name in valid_names}
    games_total = {name: 0   for name in valid_names}
    times_total = {name: 0.0 for name in valid_names}
    moves_total = {name: 0   for name in valid_names}
    matchup_count = 0
    t_start = time.time()

    for name_a, name_b in combinations(valid_names, 2):
        matchup_count += 1
        sys.stdout.write(
            f"\r  [{matchup_count}/{total_matchups}] {name_a} vs {name_b} ..."
        )
        sys.stdout.flush()

        mr = match(agents[name_a], agents[name_b], name_a, name_b, args.games)

        if csv_writer is not None:
            for i, g in enumerate(mr['games']):
                csv_writer.writerow([
                    run_stamp, name_a, name_b, i + 1,
                    int(g['a_is_blue']), g['blue_name'], g['red_name'],
                    g['winner'], g['winner_name'], g['total_moves'],
                    f"{g['avg_time_blue']:.6f}", f"{g['avg_time_red']:.6f}",
                ])
            csv_file.flush()

        results[(name_a, name_b)] = (mr['wins_a'], mr['wins_b'])
        wins_total[name_a]  += mr['wins_a']
        wins_total[name_b]  += mr['wins_b']
        games_total[name_a] += args.games
        games_total[name_b] += args.games
        times_total[name_a] += mr['avg_time_a'] * args.games
        times_total[name_b] += mr['avg_time_b'] * args.games
        moves_total[name_a] += mr['avg_moves']  * args.games
        moves_total[name_b] += mr['avg_moves']  * args.games

        wa, wb = mr['wins_a'], mr['wins_b']
        sys.stdout.write(
            f"\r  [{matchup_count}/{total_matchups}] "
            f"{name_a} {wa}-{wb} {name_b}          \n"
        )
        sys.stdout.flush()

    total_time = time.time() - t_start

    if csv_file is not None:
        csv_file.close()
        print(f"CSV : {csv_path}")

    elo     = compute_elo(valid_names, results)
    ranking = sorted(valid_names, key=lambda x: elo[x], reverse=True)

    print(f"\n{'Rang':<5} {'Modèle':<40} {'Elo':>5} {'W':>4}/{'':<4} {'Win%':>5}")
    print("─" * 62)
    for rank, name in enumerate(ranking, 1):
        w   = wins_total[name]
        g   = games_total[name]
        pct = 100.0 * w / g if g > 0 else 0
        marker = " ← MEILLEUR" if rank == 1 else ""
        print(f"#{rank:<4} {name:<40} {elo[name]:>5.0f} "
              f"{w:>4}/{g:<4} {pct:>4.0f}%{marker}")
    print(f"\nTemps total : {total_time:.0f}s")

    if not args.no_html:
        all_stats = {
            'names':             valid_names,
            'elo':               elo,
            'wins_total':        wins_total,
            'games_total':       games_total,
            'times_total':       times_total,
            'moves_total':       moves_total,
            'results':           results,
            'total_time':        total_time,
            'games_per_matchup': args.games,
            'sims':              args.sims,
        }
        generate_html_report(all_stats, html_path)


if __name__ == '__main__':
    main()

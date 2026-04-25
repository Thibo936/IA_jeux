#!/usr/bin/env python3
"""
ranking_az.py — Tournoi round-robin entre tous les modèles AlphaZero dans model/

Scanne automatiquement le dossier `model/` pour trouver tous les fichiers .pt,
puis fait jouer chaque modèle contre tous les autres.

Usage :
  python ranking_az.py                          # défaut : 20 parties/matchup
  python ranking_az.py --games 40               # plus de parties
  python ranking_az.py --sims 400               # plus de simulations MCTS
  python ranking_az.py --model-dir model        # dossier de modèles (défaut: model/)
  python ranking_az.py --no-html                # sortie texte uniquement
  python ranking_az.py --add best               # ajouter best_model.pt au tournoi
"""

import os
import sys
import argparse
import csv
import time
import json
import io
import contextlib
from itertools import combinations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_dir, 'ia'), os.path.join(_dir, 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BEST_MODEL_FILE, CHECKPOINT_DIR

TIME_PER_MOVE = 1.0


# ─── Chargement des modèles ───────────────────────────────────────────────────

def _load_agent(path: str, device: torch.device, sims: int):
    """Charge un MCTSAgent depuis un fichier .pt. Lève ValueError si incompatible."""
    from network import HexNet
    from mcts_az import MCTSAgent

    net = HexNet().to(device)
    try:
        net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        raise ValueError(f"architecture incompatible : {e}") from e
    net.eval()
    return MCTSAgent(net, device=device, sims=sims, add_dirichlet=False)


def discover_models(model_dir: str) -> list[tuple[str, str]]:
    """
    Scanne model_dir et retourne une liste de (nom_court, chemin_absolu)
    triée par nom de fichier.
    """
    if not os.path.isdir(model_dir):
        print(f"ERREUR : dossier introuvable : {model_dir}", file=sys.stderr)
        sys.exit(1)
    pts = sorted(f for f in os.listdir(model_dir) if f.endswith('.pt'))
    if not pts:
        print(f"ERREUR : aucun fichier .pt trouvé dans {model_dir}", file=sys.stderr)
        sys.exit(1)
    return [(os.path.splitext(f)[0], os.path.join(model_dir, f)) for f in pts]


# ─── Partie ───────────────────────────────────────────────────────────────────

def play_game(agent_blue, agent_red) -> dict:
    """Joue une partie. Retourne un dict avec winner, total_moves, times."""
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
    """Fait jouer agent_a vs agent_b sur num_games parties (couleurs alternées)."""
    wins_a = wins_b = 0
    times_a_all = []
    times_b_all = []
    moves_all   = []
    games       = []

    for i in range(num_games):
        a_is_blue = (i % 2 == 0)
        blue_agent  = agent_a if a_is_blue else agent_b
        red_agent   = agent_b if a_is_blue else agent_a

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
        'name_a':    name_a,
        'name_b':    name_b,
        'wins_a':    wins_a,
        'wins_b':    wins_b,
        'games':     games,
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

def _stats_bar(wr_pct: float) -> str:
    filled = min(10, int(wr_pct / 10))
    bar    = '#' * filled + '.' * (10 - filled)
    label  = ("CRITIQUE"  if wr_pct < 10 else
              "DIFFICILE" if wr_pct < 40 else
              "EQUILIBRE" if wr_pct < 75 else "DOMINANT")
    return f"[{bar}] {label}"


def generate_html_report(all_stats: dict, output_path: str):
    names       = all_stats['names']
    elo         = all_stats['elo']
    wins_total  = all_stats['wins_total']
    games_total = all_stats['games_total']
    times_total = all_stats['times_total']
    moves_total = all_stats['moves_total']
    results     = all_stats['results']
    total_time  = all_stats['total_time']
    n           = len(names)
    total_matchups = n * (n - 1) // 2

    ranking   = sorted(names, key=lambda x: elo[x], reverse=True)
    elo_data  = [elo[name] for name in ranking]
    win_data  = [100.0 * wins_total[name] / games_total[name]
                 if games_total[name] > 0 else 0 for name in ranking]
    time_data = [times_total[name] / games_total[name]
                 if games_total[name] > 0 else 0 for name in ranking]
    moves_data= [moves_total[name] / games_total[name]
                 if games_total[name] > 0 else 0 for name in ranking]

    # Matrice heatmap
    matrix = []
    for a in ranking:
        row = []
        for b in ranking:
            if a == b:
                row.append(None)
                continue
            key = (a, b) if (a, b) in results else (b, a)
            if key in results:
                wa, wb = results[key]
                total  = wa + wb
                if total == 0:
                    row.append(0.5)
                elif key == (a, b):
                    row.append(wa / total)
                else:
                    row.append(wb / total)
            else:
                row.append(0.5)
        matrix.append(row)

    # Palette de couleurs pour les modèles
    palette = [
        '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
        '#1abc9c', '#f1c40f', '#e91e63', '#00bcd4', '#95a5a6',
        '#8e44ad', '#16a085', '#d35400', '#27ae60',
    ]
    bar_colors = [palette[i % len(palette)] for i in range(len(ranking))]

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classement AlphaZero Hex 11×11</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; }}
        h1 {{ text-align: center; color: #e74c3c; margin-bottom: 0.5rem; font-size: 2rem; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{ background: #1a2634; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .card h2 {{ color: #e74c3c; margin-bottom: 1rem; font-size: 1.2rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #2a3a4a; font-size: 0.85rem; word-break: break-all; }}
        th {{ color: #e74c3c; font-weight: 600; }}
        tr:hover {{ background: #243447; }}
        .rank-1 {{ color: #ffd700; font-weight: bold; }}
        .rank-2 {{ color: #c0c0c0; font-weight: bold; }}
        .rank-3 {{ color: #cd7f32; font-weight: bold; }}
        .heatmap {{ display: grid; gap: 2px; margin-top: 1rem; overflow-x: auto; }}
        .heatmap-cell {{ padding: 0.4rem; text-align: center; font-size: 0.7rem; border-radius: 3px; }}
        .heatmap-header {{ font-weight: 600; color: #e74c3c; font-size: 0.65rem; word-break: break-all; }}
        .stats-row {{ display: flex; justify-content: space-around; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }}
        .stat-box {{ background: #1a2634; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; min-width: 150px; }}
        .stat-box .value {{ font-size: 1.8rem; font-weight: bold; color: #e74c3c; }}
        .stat-box .label {{ font-size: 0.8rem; color: #888; margin-top: 0.3rem; }}
        canvas {{ max-height: 350px; }}
        .full-width {{ grid-column: 1 / -1; }}
    </style>
</head>
<body>
    <h1>AlphaZero — Classement des modèles Hex 11×11</h1>
    <p class="subtitle">Tournoi round-robin • {n} modèles • {total_matchups} matchups • {all_stats['games_per_matchup']} parties/matchup • {all_stats['sims']} sims/coup • Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>

    <div class="stats-row">
        <div class="stat-box"><div class="value">{n}</div><div class="label">Modèles</div></div>
        <div class="stat-box"><div class="value">{total_matchups}</div><div class="label">Matchups</div></div>
        <div class="stat-box"><div class="value">{sum(games_total.values()) // 2}</div><div class="label">Parties totales</div></div>
        <div class="stat-box"><div class="value">{total_time:.0f}s</div><div class="label">Temps total</div></div>
        <div class="stat-box"><div class="value">{all_stats['sims']}</div><div class="label">Sims/coup</div></div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Score Elo</h2>
            <canvas id="eloChart"></canvas>
        </div>
        <div class="card">
            <h2>Taux de Victoire (%)</h2>
            <canvas id="winChart"></canvas>
        </div>
        <div class="card">
            <h2>Temps moyen par coup (s)</h2>
            <canvas id="timeChart"></canvas>
        </div>
        <div class="card">
            <h2>Durée moyenne des parties (coups)</h2>
            <canvas id="movesChart"></canvas>
        </div>
        <div class="card full-width">
            <h2>Matrice des confrontations (taux de victoire en ligne vs colonne)</h2>
            <div id="heatmap" class="heatmap"></div>
        </div>
        <div class="card full-width">
            <h2>Classement détaillé</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rang</th><th>Modèle</th><th>Elo</th>
                        <th>Victoires</th><th>Parties</th><th>Win%</th>
                        <th>Temps moy./coup</th><th>Coups moy./partie</th>
                    </tr>
                </thead>
                <tbody>
"""

    for rank, name in enumerate(ranking, 1):
        rank_class = f"rank-{rank}" if rank <= 3 else ""
        w  = wins_total[name]
        g  = games_total[name]
        pct       = 100.0 * w / g if g > 0 else 0
        avg_time  = times_total[name] / g if g > 0 else 0
        avg_moves = moves_total[name] / g if g > 0 else 0
        html += f"""                    <tr>
                        <td class="{rank_class}">#{rank}</td>
                        <td>{name}</td>
                        <td>{elo[name]:.0f}</td>
                        <td>{w}</td><td>{g}</td><td>{pct:.1f}%</td>
                        <td>{avg_time:.3f}s</td><td>{avg_moves:.1f}</td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>
    </div>

    <script>
        const names = """ + json.dumps(ranking) + """;
        const eloData = """ + json.dumps(elo_data) + """;
        const winData = """ + json.dumps(win_data) + """;
        const timeData = """ + json.dumps(time_data) + """;
        const movesData = """ + json.dumps(moves_data) + """;
        const barColors = """ + json.dumps(bar_colors) + """;

        const chartOpts = {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } },
                y: { grid: { display: false }, ticks: { color: '#e0e0e0', font: { size: 11 } } }
            }
        };

        new Chart(document.getElementById('eloChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: eloData, backgroundColor: barColors, borderRadius: 6 }] },
            options: chartOpts
        });
        new Chart(document.getElementById('winChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: winData, backgroundColor: barColors, borderRadius: 6 }] },
            options: { ...chartOpts, scales: { ...chartOpts.scales, x: { ...chartOpts.scales.x, max: 100 } } }
        });
        new Chart(document.getElementById('timeChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: timeData, backgroundColor: barColors, borderRadius: 6 }] },
            options: chartOpts
        });
        new Chart(document.getElementById('movesChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: movesData, backgroundColor: barColors, borderRadius: 6 }] },
            options: chartOpts
        });

        // Heatmap
        const matrix = """ + json.dumps(matrix) + """;
        const heatmapEl = document.getElementById('heatmap');
        const size = names.length;
        heatmapEl.style.gridTemplateColumns = `100px repeat(${size}, 1fr)`;

        heatmapEl.innerHTML = '<div class="heatmap-header"></div>';
        names.forEach(n => {
            heatmapEl.innerHTML += `<div class="heatmap-header" style="writing-mode:vertical-lr;text-orientation:mixed;transform:rotate(180deg)">${n}</div>`;
        });

        matrix.forEach((row, i) => {
            heatmapEl.innerHTML += `<div class="heatmap-header">${names[i]}</div>`;
            row.forEach((val, j) => {
                if (val === null) {
                    heatmapEl.innerHTML += `<div class="heatmap-cell" style="background:#1a2634">-</div>`;
                } else {
                    const pct = (val * 100).toFixed(0);
                    const r = Math.round(255 * (1 - val));
                    const g = Math.round(255 * val);
                    heatmapEl.innerHTML += `<div class="heatmap-cell" style="background:rgb(${r},${g},80);color:${val > 0.5 ? '#000' : '#fff'}">${pct}%</div>`;
                }
            });
        });
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Rapport HTML : {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tournoi round-robin entre les modèles AlphaZero dans model/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model-dir', type=str,
                        default=os.path.join(_dir, 'model'),
                        help="Dossier contenant les .pt (défaut: model/)")
    parser.add_argument('--add', nargs='*', metavar='PATH',
                        help="Chemins .pt supplémentaires à inclure "
                             "(ex: --add best checkpoints/model_iter_5.pt)")
    parser.add_argument('--games', type=int, default=20,
                        help="Parties par matchup (défaut: 20)")
    parser.add_argument('--sims', type=int, default=200,
                        help="Simulations MCTS par coup (défaut: 200)")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_dir, 'model_rank'),
                        help="Dossier de sortie (défaut: model_rank/)")
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

    # Découverte des modèles
    entries = discover_models(args.model_dir)  # [(nom, chemin), ...]

    # Modèles supplémentaires via --add
    if args.add:
        for p in args.add:
            if p.lower() in ('best', 'best_model', 'best_model.pt'):
                path = os.path.join(_dir, BEST_MODEL_FILE)
                if not os.path.isfile(path):
                    path = BEST_MODEL_FILE
            else:
                path = p
            if not os.path.isfile(path):
                alt = os.path.join(_dir, path)
                if os.path.isfile(alt):
                    path = alt
                else:
                    print(f"WARN : --add ignoré (introuvable) : {p}", file=sys.stderr)
                    continue
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in [e[0] for e in entries]:
                entries.append((name, path))

    if len(entries) < 2:
        print("ERREUR : il faut au moins 2 modèles pour un tournoi.", file=sys.stderr)
        sys.exit(1)

    n = len(entries)
    total_matchups = n * (n - 1) // 2

    print(f"Chargement de {n} modèles sur {device}...")
    agents = {}
    skipped = []
    for name, path in entries:
        try:
            agents[name] = _load_agent(path, device, args.sims)
            print(f"  OK  {name}  ({path})")
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

    # Préparer les sorties
    os.makedirs(args.output_dir, exist_ok=True)
    run_stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    if args.output:
        html_path = args.output
        base = os.path.splitext(html_path)[0]
    else:
        base = os.path.join(args.output_dir, f'ranking_{run_stamp}')
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

    # Tournoi round-robin
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

        # CSV
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

    # Elo + classement console
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
            'names':            valid_names,
            'elo':              elo,
            'wins_total':       wins_total,
            'games_total':      games_total,
            'times_total':      times_total,
            'moves_total':      moves_total,
            'results':          results,
            'total_time':       total_time,
            'games_per_matchup': args.games,
            'sims':             args.sims,
        }
        generate_html_report(all_stats, html_path)


if __name__ == '__main__':
    main()

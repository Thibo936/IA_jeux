#!/usr/bin/env python3
"""
ranking.py — Tournoi round-robin entre TOUTES les IA Hex 11×11.

IA supportées : random, alphabeta, mc_pure, mcts_light, heuristic, mohex, alphazero

Usage :
  python ranking.py                          # défaut : 100 parties/matchup
  python ranking.py --games 50               # plus rapide
  python ranking.py --output classement.html # fichier de sortie personnalisé
  python ranking.py --no-html                # sortie texte uniquement
  python ranking.py --workers 4              # parallélise les matchups
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from hex_env import HexEnv
from config import NUM_CELLS

# ─── Import des joueurs ───────────────────────────────────────────────────────

def get_player(name: str, device: torch.device = None, sims: int = 100):
    """Retourne une instance de joueur avec l'interface select_move(env, time_s)."""
    n = name.lower()

    if n == 'random':
        from random_player import RandomPlayer
        return RandomPlayer()

    if n == 'alphabeta':
        from alphabeta import AlphaBetaPlayer
        return AlphaBetaPlayer()

    if n in ('mc_pure', 'montecarlo'):
        from monte_carlo_pure import PureMonteCarloPlayer
        return PureMonteCarloPlayer()

    if n in ('mcts_light', 'mcts'):
        from mcts_light import LightMCTSPlayer
        return LightMCTSPlayer()

    if n == 'heuristic':
        from heuristic_player import HeuristicPlayer
        return HeuristicPlayer()

    if n == 'mohex':
        from mohex import MoHexPlayer
        return MoHexPlayer()

    if n == 'alphazero':
        from tournament import AlphaZeroPlayer
        return AlphaZeroPlayer(device=device, sims=sims)

    return None


def get_player_name(name: str) -> str:
    """Retourne un nom lisible pour l'affichage."""
    mapping = {
        'random': 'Random',
        'alphabeta': 'AlphaBeta',
        'mc_pure': 'MonteCarlo',
        'montecarlo': 'MonteCarlo',
        'mcts_light': 'MCTS-Light',
        'mcts': 'MCTS-Light',
        'heuristic': 'Heuristic',
        'mohex': 'MoHex',
        'alphazero': 'AlphaZero',
    }
    n = name.lower()
    if n in mapping:
        return mapping[n]
    return name


def get_player_type(name: str) -> str:
    """Catégorie du joueur."""
    n = name.lower()
    if n == 'random':
        return 'Random'
    if n == 'alphabeta':
        return 'Alpha-Beta'
    if n in ('mc_pure', 'montecarlo'):
        return 'Monte Carlo'
    if n in ('mcts_light', 'mcts'):
        return 'MCTS'
    if n == 'heuristic':
        return 'Heuristique'
    if n == 'mohex':
        return 'MoHex'
    if n == 'alphazero':
        return 'AlphaZero'
    return 'Inconnu'


# ─── Partie ───────────────────────────────────────────────────────────────────

TIME_PER_MOVE = 0.5  # secondes par coup


def play_game(player_a, player_b, game_id: int) -> dict:
    """
    Joue une partie entre player_a et player_b.
    Retourne un dict avec les stats de la partie.
    """
    env = HexEnv()
    moves = []
    times_a = []
    times_b = []
    winner = None
    t_start = time.time()

    while not env.is_terminal():
        cur = player_a if env.blue_to_play else player_b
        t0 = time.time()
        # Les IA peuvent imprimer des stats par coup sur stderr: on les masque
        # pour garder une sortie tournoi lisible.
        with contextlib.redirect_stderr(io.StringIO()):
            move = cur.select_move(env, TIME_PER_MOVE)
        elapsed = time.time() - t0

        if env.blue_to_play:
            times_a.append(elapsed)
        else:
            times_b.append(elapsed)

        if move < 0 or move >= NUM_CELLS:
            # Coup invalide → l'autre joueur gagne
            winner = 'red' if env.blue_to_play else 'blue'
            break

        r, c = divmod(move, 11)
        if env.blue[r, c] or env.red[r, c]:
            winner = 'red' if env.blue_to_play else 'blue'
            break

        env.apply_move(move)
        moves.append({
            'turn': len(moves) + 1,
            'player': 'blue' if env.blue_to_play else 'red',  # après apply, c'est l'autre
            'move': env.pos_to_str(move),
            'time': elapsed,
        })

    if winner is None:
        winner = env.winner()

    total_time = time.time() - t_start
    avg_time_a = np.mean(times_a) if times_a else 0
    avg_time_b = np.mean(times_b) if times_b else 0

    return {
        'game_id': game_id,
        'winner': winner,
        'total_moves': len(moves),
        'total_time': total_time,
        'avg_time_blue': avg_time_a,
        'avg_time_red': avg_time_b,
        'moves': moves,
    }


def _play_one_game(args):
    """Joue une seule partie (utilisé par le ThreadPoolExecutor)."""
    player_a, player_b, name_a, name_b, game_index = args
    a_is_blue = (game_index % 2 == 0)
    if a_is_blue:
        result = play_game(player_a, player_b, game_index + 1)
        blue_name, red_name = name_a, name_b
    else:
        result = play_game(player_b, player_a, game_index + 1)
        blue_name, red_name = name_b, name_a

    result['a_is_blue'] = a_is_blue
    result['blue_name'] = blue_name
    result['red_name'] = red_name
    result['winner_name'] = blue_name if result['winner'] == 'blue' else red_name
    return result


def match(player_a, player_b, name_a: str, name_b: str,
          num_games: int, game_threads: int = 1) -> dict:
    """
    Fait jouer player_a vs player_b sur num_games parties.
    game_threads > 1 : parallélise les parties via threads.
    """
    wins_a = 0
    wins_b = 0
    games = []
    times_a_all = []
    times_b_all = []
    moves_all = []

    if game_threads > 1:
        # Chaque thread a sa propre copie des joueurs pour éviter les conflits
        # d'état interne (arbres MCTS, etc.)
        from copy import deepcopy
        tasks = []
        for i in range(num_games):
            pa = deepcopy(player_a)
            pb = deepcopy(player_b)
            tasks.append((pa, pb, name_a, name_b, i))

        with ThreadPoolExecutor(max_workers=game_threads) as pool:
            results_list = list(pool.map(_play_one_game, tasks))
    else:
        results_list = []
        for i in range(num_games):
            result = _play_one_game((player_a, player_b, name_a, name_b, i))
            results_list.append(result)

    for i, result in enumerate(results_list):
        a_is_blue = (i % 2 == 0)
        if a_is_blue:
            if result['winner'] == 'blue':
                wins_a += 1
            else:
                wins_b += 1
            times_a_all.append(result['avg_time_blue'])
            times_b_all.append(result['avg_time_red'])
        else:
            if result['winner'] == 'red':
                wins_a += 1
            else:
                wins_b += 1
            times_a_all.append(result['avg_time_red'])
            times_b_all.append(result['avg_time_blue'])
        moves_all.append(result['total_moves'])
        games.append(result)

    return {
        'name_a': name_a,
        'name_b': name_b,
        'wins_a': wins_a,
        'wins_b': wins_b,
        'games': games,
        'avg_time_a': np.mean(times_a_all) if times_a_all else 0,
        'avg_time_b': np.mean(times_b_all) if times_b_all else 0,
        'avg_moves': np.mean(moves_all) if moves_all else 0,
    }


# ─── Calcul Elo ───────────────────────────────────────────────────────────────

def compute_elo(names: list[str], results: dict, k: float = 32.0,
                initial: float = 1000.0) -> dict[str, float]:
    elo = {name: initial for name in names}
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


# ─── Génération HTML ─────────────────────────────────────────────────────────

def generate_html_report(all_stats: dict, output_path: str):
    """Génère un rapport HTML avec graphiques interactifs."""
    names = all_stats['names']
    elo = all_stats['elo']
    wins_total = all_stats['wins_total']
    games_total = all_stats['games_total']
    times_total = all_stats['times_total']
    moves_total = all_stats['moves_total']
    results = all_stats['results']
    player_types = all_stats['player_types']
    total_time = all_stats['total_time']
    n = len(names)
    total_matchups = n * (n - 1) // 2

    # Préparer les données pour les graphiques
    ranking = sorted(names, key=lambda x: elo[x], reverse=True)
    elo_data = [elo[name] for name in ranking]
    win_data = [100.0 * wins_total[name] / games_total[name] if games_total[name] > 0 else 0 for name in ranking]
    time_data = [times_total[name] / games_total[name] if games_total[name] > 0 else 0 for name in ranking]
    moves_data = [moves_total[name] / games_total[name] if games_total[name] > 0 else 0 for name in ranking]
    type_colors = {
        'Random': '#95a5a6',
        'Alpha-Beta': '#3498db',
        'Monte Carlo': '#e67e22',
        'MCTS': '#2ecc71',
        'Heuristique': '#9b59b6',
        'MoHex': '#1abc9c',
        'AlphaZero': '#e74c3c',
    }
    bar_colors = [type_colors.get(player_types.get(name, 'Inconnu'), '#34495e') for name in ranking]

    # Matrice des résultats pour heatmap
    matrix = []
    for a in ranking:
        row = []
        for b in ranking:
            if a == b:
                row.append(None)
            else:
                key = (a, b) if (a, b) in results else (b, a)
                if key in results:
                    wa, wb = results[key]
                    if key == (a, b):
                        row.append(wa / (wa + wb) if (wa + wb) > 0 else 0.5)
                    else:
                        row.append(wb / (wa + wb) if (wa + wb) > 0 else 0.5)
                else:
                    row.append(0.5)
        matrix.append(row)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classement Hex 11×11 — Tournoi Round-Robin</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; }}
        h1 {{ text-align: center; color: #00d4ff; margin-bottom: 0.5rem; font-size: 2rem; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{ background: #1a2634; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .card h2 {{ color: #00d4ff; margin-bottom: 1rem; font-size: 1.2rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #2a3a4a; }}
        th {{ color: #00d4ff; font-weight: 600; }}
        tr:hover {{ background: #243447; }}
        .rank-1 {{ color: #ffd700; font-weight: bold; }}
        .rank-2 {{ color: #c0c0c0; font-weight: bold; }}
        .rank-3 {{ color: #cd7f32; font-weight: bold; }}
        .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
        .heatmap {{ display: grid; gap: 2px; margin-top: 1rem; }}
        .heatmap-cell {{ padding: 0.4rem; text-align: center; font-size: 0.75rem; border-radius: 3px; }}
        .heatmap-header {{ font-weight: 600; color: #00d4ff; font-size: 0.7rem; }}
        .stats-row {{ display: flex; justify-content: space-around; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }}
        .stat-box {{ background: #1a2634; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; min-width: 150px; }}
        .stat-box .value {{ font-size: 1.8rem; font-weight: bold; color: #00d4ff; }}
        .stat-box .label {{ font-size: 0.8rem; color: #888; margin-top: 0.3rem; }}
        canvas {{ max-height: 350px; }}
        .full-width {{ grid-column: 1 / -1; }}
    </style>
</head>
<body>
    <h1>🏆 Classement Hex 11×11</h1>
    <p class="subtitle">Tournoi round-robin • {n} IA • {total_matchups} matchups • {all_stats['games_per_matchup']} parties/matchup • Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>

    <div class="stats-row">
        <div class="stat-box"><div class="value">{n}</div><div class="label">Intelligence Artificielles</div></div>
        <div class="stat-box"><div class="value">{total_matchups}</div><div class="label">Matchups</div></div>
        <div class="stat-box"><div class="value">{sum(games_total.values()) // 2}</div><div class="label">Parties totales</div></div>
        <div class="stat-box"><div class="value">{total_time:.0f}s</div><div class="label">Temps total</div></div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>📊 Classement Elo</h2>
            <canvas id="eloChart"></canvas>
        </div>
        <div class="card">
            <h2>🏅 Taux de Victoire (%)</h2>
            <canvas id="winChart"></canvas>
        </div>
        <div class="card">
            <h2>⏱️ Temps moyen par coup (s)</h2>
            <canvas id="timeChart"></canvas>
        </div>
        <div class="card">
            <h2>🎯 Durée moyenne des parties (coups)</h2>
            <canvas id="movesChart"></canvas>
        </div>
        <div class="card full-width">
            <h2>🔥 Matrice des confrontations (taux de victoire)</h2>
            <div id="heatmap" class="heatmap"></div>
        </div>
        <div class="card full-width">
            <h2>📋 Classement détaillé</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rang</th><th>IA</th><th>Type</th><th>Elo</th>
                        <th>Victoires</th><th>Parties</th><th>Win%</th>
                        <th>Temps moy./coup</th><th>Coups moy./partie</th>
                    </tr>
                </thead>
                <tbody>
"""

    for rank, name in enumerate(ranking, 1):
        rank_class = f"rank-{rank}" if rank <= 3 else ""
        w = wins_total[name]
        g = games_total[name]
        pct = 100.0 * w / g if g > 0 else 0
        avg_time = times_total[name] / g if g > 0 else 0
        avg_moves = moves_total[name] / g if g > 0 else 0
        ptype = player_types.get(name, 'Inconnu')
        color = type_colors.get(ptype, '#34495e')
        html += f"""                    <tr>
                        <td class="{rank_class}">#{rank}</td>
                        <td>{name}</td>
                        <td><span class="badge" style="background:{color}20;color:{color};border:1px solid {color}40">{ptype}</span></td>
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
            scales: { x: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } }, y: { grid: { display: false }, ticks: { color: '#e0e0e0' } } }
        };

        new Chart(document.getElementById('eloChart'), {
            type: 'bar', data: { labels: names, datasets: [{ data: eloData, backgroundColor: barColors, borderRadius: 6 }] }, options: chartOpts
        });
        new Chart(document.getElementById('winChart'), {
            type: 'bar', data: { labels: names, datasets: [{ data: winData, backgroundColor: barColors, borderRadius: 6 }] }, options: { ...chartOpts, scales: { ...chartOpts.scales, x: { ...chartOpts.scales.x, max: 100 } } }
        });
        new Chart(document.getElementById('timeChart'), {
            type: 'bar', data: { labels: names, datasets: [{ data: timeData, backgroundColor: barColors, borderRadius: 6 }] }, options: chartOpts
        });
        new Chart(document.getElementById('movesChart'), {
            type: 'bar', data: { labels: names, datasets: [{ data: movesData, backgroundColor: barColors, borderRadius: 6 }] }, options: chartOpts
        });

        // Heatmap
        const matrix = """ + json.dumps(matrix) + """;
        const heatmapEl = document.getElementById('heatmap');
        const size = names.length;
        heatmapEl.style.gridTemplateColumns = `80px repeat(${size}, 1fr)`;

        // Header row
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

DEFAULT_IAS = ['random', 'alphabeta', 'heuristic', 'mohex', 'alphazero']


def main():
    parser = argparse.ArgumentParser(
        description="Tournoi round-robin entre TOUTES les IA Hex 11×11")
    parser.add_argument('--ias', nargs='+', default=None,
                        help=f"Liste des IA à inclure (défaut: {DEFAULT_IAS})")
    parser.add_argument('--games', type=int, default=100,
                        help="Parties par matchup (défaut: 100)")
    parser.add_argument('--sims', type=int, default=100,
                        help="Simulations MCTS pour AlphaZero (défaut: 100)")
    parser.add_argument('--time', type=float, default=0.5,
                        help="Temps par coup en secondes (défaut: 0.5)")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_dir, 'rank'),
                        help="Dossier de sortie (défaut: alphazero/rank)")
    parser.add_argument('--output', type=str, default=None,
                        help="Fichier HTML explicite (défaut: rank/ranking_<date>.html)")
    parser.add_argument('--no-html', action='store_true',
                        help="Sortie texte uniquement (pas de HTML)")
    parser.add_argument('--no-csv', action='store_true',
                        help="Ne pas écrire le CSV des parties")
    parser.add_argument('--device', type=str, default=None,
                        help="Device (cuda/cpu, défaut: auto)")
    parser.add_argument('--workers', type=int, default=1,
                        help="Matchups en parallèle (défaut: 1)")
    parser.add_argument('--game-threads', type=int, default=1,
                        help="Parties en parallèle par matchup (défaut: 1)")
    args = parser.parse_args()

    global TIME_PER_MOVE
    TIME_PER_MOVE = args.time

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Liste des IA
    ias = args.ias if args.ias else DEFAULT_IAS.copy()

    names = [get_player_name(ia) for ia in ias]
    n = len(names)
    total_matchups = n * (n - 1) // 2

    print(f"Tournoi : {n} IA, {total_matchups} matchups, {args.games} parties/matchup, device={device}")

    # Charger tous les joueurs
    players = {}
    player_types = {}
    for ia, name in zip(ias, names):
        p = get_player(ia, device=device, sims=args.sims)
        if p is not None:
            players[name] = p
            player_types[name] = get_player_type(ia)

    valid_names = list(players.keys())
    if len(valid_names) < 2:
        print("Erreur : pas assez de joueurs valides.")
        sys.exit(1)

    # Paths de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    run_dt = datetime.now()
    run_stamp = run_dt.strftime('%Y-%m-%d_%H%M')
    if args.output:
        html_path = args.output
        base, _ = os.path.splitext(html_path)
    else:
        base = os.path.join(args.output_dir, f'ranking_{run_stamp}')
        html_path = base + '.html'
    csv_path = base + '.csv'

    # CSV : une ligne par partie jouée
    csv_file = None
    csv_writer = None
    if not args.no_csv:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'run_timestamp', 'name_a', 'name_b', 'game_index',
            'a_is_blue', 'blue_name', 'red_name', 'winner',
            'winner_name', 'total_moves', 'total_time',
            'avg_time_blue', 'avg_time_red',
        ])

    # Tournoi round-robin
    results = {}
    wins_total = {name: 0 for name in valid_names}
    games_total = {name: 0 for name in valid_names}
    times_total = {name: 0.0 for name in valid_names}
    moves_total = {name: 0 for name in valid_names}
    matchup_count = 0
    t_start = time.time()

    matchups = list(combinations(valid_names, 2))
    gt = args.game_threads

    def _run_one_matchup(name_a, name_b):
        return match(players[name_a], players[name_b],
                     name_a, name_b, args.games, game_threads=gt)

    def _collect(name_a, name_b, match_result):
        if csv_writer is not None:
            for g in match_result['games']:
                csv_writer.writerow([
                    run_stamp, name_a, name_b, g['game_id'],
                    int(g['a_is_blue']), g['blue_name'], g['red_name'],
                    g['winner'], g['winner_name'], g['total_moves'],
                    f"{g['total_time']:.4f}",
                    f"{g['avg_time_blue']:.6f}",
                    f"{g['avg_time_red']:.6f}",
                ])
            csv_file.flush()
        for g in match_result['games']:
            g['moves'] = None

        results[(name_a, name_b)] = (match_result['wins_a'], match_result['wins_b'])
        wins_total[name_a] += match_result['wins_a']
        wins_total[name_b] += match_result['wins_b']
        games_total[name_a] += args.games
        games_total[name_b] += args.games
        times_total[name_a] += match_result['avg_time_a'] * args.games
        times_total[name_b] += match_result['avg_time_b'] * args.games
        moves_total[name_a] += match_result['avg_moves'] * args.games
        moves_total[name_b] += match_result['avg_moves'] * args.games

    if args.workers > 1:
        # ─── Matchups en parallèle (threads — partage GPU + modèles) ────
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_matchup = {
                executor.submit(_run_one_matchup, na, nb): (na, nb)
                for na, nb in matchups
            }
            for future in as_completed(future_to_matchup):
                name_a, name_b = future_to_matchup[future]
                matchup_count += 1
                match_result = future.result()
                sys.stdout.write(
                    f"\r  [{matchup_count}/{total_matchups}] {name_a} vs {name_b} done"
                )
                sys.stdout.flush()
                _collect(name_a, name_b, match_result)
    else:
        # ─── Mode séquentiel ────────────────────────────────────────────
        for name_a, name_b in matchups:
            matchup_count += 1
            sys.stdout.write(
                f"\r  [{matchup_count}/{total_matchups}] {name_a} vs {name_b} ..."
            )
            sys.stdout.flush()
            match_result = _run_one_matchup(name_a, name_b)
            _collect(name_a, name_b, match_result)

    total_time = time.time() - t_start
    print()  # fin de la ligne de progression

    if csv_file is not None:
        csv_file.close()
        print(f"CSV des parties : {csv_path}")

    # Calcul Elo
    elo = compute_elo(valid_names, results)
    ranking = sorted(valid_names, key=lambda x: elo[x], reverse=True)

    # Affichage console compact
    print(f"\n{'Rang':<5} {'IA':<16} {'Elo':>5} {'W':>4}/{'':<4} {'Win%':>5}")
    print("-" * 42)
    for rank, name in enumerate(ranking, 1):
        w = wins_total[name]
        g = games_total[name]
        pct = 100.0 * w / g if g > 0 else 0
        print(f"#{rank:<4} {name:<16} {elo[name]:>5.0f} {w:>4}/{g:<4} {pct:>4.0f}%")
    print(f"\nTemps total : {total_time:.0f}s")

    # Génération HTML
    if not args.no_html:
        all_stats = {
            'names': valid_names,
            'elo': elo,
            'wins_total': wins_total,
            'games_total': games_total,
            'times_total': times_total,
            'moves_total': moves_total,
            'results': results,
            'player_types': player_types,
            'total_time': total_time,
            'games_per_matchup': args.games,
        }
        generate_html_report(all_stats, html_path)


if __name__ == "__main__":
    main()

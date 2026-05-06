#!/usr/bin/env python3
"""
csv_to_html.py — Génère un rapport HTML à partir du CSV ranking.csv existant.

Usage:
  python3 csv_to_html.py
  python3 csv_to_html.py --csv ranking.csv --output ranking_from_csv.html
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict
from datetime import datetime

# ─── Mapping joueur → type / famille ─────────────────────────────────────────

def infer_player_type(name: str) -> tuple[str, str]:
    """Retourne (type, family) pour un nom de joueur."""
    n = name.lower()
    # AlphaZero
    if name.startswith('AZ-'):
        return 'AlphaZero', 'alphazero'
    # Classiques connus
    classics = {
        'random': ('Random', 'classic'),
        'alphabeta': ('Alpha-Beta', 'classic'),
        'montecarlo': ('Monte Carlo', 'classic'),
        'mc_pure': ('Monte Carlo', 'classic'),
        'mcts-light': ('MCTS', 'classic'),
        'mcts_light': ('MCTS', 'classic'),
        'mcts': ('MCTS', 'classic'),
        'heuristic': ('Heuristique', 'classic'),
        'heuristic_player': ('Heuristique', 'classic'),
        'mohex': ('MoHex', 'classic'),
        'minimax_m2': ('Minimax', 'classic'),
    }
    if n in classics:
        return classics[n]
    # LLM / autres
    return 'LLM', 'llm'


TYPE_COLORS = {
    'Random': '#95a5a6',
    'Alpha-Beta': '#3498db',
    'Monte Carlo': '#e67e22',
    'MCTS': '#2ecc71',
    'Heuristique': '#9b59b6',
    'MoHex': '#1abc9c',
    'Minimax': '#f1c40f',
    'AlphaZero': '#e74c3c',
    'LLM': '#e84393',
    'Inconnu': '#34495e',
}


# ─── Lecture & agrégation CSV ────────────────────────────────────────────────

def read_and_aggregate(csv_path: str):
    """
    Lit le CSV et agrège les résultats.
    Retourne:
      - matchups: dict[(name_a, name_b)] = {'wins_a': int, 'wins_b': int,
                                             'games': int,
                                             'sum_time_a': float, 'sum_time_b': float,
                                             'sum_moves': int}
      - player_stats: dict[name] = {'wins': int, 'games': int,
                                    'sum_time': float, 'sum_moves': int}
      - meta: dict avec sims, time_per_move, etc.
    """
    matchups = defaultdict(lambda: {
        'wins_a': 0, 'wins_b': 0, 'games': 0,
        'sum_time_a': 0.0, 'sum_time_b': 0.0, 'sum_moves': 0,
    })
    player_stats = defaultdict(lambda: {
        'wins': 0, 'games': 0, 'sum_time': 0.0, 'sum_moves': 0,
    })

    sims_set = set()
    time_per_move_set = set()
    total_rows = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            total_rows += 1
            na, nb = r['name_a'], r['name_b']
            winner = r['winner_name']
            a_is_blue = int(r['a_is_blue'])
            total_moves = int(r['total_moves'])
            avg_time_blue = float(r['avg_time_blue'])
            avg_time_red = float(r['avg_time_red'])
            sims_set.add(r.get('sims', '600'))
            time_per_move_set.add(r.get('time_per_move', '0.5'))

            key = (na, nb)
            m = matchups[key]
            m['games'] += 1
            m['sum_moves'] += total_moves

            # Temps
            if a_is_blue:
                time_a = avg_time_blue
                time_b = avg_time_red
            else:
                time_a = avg_time_red
                time_b = avg_time_blue
            m['sum_time_a'] += time_a
            m['sum_time_b'] += time_b

            # Victoires
            if winner == na:
                m['wins_a'] += 1
                player_stats[na]['wins'] += 1
            elif winner == nb:
                m['wins_b'] += 1
                player_stats[nb]['wins'] += 1

            player_stats[na]['games'] += 1
            player_stats[nb]['games'] += 1
            player_stats[na]['sum_time'] += time_a
            player_stats[nb]['sum_time'] += time_b
            player_stats[na]['sum_moves'] += total_moves
            player_stats[nb]['sum_moves'] += total_moves

    meta = {
        'total_rows': total_rows,
        'sims': int(list(sims_set)[0]) if sims_set else 600,
        'time_per_move': float(list(time_per_move_set)[0]) if time_per_move_set else 0.5,
    }
    return matchups, player_stats, meta


def resolve_input_path(path: str) -> str:
    """Résout un chemin relatif depuis le dossier du script si besoin."""
    if os.path.isabs(path):
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)
    candidate = os.path.join(_dir, path)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(path)


# ─── Elo ─────────────────────────────────────────────────────────────────────

def compute_elo(names: list[str], results: dict, k: float = 32.0, initial: float = 1000.0):
    """Calcule les Elo à partir des résultats agrégés."""
    elo = {n: initial for n in names}
    for _ in range(10):
        for (a, b), m in results.items():
            wa, wb = m['wins_a'], m['wins_b']
            total = wa + wb
            if total == 0:
                continue
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400))
            elo[a] += k * (wa / total - ea)
            elo[b] += k * (wb / total - (1.0 - ea))
    return elo


# ─── HTML Template ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classement Hex 11×11 — Depuis CSV</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; }
  h1 { text-align: center; color: #00d4ff; margin-bottom: 0.5rem; font-size: 2rem; }
  .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
  .card { background: #1a2634; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .card h2 { color: #00d4ff; margin-bottom: 1rem; font-size: 1.2rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.5rem; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #2a3a4a; font-size: 0.85rem; word-break: break-all; }
  th { color: #00d4ff; font-weight: 600; }
  tr:hover { background: #243447; }
  .rank-1 { color: #ffd700; font-weight: bold; }
  .rank-2 { color: #c0c0c0; font-weight: bold; }
  .rank-3 { color: #cd7f32; font-weight: bold; }
  .badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
  .heatmap { display: grid; gap: 2px; margin-top: 1rem; overflow-x: auto; }
  .heatmap-cell { padding: 0.4rem; text-align: center; font-size: 0.7rem; border-radius: 3px; }
  .heatmap-header { font-weight: 600; color: #00d4ff; font-size: 0.65rem; word-break: break-all; }
  .stats-row { display: flex; justify-content: space-around; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }
  .stat-box { background: #1a2634; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; min-width: 150px; }
  .stat-box .value { font-size: 1.8rem; font-weight: bold; color: #00d4ff; }
  .stat-box .label { font-size: 0.8rem; color: #888; margin-top: 0.3rem; }
  canvas { max-height: 350px; }
  .full-width { grid-column: 1 / -1; }
</style>
</head>
<body>
<h1>Classement Hex 11×11 — Depuis CSV</h1>
<p id="subtitle" class="subtitle"></p>

<div class="stats-row" id="stats-row"></div>

<div class="grid">
  <div class="card"><h2>Score Elo</h2><canvas id="eloChart"></canvas></div>
  <div class="card"><h2>Taux de Victoire (%)</h2><canvas id="winChart"></canvas></div>
  <div class="card"><h2>Temps moyen par coup (s)</h2><canvas id="timeChart"></canvas></div>
  <div class="card"><h2>Durée moyenne des parties (coups)</h2><canvas id="movesChart"></canvas></div>
  <div class="card full-width"><h2>Profil radar (Elo, Win%, Vitesse, Brièveté — tous normalisés 0-100)</h2><canvas id="radarChart" style="max-height:520px"></canvas></div>
  <div class="card full-width"><h2>Matrice des confrontations (taux de victoire ligne vs colonne)</h2><div id="heatmap" class="heatmap"></div></div>
  <div class="card full-width"><h2>Classement détaillé</h2><div id="table-wrap"></div></div>
</div>

<script id="ranking-data" type="application/json">__RANKING_JSON__</script>
<script>
const data = JSON.parse(document.getElementById('ranking-data').textContent);

function colorFor(name) {
    const t = data.playerTypes[name] || 'Inconnu';
    const colors = {
        'Random': '#95a5a6', 'Alpha-Beta': '#3498db', 'Monte Carlo': '#e67e22',
        'MCTS': '#2ecc71', 'Heuristique': '#9b59b6', 'MoHex': '#1abc9c',
        'Minimax': '#f1c40f', 'AlphaZero': '#e74c3c', 'LLM': '#e84393', 'Inconnu': '#34495e',
    };
    return colors[t] || '#34495e';
}

// ── Sous-titre & stats ────────────────────────────────────────────────────
const n = data.names.length;
const totalMatchups = n * (n - 1) / 2;
const totalGames = data.results.reduce((s, r) => s + r.winsA + r.winsB, 0);
document.getElementById('subtitle').textContent =
    `Tournoi round-robin • ${n} joueurs • ${data.matchups_played}/${totalMatchups} matchups joués • ${data.games_per_matchup} parties/matchup • ${data.sims} sims/coup • Généré le ${data.generated_at_human}`;

const statsRow = document.getElementById('stats-row');
[['Joueurs', n], ['Matchups joués', data.matchups_played], ['Parties totales', totalGames],
 ['Sims/coup', data.sims]
].forEach(([label, value]) => {
    statsRow.insertAdjacentHTML('beforeend',
        `<div class="stat-box"><div class="value">${value}</div><div class="label">${label}</div></div>`);
});

// ── Charts (barres) ───────────────────────────────────────────────────────
const names = data.names;
const eloData = names.map(n => data.elo[n]);
const winData = names.map(n => data.winPct[n]);
const timeData = names.map(n => data.avgTime[n]);
const movesData = names.map(n => data.avgMoves[n]);
const barColors = names.map(n => colorFor(n));

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

// ── Radar (4 axes normalisés 0-100) ───────────────────────────────────────
function normalize(values, invert) {
    const min = Math.min(...values), max = Math.max(...values);
    const span = max - min || 1;
    return values.map(v => {
        const x = (v - min) / span * 100;
        return invert ? 100 - x : x;
    });
}
const eloNorm   = normalize(eloData,   false);
const winNorm   = normalize(winData,   false);
const timeNorm  = normalize(timeData,  true);
const movesNorm = normalize(movesData, true);

const radarDatasets = names.map((nm, i) => {
    const c = colorFor(nm);
    return {
        label: nm,
        data: [eloNorm[i], winNorm[i], timeNorm[i], movesNorm[i]],
        borderColor: c,
        backgroundColor: c + '20',
        borderWidth: 2,
        pointRadius: 3,
    };
});
new Chart(document.getElementById('radarChart'), {
    type: 'radar',
    data: { labels: ['Elo', 'Win%', 'Vitesse', 'Brièveté'], datasets: radarDatasets },
    options: {
        responsive: true,
        plugins: { legend: { position: 'right', labels: { color: '#e0e0e0', font: { size: 11 } } } },
        scales: {
            r: {
                min: 0, max: 100,
                grid:        { color: '#2a3a4a' },
                angleLines:  { color: '#2a3a4a' },
                pointLabels: { color: '#00d4ff', font: { size: 13 } },
                ticks:       { color: '#888', backdropColor: 'transparent', stepSize: 25 },
            }
        }
    }
});

// ── Heatmap ───────────────────────────────────────────────────────────────
const resultMap = {};
data.results.forEach(r => { resultMap[r.a + '|' + r.b] = r; });

const heatmapEl = document.getElementById('heatmap');
heatmapEl.style.gridTemplateColumns = `100px repeat(${n}, 1fr)`;
heatmapEl.innerHTML = '<div class="heatmap-header"></div>';
names.forEach(nm => {
    heatmapEl.innerHTML += `<div class="heatmap-header" style="writing-mode:vertical-lr;text-orientation:mixed;transform:rotate(180deg)">${nm}</div>`;
});
names.forEach((a, i) => {
    heatmapEl.innerHTML += `<div class="heatmap-header">${a}</div>`;
    names.forEach((b, j) => {
        if (i === j) {
            heatmapEl.innerHTML += `<div class="heatmap-cell" style="background:#1a2634">-</div>`;
            return;
        }
        let val = 0.5;
        const fwd = resultMap[a + '|' + b];
        const rev = resultMap[b + '|' + a];
        if (fwd && (fwd.winsA + fwd.winsB) > 0) val = fwd.winsA / (fwd.winsA + fwd.winsB);
        else if (rev && (rev.winsA + rev.winsB) > 0) val = rev.winsB / (rev.winsA + rev.winsB);
        const pct = (val * 100).toFixed(0);
        const r = Math.round(255 * (1 - val));
        const g = Math.round(255 * val);
        heatmapEl.innerHTML += `<div class="heatmap-cell" style="background:rgb(${r},${g},80);color:${val > 0.5 ? '#000' : '#fff'}">${pct}%</div>`;
    });
});

// ── Tableau ───────────────────────────────────────────────────────────────
let tableHTML = `<table><thead><tr>
<th>Rang</th><th>Joueur</th><th>Famille</th><th>Type</th><th>Elo</th>
<th>Victoires</th><th>Parties</th><th>Win%</th>
<th>Temps moy./coup</th><th>Coups moy./partie</th></tr></thead><tbody>`;
names.forEach((nm, idx) => {
    const rank = idx + 1;
    const rankClass = rank <= 3 ? `rank-${rank}` : '';
    const t = data.playerTypes[nm];
    const f = data.playerFamilies[nm];
    const c = colorFor(nm);
    tableHTML += `<tr>
        <td class="${rankClass}">#${rank}</td>
        <td>${nm}</td>
        <td>${f}</td>
        <td><span class="badge" style="background:${c}20;color:${c};border:1px solid ${c}40">${t}</span></td>
        <td>${data.elo[nm].toFixed(0)}</td>
        <td>${data.winsTotal[nm]}</td>
        <td>${data.gamesTotal[nm]}</td>
        <td>${data.winPct[nm].toFixed(1)}%</td>
        <td>${data.avgTime[nm].toFixed(3)}s</td>
        <td>${data.avgMoves[nm].toFixed(1)}</td>
    </tr>`;
});
tableHTML += '</tbody></table>';
document.getElementById('table-wrap').innerHTML = tableHTML;
</script>
</body>
</html>"""


def generate_html_report(payload: dict, output_path: str) -> None:
    json_blob = json.dumps(payload, ensure_ascii=False, indent=2)
    json_blob = json_blob.replace('</', '<\\/')
    html = _HTML_TEMPLATE.replace('__RANKING_JSON__', json_blob)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Rapport HTML : {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Génère un HTML à partir de ranking.csv")
    parser.add_argument('--csv', type=str, default='ranking.csv',
                        help="Chemin vers le CSV (défaut: ranking.csv)")
    parser.add_argument('--output', type=str, default=None,
                        help="Chemin de sortie HTML (défaut: même dossier que le CSV)")
    args = parser.parse_args()

    csv_path = resolve_input_path(args.csv)
    if not os.path.isfile(csv_path):
        print(f"ERREUR : fichier CSV introuvable : {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Lecture de {csv_path} ...")
    matchups, player_stats, meta = read_and_aggregate(csv_path)

    # Liste des joueurs (ceux qui apparaissent dans les stats)
    all_names = sorted(player_stats.keys())
    if len(all_names) < 2:
        print("ERREUR : moins de 2 joueurs trouvés dans le CSV.", file=sys.stderr)
        sys.exit(1)

    # Elo
    elo = compute_elo(all_names, matchups)
    ranking = sorted(all_names, key=lambda k: elo[k], reverse=True)

    # Calcul des stats globales
    wins_total = {n: player_stats[n]['wins'] for n in all_names}
    games_total = {n: player_stats[n]['games'] for n in all_names}
    win_pct = {n: (100.0 * wins_total[n] / games_total[n] if games_total[n] > 0 else 0.0) for n in all_names}
    avg_time = {n: (player_stats[n]['sum_time'] / games_total[n] if games_total[n] > 0 else 0.0) for n in all_names}
    avg_moves = {n: (player_stats[n]['sum_moves'] / games_total[n] if games_total[n] > 0 else 0.0) for n in all_names}

    player_types = {n: infer_player_type(n)[0] for n in all_names}
    player_families = {n: infer_player_type(n)[1] for n in all_names}

    # Déduire games_per_matchup (max de parties par matchup)
    games_per_matchup = max(m['games'] for m in matchups.values()) if matchups else 0

    payload = {
        'schema_version': 2,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'generated_at_human': datetime.now().strftime('%d/%m/%Y à %H:%M'),
        'games_per_matchup': games_per_matchup,
        'sims': meta['sims'],
        'time_per_move': meta['time_per_move'],
        'matchups_played': len(matchups),
        'names': ranking,
        'playerTypes': player_types,
        'playerFamilies': player_families,
        'elo': {k: float(elo[k]) for k in all_names},
        'winPct': win_pct,
        'avgTime': avg_time,
        'avgMoves': avg_moves,
        'winsTotal': {k: int(wins_total[k]) for k in all_names},
        'gamesTotal': {k: int(games_total[k]) for k in all_names},
        'results': [
            {'a': a, 'b': b, 'winsA': m['wins_a'], 'winsB': m['wins_b']}
            for (a, b), m in matchups.items()
        ],
    }

    if args.output:
        output_path = (args.output if os.path.isabs(args.output)
                       else os.path.join(os.path.dirname(csv_path), args.output))
    else:
        output_path = os.path.join(
            os.path.dirname(csv_path),
            f'ranking_from_csv_{datetime.now().strftime("%Y%m%d_%H%M")}.html'
        )

    generate_html_report(payload, output_path)

    # Résumé terminal
    print(f"\n{'Rang':<5} {'Joueur':<28} {'Famille':<10} {'Elo':>5} {'W':>5}/{'':<5} {'Win%':>5}")
    print("─" * 70)
    for rank, key in enumerate(ranking, 1):
        w = wins_total[key]
        g = games_total[key]
        pct = 100.0 * w / g if g > 0 else 0.0
        print(f"#{rank:<4} {key:<28} {player_families[key]:<10} {elo[key]:>5.0f} "
              f"{w:>5}/{g:<5} {pct:>4.0f}%")


if __name__ == '__main__':
    main()

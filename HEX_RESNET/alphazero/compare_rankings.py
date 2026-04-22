#!/usr/bin/env python3
"""
compare_rankings.py — Compare plusieurs runs de `ranking.py`.

Scanne le dossier `rank/` (ou celui passé en argument), extrait les données
de chaque rapport HTML (JSON embarqué dans le <script>), et génère un rapport
HTML de comparaison avec des courbes par IA à travers le temps.

Usage :
  python compare_rankings.py
  python compare_rankings.py --dir rank --output rank/comparison.html
"""

import os
import re
import sys
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path


# ─── Parsing d'un rapport ranking_*.html ─────────────────────────────────────

_DATE_RE = re.compile(r'Généré le (\d{2})/(\d{2})/(\d{4}) à (\d{2}):(\d{2})')
_GAMES_RE = re.compile(r'(\d+)\s*parties/matchup')


def _extract_array(text: str, var_name: str):
    m = re.search(rf'const\s+{var_name}\s*=\s*(\[.*?\]);', text, re.DOTALL)
    if not m:
        return None
    return json.loads(m.group(1))


def parse_ranking_html(path: str) -> dict | None:
    """Extrait date + stats agrégées d'un rapport ranking_*.html."""
    text = Path(path).read_text(encoding='utf-8')

    m = _DATE_RE.search(text)
    if not m:
        return None
    dt = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)),
                  int(m.group(4)), int(m.group(5)))

    names = _extract_array(text, 'names')
    elo = _extract_array(text, 'eloData')
    win = _extract_array(text, 'winData')
    times = _extract_array(text, 'timeData')
    moves = _extract_array(text, 'movesData')

    if not (names and elo and win and times and moves):
        return None

    gm = _GAMES_RE.search(text)
    games_per_matchup = int(gm.group(1)) if gm else None

    return {
        'path': path,
        'date': dt,
        'names': names,
        'elo': elo,
        'win': win,
        'time': times,
        'moves': moves,
        'games_per_matchup': games_per_matchup,
    }


# ─── Génération du rapport comparatif ────────────────────────────────────────

_PALETTE = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
    '#1abc9c', '#f1c40f', '#e91e63', '#00bcd4', '#95a5a6',
    '#8e44ad', '#16a085',
]


def _build_series(runs: list[dict], metric: str, all_names: list[str]) -> list[dict]:
    """Une série par IA : valeurs successives à travers les runs (None si absente)."""
    series = []
    for idx, name in enumerate(all_names):
        data = []
        for r in runs:
            if name in r['names']:
                data.append(r[metric][r['names'].index(name)])
            else:
                data.append(None)
        color = _PALETTE[idx % len(_PALETTE)]
        series.append({
            'label': name,
            'data': data,
            'borderColor': color,
            'backgroundColor': color + '33',
            'tension': 0.25,
            'spanGaps': True,
            'pointRadius': 4,
            'borderWidth': 2,
        })
    return series


def build_comparison(runs: list[dict], output_path: str) -> None:
    runs = sorted(runs, key=lambda r: r['date'])
    all_names = sorted({n for r in runs for n in r['names']})

    labels = [r['date'].strftime('%d/%m %H:%M') for r in runs]
    full_dates = [r['date'].strftime('%Y-%m-%d %H:%M') for r in runs]

    elo_series = _build_series(runs, 'elo', all_names)
    win_series = _build_series(runs, 'win', all_names)
    time_series = _build_series(runs, 'time', all_names)
    moves_series = _build_series(runs, 'moves', all_names)

    # Tableau récapitulatif : dernier Elo, delta vs premier run où l'IA apparaît
    summary_rows = []
    for name in all_names:
        first_val = None
        last_val = None
        first_date = None
        last_date = None
        runs_count = 0
        for r in runs:
            if name in r['names']:
                v = r['elo'][r['names'].index(name)]
                if first_val is None:
                    first_val = v
                    first_date = r['date']
                last_val = v
                last_date = r['date']
                runs_count += 1
        delta = (last_val - first_val) if (first_val is not None and last_val is not None) else 0
        summary_rows.append({
            'name': name,
            'first': first_val,
            'last': last_val,
            'delta': delta,
            'runs': runs_count,
            'first_date': first_date.strftime('%d/%m/%Y') if first_date else '-',
            'last_date': last_date.strftime('%d/%m/%Y') if last_date else '-',
        })
    summary_rows.sort(key=lambda r: (r['last'] if r['last'] is not None else -1),
                      reverse=True)

    runs_info = [
        {
            'file': os.path.basename(r['path']),
            'date': r['date'].strftime('%d/%m/%Y %H:%M'),
            'ias': len(r['names']),
            'games': r['games_per_matchup'],
        }
        for r in runs
    ]

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparaison des classements Hex 11×11</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; }}
        h1 {{ text-align: center; color: #00d4ff; margin-bottom: 0.5rem; font-size: 2rem; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{ background: #1a2634; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .card h2 {{ color: #00d4ff; margin-bottom: 1rem; font-size: 1.2rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid #2a3a4a; font-size: 0.9rem; }}
        th {{ color: #00d4ff; font-weight: 600; }}
        tr:hover {{ background: #243447; }}
        .delta-pos {{ color: #2ecc71; font-weight: 600; }}
        .delta-neg {{ color: #e74c3c; font-weight: 600; }}
        .delta-zero {{ color: #888; }}
        .stats-row {{ display: flex; justify-content: space-around; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }}
        .stat-box {{ background: #1a2634; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; min-width: 150px; }}
        .stat-box .value {{ font-size: 1.8rem; font-weight: bold; color: #00d4ff; }}
        .stat-box .label {{ font-size: 0.8rem; color: #888; margin-top: 0.3rem; }}
        canvas {{ max-height: 380px; }}
        .full-width {{ grid-column: 1 / -1; }}
    </style>
</head>
<body>
    <h1>📈 Comparaison des classements Hex 11×11</h1>
    <p class="subtitle">{len(runs)} runs • du {runs[0]['date'].strftime('%d/%m/%Y')} au {runs[-1]['date'].strftime('%d/%m/%Y')} • {len(all_names)} IA suivies • Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>

    <div class="stats-row">
        <div class="stat-box"><div class="value">{len(runs)}</div><div class="label">Runs comparés</div></div>
        <div class="stat-box"><div class="value">{len(all_names)}</div><div class="label">IA suivies</div></div>
        <div class="stat-box"><div class="value">{(runs[-1]['date'] - runs[0]['date']).days + 1}</div><div class="label">Jours couverts</div></div>
    </div>

    <div class="grid">
        <div class="card"><h2>📊 Évolution du score Elo</h2><canvas id="eloChart"></canvas></div>
        <div class="card"><h2>🏅 Évolution du taux de victoire (%)</h2><canvas id="winChart"></canvas></div>
        <div class="card"><h2>⏱️ Temps moyen par coup (s)</h2><canvas id="timeChart"></canvas></div>
        <div class="card"><h2>🎯 Durée moyenne des parties (coups)</h2><canvas id="movesChart"></canvas></div>

        <div class="card full-width">
            <h2>📋 Récapitulatif par IA</h2>
            <table>
                <thead>
                    <tr>
                        <th>IA</th><th>Premier Elo</th><th>Dernier Elo</th>
                        <th>Δ Elo</th><th>Runs</th><th>Période</th>
                    </tr>
                </thead>
                <tbody>
"""
    for row in summary_rows:
        if row['delta'] > 0.5:
            cls = 'delta-pos'
            sign = '+'
        elif row['delta'] < -0.5:
            cls = 'delta-neg'
            sign = ''
        else:
            cls = 'delta-zero'
            sign = ''
        first_s = f"{row['first']:.0f}" if row['first'] is not None else '-'
        last_s = f"{row['last']:.0f}" if row['last'] is not None else '-'
        html += f"""                    <tr>
                        <td><strong>{row['name']}</strong></td>
                        <td>{first_s}</td>
                        <td>{last_s}</td>
                        <td class="{cls}">{sign}{row['delta']:.0f}</td>
                        <td>{row['runs']}</td>
                        <td>{row['first_date']} → {row['last_date']}</td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>

        <div class="card full-width">
            <h2>🗂️ Runs inclus</h2>
            <table>
                <thead><tr><th>#</th><th>Fichier</th><th>Date</th><th>IA</th><th>Parties/matchup</th></tr></thead>
                <tbody>
"""
    for i, info in enumerate(runs_info, 1):
        games_s = str(info['games']) if info['games'] is not None else '-'
        html += f"""                    <tr><td>#{i}</td><td>{info['file']}</td><td>{info['date']}</td><td>{info['ias']}</td><td>{games_s}</td></tr>
"""

    html += """                </tbody>
            </table>
        </div>
    </div>

    <script>
        const labels = """ + json.dumps(labels) + """;
        const fullDates = """ + json.dumps(full_dates) + """;
        const eloSeries = """ + json.dumps(elo_series) + """;
        const winSeries = """ + json.dumps(win_series) + """;
        const timeSeries = """ + json.dumps(time_series) + """;
        const movesSeries = """ + json.dumps(moves_series) + """;

        const baseOpts = {
            responsive: true,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { labels: { color: '#e0e0e0', boxWidth: 12 } },
                tooltip: {
                    callbacks: {
                        title: (items) => fullDates[items[0].dataIndex]
                    }
                }
            },
            scales: {
                x: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } },
                y: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } }
            }
        };

        function makeChart(id, datasets, yMax) {
            const opts = JSON.parse(JSON.stringify(baseOpts));
            if (yMax !== undefined) opts.scales.y.max = yMax;
            opts.plugins.legend.labels.color = '#e0e0e0';
            new Chart(document.getElementById(id), {
                type: 'line',
                data: { labels: labels, datasets: datasets },
                options: opts
            });
        }

        makeChart('eloChart', eloSeries);
        makeChart('winChart', winSeries, 100);
        makeChart('timeChart', timeSeries);
        makeChart('movesChart', movesSeries);
    </script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding='utf-8')
    print(f"Rapport comparatif : {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rank')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un rapport comparatif à partir des rapports ranking_*.html")
    parser.add_argument('--dir', type=str, default=_DEFAULT_DIR,
                        help=f"Dossier contenant les rapports (défaut: {_DEFAULT_DIR})")
    parser.add_argument('--output', type=str, default=None,
                        help="Fichier HTML de sortie (défaut: <dir>/comparison.html)")
    parser.add_argument('--glob', type=str, default='ranking_*.html',
                        help="Motif des fichiers à inclure (défaut: ranking_*.html)")
    args = parser.parse_args()

    pattern = os.path.join(args.dir, args.glob)
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        run = parse_ranking_html(f)
        if run is None:
            print(f"  ignoré (parsing échoué) : {os.path.basename(f)}", file=sys.stderr)
            continue
        runs.append(run)
        print(f"  chargé : {os.path.basename(f)} ({run['date'].strftime('%d/%m/%Y %H:%M')}, {len(run['names'])} IA)")

    if not runs:
        print("Aucun rapport trouvé.", file=sys.stderr)
        sys.exit(1)

    output = args.output or os.path.join(args.dir, 'comparison.html')
    build_comparison(runs, output)


if __name__ == '__main__':
    main()

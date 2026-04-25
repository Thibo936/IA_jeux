#!/usr/bin/env python3
"""
ranking_old.py — Régénère les rapports HTML à partir des CSV bruts de parties.

Scanne les dossiers passés en argument (défaut : rank/ et rank_model/),
trouve les fichiers CSV de parties, recalcule Elo / stats / heatmap,
et génère les rapports HTML manquants.

Usage :
  python ranking_old.py                    # scan rank/ + rank_model/
  python ranking_old.py --force            # régénère même si HTML existe
  python ranking_old.py --dir rank archive # scan dossiers custom
"""

import os
import sys
import csv
import json
import glob
import re
import argparse
from datetime import datetime
from pathlib import Path


# ─── Inférence type/famille ──────────────────────────────────────────────────

def infer_meta(name: str) -> tuple[str, str]:
    """Déduit (type, family) à partir du nom d'un joueur."""
    if name.startswith('AZ-') or name == 'AlphaZero':
        return 'AlphaZero', 'alphazero'
    # Modèles AlphaZero bruts (model_iter_*, best_model_*)
    if name.startswith(('model_iter_', 'best_model')):
        return 'AlphaZero', 'alphazero'
    mapping = {
        'Random': 'Random',
        'AlphaBeta': 'Alpha-Beta',
        'MonteCarlo': 'Monte Carlo',
        'MCTS-Light': 'MCTS',
        'Heuristic': 'Heuristique',
        'MoHex': 'MoHex',
    }
    if name in mapping:
        return mapping[name], 'classic'
    return 'Inconnu', 'unknown'


# ─── Calcul Elo ──────────────────────────────────────────────────────────────

def compute_elo(names, results, k=32.0, initial=1000.0):
    """Itère 10 fois sur les résultats pour converger les Elo."""
    elo = {n: initial for n in names}
    for _ in range(10):
        for (a, b), (wa, wb) in results.items():
            total = wa + wb
            if total == 0:
                continue
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400))
            elo[a] += k * (wa / total - ea)
            elo[b] += k * (wb / total - (1.0 - ea))
    return elo


# ─── Extraction date depuis nom de fichier ───────────────────────────────────

_DATE_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})')


def extract_date(fname: str) -> datetime | None:
    """Extrait YYYY-MM-DD_HHMM depuis le nom de fichier."""
    m = _DATE_RE.search(fname)
    if m:
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5))
        )
    return None


# ─── Parsing CSV → payload JSON ──────────────────────────────────────────────

def parse_csv(csv_path: str) -> dict | None:
    """Lit un CSV de parties brutes et reconstruit le payload du rapport."""
    text = Path(csv_path).read_text(encoding='utf-8')
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        return None

    rows = list(reader)
    if not rows:
        return None

    # Détection du format
    has_family = 'family_a' in reader.fieldnames
    has_type = 'type_a' in reader.fieldnames
    has_sims = 'sims' in reader.fieldnames
    has_time_per_move = 'time_per_move' in reader.fieldnames

    # Date depuis le nom de fichier, fallback sur mtime
    fname = os.path.basename(csv_path)
    dt = extract_date(fname)
    if dt is None:
        dt = datetime.fromtimestamp(os.path.getmtime(csv_path))

    # ── Agrégation ────────────────────────────────────────────────────────────
    player_games: dict[str, int] = {}
    player_wins: dict[str, int] = {}
    player_time_sum: dict[str, float] = {}
    player_moves_sum: dict[str, int] = {}

    results: dict[tuple[str, str], list[int]] = {}
    matchup_counts: dict[tuple[str, str], int] = {}
    total_time = 0.0

    for r in rows:
        name_a = r['name_a']
        name_b = r['name_b']
        a_is_blue = r['a_is_blue'] == '1' or r['a_is_blue'].lower() == 'true'
        winner_name = r['winner_name']
        moves = int(r['total_moves'])
        total_time += float(r['total_time']) if 'total_time' in r else 0.0

        avg_blue = float(r['avg_time_blue']) if 'avg_time_blue' in r else 0.0
        avg_red = float(r['avg_time_red']) if 'avg_time_red' in r else 0.0

        if a_is_blue:
            time_a, time_b = avg_blue, avg_red
        else:
            time_a, time_b = avg_red, avg_blue

        for name, t in ((name_a, time_a), (name_b, time_b)):
            player_games[name] = player_games.get(name, 0) + 1
            player_time_sum[name] = player_time_sum.get(name, 0.0) + t
            player_moves_sum[name] = player_moves_sum.get(name, 0) + moves
            player_wins[name] = player_wins.get(name, 0) + (
                1 if winner_name == name else 0
            )

        key = (name_a, name_b)
        if key not in results:
            results[key] = [0, 0]
            matchup_counts[key] = 0
        matchup_counts[key] += 1
        if winner_name == name_a:
            results[key][0] += 1
        elif winner_name == name_b:
            results[key][1] += 1

    # ── Elo & classement ──────────────────────────────────────────────────────
    names = sorted(player_games.keys())
    elo = compute_elo(names, {k: tuple(v) for k, v in results.items()})
    ranking = sorted(names, key=lambda n: elo[n], reverse=True)

    games_per_matchup = max(matchup_counts.values()) if matchup_counts else 0

    win_pct = {
        n: (100.0 * player_wins.get(n, 0) / player_games[n]) for n in names
    }
    avg_time = {
        n: (player_time_sum.get(n, 0.0) / player_games[n]) for n in names
    }
    avg_moves = {
        n: (player_moves_sum.get(n, 0) / player_games[n]) for n in names
    }

    # ── Types / Familles ──────────────────────────────────────────────────────
    player_types: dict[str, str] = {}
    player_families: dict[str, str] = {}
    for n in names:
        if has_type and has_family:
            found = False
            for r in rows:
                if r['name_a'] == n:
                    player_types[n] = r['type_a']
                    player_families[n] = r['family_a']
                    found = True
                    break
                elif r['name_b'] == n:
                    player_types[n] = r['type_b']
                    player_families[n] = r['family_b']
                    found = True
                    break
            if not found:
                t, f = infer_meta(n)
                player_types[n] = t
                player_families[n] = f
        else:
            t, f = infer_meta(n)
            player_types[n] = t
            player_families[n] = f

    sims = 0
    time_per_move = 0.0
    if has_sims:
        try:
            sims = int(rows[0]['sims'])
        except Exception:
            sims = 0
    if has_time_per_move:
        try:
            time_per_move = float(rows[0]['time_per_move'])
        except Exception:
            time_per_move = 0.0

    return {
        'schema_version': 1,
        'generated_at': dt.isoformat(timespec='seconds'),
        'generated_at_human': dt.strftime('%d/%m/%Y à %H:%M'),
        'games_per_matchup': games_per_matchup,
        'sims': sims,
        'time_per_move': time_per_move,
        'total_time': total_time,
        'names': ranking,
        'playerTypes': player_types,
        'playerFamilies': player_families,
        'playerPaths': {n: None for n in names},
        'elo': {k: float(elo[k]) for k in names},
        'winPct': win_pct,
        'avgTime': avg_time,
        'avgMoves': avg_moves,
        'winsTotal': player_wins,
        'gamesTotal': player_games,
        'results': [
            {'a': a, 'b': b, 'winsA': wa, 'winsB': wb}
            for (a, b), (wa, wb) in {k: tuple(v) for k, v in results.items()}.items()
        ],
    }


# ─── Template HTML (identique à ranking.py) ──────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classement Hex 11×11 — Tournoi unifié</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; }
        h1 { text-align: center; color: #00d4ff; margin-bottom: 0.5rem; font-size: 2rem; }
        .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
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
    <h1>Classement Hex 11×11 — Tournoi unifié</h1>
    <p id="subtitle" class="subtitle"></p>

    <div class="stats-row" id="stats-row"></div>

    <div class="grid">
        <div class="card"><h2>Score Elo</h2><canvas id="eloChart"></canvas></div>
        <div class="card"><h2>Taux de Victoire (%)</h2><canvas id="winChart"></canvas></div>
        <div class="card"><h2>Temps moyen par coup (s)</h2><canvas id="timeChart"></canvas></div>
        <div class="card"><h2>Durée moyenne des parties (coups)</h2><canvas id="movesChart"></canvas></div>
        <div class="card full-width"><h2>Matrice des confrontations (taux de victoire ligne vs colonne)</h2><div id="heatmap" class="heatmap"></div></div>
        <div class="card full-width"><h2>Classement détaillé</h2><div id="table-wrap"></div></div>
    </div>

    <script id="ranking-data" type="application/json">__RANKING_JSON__</script>
    <script>
        const data = JSON.parse(document.getElementById('ranking-data').textContent);

        const TYPE_COLORS = {
            'Random':      '#95a5a6',
            'Alpha-Beta':  '#3498db',
            'Monte Carlo': '#e67e22',
            'MCTS':        '#2ecc71',
            'Heuristique': '#9b59b6',
            'MoHex':       '#1abc9c',
            'AlphaZero':   '#e74c3c',
        };

        function colorFor(name) {
            const t = data.playerTypes[name] || 'Inconnu';
            return TYPE_COLORS[t] || '#34495e';
        }

        // ── Sous-titre & stats ───────────────────────────────────────────────
        const n = data.names.length;
        const totalMatchups = n * (n - 1) / 2;
        const totalGames = data.results.reduce((s, r) => s + r.winsA + r.winsB, 0);
        document.getElementById('subtitle').textContent =
            `Tournoi round-robin • ${n} joueurs • ${totalMatchups} matchups • ${data.games_per_matchup} parties/matchup • ${data.sims} sims/coup • Généré le ${data.generated_at_human}`;

        const statsRow = document.getElementById('stats-row');
        [['Joueurs', n],
         ['Matchups', totalMatchups],
         ['Parties totales', totalGames],
         ['Temps total', `${Math.round(data.total_time)}s`],
         ['Sims/coup', data.sims]
        ].forEach(([label, value]) => {
            statsRow.insertAdjacentHTML('beforeend',
                `<div class="stat-box"><div class="value">${value}</div><div class="label">${label}</div></div>`);
        });

        // ── Charts ───────────────────────────────────────────────────────────
        const names = data.names;  // déjà trié par Elo desc
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

        // ── Heatmap ──────────────────────────────────────────────────────────
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

        // ── Tableau ──────────────────────────────────────────────────────────
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
    """Injecte le payload JSON dans le template et écrit le fichier HTML."""
    json_blob = json.dumps(payload, ensure_ascii=False, indent=2)
    json_blob = json_blob.replace('</', '<\\/')
    html = _HTML_TEMPLATE.replace('__RANKING_JSON__', json_blob)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML : {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Régénère les rapports HTML à partir des CSV de parties.",
    )
    parser.add_argument(
        '--dir', nargs='+', default=['rank', 'rank_model'],
        help="Dossiers à scanner (défaut: rank rank_model)"
    )
    parser.add_argument(
        '--force', action='store_true',
        help="Régénère même si le HTML existe déjà"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Affiche ce qui serait fait sans écrire de fichier"
    )
    args = parser.parse_args()

    dirs = [d for d in args.dir if os.path.isdir(d)]
    if not dirs:
        print("ERREUR : aucun dossier valide trouvé.", file=sys.stderr)
        sys.exit(1)

    csv_files: list[str] = []
    for d in dirs:
        for pattern in ('ranking_*.csv', 'ranking_checkpoint_*.csv'):
            csv_files.extend(glob.glob(os.path.join(d, pattern)))
    csv_files = sorted(set(csv_files))

    processed = 0
    skipped = 0
    errors = 0

    for csv_path in csv_files:
        base = os.path.splitext(csv_path)[0]
        html_path = base + '.html'

        if not args.force and os.path.exists(html_path):
            skipped += 1
            continue

        print(f"Traitement : {os.path.basename(csv_path)}")
        payload = parse_csv(csv_path)
        if payload is None:
            print(f"  ERREUR : parsing échoué", file=sys.stderr)
            errors += 1
            continue

        if args.dry_run:
            print(f"  [dry-run] HTML -> {html_path}")
        else:
            generate_html_report(payload, html_path)
        processed += 1

    print(f"\nTerminé : {processed} généré(s), {skipped} ignoré(s) (HTML existe), {errors} erreur(s)")


if __name__ == '__main__':
    main()

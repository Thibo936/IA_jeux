#!/usr/bin/env python3
"""
compare_rankings.py — Compare plusieurs runs de `ranking.py`.

Scanne le dossier `rank/` (ou celui passé en argument), extrait les données
de chaque rapport HTML (bloc JSON embarqué, avec fallback sur l'ancien format
const eloData/winData/...) ou CSV (données brutes de parties), et génère un
rapport HTML de comparaison qui distingue visuellement deux familles :

  - **classiques** (random, alphabeta, mc_pure, mcts_light, heuristic, mohex)
    → couleur fixe par type d'IA.
  - **alphazero** (best_model.pt + variantes historiques)
    → dégradé d'une couleur unique (rouge), tri lexicographique par nom (≈
    chronologique pour les noms en `AZ-DD_MM`). `AZ-best` est tracé en
    pointillés pour rester distinct.

Les graphes et le tableau récapitulatif sont séparés par famille.

Usage :
  python compare_rankings.py
  python compare_rankings.py --dir rank --output rank/comparison.html
  python compare_rankings.py --family alphazero
  python compare_rankings.py --family classic
"""

import os
import re
import sys
import csv
import glob
import json
import argparse
import colorsys
from datetime import datetime
from pathlib import Path


# ─── Inférence type/famille pour les anciens rapports ────────────────────────

_LEGACY_TYPE_BY_NAME = {
    'Random':     'Random',
    'AlphaBeta':  'Alpha-Beta',
    'MonteCarlo': 'Monte Carlo',
    'MCTS-Light': 'MCTS',
    'Heuristic':  'Heuristique',
    'MoHex':      'MoHex',
}

CLASSIC_COLORS = {
    'Random':      '#95a5a6',
    'Alpha-Beta':  '#3498db',
    'Monte Carlo': '#e67e22',
    'MCTS':        '#2ecc71',
    'Heuristique': '#9b59b6',
    'MoHex':       '#1abc9c',
    'Inconnu':     '#34495e',
}


def _infer_legacy_meta(name: str) -> tuple[str, str]:
    """(type, family) pour un nom issu d'un rapport legacy."""
    if name.startswith('AZ-') or name == 'AlphaZero':
        return 'AlphaZero', 'alphazero'
    if name in _LEGACY_TYPE_BY_NAME:
        return _LEGACY_TYPE_BY_NAME[name], 'classic'
    return 'Inconnu', 'unknown'


# ─── Parsing d'un rapport ranking_*.html ─────────────────────────────────────

_JSON_RE = re.compile(
    r'<script id="ranking-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_DATE_RE = re.compile(r'Généré le (\d{2})/(\d{2})/(\d{4}) à (\d{2}):(\d{2})')
_GAMES_RE = re.compile(r'(\d+)\s*parties/matchup')


def _legacy_extract_array(text: str, var_name: str):
    m = re.search(rf'const\s+{var_name}\s*=\s*(\[.*?\]);', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_ranking_html(path: str) -> dict | None:
    """Extrait date + stats agrégées + types/familles d'un rapport HTML."""
    text = Path(path).read_text(encoding='utf-8')

    m = _JSON_RE.search(text)
    if m:
        # ── Nouveau format : bloc JSON embarqué ───────────────────────────────
        try:
            blob = m.group(1).replace('<\\/', '</')
            data = json.loads(blob)
        except json.JSONDecodeError:
            return None

        names = data.get('names') or []
        if not names:
            return None

        dt = None
        if data.get('generated_at'):
            try:
                dt = datetime.fromisoformat(data['generated_at'])
            except ValueError:
                dt = None
        if dt is None:
            md = _DATE_RE.search(text)
            if md:
                dt = datetime(int(md.group(3)), int(md.group(2)), int(md.group(1)),
                              int(md.group(4)), int(md.group(5)))
            else:
                return None

        try:
            elo   = [data['elo'][n]      for n in names]
            win   = [data['winPct'][n]   for n in names]
            times = [data['avgTime'][n]  for n in names]
            moves = [data['avgMoves'][n] for n in names]
        except KeyError:
            return None

        return {
            'path':              path,
            'date':              dt,
            'names':             names,
            'elo':               elo,
            'win':               win,
            'time':              times,
            'moves':             moves,
            'playerTypes':       dict(data.get('playerTypes', {})),
            'playerFamilies':    dict(data.get('playerFamilies', {})),
            'games_per_matchup': data.get('games_per_matchup'),
        }

    # ── Ancien format : const namesetc. ──────────────────────────────────────
    md = _DATE_RE.search(text)
    if not md:
        return None
    dt = datetime(int(md.group(3)), int(md.group(2)), int(md.group(1)),
                  int(md.group(4)), int(md.group(5)))

    names = _legacy_extract_array(text, 'names')
    elo   = _legacy_extract_array(text, 'eloData')
    win   = _legacy_extract_array(text, 'winData')
    times = _legacy_extract_array(text, 'timeData')
    moves = _legacy_extract_array(text, 'movesData')
    if not (names and elo and win and times and moves):
        return None

    types: dict[str, str] = {}
    families: dict[str, str] = {}
    for n in names:
        t, f = _infer_legacy_meta(n)
        types[n] = t
        families[n] = f

    gm = _GAMES_RE.search(text)
    return {
        'path':              path,
        'date':              dt,
        'names':             names,
        'elo':               elo,
        'win':               win,
        'time':              times,
        'moves':             moves,
        'playerTypes':       types,
        'playerFamilies':    families,
        'games_per_matchup': int(gm.group(1)) if gm else None,
    }


# ─── Calcul Elo (copié de ranking.py) ────────────────────────────────────────

def compute_elo(names: list[str], results: dict[tuple[str, str], tuple[int, int]],
                k: float = 32.0, initial: float = 1000.0) -> dict[str, float]:
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


# ─── Parsing d'un rapport ranking_*.csv (données brutes) ─────────────────────

_CSV_DATE_RE = re.compile(r'ranking_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})')


def parse_ranking_csv(path: str) -> dict | None:
    """Lit un CSV brut de parties et agrège en stats par joueur + Elo."""
    text = Path(path).read_text(encoding='utf-8')

    # Extraction de la date depuis le nom de fichier
    fname = os.path.basename(path)
    md = _CSV_DATE_RE.search(fname)
    if md:
        dt = datetime(int(md.group(1)), int(md.group(2)), int(md.group(3)),
                      int(md.group(4)), int(md.group(5)))
    else:
        return None

    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return None

    # Agrégation par matchup + par joueur
    matchup_results: dict[tuple[str, str], list[dict]] = {}
    player_stats: dict[str, dict] = {}

    for r in rows:
        name_a = r['name_a']
        name_b = r['name_b']
        a_is_blue = r['a_is_blue'] == '1' or r['a_is_blue'].lower() == 'true'
        winner = r['winner']
        winner_name = r['winner_name']
        total_moves = int(r['total_moves'])
        avg_time_blue = float(r['avg_time_blue'])
        avg_time_red = float(r['avg_time_red'])

        # Init joueurs
        for name in (name_a, name_b):
            if name not in player_stats:
                player_stats[name] = {
                    'games': 0, 'wins': 0,
                    'time_sum': 0.0, 'time_count': 0,
                    'moves_sum': 0, 'moves_count': 0,
                }

        # Résultat du matchup
        key = (name_a, name_b)
        if key not in matchup_results:
            matchup_results[key] = []
        matchup_results[key].append(r)

        # Stats joueur
        for name in (name_a, name_b):
            player_stats[name]['games'] += 1
            player_stats[name]['moves_sum'] += total_moves
            player_stats[name]['moves_count'] += 1

        if a_is_blue:
            player_stats[name_a]['time_sum'] += avg_time_blue
            player_stats[name_a]['time_count'] += 1
            player_stats[name_b]['time_sum'] += avg_time_red
            player_stats[name_b]['time_count'] += 1
        else:
            player_stats[name_a]['time_sum'] += avg_time_red
            player_stats[name_a]['time_count'] += 1
            player_stats[name_b]['time_sum'] += avg_time_blue
            player_stats[name_b]['time_count'] += 1

        if winner_name == name_a:
            player_stats[name_a]['wins'] += 1
        elif winner_name == name_b:
            player_stats[name_b]['wins'] += 1
        # Si draw, personne ne gagne

    # Agrégation des matchups pour Elo
    results: dict[tuple[str, str], tuple[int, int]] = {}
    games_per_matchup = 0
    for (name_a, name_b), games in matchup_results.items():
        wins_a = sum(1 for g in games if g['winner_name'] == name_a)
        wins_b = sum(1 for g in games if g['winner_name'] == name_b)
        results[(name_a, name_b)] = (wins_a, wins_b)
        games_per_matchup = max(games_per_matchup, len(games))

    names = sorted(player_stats.keys())
    elo = compute_elo(names, results)

    # Tri par Elo décroissant pour cohérence avec les rapports HTML
    names_sorted = sorted(names, key=lambda n: elo[n], reverse=True)

    elo_list = [elo[n] for n in names_sorted]
    win_list = [
        (player_stats[n]['wins'] / player_stats[n]['games'] * 100)
        if player_stats[n]['games'] > 0 else 0.0
        for n in names_sorted
    ]
    time_list = [
        (player_stats[n]['time_sum'] / player_stats[n]['time_count'])
        if player_stats[n]['time_count'] > 0 else 0.0
        for n in names_sorted
    ]
    moves_list = [
        (player_stats[n]['moves_sum'] / player_stats[n]['moves_count'])
        if player_stats[n]['moves_count'] > 0 else 0.0
        for n in names_sorted
    ]

    types: dict[str, str] = {}
    families: dict[str, str] = {}
    for n in names_sorted:
        t, f = _infer_legacy_meta(n)
        types[n] = t
        families[n] = f

    return {
        'path':              path,
        'date':              dt,
        'names':             names_sorted,
        'elo':               elo_list,
        'win':               win_list,
        'time':              time_list,
        'moves':             moves_list,
        'playerTypes':       types,
        'playerFamilies':    families,
        'games_per_matchup': games_per_matchup if games_per_matchup > 0 else None,
    }


# ─── Couleurs ────────────────────────────────────────────────────────────────

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """h,s,l ∈ [0,1] → '#rrggbb'."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def _az_color(idx: int, n_az: int) -> str:
    """Dégradé rouge : idx ∈ [0, n_az), L varie de 35% (ancien) à 75% (récent)."""
    if n_az <= 1:
        return '#e74c3c'
    t = idx / max(n_az - 1, 1)
    l = 0.35 + 0.40 * t
    return _hsl_to_hex(0.0, 0.70, l)


def _color_for(name: str, ptype: str, family: str,
               az_order: list[str]) -> str:
    if family == 'classic' or family == 'unknown':
        return CLASSIC_COLORS.get(ptype, '#34495e')
    # alphazero
    try:
        idx = az_order.index(name)
    except ValueError:
        idx = 0
    return _az_color(idx, len(az_order))


# ─── Construction des séries ─────────────────────────────────────────────────

def _build_series(runs: list[dict], metric: str, names: list[str],
                  types: dict[str, str], families: dict[str, str],
                  az_order: list[str]) -> list[dict]:
    """Une série Chart.js par AI : valeurs successives à travers les runs."""
    series = []
    for name in names:
        data = []
        for r in runs:
            if name in r['names']:
                data.append(r[metric][r['names'].index(name)])
            else:
                data.append(None)
        family = families.get(name, 'unknown')
        ptype = types.get(name, 'Inconnu')
        color = _color_for(name, ptype, family, az_order)
        ds = {
            'label':           name,
            'data':            data,
            'borderColor':     color,
            'backgroundColor': color + '33',
            'tension':         0.25,
            'spanGaps':        True,
            'pointRadius':     4,
            'borderWidth':     2,
        }
        # AZ-best en pointillés pour le distinguer du dégradé
        if name == 'AZ-best' or (family == 'alphazero' and name == 'AlphaZero'):
            ds['borderDash'] = [4, 4]
            ds['borderWidth'] = 3
        series.append(ds)
    return series


# ─── Génération du rapport comparatif ────────────────────────────────────────

def build_comparison(runs: list[dict], output_path: str,
                     family_filter: str = 'all') -> None:
    runs = sorted(runs, key=lambda r: r['date'])

    # Agrégation des types/familles globaux (privilégie le run le plus récent)
    all_types: dict[str, str] = {}
    all_families: dict[str, str] = {}
    for r in runs:
        for n in r['names']:
            all_types[n] = r['playerTypes'].get(n, all_types.get(n, 'Inconnu'))
            all_families[n] = r['playerFamilies'].get(n, all_families.get(n, 'unknown'))

    # Filtrage par famille demandée
    def _keep(name: str) -> bool:
        f = all_families.get(name, 'unknown')
        if family_filter == 'all':
            return True
        return f == family_filter

    classic_names = sorted(n for n in all_types
                           if all_families.get(n) in ('classic', 'unknown') and _keep(n))
    az_names_unsorted = [n for n in all_types
                         if all_families.get(n) == 'alphazero' and _keep(n)]

    # Ordre AZ : tri lex (≈ chronologique pour AZ-DD_MM) + AZ-best en milieu
    az_order = sorted(az_names_unsorted)

    show_classic = (family_filter in ('all', 'classic')) and bool(classic_names)
    show_az      = (family_filter in ('all', 'alphazero')) and bool(az_order)

    if not (show_classic or show_az):
        print("ERREUR : aucun joueur ne correspond au filtre famille.", file=sys.stderr)
        sys.exit(1)

    labels     = [r['date'].strftime('%d/%m %H:%M')   for r in runs]
    full_dates = [r['date'].strftime('%Y-%m-%d %H:%M') for r in runs]

    classic_elo   = _build_series(runs, 'elo',   classic_names, all_types, all_families, az_order) if show_classic else []
    classic_win   = _build_series(runs, 'win',   classic_names, all_types, all_families, az_order) if show_classic else []
    classic_time  = _build_series(runs, 'time',  classic_names, all_types, all_families, az_order) if show_classic else []
    classic_moves = _build_series(runs, 'moves', classic_names, all_types, all_families, az_order) if show_classic else []

    az_elo   = _build_series(runs, 'elo',   az_order, all_types, all_families, az_order) if show_az else []
    az_win   = _build_series(runs, 'win',   az_order, all_types, all_families, az_order) if show_az else []
    az_time  = _build_series(runs, 'time',  az_order, all_types, all_families, az_order) if show_az else []
    az_moves = _build_series(runs, 'moves', az_order, all_types, all_families, az_order) if show_az else []

    # Récapitulatif par AI : premier/dernier Elo, delta, runs, période
    def _summary(names: list[str]) -> list[dict]:
        rows = []
        for name in names:
            first_val = last_val = None
            first_date = last_date = None
            runs_count = 0
            for r in runs:
                if name in r['names']:
                    v = r['elo'][r['names'].index(name)]
                    if first_val is None:
                        first_val, first_date = v, r['date']
                    last_val, last_date = v, r['date']
                    runs_count += 1
            delta = (last_val - first_val) if (first_val is not None and last_val is not None) else 0
            rows.append({
                'name':       name,
                'type':       all_types.get(name, 'Inconnu'),
                'first':      first_val,
                'last':       last_val,
                'delta':      delta,
                'runs':       runs_count,
                'first_date': first_date.strftime('%d/%m/%Y') if first_date else '-',
                'last_date':  last_date.strftime('%d/%m/%Y')  if last_date  else '-',
            })
        rows.sort(key=lambda r: (r['last'] if r['last'] is not None else -1), reverse=True)
        return rows

    classic_rows = _summary(classic_names) if show_classic else []
    az_rows      = _summary(az_order)      if show_az else []

    runs_info = [
        {
            'file':  os.path.basename(r['path']),
            'date':  r['date'].strftime('%d/%m/%Y %H:%M'),
            'count': len(r['names']),
            'games': r['games_per_matchup'],
        }
        for r in runs
    ]

    # ── Construction du HTML ─────────────────────────────────────────────────
    period_label = (f"du {runs[0]['date'].strftime('%d/%m/%Y')} "
                    f"au {runs[-1]['date'].strftime('%d/%m/%Y')}")
    days_covered = (runs[-1]['date'] - runs[0]['date']).days + 1

    n_classic = len(classic_names)
    n_az = len(az_order)

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; padding: 2rem; }
    h1 { text-align: center; color: #00d4ff; margin-bottom: 0.5rem; font-size: 2rem; }
    .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
    .stats-row { display: flex; justify-content: space-around; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }
    .stat-box { background: #1a2634; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; min-width: 150px; }
    .stat-box .value { font-size: 1.8rem; font-weight: bold; color: #00d4ff; }
    .stat-box .label { font-size: 0.8rem; color: #888; margin-top: 0.3rem; }
    .section-title { color: #00d4ff; font-size: 1.4rem; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #2a3a4a; }
    .family-grid { display: grid; gap: 1.5rem; margin-bottom: 2rem; }
    .family-grid.dual { grid-template-columns: 1fr 1fr; }
    .family-grid.single { grid-template-columns: 1fr; }
    @media (max-width: 1100px) { .family-grid.dual { grid-template-columns: 1fr; } }
    .family-col h3 { color: #00d4ff; font-size: 1.05rem; margin-bottom: 0.7rem; }
    .family-col.alphazero h3 { color: #e74c3c; }
    .card { background: #1a2634; border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 1rem; }
    .card h2 { color: #00d4ff; margin-bottom: 0.8rem; font-size: 1.05rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.4rem; }
    .family-col.alphazero .card h2 { color: #e74c3c; }
    canvas { max-height: 300px; }
    table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
    th, td { padding: 0.5rem 0.7rem; text-align: left; border-bottom: 1px solid #2a3a4a; font-size: 0.85rem; }
    th { color: #00d4ff; font-weight: 600; }
    .alphazero-table th { color: #e74c3c; }
    tr:hover { background: #243447; }
    .delta-pos { color: #2ecc71; font-weight: 600; }
    .delta-neg { color: #e74c3c; font-weight: 600; }
    .delta-zero { color: #888; }
    .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
    """

    head = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparaison des classements Hex 11×11</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>{css}</style>
</head>
<body>
    <h1>Comparaison des classements Hex 11×11</h1>
    <p class="subtitle">{len(runs)} runs • {period_label} • {n_classic} IA classiques • {n_az} variantes AlphaZero • Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}{f' • filtre = {family_filter}' if family_filter != 'all' else ''}</p>

    <div class="stats-row">
        <div class="stat-box"><div class="value">{len(runs)}</div><div class="label">Runs comparés</div></div>
        <div class="stat-box"><div class="value">{n_classic}</div><div class="label">IA classiques</div></div>
        <div class="stat-box"><div class="value">{n_az}</div><div class="label">Variantes AlphaZero</div></div>
        <div class="stat-box"><div class="value">{days_covered}</div><div class="label">Jours couverts</div></div>
    </div>
"""

    # Section graphes (2 colonnes ou 1 selon la famille filtrée)
    grid_class = 'dual' if (show_classic and show_az) else 'single'
    charts_html = f'<div class="family-grid {grid_class}">'

    if show_classic:
        charts_html += """
        <div class="family-col classic">
            <h3>IA classiques</h3>
            <div class="card"><h2>Évolution du score Elo</h2><canvas id="classicElo"></canvas></div>
            <div class="card"><h2>Évolution du taux de victoire (%)</h2><canvas id="classicWin"></canvas></div>
            <div class="card"><h2>Temps moyen par coup (s)</h2><canvas id="classicTime"></canvas></div>
            <div class="card"><h2>Durée moyenne des parties (coups)</h2><canvas id="classicMoves"></canvas></div>
        </div>"""
    if show_az:
        charts_html += """
        <div class="family-col alphazero">
            <h3>Variantes AlphaZero</h3>
            <div class="card"><h2>Évolution du score Elo</h2><canvas id="azElo"></canvas></div>
            <div class="card"><h2>Évolution du taux de victoire (%)</h2><canvas id="azWin"></canvas></div>
            <div class="card"><h2>Temps moyen par coup (s)</h2><canvas id="azTime"></canvas></div>
            <div class="card"><h2>Durée moyenne des parties (coups)</h2><canvas id="azMoves"></canvas></div>
        </div>"""
    charts_html += '</div>'

    # Section tableaux récap
    def _row_html(row: dict, with_type: bool) -> str:
        if row['delta'] > 0.5:
            cls, sign = 'delta-pos', '+'
        elif row['delta'] < -0.5:
            cls, sign = 'delta-neg', ''
        else:
            cls, sign = 'delta-zero', ''
        first_s = f"{row['first']:.0f}" if row['first'] is not None else '-'
        last_s  = f"{row['last']:.0f}"  if row['last']  is not None else '-'
        type_cell = ''
        if with_type:
            c = CLASSIC_COLORS.get(row['type'], '#34495e')
            type_cell = (f'<td><span class="badge" '
                         f'style="background:{c}20;color:{c};border:1px solid {c}40">'
                         f"{row['type']}</span></td>")
        return (f"<tr><td><strong>{row['name']}</strong></td>{type_cell}"
                f"<td>{first_s}</td><td>{last_s}</td>"
                f'<td class="{cls}">{sign}{row["delta"]:.0f}</td>'
                f"<td>{row['runs']}</td>"
                f"<td>{row['first_date']} → {row['last_date']}</td></tr>")

    tables_html = ''
    if show_classic:
        tables_html += '<h2 class="section-title">Récapitulatif IA classiques</h2>'
        tables_html += '<div class="card"><table><thead><tr>'
        tables_html += '<th>IA</th><th>Type</th><th>Premier Elo</th><th>Dernier Elo</th>'
        tables_html += '<th>Δ Elo</th><th>Runs</th><th>Période</th></tr></thead><tbody>'
        for row in classic_rows:
            tables_html += _row_html(row, with_type=True)
        tables_html += '</tbody></table></div>'

    if show_az:
        tables_html += '<h2 class="section-title">Récapitulatif variantes AlphaZero</h2>'
        tables_html += '<div class="card alphazero-table"><table><thead><tr>'
        tables_html += '<th>Variante</th><th>Premier Elo</th><th>Dernier Elo</th>'
        tables_html += '<th>Δ Elo</th><th>Runs</th><th>Période</th></tr></thead><tbody>'
        for row in az_rows:
            tables_html += _row_html(row, with_type=False)
        tables_html += '</tbody></table></div>'

    # Section runs
    runs_html = '<h2 class="section-title">Runs inclus</h2>'
    runs_html += '<div class="card"><table><thead><tr>'
    runs_html += '<th>#</th><th>Fichier</th><th>Date</th><th>Joueurs</th>'
    runs_html += '<th>Parties/matchup</th></tr></thead><tbody>'
    for i, info in enumerate(runs_info, 1):
        games_s = str(info['games']) if info['games'] is not None else '-'
        runs_html += (f"<tr><td>#{i}</td><td>{info['file']}</td>"
                      f"<td>{info['date']}</td><td>{info['count']}</td>"
                      f"<td>{games_s}</td></tr>")
    runs_html += '</tbody></table></div>'

    # JS : configuration des charts
    js = """
    const labels = """ + json.dumps(labels) + """;
    const fullDates = """ + json.dumps(full_dates) + """;

    const baseOpts = {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
            legend: { labels: { color: '#e0e0e0', boxWidth: 12, font: { size: 11 } } },
            tooltip: { callbacks: { title: (items) => fullDates[items[0].dataIndex] } }
        },
        scales: {
            x: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } },
            y: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } }
        }
    };

    function makeChart(id, datasets, yMax) {
        const opts = JSON.parse(JSON.stringify(baseOpts));
        if (yMax !== undefined) opts.scales.y.max = yMax;
        new Chart(document.getElementById(id), {
            type: 'line',
            data: { labels: labels, datasets: datasets },
            options: opts
        });
    }
    """

    if show_classic:
        js += "\n    makeChart('classicElo',   "   + json.dumps(classic_elo)   + ");"
        js += "\n    makeChart('classicWin',   "   + json.dumps(classic_win)   + ", 100);"
        js += "\n    makeChart('classicTime',  "  + json.dumps(classic_time)  + ");"
        js += "\n    makeChart('classicMoves', " + json.dumps(classic_moves) + ");"
    if show_az:
        js += "\n    makeChart('azElo',   "   + json.dumps(az_elo)   + ");"
        js += "\n    makeChart('azWin',   "   + json.dumps(az_win)   + ", 100);"
        js += "\n    makeChart('azTime',  "  + json.dumps(az_time)  + ");"
        js += "\n    makeChart('azMoves', " + json.dumps(az_moves) + ");"

    html = head + charts_html + tables_html + runs_html + f"""
    <script>{js}</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding='utf-8')
    print(f"Rapport comparatif : {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rank')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un rapport comparatif à partir des rapports ranking_*.html et ranking_*.csv",
    )
    parser.add_argument('--dir', type=str, default=_DEFAULT_DIR,
                        help=f"Dossier contenant les rapports (défaut: {_DEFAULT_DIR})")
    parser.add_argument('--output', type=str, default=None,
                        help="Fichier HTML de sortie (défaut: <dir>/comparison.html)")
    parser.add_argument('--glob', type=str, default='ranking_*',
                        help="Motif des fichiers à inclure (défaut: ranking_*)")
    parser.add_argument('--family', choices=('all', 'classic', 'alphazero'),
                        default='all',
                        help="Filtre par famille (défaut: all)")
    args = parser.parse_args()

    # On scanne HTML et CSV
    pattern = os.path.join(args.dir, args.glob)
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        if f.endswith('.csv'):
            run = parse_ranking_csv(f)
        elif f.endswith('.html'):
            run = parse_ranking_html(f)
        else:
            continue
        if run is None:
            print(f"  ignoré (parsing échoué) : {os.path.basename(f)}",
                  file=sys.stderr)
            continue
        runs.append(run)
        print(f"  chargé : {os.path.basename(f)} "
              f"({run['date'].strftime('%d/%m/%Y %H:%M')}, "
              f"{len(run['names'])} joueurs)")

    if not runs:
        print("Aucun rapport trouvé.", file=sys.stderr)
        sys.exit(1)

    output = args.output or os.path.join(args.dir, 'comparison.html')
    build_comparison(runs, output, family_filter=args.family)


if __name__ == '__main__':
    main()

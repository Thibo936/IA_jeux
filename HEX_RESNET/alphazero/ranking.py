#!/usr/bin/env python3
"""
ranking.py — Tournoi round-robin unifié Hex 11×11 (outil unique).

Fait jouer ensemble dans un même round-robin :
  - Les IA classiques  : random, alphabeta, mc_pure, mcts_light, heuristic, mohex
  - Toutes les variantes AlphaZero : .pt du dossier model/ + checkpoints/best_model.pt

Usage :
  python ranking.py                          # tout par défaut, 50 parties/matchup
  python ranking.py --games 50               # 50 parties/matchup
  python ranking.py --no-models              # sans les .pt de model/
  python ranking.py --no-classics            # tournoi 100% AlphaZero
  python ranking.py --mode classic           # seulement classiques
  python ranking.py --mode alphazero         # seulement AZ
  python ranking.py --workers 2 --game-threads 2

Le CSV est incrémental : seuls les matchups manquants sont joués.
Le HTML est régénéré à chaque exécution.
"""

import os
import sys
import argparse
import csv
import time
import json
import io
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_dir, os.path.join(_dir, 'ia'), os.path.join(_dir, 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BEST_MODEL_FILE, MODEL_DIR
import model_naming


# ─── Constantes & catégorisation ─────────────────────────────────────────────

DEFAULT_CLASSICS = ['random', 'alphabeta', 'mc_pure', 'mcts_light', 'heuristic', 'mohex']

CLASSIC_DISPLAY = {
    'random':     'Random',
    'alphabeta':  'AlphaBeta',
    'mc_pure':    'MonteCarlo',
    'montecarlo': 'MonteCarlo',
    'mcts_light': 'MCTS-Light',
    'mcts':       'MCTS-Light',
    'heuristic':  'Heuristic',
    'mohex':      'MoHex',
}

CLASSIC_TYPE = {
    'random':     'Random',
    'alphabeta':  'Alpha-Beta',
    'mc_pure':    'Monte Carlo',
    'montecarlo': 'Monte Carlo',
    'mcts_light': 'MCTS',
    'mcts':       'MCTS',
    'heuristic':  'Heuristique',
    'mohex':      'MoHex',
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

TIME_PER_MOVE = 0.5


# ─── Inférence type/famille (compatibilité legacy) ──────────────────────────

_LEGACY_TYPE_BY_NAME = {
    'Random':     'Random',
    'AlphaBeta':  'Alpha-Beta',
    'MonteCarlo': 'Monte Carlo',
    'MCTS-Light': 'MCTS',
    'Heuristic':  'Heuristique',
    'MoHex':      'MoHex',
}


def infer_meta(name: str) -> tuple[str, str]:
    """(type, family) pour un nom de joueur."""
    if name.startswith('AZ-') or name == 'AlphaZero':
        return 'AlphaZero', 'alphazero'
    if name.startswith(('model_', 'best_model')):
        return 'AlphaZero', 'alphazero'
    if name in _LEGACY_TYPE_BY_NAME:
        return _LEGACY_TYPE_BY_NAME[name], 'classic'
    return 'Inconnu', 'unknown'


# ─── Joueurs classiques ──────────────────────────────────────────────────────

import importlib


def _make_classic(classic_id: str):
    """Instancie une IA classique (legacy ou découverte dynamiquement dans ia/)."""
    n = classic_id.lower()
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

    # Fallback dynamique : importer le module par son nom de fichier
    try:
        mod = importlib.import_module(classic_id)
    except Exception as exc:
        raise ValueError(f"IA classique inconnue : {classic_id}") from exc

    # Cherche une classe avec select_move (et pas un helper interne)
    candidates = []
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if isinstance(obj, type) and hasattr(obj, 'select_move'):
            candidates.append(obj)

    if not candidates:
        raise ValueError(f"IA classique inconnue : {classic_id}")

    # Préfère celle qui finit par 'Player' si elle existe
    for c in candidates:
        if c.__name__.endswith('Player'):
            return c()

    return candidates[0]()


_CLASSIC_DENYLIST = {'__init__', 'humain', 'katahex', 'mcts_az'}


def discover_classic_players(ia_dir: str) -> list[str]:
    """Scanne ia/ et retourne les stems de tous les .py joueurs disponibles."""
    if not os.path.isdir(ia_dir):
        return DEFAULT_CLASSICS
    discovered: list[str] = []
    for fname in sorted(os.listdir(ia_dir)):
        if not fname.endswith('.py'):
            continue
        stem = os.path.splitext(fname)[0]
        if stem in _CLASSIC_DENYLIST:
            continue
        discovered.append(stem)
    return discovered if discovered else DEFAULT_CLASSICS


# ─── Découverte des modèles AlphaZero ────────────────────────────────────────

def _az_name_from_path(path: str) -> str:
    """
    Nommage déterministe pour un .pt AlphaZero.
      checkpoints/best_model.pt          → AZ-best
      model/model_01_00_17_04.pt         → AZ-01
      model/model_03_01_18_04.pt         → AZ-03
    """
    base = os.path.splitext(os.path.basename(path))[0]
    if base == 'best_model':
        return 'AZ-best'
    if base.startswith('model_'):
        # model_NN_PP_DD_MM → prendre le numéro NN
        parts = base.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            return f"AZ-{int(parts[1]):02d}"
    return 'AZ-' + base


def discover_alphazero_models(model_dir: str) -> list[tuple[str, str]]:
    """Liste [(name, abs_path), ...] des .pt du dossier (vide si absent)."""
    if not os.path.isdir(model_dir):
        return []
    pts = sorted(f for f in os.listdir(model_dir) if f.endswith('.pt'))
    return [(_az_name_from_path(f), os.path.abspath(os.path.join(model_dir, f)))
            for f in pts]


# ─── Registre des joueurs ────────────────────────────────────────────────────

@dataclass
class PlayerEntry:
    key: str
    family: str
    type: str
    classic_id: str | None = None
    source_path: str | None = None
    player: object = field(default=None)


def build_player_registry(args) -> list[PlayerEntry]:
    """Construit le registre dédupliqué (pas encore chargé)."""
    entries: list[PlayerEntry] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()

    def _unique_key(base: str) -> str:
        if base not in seen_keys:
            return base
        i = 2
        while f"{base}_{i}" in seen_keys:
            i += 1
        return f"{base}_{i}"

    def _add(entry: PlayerEntry) -> None:
        if entry.source_path:
            ap = os.path.abspath(entry.source_path)
            if ap in seen_paths:
                return
            seen_paths.add(ap)
            entry.source_path = ap
        entry.key = _unique_key(entry.key)
        seen_keys.add(entry.key)
        entries.append(entry)

    # 1. Classiques
    if not args.no_classics:
        if args.ias:
            ias = args.ias
        else:
            ias = discover_classic_players(os.path.join(_dir, 'ia'))
        for ia in ias:
            n = ia.lower()
            display = CLASSIC_DISPLAY.get(n, ia)
            ptype = CLASSIC_TYPE.get(n, 'Inconnu')
            _add(PlayerEntry(key=display, family='classic',
                             type=ptype, classic_id=n))

    # 2. .pt du dossier model/
    if not args.no_models:
        model_dir = args.model_dir if args.model_dir else os.path.join(_dir, MODEL_DIR)
        for name, path in discover_alphazero_models(model_dir):
            _add(PlayerEntry(key=name, family='alphazero',
                             type='AlphaZero', source_path=path))

    # 4. --add
    for p in args.add or []:
        if p.lower() in ('best', 'best_model', 'best_model.pt'):
            ap = os.path.join(_dir, BEST_MODEL_FILE)
            if not os.path.isfile(ap):
                ap = BEST_MODEL_FILE
        else:
            ap = p
        if not os.path.isfile(ap):
            alt = os.path.join(_dir, ap)
            if os.path.isfile(alt):
                ap = alt
            else:
                print(f"WARN : --add ignoré (introuvable) : {p}", file=sys.stderr)
                continue
        _add(PlayerEntry(key=_az_name_from_path(ap), family='alphazero',
                         type='AlphaZero', source_path=ap))

    return entries


def instantiate_players(entries: list[PlayerEntry], device, sims: int) -> list[PlayerEntry]:
    """Charge chaque joueur. Skip ceux qui échouent à l'init."""
    from tournament import AlphaZeroPlayer
    valid: list[PlayerEntry] = []
    for e in entries:
        try:
            if e.family == 'classic':
                e.player = _make_classic(e.classic_id)
            else:
                e.player = AlphaZeroPlayer(model_path=e.source_path,
                                           device=device, sims=sims)
        except Exception as exc:
            print(f"  SKIP {e.key:<28} ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            continue
        suffix = f" ← {os.path.relpath(e.source_path, _dir)}" if e.source_path else ""
        print(f"  OK   {e.key:<28} [{e.type}]{suffix}")
        valid.append(e)
    return valid


# ─── Lecture CSV incrémental ─────────────────────────────────────────────────

def read_csv_db(csv_path: str) -> tuple[list[dict], dict[tuple[str, str], int]]:
    """
    Lit le CSV existant. Retourne (lignes, matchup_counts).
    matchup_counts : {(name_a, name_b): nombre de parties déjà jouées}
    """
    rows: list[dict] = []
    counts: dict[tuple[str, str], int] = {}
    if not os.path.isfile(csv_path):
        return rows, counts
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            na, nb = r['name_a'], r['name_b']
            key = tuple(sorted([na, nb]))
            counts[key] = counts.get(key, 0) + 1
    return rows, counts


# ─── Partie ──────────────────────────────────────────────────────────────────

def play_game(player_blue, player_red, game_id: int) -> dict:
    """Joue une partie. player_blue joue Blue, player_red joue Red."""
    env = HexEnv()
    times_blue: list[float] = []
    times_red: list[float] = []
    total_moves = 0
    winner: str | None = None
    t_start = time.time()

    while not env.is_terminal():
        cur = player_blue if env.blue_to_play else player_red
        t0 = time.time()
        with contextlib.redirect_stderr(io.StringIO()):
            move = cur.select_move(env, TIME_PER_MOVE)
        elapsed = time.time() - t0

        if env.blue_to_play:
            times_blue.append(elapsed)
        else:
            times_red.append(elapsed)

        if move < 0 or move >= NUM_CELLS:
            winner = 'red' if env.blue_to_play else 'blue'
            break
        r, c = divmod(move, 11)
        if env.blue[r, c] or env.red[r, c]:
            winner = 'red' if env.blue_to_play else 'blue'
            break

        env.apply_move(move)
        total_moves += 1

    if winner is None:
        winner = env.winner()

    return {
        'game_id':       game_id,
        'winner':        winner,
        'total_moves':   total_moves,
        'total_time':    time.time() - t_start,
        'avg_time_blue': float(np.mean(times_blue)) if times_blue else 0.0,
        'avg_time_red':  float(np.mean(times_red))  if times_red  else 0.0,
    }


def _play_one_game(task) -> dict:
    player_a, player_b, name_a, name_b, game_index = task
    a_is_blue = (game_index % 2 == 0)
    if a_is_blue:
        g = play_game(player_a, player_b, game_index + 1)
        blue_name, red_name = name_a, name_b
    else:
        g = play_game(player_b, player_a, game_index + 1)
        blue_name, red_name = name_b, name_a
    g['a_is_blue']   = a_is_blue
    g['blue_name']   = blue_name
    g['red_name']    = red_name
    g['winner_name'] = blue_name if g['winner'] == 'blue' else red_name
    return g


def match(player_a, player_b, name_a: str, name_b: str,
          num_games: int, game_threads: int = 1) -> dict:
    """Round entre player_a et player_b sur num_games parties (couleurs alternées)."""
    if game_threads > 1:
        tasks = [(deepcopy(player_a), deepcopy(player_b), name_a, name_b, i)
                 for i in range(num_games)]
        with ThreadPoolExecutor(max_workers=game_threads) as pool:
            games = list(pool.map(_play_one_game, tasks))
    else:
        games = [_play_one_game((player_a, player_b, name_a, name_b, i))
                 for i in range(num_games)]

    wins_a = wins_b = 0
    times_a_all: list[float] = []
    times_b_all: list[float] = []
    moves_all: list[int] = []
    for g in games:
        a_is_blue = g['a_is_blue']
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


# ─── Calcul Elo ──────────────────────────────────────────────────────────────

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


# ─── HTML template ───────────────────────────────────────────────────────────

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
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
  .card { background: #1a2634; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .card h2 { color: #00d4ff; margin-bottom: 1rem; font-size: 1.2rem; border-bottom: 1px solid #2a3a4a; padding-bottom: 0.5rem; }
  .card.az h2 { color: #e74c3c; }
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
  .section-title { color: #00d4ff; font-size: 1.4rem; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #2a3a4a; }
  .az-title { color: #e74c3c; }
  .delta-pos { color: #2ecc71; font-weight: 600; }
  .delta-neg { color: #e74c3c; font-weight: 600; }
  .delta-zero { color: #888; }
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
  <div class="card full-width"><h2>Évolution temporelle — Elo</h2><canvas id="evoElo"></canvas></div>
  <div class="card full-width"><h2>Évolution temporelle — Win%</h2><canvas id="evoWin"></canvas></div>
  <div class="card full-width"><h2>Classement détaillé</h2><div id="table-wrap"></div></div>
</div>

<script id="ranking-data" type="application/json">__RANKING_JSON__</script>
<script>
const data = JSON.parse(document.getElementById('ranking-data').textContent);

const TYPE_COLORS = {
    'Random': '#95a5a6',
    'Alpha-Beta': '#3498db',
    'Monte Carlo': '#e67e22',
    'MCTS': '#2ecc71',
    'Heuristique': '#9b59b6',
    'MoHex': '#1abc9c',
    'AlphaZero': '#e74c3c',
};

function colorFor(name) {
    const t = data.playerTypes[name] || 'Inconnu';
    return TYPE_COLORS[t] || '#34495e';
}

// ── Sous-titre & stats ────────────────────────────────────────────────────
const n = data.names.length;
const totalMatchups = n * (n - 1) / 2;
const totalGames = data.results.reduce((s, r) => s + r.winsA + r.winsB, 0);
const runsCount = data.runs ? data.runs.length : 1;
document.getElementById('subtitle').textContent =
    `Tournoi round-robin • ${n} joueurs • ${totalMatchups} matchups • ${data.games_per_matchup} parties/matchup • ${data.sims} sims/coup • ${runsCount} runs • Généré le ${data.generated_at_human}`;

const statsRow = document.getElementById('stats-row');
[['Joueurs', n], ['Matchups', totalMatchups], ['Parties totales', totalGames],
 ['Runs', runsCount], ['Temps total', `${Math.round(data.total_time)}s`],
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

// ── Évolution temporelle (courbes) ────────────────────────────────────────
if (data.runs && data.runs.length > 1) {
    const evoLabels = data.runs.map(r => r.date);
    const evoDates = data.runs.map(r => r.fullDate);

    function buildEvoSeries(metric, allNames) {
        return allNames.map(name => {
            const color = colorFor(name);
            return {
                label: name,
                data: data.runs.map(r => r[metric][name] !== undefined ? r[metric][name] : null),
                borderColor: color,
                backgroundColor: color + '33',
                tension: 0.25,
                spanGaps: true,
                pointRadius: 4,
                borderWidth: 2
            };
        });
    }

    const evoOpts = {
        responsive: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
            legend: { labels: { color: '#e0e0e0', boxWidth: 12, font: { size: 11 } } },
            tooltip: { callbacks: { title: (items) => evoDates[items[0].dataIndex] } }
        },
        scales: {
            x: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } },
            y: { grid: { color: '#2a3a4a' }, ticks: { color: '#aaa' } }
        }
    };

    new Chart(document.getElementById('evoElo'), {
        type: 'line',
        data: { labels: evoLabels, datasets: buildEvoSeries('elo', data.allNames) },
        options: evoOpts
    });
    new Chart(document.getElementById('evoWin'), {
        type: 'line',
        data: { labels: evoLabels, datasets: buildEvoSeries('win', data.allNames) },
        options: { ...evoOpts, scales: { ...evoOpts.scales, y: { ...evoOpts.scales.y, max: 100 } } }
    });
}

// ── Tableau ───────────────────────────────────────────────────────────────
let tableHTML = `<table><thead><tr>
<th>Rang</th><th>Joueur</th><th>Modèle</th><th>Famille</th><th>Type</th><th>Elo</th>
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
        <td>${data.playerFiles[nm] || '-'}</td>
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
    parser = argparse.ArgumentParser(
        description="Tournoi round-robin unifié Hex 11×11 (classiques + AlphaZero)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ias', nargs='+', default=None,
                        help=f"IA classiques (défaut: {DEFAULT_CLASSICS})")
    parser.add_argument('--no-classics', action='store_true',
                        help="Aucune IA classique")
    parser.add_argument('--no-models', action='store_true',
                        help="Ignore les .pt de --model-dir")
    parser.add_argument('--model-dir', type=str,
                        default=os.path.join(_dir, MODEL_DIR),
                        help=f"Dossier des .pt AZ (défaut: {MODEL_DIR})")
    parser.add_argument('--add', nargs='*', default=[], metavar='PATH',
                        help=".pt supplémentaires")
    parser.add_argument('--games', type=int, default=50,
                        help="Parties par matchup (défaut: 50)")
    parser.add_argument('--sims', type=int, default=600,
                        help="Simulations MCTS AZ (défaut: 600)")
    parser.add_argument('--time', type=float, default=0.5,
                        help="Budget temps classiques (défaut: 0.5s)")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_dir, 'rank'),
                        help="Dossier de sortie (défaut: rank/)")
    parser.add_argument('--csv', type=str, default=None,
                        help="Fichier CSV (défaut: rank/ranking.csv)")
    parser.add_argument('--output', type=str, default=None,
                        help="Fichier HTML (défaut: rank/ranking.html)")
    parser.add_argument('--no-html', action='store_true',
                        help="Pas de HTML")
    parser.add_argument('--no-csv', action='store_true',
                        help="Pas de CSV")
    parser.add_argument('--device', type=str, default=None,
                        help="Device (cuda/cpu, défaut: auto)")
    parser.add_argument('--workers', type=int, default=1,
                        help="Matchups parallèles (défaut: 1)")
    parser.add_argument('--game-threads', type=int, default=1,
                        help="Parties parallèles par matchup (défaut: 1)")
    parser.add_argument('--mode', choices=['all', 'classic', 'alphazero'],
                        default='all', help="Filtre par famille (défaut: all)")
    args = parser.parse_args()

    global TIME_PER_MOVE
    TIME_PER_MOVE = args.time

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Registre
    print("Construction du registre des joueurs...")
    entries = build_player_registry(args)
    if not entries:
        print("ERREUR : aucun joueur sélectionné.", file=sys.stderr)
        sys.exit(1)

    az_count = sum(1 for e in entries if e.family == 'alphazero')
    if az_count > 10:
        print(f"AVERT : {az_count} modèles AZ — charge VRAM élevée.",
              file=sys.stderr)

    print(f"Chargement de {len(entries)} joueurs sur {device}...")
    valid = instantiate_players(entries, device, args.sims)
    if len(valid) < 2:
        print("ERREUR : moins de 2 joueurs valides.", file=sys.stderr)
        sys.exit(1)

    # Filtrage par mode
    if args.mode == 'classic':
        valid = [e for e in valid if e.family == 'classic']
    elif args.mode == 'alphazero':
        valid = [e for e in valid if e.family == 'alphazero']

    if len(valid) < 2:
        print(f"ERREUR : moins de 2 joueurs après filtre --mode {args.mode}.",
              file=sys.stderr)
        sys.exit(1)

    n = len(valid)
    matchups = list(combinations(range(n), 2))
    total_matchups = len(matchups)

    # 2. Fichiers
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(args.output_dir, 'ranking.csv')
    html_path = args.output or os.path.join(
        args.output_dir,
        f'ranking_{datetime.now().strftime("%Y%m%d_%H%M")}.html'
    )

    # 3. Lecture CSV incrémental
    existing_matchups: dict[tuple[str, str], dict] = {}
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                na, nb = r['name_a'], r['name_b']
                key = (na, nb)
                if key not in existing_matchups:
                    existing_matchups[key] = {
                        'wins_a': 0, 'wins_b': 0, 'games': 0,
                        'sum_time_a': 0.0, 'sum_time_b': 0.0, 'sum_moves': 0,
                    }
                m = existing_matchups[key]
                m['games'] += 1
                if r['winner_name'] == na:
                    m['wins_a'] += 1
                else:
                    m['wins_b'] += 1
                a_is_blue = int(r['a_is_blue'])
                if a_is_blue:
                    m['sum_time_a'] += float(r['avg_time_blue'])
                    m['sum_time_b'] += float(r['avg_time_red'])
                else:
                    m['sum_time_a'] += float(r['avg_time_red'])
                    m['sum_time_b'] += float(r['avg_time_blue'])
                m['sum_moves'] += int(r['total_moves'])
        print(f"CSV existant : {sum(m['games'] for m in existing_matchups.values())} lignes, "
              f"{len(existing_matchups)} matchups déjà joués.")
    else:
        print("Aucun CSV existant, tournoi complet.")

    # 4. Calcul des matchups manquants
    by_key = {e.key: e for e in valid}
    results: dict[tuple[str, str], tuple[int, int]] = {}
    wins_total = {e.key: 0 for e in valid}
    games_total = {e.key: 0 for e in valid}
    times_total = {e.key: 0.0 for e in valid}
    moves_total = {e.key: 0.0 for e in valid}

    pair_keys = [(valid[i].key, valid[j].key) for i, j in matchups]
    matchup_count = 0
    t_start = time.time()

    def _run_matchup(name_a: str, name_b: str, num_games: int) -> dict:
        if args.workers > 1:
            pa = deepcopy(by_key[name_a].player)
            pb = deepcopy(by_key[name_b].player)
        else:
            pa = by_key[name_a].player
            pb = by_key[name_b].player
        return match(pa, pb, name_a, name_b, num_games,
                     game_threads=args.game_threads)

    csv_file = None
    csv_writer = None
    if not args.no_csv:
        csv_is_new = not os.path.isfile(csv_path)
        csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        if csv_is_new:
            csv_writer.writerow([
                'run_timestamp', 'name_a', 'name_b',
                'game_index', 'a_is_blue',
                'winner_name', 'total_moves', 'total_time',
                'avg_time_blue', 'avg_time_red',
                'sims', 'time_per_move',
            ])

    for na, nb in pair_keys:
        matchup_count += 1
        key = (na, nb)
        existing = existing_matchups.get(key, {'wins_a': 0, 'wins_b': 0, 'games': 0,
                                                'sum_time_a': 0.0, 'sum_time_b': 0.0, 'sum_moves': 0})
        missing = args.games - existing['games']

        if missing <= 0:
            sys.stdout.write(f"\r  [{matchup_count}/{total_matchups}] {na} vs {nb} "
                             f"({existing['games']}/{args.games} déjà jouées) — skip\n")
            sys.stdout.flush()
            results[key] = (existing['wins_a'], existing['wins_b'])
            wins_total[na] += existing['wins_a']
            wins_total[nb] += existing['wins_b']
            games_total[na] += existing['games']
            games_total[nb] += existing['games']
            times_total[na] += existing['sum_time_a']
            times_total[nb] += existing['sum_time_b']
            moves_total[na] += existing['sum_moves']
            moves_total[nb] += existing['sum_moves']
            continue

        sys.stdout.write(f"\r  [{matchup_count}/{total_matchups}] {na} vs {nb} "
                         f"(+{missing} parties) ...")
        sys.stdout.flush()
        mr = _run_matchup(na, nb, missing)

        if csv_writer is not None:
            for g in mr['games']:
                csv_writer.writerow([
                    datetime.now().strftime('%Y-%m-%d_%H%M'),
                    na, nb,
                    g['game_id'], int(g['a_is_blue']),
                    g['winner_name'], g['total_moves'],
                    f"{g['total_time']:.4f}",
                    f"{g['avg_time_blue']:.6f}", f"{g['avg_time_red']:.6f}",
                    args.sims, args.time,
                ])
            csv_file.flush()

        total_wins_a = existing['wins_a'] + mr['wins_a']
        total_wins_b = existing['wins_b'] + mr['wins_b']
        results[key] = (total_wins_a, total_wins_b)
        wins_total[na] += total_wins_a
        wins_total[nb] += total_wins_b
        games_total[na] += args.games
        games_total[nb] += args.games
        times_total[na] += existing['sum_time_a'] + mr['avg_time_a'] * missing
        times_total[nb] += existing['sum_time_b'] + mr['avg_time_b'] * missing
        moves_total[na] += existing['sum_moves'] + mr['avg_moves'] * missing
        moves_total[nb] += existing['sum_moves'] + mr['avg_moves'] * missing

        sys.stdout.write(
            f"\r  [{matchup_count}/{total_matchups}] "
            f"{na} {total_wins_a}-{total_wins_b} {nb}"
            + " " * 20 + "\n"
        )
        sys.stdout.flush()

    total_time = time.time() - t_start

    if csv_file is not None:
        csv_file.close()
        print(f"CSV : {csv_path}")

    # 5. Elo + classement
    valid_keys = [e.key for e in valid]
    elo = compute_elo(valid_keys, results)
    ranking = sorted(valid_keys, key=lambda k: elo[k], reverse=True)

    print(f"\n{'Rang':<5} {'Joueur':<28} {'Famille':<10} {'Elo':>5} "
          f"{'W':>5}/{'':<5} {'Win%':>5}")
    print("─" * 70)
    for rank, key in enumerate(ranking, 1):
        e = by_key[key]
        w = wins_total[key]
        g = games_total[key]
        pct = 100.0 * w / g if g > 0 else 0.0
        marker = " ← MEILLEUR" if rank == 1 else ""
        print(f"#{rank:<4} {key:<28} {e.family:<10} {elo[key]:>5.0f} "
              f"{w:>5}/{g:<5} {pct:>4.0f}%{marker}")
    print(f"\nTemps total : {total_time:.0f}s")

    # 6. HTML
    if not args.no_html:
        win_pct = {k: (100.0 * wins_total[k] / games_total[k]
                       if games_total[k] > 0 else 0.0) for k in valid_keys}
        avg_time = {k: (times_total[k] / games_total[k]
                        if games_total[k] > 0 else 0.0) for k in valid_keys}
        avg_moves = {k: (moves_total[k] / games_total[k]
                         if games_total[k] > 0 else 0.0) for k in valid_keys}

        payload = {
            'schema_version':    2,
            'generated_at':      datetime.now().isoformat(timespec='seconds'),
            'generated_at_human': datetime.now().strftime('%d/%m/%Y à %H:%M'),
            'games_per_matchup': args.games,
            'sims':              args.sims,
            'time_per_move':     args.time,
            'total_time':        total_time,
            'names':             ranking,
            'playerTypes':       {k: by_key[k].type   for k in valid_keys},
            'playerFamilies':    {k: by_key[k].family for k in valid_keys},
            'playerPaths':       {k: (os.path.relpath(by_key[k].source_path, _dir)
                                      if by_key[k].source_path else None)
                                  for k in valid_keys},
            'playerFiles':       {k: (os.path.basename(by_key[k].source_path)
                                      if by_key[k].source_path else None)
                                  for k in valid_keys},
            'elo':               {k: float(elo[k])   for k in valid_keys},
            'winPct':            win_pct,
            'avgTime':           avg_time,
            'avgMoves':          avg_moves,
            'winsTotal':         {k: int(wins_total[k])  for k in valid_keys},
            'gamesTotal':        {k: int(games_total[k]) for k in valid_keys},
            'results': [
                {'a': a, 'b': b, 'winsA': wa, 'winsB': wb}
                for (a, b), (wa, wb) in results.items()
            ],
        }
        generate_html_report(payload, html_path)


if __name__ == '__main__':
    main()

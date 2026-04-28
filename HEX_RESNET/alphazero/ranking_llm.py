#!/usr/bin/env python3
"""
ranking_llm.py — Tournoi round-robin lite pour les IA du dossier model_LLM/.

100 parties par duel, couleurs alternées. Sortie terminal + log dans
model_LLM/modelLLM.log. Pas de HTML, pas de CSV.

Si un modèle échoue à se charger ou crash en partie, il prend 0 sur les
parties restantes et le tournoi continue.
"""

import os
import sys
import time
import inspect
import importlib.util
import io
import contextlib
import traceback
from itertools import combinations
from datetime import datetime

_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_dir, os.path.join(_dir, 'ia'), os.path.join(_dir, 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS

MODEL_LLM_DIR = os.path.join(_dir, 'model_LLM')
LOG_PATH = os.path.join(MODEL_LLM_DIR, 'modelLLM.log')
GAMES_PER_MATCHUP = 20
TIME_PER_MOVE = 0.5


# ─── Tee : écrit terminal + log ──────────────────────────────────────────────

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# ─── Chargement dynamique des modèles ────────────────────────────────────────

def load_player_module(path: str):
    """Importe un .py isolément. Retourne le module ou lève."""
    name = 'llm_' + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def find_player_class(mod):
    """Trouve la classe avec une méthode select_move."""
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != mod.__name__:
            continue
        if hasattr(cls, 'select_move') and callable(getattr(cls, 'select_move')):
            return cls
    return None


def discover_llm_players():
    """Liste [(display_name, file, instance_or_None, error_or_None)]."""
    out = []
    if not os.path.isdir(MODEL_LLM_DIR):
        return out
    files = sorted(f for f in os.listdir(MODEL_LLM_DIR) if f.endswith('.py'))
    for f in files:
        path = os.path.join(MODEL_LLM_DIR, f)
        display = f[:-3]
        try:
            mod = load_player_module(path)
            cls = find_player_class(mod)
            if cls is None:
                out.append((display, f, None, "aucune classe avec select_move"))
                continue
            display = getattr(mod, 'MODEL_NAME', None) or cls.__name__
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                inst = cls()
            out.append((display, f, inst, None))
        except Exception as e:
            out.append((display, f, None, f"{type(e).__name__}: {e}"))
    return out


# ─── Partie ──────────────────────────────────────────────────────────────────

def play_game(player_blue, player_red):
    """
    Joue une partie. Retourne ('blue'|'red', faulty_side|None).
    faulty_side = côté qui a planté ou joué un coup illégal/exception.
    """
    env = HexEnv()
    while not env.is_terminal():
        cur = player_blue if env.blue_to_play else player_red
        side = 'blue' if env.blue_to_play else 'red'
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                move = cur.select_move(env, TIME_PER_MOVE)
        except Exception:
            return ('red' if side == 'blue' else 'blue'), side

        if move is None or move < 0 or move >= NUM_CELLS:
            return ('red' if side == 'blue' else 'blue'), side
        r, c = divmod(int(move), 11)
        if env.blue[r, c] or env.red[r, c]:
            return ('red' if side == 'blue' else 'blue'), side
        env.apply_move(int(move))

    return env.winner(), None


def match(p_a, p_b, name_a, name_b, num_games=GAMES_PER_MATCHUP):
    """Round entre A et B. Retourne (wins_a, wins_b, faults_a, faults_b)."""
    wins_a = wins_b = 0
    faults_a = faults_b = 0
    a_disabled = b_disabled = False

    for i in range(num_games):
        a_blue = (i % 2 == 0)

        # Si un joueur est disable, l'autre gagne d'office
        if a_disabled and b_disabled:
            continue
        if a_disabled:
            wins_b += 1
            continue
        if b_disabled:
            wins_a += 1
            continue

        if a_blue:
            winner, fault = play_game(p_a, p_b)
            a_side, b_side = 'blue', 'red'
        else:
            winner, fault = play_game(p_b, p_a)
            a_side, b_side = 'red', 'blue'

        if fault == a_side:
            faults_a += 1
            if faults_a >= 5:
                a_disabled = True
        elif fault == b_side:
            faults_b += 1
            if faults_b >= 5:
                b_disabled = True

        if winner == a_side:
            wins_a += 1
        else:
            wins_b += 1

        print(f"   partie {i+1}/{num_games}  →  {name_a} {wins_a} - {wins_b} {name_b}", flush=True)

    return wins_a, wins_b, faults_a, faults_b


# ─── Elo ─────────────────────────────────────────────────────────────────────

def compute_elo(names, results, k=32.0, initial=1000.0):
    elo = {n: initial for n in names}
    for _ in range(20):
        for (a, b), (wa, wb) in results.items():
            total = wa + wb
            if total == 0:
                continue
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400))
            elo[a] += k * (wa / total - ea)
            elo[b] += k * (wb / total - (1.0 - ea))
    return elo


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log_f = open(LOG_PATH, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_f)

    print(f"=== Tournoi LLM Hex 11×11 — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"Dossier : {MODEL_LLM_DIR}")
    print(f"{GAMES_PER_MATCHUP} parties/duel, {TIME_PER_MOVE}s/coup\n")

    print("Chargement des modèles...")
    players = discover_llm_players()
    if not players:
        print("ERREUR : aucun fichier .py dans model_LLM/.")
        return

    valid = []
    failed = []
    for display, fname, inst, err in players:
        if inst is None:
            print(f"  KO   {display:<35} ({fname})  →  {err}")
            failed.append(display)
        else:
            print(f"  OK   {display:<35} ({fname})")
            valid.append((display, inst))
    print()

    if len(valid) < 2:
        print("ERREUR : moins de 2 joueurs valides.")
        for f in failed:
            print(f"  {f}: 0 victoire (modèle non chargé)")
        return

    names = [d for d, _ in valid]
    by_name = {d: inst for d, inst in valid}

    matchups = list(combinations(names, 2))
    total = len(matchups)
    results = {}
    wins_total = {n: 0 for n in names}
    games_total = {n: 0 for n in names}
    faults_total = {n: 0 for n in names}

    t0 = time.time()
    for i, (a, b) in enumerate(matchups, 1):
        print(f"[{i}/{total}] {a}  vs  {b} ...", flush=True)
        t_m = time.time()
        try:
            wa, wb, fa, fb = match(by_name[a], by_name[b], a, b)
        except Exception as e:
            print(f"   ERREUR fatale : {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.__stderr__)
            wa = wb = 0
            fa = fb = 0
        dt = time.time() - t_m
        results[(a, b)] = (wa, wb)
        wins_total[a]  += wa
        wins_total[b]  += wb
        games_total[a] += wa + wb
        games_total[b] += wa + wb
        faults_total[a] += fa
        faults_total[b] += fb
        flag = ""
        if fa or fb:
            flag = f"   [faults A={fa} B={fb}]"
        print(f"   → {a} {wa} - {wb} {b}   ({dt:.1f}s){flag}")

    total_time = time.time() - t0

    # Pour les modèles qui n'ont pas chargé : 0 victoire, présents dans le log final
    for f in failed:
        wins_total[f] = 0
        games_total[f] = 0
        faults_total[f] = 0
        names_with_failed = True

    # Elo (uniquement sur les valides)
    elo = compute_elo(names, results)
    for f in failed:
        elo[f] = 0.0

    all_names = names + failed
    ranking = sorted(all_names, key=lambda n: (elo.get(n, 0.0), wins_total.get(n, 0)), reverse=True)

    print()
    print("=" * 78)
    print(f"{'Rang':<5} {'Modèle':<35} {'Elo':>6} {'V':>5} {'/':<1} {'Parties':<7} {'Win%':>6} {'Faults':>7}")
    print("-" * 78)
    for rank, n in enumerate(ranking, 1):
        w = wins_total.get(n, 0)
        g = games_total.get(n, 0)
        pct = (100.0 * w / g) if g else 0.0
        print(f"#{rank:<4} {n:<35} {elo.get(n,0.0):>6.0f} {w:>5} / {g:<7} {pct:>5.1f}% {faults_total.get(n,0):>7}")
    print("=" * 78)
    print(f"\nTemps total : {total_time:.1f}s")
    if failed:
        print("\nModèles non chargés (0 victoire) :")
        for f in failed:
            print(f"  - {f}")

    sys.stdout = sys.__stdout__
    log_f.close()
    print(f"\nLog écrit : {LOG_PATH}")


if __name__ == '__main__':
    main()

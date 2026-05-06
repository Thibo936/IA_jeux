#!/usr/bin/env python3
"""
benchmark_ranking_params.py - Micro-benchmark pour dimensionner ranking.py.

Le script lance de petits tournois representatifs avec ranking.py, mesure le
temps reel, puis estime le cout d'un tournoi complet avec les joueurs presents.

Exemples :
  python benchmark_ranking_params.py
  python benchmark_ranking_params.py --include-slow
  python benchmark_ranking_params.py --sample-games 2 --timeout 900
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RANKING_PY = os.path.join(SCRIPT_DIR, 'ranking.py')
MODEL_DIR = 'model'
BEST_MODEL_FILE = os.path.join('checkpoints', 'best_model.pt')
CLASSIC_DENYLIST = {'__init__', 'humain', 'katahex', 'mcts_az'}

CLASSIC_BENCH_IDS = ['alphabeta', 'heuristic', 'mcts_light', 'mohex']


@dataclass
class BenchCase:
    name: str
    args: list[str]
    family: str
    sims: int | None = None
    workers: int | None = None
    gpu_workers: int | None = None


@dataclass
class BenchResult:
    case: BenchCase
    ok: bool
    seconds: float
    games: int
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def games_per_second(self) -> float:
        return self.games / self.seconds if self.ok and self.seconds > 0 else 0.0

    @property
    def seconds_per_game(self) -> float:
        return self.seconds / self.games if self.ok and self.games > 0 else 0.0


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _discover_classics() -> list[str]:
    ia_dir = os.path.join(SCRIPT_DIR, 'ia')
    if not os.path.isdir(ia_dir):
        return []
    discovered: list[str] = []
    for fname in sorted(os.listdir(ia_dir)):
        if not fname.endswith('.py'):
            continue
        stem = os.path.splitext(fname)[0]
        if stem in CLASSIC_DENYLIST:
            continue
        discovered.append(stem)
    return discovered


def _az_name_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    if base == 'best_model':
        return 'AZ-best'
    if base.startswith('model_'):
        parts = base.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            return f"AZ-{int(parts[1]):02d}"
    return 'AZ-' + base


def _discover_az_models() -> list[tuple[str, str]]:
    model_dir = os.path.join(SCRIPT_DIR, MODEL_DIR)
    models: list[tuple[str, str]] = []
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if fname.endswith('.pt'):
                path = os.path.abspath(os.path.join(model_dir, fname))
                models.append((_az_name_from_path(path), path))

    best_ckpt = os.path.abspath(os.path.join(SCRIPT_DIR, BEST_MODEL_FILE))
    if os.path.isfile(best_ckpt):
        models.append((_az_name_from_path(best_ckpt), best_ckpt))

    # Dedup par chemin absolu, comme ranking.py le fait dans son registre.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for name, path in models:
        ap = os.path.abspath(path)
        if ap in seen:
            continue
        seen.add(ap)
        deduped.append((name, ap))
    return deduped


def _torch_info() -> tuple[bool, bool, str | None]:
    try:
        import torch  # type: ignore
    except Exception:
        return False, False, None
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return True, cuda_available, gpu_name


def _total_matchups(classic_count: int, az_count: int) -> tuple[int, int, int, int]:
    total_players = classic_count + az_count
    total = total_players * (total_players - 1) // 2
    cpu = classic_count * (classic_count - 1) // 2
    mixed = classic_count * az_count
    gpu = az_count * (az_count - 1) // 2
    return total, cpu, mixed, gpu


def _extract_games(stdout: str) -> int:
    match = re.search(r"Parties . jouer au total\s*:\s*(\d+)", stdout)
    if match:
        return int(match.group(1))
    return 0


def _run_case(case: BenchCase, timeout: int) -> BenchResult:
    with tempfile.TemporaryDirectory(prefix='ranking_bench_') as tmpdir:
        cmd = [
            sys.executable,
            RANKING_PY,
            '--output-dir', tmpdir,
            '--csv', os.path.join(tmpdir, 'ranking.csv'),
            '--no-html',
            '--no-csv',
            *case.args,
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=SCRIPT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            seconds = time.time() - t0
            stdout = exc.stdout if isinstance(exc.stdout, str) else ''
            stderr = exc.stderr if isinstance(exc.stderr, str) else ''
            return BenchResult(case, False, seconds, _extract_games(stdout), stdout, stderr, 'timeout')

    seconds = time.time() - t0
    games = _extract_games(proc.stdout)
    ok = proc.returncode == 0 and games > 0
    error = None if ok else f"exit={proc.returncode}"
    return BenchResult(case, ok, seconds, games, proc.stdout, proc.stderr, error)


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return '0s'
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    if days >= 1.0:
        return f"{days:.1f}j"
    if hours >= 1.0:
        return f"{hours:.1f}h"
    if minutes >= 1.0:
        return f"{minutes:.1f}min"
    return f"{seconds:.1f}s"


def _estimate_full_time(results: list[BenchResult], classic_count: int, az_count: int,
                        games_per_matchup: int, sims: int) -> tuple[float | None, str]:
    _, cpu_matchups, mixed_matchups, gpu_matchups = _total_matchups(classic_count, az_count)

    classic = [r for r in results if r.ok and r.case.family == 'classic']
    az_same_sims = [r for r in results if r.ok and r.case.family == 'az' and r.case.sims == sims]
    az_any = [r for r in results if r.ok and r.case.family == 'az']
    mixed = [r for r in results if r.ok and r.case.family == 'mixed']

    if not classic and not az_any and not mixed:
        return None, 'pas assez de mesures valides'

    cpu_spg = min((r.seconds_per_game for r in classic), default=0.0)

    if az_same_sims:
        az_spg = min(r.seconds_per_game for r in az_same_sims)
        scale_note = ''
    elif az_any:
        base = min(az_any, key=lambda r: r.seconds_per_game)
        base_sims = base.case.sims or sims
        az_spg = base.seconds_per_game * (sims / base_sims)
        scale_note = f"AZ extrapole depuis sims={base_sims}"
    else:
        az_spg = 0.0
        scale_note = 'pas de mesure AZ'

    mixed_spg = min((r.seconds_per_game for r in mixed), default=az_spg or cpu_spg)

    cpu_seconds = cpu_matchups * games_per_matchup * cpu_spg
    az_seconds = (gpu_matchups * games_per_matchup * az_spg) + (mixed_matchups * games_per_matchup * mixed_spg)

    # ranking.py lance CPU et AZ en parallele quand les deux existent, donc on
    # prend max(cpu, az) puis on ajoute une marge pour chargement/overheads.
    estimate = max(cpu_seconds, az_seconds) * 1.15
    notes = []
    if scale_note:
        notes.append(scale_note)
    notes.append('marge overhead 15%')
    return estimate, ', '.join(notes)


def _build_cases(args, az_models: list[tuple[str, str]]) -> list[BenchCase]:
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count - 2)
    worker_values = args.workers or sorted({1, min(4, default_workers), default_workers})
    _, cuda_available, _ = _torch_info()
    gpu_worker_values = args.gpu_workers or ([1, 2] if cuda_available else [1])
    sims_values = args.sims or ([50, 100, 200, 400] if args.include_slow else [50, 100])

    classic_ids = [c for c in CLASSIC_BENCH_IDS]
    cases: list[BenchCase] = []

    if len(classic_ids) >= 2:
        for workers in worker_values:
            cases.append(BenchCase(
                name=f"classic workers={workers}",
                family='classic',
                workers=workers,
                args=[
                    '--no-models',
                    '--ias', *classic_ids,
                    '--mode', 'classic',
                    '--games', str(args.sample_games),
                    '--workers', str(workers),
                    '--game-threads', '1',
                ],
            ))

    az_subset = az_models[:max(2, min(3, len(az_models)))]
    az_paths = [path for _, path in az_subset]
    if len(az_paths) >= 2:
        for sims in sims_values:
            for gpu_workers in gpu_worker_values:
                cases.append(BenchCase(
                    name=f"az sims={sims} gpu-workers={gpu_workers}",
                    family='az',
                    sims=sims,
                    gpu_workers=gpu_workers,
                    args=[
                        '--no-classics',
                        '--no-models',
                        '--add', *az_paths,
                        '--mode', 'alphazero',
                        '--games', str(args.sample_games),
                        '--sims', str(sims),
                        '--gpu-workers', str(gpu_workers),
                        '--game-threads', '1',
                    ],
                ))

    if len(az_paths) >= 1:
        for sims in sims_values[:2]:
            cases.append(BenchCase(
                name=f"mixed sims={sims}",
                family='mixed',
                sims=sims,
                args=[
                    '--no-models',
                    '--ias', 'alphabeta', 'heuristic',
                    '--add', az_paths[0],
                    '--games', str(args.sample_games),
                    '--sims', str(sims),
                    '--workers', str(default_workers),
                    '--gpu-workers', '1',
                    '--game-threads', '1',
                ],
            ))

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark court pour trouver de bons parametres ranking.py",
    )
    parser.add_argument('--sample-games', type=int, default=1,
                        help="Parties par matchup pendant le benchmark (defaut: 1)")
    parser.add_argument('--timeout', type=int, default=600,
                        help="Timeout par test en secondes (defaut: 600)")
    parser.add_argument('--include-slow', action='store_true',
                        help="Teste aussi sims=200 et sims=400")
    parser.add_argument('--sims', type=_parse_int_list, default=None,
                        help="Liste sims separee par virgules, ex: 50,100,200")
    parser.add_argument('--workers', type=_parse_int_list, default=None,
                        help="Liste workers CPU separee par virgules")
    parser.add_argument('--gpu-workers', type=_parse_int_list, default=None,
                        help="Liste gpu-workers separee par virgules")
    parser.add_argument('--estimate-games', type=int, default=50,
                        help="Games/matchup pour l'estimation complete (defaut: 50)")
    parser.add_argument('--estimate-sims', type=int, default=400,
                        help="Sims pour l'estimation complete (defaut: 400)")
    args = parser.parse_args()

    classics = _discover_classics()
    az_models = _discover_az_models()
    total, cpu_matchups, mixed_matchups, gpu_matchups = _total_matchups(len(classics), len(az_models))
    torch_available, cuda_available, gpu_name = _torch_info()

    print("=== Benchmark ranking.py ===")
    print(f"CPU cores       : {os.cpu_count() or 1}")
    print(f"Torch importable: {'oui' if torch_available else 'non'}")
    print(f"CUDA disponible : {'oui' if cuda_available else 'non'}")
    if gpu_name:
        print(f"GPU             : {gpu_name}")
    if not torch_available:
        print("WARN            : cet interpreteur ne peut pas importer torch ; ranking.py echouera probablement.")
    print(f"Classiques      : {len(classics)}")
    print(f"Modeles AZ      : {len(az_models)}")
    print(f"Joueurs total   : {len(classics) + len(az_models)}")
    print(f"Matchups total  : {total} ({cpu_matchups} CPU, {mixed_matchups} mixtes, {gpu_matchups} AZ)")
    print(f"Parties estimees: {total * args.estimate_games} avec --games {args.estimate_games}\n")

    cases = _build_cases(args, az_models)
    if not cases:
        print("ERREUR : pas assez de joueurs/modeles pour benchmarker.", file=sys.stderr)
        return 1

    results: list[BenchResult] = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case.name} ...", flush=True)
        result = _run_case(case, args.timeout)
        results.append(result)
        if result.ok:
            print(f"  OK  {result.games} parties en {_format_duration(result.seconds)} "
                  f"({result.seconds_per_game:.2f}s/partie, {result.games_per_second:.3f} parties/s)")
        else:
            print(f"  KO  {result.error} apres {_format_duration(result.seconds)}")
            if result.stderr.strip():
                last_lines = result.stderr.strip().splitlines()[-3:]
                print("  stderr:")
                for line in last_lines:
                    print(f"    {line}")

    print("\n=== Resultats utiles ===")
    valid = [r for r in results if r.ok]
    if not valid:
        print("Aucun benchmark valide.")
        return 1

    for family in ['classic', 'az', 'mixed']:
        family_results = [r for r in valid if r.case.family == family]
        if not family_results:
            continue
        best = min(family_results, key=lambda r: r.seconds_per_game)
        print(f"Meilleur {family:<7}: {best.case.name:<32} {best.seconds_per_game:.2f}s/partie")

    print("\n=== Estimations tournoi complet ===")
    for games in [10, 30, args.estimate_games]:
        estimate, note = _estimate_full_time(
            valid,
            len(classics),
            len(az_models),
            games_per_matchup=games,
            sims=args.estimate_sims,
        )
        if estimate is None:
            print(f"--games {games:<3} --sims {args.estimate_sims:<4}: estimation impossible ({note})")
        else:
            print(f"--games {games:<3} --sims {args.estimate_sims:<4}: environ {_format_duration(estimate)} ({note})")

    print("\n=== Recommandation ===")
    best_classic = min((r for r in valid if r.case.family == 'classic'),
                       key=lambda r: r.seconds_per_game, default=None)
    best_az = min((r for r in valid if r.case.family == 'az'),
                  key=lambda r: r.seconds_per_game, default=None)
    workers = best_classic.case.workers if best_classic and best_classic.case.workers else max(1, (os.cpu_count() or 2) - 2)
    gpu_workers = best_az.case.gpu_workers if best_az and best_az.case.gpu_workers else 1
    sims = best_az.case.sims if best_az and best_az.case.sims else min(args.sims or [100])
    print("Classement rapide :")
    print(f"  python ranking.py --games 10 --sims {sims} --workers {workers} --gpu-workers {gpu_workers}")
    print("Classement plus serieux :")
    print(f"  python ranking.py --games 30 --sims {max(sims, 200)} --workers {workers} --gpu-workers {gpu_workers}")
    print("Classement complet lourd :")
    print(f"  python ranking.py --games 50 --sims {args.estimate_sims} --workers {workers} --gpu-workers {gpu_workers}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

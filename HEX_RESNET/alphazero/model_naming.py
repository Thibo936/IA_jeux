#!/usr/bin/env python3
"""
model_naming.py — Gestion du nommage des modèles AlphaZero dans model/.

Format : model_<num>_<parent>_<day>_<month>.pt
  num    : numéro séquentiel du modèle (01, 02, …)
  parent : numéro du dernier modèle présent au moment de la création
  day    : jour de création (01–31)
  month  : mois de création (01–12)
"""

import os
import re
import shutil
from datetime import datetime


# Regex pour parser les noms de modèles
_MODEL_RE = re.compile(r'^model_(\d+)_(\d+)_(\d+)_(\d+)\.pt$')


def parse_model_name(filename: str) -> tuple[int, int, int, int] | None:
    """Parse 'model_NN_PP_DD_MM.pt' → (num, parent, day, month). Retourne None si non-match."""
    m = _MODEL_RE.match(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def scan_models(model_dir: str) -> list[tuple[int, int, int, int, str]]:
    """
    Scanne model_dir et retourne une liste triée par num :
      [(num, parent, day, month, abs_path), ...]
    """
    results: list[tuple[int, int, int, int, str]] = []
    if not os.path.isdir(model_dir):
        return results
    for fname in os.listdir(model_dir):
        parsed = parse_model_name(fname)
        if parsed is None:
            continue
        num, parent, day, month = parsed
        results.append((num, parent, day, month, os.path.abspath(os.path.join(model_dir, fname))))
    results.sort(key=lambda x: x[0])
    return results


def last_model_number(model_dir: str) -> int | None:
    """Retourne le plus grand numéro de modèle dans model_dir, ou None si vide."""
    models = scan_models(model_dir)
    if not models:
        return None
    return models[-1][0]


def next_model_number(model_dir: str) -> int:
    """Prochain numéro disponible (1 si dossier vide)."""
    last = last_model_number(model_dir)
    return 1 if last is None else last + 1


def build_model_name(num: int, parent: int, date: datetime | None = None) -> str:
    """Construit le nom de fichier model_NN_PP_DD_MM.pt."""
    if date is None:
        date = datetime.now()
    return f"model_{num:02d}_{parent:02d}_{date.day:02d}_{date.month:02d}.pt"


def copy_best_to_model(best_model_path: str, model_dir: str) -> str:
    """
    Copie best_model.pt dans model_dir avec le bon nommage incrémental.
    Retourne le chemin absolu du nouveau fichier.
    """
    os.makedirs(model_dir, exist_ok=True)
    num = next_model_number(model_dir)
    parent = last_model_number(model_dir)
    if parent is None:
        parent = 0
    fname = build_model_name(num, parent)
    dest = os.path.join(model_dir, fname)
    shutil.copy2(best_model_path, dest)
    print(f"  → Copié dans model/ : {fname}")
    return dest


def list_model_entries(model_dir: str) -> list[dict]:
    """
    Liste les modèles sous forme de dicts lisibles :
      [{'num': 1, 'parent': 0, 'date_str': '17/04', 'path': ...}, ...]
    """
    models = scan_models(model_dir)
    return [
        {
            'num': num,
            'parent': parent,
            'date_str': f"{day:02d}/{month:02d}",
            'path': path,
        }
        for num, parent, day, month, path in models
    ]


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else 'model'
    for e in list_model_entries(d):
        print(f"  #{e['num']:02d}  parent={e['parent']:02d}  date={e['date_str']}  {e['path']}")

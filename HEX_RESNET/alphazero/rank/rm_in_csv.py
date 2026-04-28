#!/usr/bin/env python3
"""
rm_in_csv.py — Supprime du CSV ranking toutes les parties impliquant un joueur donné.

Usage :
    python rm_in_csv.py <joueur> [chemin_csv]

Exemple :
    python rm_in_csv.py Random
    python rm_in_csv.py AZ-best ../checkpoints/ranking.csv
"""
import csv
import os
import sys


def remove_player(player_name: str, csv_path: str) -> int:
    """Filtre les lignes où name_a ou name_b == player_name. Retourne nb supprimé."""
    removed = 0
    kept = []

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        kept.append(header)
        name_a_idx = header.index('name_a')
        name_b_idx = header.index('name_b')

        for row in reader:
            if row[name_a_idx] == player_name or row[name_b_idx] == player_name:
                removed += 1
            else:
                kept.append(row)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(kept)

    return removed


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    player = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ranking.csv'
    )

    if not os.path.isfile(csv_path):
        print(f"ERREUR : fichier CSV introuvable : {csv_path}", file=sys.stderr)
        sys.exit(1)

    removed = remove_player(player, csv_path)
    print(f"Supprimé {removed} ligne(s) impliquant '{player}'.")
    print(f"CSV mis à jour : {csv_path}")


if __name__ == '__main__':
    main()

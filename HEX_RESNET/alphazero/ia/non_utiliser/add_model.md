# Guide : Ajouter une nouvelle IA dans `alphazero/ia/`

Ce document détaille les conventions, l'architecture et les étapes d'intégration pour ajouter un nouveau joueur IA au projet AlphaZero Hex.

---

## 1. Interface commune obligatoire

Toute IA Python doit exposer **exactement** cette interface pour être compatible avec le tournoi et le classement :

```python
class MaNouvelleIA:
    def __init__(self, ...):
        self.last_stats: dict = {}   # ← obligatoire

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        """
        Retourne l'index du coup choisi (0 .. 120 pour un plateau 11×11).
        - env       : instance de HexEnv (moteur de jeu unique)
        - time_s    : budget temps en secondes (information, pas de contrainte stricte)
        """
        ...
```

### Règles
- `select_move` ne doit **jamais** modifier `env` en place (utiliser `env.copy()` si besoin).
- `self.last_stats` doit être mis à jour **à chaque appel** avec au moins les clés attendues par le format de stats (voir §4).
- L'IA peut ignorer `time_s` ou s'en servir comme deadline (`time.time() + time_s`).

---

## 2. Structure minimale d'un fichier

Créer un nouveau fichier dans `alphazero/ia/<nom_snake_case>.py` avec cette structure :

```python
# mon_ia.py — Description courte de l'IA
# Interface CLI : python mon_ia.py BOARD PLAYER [time_s]

import sys
import os

# ─── Bootstrap des imports train/ ─────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv
from config import NUM_CELLS, BOARD_SIZE  # si besoin

# ─── Logique de l'IA ──────────────────────────────────────────────────────────

class MonIA:
    def __init__(self, param: int = 42):
        self.param = param
        self.last_stats: dict = {}

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        # Exemple : jouer le premier coup légal
        move = int(moves[0])
        self.last_stats = {}  # ou stats pertinentes
        return move

# ─── Interface CLI (protocole BOARD/PLAYER) ───────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python mon_ia.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = MonIA()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))
```

### Points importants
- **Ne jamais** utiliser `cd` dans le code. Utiliser le bloc `sys.path` ci-dessus pour importer `hex_env` et `config`.
- Le fichier doit être exécutable en ligne de commande pour servir aussi d'IA externe via `subprocess` (voir `tournament.py`).

---

## 3. Protocole BOARD / PLAYER

Le tournoi et les CLI communiquent via ce protocole standard :

| Argument | Format | Description |
|----------|--------|-------------|
| `BOARD`  | Chaîne de 121 caractères | `.` = vide, `O` = Blue (Nord-Sud), `@` = Red (Ouest-Est) |
| `PLAYER` | `'O'` ou `'@'` | Joueur dont c'est le tour |
| `time_s` | `float` (optionnel) | Budget temps en secondes |

**Sortie stdout** : le coup en notation alphanumérique, ex. `A1`, `F6`, `K11`.  
**Sortie stderr** : stats de recherche (formatés parsés par `tournament.py`, voir §4).

---

## 4. Conventions de stats (`last_stats` & stderr)

Le tournoi formate automatiquement les stats affichées en verbose selon les clés présentes dans `last_stats`.  
Deux familles de formats sont reconnues :

### Alpha-Beta / Minimax
```python
self.last_stats = {
    'score': 1234,    # int, score interne
    'nodes': 150_000, # int, nœuds explorés
    'depth': 5,       # int, profondeur atteinte
}
# stderr attendu :
# SCORE:1234 NODES:150000 DEPTH:5
```

### MCTS / Monte-Carlo
```python
self.last_stats = {
    'iters':   800,      # int, nombre de simulations/itérations
    'visits':  12000,    # int, visites totales dans l'arbre
    'winrate': 0.62,     # float, winrate estimé du coup joué
    'time':    1.234,    # float, temps réel écoulé
}
# stderr attendu :
# ITERS:800 VISITS:12000 WINRATE:0.6200 TIME:1.234
```

**Bonne pratique** : émettre ces lignes sur `stderr` dans la CLI `__main__` pour que les IA externes soient aussi parsées correctement :

```python
print(f"ITERS:{sims} VISITS:{visits} WINRATE:{winrate:.4f} TIME:{elapsed:.3f}", file=sys.stderr)
```

---

## 5. Coup gagnant immédiat

Avant de lancer une recherche coûteuse, **toujours** vérifier si un coup gagnant immédiat existe :

```python
moves = env.get_legal_moves()
root_blue = env.blue_to_play

for move in moves:
    m = int(move)
    env.apply_move(m)
    w = env.winner()
    env.undo_move(m, root_blue)   # ou env.copy().apply_move(m) si pas d'undo
    if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
        self.last_stats = { ... }  # stats minimales
        return m
```

Cette optimisation est présente dans **toutes** les IA du projet (`alphabeta`, `mcts_light`, `mc_pure`, etc.).

---

## 6. Intégration dans `tournament.py`

Pour rendre l'IA accessible via la CLI `tournament.py`, l'ajouter dans la fonction `_resolve_ai` :

```python
# alphazero/tournament.py

def _resolve_ai(name: str, time_s: float):
    n = name.lower()
    if n == 'alphabeta':
        from alphabeta import AlphaBetaPlayer
        return AlphaBetaPlayer(), 'AlphaBeta'
    # ...
    if n == 'mon_ia':                     # ← AJOUTER
        from mon_ia import MonIA           # ← AJOUTER
        return MonIA(), 'MonIA'            # ← AJOUTER
    # ...
    return name, name
```

- Le premier élément du tuple est l'instance (ou une chaîne de commande externe).
- Le deuxième élément est le nom d'affichage.

### Commande externe
Si l'IA est un exécutable externe (binaire, script shell, etc.), il suffit de passer son chemin directement :

```bash
python tournament.py alphabeta ./mon_executable 20 -t 1.5
```

L'exécutable doit respecter le protocole BOARD/PLAYER (§3).

---

## 7. Intégration dans `ranking.py` (round-robin unifié)

Pour inclure la nouvelle IA dans le classement automatique, modifier **5 endroits** :

### 7.1 Liste des classiques par défaut
```python
# alphazero/ranking.py
DEFAULT_CLASSICS = ['random', 'alphabeta', 'mc_pure', 'mcts_light',
                    'heuristic', 'mohex', 'mon_ia']   # ← AJOUTER
```

### 7.2 Nom d'affichage
```python
CLASSIC_DISPLAY = {
    # ...
    'mon_ia': 'MonIA',   # ← AJOUTER
}
```

### 7.3 Type / famille
```python
CLASSIC_TYPE = {
    # ...
    'mon_ia': 'MonType',  # ← AJOUTER (ex: 'Neural', 'Heuristique', 'MCTS'...)
}
```

### 7.4 Couleur dans les graphes HTML
```python
CLASSIC_COLORS = {
    # ...
    'MonType': '#ff00ff',   # ← AJOUTER (doit correspondre à CLASSIC_TYPE)
}
```

### 7.5 Instanciation
```python
def _make_classic(classic_id: str):
    n = classic_id.lower()
    # ...
    if n == 'mon_ia':                  # ← AJOUTER
        from mon_ia import MonIA       # ← AJOUTER
        return MonIA()                  # ← AJOUTER
    raise ValueError(f"IA classique inconnue : {classic_id}")
```

---

## 8. API utile de `HexEnv`

| Méthode | Description |
|---------|-------------|
| `env.get_legal_moves()` | `np.ndarray` des indices 0..120 libres |
| `env.apply_move(pos)` | Joue `pos` sur le plateau **en place** |
| `env.undo_move(pos, was_blue)` | Annule `apply_move` (plus rapide que `copy`) |
| `env.copy()` | Copie profonde (utile pour les simulations) |
| `env.is_terminal()` | `True` si partie finie |
| `env.winner()` | `'blue'`, `'red'` ou `None` |
| `env.blue_to_play` | `bool`, `True` si Blue (O) doit jouer |
| `env.blue`, `env.red` | Tableaux `bool (11, 11)` des pierres posées |
| `env.to_string()` | Sérialise en chaîne de 121 caractères |
| `HexEnv.from_string(board, player)` | Constructeur depuis chaîne CLI |
| `env.pos_to_str(pos)` | `0 → "A1"`, `120 → "K11"` |
| `HexEnv.str_to_pos(s)` | `"F6" → 59` |
| `env.get_state_tensor()` | `np.ndarray (3, 11, 11)` pour le réseau |
| `env.mirror()` | Réflexion diagonale (symétrie Hex) |

---

## 9. Checklist avant de commit

- [ ] Le fichier est dans `alphazero/ia/<nom>.py`
- [ ] `select_move(env, time_s) -> int` est implémenté
- [ ] `self.last_stats` est renseigné après chaque coup
- [ ] Le coup gagnant immédiat est détecté
- [ ] La CLI `__main__` respecte le protocole BOARD/PLAYER
- [ ] Les stats sont écrites sur `stderr` dans le format attendu
- [ ] Ajouté dans `_resolve_ai` de `tournament.py`
- [ ] Ajouté dans les 5 sections de `ranking.py`
- [ ] Testé via : `python alphazero/tournament.py mon_ia random 10 -t 1.0`

---

## 10. Exemple complet : squelette prêt à l'emploi

```python
# alphazero/ia/template_ia.py
# Copier ce fichier et renommer/adapter.

import sys
import os
import time

_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), 'train')
for _p in [_dir, _train]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv


class TemplateIA:
    """
    Squelette minimal d'IA pour Hex 11×11.
    Remplacez la logique de select_move par votre algorithme.
    """

    def __init__(self, seed: int | None = None):
        self.last_stats: dict = {}
        if seed is not None:
            import numpy as np
            np.random.seed(seed)

    def select_move(self, env: HexEnv, time_s: float = 1.5) -> int:
        moves = env.get_legal_moves()
        if len(moves) == 0:
            return -1

        root_blue = env.blue_to_play

        # 1) Coup gagnant immédiat
        for move in moves:
            m = int(move)
            env.apply_move(m)
            w = env.winner()
            env.undo_move(m, root_blue)
            if (root_blue and w == 'blue') or (not root_blue and w == 'red'):
                self.last_stats = {'iters': 1, 'visits': 1,
                                   'winrate': 1.0, 'time': 0.0}
                return m

        # 2) Votre algorithme ici
        t0 = time.time()
        best = int(moves[0])   # ← remplacer par vraie sélection
        elapsed = time.time() - t0

        self.last_stats = {
            'iters': 1,
            'visits': 1,
            'winrate': 0.5,
            'time': elapsed,
        }
        return best


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python template_ia.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    _player = TemplateIA()
    _move = _player.select_move(_env, _time_s)
    print(_env.pos_to_str(_move))
```

---

*Dernière mise à jour : avril 2026*

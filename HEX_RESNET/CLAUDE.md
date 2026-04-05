# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Aucune compilation. Dépendances Python uniquement :

```bash
pip install torch numpy numba
```

## Running

### Tournoi Python

```bash
cd alphazero

# AlphaBeta vs Random (20 parties)
python tournament.py alphabeta random 20

# AlphaBeta vs AlphaZero (verbose, 2s/coup)
python tournament.py alphabeta alphazero 10 -v -t 2.0
```

Mots-clés reconnus : `alphabeta`, `random`, `alphazero`. Tout autre argument est traité comme une commande externe via subprocess.

### CLI individuelle (protocole BOARD/PLAYER commun)

```bash
python alphabeta.py "........................................................................................................................................................................................................." O
python random_player.py "........................................................................................................................................................................................................." @
python play.py "........................................................................................................................................................................................................." O 1.5
```

- **BOARD** : 121 chars row-major : `.` vide, `O` Blue, `@` Red
- **PLAYER** : `O` (Blue, Nord-Sud) ou `@` (Red, Ouest-Est)
- Sortie stdout : coup en notation `A1`..`K11` — stderr : stats

### Entraînement AlphaZero

**Serveur local (Ryzen 7 / RX 6600XT / ROCm) :**
```bash
cd alphazero
python trainer.py --iterations 100 --games 100 --simulations 800 --steps 500 --device cuda
python trainer.py --iterations 1 --games 10 --simulations 100   # test rapide
```

**Google Colab (GPU puissant) :**
Utiliser `train_colab.ipynb` à la racine du projet. Supporte :
- Mixed precision (FP16) pour training et inférence
- Self-play parallèle (multi-processus)
- Upload/download des checkpoints
- Monitoring avec graphes matplotlib

Checkpoints dans `alphazero/checkpoints/`. Meilleur modèle : `best_model.pt`. Accepté si win rate ≥ 55 % sur 40 parties d'évaluation.

**Important** : si l'architecture réseau change (ex: nombre de blocs), supprimer les anciens checkpoints avant de relancer :
```bash
rm checkpoints/best_model.pt checkpoints/model_iter_*.pt checkpoints/replay_buffer.npz
```

## Architecture

### Moteur de jeu — `hex_env.py` (source de vérité unique)

État : deux tableaux numpy `bool` 11×11 (`blue`, `red`).
- `get_state_tensor()` → `float32 (3, 11, 11)` : plans Blue, Red, joueur courant
- `is_terminal()` / `winner()` → BFS flood-fill
- `mirror()` → augmentation par réflexion diagonale (symétrie naturelle du Hex)
- `copy()`, `apply_move(pos)`, `undo_move(pos, was_blue)`, `to_string()`, `pos_to_str()`, `str_to_pos()`

### IA Python

- **`alphabeta.py`** : Alpha-Beta avec heuristique BFS 0-1 Numba JIT (Red_path − Blue_path). Profondeur de base 5, adaptative (+1 si ≤60 coups, +1 si ≤30). Table de transposition Zobrist, killer moves, history heuristic, ordonnancement à tous les niveaux. `undo_move` au lieu de `copy`. Exporte `AlphaBetaPlayer` et `eval_heuristic()`.
- **`random_player.py`** : Joueur aléatoire uniforme. Exporte `RandomPlayer`.
- **`play.py`** : Wrapper CLI pour AlphaZero (charge `best_model.pt`, adapte les simulations au temps alloué).

### AlphaZero — `alphazero/`

- **`network.py`** : ResNet 10 blocs, 128 filtres, têtes politique + valeur (~3 M params). Supporte `predict()` (single) et `batch_predict()` (batché). Inférence FP16 automatique sur GPU (torch.amp.autocast).
- **`mcts_az.py`** : MCTS UCB-PUCT (c_puct=1.0), bruit Dirichlet à la racine (α=0.03, ε=0.25), température τ=1 pour les 20 premiers coups puis τ→0. Inférence batchée GPU avec virtual loss (16 feuilles/batch). Expansion paresseuse. Tree reuse entre coups.
- **`self_play.py`** : Génération de parties avec tree reuse + buffer circulaire (200 000 positions).
- **`trainer.py`** : Boucle self-play → train → évaluation. Loss = MSE(v,z) + CrossEntropy(p,π) + λ·L2. Adam lr=1e-3 + cosine annealing (→ 1e-5). Gestion des checkpoints incompatibles.
- **`evaluate.py`** : Comparaison de modèles et test vs joueur aléatoire.
- **`tournament.py`** : Tournoi full Python. Supporte `AlphaBetaPlayer`, `RandomPlayer`, `AlphaZeroPlayer`, et commandes externes via subprocess.

### Interface commune entre les IA Python

Chaque joueur expose : `select_move(env: HexEnv, time_s: float) -> int` et stocke les stats dans `self.last_stats` après chaque coup.

## Language

Commentaires, noms de variables, et messages en français.

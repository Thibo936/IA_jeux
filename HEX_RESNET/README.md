# TC_MG_hex — IA pour le jeu de Hex 11×11

Projet L3 (Paris 8). Trois IA jouent au Hex 11×11 en **full Python** :
- **Alpha-Beta** — recherche alpha-bêta avec heuristique BFS 0-1
- **MCTS guidé** — Monte-Carlo Tree Search avec réseau de neurones (AlphaZero)
- **Aléatoire** — joueur de référence

---

## Installation

```bash
pip install torch numpy
```

GPU NVIDIA (recommandé pour l'entraînement) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

GPU AMD (ROCm) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

> **Note RX 6600 / gfx1032** : ajouter `export HSA_OVERRIDE_GFX_VERSION=10.3.0` dans `~/.bashrc`.

Dépendances optionnelles (notebook Colab) : `matplotlib`

---

## Démarrage rapide

```bash
cd alphazero

# Tournoi Alpha-Beta vs joueur aléatoire (20 parties)
python tournament.py alphabeta random 20

# Tournoi avec affichage du plateau
python tournament.py alphabeta random 4 -v

# Entraîner AlphaZero (test rapide)
python trainer.py --iterations 1 --games 10 --simulations 100

# Tournoi Alpha-Beta vs AlphaZero entraîné
python tournament.py alphabeta alphazero 10 -t 2.0
```

---

## Protocole commun des IA

Toutes les IA partagent le même protocole en ligne de commande :

```bash
python <ia>.py BOARD PLAYER [time_s]
```

| Argument | Description |
|---|---|
| `BOARD` | 121 caractères row-major : `.` vide, `O` Blue, `@` Red |
| `PLAYER` | `O` (Blue, Nord→Sud) ou `@` (Red, Ouest→Est) |
| `time_s` | Temps alloué en secondes (optionnel) |

**Sortie stdout** : coup en notation `A1`..`K11`
**Sortie stderr** : stats de recherche

```bash
# Exemples
python alphabeta.py "........................................................................................................................................................................................................." O
python random_player.py "........................................................................................................................................................................................................." @
python play.py "........................................................................................................................................................................................................." O 1.5
```

---

## Alpha-Beta

Recherche alpha-bêta avec profondeur **adaptative** selon le nombre de coups légaux :

| Coups légaux | Profondeur | Temps typique |
|---|---|---|
| > 60 (début) | 1 | ~0.09 s |
| 30–60 (milieu) | 2 | ~0.4 s |
| < 30 (fin) | 3 | ~0.5 s |

Heuristique : `score = Red_path − Blue_path` (BFS 0-1, plus court chemin virtuel).
Les coups sont ordonnés à la racine par cette heuristique pour améliorer les coupures.

---

## AlphaZero

Pipeline complet dans `alphazero/`. Lancer les scripts **depuis `alphazero/`**.

### Entraînement

**Serveur local :**
```bash
cd alphazero

# Test rapide (CPU)
python trainer.py --iterations 1 --games 10 --simulations 100

# Entraînement standard (GPU recommandé)
python trainer.py --iterations 100 --games 100 --simulations 800 --steps 500 --device cuda

# Sans évaluation (plus rapide)
python trainer.py --iterations 20 --games 50 --simulations 200 --no-eval
```

| Option | Défaut | Description |
|---|---|---|
| `--iterations` | 20 | Itérations AlphaZero |
| `--games` | 100 | Parties de self-play par itération |
| `--simulations` | 800 | Simulations MCTS par coup |
| `--steps` | 300 | Pas d'entraînement par itération |
| `--batch` | 512 | Taille du batch |
| `--device` | auto | `cuda`, `cpu`, ou `auto` |
| `--eval-games` | 40 | Parties pour l'évaluation |
| `--no-eval` | — | Désactive l'évaluation |

**Google Colab :**
Utiliser `train_colab.ipynb` à la racine du projet pour un entraînement intensif avec :
- Mixed precision (FP16) pour le training et l'inférence
- Self-play parallèle (multi-processus)
- Graphes de suivi (loss, win rate)
- Upload/download des checkpoints

Checkpoints sauvegardés dans `alphazero/checkpoints/`.
Le meilleur modèle est dans `best_model.pt` — remplacé si win rate ≥ 55 % sur 40 parties.

### Évaluation

```bash
cd alphazero

# Win rate vs joueur aléatoire
python evaluate.py
# → Win rate vs random : ~95% après quelques itérations

# Test du moteur
python -c "from hex_env import HexEnv; e = HexEnv(); print(e.get_state_tensor().shape)"
# → (3, 11, 11)

# Test du réseau
python network.py
# → PASS si la loss descend correctement
```

---

## Tournois

```bash
cd alphazero
python tournament.py <ia1> <ia2> [nb_parties] [-v] [-t <s>]
```

| Option | Description |
|---|---|
| `-v` | Affichage du plateau après chaque coup |
| `-t <s>` | Temps par coup en secondes (défaut : 1.5) |

Mots-clés reconnus : `alphabeta`, `random`, `alphazero`.
Tout autre argument est une commande externe appelée via subprocess.

```bash
# Exemples
python tournament.py alphabeta random 20
python tournament.py alphabeta alphazero 10 -v -t 2.0
python tournament.py alphabeta ./TC_MG_mcts_hex 20   # IA externe
```

Les couleurs (Blue/Red) alternent automatiquement à chaque partie.

---

## Architecture

```
alphazero/
├── hex_env.py        # Moteur Hex (état, BFS, tenseur 3×11×11, mirror)
├── config.py         # Hyperparamètres centralisés
├── alphabeta.py      # Joueur Alpha-Beta (heuristique BFS 0-1)
├── random_player.py  # Joueur aléatoire
├── network.py        # ResNet 10 blocs, 128 filtres — têtes politique + valeur (~3M params)
├── mcts_az.py        # MCTS UCB-PUCT guidé par réseau (batch 16, virtual loss)
├── self_play.py      # Génération de parties + buffer circulaire (200 000 pos.)
├── trainer.py        # Boucle AlphaZero : self-play → train → eval → checkpoint
├── evaluate.py       # Comparaison de modèles, test vs random
├── play.py           # Wrapper CLI AlphaZero (protocole BOARD/PLAYER)
├── tournament.py     # Tournoi Python (appels directs ou subprocess)
train_colab.ipynb     # Notebook Colab (FP16, self-play parallèle, monitoring)
```

### Moteur (`hex_env.py`)

- État : deux tableaux numpy `bool` 11×11 — `blue` (Nord→Sud) et `red` (Ouest→Est)
- Détection de victoire par BFS flood-fill
- `mirror()` : réflexion diagonale (symétrie naturelle du Hex) — utilisée pour l'augmentation de données

### Réseau (`network.py`)

```
Entrée (batch, 3, 11, 11)
  → Conv 3→128 + BN + ReLU
  → 6 × ResBlock(128) : Conv + BN + ReLU + Conv + BN + connexion résiduelle
  ┌→ Tête Politique : Conv(128→2) + BN + ReLU + Linear(242→121) → log_softmax
  └→ Tête Valeur    : Conv(128→1) + BN + ReLU + Linear(121→256→1) → tanh
```

~1,8 M paramètres.

### MCTS AlphaZero (`mcts_az.py`)

Sélection par score UCB-PUCT :

```
U(s,a) = Q(s,a) + c_puct × P(s,a) × √ΣN(s) / (1 + N(s,a))
```

- `c_puct = 1.0` — bruit Dirichlet à la racine en self-play (α=0.03, ε=0.25)
- Température τ=1 pour les 20 premiers coups, puis τ→0 (argmax)
- Inférence FP16 automatique sur GPU (torch.amp.autocast)
- Batch MCTS : 16 feuilles par inférence GPU

### Boucle d'entraînement (`trainer.py`)

```
L = MSE(v, z) + CrossEntropy(p, π) + λ·L2
```

- `z` = résultat de la partie (+1 / −1) du point de vue du joueur courant
- `π` = distribution de visites MCTS normalisée
- Adam + cosine scheduler, `lr = 1e-3`, `λ = 1e-4`

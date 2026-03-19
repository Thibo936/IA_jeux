# TC_MG_hex — IA pour le jeu de Hex 11×11

Projet L3 (Paris 8). Trois IA jouent au Hex 11×11 : Alpha-Beta, MCTS multi-threadé, et AlphaZero (réseau de neurones + MCTS guidé).

---

## Table des matières

1. [Prérequis](#prérequis)
2. [Compilation des binaires C++](#compilation-des-binaires-c)
3. [Protocole commun des IA](#protocole-commun-des-ia)
4. [Alpha-Beta](#alpha-beta)
5. [MCTS C++](#mcts-c)
6. [AlphaZero (Python)](#alphazero-python)
   - [Installation des dépendances](#installation-des-dépendances)
   - [Entraînement](#entraînement)
   - [Évaluation](#évaluation)
   - [Jouer un coup](#jouer-un-coup)
7. [Tournois](#tournois)
8. [Architecture du code](#architecture-du-code)

---

## Prérequis

| Composant | Version minimale |
|---|---|
| g++ | C++11 |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| NumPy | 1.24+ |
| GPU NVIDIA (optionnel) | CUDA 11.8+ |

---

## Compilation des binaires C++

```bash
make hex        # compile les 4 binaires Hex
make clean      # supprime les binaires
```

Binaires produits :
- `TC_MG_alphabeta_hex` — Alpha-Beta profondeur 4
- `TC_MG_mcts_hex` — MCTS multi-threadé
- `TC_MG_random_hex` — joueur aléatoire (référence)
- `tournament_hex` — organisateur de tournois

---

## Protocole commun des IA

Toutes les IA (C++ et Python) partagent le même protocole en ligne de commande :

```
./TC_MG_<ai>_hex BOARD PLAYER [time_s]
```

| Argument | Description |
|---|---|
| `BOARD` | Chaîne de 121 caractères (row-major, gauche→droite, haut→bas) : `.` vide, `O` Blue, `@` Red |
| `PLAYER` | `O` (Blue, direction Nord-Sud) ou `@` (Red, direction Ouest-Est) |
| `time_s` | Temps alloué par coup en secondes (ignoré par Alpha-Beta) |

**Sortie stdout** : coup en notation `A1`..`K11` (lettre = colonne, chiffre = ligne)

**Sortie stderr** :
- Alpha-Beta : `SCORE:%d NODES:%d DEPTH:%d`
- MCTS C++ : `ITERS:%d VISITS:%d WINRATE:%f TIME:%f`
- AlphaZero : `ITERS:%d VISITS:%d WINRATE:%f TIME:%f`

**Exemple** — plateau vide, Blue joue :
```bash
./TC_MG_alphabeta_hex "........................................................................................................................................................................................................." O
# → F6  (coup au centre)
```

---

## Alpha-Beta

Recherche alpha-bêta à profondeur fixe 4. L'heuristique est la différence de longueur du plus court chemin virtuel entre les deux joueurs (calculée par BFS 0-1).

```bash
# Un coup sur plateau vide, Blue joue, 1.5s alloués (ignoré)
./TC_MG_alphabeta_hex "........................................................................................................................................................................................................." O 1.5
```

---

## MCTS C++

MCTS multi-threadé avec UCB1, virtual loss et filtrage tactique (évite de donner une victoire immédiate à l'adversaire). L'évaluation des feuilles utilise 4 playouts aléatoires.

```bash
# Plateau vide, Red joue, 2 secondes
./TC_MG_mcts_hex "........................................................................................................................................................................................................." @ 2.0
```

---

## AlphaZero (Python)

Pipeline complet implémenté dans le répertoire `alphazero/`. Les modules s'importent entre eux — lancer les scripts **depuis `alphazero/`** ou via `python alphazero/play.py` depuis la racine.

### Installation des dépendances

```bash
pip install torch numpy
```

Avec GPU NVIDIA (CUDA) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Avec GPU AMD (ROCm) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

> **Note RX 6600 / gfx1032** : ce GPU nécessite un override pour être reconnu par PyTorch ROCm :
> ```bash
> export HSA_OVERRIDE_GFX_VERSION=10.3.0
> ```
> Ajouter cette ligne dans `~/.bashrc` pour la rendre permanente.

---

### Entraînement

Le script `trainer.py` orchestre la boucle AlphaZero complète :

```
self-play → entraînement réseau → évaluation → checkpoint
```

```bash
cd alphazero

# Lancement minimal (test rapide)
python trainer.py --iterations 1 --games 10 --simulations 100

# Entraînement standard (GPU recommandé)
python trainer.py --iterations 50 --games 100 --simulations 400 --device cuda

# Sans étape d'évaluation (plus rapide)
python trainer.py --iterations 20 --games 50 --simulations 200 --no-eval
```

**Options disponibles :**

| Option | Défaut | Description |
|---|---|---|
| `--iterations` | 20 | Nombre d'itérations AlphaZero |
| `--games` | 100 | Parties de self-play par itération |
| `--simulations` | 400 | Simulations MCTS par coup |
| `--steps` | 1000 | Pas d'entraînement par itération |
| `--batch` | 512 | Taille du batch |
| `--device` | auto | `cuda`, `cpu`, ou `auto` |
| `--eval-games` | 40 | Parties pour l'évaluation |
| `--no-eval` | — | Désactive l'évaluation (plus rapide) |

**Checkpoints** : sauvegardés dans `alphazero/checkpoints/`. Le meilleur modèle accepté est toujours dans `alphazero/checkpoints/best_model.pt`.

**Critère d'acceptation** : le nouveau modèle remplace l'ancien si son win rate est ≥ 55 % sur 40 parties (20 par couleur).

---

### Évaluation

**Test rapide contre un joueur aléatoire :**
```bash
cd alphazero
python evaluate.py
# → Win rate vs random : 95%  (attendu après quelques itérations)
```

**Test unitaire du moteur Python :**
```bash
python -c "
from alphazero.hex_env import HexEnv
e = HexEnv()
print(e.get_state_tensor().shape)   # → (3, 11, 11)
print(len(e.get_legal_moves()))     # → 121
"
```

**Test du réseau (overfit sur données synthétiques) :**
```bash
cd alphazero
python network.py
# → PASS si la loss descend correctement
```

**Test MCTS sans réseau (politique uniforme) :**
```bash
python -c "
from alphazero.mcts_az import MCTSAgent
a = MCTSAgent(None, sims=50)
print('ok')
"
```

---

### Jouer un coup

`play.py` est le wrapper CLI compatible avec le protocole BOARD/PLAYER. Il charge automatiquement `checkpoints/best_model.pt` et retourne un coup.

```bash
cd alphazero

# Plateau vide, Blue joue, 1.5 secondes
python play.py "........................................................................................................................................................................................................." O 1.5
# → F6

# Utilisation depuis la racine du projet
python alphazero/play.py "........................................................................................................................................................................................................." @ 2.0
```

Le nombre de simulations MCTS s'adapte automatiquement au temps alloué :

| Temps alloué | Simulations |
|---|---|
| ≤ 0.5 s | 100 |
| ≤ 1.0 s | 200 |
| ≤ 2.0 s | 400 |
| > 2.0 s | 800 |

---

## Tournois

Le binaire `tournament_hex` orchestre des matchs entre deux IA quelconques. Chaque IA est appelée via `popen` selon le protocole BOARD/PLAYER.

**Syntaxe :**
```bash
./tournament_hex <IA1> <IA2> <nb_parties> [options]
```

**Options :**

| Option | Description |
|---|---|
| `-v` | Affichage du plateau après chaque coup |
| `-t <s>` | Temps alloué par coup (défaut : 1.5 s) |

---

**Alpha-Beta vs MCTS C++ (20 parties) :**
```bash
./tournament_hex ./TC_MG_alphabeta_hex ./TC_MG_mcts_hex 20 -t 1.5
```

**MCTS C++ vs AlphaZero Python (20 parties, 2 s/coup) :**
```bash
./tournament_hex ./TC_MG_mcts_hex "python alphazero/play.py" 20 -t 2.0
```

**AlphaZero vs Alpha-Beta (10 parties, verbeux) :**
```bash
./tournament_hex "python alphazero/play.py" ./TC_MG_alphabeta_hex 10 -v -t 2.0
```

> Les couleurs (Blue/Red) alternent automatiquement à chaque partie.

---

## Architecture du code

```
TC_MG_hex/
├── hexbb.h               # Moteur Hex C++ (bitboard 2×uint64, BFS, eval)
├── alphabeta_hex.cpp     # Alpha-Beta profondeur 4
├── mcts_hex.cpp          # MCTS multi-threadé (UCB1 + virtual loss)
├── random_hex.cpp        # Joueur aléatoire
├── tournament.cpp        # Organisateur de tournois (popen)
├── Makefile
└── alphazero/
    ├── config.py         # Hyperparamètres centralisés
    ├── hex_env.py        # Moteur Hex Python/numpy (BFS, tenseur 3×11×11)
    ├── network.py        # CNN ResNet 6 blocs, 128 filtres (PyTorch)
    ├── mcts_az.py        # MCTS UCB-PUCT guidé par réseau
    ├── self_play.py      # Génération de parties + buffer circulaire
    ├── trainer.py        # Boucle AlphaZero (self-play → train → eval)
    ├── evaluate.py       # Comparaison de modèles, test vs random
    └── play.py           # Wrapper CLI compatible protocole BOARD/PLAYER
```

### Moteur Python (`hex_env.py`)

- État : deux tableaux numpy `bool` 11×11 (`blue`, `red`)
- `get_state_tensor()` → `float32 (3, 11, 11)` : plans Blue, Red, joueur courant
- `is_terminal()` → BFS identique à `hexbb.h`
- `mirror()` → augmentation par réflexion diagonale (symétrie naturelle du Hex)

### Réseau (`network.py`)

```
Entrée (batch, 3, 11, 11)
  → Conv 3→128 + BN + ReLU
  → 6 × ResBlock(128) : Conv + BN + ReLU + Conv + BN + skip
  ┌→ Tête Politique : Conv(128→2,1×1) + BN + ReLU + Linear(242→121) → log_softmax
  └→ Tête Valeur    : Conv(128→1,1×1) + BN + ReLU + Linear(121→256→1) → tanh
```

1,8 M paramètres au total.

### MCTS AlphaZero (`mcts_az.py`)

Score UCB-PUCT pour sélectionner l'enfant à explorer :

```
U(s,a) = Q(s,a) + c_puct × P(s,a) × √ΣN(s) / (1 + N(s,a))
```

- `c_puct = 1.0`, `P(s,a)` = probabilité issue de la tête Politique
- Bruit Dirichlet à la racine en self-play (α=0.3, ε=0.25)
- Température τ=1 pour les 15 premiers coups, puis τ→0 (argmax)

### Boucle d'entraînement (`trainer.py`)

Loss totale minimisée par Adam :

```
L = MSE(v, z) + CrossEntropy(p, π) + λ × L2
```

- `z` = résultat de la partie (+1 vainqueur, -1 perdant) du point de vue du joueur courant
- `π` = distribution de visites MCTS normalisée
- `λ = 1e-4`, `lr = 1e-3`, scheduler cosine

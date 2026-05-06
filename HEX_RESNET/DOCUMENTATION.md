# HEX_RESNET — Documentation Technique Complète

> Framework AlphaZero complet pour le jeu de Hex 11×11
> Langage : Python 3.12 · PyTorch · NumPy · Numba

---

## Sommaire

- [Architecture globale](#architecture-globale)
- [Installation & prérequis](#installation--prérequis)
- [Démarrage rapide](#démarrage-rapide)
- [Module 1 — Moteur de jeu (`train/hex_env.py`)](#module-1--moteur-de-jeu)
- [Module 2 — Configuration (`train/config.py`)](#module-2--configuration)
- [Module 3 — Réseau de neurones (`train/network.py`)](#module-3--réseau-de-neurones)
- [Module 4 — MCTS guidé (`ia/mcts_az.py`)](#module-4--mcts-guidé)
- [Module 5 — Self-play (`train/self_play.py`)](#module-5--self-play)
- [Module 6 — Entraînement (`train/trainer.py`)](#module-6--entraînement)
- [Module 7 — Évaluation (`train/evaluate.py`)](#module-7--évaluation)
- [Module 8 — IA adverses (`ia/`)](#module-8--ia-adverses)
  - [Alpha-Beta (`alphabeta.py`)](#alpha-beta)
  - [MCTS Léger (`mcts_light.py`)](#mcts-léger)
  - [Heuristique (`heuristic_player.py`)](#joueur-heuristique)
  - [KataHex (`katahex.py`)](#katahex)
  - [MoHex (`mohex.py`)](#mohex)
  - [IA LLM (`ia/model_LLM/`)](#ia-llm)
- [Module 9 — Tournoi & Classement](#module-9--tournoi--classement)
  - [Tournoi (`tournament.py`)](#tournoi)
  - [Classement round-robin (`ranking.py`)](#classement-round-robin)
  - [Classement LLM (`ranking_llm.py`)](#classement-llm)
  - [Export HTML (`rank/csv_to_html.py`)](#export-html)
- [Module 10 — Outils](#module-10--outils)
  - [Interface CLI (`play.py`)](#interface-cli)
  - [Nommage des modèles (`model_naming.py`)](#nommage-des-modèles)
- [Glossaire](#glossaire)
- [FAQ](#faq)
- [Références](#références)

---

## Architecture globale

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ALPHAZERO HEX_RESNET                     │
└──────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐       ┌─────────────────────────┐
   │  HexNet     │◄──────│  Checkpoint best_model  │
   │  (ResNet)   │       │  checkpoints/best_model │
   └──────┬──────┘       └─────────────────────────┘
          │  guide (π, v)
          ▼
   ┌─────────────┐     ┌──────────────────────────────────┐
   │ MCTSAgent   │────►│  Self-play parallèle             │
   │ (UCB-PUCT)  │     │  N_PARALLEL_GAMES slots          │
   │ virtual loss│     │  inférence batchée cross-games   │
   └─────────────┘     └──────────────┬───────────────────┘
                                       │ positions (s, π, z)
                                       ▼
                        ┌──────────────────────────┐
                        │  ReplayBuffer circulaire  │
                        │  (max 150 000 positions)  │
                        └──────────────┬────────────┘
                                       │ batch aléatoire
                                       ▼
                        ┌──────────────────────────┐
                        │  Entraînement             │
                        │  loss_π (cross-entropie)  │
                        │  loss_v (MSE)             │
                        │  Adam + CosineAnnealing   │
                        └──────────────┬────────────┘
                                       │ nouveau modèle
                                       ▼
                        ┌──────────────────────────┐
                        │  Évaluation              │
                        │  nouveau vs meilleur      │
                        │  win rate ≥ 55% ?         │
                        └──────────┬───┬────────────┘
                                   │   │
                             oui   │   │ non
                                   ▼   ▼
                          best_model   rollback
                             mis à    (ancien
                             jour      modèle)
                                   │
                      ────────────►│ (itération suivante)

Couche de jeu (partagée par tous les modules) :
  HexEnv ──► get_state_tensor() ──► (3, 11, 11)  float32
               ──► legal_mask()     ──► (121,)     bool

IA adverses disponibles pour les tournois :
  AlphaZero · AlphaBeta · MoHex · KataHex · MCTSLight
  Heuristique · MonteCarloPur · Random
  + 8 IA LLM (Claude, DeepSeek, Gemini, GPT, Kimi, Mimo, MiniMax, Qwen)
```

---

## Installation & prérequis

**Python 3.12+** requis. GPU CUDA ou ROCm recommandé pour l'entraînement.

```bash
# Cloner le dépôt
git clone <url_du_depot> HEX_RESNET
cd HEX_RESNET/alphazero

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Installer les dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy numba
```

**Dépendances par module :**

| Module | Dépendances Python |
|--------|--------------------|
| `hex_env.py` | `numpy`, `collections` |
| `network.py` | `torch`, `torch.nn` |
| `mcts_az.py` | `numpy`, `torch`, `math` |
| `self_play.py` | `numpy`, `collections` |
| `trainer.py` | `torch`, `numpy`, `argparse` |
| `alphabeta.py` | `numpy`, `numba` |
| `mohex.py` | `numpy`, `numba`, `math` |
| `tournament.py` | `subprocess`, `shlex` |
| `ranking.py` | `csv`, `torch`, `concurrent.futures` |

---

## Démarrage rapide

```bash
# Depuis alphazero/train/

# 1. Vérifier que l'architecture réseau est correcte
python network.py

# 2. Lancer un entraînement complet (20 itérations)
python trainer.py --iterations 20 --games 100 --simulations 800 --device auto

# 3. Reprendre un entraînement existant (le checkpoint est chargé automatiquement)
python trainer.py --iterations 10 --no-eval

# 4. Tester le modèle entraîné contre un joueur aléatoire
python evaluate.py

# 5. Lancer un tournoi entre deux IA
cd ..
python tournament.py alphazero alphabeta 20 -v -t 2.0
```

---

## Module 1 — Moteur de jeu

**Fichier :** `alphazero/train/hex_env.py`

Implémente le moteur complet du jeu de Hex sur un plateau 11×11. Utilise des tableaux NumPy booléens pour représenter les pièces, et une BFS flood-fill pour détecter la victoire. C'est la couche fondamentale partagée par tous les autres modules.

**Règles du Hex :**
- Blue (`O`) connecte le bord Nord (ligne 0) au bord Sud (ligne 10)
- Red (`@`) connecte le bord Ouest (colonne 0) au bord Est (colonne 10)
- Chaque case a 6 voisins hexagonaux : `(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)`
- Le premier joueur à réaliser sa connexion gagne

**Imports requis :**
```python
from hex_env import HexEnv
```

**Constante de module :**

| Nom | Type | Valeur | Description |
|-----|------|--------|-------------|
| `_HEX_NEIGHBORS` | `list[tuple]` | `[(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0)]` | Deltas (dr, dc) pour les 6 voisins hexagonaux |

---

### `HexEnv`

État complet d'une partie de Hex 11×11. Toutes les opérations sont vectorisées sur des tableaux NumPy `bool (11, 11)`.

**Attributs :**

| Nom | Type | Valeur par défaut | Description |
|-----|------|-------------------|-------------|
| `blue` | `np.ndarray[bool, (11,11)]` | `zeros` | Présence des pièces Blue |
| `red` | `np.ndarray[bool, (11,11)]` | `zeros` | Présence des pièces Red |
| `blue_to_play` | `bool` | `True` | `True` = Blue joue, `False` = Red joue |
| `_winner` | `str \| None` | `None` | Cache du vainqueur (`'blue'`, `'red'`, ou `None`) |

**Exemple d'instanciation :**
```python
env = HexEnv()           # plateau vide, Blue joue en premier
env2 = env.copy()        # copie indépendante
env3 = HexEnv.from_string("." * 121, 'O')  # depuis chaîne
```

---

#### `HexEnv.from_string(board_str, player_char) -> HexEnv`

**Description**
Constructeur alternatif (classmethod). Reconstruit un état de jeu complet depuis une représentation texte en 121 caractères. Utilisé par le protocole CLI pour communiquer entre processus.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `board_str` | `str` | — | 121 caractères `'.'` (vide), `'O'` (Blue), `'@'` (Red), ordre row-major |
| `player_char` | `str` | — | `'O'` si Blue doit jouer, `'@'` si Red doit jouer |

**Retourne**
`HexEnv` — Nouvel état de jeu initialisé depuis la chaîne.

**Exemple**
```python
board = "." * 60 + "O" + "." * 60   # Blue a joué au centre (case 60)
env = HexEnv.from_string(board, '@')  # c'est au tour de Red
```

---

#### `get_legal_moves() -> np.ndarray`

**Description**
Calcule la liste des coups légaux (cases vides) en une seule opération NumPy.

**Retourne**
`np.ndarray[int64, (N,)]` — Tableau 1D d'indices `0..120` des cases vides. La longueur varie de 0 à 121.

**Exemple**
```python
env = HexEnv()
moves = env.get_legal_moves()  # → array([0, 1, 2, ..., 120])
print(len(moves))              # 121 (plateau vide)
```

---

#### `legal_mask() -> np.ndarray`

**Description**
Retourne un masque booléen `(121,)` indiquant les coups légaux. Plus efficace que `get_legal_moves()` pour masquer une distribution de politique (opération element-wise directe).

**Retourne**
`np.ndarray[bool, (121,)]` — `True` aux positions légales, `False` aux positions occupées.

**Notes**
> Utilisé par `HexNet.predict()` et `HexNet.batch_predict()` pour masquer les coups illégaux avant renormalisation de la politique.

---

#### `apply_move(pos) -> None`

**Description**
Joue le coup `pos` pour le joueur courant. Met à jour le tableau `blue` ou `red`, bascule `blue_to_play`, et invalide le cache de vainqueur.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `pos` | `int` | — | Index de la case à jouer, `0` (A1) à `120` (K11) |

**Retourne**
`None`

⚠️ **Modification in-place.** Ne vérifie pas si la case est déjà occupée.

**Exemple**
```python
env = HexEnv()
env.apply_move(60)           # Blue joue au centre (F6)
print(env.blue_to_play)      # False — c'est maintenant au tour de Red
```

---

#### `undo_move(pos, was_blue) -> None`

**Description**
Annule un coup sans allouer de mémoire. Conçu pour l'Alpha-Beta qui fait des `apply`/`undo` en rafale sans copier l'état.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `pos` | `int` | — | Index de la case à annuler |
| `was_blue` | `bool` | — | `True` si le coup avait été joué par Blue |

**Retourne**
`None`

⚠️ **Modification in-place.** Invalide le cache `_winner`.

---

#### `_blue_wins() -> bool`

**Description**
Détection de victoire pour Blue par BFS flood-fill. Initialise la file avec toutes les cases Blue de la ligne 0, puis explore les voisins. Retourne `True` dès qu'un nœud de la ligne 10 est atteint.

**Contexte algorithmique**
La BFS est de complexité O(N) avec N = 121 cases. Elle est appelée au plus une fois par `is_terminal()` (résultat mis en cache dans `_winner`).

**Retourne**
`bool` — `True` si Blue a une connexion Nord-Sud complète.

---

#### `_red_wins() -> bool`

**Description**
Identique à `_blue_wins()` mais pour Red : connexion Ouest (`col=0`) → Est (`col=10`).

**Retourne**
`bool` — `True` si Red a une connexion Ouest-Est complète.

---

#### `is_terminal() -> bool`

**Description**
Teste si la partie est terminée. Utilise le cache `_winner` pour éviter de relancer les BFS si la réponse est déjà connue.

**Retourne**
`bool` — `True` si un joueur a gagné.

**Notes**
> Effet de bord : remplit `_winner` si la partie vient de se terminer.

---

#### `winner() -> str | None`

**Description**
Retourne l'identité du vainqueur en appelant `is_terminal()`.

**Retourne**
`str | None` — `'blue'`, `'red'`, ou `None` si la partie n'est pas terminée.

---

#### `get_state_tensor() -> np.ndarray`

**Description**
Encode l'état courant en un tenseur à 3 plans utilisable par le réseau de neurones. C'est l'interface principale entre le moteur de jeu et le réseau.

**Retourne**
`np.ndarray[float32, (3, 11, 11)]`

| Plan | Contenu |
|------|---------|
| `[0]` | Pièces Blue : `1.0` si Blue occupe la case, `0.0` sinon |
| `[1]` | Pièces Red : `1.0` si Red occupe la case, `0.0` sinon |
| `[2]` | Joueur courant : `1.0` partout si Blue joue, `0.0` partout si Red joue |

**Exemple**
```python
env = HexEnv()
env.apply_move(60)
tensor = env.get_state_tensor()
print(tensor.shape)   # (3, 11, 11)
print(tensor[2, 0, 0])  # 0.0 — c'est au tour de Red
```

---

#### `copy() -> HexEnv`

**Description**
Retourne une copie profonde et indépendante de l'état courant. Copie explicitement `blue`, `red`, `blue_to_play` et `_winner`.

**Retourne**
`HexEnv` — Nouvel objet entièrement indépendant.

**Notes**
> Utilisé massivement par le MCTS pour matérialiser les états fils lors de la descente dans l'arbre.

---

#### `mirror() -> HexEnv`

**Description**
Applique la réflexion diagonale principale (transposition) au plateau. C'est la seule symétrie naturelle du Hex : elle échange exactement les deux directions de connexion (Nord-Sud ↔ Ouest-Est), ce qui revient à échanger les rôles Blue et Red.

**Retourne**
`HexEnv` — Nouvel état transposé avec Blue/Red échangés.

**Notes**
> Utilisée par `_augment()` dans `self_play.py` pour doubler le nombre d'exemples d'entraînement sans coût de génération supplémentaire.

---

#### `to_string() -> str`

**Description**
Sérialise l'état en 121 caractères `'.'`, `'O'`, `'@'` (row-major). Utilisé par le protocole CLI pour passer l'état aux IA externes via `argv`.

**Retourne**
`str` — Chaîne de 121 caractères.

---

#### `pos_to_str(pos) -> str`

**Description**
Convertit un index `0..120` en notation algébrique `'A1'..'K11'`.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `pos` | `int` | — | Index de la case (0-indexé, row-major) |

**Retourne**
`str` — Exemple : `60` → `'F6'` (colonne F, ligne 6).

---

#### `HexEnv.str_to_pos(s) -> int`

**Description**
Convertit une notation algébrique `'A1'..'K11'` en index `0..120`. Méthode statique.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `s` | `str` | — | Notation algébrique, ex. `'F6'` |

**Retourne**
`int` — Index de la case (0-indexé, row-major).

---

#### `__str__() -> str`

**Description**
Affichage ASCII du plateau avec entêtes de colonnes (`A..K`) et numéros de lignes (`1..11`). Utilise l'indentation diagonale caractéristique du Hex.

**Retourne**
`str` — Représentation multi-lignes du plateau.

---

## Module 2 — Configuration

**Fichier :** `alphazero/train/config.py`

Centralise tous les hyperparamètres du système. Importé par la quasi-totalité des modules. Modifier ce fichier impacte l'ensemble du pipeline.

**Constantes globales :**

| Nom | Type | Valeur | Description |
|-----|------|--------|-------------|
| `BOARD_SIZE` | `int` | `11` | Taille du plateau (11×11) |
| `NUM_CELLS` | `int` | `121` | Nombre total de cases |
| `NUM_CHANNELS` | `int` | `128` | Filtres par couche convolutive dans le ResNet |
| `NUM_RES_BLOCKS` | `int` | `10` | Nombre de blocs résiduels |
| `INPUT_CHANNELS` | `int` | `3` | Plans d'entrée du réseau (Blue, Red, joueur) |
| `MCTS_SIMULATIONS` | `int` | `800` | Simulations MCTS par coup (entraînement) |
| `MCTS_SIMULATIONS_EVAL` | `int` | `400` | Simulations MCTS par coup (évaluation) |
| `C_PUCT` | `float` | `1.0` | Constante d'exploration UCB-PUCT |
| `DIRICHLET_ALPHA` | `float` | `0.08` | Paramètre α du bruit Dirichlet (~10/n_actions) |
| `DIRICHLET_EPS` | `float` | `0.25` | Poids du bruit Dirichlet dans la politique racine |
| `TEMPERATURE_MOVES` | `int` | `20` | Coups joués avec τ=1 (exploration) avant de passer à τ→0 |
| `GAMES_PER_ITER` | `int` | `100` | Parties de self-play par itération AlphaZero |
| `REPLAY_BUFFER_SIZE` | `int` | `150_000` | Capacité maximale du buffer circulaire |
| `N_PARALLEL_GAMES` | `int` | `8` | Parties simultanées en self-play parallèle |
| `LEAVES_PER_GAME` | `int` | `8` | Feuilles MCTS collectées par partie par round GPU |
| `BATCH_SIZE` | `int` | `512` | Taille du batch d'entraînement |
| `LEARNING_RATE` | `float` | `1e-3` | Taux d'apprentissage initial (Adam) |
| `LR_SCHEDULER` | `str` | `"cosine"` | Type de scheduler LR |
| `LR_ETA_MIN` | `float` | `1e-5` | LR minimale pour le cosine annealing |
| `WEIGHT_DECAY` | `float` | `1e-4` | Régularisation L2 (Adam `weight_decay`) |
| `TRAIN_STEPS` | `int` | `1500` | Pas d'optimisation par itération (~4 passes sur le buffer) |
| `EVAL_GAMES` | `int` | `120` | Parties d'évaluation (60 par couleur) |
| `WIN_RATE_THRESHOLD` | `float` | `0.55` | Seuil de win rate pour accepter le nouveau modèle |
| `CHECKPOINT_DIR` | `str` | `"checkpoints"` | Dossier des checkpoints intermédiaires |
| `BEST_MODEL_FILE` | `str` | `"checkpoints/best_model.pt"` | Chemin du meilleur modèle courant |
| `MODEL_DIR` | `str` | `"model"` | Dossier des modèles acceptés (copies incrémentales) |

---

## Module 3 — Réseau de neurones

**Fichier :** `alphazero/train/network.py`

Implémente l'architecture ResNet d'AlphaZero pour Hex 11×11. Le réseau produit simultanément une distribution de politique (sur les 121 cases) et une estimation de valeur de la position.

**Imports requis :**
```python
import torch
from network import HexNet
```

---

### `ResBlock`

Bloc résiduel standard avec deux convolutions 3×3, BatchNorm, et connexion sautée (skip connection).

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `conv1` | `nn.Conv2d(C→C, 3×3, padding=1)` | Première convolution |
| `bn1` | `nn.BatchNorm2d(C)` | Normalisation après conv1 |
| `conv2` | `nn.Conv2d(C→C, 3×3, padding=1)` | Deuxième convolution |
| `bn2` | `nn.BatchNorm2d(C)` | Normalisation après conv2 |

**Flux de données :**
```
x → conv1 → bn1 → ReLU → conv2 → bn2 → (+x) → ReLU
```

#### `ResBlock.forward(x) -> torch.Tensor`

**Description**
Passe avant d'un bloc résiduel. Calcule `ReLU(bn2(conv2(ReLU(bn1(conv1(x))))) + x)`.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `x` | `torch.Tensor` | — | Tenseur `(batch, C, 11, 11)` |

**Retourne**
`torch.Tensor` — Même forme `(batch, C, 11, 11)`.

---

### `HexNet`

Réseau AlphaZero complet pour Hex 11×11. Architecture à double tête : **politique** et **valeur**.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `stem` | `nn.Sequential` | Couche d'entrée : Conv(3→C, 3×3) + BN + ReLU |
| `res_blocks` | `nn.Sequential` | Pile de `NUM_RES_BLOCKS` blocs résiduels |
| `policy_conv` | `nn.Conv2d(C→2, 1×1)` | Tête politique : réduction à 2 plans |
| `policy_bn` | `nn.BatchNorm2d(2)` | BN de la tête politique |
| `policy_fc` | `nn.Linear(2×121→121)` | Couche linéaire finale de la politique |
| `value_conv` | `nn.Conv2d(C→1, 1×1)` | Tête valeur : réduction à 1 plan |
| `value_bn` | `nn.BatchNorm2d(1)` | BN de la tête valeur |
| `value_fc1` | `nn.Linear(121→256)` | Première couche linéaire de la valeur |
| `value_fc2` | `nn.Linear(256→1)` | Couche de sortie de la valeur |

**Exemple d'instanciation :**
```python
import torch
from network import HexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HexNet().to(device)  # utilise NUM_CHANNELS=128, NUM_RES_BLOCKS=10
net.eval()
```

---

#### `HexNet.forward(x) -> tuple[torch.Tensor, torch.Tensor]`

**Description**
Passe avant du réseau. Calcule simultanément la politique et la valeur pour un batch d'états.

**Contexte algorithmique**
Correspond à l'évaluation `(p, v) = f_θ(s)` de la formule AlphaZero (voir papier, §3.1). La politique est renvoyée en log-softmax pour la stabilité numérique lors du calcul de la cross-entropie.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `x` | `torch.Tensor` | — | Batch d'états `(batch, 3, 11, 11)` float32 |

**Retourne**
`tuple[torch.Tensor, torch.Tensor]`
- `log_policy` : `(batch, 121)` — Log-probabilités (log_softmax). Non masquées par les coups légaux.
- `value` : `(batch, 1)` — Estimation tanh ∈ [-1, 1]. `+1` = victoire certaine du joueur courant.

**Flux de données (dimensions) :**
```
(B, 3, 11, 11)
  → stem     → (B, 128, 11, 11)
  → res×10   → (B, 128, 11, 11)
  ┌──────────────────────────────┐
  │ Tête politique                │
  │ policy_conv → (B, 2, 11, 11) │
  │ policy_bn + ReLU             │
  │ view → (B, 242)              │
  │ policy_fc → (B, 121)         │
  │ log_softmax → (B, 121)       │
  └──────────────────────────────┘
  ┌──────────────────────────────┐
  │ Tête valeur                   │
  │ value_conv → (B, 1, 11, 11)  │
  │ value_bn + ReLU              │
  │ view → (B, 121)              │
  │ value_fc1 → (B, 256) + ReLU  │
  │ value_fc2 → (B, 1) + tanh   │
  └──────────────────────────────┘
```

---

#### `HexNet.predict(state_tensor, legal_mask, device) -> tuple[np.ndarray, float]`

**Description**
Inférence pour un seul état, sans gradient. Masque les coups illégaux et renormalise la politique sur les coups légaux uniquement.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `state_tensor` | `np.ndarray[float32, (3,11,11)]` | — | Tenseur d'état (depuis `get_state_tensor()`) |
| `legal_mask` | `np.ndarray[bool, (121,)]` | — | Masque des coups légaux (depuis `legal_mask()`) |
| `device` | `torch.device` | — | Device cible |

**Retourne**
`tuple[np.ndarray, float]`
- `policy_probs` : `(121,)` float32, normalisée sur les coups légaux uniquement
- `value` : float ∈ [-1, 1]

**Notes**
> Si la somme des probabilités sur les coups légaux est < 1e-8 (réseau saturé), bascule sur une politique uniforme sur les coups légaux.

---

#### `HexNet.batch_predict(states, legal_masks, device) -> tuple[np.ndarray, np.ndarray]`

**Description**
Inférence batchée pour plusieurs états simultanément. Optimisée pour l'utilisation GPU dans le self-play parallèle. Utilise `torch.amp.autocast` si CUDA est disponible.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `states` | `np.ndarray[float32, (B,3,11,11)]` | — | Batch d'états |
| `legal_masks` | `np.ndarray[bool, (B,121)]` | — | Masques des coups légaux |
| `device` | `torch.device` | — | Device cible |

**Retourne**
`tuple[np.ndarray, np.ndarray]`
- `policies` : `(B, 121)` float32, chaque ligne normalisée sur les légaux
- `values` : `(B,)` float32

---

#### `test_overfit(device_str, steps) -> bool`

**Description**
Test de santé de l'architecture : vérifie que le réseau peut mémoriser 10 positions synthétiques aléatoires. Si la loss descend sous 0.5 en 200 étapes, l'architecture est correcte.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `device_str` | `str` | `"cpu"` | Device de test |
| `steps` | `int` | `200` | Nombre de pas d'optimisation |

**Retourne**
`bool` — `True` si la loss finale < 0.5 (test réussi).

**Exemple**
```python
# Lancer directement
python network.py
# Sortie : PASS ou FAIL
```

---

## Module 4 — MCTS guidé

**Fichier :** `alphazero/ia/mcts_az.py`

Implémente le MCTS (Monte Carlo Tree Search) guidé par le réseau de neurones selon la formulation AlphaZero (UCB-PUCT). Optimisé pour maximiser l'utilisation GPU via l'inférence batchée et le virtual loss.

**Constantes de module :**

| Nom | Type | Valeur | Description |
|-----|------|--------|-------------|
| `_VIRTUAL_LOSS` | `int` | `3` | Pénalité appliquée lors de la sélection batchée |
| `_BATCH_SIZE` | `int` | `32` | Taille du batch de feuilles pour l'inférence GPU |

---

### `MCTSNode`

Nœud de l'arbre MCTS. Utilise `__slots__` pour minimiser l'empreinte mémoire (les arbres peuvent contenir des millions de nœuds).

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `env` | `HexEnv \| None` | État du jeu à ce nœud. `None` si expansion paresseuse. |
| `parent` | `MCTSNode \| None` | Nœud parent dans l'arbre |
| `move` | `int` | Coup qui a mené à ce nœud (-1 pour la racine) |
| `children` | `dict[int, MCTSNode]` | Enfants indexés par coup |
| `N` | `int` | Nombre de visites |
| `W` | `float` | Somme des valeurs remontées |
| `Q` | `float` | Valeur moyenne Q = W/N |
| `P` | `float` | Prior : probabilité a priori issue du réseau |
| `is_expanded` | `bool` | `True` si le nœud a déjà été expandé |
| `is_terminal` | `bool` | `True` si la position est terminale |

**Notes**
> L'expansion est paresseuse : `env` est initialisé à `None` à la création de l'enfant et matérialisé uniquement lors de la première visite (appel à `_select_leaf`). Cela évite d'allouer 121 copies d'état pour chaque nœud expandé.

---

### `MCTSAgent`

Agent MCTS complet guidé par un réseau de neurones (AlphaZero UCB-PUCT). Gère les simulations séquentielles et batchées, le virtual loss, et le bruit Dirichlet.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `net` | `HexNet \| None` | Réseau de neurones. `None` → politique uniforme (tests) |
| `device` | `torch.device` | Device d'inférence |
| `sims` | `int` | Nombre de simulations par coup |
| `c_puct` | `float` | Constante d'exploration UCB-PUCT |
| `add_dirichlet` | `bool` | Ajoute le bruit Dirichlet à la racine (self-play uniquement) |

**Exemple d'instanciation :**
```python
import torch
from network import HexNet
from mcts_az import MCTSAgent

device = torch.device("cuda")
net = HexNet().to(device)
agent = MCTSAgent(net, device=device, sims=800, add_dirichlet=True)
```

---

#### `MCTSAgent._evaluate(env) -> tuple[np.ndarray, float]`

**Description**
Appelle le réseau sur l'état `env` et retourne `(policy, value)` depuis le point de vue du joueur courant. Si `net=None`, retourne une politique uniforme et valeur 0.

**Retourne**
`tuple[np.ndarray[float32,(121,)], float]`

---

#### `MCTSAgent._select_child(node) -> int`

**Description**
Sélectionne le coup enfant qui maximise le score UCB-PUCT. Formule utilisée (voir papier AlphaZero, eq. 2) :

```
score(a) = Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
```

où `Q` est mesuré du point de vue du parent (d'où le signe `-child.Q`).

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `node` | `MCTSNode` | — | Nœud parent dont on sélectionne le meilleur enfant |

**Retourne**
`int` — Index du coup sélectionné, ou -1 si pas d'enfants.

---

#### `MCTSAgent._expand_with_policy(node, policy) -> None`

**Description**
Expanse un nœud en créant les nœuds enfants pour tous les coups avec `policy[move] > 0`. Utilise l'expansion paresseuse : `env=None` pour tous les enfants.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `node` | `MCTSNode` | — | Nœud à expanser (marqué `is_expanded=True`) |
| `policy` | `np.ndarray[float32,(121,)]` | — | Distribution de politique sur les coups |

---

#### `MCTSAgent._expand(node) -> float`

**Description**
Expanse le nœud en évaluant via le réseau. Détecte les terminaux (retourne -1.0) et appelle `_expand_with_policy`.

**Retourne**
`float` — Valeur du nœud du point de vue du joueur courant. `-1.0` si terminal.

---

#### `MCTSAgent._backprop(node, value) -> None`

**Description**
Remonte la valeur depuis la feuille vers la racine. Alterne le signe à chaque niveau : `v → -v → v → ...` car la valeur est toujours du point de vue du joueur qui joue au nœud.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `node` | `MCTSNode` | — | Feuille évaluée |
| `value` | `float` | — | Valeur estimée du point de vue du joueur à `node` |

---

#### `MCTSAgent._apply_virtual_loss(node) -> None`

**Description**
Applique une virtual loss sur tout le chemin de la feuille à la racine. Fait croire à d'autres threads/batchs que ce chemin est "déjà en cours d'exploration", les encourageant à choisir d'autres chemins.

**Formule :** `N += _VIRTUAL_LOSS`, `W -= _VIRTUAL_LOSS`, mise à jour de `Q`.

**Notes**
> Le virtual loss est une technique de parallélisation pour le MCTS. Il permet à plusieurs feuilles d'être sélectionnées simultanément (pour être évaluées en batch) sans que plusieurs threads ne descendent dans le même chemin.

---

#### `MCTSAgent._undo_virtual_loss(node) -> None`

**Description**
Annule le virtual loss après l'expansion et la backpropagation réelles.

---

#### `MCTSAgent._select_leaf(root) -> MCTSNode`

**Description**
Descend de la racine jusqu'à une feuille non-expandée en suivant la politique UCB-PUCT. Matérialise les environnements fils à la volée (expansion paresseuse).

**Retourne**
`MCTSNode` — Feuille non-expandée.

---

#### `MCTSAgent._simulate_batch(root, batch_size) -> int`

**Description**
Sélectionne `batch_size` feuilles avec virtual loss, évalue tout le batch en une seule inférence GPU, puis expand et backpropage chaque feuille.

**Contexte algorithmique**
C'est le cœur de l'optimisation GPU : au lieu d'alterner sélection/inférence/backprop pour chaque simulation, on collecte `batch_size` feuilles, on fait **une seule** inférence GPU batchée, puis on traite toutes les feuilles. Cela multiplie l'utilisation du GPU par un facteur proche de `batch_size`.

**Retourne**
`int` — Nombre de simulations effectivement réalisées.

---

#### `MCTSAgent.get_policy(env, move_count, return_root, reuse_root) -> np.ndarray | tuple`

**Description**
Lance `self.sims` simulations depuis l'état `env` et retourne la distribution de visites π.

**Contexte algorithmique**
Après les simulations, la politique retournée est :
- Si `move_count < TEMPERATURE_MOVES` (τ=1) : π(a) = N(a) / ΣN, distribution proportionnelle aux visites
- Sinon (τ→0) : π(meilleur_coup) = 1, tout le reste = 0

Le bruit Dirichlet est ajouté à la racine uniquement si `add_dirichlet=True` (self-play) :
`P(a) ← (1-ε)·P(a) + ε·η(a)` où `η ~ Dir(α)` et `ε=0.25, α=0.08`.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `env` | `HexEnv` | — | État courant |
| `move_count` | `int` | `0` | Numéro du coup (détermine la température τ) |
| `return_root` | `bool` | `False` | Si `True`, retourne aussi le nœud racine |
| `reuse_root` | `MCTSNode \| None` | `None` | Sous-arbre à réutiliser (tree reuse) |

**Retourne**
`np.ndarray[float32,(121,)]` — Distribution de politique π.
Ou `tuple[np.ndarray, MCTSNode]` si `return_root=True`.

**Exemple**
```python
env = HexEnv()
pi = agent.get_policy(env, move_count=0)
best_move = int(pi.argmax())
```

---

#### `MCTSAgent.select_move(env, move_count) -> int`

**Description**
Raccourci : appelle `get_policy()` et sélectionne un coup selon la température.

**Retourne**
`int` — Index du coup sélectionné (0..120).

---

## Module 5 — Self-play

**Fichier :** `alphazero/train/self_play.py`

Génère les données d'entraînement par self-play. Implémente un buffer circulaire et le self-play parallèle avec batching cross-games pour maximiser l'utilisation GPU.

---

### `ReplayBuffer`

Buffer circulaire stockant les triplets `(état, politique, valeur)` issus du self-play.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `max_size` | `int` | Capacité maximale |
| `_states` | `deque[np.ndarray]` | États `(3, 11, 11)` |
| `_policies` | `deque[np.ndarray]` | Politiques MCTS `(121,)` |
| `_values` | `deque[float]` | Valeurs z ∈ {-1, +1} |

---

#### `ReplayBuffer.add(state, policy, value) -> None`

**Description**
Ajoute un exemple au buffer. Si le buffer est plein, l'exemple le plus ancien est automatiquement éjecté (comportement `deque(maxlen=...)`).

⚠️ **Modification in-place du buffer.** Les tableaux ne sont pas copiés, il faut les passer déjà copiés.

---

#### `ReplayBuffer.sample(batch_size) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

**Description**
Tire un batch aléatoire sans remise.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `batch_size` | `int` | — | Taille du batch |

**Retourne**
`tuple` de :
- `states` : `(batch_size, 3, 11, 11)` float32
- `policies` : `(batch_size, 121)` float32
- `values` : `(batch_size,)` float32

**Lève**
- `ValueError` — Si `batch_size > len(buffer)` (numpy échoue sur `np.random.choice`)

---

#### `ReplayBuffer.save(path) -> None`

**Description**
Sauvegarde le buffer compressé sur disque au format `.npz`.

⚠️ Ne fait rien si le buffer est vide.

---

#### `ReplayBuffer.load(path) -> int`

**Description**
Restaure le buffer depuis un fichier `.npz`. Utilisé pour reprendre un entraînement.

**Retourne**
`int` — Nombre d'exemples chargés. `0` si le fichier n'existe pas.

---

#### `_augment(state, policy, value) -> tuple`

**Description**
Augmentation de données par réflexion diagonale (transposition). Double le nombre d'exemples d'entraînement sans coût de génération supplémentaire.

**Transformation appliquée :**
- `state_aug[0] = state[1].T` — Red transposé devient nouveau Blue
- `state_aug[1] = state[0].T` — Blue transposé devient nouveau Red
- `state_aug[2] = 1 - state[2]` — Joueur inversé
- `policy_aug[j] = policy[i]` où `j = c*11 + r` pour `i = r*11 + c` (transposition des indices)
- `value` : conservé tel quel (la symétrie échange totalement les rôles)

**Retourne**
`tuple[np.ndarray, np.ndarray, float]` — `(state_aug, policy_aug, value)`.

---

#### `play_one_game(agent, buffer, augment, verbose) -> str`

**Description**
Joue une partie complète en self-play séquentiel. Enregistre tous les états dans le buffer avec la valeur z correcte après la fin de partie.

**Contexte algorithmique**
La valeur z est assignée rétrospectivement (après la fin de partie) :
- Pour un état où le joueur `X` devait jouer : `z = +1` si `X` a gagné, `-1` sinon.

Le tree reuse est activé : le sous-arbre du coup joué est réutilisé comme racine au coup suivant, économisant ~800 simulations par coup.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `agent` | `MCTSAgent` | — | Agent MCTS |
| `buffer` | `ReplayBuffer` | — | Buffer cible |
| `augment` | `bool` | `True` | Si `True`, ajoute aussi la version miroir |
| `verbose` | `bool` | `False` | Affiche les coups dans la console |

**Retourne**
`str` — `'blue'` ou `'red'` (vainqueur).

---

### `GameSlot`

Encapsule une partie en cours pour le self-play parallèle. Chaque slot est une machine à états qui peut être avancée indépendamment entre deux rounds d'inférence GPU.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `env` | `HexEnv` | État courant de la partie |
| `history` | `list` | `[(state, pi, blue_to_play), ...]` |
| `move_count` | `int` | Nombre de coups joués |
| `sims_remaining` | `int` | Simulations restantes pour le coup courant |
| `reuse_node` | `MCTSNode \| None` | Sous-arbre pour le tree reuse |
| `root` | `MCTSNode` | Racine MCTS courante |

---

#### `GameSlot.collect_leaves(agent, max_leaves) -> list`

**Description**
Sélectionne jusqu'à `max_leaves` feuilles MCTS pour ce slot. Traite immédiatement les terminaux et les collisions (nœuds déjà expandés). Les feuilles retournées nécessitent une inférence réseau.

---

#### `GameSlot.dispatch_results(agent, leaves, policies, values) -> None`

**Description**
Reçoit les résultats de l'inférence GPU et applique l'expansion + backpropagation pour chaque feuille.

---

#### `GameSlot.advance_move(agent) -> bool`

**Description**
Calcule la distribution π à partir des visites, applique la température, choisit et joue un coup, et initialise le prochain MCTS (avec tree reuse si possible).

**Retourne**
`bool` — `True` si la partie est terminée après ce coup.

---

#### `GameSlot.finalize(buffer, augment) -> str`

**Description**
Écrit toutes les positions de la partie dans le buffer avec les valeurs z correctes.

**Retourne**
`str` — `'blue'` ou `'red'` (vainqueur).

---

#### `run_self_play(agent, buffer, num_games, verbose) -> dict`

**Description**
Lance `num_games` parties en parallèle. Les slots sont gérés en round-robin : à chaque itération, on collecte les feuilles de tous les slots actifs, on fait **une seule** inférence GPU batchée cross-games, puis on dispatche les résultats. Affiche une barre de progression.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `agent` | `MCTSAgent` | — | Agent MCTS partagé entre tous les slots |
| `buffer` | `ReplayBuffer` | — | Buffer cible |
| `num_games` | `int` | `GAMES_PER_ITER` | Nombre de parties à jouer |
| `verbose` | `bool` | `False` | Affiche les coups |

**Retourne**
`dict` — `{'blue_wins': int, 'red_wins': int, 'total': int}`

---

## Module 6 — Entraînement

**Fichier :** `alphazero/train/trainer.py`

Implémente la boucle principale AlphaZero : self-play → entraînement → évaluation. Accessible en ligne de commande.

**Usage :**
```bash
python trainer.py --iterations 20 --games 100 --simulations 800 --device auto
```

---

#### `train_step(net, optimizer, states, policies, values, device) -> tuple[float, float, float]`

**Description**
Un pas d'optimisation complet. Calcule les deux pertes (politique et valeur), leur somme, rétropropage, clip le gradient, et met à jour les paramètres.

**Contexte algorithmique**
Selon la formulation AlphaZero :
- `loss_π = -Σ π_MCTS · log(π_réseau)` (cross-entropie — `π_MCTS` est la cible fixe)
- `loss_v = MSE(v_réseau, z)` (erreur quadratique entre prédiction et résultat réel)
- `loss = loss_π + loss_v`
- Gradient clippé à 1.0 pour la stabilité.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `net` | `HexNet` | — | Réseau en mode `train()` |
| `optimizer` | `torch.optim.Optimizer` | — | Optimiseur (Adam) |
| `states` | `np.ndarray[float32,(B,3,11,11)]` | — | Batch d'états |
| `policies` | `np.ndarray[float32,(B,121)]` | — | Cibles de politique (distributions MCTS) |
| `values` | `np.ndarray[float32,(B,)]` | — | Cibles de valeur (résultats z ∈ {-1,+1}) |
| `device` | `torch.device` | — | Device |

**Retourne**
`tuple[float, float, float]` — `(loss_totale, loss_valeur, loss_politique)`.

---

#### `train_epoch(net, optimizer, scheduler, buffer, steps, batch_size, device) -> dict`

**Description**
Lance `steps` pas d'entraînement consécutifs avec échantillonnage aléatoire du buffer. Affiche une barre de progression. Arrête si le buffer est trop petit.

**Retourne**
`dict` — `{'loss': float, 'loss_value': float, 'loss_policy': float}` (moyennes sur tous les steps).

---

#### `save_checkpoint(net, path) -> None`

**Description**
Sauvegarde le `state_dict` du réseau dans un fichier `.pt`.

⚠️ Crée le dossier parent si nécessaire.

---

#### `load_checkpoint(net, path, device) -> bool`

**Description**
Charge un checkpoint. Gère le cas d'incompatibilité d'architecture (ex: changement de `NUM_CHANNELS`).

**Retourne**
`bool` — `True` si le chargement a réussi, `False` si le fichier n'existe pas ou si l'architecture est incompatible.

---

#### `main()`

**Description**
Boucle principale AlphaZero. Enchaîne pour chaque itération :
1. Self-play → génération de `GAMES_PER_ITER` parties
2. Entraînement → `TRAIN_STEPS` pas d'optimisation sur le buffer
3. Évaluation → comparaison nouveau modèle vs meilleur modèle

**Stratégie d'évaluation :**
- Si win rate ≥ `WIN_RATE_THRESHOLD` (55%) : le nouveau modèle devient le meilleur, le buffer est sauvegardé.
- Sinon : rollback vers l'ancien modèle, le buffer n'est pas sauvegardé (évite de polluer avec des positions d'un modèle écarté).
- L'optimiseur et le scheduler continuent leur trajectoire sans discontinuité dans les deux cas.

**Arguments CLI :**

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--iterations` | `20` | Nombre d'itérations AlphaZero |
| `--games` | `100` | Parties par itération |
| `--simulations` | `800` | Simulations MCTS par coup |
| `--steps` | `1500` | Steps d'entraînement par itération |
| `--batch` | `512` | Batch size |
| `--device` | `"auto"` | `cuda` / `cpu` / `auto` |
| `--eval-games` | `120` | Parties d'évaluation |
| `--no-eval` | `False` | Désactive l'évaluation (accélère le bootstrap) |

---

## Module 7 — Évaluation

**Fichier :** `alphazero/train/evaluate.py`

Compare deux modèles par tournoi interne pour décider si le nouveau modèle est meilleur que le précédent.

---

#### `_play_game(agent_blue, agent_red) -> str`

**Description**
Joue une partie complète entre deux agents MCTS. Utilise toujours τ→0 (move_count=999, argmax) pour une comparaison déterministe.

**Retourne**
`str` — `'blue'` ou `'red'`.

---

#### `evaluate_models(new_net, best_net, device, num_games, sims) -> float`

**Description**
Fait jouer `new_net` contre `best_net` sur `num_games` parties, en alternant les couleurs (moitié avec `new_net=Blue`, moitié avec `new_net=Red`) pour éliminer le biais de premier joueur.

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `new_net` | `HexNet` | — | Nouveau modèle candidat |
| `best_net` | `HexNet` | — | Meilleur modèle actuel |
| `device` | `torch.device` | — | Device d'inférence |
| `num_games` | `int` | `EVAL_GAMES` | Nombre total de parties |
| `sims` | `int` | `MCTS_SIMULATIONS_EVAL` | Simulations par coup (400 par défaut) |

**Retourne**
`float` — Win rate de `new_net` ∈ [0, 1].

---

#### `evaluate_vs_random(net, device, num_games, sims) -> float`

**Description**
Évalue un modèle contre un joueur aléatoire. Utile pour valider rapidement qu'un modèle en début d'entraînement a bien appris quelque chose.

**Retourne**
`float` — Win rate du modèle ∈ [0, 1]. Un modèle correct devrait atteindre > 90% rapidement.

---

## Module 8 — IA adverses

### Alpha-Beta

**Fichier :** `alphazero/ia/alphabeta.py`

Joueur déterministe basé sur l'algorithme négamax Alpha-Beta. Heuristique : différence de plus court chemin virtuel BFS 0-1 (acceleré par Numba JIT). Optimisations : table de transposition Zobrist, killer moves, history heuristic.

---

#### `_shortest_path_jit(player_flat, blocker_flat, start_row, N, nb_dr, nb_dc) -> int`

**Description**
BFS 0-1 compilée Numba (JIT, mise en cache). Calcule le plus court chemin virtuel d'un joueur d'un bord à l'autre, en comptant le nombre de cases vides à traverser.

**Coût de passage :**
- Case déjà occupée par le joueur : coût 0 (arc "gratuit")
- Case vide : coût 1
- Case occupée par l'adversaire : bloquée (inaccessible)

**Contexte algorithmique**
La BFS 0-1 (deque avec priorité) est équivalente à Dijkstra pour des poids 0/1. Elle trouve en O(N) le chemin de connexion le plus court en nombre de cases vides à placer. Cette métrique est aussi appelée "distance virtuelle" ou "resistance".

**Paramètres**

| Nom | Type | Défaut | Description |
|-----|------|--------|-------------|
| `player_flat` | `np.ndarray[bool,(121,)]` | — | Cases du joueur (tableau 1D) |
| `blocker_flat` | `np.ndarray[bool,(121,)]` | — | Cases de l'adversaire |
| `start_row` | `bool` | — | `True` = Blue (ligne 0→10), `False` = Red (col 0→10) |
| `N` | `int` | — | Taille du plateau (11) |
| `nb_dr`, `nb_dc` | `np.ndarray[int32,(6,)]` | — | Deltas voisins hexagonaux |

**Retourne**
`int` — Distance minimale, ou `1_000_000_000` si impossible.

---

#### `eval_heuristic(env, blue_to_play) -> int`

**Description**
Score heuristique statique = `Red_path - Blue_path`, du point de vue du joueur courant. Un score positif indique un avantage pour le joueur courant.

**Retourne**
`int` — Score heuristique ∈ [-SCORE_WIN, +SCORE_WIN].

---

#### `AlphaBetaPlayer`

Joueur Alpha-Beta pour Hex 11×11.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `depth` | `int` | Profondeur de recherche (défaut: 4) |
| `last_stats` | `dict` | Stats du dernier coup : `{'score', 'nodes', 'depth'}` |
| `_tt` | `dict` | Table de transposition persistante entre les coups |

---

#### `AlphaBetaPlayer.select_move(env, time_s) -> int`

**Description**
Sélectionne le meilleur coup par Alpha-Beta négamax avec :
1. Détection de coup gagnant immédiat (avant toute recherche)
2. Profondeur adaptative (+1 si ≤60 légaux, +2 si ≤30)
3. Ordonnancement à la racine par évaluation heuristique
4. Récursion Alpha-Beta avec table de transposition, killers, history

**Retourne**
`int` — Index du meilleur coup (0..120).

---

### MCTS Léger

**Fichier :** `alphazero/ia/mcts_light.py`

MCTS classique (UCT) avec rollouts aléatoires, sans réseau de neurones. Formule UCT standard (Kocsis & Szepesvári, 2006).

---

#### `LightMCTSPlayer`

MCTS léger avec rollouts aléatoires jusqu'en fin de partie.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `c_uct` | `float` | Constante d'exploration UCT (défaut: 1.4) |
| `min_simulations` | `int` | Simulations minimales (défaut: 128) |
| `last_stats` | `dict` | Stats : `{'iters', 'visits', 'winrate', 'time'}` |

**Formule UCT :**
```
score = (1 - child.wins/child.visits) + c_uct × √(ln(N) / child.visits)
```
Note : `1 - win_rate_enfant` = win rate du parent pour ce coup (inversion de perspective).

---

#### `LightMCTSPlayer.select_move(env, time_s) -> int`

**Description**
Lance le MCTS jusqu'à la deadline (temps alloué) ou le minimum de simulations, puis retourne le coup le plus visité.

**Retourne**
`int` — Index du meilleur coup (0..120).

---

### Joueur heuristique

**Fichier :** `alphazero/ia/heuristic_player.py`

Joueur glouton (depth-1) : évalue tous les coups légaux par BFS 0-1 et choisit celui qui maximise le score heuristique.

---

#### `HeuristicPlayer.select_move(env, time_s) -> int`

**Description**
Pour chaque coup légal, applique le coup, évalue la position (BFS 0-1), annule, et garde le meilleur. Détecte les coups gagnants immédiats avant l'évaluation.

**Retourne**
`int` — Index du meilleur coup (0..120).

---

### KataHex

**Fichier :** `alphazero/ia/katahex.py`

MCTS avec PUCT et prior heuristique calculé par patterns (centre, voisins amis, bords, complétion de ponts). Rollouts aléatoires. Maintient une **ownership map** (carte de possession) agrégée sur toutes les simulations.

---

#### `_compute_prior(env) -> dict[int, float]`

**Description**
Calcule une distribution prior softmax sur les coups légaux. Features utilisées :
- **Centre** : bonus décroissant avec la distance Manhattan au centre
- **Voisins amis** : bonus si la case a des voisins de la même couleur
- **Bord propre** : Blue bonus lignes 0/10, Red bonus colonnes 0/10
- **Complétion de pont** : fort bonus si la case est un intermédiaire d'un pont existant

**Retourne**
`dict[int, float]` — Prior normalisé par softmax (T=2).

---

#### `KataHexPlayer`

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `c_puct` | `float` | Constante PUCT (défaut: 1.4) |
| `min_simulations` | `int` | Simulations minimales (défaut: 100) |
| `ownership` | `np.ndarray \| None` | Carte d'ownership normalisée après la dernière sélection |

---

### MoHex

**Fichier :** `alphazero/ia/mohex.py`

Implémentation avancée inspirée de MoHex (Université d'Alberta). Combine MCTS + RAVE/AMAF + prior heuristique + tree reuse + rollouts Numba.

**Optimisations clés :**
- **RAVE/AMAF** : mise à jour du score estimé de chaque coup basée sur toutes les simulations où ce coup a été joué (pas seulement celles descendant dans ce sous-arbre)
- **Tree reuse** : la racine de l'arbre est avancée au coup joué entre les appels
- **Blocage forcé** : détecte et bloque immédiatement les menaces à 1 coup uniques
- **Cases mortes** : filtre les cases sans impact stratégique (fill-in trivial)
- **Prior Numba** : le calcul du prior heuristique est compilé JIT

---

#### `MoHexPlayer._rave_select(node) -> _RAVENode`

**Description**
Sélection RAVE-UCT avec progressive bias. Formule combinée :

```
β = √(rave_k / (3·N + rave_k))                    # poids RAVE
Q_combiné = (1-β)·Q_MC + β·Q_AMAF                  # mélange MCTS/RAVE
score = Q_combiné + c_uct·√(ln N / n_enfant) + c_bias·P·√N/(1+n_enfant)
```

où `Q_AMAF = amaf_wins/amaf_visits` pour le coup considéré.

---

#### `MoHexPlayer._try_reuse_tree(env) -> _RAVENode | None`

**Description**
Tente de réutiliser l'arbre en trouvant le sous-arbre correspondant à l'état actuel (0 ou 1 coup d'avance sur la racine mémorisée). Retourne `None` si la position a trop changé (partie différente ou plus de 1 coup ajouté).

---

#### `MoHexPlayer.select_move(env, time_s) -> int`

**Description**
Sélection en 6 étapes :
1. Coup gagnant immédiat
2. Blocage forcé si l'adversaire a une unique menace à 1 coup
3. Tree reuse si possible
4. Boucle MCTS jusqu'à la deadline
5. Meilleur coup par nombre de visites
6. Avance la racine dans le coup choisi (pour le prochain appel)

---

### IA LLM

**Répertoire :** `alphazero/ia/` (fichiers `*_V2.py`) et `alphazero/model_LLM/`

8 IA autonomes générées par différents LLMs, chacune implémentant sa propre variante de MCTS. Toutes respectent l'interface `select_move(env: HexEnv, time_s: float) -> int`.

| Fichier | Classe | Algorithme principal |
|---------|--------|----------------------|
| `claude_opus_4_7_xhigh_V2.py` | `ClaudeOpus47Xhigh` | MCTS-RAVE + union-find incrémental |
| `deepseek_v4_pro_max_V2.py` | — | MCTS + RAVE + prior Numba + BFS |
| `kimi_k26_V2.py` | — | MCTS + RAVE |
| `mimo_v25_pro_V2.py` | — | MCTS |
| `minimax_m2_V2.py` | — | MCTS + Minimax |
| `gemini_3_1_pro_preview.py` | — | MCTS |
| `gpt53_codex_xhigh.py` | — | MCTS |
| `qwen36plus.py` | — | MCTS |

**Caractéristiques communes :**
- Interface CLI : `python <ia>.py BOARD PLAYER [time_s]`
- Sortie stdout : coup en notation `A1..K11`
- Sortie stderr : `ITERS:N VISITS:V WINRATE:f TIME:f`
- Toutes détectent les coups gagnants immédiats et les blocages forcés

**Dossier `model_LLM/`** : versions allégées pour le tournoi LLM interne (`ranking_llm.py`), chargées dynamiquement via `importlib`.

---

## Module 9 — Tournoi & Classement

### Tournoi

**Fichier :** `alphazero/tournament.py`

Orchestre un tournoi entre deux IA quelconques. Supporte les objets Python et les processus externes.

**Usage :**
```bash
python tournament.py alphabeta alphazero 20 -v -t 2.0
python tournament.py mohex ./external_engine 10 -t 1.5
```

---

#### `AlphaZeroPlayer`

Wrapper `MCTSAgent + HexNet` pour l'usage dans les tournois. Charge automatiquement `checkpoints/best_model.pt`.

**Attributs :**

| Nom | Type | Description |
|-----|------|-------------|
| `device` | `torch.device` | Device d'inférence |
| `_agent` | `MCTSAgent` | Agent MCTS sous-jacent |
| `_sims` | `int` | Nombre de simulations |
| `last_stats` | `dict` | `{'iters', 'visits', 'winrate', 'time'}` |

---

#### `_call_external(cmd, env, time_s) -> tuple[int, str]`

**Description**
Appelle un processus externe (binaire C++, autre script Python) selon le protocole `BOARD PLAYER [time_s]`. Parse la réponse stdout et stderr.

**Protocole :**
- `argv`: `[cmd, board_str(121), player_char, time_s]`
- `stdout`: coup en notation algébrique (`"F6"`)
- `stderr`: stats `"ITERS:N VISITS:V WINRATE:f TIME:f"` (parsées et affichées)

**Retourne**
`tuple[int, str]` — `(move_index, stats_stderr_str)`, ou `(-1, "")` en cas d'erreur.

---

#### `run_tournament(ai1, ai2, n_games, verbose, time_s, ai1_name, ai2_name) -> tuple[int, int, int]`

**Description**
Lance un tournoi complet. Les couleurs alternent à chaque partie (partie paire → ai1=Blue). Affiche les résultats en temps réel et le résumé final.

**Retourne**
`tuple[int, int, int]` — `(wins_ai1, wins_ai2, errors)`.

---

#### `_resolve_ai(name, time_s) -> tuple[Any, str]`

**Description**
Résout un nom d'IA textuel en objet Python ou en chaîne de commande.

| Nom | Objet retourné |
|-----|----------------|
| `'alphabeta'` | `AlphaBetaPlayer()` |
| `'random'` | `RandomPlayer()` |
| `'alphazero'` | `AlphaZeroPlayer(sims=...)` |
| `'mc_pure'` | `PureMonteCarloPlayer()` |
| `'mcts_light'` | `LightMCTSPlayer()` |
| `'heuristic'` | `HeuristicPlayer()` |
| `'mohex'` | `MoHexPlayer()` |
| `'humain'` | `HumanPlayer()` |
| autre | Chaîne de commande externe |

---

### Classement round-robin

**Fichier :** `alphazero/ranking.py`

Tournoi round-robin unifié entre toutes les IA (classiques + AlphaZero). Stockage incrémental CSV : seuls les matchups manquants sont rejoués. Génère un rapport HTML.

**Usage :**
```bash
python ranking.py --games 50
python ranking.py --mode alphazero --workers 2
python ranking.py --no-classics
```

**Constantes :**

| Nom | Valeur | Description |
|-----|--------|-------------|
| `DEFAULT_CLASSICS` | `['random','alphabeta','mc_pure','mcts_light','heuristic','mohex']` | IA classiques par défaut |
| `TIME_PER_MOVE` | `0.5` | Secondes par coup dans le tournoi |

**Fonctionnement incrémental :**
Le CSV `rank/ranking.csv` enregistre chaque partie. À chaque lancement, le script lit le CSV existant et ne rejoue que les matchups non encore couverts. Le HTML est régénéré à chaque fois.

---

### Classement LLM

**Fichier :** `alphazero/ranking_llm.py`

Tournoi round-robin simplifié entre les IA du dossier `model_LLM/`. Charge les modules dynamiquement via `importlib`. Sorties : console + log `model_LLM/modelLLM.log`.

**Usage :**
```bash
python ranking_llm.py
```

**Constantes :**

| Nom | Valeur | Description |
|-----|--------|-------------|
| `GAMES_PER_MATCHUP` | `20` | Parties par duel |
| `TIME_PER_MOVE` | `0.5` | Secondes par coup |

---

#### `load_player_module(path) -> module`

**Description**
Importe un fichier `.py` de manière isolée (`importlib`). Redirige stdout/stderr pendant l'import pour éviter les sorties parasites.

**Retourne**
`module` — Module Python importé.

**Lève**
- Toute exception levée lors de l'exécution du module.

---

### Export HTML

**Fichier :** `alphazero/rank/csv_to_html.py`

Génère un rapport HTML interactif depuis `ranking.csv`.

**Usage :**
```bash
python3 rank/csv_to_html.py
python3 rank/csv_to_html.py --csv ranking.csv --output ranking.html
```

---

#### `infer_player_type(name) -> tuple[str, str]`

**Description**
Déduit le type et la famille d'un joueur depuis son nom.

| Nom (préfixe/exact) | Type retourné | Famille |
|---------------------|---------------|---------|
| `AZ-*` | `'AlphaZero'` | `'alphazero'` |
| `random` | `'Random'` | `'classic'` |
| `alphabeta` | `'Alpha-Beta'` | `'classic'` |
| `montecarlo`, `mc_pure` | `'Monte Carlo'` | `'classic'` |
| `mcts_light`, `mcts` | `'MCTS'` | `'classic'` |
| `heuristic` | `'Heuristique'` | `'classic'` |
| `mohex` | `'MoHex'` | `'classic'` |
| autre | `'LLM'` | `'llm'` |

---

## Module 10 — Outils

### Interface CLI

**Fichier :** `alphazero/play.py`

Wrapper CLI AlphaZero compatible avec le protocole BOARD/PLAYER utilisé par les tournois externes (binaires C++, autres frameworks).

**Usage :**
```bash
python play.py BOARD PLAYER [time_s]
# Exemple :
python play.py "$(python -c "print('.'*121)")" O 2.0
```

**Protocole :**
- **Stdin** : rien
- **argv[1]** : BOARD — 121 chars `'.'`/`'O'`/`'@'`
- **argv[2]** : PLAYER — `'O'` ou `'@'`
- **argv[3]** : time_s — Temps alloué en secondes
- **stdout** : coup en notation algébrique (`"F6"`)
- **stderr** : `"ITERS:N VISITS:V WINRATE:f TIME:f"`

---

#### `compute_sims(time_limit) -> int`

**Description**
Adapte le nombre de simulations au temps alloué.

| Temps | Simulations |
|-------|-------------|
| ≤ 0.5s | 200 |
| ≤ 1.0s | 400 |
| ≤ 2.0s | 800 |
| > 2.0s | 1600 |

---

#### `load_best_model(device) -> HexNet | None`

**Description**
Charge le meilleur modèle disponible. Cherche d'abord dans `alphazero/BEST_MODEL_FILE`, puis dans le répertoire courant.

**Retourne**
`HexNet` en mode `eval()`, ou `None` si aucun modèle n'est trouvé (fallback sur politique uniforme).

---

### Nommage des modèles

**Fichier :** `alphazero/model_naming.py`

Gère le nommage incrémental des modèles acceptés dans `model/`. Format : `model_<num>_<parent>_<day>_<month>.pt`.

---

#### `parse_model_name(filename) -> tuple | None`

**Description**
Parse un nom de fichier modèle.

**Retourne**
`tuple[int, int, int, int]` — `(num, parent, day, month)`, ou `None` si le format ne correspond pas.

---

#### `scan_models(model_dir) -> list`

**Description**
Scanne un dossier et retourne la liste triée de tous les modèles valides.

**Retourne**
`list[tuple[int, int, int, int, str]]` — `[(num, parent, day, month, abs_path), ...]` trié par `num`.

---

#### `copy_best_to_model(best_model_path, model_dir) -> str`

**Description**
Copie `best_model.pt` dans `model/` avec le bon nom incrémental. Détermine automatiquement le prochain numéro séquentiel et le numéro de parent.

⚠️ Crée `model_dir` si nécessaire.

**Retourne**
`str` — Chemin absolu du nouveau fichier copié.

---

#### `list_model_entries(model_dir) -> list[dict]`

**Description**
Retourne une liste de dicts lisibles pour l'affichage.

**Retourne**
`list[dict]` — `[{'num': int, 'parent': int, 'date_str': str, 'path': str}, ...]`.

---

## Glossaire

**AlphaZero**
: Algorithme de DeepMind (2017) pour apprendre à jouer à des jeux de plateau par auto-apprentissage (*self-play*). Combine un réseau de neurones ResNet avec un MCTS guidé. Ne nécessite aucune connaissance humaine au-delà des règles.

**AMAF (All Moves As First)**
: Heuristique pour le MCTS. Suppose qu'un coup est aussi bon (en espérance) quel que soit le moment de la partie où on le joue. Permet de mettre à jour les statistiques d'un coup pour tous les ancêtres de l'arbre qui le contiennent, accélérant la convergence.

**BFS 0-1**
: Variante de la BFS (Breadth-First Search) pour les graphes avec des poids 0 ou 1. Utilise une deque (double-ended queue) : les arcs de coût 0 sont mis en tête (`appendleft`), les arcs de coût 1 en queue (`append`). Équivalent à Dijkstra avec complexité O(V+E) au lieu de O((V+E)log V).

**Bruit Dirichlet**
: Bruit ajouté aux probabilités a priori à la racine du MCTS en self-play. `P(a) ← (1-ε)·P(a) + ε·η(a)` où `η ~ Dirichlet(α)`. Garantit l'exploration de coups improbables selon le réseau. Paramètres : `α=0.08` (≈10/121), `ε=0.25`.

**Distance virtuelle (Resistance)**
: Nombre minimum de cases vides qu'un joueur doit encore placer pour créer une connexion d'un bord à l'autre. Calculée par BFS 0-1 (coût 0 sur les cases déjà occupées, coût 1 sur les cases vides).

**ELO**
: Système de classement pour les jeux à deux joueurs. Le rating d'un joueur représente son niveau relatif : une différence de 400 points correspond à ~91% de win rate. Non implémenté nativement dans HEX_RESNET, les classements utilisent le win rate brut.

**Flood-fill**
: Algorithme BFS/DFS qui "remplit" une région connexe depuis un point de départ. Utilisé dans `_blue_wins()` et `_red_wins()` pour détecter si un joueur a réalisé sa connexion d'un bord à l'autre.

**History heuristic**
: Heuristique d'ordonnancement pour Alpha-Beta. Maintient un tableau `history[couleur][coup]` incrémenté de `depth²` à chaque coupure β. Les coups qui ont provoqué des coupures fréquemment (et à grande profondeur) sont explorés en priorité.

**Killer moves**
: Heuristique d'ordonnancement pour Alpha-Beta. Mémorise les 2 derniers coups ayant provoqué une coupure β à chaque profondeur. Ces coups sont essayés en priorité aux nœuds frères du même niveau.

**MCTS (Monte Carlo Tree Search)**
: Algorithme de recherche arborescente pour les jeux. Alterne 4 phases : Sélection (descendre par UCT/PUCT), Expansion (créer un nouveau nœud), Simulation (rollout jusqu'à la fin), Backpropagation (remonter le résultat).

**Nœud virtuel (union-find)**
: Dans les IA LLM, des nœuds virtuels représentent les bords du plateau (Nord, Sud, Ouest, Est). La victoire est détectée en vérifiant si le nœud Nord est dans la même composante connexe que le nœud Sud (pour Blue), sans relancer une BFS.

**Politique (Policy head)**
: Tête de sortie du réseau qui prédit la distribution de probabilité sur les 121 coups possibles. Entraînée par cross-entropie avec les distributions MCTS.

**PUCT (Polynomial Upper Confidence Trees)**
: Variante de UCT pour les MCTS guidés par réseau. Formule : `Q + c·P·√N/(1+n)` où `P` est le prior issu du réseau.

**RAVE/AMAF (Rapid Action Value Estimation)**
: Extension du MCTS : les statistiques d'un coup sont mises à jour pour tous les nœuds ancêtres, pas seulement pour le chemin de la simulation. Converge plus vite en début de partie.

**Replay buffer**
: Buffer circulaire stockant les triplets `(état, politique, valeur)` générés par le self-play. L'entraînement échantillonne des batchs aléatoires depuis ce buffer, ce qui brise les corrélations temporelles entre les données.

**ResNet**
: Réseau de neurones avec connexions résiduelles (He et al., 2015). Chaque bloc calcule `F(x) + x`, permettant l'apprentissage de transformations résiduelles et facilitant l'entraînement de réseaux très profonds.

**Round-robin**
: Format de tournoi où chaque participant affronte tous les autres. Dans HEX_RESNET, chaque matchup est joué en nombre pair de parties avec alternance des couleurs.

**Self-play**
: Le modèle joue contre lui-même pour générer des données d'entraînement. À chaque état, le MCTS génère une distribution de politique améliorée. La valeur z est assignée rétrospectivement selon le vainqueur final.

**Température τ**
: Paramètre contrôlant l'exploration lors de la sélection du coup. τ=1 : sélection proportionnelle aux visites (exploratoire). τ→0 : sélection du coup le plus visité (exploitatoire, argmax).

**Tree reuse**
: Optimisation qui réutilise le sous-arbre MCTS du coup joué comme nouvelle racine au coup suivant. Économise `sims-1` simulations par coup.

**UCT (Upper Confidence Trees)**
: Formule de sélection MCTS de base : `Q + c·√(ln N / n)`. Équilibre exploitation (Q élevé) et exploration (n faible).

**Union-Find (Disjoint Set Union)**
: Structure de données pour gérer des ensembles disjoints. Opérations `find()` et `union()` quasi-constantes avec compression de chemin. Utilisée dans les IA LLM pour détecter la victoire de manière incrémentale.

**Valeur (Value head)**
: Tête de sortie du réseau qui prédit la probabilité de victoire du joueur courant (tanh ∈ [-1, 1]). Entraînée par MSE avec le résultat réel de la partie.

**Virtual loss**
: Technique de parallélisation MCTS. Avant l'inférence GPU d'un batch, on applique une pénalité temporaire sur chaque chemin sélectionné, forçant les sélections suivantes à choisir d'autres chemins. La pénalité est annulée après la vraie backpropagation.

**Win rate threshold**
: Seuil de win rate (55% par défaut) en dessous duquel le nouveau modèle est rejeté et l'ancien conservé. Évite d'accepter un modèle marginalement meilleur à cause de la variance des parties d'évaluation.

**Zobrist hashing**
: Technique de hashing incrémental pour les jeux de plateau. Le hash de l'état est calculé par XOR de clés aléatoires associées à chaque (position, pièce). Mise à jour O(1) après un coup. Utilisé pour la table de transposition de l'Alpha-Beta.

---

## FAQ

**1. Comment ajouter une nouvelle IA ?**

Créez un fichier `alphazero/ia/mon_ia.py` avec une classe exposant `select_move(env: HexEnv, time_s: float) -> int`. Optionnellement, ajoutez `last_stats: dict` avec les clés `{'iters', 'visits', 'winrate', 'time'}` pour l'affichage dans le tournoi. Ajoutez le cas correspondant dans `tournament.py/_resolve_ai()`.

```python
class MonIA:
    def __init__(self): self.last_stats = {}
    def select_move(self, env: HexEnv, time_s: float) -> int:
        # Votre logique ici
        return env.get_legal_moves()[0]
```

**2. Comment reprendre un entraînement interrompu ?**

Le script `trainer.py` recharge automatiquement `checkpoints/best_model.pt` et `checkpoints/replay_buffer.npz` au démarrage. Relancez simplement :
```bash
python trainer.py --iterations 50  # Le buffer et le modèle sont restaurés
```

**3. Comment évaluer un checkpoint spécifique ?**

Chargez le checkpoint manuellement et utilisez `evaluate_vs_random()` ou `evaluate_models()` :
```python
import torch
from network import HexNet
from evaluate import evaluate_vs_random

device = torch.device("cuda")
net = HexNet().to(device)
net.load_state_dict(torch.load("checkpoints/model_iter_0010.pt", map_location=device))
wr = evaluate_vs_random(net, device, num_games=100)
print(f"Win rate vs random : {wr:.1%}")
```

**4. Comment lancer un tournoi entre tous les modèles disponibles ?**

Utilisez `ranking.py` depuis le dossier `alphazero/` :
```bash
python ranking.py --games 50 --workers 2
```
Le rapport HTML est généré dans `rank/ranking_<date>.html`.

**5. Pourquoi le réseau AlphaZero joue toujours de façon identique après l'évaluation ?**

En évaluation (dans `evaluate.py` et `tournament.py`), le MCTS est lancé avec `move_count=999`, forçant τ→0 (argmax). Ce comportement déterministe est voulu pour des comparaisons reproductibles. En self-play, les 20 premiers coups utilisent τ=1 (sélection stochastique) pour l'exploration.

---

## Références

**Papier fondateur :**
- Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. DeepMind. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)

**Jeu de Hex :**
- Nash, J. (1952). *Some games and machines for playing them*. Princeton.
- Browne, C. (2000). *Hex Strategy: Making the Right Connections*. A K Peters.
- Règles officielles : [https://en.wikipedia.org/wiki/Hex_(board_game)](https://en.wikipedia.org/wiki/Hex_(board_game))

**MoHex (inspiration) :**
- Arneson, B., Hayward, R., Henderson, P. (2010). *Monte Carlo Tree Search in Hex*. IEEE Transactions on Computational Intelligence and AI in Games.

**RAVE/AMAF :**
- Gelly, S., Silver, D. (2007). *Combining Online and Offline Knowledge in UCT*. ICML.

**BFS 0-1 / Distance virtuelle Hex :**
- Anshelevich, V. (2002). *A hierarchical approach to computer Hex*. Artificial Intelligence.

**Bibliothèques utilisées :**
- PyTorch : [https://pytorch.org](https://pytorch.org)
- NumPy : [https://numpy.org](https://numpy.org)
- Numba : [https://numba.pydata.org](https://numba.pydata.org)

---

*Documentation générée pour HEX_RESNET · Branche `one-rank` · 2026-05-01*

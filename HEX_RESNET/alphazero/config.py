# config.py — Hyperparamètres centralisés pour AlphaZero Hex 11×11

# ─── Plateau ──────────────────────────────────────────────────────────────────
BOARD_SIZE = 11
NUM_CELLS  = BOARD_SIZE * BOARD_SIZE  # 121

# ─── Réseau ───────────────────────────────────────────────────────────────────
NUM_CHANNELS    = 128   # filtres par couche convolutive
NUM_RES_BLOCKS  = 10    # blocs résiduels (ou couches conv simples)
INPUT_CHANNELS  = 3     # (Blue, Red, joueur_courant)

# ─── MCTS ─────────────────────────────────────────────────────────────────────
MCTS_SIMULATIONS    = 800   # simulations par coup (entraînement)
MCTS_SIMULATIONS_EVAL = 400 # simulations par coup (évaluation)
C_PUCT              = 1.0   # constante d'exploration UCB-PUCT
DIRICHLET_ALPHA     = 0.03  # bruit Dirichlet à la racine (~10/n_actions)
DIRICHLET_EPS       = 0.25  # poids du bruit Dirichlet
TEMPERATURE_MOVES   = 20    # coups joués avec τ=1 avant de passer à τ→0

# ─── Self-play ────────────────────────────────────────────────────────────────
GAMES_PER_ITER      = 100   # parties par itération de self-play
REPLAY_BUFFER_SIZE  = 150_000  # taille du buffer circulaire (positions)
N_PARALLEL_GAMES    = 8     # parties simultanées en self-play parallèle
LEAVES_PER_GAME     = 8     # feuilles collectées par partie par round GPU

# ─── Entraînement ─────────────────────────────────────────────────────────────
BATCH_SIZE          = 512
LEARNING_RATE       = 1e-3
LR_SCHEDULER        = "cosine"
LR_ETA_MIN          = 1e-5   # LR minimale pour le cosine annealing
WEIGHT_DECAY        = 1e-4
TRAIN_STEPS         = 1500   # steps par itération (~4 passes sur le buffer)

# ─── Évaluation ───────────────────────────────────────────────────────────────
EVAL_GAMES          = 80     # parties d'évaluation (40 par couleur)
WIN_RATE_THRESHOLD  = 0.58   # seuil pour accepter le nouveau modèle

# ─── Checkpoints ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_FILE = "checkpoints/best_model.pt"

# config.py — Hyperparamètres centralisés pour AlphaZero Hex 11×11

# ─── Plateau ──────────────────────────────────────────────────────────────────
BOARD_SIZE = 11
NUM_CELLS  = BOARD_SIZE * BOARD_SIZE  # 121

# ─── Réseau ───────────────────────────────────────────────────────────────────
NUM_CHANNELS    = 128   # filtres par couche convolutive
NUM_RES_BLOCKS  = 6     # blocs résiduels (ou couches conv simples)
INPUT_CHANNELS  = 3     # (Blue, Red, joueur_courant)

# ─── MCTS ─────────────────────────────────────────────────────────────────────
MCTS_SIMULATIONS    = 400   # simulations par coup (entraînement)
MCTS_SIMULATIONS_EVAL = 200 # simulations par coup (évaluation)
C_PUCT              = 1.0   # constante d'exploration UCB-PUCT
DIRICHLET_ALPHA     = 0.3   # bruit Dirichlet à la racine
DIRICHLET_EPS       = 0.25  # poids du bruit Dirichlet
TEMPERATURE_MOVES   = 15    # coups joués avec τ=1 avant de passer à τ→0

# ─── Self-play ────────────────────────────────────────────────────────────────
GAMES_PER_ITER      = 100   # parties par itération de self-play
REPLAY_BUFFER_SIZE  = 50_000  # taille du buffer circulaire (positions)

# ─── Entraînement ─────────────────────────────────────────────────────────────
BATCH_SIZE          = 512
LEARNING_RATE       = 2e-4
LR_SCHEDULER        = "cosine"
WEIGHT_DECAY        = 1e-4
TRAIN_STEPS         = 500    # steps par itération (limité pour éviter l'overfitting du buffer)

# ─── Évaluation ───────────────────────────────────────────────────────────────
EVAL_GAMES          = 40     # parties d'évaluation (20 par couleur)
WIN_RATE_THRESHOLD  = 0.55   # seuil pour accepter le nouveau modèle

# ─── Checkpoints ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_FILE = "checkpoints/best_model.pt"

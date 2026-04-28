# Pistes d'amélioration — HEX_RESNET

Synthèse des améliorations possibles, classées par impact attendu. Chaque entrée
indique le ou les fichiers concernés et un repère de difficulté
(★ rapide · ★★ moyen · ★★★ gros chantier).

---

## 1. Bugs / incohérences à corriger en priorité

### 1.1. La cible π enregistrée devient one-hot après `TEMPERATURE_MOVES` ★
- **Fichier** : `alphazero/train/self_play.py:241-265` (méthode `GameSlot.advance_move`).
- **Symptôme** : pour `move_count >= TEMPERATURE_MOVES`, le code écrase `pi` en
  vecteur one-hot avant de l'enregistrer dans `self.history`. La tête politique
  est donc entraînée sur des distributions effondrées pendant ~80 % de la partie,
  ce qui détruit l'information de visites MCTS.
- **Fix** : conserver la distribution de visites normalisée pour l'entraînement et
  n'appliquer la température qu'à la **sélection du coup** (`np.random.choice` vs
  `argmax`). La logique correcte existe déjà dans `mcts_az.MCTSAgent.get_policy`
  mais est dupliquée — et incorrectement — dans `GameSlot`.

### 1.2. Doc CLAUDE.md désynchronisée avec `trainer.py` ★
- `CLAUDE.md` affirme : *« Quand l'évaluation rejette un modèle, l'optimiseur +
  scheduler sont reset alongside the revert »*.
- `trainer.py:248-252` fait l'inverse : commentaire explicite *« On conserve
  optimizer et scheduler globaux »*. Décider du comportement souhaité, puis
  aligner code **et** documentation. Idem pour le nombre de blocs ResNet
  (CLAUDE.md dit ~6, `config.py` dit 10).

### 1.3. Augmentation : valeur conservée alors que les rôles s'échangent ★★
- `alphazero/train/self_play.py:74-102` (`_augment`) transpose le plateau, échange
  Blue/Red **et** inverse le plan « joueur courant », mais conserve `value`. Le
  commentaire le justifie ; c'est cohérent **uniquement** parce que les rôles
  s'échangent intégralement. Ajouter un test unitaire qui vérifie cette
  invariance sur quelques positions terminales (sinon une régression future
  passera inaperçue et empoisonnera le buffer).

### 1.4. `torch.load` sans `weights_only=True` ★
- `trainer.py:124`, `evaluate.py:150`, et toutes les routines de chargement
  d'`ia/`. PyTorch 2.x émet un `FutureWarning` et passera bientôt en `True` par
  défaut. Passer dès maintenant `weights_only=True` pour éviter les surprises et
  réduire la surface d'attaque (pickle arbitraire).

---

## 2. Performance — moteur de jeu et MCTS

### 2.1. JIT-compiler la détection de victoire ★★
- `hex_env._blue_wins` / `_red_wins` (`hex_env.py:85-123`) tournent en Python
  pur avec `collections.deque`. Appelée dans **chaque** `is_terminal()` (donc à
  chaque nœud MCTS et chaque coup AlphaBeta), c'est un goulot direct.
- Adopter la même approche que `_shortest_path_jit` (Numba) ou, mieux, un
  **union-find incrémental** mis à jour dans `apply_move`/`undo_move` : O(α(n))
  par coup au lieu d'un BFS complet à chaque test.

### 2.2. Virtual loss : remontée O(d) par feuille ★★
- `mcts_az._apply_virtual_loss` / `_undo_virtual_loss` (`mcts_az.py:148-164`)
  remontent à la racine pour **chaque** feuille puis l'annulent juste après.
  Pour un batch de 32 feuilles à profondeur 20, c'est 1 280 mises à jour de
  nœuds qui pourraient être évitées en mémorisant le chemin renvoyé par
  `_select_leaf` (liste `path`) et en n'appliquant la VL qu'une seule fois le
  long de ce chemin.

### 2.3. Hash Zobrist incrémental côté AlphaBeta ★
- `_compute_hash` (`alphabeta.py:51-62`) parcourt les 121 cases à chaque
  `select_move`. Maintenir le hash incrémentalement dans `HexEnv` (mis à jour
  par `apply_move`/`undo_move`), ce qui évite aussi de tout recalculer entre
  deux coups successifs.

### 2.4. Réutilisation de la table de transposition entre parties ★
- `AlphaBetaPlayer._tt` est persistante au sein d'une partie mais jamais
  bornée : sur de longs tournois elle peut consommer beaucoup de RAM. Ajouter
  un **plafond LRU** (ou la vider à chaque nouvelle partie) ; pour Hex, la
  réutilisation inter-parties n'apporte presque rien car les positions ne se
  recroisent pas.

### 2.5. `torch.compile` sur `HexNet` ★
- Sur GPU récents, `torch.compile(net, mode="reduce-overhead")` apporte
  typiquement 1.3–1.7× sur les forwards batchés de petit modèle. À tenter dans
  `trainer.py` et dans le chargement d'inférence.

### 2.6. Migration FP16 → BF16 sur GPU compatible ★
- `network.py:109` et `:146` utilisent `torch.amp.autocast(...)`. BF16
  (Ampère/RDNA3+) supprime les soucis d'underflow des heads `tanh`/`log_softmax`
  et permet en général de retirer le `GradScaler` côté entraînement (pas encore
  utilisé ici, mais à anticiper).

### 2.7. Pré-allocation des tenseurs et `pin_memory` côté `batch_predict` ★★
- `network.batch_predict` appelle `torch.from_numpy(states).to(device)` à chaque
  inférence. Allouer une fois un buffer GPU `pinned` réutilisé en
  `copy_(non_blocking=True)` réduit la latence par batch — utile car
  `_simulate_batch` fait beaucoup de petits batchs (32).

### 2.8. Mutualiser `predict()` et `batch_predict()` ★
- `network.predict` (état unique) est désormais redondant : tous les chemins
  chauds passent par `batch_predict`. Le supprimer ou le réécrire comme
  `batch_predict(states[None], masks[None])[0]`, pour éviter deux chemins
  d'inférence à maintenir.

---

## 3. AlphaZero — qualité de l'apprentissage

### 3.1. Sauvegarde de la distribution MCTS sans température ★
- Conséquence directe du bug 1.1 : la tête politique apprend bien plus vite si
  on lui donne la **vraie** distribution de visites (normalisée), pas un
  one-hot. Vérifier après correction que la `loss_policy` descend plus
  rapidement et que le win-rate vs random monte plus vite.

### 3.2. Symétries supplémentaires pour l'augmentation ★
- Hex possède **deux** symétries indépendantes : la diagonale principale
  (déjà implémentée dans `_augment`) **et** la rotation 180° (qui préserve les
  rôles). Ajouter cette seconde transformation double encore le buffer
  effectif sans coût de self-play.

### 3.3. EMA des poids pour l'évaluation et l'inférence ★★
- Conserver une moyenne exponentielle des poids (`decay≈0.999`) et l'utiliser
  pour la phase d'évaluation et pour le snapshot `best_model.pt`. Stabilise
  fortement les comparaisons et réduit la variance des décisions d'acceptation.

### 3.4. Évaluation Elo plutôt que win-rate seuil 55 % ★★
- `WIN_RATE_THRESHOLD = 0.55` (`config.py:36`) est binaire et bruité. Maintenir
  une échelle Elo glissante en faisant jouer chaque candidat contre un **pool**
  des derniers acceptés (au moins 3-5), comme dans KataGo / LCZero. Le seuil
  d'acceptation devient « +20 Elo significatif » et l'on évite les cycles.

### 3.5. Réintroduire un FPU (First-Play Urgency) ★
- Dans `_select_child` (`mcts_az.py:92-107`), `q = -child.Q if child.N > 0 else
  0.0`. La valeur 0 pour un enfant non visité est connue pour sur-explorer
  (notamment en début de partie). Implémenter le FPU « parent − fpu_reduction »
  (~0.2) comme dans Leela / KataZero apporte plusieurs % de force MCTS.

### 3.6. Politique adaptive `c_puct` ★
- Formule originale d'AZ Go : `c_puct(s) = log((1 + N(s) + c_base) / c_base) +
  c_init`. Aujourd'hui le code utilise une constante (`C_PUCT = 1.0`).
  L'adapter coûte deux additions par sélection et améliore l'exploration en
  fin d'arbre quand `N` croît.

### 3.7. Resign automatique en self-play ★★
- Quand `value` au nœud racine reste très négative (< −0.9) sur plusieurs
  coups, abandonner et propager `z = −1`. Économie de ~30 % de coups joués
  pendant le self-play (chiffre AlphaZero), à condition de garder un petit
  pourcentage de parties « no-resign » pour calibrer le seuil.

### 3.8. Buffer pondéré par âge ou par `|TD-error|` ★★★
- Actuellement `ReplayBuffer.sample` tire uniformément. Pondérer par récence
  (les positions générées par l'itération courante valent plus) est un gain
  classique. PER (Prioritized Experience Replay) sur l'erreur de valeur est
  une étape suivante.

### 3.9. Sauvegarde du `replay_buffer` en `mmap` / format colonne ★
- `np.savez_compressed` puis `np.load` recharge tout en RAM au démarrage : long
  pour 150 k positions × (3, 11, 11) × float32. Passer en `np.memmap` ou en
  `safetensors` accélère le warm-start et limite la fragmentation mémoire.

### 3.10. Logs métriques structurés (TensorBoard ou W&B) ★
- Aujourd'hui tout passe par `print` : impossible de comparer plusieurs runs ou
  de tracer la courbe `loss_value` / `win_rate` / `Elo`. Ajouter
  `torch.utils.tensorboard.SummaryWriter` (zéro dépendance externe) ou
  `wandb` (optionnel via flag CLI).

### 3.11. Reproductibilité ★
- Aucune seed fixe explicite côté `trainer.py` (on en voit une dans
  `alphabeta.py` pour Zobrist, c'est tout). Ajouter un `--seed` qui propage à
  `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.

---

## 4. Architecture du réseau

### 4.1. Squeeze-and-Excitation dans les blocs résiduels ★★
- Ajouter un module SE (Hu et al.) dans `ResBlock` (`network.py:10-24`) coûte
  ~2 % de paramètres et apporte régulièrement +30 à +50 Elo en jeux de
  plateau, prouvé par KataGo.

### 4.2. Tête politique en 1×1 directe ★
- Tête actuelle : Conv 1×1 → 2 canaux → BN → FC(242→121). KataGo et LCZero
  utilisent une simple Conv 1×1 → 1 canal → flatten (121 → 121, pas de FC).
  Moins de paramètres et aucun goulot 242→121 douteux.

### 4.3. Tête valeur avec global average pooling ★
- Remplacer le `flatten` 121 → 256 → 1 par GAP → FC(C → 1). Plus stable
  numériquement et invariant à la translation des features.

### 4.4. Activations modernes ★
- ReLU partout aujourd'hui ; tester GELU ou SiLU/Swish. Gains marginaux mais
  presque gratuits (changement d'une ligne).

### 4.5. Channels-last memory format ★
- Sur GPU, `net = net.to(memory_format=torch.channels_last)` accélère
  significativement les Conv2d en BF16/FP16.

### 4.6. Initialisation et Norm ★★
- BatchNorm est fragile pour l'inférence single-state (`predict`) : la moyenne
  glissante apprise dépend du régime self-play. Tester GroupNorm ou simplement
  ne plus jamais appeler `predict` (cf. 2.8).

---

## 5. Spécifique Hex — gameplay et heuristiques

### 5.1. Règle du swap (pie rule) ★★
- Hex tournoi se joue avec swap : Blue ouvre, Red peut « voler » l'ouverture.
  Aucune trace dans le code. À implémenter dans `HexEnv` (un coup spécial) et à
  enseigner aux agents (sinon Blue ouvre toujours au centre, exploitable).

### 5.2. Connexions virtuelles / ponts ★★★
- Le **bridge** (deux pions à distance 2 sur les bonnes diagonales) est la
  pierre angulaire des heuristiques Hex. Détecter les ponts permet :
  - de raccourcir l'évaluation BFS d'AlphaBeta (cellules connectées
    « virtuellement » même si vides) ;
  - d'élaguer fortement le MCTS en évitant les coups qui détruisent un pont.
  Voir Henderson & Hayward, ou la base de templates de MoHex.

### 5.3. Détection de cellules mortes ★★
- Une cellule est *morte* si la jouer ne change ni le gagnant ni l'évaluation
  optimale. Les couper du plateau réduit drastiquement le branching factor en
  fin de partie.

### 5.4. Ouvertures fixes pour l'évaluation ★
- Dans `evaluate_models`, toutes les parties partent du plateau vide : forte
  variance et chaque match revisite les mêmes situations. Tirer un set de ~20
  ouvertures fortes, jouer chaque ouverture deux fois (couleurs inversées),
  réduit énormément la variance et permet des comparaisons sur 40 parties au
  lieu de 120.

### 5.5. Tablebase de fin de partie ★★★
- Pour ≤ 8 cases vides, une recherche exhaustive est faisable et donne le
  résultat exact. Brancher cette table dans `is_terminal` court-circuite MCTS
  et AlphaBeta sur la phase finale.

---

## 6. Tournois & ranking

### 6.1. Elo bayésien plutôt que % de victoires ★★
- Implémenter un BayesElo / Glicko-2 sur `ranking.py` donne un score avec
  intervalle de confiance, indispensable quand certains matchups n'ont que
  10-20 parties.

### 6.2. Persistance JSON en plus du CSV ★
- Le CSV incrémental est utile mais peu structuré. Sortir aussi un `ranking.json`
  (avec timestamps, hyperparamètres MCTS de chaque agent, version git) facilite
  l'analyse externe.

### 6.3. Multi-time-control ★★
- Aujourd'hui un seul time budget par run. Faire tourner chaque matchup à 0.5 s,
  1 s, 2 s donnerait une **courbe** force-vs-temps, beaucoup plus informative
  qu'un classement unique.

---

## 7. Qualité de code & infrastructure

### 7.1. Packaging propre ★★
- Boilerplate `sys.path.insert` dans presque chaque fichier (`trainer.py:11-14`,
  `mcts_az.py:11-14`, `evaluate.py:9-13`, …). Convertir `alphazero/` en vrai
  package (`__init__.py`, `pyproject.toml` avec `pip install -e .`) supprime
  tout ce code et permet `from alphazero.train.network import HexNet`.

### 7.2. Suite de tests ★★
- Aucun dossier `tests/`. Quelques tests unitaires haute valeur :
  - terminaisons triviales (`is_terminal`),
  - invariance de `_augment` (cf. 1.3),
  - `apply_move` ↔ `undo_move`,
  - smoke-test trainer 1 itération, 2 parties (déjà documenté dans
    `CLAUDE.md`, à automatiser).

### 7.3. CI minimaliste ★
- `.github/workflows/ci.yml` qui lance `ruff`, `pytest`, et le smoke-test
  trainer. Évite de casser silencieusement l'API entre `ia/` et `train/`.

### 7.4. Logging structuré ★
- Remplacer les `print` mêlés à `sys.stderr` par le module `logging` standard
  avec niveaux. Particulièrement utile dans `tournament.py` qui mélange
  pilotage et messages utilisateur.

### 7.5. Type-checking ★★
- Le code utilise déjà beaucoup de hints (parfois en string forward-refs comme
  `"np.ndarray"`). Importer directement `numpy as np` dans la signature et
  faire passer `mypy` en mode `--ignore-missing-imports` permettrait de
  capturer pas mal d'incohérences entre `_simulate_batch`, `dispatch_results`
  et `batch_predict`.

### 7.6. Nettoyage des joueurs « LLM » dupliqués ★★
- `alphazero/ia/` contient ~9 fichiers `claude_*`, `deepseek_*`, `gpt53_*`,
  `kimi_*`, `mimo_*`, `qwen_*`, etc. (10 à 25 ko chacun). Si ce sont des
  benchmarks ponctuels, les déplacer dans un sous-dossier `experiments/` ou
  `bench/`, pour ne pas polluer l'API publique du package.

### 7.7. Makefile / Justfile complet ★
- `make clean` existe. Ajouter `make test`, `make train-smoke`, `make rank`,
  `make lint` pour homogénéiser les commandes (la doc CLAUDE.md liste 8
  invocations différentes que l'on retape sans cesse).

### 7.8. Dockerfile + lockfile ★★
- Reproductibilité : `requirements.txt` ou `uv.lock`/`poetry.lock` figeant
  `torch`, `numba`, `numpy`. Un `Dockerfile` ROCm officiel (RX 6600 mentionné
  dans CLAUDE.md) éviterait à un nouvel utilisateur de batailler avec
  `HSA_OVERRIDE_GFX_VERSION`.

---

## 8. Outillage utilisateur

### 8.1. Visualiseur Web minimal ★★
- Petit serveur Flask/FastAPI qui sert `play.py` : on joue contre `best_model.pt`
  dans le navigateur avec affichage de la heatmap de la politique et de la
  valeur. Beaucoup plus pédagogique qu'une CLI 121 caractères.

### 8.2. Rejeu de parties ★
- Sauvegarder en SGF-like (ou JSON) chaque partie d'évaluation pour rejouer
  ultérieurement les défaites du modèle. Très utile pour debug.

### 8.3. CLI unifiée ★
- Aujourd'hui : `tournament.py`, `ranking.py`, `versus.py`, `play.py`,
  `train/trainer.py`, `train/evaluate.py`. Une commande parente unique
  (`hex train`, `hex play`, `hex rank`, `hex bench`) — par exemple via
  `typer` — réduit la friction.

---

## 9. Quick wins (≤ 1 h chacun)

1. Corriger le bug π one-hot (1.1).
2. Aligner CLAUDE.md avec `trainer.py` et `config.py` (1.2).
3. Ajouter `weights_only=True` à tous les `torch.load` (1.4).
4. Supprimer la duplication `predict` / `batch_predict` (2.8).
5. Ajouter une seed CLI à `trainer.py` (3.11).
6. Remplacer la tête politique par Conv 1×1 → 1 canal (4.2).
7. Sortir un `ranking.json` à côté du CSV (6.2).
8. Ajouter `make test` et un `tests/test_hex_env.py` minimal (7.2).

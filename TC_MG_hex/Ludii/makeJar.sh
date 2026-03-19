rm -f TC_MG_alphabeta_player.jar TC_MG_montecarlo.jar TC_MG_mcts.jar
rm -f TC_MG_alphabeta_hex.jar TC_MG_mcts_hex.jar
rm -rf bk

# ── Joueurs Hex 11x11 ────────────────────────────────────────────────────────
javac -cp Ludii-1.3.14.jar -d . TC_MG_alphabeta_hex.java
jar cf TC_MG_alphabeta_hex.jar bk/TC_MG_alphabeta_hex.class

javac -cp Ludii-1.3.14.jar -d . TC_MG_mcts_hex.java
jar cf TC_MG_mcts_hex.jar bk/TC_MG_mcts_hex.class

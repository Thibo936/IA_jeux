CC=g++
CFLAGS=-std=c++11 -Wall -O3

.PHONY: all clean hex breakthrough

# ── Cibles par défaut (Hex) ──────────────────────────────────────────────────
all: clean hex

hex: TC_MG_alphabeta_hex TC_MG_mcts_hex TC_MG_random_hex tournament_hex

breakthrough: TC_MG_alphabeta_player TC_MG_montecarlo TC_MG_mcts tournament benchmark_mcts

# ── Jeu de Hex 11x11 ─────────────────────────────────────────────────────────
TC_MG_alphabeta_hex: hexbb.h alphabeta_hex.cpp
	$(CC) $(CFLAGS) alphabeta_hex.cpp -o $@

TC_MG_mcts_hex: hexbb.h mcts_hex.cpp
	$(CC) $(CFLAGS) mcts_hex.cpp -o $@ -lpthread

TC_MG_random_hex: hexbb.h random_hex.cpp
	$(CC) $(CFLAGS) random_hex.cpp -o $@

tournament_hex: hexbb.h tournament.cpp
	$(CC) $(CFLAGS) tournament.cpp -o $@

# ── Breakthrough 8x8 (conservés) ─────────────────────────────────────────────
TC_MG_alphabeta_player: bkbb64.h alphabeta_bk.cpp
	$(CC) $(CFLAGS) alphabeta_bk.cpp -o $@

TC_MG_montecarlo: bkbb64.h montecarlo_bk.cpp
	$(CC) $(CFLAGS) montecarlo_bk.cpp -o $@

TC_MG_mcts: bkbb64.h mcts_bk.cpp
	$(CC) $(CFLAGS) mcts_bk.cpp -o $@

tournament: bkbb64.h tournament.cpp
	$(CC) $(CFLAGS) tournament.cpp -o $@

benchmark_mcts: bkbb64.h benchmark_mcts.cpp
	$(CC) $(CFLAGS) benchmark_mcts.cpp -o $@

clean:
	rm -f TC_MG_alphabeta_hex TC_MG_mcts_hex TC_MG_random_hex tournament_hex
	rm -f TC_MG_alphabeta_player TC_MG_montecarlo TC_MG_mcts tournament tournament_ab_mcts benchmark_mcts_test benchmark_mcts
	rm -f ./Ludii/TC_MG_alphabeta_hex.jar ./Ludii/TC_MG_mcts_hex.jar
	rm -f ./Ludii/TC_MG_alphabeta_player.jar ./Ludii/TC_MG_montecarlo.jar ./Ludii/TC_MG_mcts.jar
	rm -f ./Ludii/bk/*.class

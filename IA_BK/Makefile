CC=g++
CFLAGS=-std=c++11 -Wall -O3

.PHONY: all clean

all: TC_MG_alphabeta_player TC_MG_montecarlo TC_MG_mcts \
     TC_MG_random TC_MG_uct TC_MG_mast TC_MG_rave \
     tournament big_tournament benchmark_mcts

# ── Modèles originaux (non modifiés) ─────────────────────────────────────────

TC_MG_alphabeta_player: bkbb64.h alphabeta_bk.cpp
	$(CC) $(CFLAGS) alphabeta_bk.cpp -o $@

TC_MG_mcts: bkbb64.h mcts_bk.cpp
	$(CC) $(CFLAGS) mcts_bk.cpp -o $@ -lpthread

# ── Modèles améliorés / nouveaux ─────────────────────────────────────────────

TC_MG_montecarlo: bkbb64.h montecarlo_bk.cpp
	$(CC) $(CFLAGS) montecarlo_bk.cpp -o $@ -lpthread

TC_MG_random: bkbb64.h random_bk.cpp
	$(CC) $(CFLAGS) random_bk.cpp -o $@

TC_MG_uct: bkbb64.h uct_bk.cpp
	$(CC) $(CFLAGS) uct_bk.cpp -o $@ -lpthread

TC_MG_mast: bkbb64.h mast_bk.cpp
	$(CC) $(CFLAGS) mast_bk.cpp -o $@ -lpthread

TC_MG_rave: bkbb64.h rave_bk.cpp
	$(CC) $(CFLAGS) rave_bk.cpp -o $@ -lpthread

# ── Outils ───────────────────────────────────────────────────────────────────

tournament: bkbb64.h tournament.cpp
	$(CC) $(CFLAGS) tournament.cpp -o $@

big_tournament: bkbb64.h big_tournament.cpp
	$(CC) $(CFLAGS) big_tournament.cpp -o TC_MG_big_tournament -lpthread

benchmark_mcts: bkbb64.h benchmark_mcts.cpp
	$(CC) $(CFLAGS) benchmark_mcts.cpp -o $@

clean:
	rm -f TC_MG_alphabeta_player TC_MG_montecarlo TC_MG_mcts \
	      TC_MG_random TC_MG_uct TC_MG_mast TC_MG_rave \
	      TC_MG_big_tournament \
	      tournament tournament_ab_mcts benchmark_mcts_test benchmark_mcts \
	      ./Ludii/TC_MG_alphabeta_player.jar ./Ludii/TC_MG_montecarlo.jar ./Ludii/TC_MG_mcts.jar
	rm -f ./Ludii/bk/*.class

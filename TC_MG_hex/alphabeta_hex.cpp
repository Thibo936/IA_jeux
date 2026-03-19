// alphabeta_hex.cpp — joueur Alpha-Beta pour le jeu de Hex 11x11
// Usage : ./TC_MG_alphabeta_hex BOARD PLAYER
//   BOARD  : 121 chars, gauche→droite, haut→bas : '.' vide, 'O' Blue, '@' Red
//   PLAYER : 'O' (Blue, Nord-Sud) ou '@' (Red, Ouest-Est)
// Sortie stdout : "A1" (notation colonne+ligne, ex: "F6")

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <string>
#include "hexbb.h"

#define BLUE_PLAYER 0
#define RED_PLAYER  1
#define MAX_DEPTH   4   // Hex a un branching factor très élevé (~60-80 coups)
                        // profondeur 4 reste raisonnable

const int SCORE_WIN  =  100000;
const int SCORE_LOSE = -100000;

static int g_nodes = 0;

// ─── Alpha-Beta ──────────────────────────────────────────────────────────────
// Retourne un score du point de vue du joueur courant (positif = bien)

static int alphabeta(HexBoard board, int depth, int alpha, int beta, bool isBlue) {
    g_nodes++;

    if (board.blue_win()) return isBlue ?  SCORE_WIN + depth : SCORE_LOSE - depth;
    if (board.red_win())  return isBlue ? SCORE_LOSE - depth : SCORE_WIN + depth;
    if (depth == 0)       return board.eval(isBlue);

    std::vector<HexMove> moves = board.get_legal_moves();
    if (moves.empty()) return board.eval(isBlue);

    int best = SCORE_LOSE;
    for (const auto& m : moves) {
        HexBoard next = board;
        next.apply_move(m, isBlue);
        // L'adversaire joue, on inverse le score
        int val = -alphabeta(next, depth - 1, -beta, -alpha, !isBlue);
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break; // coupure
    }
    return best;
}

// ─── Choix du meilleur coup ──────────────────────────────────────────────────
static std::string get_best_move(HexBoard& board, bool isBlue) {
    std::vector<HexMove> moves = board.get_legal_moves();
    if (moves.empty()) return "resign";

    // Vérifier coup gagnant immédiat
    for (const auto& m : moves) {
        HexBoard next = board;
        next.apply_move(m, isBlue);
        if (next.is_win(isBlue))
            return m.to_str();
    }

    g_nodes = 0;
    HexMove bestMove = moves[0];
    int bestValue = SCORE_LOSE;

    for (const auto& m : moves) {
        HexBoard next = board;
        next.apply_move(m, isBlue);
        int val = -alphabeta(next, MAX_DEPTH - 1, SCORE_LOSE, SCORE_WIN, !isBlue);
        if (val > bestValue) {
            bestValue = val;
            bestMove = m;
        }
    }

    fprintf(stderr, "SCORE:%d NODES:%d DEPTH:%d\n", bestValue, g_nodes, MAX_DEPTH);
    return bestMove.to_str();
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]);
        fprintf(stderr, "  BOARD  : 121 chars ('.' 'O' '@')\n");
        fprintf(stderr, "  PLAYER : 'O' (Blue) ou '@' (Red)\n");
        fprintf(stderr, "  time_s : ignoré (profondeur fixe), accepté pour compatibilité\n");
        return 1;
    }

    HexBoard board(argv[1]);
    std::string playerStr(argv[2]);

    bool isBlue = (playerStr == "O");
    printf("%s\n", get_best_move(board, isBlue).c_str());
    return 0;
}

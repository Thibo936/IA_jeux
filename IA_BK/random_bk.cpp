// random_bk.cpp - Joueur aléatoire pur pour Breakthrough
// Stratégie : choisit un coup légal uniformément au hasard
// Utile comme baseline de référence (niveau 0)

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <vector>
#include "bkbb64.h"

#define WHITE 0
#define BLACK 1

static std::vector<Move64_t> get_legal_moves(const Board64_t& board, bool isWhite) {
    uint64_t l = isWhite ? board.white_left() : board.black_left();
    uint64_t f = isWhite ? board.white_forward() : board.black_forward();
    uint64_t r = isWhite ? board.white_right() : board.black_right();
    return Lfr_t(l, f, r).get_moves(isWhite);
}

std::string get_best_move(Board64_t& board, int color) {
    bool isWhite = (color == WHITE);

    std::vector<Move64_t> moves = get_legal_moves(board, isWhite);
    if (moves.empty()) return "resign";

    // Coup aléatoire uniforme
    int idx = rand() % (int)moves.size();
    Move64_t chosen = moves[idx];

    fprintf(stderr, "RANDOM: %zu moves available, picked #%d\n", moves.size(), idx);

    return pos_to_coord(chosen.pi) + "-" + pos_to_coord(chosen.pf);
}

int main(int argc, char** argv) {
    srand(time(nullptr));
    if (argc < 3) {
        fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]);
        return 0;
    }

    Board64_t B(argv[1]);
    std::string playerStr(argv[2]);

    if (playerStr == "O") {
        printf("%s\n", get_best_move(B, WHITE).c_str());
    } else if (playerStr == "@") {
        printf("%s\n", get_best_move(B, BLACK).c_str());
    }

    return 0;
}

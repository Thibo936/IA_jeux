#include <cstdlib>   
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include "bkbb64.h"

#define WHITE 0
#define BLACK 1
#define MAX_DEPTH 6

const int SCORE_WIN = 100000;
const int SCORE_LOSE = -100000;
const int MATERIAL_WEIGHT = 10;
const int ADVANCE_WEIGHT = 5;
const int SAFE_WEIGHT = 3;

int evaluate_board(const Board64_t& board) {
    if (board.white_win()) return SCORE_WIN;
    if (board.black_win()) return SCORE_LOSE;

    int score = 0;

    uint64_t w = board.white;
    while (w) {
        int idx = __builtin_ctzll(w);
        int row = idx / 8;

        score += MATERIAL_WEIGHT;
        score += (7 - row) * ADVANCE_WEIGHT;
        if (row == 1) score += 50;

        w &= (w - 1);
    }

    uint64_t b = board.black;
    while (b) {
        int idx = __builtin_ctzll(b);
        int row = idx / 8;

        score -= MATERIAL_WEIGHT;
        score -= row * ADVANCE_WEIGHT;
        if (row == 6) score -= 50;

        b &= (b - 1);
    }

    uint64_t w_protected_mask = (board.white << 7) | (board.white << 9);
    uint64_t w_safe = board.white & w_protected_mask;
    score += __builtin_popcountll(w_safe) * SAFE_WEIGHT;

    uint64_t b_protected_mask = (board.black >> 7) | (board.black >> 9);
    uint64_t b_safe = board.black & b_protected_mask;
    score -= __builtin_popcountll(b_safe) * SAFE_WEIGHT;

    return score;
}

static int g_nodes = 0;

int alphabeta(Board64_t& board, int depth, int alpha, int beta, bool isWhite) {
    g_nodes++;
    // Conditions d'arrêt
    if (board.white_win()) return SCORE_WIN + depth; // Favorise les victoires rapides
    if (board.black_win()) return SCORE_LOSE - depth; // Favorise les défaites tardives
    if (depth == 0) {
        return evaluate_board(board);
    }

    if (isWhite) {
        int maxEval = INT_MIN;
        uint64_t wl = board.white_left();
        uint64_t wf = board.white_forward();
        uint64_t wr = board.white_right();
        std::vector<Move64_t> moves = Lfr_t(wl, wf, wr).get_white_moves();
        
        if (moves.empty()) return evaluate_board(board);

        for (const auto& m : moves) {
            Board64_t nextBoard = board;
            nextBoard.apply_white_move(m);
            int score = alphabeta(nextBoard, depth - 1, alpha, beta, false);
            maxEval = std::max(maxEval, score);
            alpha = std::max(alpha, score);
            if (beta <= alpha) break; // Coupure Beta
        }
        return maxEval;
    } else {
        int minEval = INT_MAX;
        uint64_t bl = board.black_left();
        uint64_t bf = board.black_forward();
        uint64_t br = board.black_right();
        std::vector<Move64_t> moves = Lfr_t(bl, bf, br).get_black_moves();

        if (moves.empty()) return evaluate_board(board);

        for (const auto& m : moves) {
            Board64_t nextBoard = board;
            nextBoard.apply_black_move(m);
            int score = alphabeta(nextBoard, depth - 1, alpha, beta, true);
            minEval = std::min(minEval, score);
            beta = std::min(beta, score);
            if (beta <= alpha) break; // Coupure Alpha
        }
        return minEval;
    }
}

std::string get_best_move(Board64_t& board, int color) {
    bool isWhite = (color == WHITE);
    Move64_t bestMove;
    int bestValue = isWhite ? INT_MIN : INT_MAX;

    uint64_t l = isWhite ? board.white_left() : board.black_left();
    uint64_t f = isWhite ? board.white_forward() : board.black_forward();
    uint64_t r = isWhite ? board.white_right() : board.black_right();
    std::vector<Move64_t> moves = Lfr_t(l, f, r).get_moves(isWhite);

    if (moves.empty()) return "resign";
    bestMove = moves[0]; // Initialisation pour éviter le warning compiler

    g_nodes = 0;
    for (const auto& m : moves) {
        Board64_t nextBoard = board;
        nextBoard.apply_move(m, isWhite);
        int boardValue = alphabeta(nextBoard, MAX_DEPTH - 1, INT_MIN, INT_MAX, !isWhite);

        if (isWhite) {
            if (boardValue > bestValue) {
                bestValue = boardValue;
                bestMove = m;
            }
        } else {
            if (boardValue < bestValue) {
                bestValue = boardValue;
                bestMove = m;
            }
        }
    }

    fprintf(stderr, "SCORE:%d NODES:%d DEPTH:%d\n", bestValue, g_nodes, MAX_DEPTH);

    std::string stri = pos_to_coord(bestMove.pi);
    std::string strf = pos_to_coord(bestMove.pf);
    return stri + "-" + strf;
}

int main(int argc, char** argv) {
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

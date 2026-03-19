#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include "bkbb64.h"

#define WHITE 0
#define BLACK 1
#define MAX_MOVES 200
#define GAMES_PER_COLOR 10
#define ITER_START   50000
#define ITER_END   1500000
#define ITER_STEP    50000

static const char* MCTS_EXE = "./TC_MG_mcts";
static const char* AB_EXE   = "./TC_MG_alphabeta_player";

// ── helpers identiques à tournament.cpp ────────────────────────────────────

static std::string board_to_str(const Board64_t& board) {
    std::string s(64, '.');
    for (int i = 0; i < 64; i++) {
        uint64_t bit = (1ULL << i);
        if (board.black & bit)      s[i] = '@';
        else if (board.white & bit) s[i] = 'O';
    }
    return s;
}

static uint64_t coord_to_pos(const std::string& coord) {
    if (coord.size() < 2) return 0ULL;
    int col = coord[0] - 'A';
    int row = 8 - (coord[1] - '0');
    return (1ULL << (row * 8 + col));
}

static bool apply_move_str(Board64_t& board, const std::string& move, int color) {
    if (move == "resign" || move.size() < 5) return false;
    Move64_t m;
    m.pi = coord_to_pos(move.substr(0, 2));
    m.pf = coord_to_pos(move.substr(3, 2));
    if (m.pi == 0 || m.pf == 0) return false;
    board.apply_move(m, color == WHITE);
    return true;
}

// ── appel IA avec mesure de temps ──────────────────────────────────────────

static std::string call_ai(const std::string& exe, const Board64_t& board,
                            int color, int nb_iter, double& elapsed_sec) {
    std::string boardStr = board_to_str(board);
    std::string playerStr = (color == WHITE) ? "O" : "@";
    std::string cmd = exe + " \"" + boardStr + "\" " + playerStr;
    if (nb_iter > 0)
        cmd += " " + std::to_string(nb_iter);

    auto t0 = std::chrono::high_resolution_clock::now();
    FILE* pipe = popen(cmd.c_str(), "r");
    char buf[32];
    std::string result;
    if (pipe) {
        while (fgets(buf, sizeof(buf), pipe)) result += buf;
        pclose(pipe);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    while (!result.empty() &&
           (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
        result.pop_back();
    return result;
}

// ── une partie ; retourne 0=MCTS gagne, 1=AB gagne, -1=nul ────────────────

static int play_game(int mcts_color, int nb_iter,
                     double& mcts_total_time, int& mcts_total_moves) {
    Board64_t board;
    bool whiteTurn = true;

    for (int turn = 0; turn < MAX_MOVES; turn++) {
        if (board.white_win()) return (mcts_color == WHITE) ? 0 : 1;
        if (board.black_win()) return (mcts_color == BLACK) ? 0 : 1;

        int currentColor = whiteTurn ? WHITE : BLACK;
        bool isMctsMove  = (currentColor == mcts_color);

        double elapsed = 0.0;
        std::string move;
        if (isMctsMove) {
            move = call_ai(MCTS_EXE, board, currentColor, nb_iter, elapsed);
            mcts_total_time  += elapsed;
            mcts_total_moves += 1;
        } else {
            move = call_ai(AB_EXE, board, currentColor, -1, elapsed);
        }

        if (move == "resign" || !apply_move_str(board, move, currentColor))
            return isMctsMove ? 1 : 0;

        whiteTurn = !whiteTurn;
    }
    return -1; // match nul (partie trop longue)
}

// ── main ───────────────────────────────────────────────────────────────────

int main() {
    // En-tête horodaté
    time_t now = time(nullptr);
    char timebuf[64];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    FILE* out = fopen("benchmark_results.txt", "w");
    if (!out) { fprintf(stderr, "Impossible d'ouvrir benchmark_results.txt\n"); return 1; }

    fprintf(out, "# Benchmark MCTS vs AlphaBeta (profondeur=6) - %s\n", timebuf);
    fprintf(out, "# %d parties MCTS=blanc + %d parties MCTS=noir par niveau\n",
            GAMES_PER_COLOR, GAMES_PER_COLOR);
    fprintf(out, "# %-13s %-11s %-10s %-9s %-16s %s\n",
            "iterations", "vic_blanc", "vic_noir", "total/20",
            "temps_total_s", "temps_moy_s/coup");

    int nb_levels = (ITER_END - ITER_START) / ITER_STEP + 1;
    printf("=== Benchmark MCTS vs AlphaBeta  [%s] ===\n", timebuf);
    printf("Niveaux : %d à %d, pas %d  (%d niveaux x %d parties)\n\n",
           ITER_START, ITER_END, ITER_STEP, nb_levels, 2 * GAMES_PER_COLOR);
    fflush(stdout);

    for (int nb_iter = ITER_START; nb_iter <= ITER_END; nb_iter += ITER_STEP) {
        int    wins_white = 0, wins_black = 0;
        double total_time = 0.0;
        int    total_moves = 0;

        printf("[%7d iters]\n", nb_iter);

        // 10 parties MCTS = blanc
        for (int g = 0; g < GAMES_PER_COLOR; g++) {
            int result = play_game(WHITE, nb_iter, total_time, total_moves);
            if (result == 0) wins_white++;
            printf("  Partie %2d/20 [MCTS=blanc]: %s gagne\n",
                   g + 1, result == 0 ? "MCTS" : "AlphaBeta");
            fflush(stdout);
        }

        // 10 parties MCTS = noir
        for (int g = 0; g < GAMES_PER_COLOR; g++) {
            int result = play_game(BLACK, nb_iter, total_time, total_moves);
            if (result == 0) wins_black++;
            printf("  Partie %2d/20 [MCTS=noir ]: %s gagne\n",
                   g + GAMES_PER_COLOR + 1, result == 0 ? "MCTS" : "AlphaBeta");
            fflush(stdout);
        }

        double avg = (total_moves > 0) ? total_time / total_moves : 0.0;

        printf("  ==> BILAN: blanc=%2d/10  noir=%2d/10  total=%2d/20  temps=%.1fs  moy=%.3fs/coup\n\n",
               wins_white, wins_black, wins_white + wins_black, total_time, avg);

        fprintf(out, "  %-13d %-11d %-10d %-9d %-16.2f %.4f\n",
                nb_iter, wins_white, wins_black, wins_white + wins_black,
                total_time, avg);
        fflush(out);
    }

    fprintf(out, "# Fin du benchmark\n");
    fclose(out);
    printf("\nResultats enregistres dans benchmark_results.txt\n");
    return 0;
}

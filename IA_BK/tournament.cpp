#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include "bkbb64.h"

#define WHITE 0
#define BLACK 1
#define MAX_MOVES 200  // sécurité anti-boucle infinie

// Affiche le plateau en ASCII avec infos
void print_board_verbose(const Board64_t& board, const std::string& lastMove, int turnNum,
                          const std::string& aiName, const std::string& color,
                          int captured, const std::string& ai_stats, bool isWhiteTurn) {
    int w_count = __builtin_popcountll(board.white);
    int b_count = __builtin_popcountll(board.black);

    printf("\n  Coup #%d [%s] %s joue: %s\n", turnNum, color.c_str(), aiName.c_str(), lastMove.c_str());
    printf("  +--------+\n");
    for (int row = 0; row < 8; row++) {
        printf("%d |", 8 - row);
        for (int col = 0; col < 8; col++) {
            int idx = row * 8 + col;
            uint64_t bit = (1ULL << idx);
            if (board.black & bit)       printf("@");
            else if (board.white & bit)  printf("O");
            else                         printf(".");
        }
        printf("|\n");
    }
    printf("  +--------+\n");
    printf("   ABCDEFGH\n");
    if (captured > 0)
        printf("  O(blanc):%d  @(noir):%d  >>> CAPTURE\n", w_count, b_count);
    else
        printf("  O(blanc):%d  @(noir):%d\n", w_count, b_count);

    // Score heuristique depuis la perspective des blancs (positif = blanc gagne)
    int eval = board.eval(true);
    printf("  Eval board: %+d", eval);

    // Stats IA
    if (!ai_stats.empty()) {
        int score, nodes, depth;
        if (sscanf(ai_stats.c_str(), "SCORE:%d NODES:%d DEPTH:%d", &score, &nodes, &depth) == 3) {
            printf("  |  AlphaBeta: score=%d, %dk noeuds, profondeur %d", score, nodes / 1000, depth);
        } else {
            int iters, visits;
            float winrate, time_s;
            if (sscanf(ai_stats.c_str(), "ITERS:%d VISITS:%d WINRATE:%f TIME:%f",
                       &iters, &visits, &winrate, &time_s) == 4) {
                float wr_pct = winrate * 100.0f;
                int filled = (int)(wr_pct / 10.0f);
                if (filled > 10) filled = 10;
                char bar[11];
                for (int i = 0; i < 10; i++) bar[i] = (i < filled) ? '#' : '.';
                bar[10] = '\0';
                const char* label;
                if      (wr_pct <  10.0f) label = "CRITIQUE";
                else if (wr_pct <  40.0f) label = "DIFFICILE";
                else if (wr_pct <  75.0f) label = "EQUILIBRE";
                else                      label = "DOMINANT";
                printf("  |  MCTS: %d iters, meilleur coup visite %dx (winrate: %.1f%% [%s] %s), temps: %.2fs",
                       iters, visits, wr_pct, bar, label, time_s);
            } else {
                printf("  |  %s", ai_stats.c_str());
            }
        }
    }
    printf("\n\n");
}

// Sérialise le board en chaîne de 64 caractères pour les IA
std::string board_to_str(const Board64_t& board) {
    std::string s(64, '.');
    for (int i = 0; i < 64; i++) {
        uint64_t bit = (1ULL << i);
        if (board.black & bit) s[i] = '@';
        else if (board.white & bit) s[i] = 'O';
    }
    return s;
}

// Convertit "A8" -> position bitboard
uint64_t coord_to_pos(const std::string& coord) {
    if (coord.size() < 2) return 0ULL;
    int col = coord[0] - 'A';
    int row = 8 - (coord[1] - '0');
    return (1ULL << (row * 8 + col));
}

// Appelle l'exécutable IA et retourne (coup, stats_stderr)
std::pair<std::string, std::string> call_ai(const std::string& exe, const Board64_t& board, int color) {
    std::string boardStr = board_to_str(board);
    std::string playerStr = (color == WHITE) ? "O" : "@";
    std::string tmpFile = "/tmp/tc_mg_" + std::to_string(getpid()) + "_stderr";
    std::string cmd = exe + " \"" + boardStr + "\" " + playerStr + " 2>" + tmpFile;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {"resign", ""};

    char buf[32];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe)) {
        result += buf;
    }
    pclose(pipe);

    // Lire le stderr capturé
    std::string stats;
    FILE* sf = fopen(tmpFile.c_str(), "r");
    if (sf) {
        char sbuf[512];
        while (fgets(sbuf, sizeof(sbuf), sf)) {
            stats += sbuf;
        }
        fclose(sf);
        remove(tmpFile.c_str());
    }

    // Nettoyer les espaces/newlines
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
        result.pop_back();
    while (!stats.empty() && (stats.back() == '\n' || stats.back() == '\r'))
        stats.pop_back();

    return {result, stats};
}

// Applique un coup "XX-YY" au board, retourne false si coup invalide
bool apply_move_str(Board64_t& board, const std::string& move, int color) {
    if (move == "resign") return false;
    if (move.size() < 5) return false;

    std::string from = move.substr(0, 2);
    std::string to   = move.substr(3, 2);

    Move64_t m;
    m.pi = coord_to_pos(from);
    m.pf = coord_to_pos(to);

    if (m.pi == 0 || m.pf == 0) return false;

    board.apply_move(m, color == WHITE);
    return true;
}

// Joue une partie entre ai1 (color1) et ai2 (color2)
// Retourne 0 si ai1 gagne, 1 si ai2 gagne, -1 si null/erreur
int play_game(const std::string& ai1, int color1,
              const std::string& ai2, int color2, bool verbose) {
    Board64_t board;
    bool whiteTurn = true;  // blanc commence toujours

    for (int turn = 0; turn < MAX_MOVES; turn++) {
        if (board.white_win()) return (color1 == WHITE) ? 0 : 1;
        if (board.black_win()) return (color1 == BLACK) ? 0 : 1;

        int currentColor = whiteTurn ? WHITE : BLACK;
        std::string* currentAI;
        int aiIdx;

        if (currentColor == color1) {
            currentAI = const_cast<std::string*>(&ai1);
            aiIdx = 0;
        } else {
            currentAI = const_cast<std::string*>(&ai2);
            aiIdx = 1;
        }

        int w_before = __builtin_popcountll(board.white);
        int b_before = __builtin_popcountll(board.black);

        auto ai_result = call_ai(*currentAI, board, currentColor);
        std::string move = ai_result.first;
        std::string ai_stats = ai_result.second;

        if (move == "resign" || !apply_move_str(board, move, currentColor)) {
            if (verbose)
                printf("  [%s] %s -> ABANDON\n", whiteTurn ? "W" : "B", currentAI->c_str());
            // L'IA qui abandonne perd
            return (aiIdx == 0) ? 1 : 0;
        }

        if (verbose) {
            int w_after = __builtin_popcountll(board.white);
            int b_after = __builtin_popcountll(board.black);
            int captured = (w_before + b_before) - (w_after + b_after);
            std::string colorStr = whiteTurn ? "Blanc" : "Noir";
            print_board_verbose(board, move, turn + 1, *currentAI, colorStr,
                                captured, ai_stats, whiteTurn);
        }

        whiteTurn = !whiteTurn;
    }
    return -1; // match nul (trop long)
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <ai1_exe> <ai2_exe> [nb_games] [-v]\n", argv[0]);
        fprintf(stderr, "  Ex: %s ./TC_MG_montecarlo ./TC_MG_mcts 100\n", argv[0]);
        return 1;
    }

    std::string ai1 = argv[1];
    std::string ai2 = argv[2];
    int nbGames = (argc >= 4) ? atoi(argv[3]) : 100;
    bool verbose = false;
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "-v") verbose = true;

    int wins1 = 0, wins2 = 0, draws = 0;
    // Stats par couleur : wins1_as_white = victoires de ai1 quand il joue blanc
    int wins1_as_white = 0, wins1_as_black = 0;
    int wins2_as_white = 0, wins2_as_black = 0;

    printf("Tournoi: %s vs %s (%d parties)\n", ai1.c_str(), ai2.c_str(), nbGames);
    printf("Les couleurs alternent à chaque partie pour un tournoi équitable.\n");
    printf("------------------------------------------------\n");

    for (int g = 0; g < nbGames; g++) {
        // Alterner les couleurs : parties paires -> ai1=blanc, impaires -> ai1=noir
        int color1 = (g % 2 == 0) ? WHITE : BLACK;
        int color2 = (color1 == WHITE) ? BLACK : WHITE;
        const char* c1str = (color1 == WHITE) ? "B" : "N";
        const char* c2str = (color2 == WHITE) ? "B" : "N";

        if (verbose)
            printf("Partie %d (%s=%s, %s=%s):\n",
                   g + 1,
                   ai1.c_str(), color1 == WHITE ? "Blanc" : "Noir",
                   ai2.c_str(), color2 == WHITE ? "Blanc" : "Noir");

        int result = play_game(ai1, color1, ai2, color2, verbose);

        if (result == 0) {
            wins1++;
            if (color1 == WHITE) wins1_as_white++; else wins1_as_black++;
            if (!verbose) printf("Partie %3d [ai1=%s ai2=%s]: %s gagne\n", g+1, c1str, c2str, ai1.c_str());
        } else if (result == 1) {
            wins2++;
            if (color2 == WHITE) wins2_as_white++; else wins2_as_black++;
            if (!verbose) printf("Partie %3d [ai1=%s ai2=%s]: %s gagne\n", g+1, c1str, c2str, ai2.c_str());
        } else {
            draws++;
            if (!verbose) printf("Partie %3d [ai1=%s ai2=%s]: nul\n", g+1, c1str, c2str);
        }

        fflush(stdout);
    }

    int gamesPerColor = nbGames / 2;
    printf("================================================\n");
    printf("Résultats finaux (%d parties, ~%d en blanc, ~%d en noir chacun):\n",
           nbGames, gamesPerColor, gamesPerColor);
    printf("  %-30s : %d victoires (%.1f%%)  [blanc: %d  noir: %d]\n",
           ai1.c_str(), wins1, 100.0 * wins1 / nbGames, wins1_as_white, wins1_as_black);
    printf("  %-30s : %d victoires (%.1f%%)  [blanc: %d  noir: %d]\n",
           ai2.c_str(), wins2, 100.0 * wins2 / nbGames, wins2_as_white, wins2_as_black);
    printf("  Nuls                           : %d\n", draws);

    return 0;
}

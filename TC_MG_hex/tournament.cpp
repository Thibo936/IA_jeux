// tournament_hex.cpp — Tournoi entre deux IA Hex 11x11
// Usage : ./tournament_hex <ai1_exe> <ai2_exe> [nb_games] [-v]
//   Ex  : ./tournament_hex ./TC_MG_alphabeta_hex ./TC_MG_mcts_hex 20

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include "hexbb.h"

#define BLUE 0   // joueur 1 (O) : connecte Nord → Sud  (lignes)
#define RED  1   // joueur 2 (@) : connecte Ouest → Est (colonnes)

// En Hex, une partie ne peut pas dépasser HEX_CELLS coups (121 placements)
#define MAX_MOVES (HEX_CELLS + 5)

// ─── Affichage ASCII du plateau avec informations ─────────────────────────────
void print_board_verbose(const HexBoard& board, const std::string& lastMove, int turnNum,
                         const std::string& aiName, const std::string& playerLabel,
                         const std::string& ai_stats) {
    int blue_count = __builtin_popcountll(board.blue_lo)
                   + __builtin_popcountll(board.blue_hi & 0x1FFFFFFFFFFFFFFULL);
    int red_count  = __builtin_popcountll(board.red_lo)
                   + __builtin_popcountll(board.red_hi  & 0x1FFFFFFFFFFFFFFULL);

    printf("\n  Coup #%d [%s] %s joue: %s\n", turnNum, playerLabel.c_str(), aiName.c_str(), lastMove.c_str());

    // En-tête colonnes
    printf("      ");
    for (int c = 0; c < HEX_SIZE; c++) printf("%c ", 'A' + c);
    printf("\n");

    for (int r = 0; r < HEX_SIZE; r++) {
        printf("  %2d  ", r + 1);
        // Décalage hexagonal
        for (int s = 0; s < r; s++) printf(" ");
        for (int c = 0; c < HEX_SIZE; c++) {
            int idx = r * HEX_SIZE + c;
            if      (bit_get(board.blue_lo, board.blue_hi, idx)) printf("O ");
            else if (bit_get(board.red_lo,  board.red_hi,  idx)) printf("@ ");
            else                                                  printf(". ");
        }
        printf("\n");
    }

    printf("  O(Blue/Nord-Sud):%d  @(Red/Ouest-Est):%d\n", blue_count, red_count);

    // Score heuristique (positif = Blue mieux)
    int eval_blue = board.eval(true);
    printf("  Eval (Blue): %+d", eval_blue);

    // Stats IA issues du stderr
    if (!ai_stats.empty()) {
        int score, nodes, depth;
        if (sscanf(ai_stats.c_str(), "SCORE:%d NODES:%d DEPTH:%d", &score, &nodes, &depth) == 3) {
            printf("  |  AlphaBeta: score=%+d, %dk noeuds, profondeur %d", score, nodes / 1000, depth);
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
                printf("  |  MCTS: %d iters, meilleur visite %dx (winrate: %.1f%% [%s] %s), temps: %.2fs",
                       iters, visits, wr_pct, bar, label, time_s);
            } else {
                printf("  |  %s", ai_stats.c_str());
            }
        }
    }
    printf("\n\n");
}

// ─── Sérialise le plateau en chaîne de 121 chars ─────────────────────────────
std::string board_to_str(const HexBoard& board) {
    std::string s(HEX_CELLS, '.');
    for (int i = 0; i < HEX_CELLS; i++) {
        if      (bit_get(board.blue_lo, board.blue_hi, i)) s[i] = 'O';
        else if (bit_get(board.red_lo,  board.red_hi,  i)) s[i] = '@';
    }
    return s;
}

// Limite de temps par coup passée aux IA (en secondes), modifiable via -t
static double g_time_per_move = 1.5;

// ─── Appelle l'exécutable IA et retourne (coup, stats_stderr) ─────────────────
// Le coup retourné est de la forme "F6" (colonne + ligne)
std::pair<std::string, std::string> call_ai(const std::string& exe,
                                             const HexBoard& board, int player) {
    std::string boardStr   = board_to_str(board);
    std::string playerStr  = (player == BLUE) ? "O" : "@";
    std::string tmpFile    = "/tmp/tc_mg_hex_" + std::to_string(getpid()) + "_stderr";
    // Préfixer avec "./" si le chemin est un nom simple (sans séparateur de répertoire)
    // pour que popen trouve l'exécutable dans le répertoire courant
    std::string exePath = exe;
    if (exe.find('/') == std::string::npos)
        exePath = "./" + exe;
    // Passer la limite de temps comme 3e argument (les IA MCTS et AB l'acceptent)
    char timeBuf[32];
    snprintf(timeBuf, sizeof(timeBuf), "%.2f", g_time_per_move);
    std::string cmd = exePath + " \"" + boardStr + "\" " + playerStr
                    + " " + timeBuf + " 2>" + tmpFile;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {"resign", ""};

    char buf[32];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe))
        result += buf;
    pclose(pipe);

    // Lire le stderr capturé
    std::string stats;
    FILE* sf = fopen(tmpFile.c_str(), "r");
    if (sf) {
        char sbuf[512];
        while (fgets(sbuf, sizeof(sbuf), sf))
            stats += sbuf;
        fclose(sf);
        remove(tmpFile.c_str());
    }

    // Supprimer trailing whitespace
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
        result.pop_back();
    while (!stats.empty() && (stats.back() == '\n' || stats.back() == '\r'))
        stats.pop_back();

    return {result, stats};
}

// ─── Applique un coup "F6" au plateau ────────────────────────────────────────
// Retourne false si le coup est invalide
bool apply_move_str(HexBoard& board, const std::string& move, int player) {
    if (move == "resign" || move.size() < 2) return false;

    int pos = hex_str_to_pos(move);
    if (pos < 0 || pos >= HEX_CELLS) return false;

    // Vérifier que la case est bien vide
    if (bit_get(board.blue_lo, board.blue_hi, pos)) return false;
    if (bit_get(board.red_lo,  board.red_hi,  pos)) return false;

    board.apply_move(HexMove(pos), player == BLUE);
    return true;
}

// ─── Joue une partie ─────────────────────────────────────────────────────────
// Retourne 0 si ai1 gagne, 1 si ai2 gagne, -1 si erreur/timeout
int play_game(const std::string& ai1, int player1,
              const std::string& ai2, int player2, bool verbose) {
    HexBoard board;
    bool blueTurn = true; // Blue (joueur 1) commence toujours

    for (int turn = 0; turn < MAX_MOVES; turn++) {
        if (board.blue_win()) return (player1 == BLUE) ? 0 : 1;
        if (board.red_win())  return (player1 == RED)  ? 0 : 1;

        int currentPlayer = blueTurn ? BLUE : RED;
        const std::string* currentAI;
        int aiIdx;

        if (currentPlayer == player1) {
            currentAI = &ai1; aiIdx = 0;
        } else {
            currentAI = &ai2; aiIdx = 1;
        }

        std::pair<std::string,std::string> ai_result = call_ai(*currentAI, board, currentPlayer);
        std::string move     = ai_result.first;
        std::string ai_stats = ai_result.second;

        if (move == "resign" || !apply_move_str(board, move, currentPlayer)) {
            if (verbose)
                printf("  [%s] %s -> ABANDON\n",
                       blueTurn ? "Blue" : "Red", currentAI->c_str());
            return (aiIdx == 0) ? 1 : 0;
        }

        if (verbose) {
            std::string playerLabel = blueTurn ? "Blue(O)" : "Red(@)";
            print_board_verbose(board, move, turn + 1, *currentAI, playerLabel, ai_stats);
        }

        blueTurn = !blueTurn;
    }
    // Ne devrait pas arriver en Hex (la partie finit forcément en ≤ 121 coups)
    return -1;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <ai1_exe> <ai2_exe> [nb_games] [-v] [-t time_s]\n", argv[0]);
        fprintf(stderr, "  Ex  : %s ./TC_MG_alphabeta_hex ./TC_MG_mcts_hex 20\n", argv[0]);
        fprintf(stderr, "  -v  : affichage verbose du plateau apres chaque coup\n");
        fprintf(stderr, "  -t  : limite de temps par coup en secondes (defaut 1.3)\n");
        fprintf(stderr, "  Blue(O) connecte Nord-Sud, Red(@) connecte Ouest-Est\n");
        return 1;
    }

    std::string ai1  = argv[1];
    std::string ai2  = argv[2];
    int nbGames      = (argc >= 4 && argv[3][0] != '-') ? atoi(argv[3]) : 20;
    bool verbose     = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v") verbose = true;
        if (std::string(argv[i]) == "-t" && i + 1 < argc) {
            g_time_per_move = atof(argv[i + 1]);
            if (g_time_per_move <= 0) g_time_per_move = 1.3;
        }
    }

    int wins1 = 0, wins2 = 0, draws = 0;
    int wins1_as_blue = 0, wins1_as_red = 0;
    int wins2_as_blue = 0, wins2_as_red = 0;

    printf("Tournoi Hex 11x11: %s vs %s (%d parties, %.2fs/coup)\n",
           ai1.c_str(), ai2.c_str(), nbGames, g_time_per_move);
    printf("Blue(O) : connecte Nord -> Sud\n");
    printf("Red (@) : connecte Ouest -> Est\n");
    printf("Les couleurs alternent à chaque partie.\n");
    printf("------------------------------------------------\n");

    for (int g = 0; g < nbGames; g++) {
        // Alterner : parties paires → ai1=Blue, impaires → ai1=Red
        int player1 = (g % 2 == 0) ? BLUE : RED;
        int player2 = (player1 == BLUE) ? RED : BLUE;
        const char* c1str = (player1 == BLUE) ? "Blue" : "Red ";
        const char* c2str = (player2 == BLUE) ? "Blue" : "Red ";

        if (verbose)
            printf("Partie %d (%s=%s, %s=%s):\n",
                   g + 1,
                   ai1.c_str(), player1 == BLUE ? "Blue" : "Red",
                   ai2.c_str(), player2 == BLUE ? "Blue" : "Red");

        int result = play_game(ai1, player1, ai2, player2, verbose);

        if (result == 0) {
            wins1++;
            if (player1 == BLUE) wins1_as_blue++; else wins1_as_red++;
            if (!verbose)
                printf("Partie %3d [ai1=%s ai2=%s]: %s gagne\n",
                       g + 1, c1str, c2str, ai1.c_str());
        } else if (result == 1) {
            wins2++;
            if (player2 == BLUE) wins2_as_blue++; else wins2_as_red++;
            if (!verbose)
                printf("Partie %3d [ai1=%s ai2=%s]: %s gagne\n",
                       g + 1, c1str, c2str, ai2.c_str());
        } else {
            draws++;
            if (!verbose)
                printf("Partie %3d [ai1=%s ai2=%s]: nul/erreur\n", g + 1, c1str, c2str);
        }

        fflush(stdout);
    }

    int gamesPerColor = nbGames / 2;
    printf("================================================\n");
    printf("Résultats finaux (%d parties, ~%d en Blue, ~%d en Red chacun):\n",
           nbGames, gamesPerColor, gamesPerColor);
    printf("  %-32s : %d victoires (%.1f%%)  [Blue: %d  Red: %d]\n",
           ai1.c_str(), wins1, 100.0 * wins1 / nbGames, wins1_as_blue, wins1_as_red);
    printf("  %-32s : %d victoires (%.1f%%)  [Blue: %d  Red: %d]\n",
           ai2.c_str(), wins2, 100.0 * wins2 / nbGames, wins2_as_blue, wins2_as_red);
    if (draws > 0)
        printf("  Nuls/erreurs                     : %d\n", draws);

    return 0;
}

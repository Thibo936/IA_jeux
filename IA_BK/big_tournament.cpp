// big_tournament.cpp - Tournoi complet multi-modèles pour Breakthrough
//
// Fonctionnement :
//   1. Détecte automatiquement tous les exécutables TC_MG_* dans ./
//      (ou prend la liste en arguments)
//   2. Détecte si un modèle est multi-threadé (utilise tous les cœurs)
//      → les modèles mono-thread peuvent jouer plusieurs matchs en parallèle
//   3. Organise une phase de POULE en round-robin, matchs joués en parallèle :
//      - Modèles mono-thread : autant de matchs simultanés que de cœurs dispo
//      - Modèle multi-thread  : réserve tous les cœurs, 1 seul match à la fois
//      Chaque match est adaptatif :
//        - Commence à 10 parties (5 en blanc / 5 en noir)
//        - Si écart ≤ 1 : prolonge par +2 jusqu'à max 19 parties
//        - Égalité parfaite à 19 → match nul
//   4. Classement final :
//      3 pts victoire match / 1 pt nul / 0 pt défaite
//      Départage : (1) différentiel W-L, (2) % victoires parties
//   5. Affichage :
//      - Progression live de chaque partie (avec mutex pour éviter mélange)
//      - Classement partiel après chaque match terminé
//      - Tableau de poule + classement final coloré
//
// Usage :
//   ./TC_MG_big_tournament                        # auto-détection TC_MG_*
//   ./TC_MG_big_tournament 1.0                    # 1s par coup
//   ./TC_MG_big_tournament 1.0 ./model1 ./model2  # liste explicite
//   ./TC_MG_big_tournament -v 1.0                 # verbose (affiche chaque coup)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <sstream>
#include <unistd.h>
#include <dirent.h>
#include "bkbb64.h"

// ─── Constantes ─────────────────────────────────────────────────────────────

#define WHITE        0
#define BLACK        1
#define MAX_MOVES    200
#define MIN_GAMES    10
#define MAX_GAMES    19
#define TIE_THRESH   1

// ─── Couleurs ANSI ──────────────────────────────────────────────────────────

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_WHITE   "\033[37m"
#define ANSI_DIM     "\033[2m"

static bool   g_verbose      = false;
static double g_time_limit   = 1.0;
static int    g_total_cores  = 1;

// Mutex global pour tous les printf (évite le mélange entre threads)
static std::mutex g_print_mtx;

// ─── Détection modèles multi-threadés ───────────────────────────────────────
// Un modèle est considéré multi-threadé s'il contient l'un de ces suffixes.
// Il utilisera tous les cœurs dispo → on ne peut pas lancer d'autres matchs
// en parallèle avec lui.

static bool is_multithreaded(const std::string& name) {
    // Liste des modèles utilisant std::thread hardware_concurrency()
    static const char* mt_models[] = {
        "TC_MG_mcts",        // arbre partagé + virtual loss
        "TC_MG_montecarlo",  // parallel playouts round-robin
        "TC_MG_uct",         // root parallelization
        "TC_MG_mast",        // root parallelization + tables MAST locales
        "TC_MG_rave",        // root parallelization + amaf[] locaux
        nullptr
    };
    for (int i = 0; mt_models[i]; i++) {
        if (name.find(mt_models[i]) != std::string::npos)
            return true;
    }
    return false;
}

// Retourne le nombre de cœurs "consommés" par un match entre deux joueurs
static int match_core_cost(const std::string& nameA, const std::string& nameB) {
    bool mtA = is_multithreaded(nameA);
    bool mtB = is_multithreaded(nameB);
    if (mtA || mtB) return g_total_cores;  // monopolise tout
    return 1;                               // 1 cœur suffit
}

// ─── Utilitaires board ──────────────────────────────────────────────────────

static std::string board_to_str(const Board64_t& board) {
    std::string s(64, '.');
    for (int i = 0; i < 64; i++) {
        uint64_t bit = (1ULL << i);
        if      (board.black & bit) s[i] = '@';
        else if (board.white & bit) s[i] = 'O';
    }
    return s;
}

static uint64_t coord_to_pos(const std::string& coord) {
    if (coord.size() < 2) return 0ULL;
    int col = coord[0] - 'A';
    int row = 8 - (coord[1] - '0');
    if (col < 0 || col > 7 || row < 0 || row > 7) return 0ULL;
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

// ─── Appel IA externe ────────────────────────────────────────────────────────
// Chaque appel utilise un fichier tmp unique (pid + thread_id) pour éviter
// les collisions entre matchs parallèles.

static std::pair<std::string,std::string>
call_ai(const std::string& exe, const Board64_t& board, int color) {
    std::string boardStr  = board_to_str(board);
    std::string playerStr = (color == WHITE) ? "O" : "@";

    // Nom de fichier tmp unique par thread
    std::ostringstream oss;
    oss << "/tmp/bktour_" << getpid() << "_" << std::this_thread::get_id();
    std::string tmpFile = oss.str();

    char timeBuf[32];
    snprintf(timeBuf, sizeof(timeBuf), "%.2f", g_time_limit);
    std::string cmd = exe + " \"" + boardStr + "\" " + playerStr
                    + " " + timeBuf + " 2>" + tmpFile;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {"resign", ""};
    char buf[64];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);

    std::string stats;
    FILE* sf = fopen(tmpFile.c_str(), "r");
    if (sf) {
        char sbuf[512];
        while (fgets(sbuf, sizeof(sbuf), sf)) stats += sbuf;
        fclose(sf);
        remove(tmpFile.c_str());
    }
    while (!result.empty() && (result.back()=='\n'||result.back()=='\r'||result.back()==' '))
        result.pop_back();
    while (!stats.empty() && (stats.back()=='\n'||stats.back()=='\r'))
        stats.pop_back();
    return {result, stats};
}

// ─── Affichage board verbose ─────────────────────────────────────────────────

static void print_board_live(const Board64_t& board, const std::string& move,
                              int turnNum, const std::string& aiName,
                              bool isWhite, int captured) {
    int w = __builtin_popcountll(board.white);
    int b = __builtin_popcountll(board.black);
    printf("\n  Coup #%d [%s] %s%s%s joue: %s%s%s",
           turnNum, isWhite ? "W" : "B",
           ANSI_CYAN, aiName.c_str(), ANSI_RESET,
           ANSI_YELLOW, move.c_str(), ANSI_RESET);
    if (captured > 0) printf(" " ANSI_RED "CAPTURE!" ANSI_RESET);
    printf("\n  +--------+\n");
    for (int row = 0; row < 8; row++) {
        printf("%d |", 8-row);
        for (int col2 = 0; col2 < 8; col2++) {
            uint64_t bit = (1ULL << (row*8+col2));
            if      (board.black & bit) printf(ANSI_MAGENTA "@" ANSI_RESET);
            else if (board.white & bit) printf(ANSI_WHITE "O" ANSI_RESET);
            else                        printf(ANSI_DIM "." ANSI_RESET);
        }
        printf("|\n");
    }
    printf("  +--------+\n");
    printf("   ABCDEFGH   O:%d  @:%d\n\n", w, b);
}

// ─── Joue une partie simple ──────────────────────────────────────────────────
// Retourne : 0=ai1 gagne, 1=ai2 gagne, -1=nul
// matchTag : identifiant court du match pour les logs en mode parallèle

static int play_one_game(const std::string& ai1, int color1,
                          const std::string& ai2, int color2,
                          int gameNum, int totalGames,
                          const std::string& name1, const std::string& name2,
                          const std::string& matchTag) {
    Board64_t board;
    bool whiteTurn = true;

    if (!g_verbose) {
        std::lock_guard<std::mutex> lk(g_print_mtx);
        printf("  [%s] Partie %2d/%d  %s(%s) vs %s(%s) ... ",
               matchTag.c_str(),
               gameNum, totalGames,
               name1.c_str(), color1==WHITE?"B":"N",
               name2.c_str(), color2==WHITE?"B":"N");
        fflush(stdout);
    }

    for (int turn = 0; turn < MAX_MOVES; turn++) {
        if (board.white_win()) {
            int winner = (color1==WHITE) ? 0 : 1;
            if (!g_verbose) {
                std::lock_guard<std::mutex> lk(g_print_mtx);
                printf(ANSI_GREEN "%s gagne\n" ANSI_RESET,
                       winner==0 ? name1.c_str() : name2.c_str());
            }
            return winner;
        }
        if (board.black_win()) {
            int winner = (color1==BLACK) ? 0 : 1;
            if (!g_verbose) {
                std::lock_guard<std::mutex> lk(g_print_mtx);
                printf(ANSI_GREEN "%s gagne\n" ANSI_RESET,
                       winner==0 ? name1.c_str() : name2.c_str());
            }
            return winner;
        }

        int currentColor = whiteTurn ? WHITE : BLACK;
        const std::string* currentAI;
        int aiIdx;
        const std::string* currentName;

        if (currentColor == color1) {
            currentAI   = &ai1;
            currentName = &name1;
            aiIdx       = 0;
        } else {
            currentAI   = &ai2;
            currentName = &name2;
            aiIdx       = 1;
        }

        int w_before = __builtin_popcountll(board.white);
        int b_before = __builtin_popcountll(board.black);

        std::pair<std::string,std::string> ai_res = call_ai(*currentAI, board, currentColor);
        std::string move  = ai_res.first;
        std::string stats = ai_res.second;

        if (move == "resign" || !apply_move_str(board, move, currentColor)) {
            if (!g_verbose) {
                std::lock_guard<std::mutex> lk(g_print_mtx);
                printf(ANSI_RED "%s abandonne → %s gagne\n" ANSI_RESET,
                       currentName->c_str(),
                       (aiIdx==0 ? name2 : name1).c_str());
            }
            return (aiIdx==0) ? 1 : 0;
        }

        if (g_verbose) {
            int w_after = __builtin_popcountll(board.white);
            int b_after = __builtin_popcountll(board.black);
            int cap = (w_before+b_before) - (w_after+b_after);
            std::lock_guard<std::mutex> lk(g_print_mtx);
            print_board_live(board, move, turn+1, *currentName, whiteTurn, cap);
        }

        whiteTurn = !whiteTurn;
    }

    if (!g_verbose) {
        std::lock_guard<std::mutex> lk(g_print_mtx);
        printf(ANSI_DIM "nul (limite tours)\n" ANSI_RESET);
    }
    return -1;
}

// ─── Structures tournoi ──────────────────────────────────────────────────────

struct Player {
    std::string exe;
    std::string name;
    bool        multithreaded = false;

    // Stats (accès protégé par g_stats_mtx)
    int match_pts    = 0;
    int match_wins   = 0;
    int match_draws  = 0;
    int match_losses = 0;
    int game_wins    = 0;
    int game_losses  = 0;
    int game_draws   = 0;
    int total_games  = 0;
};

struct MatchResult {
    int idxA, idxB;
    int winsA, winsB, draws;
    int total_games;
    int winner;   // 0=A, 1=B, 2=nul
};

// ─── Match adaptatif ─────────────────────────────────────────────────────────

static MatchResult play_match(int idxA, int idxB,
                               const Player& A, const Player& B,
                               int matchNum, int totalMatches) {
    MatchResult res;
    res.idxA        = idxA;
    res.idxB        = idxB;
    res.winsA       = 0;
    res.winsB       = 0;
    res.draws       = 0;
    res.total_games = 0;

    // Tag court pour identifier le match dans les logs parallèles
    std::string tag = std::to_string(matchNum) + "/" + std::to_string(totalMatches);

    {
        std::lock_guard<std::mutex> lk(g_print_mtx);
        printf(ANSI_BOLD "\n══════════════════════════════════════════════════════\n" ANSI_RESET);
        printf(ANSI_BOLD "  [Match %s] %s%s%s  vs  %s%s%s\n" ANSI_RESET,
               tag.c_str(),
               ANSI_CYAN,  A.name.c_str(), ANSI_RESET,
               ANSI_MAGENTA, B.name.c_str(), ANSI_RESET);
        if (A.multithreaded || B.multithreaded)
            printf(ANSI_DIM "  (match mono — au moins un modèle multi-threadé)\n" ANSI_RESET);
        printf(ANSI_BOLD "══════════════════════════════════════════════════════\n" ANSI_RESET);
        fflush(stdout);
    }

    int gameNum = 0;

    while (res.total_games < MAX_GAMES) {
        int target = (res.total_games == 0) ? MIN_GAMES : res.total_games + 2;
        if (target > MAX_GAMES) target = MAX_GAMES;

        while (res.total_games < target) {
            gameNum++;
            int colorA = (res.total_games % 2 == 0) ? WHITE : BLACK;
            int colorB = (colorA == WHITE) ? BLACK : WHITE;

            int r = play_one_game(A.exe, colorA, B.exe, colorB,
                                  gameNum, MAX_GAMES,
                                  A.name, B.name, tag);
            res.total_games++;
            if      (r == 0) res.winsA++;
            else if (r == 1) res.winsB++;
            else             res.draws++;
        }

        int diff = std::abs(res.winsA - res.winsB);

        {
            std::lock_guard<std::mutex> lk(g_print_mtx);
            printf("  [%s] " ANSI_DIM "Score après %d parties : %s%s%s %d – %d %s%s%s",
                   tag.c_str(), res.total_games,
                   ANSI_CYAN,  A.name.c_str(), ANSI_RESET, res.winsA,
                   res.winsB,
                   ANSI_MAGENTA, B.name.c_str(), ANSI_RESET);
            if (res.draws > 0) printf(" (%d nuls)", res.draws);
            printf(ANSI_RESET "\n");

            if (diff <= TIE_THRESH && res.total_games < MAX_GAMES)
                printf("  [%s] " ANSI_YELLOW "⟳  Trop serré (%d), prolongation +2 parties...\n" ANSI_RESET,
                       tag.c_str(), diff);
            fflush(stdout);
        }

        if (diff > TIE_THRESH || res.total_games >= MAX_GAMES)
            break;
    }

    int diff = res.winsA - res.winsB;
    if (diff > 0)      res.winner = 0;
    else if (diff < 0) res.winner = 1;
    else               res.winner = 2;

    {
        std::lock_guard<std::mutex> lk(g_print_mtx);
        printf(ANSI_BOLD "  [%s] → ", tag.c_str());
        if (res.winner == 0)
            printf(ANSI_GREEN "%s gagne le match (%d-%d)\n" ANSI_RESET,
                   A.name.c_str(), res.winsA, res.winsB);
        else if (res.winner == 1)
            printf(ANSI_GREEN "%s gagne le match (%d-%d)\n" ANSI_RESET,
                   B.name.c_str(), res.winsB, res.winsA);
        else
            printf(ANSI_YELLOW "Match nul (%d-%d)\n" ANSI_RESET, res.winsA, res.winsB);
        printf(ANSI_BOLD "══════════════════════════════════════════════════════\n" ANSI_RESET);
        fflush(stdout);
    }

    return res;
}

// ─── Auto-détection TC_MG_* ──────────────────────────────────────────────────

static std::vector<std::string> auto_detect_players() {
    std::vector<std::string> players;
    DIR* dir = opendir(".");
    if (!dir) return players;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string fname(ent->d_name);
        if (fname.substr(0, 6) != "TC_MG_") continue;
        if (fname.find("tournament") != std::string::npos) continue;
        if (fname.find("benchmark")  != std::string::npos) continue;
        std::string path = "./" + fname;
        if (access(path.c_str(), X_OK) == 0)
            players.push_back(path);
    }
    closedir(dir);
    std::sort(players.begin(), players.end());
    return players;
}

static std::string short_name(const std::string& path) {
    size_t pos = path.rfind('/');
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

template<typename T>
static void shuffle_vec(std::vector<T>& v) {
    for (int i = (int)v.size()-1; i > 0; i--) {
        int j = rand() % (i+1);
        std::swap(v[i], v[j]);
    }
}

// ─── Affichage classement ────────────────────────────────────────────────────

static void print_ranking(std::vector<Player>& players,
                           const std::vector<MatchResult>& results) {
    int n = (int)players.size();

    printf(ANSI_BOLD "\n\n╔══════════════════════════════════════════════════════════╗\n");
    printf(         "║         TABLEAU DE POULE (victoires de la ligne vs col)  ║\n");
    printf(         "╚══════════════════════════════════════════════════════════╝\n" ANSI_RESET);

    const int COL = 6;
    printf("  %-18s", "");
    for (int j = 0; j < n; j++) {
        std::string lbl = players[j].name;
        size_t p = lbl.find('_', 3);
        if (p != std::string::npos) lbl = lbl.substr(p+1);
        if ((int)lbl.size() > COL-1) lbl = lbl.substr(0, COL-1);
        printf(" %*s", COL, lbl.c_str());
    }
    printf("  | Pts  W   L   D  %%Win\n");

    printf("  %-18s", "");
    for (int j = 0; j < n; j++) printf(" ------");
    printf("  +----+---+---+---+------\n");

    std::vector<std::vector<int>> mat(n, std::vector<int>(n, -1));
    for (const auto& r : results) {
        mat[r.idxA][r.idxB] = r.winsA;
        mat[r.idxB][r.idxA] = r.winsB;
    }

    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (players[a].match_pts != players[b].match_pts)
            return players[a].match_pts > players[b].match_pts;
        int da = players[a].game_wins - players[a].game_losses;
        int db = players[b].game_wins - players[b].game_losses;
        if (da != db) return da > db;
        return players[a].game_wins > players[b].game_wins;
    });

    for (int rank = 0; rank < n; rank++) {
        int i = order[rank];
        const Player& p = players[i];
        std::string rowlbl = "#" + std::to_string(rank+1) + " " + p.name;
        if ((int)rowlbl.size() > 18) rowlbl = rowlbl.substr(0, 18);
        printf("  %-18s", rowlbl.c_str());
        for (int j = 0; j < n; j++) {
            if (i == j) {
                printf(" " ANSI_DIM "  --- " ANSI_RESET);
            } else if (mat[i][j] < 0) {
                printf("   N/A");
            } else {
                int vi = mat[i][j];
                int vj = mat[j][i] >= 0 ? mat[j][i] : 0;
                if      (vi > vj) printf(" " ANSI_GREEN "%*d" ANSI_RESET, COL, vi);
                else if (vi < vj) printf(" " ANSI_RED   "%*d" ANSI_RESET, COL, vi);
                else              printf(" " ANSI_YELLOW "%*d" ANSI_RESET, COL, vi);
            }
        }
        int tot = p.game_wins + p.game_losses + p.game_draws;
        double pct = tot > 0 ? 100.0 * p.game_wins / tot : 0.0;
        printf("  | %3d  %3d %3d %3d  %5.1f%%\n",
               p.match_pts, p.game_wins, p.game_losses, p.game_draws, pct);
    }

    printf(ANSI_BOLD "\n╔══════════════════════════════════════════════════════════╗\n");
    printf(         "║                    CLASSEMENT FINAL                     ║\n");
    printf(         "╚══════════════════════════════════════════════════════════╝\n" ANSI_RESET);

    printf("  %-4s  %-22s  %-3s  %4s  %4s  %4s  %4s  %6s  %5s  %5s\n",
           "Rang", "Modèle", "MT", "Pts", "MW", "ML", "MD", "Parties", "Wins", "%Win");
    printf("  ────  ──────────────────────  ───  ────  ────  ────  ────  ──────  ─────  ─────\n");

    const char* medals[] = {
        ANSI_YELLOW "🥇" ANSI_RESET,
        ANSI_WHITE  "🥈" ANSI_RESET,
        "\033[33m🥉" ANSI_RESET,
    };

    for (int rank = 0; rank < n; rank++) {
        int i = order[rank];
        const Player& p = players[i];
        int tot = p.game_wins + p.game_losses + p.game_draws;
        double pct = tot > 0 ? 100.0 * p.game_wins / tot : 0.0;
        const char* medal = (rank < 3) ? medals[rank] : "  ";
        const char* mt_lbl = p.multithreaded ? ANSI_CYAN " MT" ANSI_RESET : ANSI_DIM " ST" ANSI_RESET;
        printf("  %s%-2d  %-22s  %s  %4d  %4d  %4d  %4d  %6d  %5d  %5.1f%%\n",
               medal, rank+1, p.name.c_str(), mt_lbl,
               p.match_pts, p.match_wins, p.match_losses, p.match_draws,
               tot, p.game_wins, pct);
    }

    printf(ANSI_BOLD "\n╔══════════════════════════════════════════════════════════╗\n");
    printf(         "║                  DÉTAIL DES MATCHS                      ║\n");
    printf(         "╚══════════════════════════════════════════════════════════╝\n" ANSI_RESET);

    for (const auto& r : results) {
        const std::string& na = players[r.idxA].name;
        const std::string& nb = players[r.idxB].name;
        const char* verdict;
        if      (r.winner == 0) verdict = ANSI_GREEN " → A gagne" ANSI_RESET;
        else if (r.winner == 1) verdict = ANSI_RED   " → B gagne" ANSI_RESET;
        else                    verdict = ANSI_YELLOW " → Nul"     ANSI_RESET;
        printf("  %s%-22s%s %2d - %-2d %s%-22s%s  (%d parties)%s\n",
               ANSI_CYAN,  na.c_str(), ANSI_RESET, r.winsA,
               r.winsB,
               ANSI_MAGENTA, nb.c_str(), ANSI_RESET,
               r.total_games, verdict);
    }
    printf("\n");
}

// ─── Classement partiel (appelé sous g_print_mtx) ────────────────────────────

static void print_partial_ranking(const std::vector<Player>& players,
                                   int doneMatches, int totalMatches) {
    int n = (int)players.size();
    printf(ANSI_DIM "\n  Classement partiel (%d/%d matchs joués):\n", doneMatches, totalMatches);

    std::vector<int> ord(n);
    for (int i = 0; i < n; i++) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (players[a].match_pts != players[b].match_pts)
            return players[a].match_pts > players[b].match_pts;
        return (players[a].game_wins - players[a].game_losses) >
               (players[b].game_wins - players[b].game_losses);
    });
    for (int r = 0; r < n; r++) {
        int i = ord[r];
        int tot = players[i].game_wins + players[i].game_losses + players[i].game_draws;
        double pct = tot > 0 ? 100.0 * players[i].game_wins / tot : 0.0;
        printf("    %d. %-22s  %3dpts  %dW-%dL-%dD  (%.0f%% wins)\n",
               r+1, players[i].name.c_str(),
               players[i].match_pts,
               players[i].game_wins, players[i].game_losses, players[i].game_draws,
               pct);
    }
    printf(ANSI_RESET);
    fflush(stdout);
}

// ─── Usage ───────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s                           # auto-détecte tous les TC_MG_*\n"
        "  %s [time_s] [exe1 exe2 ...]  # spécifie durée et/ou liste d'IA\n"
        "  %s -v [time_s] [...]         # verbose (affiche chaque coup)\n"
        "\nExemples:\n"
        "  %s 0.5\n"
        "  %s 1.0 ./TC_MG_mcts ./TC_MG_uct ./TC_MG_rave\n"
        "\nParallélisme automatique :\n"
        "  Les modèles mono-thread jouent plusieurs matchs en simultané\n"
        "  (autant que de cœurs CPU disponibles).\n"
        "  Les modèles multi-threadés (ex: TC_MG_mcts) monopolisent tous\n"
        "  les cœurs : leurs matchs s'exécutent seuls.\n"
        "\nRègles match adaptatif:\n"
        "  - Démarre à %d parties (alternance couleurs)\n"
        "  - Si écart ≤ %d : prolonge par +2 jusqu'à %d parties max\n"
        "  - Vainqueur = plus de victoires ; égalité parfaite = nul\n",
        prog, prog, prog, prog, prog, MIN_GAMES, TIE_THRESH, MAX_GAMES);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));

    g_total_cores = (int)std::thread::hardware_concurrency();
    if (g_total_cores < 1) g_total_cores = 1;

    // ── Parse arguments ───────────────────────────────────────────────────
    std::vector<std::string> exes;
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "-v" || a == "--verbose") {
            g_verbose = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            char* end;
            double t = strtod(a.c_str(), &end);
            if (end != a.c_str() && *end == '\0' && t > 0)
                g_time_limit = t;
            else
                exes.push_back(a);
        }
    }

    if (exes.empty()) {
        exes = auto_detect_players();
        if (exes.empty()) {
            fprintf(stderr, ANSI_RED "Aucun exécutable TC_MG_* trouvé dans ./\n" ANSI_RESET);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (exes.size() < 2) {
        fprintf(stderr, ANSI_RED "Il faut au moins 2 modèles.\n" ANSI_RESET);
        return 1;
    }

    // ── Construire les joueurs ─────────────────────────────────────────────
    std::vector<Player> players;
    players.reserve(exes.size());
    for (const auto& e : exes) {
        Player p;
        p.exe           = e;
        p.name          = short_name(e);
        p.multithreaded = is_multithreaded(p.name);
        players.push_back(p);
    }
    int n = (int)players.size();

    // ── Bandeau ───────────────────────────────────────────────────────────
    printf(ANSI_BOLD ANSI_CYAN
           "\n╔══════════════════════════════════════════════════════════════╗\n"
           "║       TOURNOI BREAKTHROUGH — ROUND-ROBIN PARALLÈLE          ║\n"
           "╚══════════════════════════════════════════════════════════════╝\n"
           ANSI_RESET);
    printf("  %d modèles  |  %.1fs/coup  |  %d-%d parties/match  |  %d cœurs CPU\n\n",
           n, g_time_limit, MIN_GAMES, MAX_GAMES, g_total_cores);

    for (int i = 0; i < n; i++) {
        const char* mt = players[i].multithreaded
            ? ANSI_CYAN "[MT — multi-threadé]" ANSI_RESET
            : ANSI_DIM  "[ST — mono-thread]"   ANSI_RESET;
        printf("  [%d] %s%s%s  %s\n", i,
               ANSI_BOLD, players[i].name.c_str(), ANSI_RESET, mt);
    }

    // ── Liste des matchs ──────────────────────────────────────────────────
    std::vector<std::pair<int,int>> matchups;
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            matchups.push_back({i, j});
    shuffle_vec(matchups);

    int totalMatches = (int)matchups.size();
    printf("\n  %d matchs au programme\n", totalMatches);
    printf(ANSI_DIM "  Ordre mélangé aléatoirement\n" ANSI_RESET);
    printf(ANSI_DIM "  Matchs mono-thread joués en parallèle (%d cœurs)\n\n" ANSI_RESET,
           g_total_cores);

    // ── Phase de poule avec parallélisme adaptatif ────────────────────────
    //
    // On maintient un compteur de "cœurs occupés".
    // Quand un match implique un modèle MT → coût = g_total_cores
    // Sinon → coût = 1
    // On lance un nouveau match uniquement si cœurs_dispo >= coût du match
    //
    // On utilise une file de travail (matchups), des threads workers,
    // et un mutex + condition_variable pour la coordination.

    std::vector<MatchResult> allResults;
    allResults.reserve(totalMatches);
    std::mutex              g_stats_mtx;   // protège players[] et allResults
    std::mutex              g_queue_mtx;   // protège la file de matchs
    std::condition_variable g_queue_cv;

    std::atomic<int> g_cores_used{0};
    std::atomic<int> g_matches_done{0};

    // Index courant dans matchups
    std::atomic<int> g_next_match{0};

    // Fonction worker : prend des matchs dans la file dès qu'il y a de la place
    auto worker = [&]() {
        while (true) {
            // Attendre qu'un match soit disponible ET que les cœurs le permettent
            int mi = -1;
            int cost = 0;
            {
                std::unique_lock<std::mutex> lk(g_queue_mtx);
                g_queue_cv.wait(lk, [&]() {
                    if (g_next_match.load() >= totalMatches) return true; // terminé
                    int idx = g_next_match.load();
                    int c = match_core_cost(
                        players[matchups[idx].first].name,
                        players[matchups[idx].second].name);
                    return (g_total_cores - g_cores_used.load()) >= c;
                });

                if (g_next_match.load() >= totalMatches) break;

                mi   = g_next_match.fetch_add(1);
                cost = match_core_cost(
                    players[matchups[mi].first].name,
                    players[matchups[mi].second].name);
                g_cores_used.fetch_add(cost);
            }

            int idxA = matchups[mi].first;
            int idxB = matchups[mi].second;
            int matchNum = mi + 1;

            MatchResult res = play_match(idxA, idxB,
                                          players[idxA], players[idxB],
                                          matchNum, totalMatches);

            // Libérer les cœurs et notifier
            g_cores_used.fetch_sub(cost);
            g_matches_done.fetch_add(1);
            g_queue_cv.notify_all();

            // Mettre à jour les stats (protégé)
            {
                std::lock_guard<std::mutex> lk(g_stats_mtx);
                allResults.push_back(res);

                Player& A = players[idxA];
                Player& B = players[idxB];

                A.game_wins   += res.winsA;
                A.game_losses += res.winsB;
                A.game_draws  += res.draws;
                A.total_games += res.total_games;

                B.game_wins   += res.winsB;
                B.game_losses += res.winsA;
                B.game_draws  += res.draws;
                B.total_games += res.total_games;

                if (res.winner == 0) {
                    A.match_pts += 3; A.match_wins++;
                    B.match_losses++;
                } else if (res.winner == 1) {
                    B.match_pts += 3; B.match_wins++;
                    A.match_losses++;
                } else {
                    A.match_pts += 1; A.match_draws++;
                    B.match_pts += 1; B.match_draws++;
                }
            }

            // Classement partiel (sous print mutex + stats lock)
            {
                std::lock_guard<std::mutex> slk(g_stats_mtx);
                std::lock_guard<std::mutex> plk(g_print_mtx);
                print_partial_ranking(players, g_matches_done.load(), totalMatches);
            }
        }
    };

    // Lancer autant de threads workers que de cœurs (au max totalMatches)
    int nworkers = std::min(g_total_cores, totalMatches);
    std::vector<std::thread> threads;
    threads.reserve(nworkers);

    // Déclencher la condition initiale
    g_queue_cv.notify_all();
    for (int t = 0; t < nworkers; t++)
        threads.emplace_back(worker);

    // Notifier régulièrement pour débloquer les workers en attente
    // (nécessaire si des matchs MT libèrent des cœurs)
    std::thread notifier([&]() {
        while (g_matches_done.load() < totalMatches) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            g_queue_cv.notify_all();
        }
    });

    for (auto& th : threads) th.join();
    notifier.join();

    // ── Classement final ──────────────────────────────────────────────────
    // Trier allResults par (idxA, idxB) pour un affichage cohérent
    std::sort(allResults.begin(), allResults.end(), [](const MatchResult& a, const MatchResult& b) {
        if (a.idxA != b.idxA) return a.idxA < b.idxA;
        return a.idxB < b.idxB;
    });

    print_ranking(players, allResults);
    return 0;
}

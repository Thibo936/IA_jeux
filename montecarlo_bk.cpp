// montecarlo_bk.cpp - Monte-Carlo pur parallèle pour Breakthrough
//
// Parallélisation : chaque thread effectue des playouts indépendants sur son
// propre RNG (thread-local xorshift) et accumule dans des compteurs locaux.
// Fusion des scores par réduction à la fin du budget temps.
// Stratégie : root parallelization avec round-robin par thread.
//
// Usage : ./TC_MG_montecarlo BOARD PLAYER [time_s]

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include "bkbb64.h"

#define WHITE 0
#define BLACK 1

double g_time_limit_s = 1.0;

static const double CAPTURE_PROB   = 0.80;
static const int    MAX_PLAYOUT_DEPTH = 200;

// ─── RNG thread-local ────────────────────────────────────────────────────────
// Chaque thread a son propre état xorshift → pas de contention sur rand()

static thread_local uint32_t tl_seed = 0;

static inline uint32_t tl_rand() {
    tl_seed ^= tl_seed << 13;
    tl_seed ^= tl_seed >> 17;
    tl_seed ^= tl_seed << 5;
    return tl_seed;
}

static inline void tl_seed_init(uint32_t base, int thread_id) {
    tl_seed = base ^ (uint32_t)(thread_id * 2654435761u);
    if (tl_seed == 0) tl_seed = 1;
}

// ─── Évaluation heuristique ──────────────────────────────────────────────────

static int evaluate_board(const Board64_t& board) {
    if (board.white_win()) return  100000;
    if (board.black_win()) return -100000;

    int score = 0;
    int w_best_row = 7, b_best_row = 0;

    uint64_t w = board.white;
    while (w) {
        int idx = __builtin_ctzll(w);
        int row = idx / 8;
        score += 10;
        score += (7 - row) * 5;
        if (row <= 1) score += (2 - row) * 30;
        if (row < w_best_row) w_best_row = row;
        w &= w - 1;
    }
    uint64_t b = board.black;
    while (b) {
        int idx = __builtin_ctzll(b);
        int row = idx / 8;
        score -= 10;
        score -= row * 5;
        if (row >= 6) score -= (row - 5) * 30;
        if (row > b_best_row) b_best_row = row;
        b &= b - 1;
    }
    score += (7 - w_best_row) * 10;
    score -= b_best_row * 10;
    uint64_t wp = board.white & ((board.white << 7) | (board.white << 9));
    score += __builtin_popcountll(wp) * 3;
    uint64_t bp = board.black & ((board.black >> 7) | (board.black >> 9));
    score -= __builtin_popcountll(bp) * 3;
    return score;
}

// ─── Utilitaires légaux ──────────────────────────────────────────────────────

static std::vector<Move64_t> get_legal_moves(const Board64_t& board, bool isWhite) {
    uint64_t l = isWhite ? board.white_left()    : board.black_left();
    uint64_t f = isWhite ? board.white_forward() : board.black_forward();
    uint64_t r = isWhite ? board.white_right()   : board.black_right();
    return Lfr_t(l, f, r).get_moves(isWhite);
}

static bool is_immediate_win(const Board64_t& board, const Move64_t& move, bool isWhite) {
    Board64_t nb = board;
    nb.apply_move(move, isWhite);
    return isWhite ? nb.white_win() : nb.black_win();
}

static bool find_immediate_winning_move(const Board64_t& board, bool isWhite, Move64_t& out) {
    for (const auto& m : get_legal_moves(board, isWhite))
        if (is_immediate_win(board, m, isWhite)) { out = m; return true; }
    return false;
}

static std::vector<Move64_t> get_safe_moves(const Board64_t& board, bool isWhite) {
    std::vector<Move64_t> safe;
    for (const auto& m : get_legal_moves(board, isWhite)) {
        Board64_t nb = board;
        nb.apply_move(m, isWhite);
        if (nb.white_win() || nb.black_win()) { safe.push_back(m); continue; }
        Move64_t dummy;
        if (!find_immediate_winning_move(nb, !isWhite, dummy))
            safe.push_back(m);
    }
    return safe;
}

// ─── Playout epsilon-greedy (utilise le RNG thread-local) ───────────────────

static int playout_score(Board64_t board, bool currentPlayer) {
    for (int depth = 0; depth < MAX_PLAYOUT_DEPTH; depth++) {
        if (board.white_win()) return  100000;
        if (board.black_win()) return -100000;

        uint64_t l = currentPlayer ? board.white_left()    : board.black_left();
        uint64_t f = currentPlayer ? board.white_forward() : board.black_forward();
        uint64_t r = currentPlayer ? board.white_right()   : board.black_right();
        if (count64(l) + count64(f) + count64(r) == 0)
            return currentPlayer ? -100000 : 100000;

        uint64_t enemy    = currentPlayer ? board.black : board.white;
        uint64_t real_caps = (l | r) & enemy;

        // Utilise le RNG thread-local au lieu de rand()
        if (real_caps && (double)(tl_rand() & 0xFFFF) / 65535.0 < CAPTURE_PROB) {
            uint64_t cap_l = l & enemy;
            uint64_t cap_r = r & enemy;
            // Seed du board non utilisé ici : on utilise directement tl_rand
            board.seed = tl_rand();
            board.apply_move(board.get_rand_move(Lfr_t(cap_l, 0ULL, cap_r), currentPlayer),
                             currentPlayer);
        } else {
            board.seed = tl_rand();
            board.rand_move(currentPlayer);
        }
        currentPlayer = !currentPlayer;
    }
    return evaluate_board(board);
}

// ─── Sélection du meilleur coup (multi-threadé) ──────────────────────────────

std::string get_best_move(Board64_t& board, int color) {
    bool isWhite = (color == WHITE);

    Move64_t forcedWin;
    if (find_immediate_winning_move(board, isWhite, forcedWin))
        return pos_to_coord(forcedWin.pi) + "-" + pos_to_coord(forcedWin.pf);

    std::vector<Move64_t> moves = get_safe_moves(board, isWhite);
    if (moves.empty()) moves = get_legal_moves(board, isWhite);
    if (moves.empty()) return "resign";

    uint64_t enemy = isWhite ? board.black : board.white;
    std::stable_sort(moves.begin(), moves.end(),
        [enemy](const Move64_t& a, const Move64_t& b_) {
            return (int)((a.pf & enemy) != 0) > (int)((b_.pf & enemy) != 0);
        });

    int N = (int)moves.size();

    // Compteurs partagés (atomiques pour la réduction)
    std::vector<std::atomic<long long>> scores(N);
    std::vector<std::atomic<int>>       counts(N);
    for (int i = 0; i < N; i++) { scores[i] = 0; counts[i] = 0; }

    std::atomic<int>  g_total_playouts{0};
    std::atomic<bool> g_stop{false};

    auto start = std::chrono::high_resolution_clock::now();

    int nthreads = (int)std::max(1u, std::thread::hardware_concurrency());

    auto worker = [&](int tid) {
        tl_seed_init((uint32_t)time(nullptr), tid);
        int local_round = tid; // chaque thread commence à un offset différent

        while (!g_stop.load(std::memory_order_relaxed)) {
            // Vérification du temps toutes les 64 itérations
            if ((local_round / N) % 64 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(now - start).count() >= g_time_limit_s) {
                    g_stop.store(true);
                    break;
                }
            }

            int idx = local_round % N;
            Board64_t nb = board;
            nb.apply_move(moves[idx], isWhite);

            int sc = playout_score(nb, !isWhite);
            scores[idx].fetch_add(sc,    std::memory_order_relaxed);
            counts[idx].fetch_add(1,     std::memory_order_relaxed);
            g_total_playouts.fetch_add(1, std::memory_order_relaxed);

            local_round += nthreads; // round-robin inter-threads
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; t++)
        threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();

    // Sélection du meilleur coup par score moyen
    Move64_t bestMove = moves[0];
    double bestAvg = isWhite ? -1e18 : 1e18;
    for (int i = 0; i < N; i++) {
        int cnt = counts[i].load();
        if (cnt == 0) continue;
        double avg = (double)scores[i].load() / cnt;
        bool better = isWhite ? (avg > bestAvg) : (avg < bestAvg);
        if (better) { bestAvg = avg; bestMove = moves[i]; }
    }

    fprintf(stderr, "MC: playouts=%d threads=%d moves=%d time=%.2fs\n",
            g_total_playouts.load(), nthreads, N, elapsed);

    return pos_to_coord(bestMove.pi) + "-" + pos_to_coord(bestMove.pf);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    srand(time(nullptr));
    if (argc < 3) {
        fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]);
        return 0;
    }
    if (argc >= 4) {
        g_time_limit_s = atof(argv[3]);
        if (g_time_limit_s <= 0) g_time_limit_s = 1.0;
    }

    Board64_t B(argv[1]);
    B.seed = (uint32_t)time(nullptr);
    std::string playerStr(argv[2]);

    if (playerStr == "O")
        printf("%s\n", get_best_move(B, WHITE).c_str());
    else if (playerStr == "@")
        printf("%s\n", get_best_move(B, BLACK).c_str());

    return 0;
}

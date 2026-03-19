// rave_bk.cpp - RAVE/AMAF parallèle pour Breakthrough
//
// Parallélisation : Root Parallelization
//   - N threads, chacun avec son propre arbre RAVE complet (nœuds + amaf[])
//   - RNG thread-local → zéro contention, zéro mutex pendant la recherche
//   - Fusion finale des enfants directs de la racine :
//     visits, wins, et amaf[] sommés pour choisir le coup le plus robuste
//
// Référence : Gelly & Silver, RAVE ICML 2007
// Usage : ./TC_MG_rave BOARD PLAYER [time_s]

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <map>
#include "bkbb64.h"

#define WHITE     0
#define BLACK     1
#define UCT_C     0.5
#define RAVE_K    1000
#define MAX_DEPTH 200

double g_time_limit_s = 1.0;

// ─── RNG thread-local ────────────────────────────────────────────────────────

static thread_local uint32_t tl_seed = 1;

static inline uint32_t tl_rand() {
    tl_seed ^= tl_seed << 13;
    tl_seed ^= tl_seed >> 17;
    tl_seed ^= tl_seed << 5;
    return tl_seed;
}
static inline void tl_seed_init(uint32_t base, int tid) {
    tl_seed = base ^ (uint32_t)(tid * 2654435761u);
    if (!tl_seed) tl_seed = 1;
}

// ─── Utilitaires ────────────────────────────────────────────────────────────

static std::vector<Move64_t> get_legal_moves(const Board64_t& board, bool isWhite) {
    uint64_t l = isWhite ? board.white_left() : board.black_left();
    uint64_t f = isWhite ? board.white_forward() : board.black_forward();
    uint64_t r = isWhite ? board.white_right() : board.black_right();
    return Lfr_t(l, f, r).get_moves(isWhite);
}

static inline int move_amaf_id(const Move64_t& m) {
    return __builtin_ctzll(m.pi) * 64 + __builtin_ctzll(m.pf);
}

// Playout aléatoire avec trace AMAF (utilise RNG thread-local)
static double random_playout_amaf(Board64_t board, bool cur,
                                   std::vector<int>& amaf_w,
                                   std::vector<int>& amaf_b) {
    for (int d = 0; d < MAX_DEPTH; d++) {
        if (board.white_win()) return 1.0;
        if (board.black_win()) return 0.0;
        uint64_t l = cur ? board.white_left()    : board.black_left();
        uint64_t f = cur ? board.white_forward() : board.black_forward();
        uint64_t r = cur ? board.white_right()   : board.black_right();
        if (count64(l) + count64(f) + count64(r) == 0) return cur ? 0.0 : 1.0;
        board.seed = tl_rand();
        Move64_t mv = board.get_rand_move(Lfr_t(l, f, r), cur);
        (cur ? amaf_w : amaf_b).push_back(move_amaf_id(mv));
        board.apply_move(mv, cur);
        cur = !cur;
    }
    return __builtin_popcountll(board.white) >= __builtin_popcountll(board.black) ? 1.0 : 0.0;
}

// ─── Nœud RAVE (par thread, aucun mutex) ─────────────────────────────────────

struct RAVENode {
    Board64_t board;
    bool      isWhiteToPlay;
    Move64_t  moveFromParent;
    RAVENode* parent;

    std::vector<RAVENode*> children;
    std::vector<Move64_t>  untriedMoves;
    int    visits;
    double wins;

    struct AmafEntry { double wins = 0.0; int visits = 0; };
    std::vector<AmafEntry> amaf; // 4096 entrées

    RAVENode(const Board64_t& b, bool w, RAVENode* p, Move64_t m)
        : board(b), isWhiteToPlay(w), moveFromParent(m),
          parent(p), visits(0), wins(0.0)
    {
        untriedMoves = get_legal_moves(b, w);
        amaf.resize(4096);
    }

    ~RAVENode() { for (auto* c : children) delete c; }

    bool isTerminal() const { return board.white_win() || board.black_win(); }

    double rave_score(const RAVENode* child) const {
        int aid = move_amaf_id(child->moveFromParent);
        const AmafEntry& ae = amaf[aid];
        double q_uct  = child->wins / (child->visits + 1e-9);
        double q_amaf = ae.visits > 0 ? ae.wins / ae.visits : 0.5;
        double beta   = std::sqrt((double)RAVE_K / (3.0 * visits + RAVE_K));
        double exploit = (1.0 - beta) * q_uct + beta * q_amaf;
        double explore = UCT_C * std::sqrt(std::log(visits + 1.0) / (child->visits + 1e-9));
        return exploit + explore;
    }

    RAVENode* bestChild() const {
        RAVENode* best = nullptr; double bestV = -1e18;
        for (auto* c : children) {
            double v = rave_score(c);
            if (v > bestV) { bestV = v; best = c; }
        }
        return best;
    }

    // Expansion : choisit le coup non essayé avec meilleur score AMAF
    RAVENode* expand() {
        int bestIdx = (int)untriedMoves.size() - 1;
        double bestA = -1e9;
        for (int i = 0; i < (int)untriedMoves.size(); i++) {
            const AmafEntry& ae = amaf[move_amaf_id(untriedMoves[i])];
            double sc = ae.visits > 0 ? ae.wins / ae.visits : 0.5;
            if (sc > bestA) { bestA = sc; bestIdx = i; }
        }
        Move64_t move = untriedMoves[bestIdx];
        untriedMoves.erase(untriedMoves.begin() + bestIdx);
        Board64_t nb = board; nb.apply_move(move, isWhiteToPlay);
        RAVENode* child = new RAVENode(nb, !isWhiteToPlay, this, move);
        children.push_back(child);
        return child;
    }
};

static RAVENode* rave_select(RAVENode* root) {
    RAVENode* node = root;
    while (!node->isTerminal()) {
        if (!node->untriedMoves.empty()) return node->expand();
        RAVENode* c = node->bestChild();
        if (!c) break;
        node = c;
    }
    return node;
}

static void rave_backprop(RAVENode* leaf, double prob,
                           const std::vector<int>& amaf_w,
                           const std::vector<int>& amaf_b) {
    RAVENode* node = leaf;
    while (node) {
        node->visits++;
        bool byWhite = node->parent ? node->parent->isWhiteToPlay : false;
        node->wins += byWhite ? prob : (1.0 - prob);

        const std::vector<int>& ids = node->isWhiteToPlay ? amaf_w : amaf_b;
        double reward = node->isWhiteToPlay ? prob : (1.0 - prob);
        for (int aid : ids) {
            node->amaf[aid].visits++;
            node->amaf[aid].wins += reward;
        }
        node = node->parent;
    }
}

// ─── Stats de racine (pour fusion) ──────────────────────────────────────────

struct RootChildStats {
    Move64_t move;
    int      visits;
    double   wins;
};

// ─── Worker : un arbre RAVE indépendant par thread ────────────────────────────

static std::vector<RootChildStats>
run_rave_tree(const Board64_t& board, bool isWhite,
              const std::chrono::high_resolution_clock::time_point& start,
              std::atomic<bool>& stop, int tid) {
    tl_seed_init((uint32_t)time(nullptr), tid);

    RAVENode* root = new RAVENode(board, isWhite, nullptr, Move64_t());
    int iters = 0;

    while (!stop.load(std::memory_order_relaxed)) {
        if (iters % 256 == 0) {
            if (std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - start).count()
                    >= g_time_limit_s) {
                stop.store(true);
                break;
            }
        }
        iters++;

        RAVENode* leaf = rave_select(root);
        if (!leaf) break;

        double result;
        std::vector<int> amaf_w, amaf_b;

        if (leaf->isTerminal())
            result = leaf->board.white_win() ? 1.0 : 0.0;
        else
            result = random_playout_amaf(leaf->board, leaf->isWhiteToPlay, amaf_w, amaf_b);

        rave_backprop(leaf, result, amaf_w, amaf_b);
    }

    std::vector<RootChildStats> stats;
    stats.reserve(root->children.size());
    for (auto* c : root->children)
        stats.push_back({c->moveFromParent, c->visits, c->wins});

    delete root;
    return stats;
}

// ─── get_best_move ───────────────────────────────────────────────────────────

std::string get_best_move(Board64_t& board, int color) {
    bool isWhite = (color == WHITE);

    for (const auto& m : get_legal_moves(board, isWhite)) {
        Board64_t nb = board; nb.apply_move(m, isWhite);
        if (isWhite ? nb.white_win() : nb.black_win())
            return pos_to_coord(m.pi) + "-" + pos_to_coord(m.pf);
    }
    if (get_legal_moves(board, isWhite).empty()) return "resign";

    int nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
    auto start   = std::chrono::high_resolution_clock::now();
    std::atomic<bool> stop{false};

    std::vector<std::vector<RootChildStats>> thread_results(nthreads);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back([&, t]() {
            thread_results[t] = run_rave_tree(board, isWhite, start, stop, t);
        });
    }
    for (auto& th : threads) th.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();

    // Fusion des stats racine
    struct Merged { int visits = 0; double wins = 0.0; Move64_t move; };
    std::map<std::pair<uint64_t,uint64_t>, Merged> fusion;
    for (const auto& tvec : thread_results)
        for (const auto& s : tvec) {
            auto key = std::make_pair(s.move.pi, s.move.pf);
            auto& mg = fusion[key];
            mg.move    = s.move;
            mg.visits += s.visits;
            mg.wins   += s.wins;
        }

    if (fusion.empty()) return "resign";

    Move64_t bestMove = {0ULL, 0ULL}; int bestVisits = -1; double bestWins = 0.0;
    for (const auto& kv : fusion)
        if (kv.second.visits > bestVisits) {
            bestVisits = kv.second.visits;
            bestWins   = kv.second.wins;
            bestMove   = kv.second.move;
        }

    int totalIters = 0;
    for (const auto& kv : fusion) totalIters += kv.second.visits;
    double wr = bestWins / (bestVisits + 1e-9);

    fprintf(stderr, "RAVE: iters=%d visits=%d winrate=%.4f threads=%d time=%.2fs\n",
            totalIters, bestVisits, wr, nthreads, elapsed);

    return pos_to_coord(bestMove.pi) + "-" + pos_to_coord(bestMove.pf);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    srand(time(nullptr));
    if (argc < 3) { fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]); return 0; }
    if (argc >= 4) { g_time_limit_s = atof(argv[3]); if (g_time_limit_s <= 0) g_time_limit_s = 1.0; }

    Board64_t B(argv[1]); B.seed = (uint32_t)time(nullptr);
    std::string playerStr(argv[2]);

    if      (playerStr == "O") printf("%s\n", get_best_move(B, WHITE).c_str());
    else if (playerStr == "@") printf("%s\n", get_best_move(B, BLACK).c_str());
    return 0;
}

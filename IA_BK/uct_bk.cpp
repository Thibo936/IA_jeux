// uct_bk.cpp - UCT parallèle pour Breakthrough
//
// Parallélisation : Root Parallelization
//   - N threads, chacun avec son propre arbre UCT complet et son RNG thread-local
//   - Aucun verrou pendant la recherche → zéro contention
//   - À la fin, fusion des statistiques des enfants directs de la racine :
//     visits et wins sont sommés pour choisir le coup le plus robuste
//
// Usage : ./TC_MG_uct BOARD PLAYER [time_s]

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

#define WHITE  0
#define BLACK  1
#define UCT_C  0.7

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

static double random_playout(Board64_t board, bool cur) {
    for (int d = 0; d < 200; d++) {
        if (board.white_win()) return 1.0;
        if (board.black_win()) return 0.0;
        board.seed = tl_rand();
        board.rand_move(cur);
        cur = !cur;
    }
    return __builtin_popcountll(board.white) > __builtin_popcountll(board.black) ? 1.0 : 0.0;
}

// ─── Nœud UCT (par thread, pas de mutex nécessaire) ─────────────────────────

struct UCTNode {
    Board64_t board;
    bool      isWhiteToPlay;
    Move64_t  moveFromParent;
    UCTNode*  parent;

    std::vector<UCTNode*> children;
    std::vector<Move64_t> untriedMoves;
    int    visits;
    double wins;

    UCTNode(const Board64_t& b, bool w, UCTNode* p, Move64_t m)
        : board(b), isWhiteToPlay(w), moveFromParent(m),
          parent(p), visits(0), wins(0.0)
    { untriedMoves = get_legal_moves(b, w); }

    ~UCTNode() { for (auto* c : children) delete c; }

    bool isTerminal()     const { return board.white_win() || board.black_win(); }
    bool isFullyExpanded()const { return untriedMoves.empty(); }

    UCTNode* bestChild() const {
        UCTNode* best = nullptr; double bestV = -1e18;
        for (auto* c : children) {
            double v = c->wins / (c->visits + 1e-9)
                     + UCT_C * std::sqrt(std::log(visits + 1.0) / (c->visits + 1e-9));
            if (v > bestV) { bestV = v; best = c; }
        }
        return best;
    }

    UCTNode* expand() {
        // Choisit un coup non essayé au hasard (pas de biais)
        int idx = tl_rand() % (uint32_t)untriedMoves.size();
        Move64_t move = untriedMoves[idx];
        untriedMoves.erase(untriedMoves.begin() + idx);
        Board64_t nb = board; nb.apply_move(move, isWhiteToPlay);
        UCTNode* child = new UCTNode(nb, !isWhiteToPlay, this, move);
        children.push_back(child);
        return child;
    }
};

static UCTNode* uct_select(UCTNode* root) {
    UCTNode* node = root;
    while (!node->isTerminal()) {
        if (!node->isFullyExpanded()) return node->expand();
        UCTNode* c = node->bestChild();
        if (!c) break;
        node = c;
    }
    return node;
}

static void uct_backprop(UCTNode* node, double prob) {
    while (node) {
        node->visits++;
        bool byWhite = node->parent ? node->parent->isWhiteToPlay : false;
        node->wins += byWhite ? prob : (1.0 - prob);
        node = node->parent;
    }
}

// ─── Worker : un arbre UCT indépendant par thread ────────────────────────────

struct RootChildStats {
    Move64_t move;
    int      visits;
    double   wins;
};

static std::vector<RootChildStats>
run_uct_tree(const Board64_t& board, bool isWhite,
             const std::chrono::high_resolution_clock::time_point& start,
             std::atomic<bool>& stop, int tid) {
    tl_seed_init((uint32_t)time(nullptr), tid);

    UCTNode* root = new UCTNode(board, isWhite, nullptr, Move64_t());
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

        UCTNode* leaf = uct_select(root);
        if (!leaf) break;

        double result;
        if (leaf->isTerminal())
            result = leaf->board.white_win() ? 1.0 : 0.0;
        else
            result = random_playout(leaf->board, leaf->isWhiteToPlay);

        uct_backprop(leaf, result);
    }

    // Extraire les stats des enfants directs de la racine
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

    // Victoire immédiate
    for (const auto& m : get_legal_moves(board, isWhite)) {
        Board64_t nb = board; nb.apply_move(m, isWhite);
        if (isWhite ? nb.white_win() : nb.black_win())
            return pos_to_coord(m.pi) + "-" + pos_to_coord(m.pf);
    }
    if (get_legal_moves(board, isWhite).empty()) return "resign";

    int nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
    auto start   = std::chrono::high_resolution_clock::now();
    std::atomic<bool> stop{false};

    // Chaque thread produit ses stats de racine
    std::vector<std::vector<RootChildStats>> thread_results(nthreads);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back([&, t]() {
            thread_results[t] = run_uct_tree(board, isWhite, start, stop, t);
        });
    }
    for (auto& th : threads) th.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();

    // Fusion : somme visits+wins par coup (clé = (pi,pf))
    struct Merged { int visits = 0; double wins = 0.0; Move64_t move; };
    std::map<std::pair<uint64_t,uint64_t>, Merged> fusion;

    for (const auto& trvec : thread_results) {
        for (const auto& s : trvec) {
            auto key = std::make_pair(s.move.pi, s.move.pf);
            auto& mg = fusion[key];
            mg.move    = s.move;
            mg.visits += s.visits;
            mg.wins   += s.wins;
        }
    }

    if (fusion.empty()) return "resign";

    // Meilleur coup = plus visité (robustesse)
    Move64_t bestMove = {0ULL, 0ULL};
    int      bestVisits = -1;
    double   bestWins   = 0.0;
    for (const auto& kv : fusion) {
        if (kv.second.visits > bestVisits) {
            bestVisits = kv.second.visits;
            bestWins   = kv.second.wins;
            bestMove   = kv.second.move;
        }
    }

    double wr = bestWins / (bestVisits + 1e-9);
    int totalIters = 0;
    for (const auto& kv : fusion) totalIters += kv.second.visits;

    fprintf(stderr, "UCT: iters=%d visits=%d winrate=%.4f threads=%d time=%.2fs\n",
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

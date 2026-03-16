// mast_bk.cpp - MAST parallèle pour Breakthrough
//
// Parallélisation : Root Parallelization
//   - N threads, chacun avec son propre arbre UCT et sa propre table MAST locale
//   - RNG thread-local → zéro contention
//   - Fusion finale : somme visits/wins des enfants racine + merge tables MAST
//     (les tables MAST fusionnées améliorent le tri initial d'un éventuel
//     second appel, e.g. en mode analyse — pas utilisé ici mais bonne pratique)
//
// Référence : Finnsson & Björnsson, AAAI 2008
// Usage : ./TC_MG_mast BOARD PLAYER [time_s]

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <map>
#include "bkbb64.h"

#define WHITE     0
#define BLACK     1
#define UCT_C     0.7
#define MAST_TEMP 0.3

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

// ─── Table MAST (locale par thread) ─────────────────────────────────────────

struct MastStats {
    double wins   = 0.0;
    int    visits = 0;
    double score() const { return visits > 0 ? wins / visits : 0.5; }
};
using MastTable = std::unordered_map<uint64_t, MastStats>;

static inline uint64_t mast_key(const Move64_t& m, bool isWhite) {
    int pi = __builtin_ctzll(m.pi);
    int pf = __builtin_ctzll(m.pf);
    return (uint64_t)pi | ((uint64_t)pf << 8) | ((uint64_t)(isWhite ? 1 : 0) << 16);
}

static void mast_update(MastTable& tbl, const Move64_t& m,
                         bool byWhite, double whiteWinProb) {
    auto& s = tbl[mast_key(m, byWhite)];
    s.visits++;
    s.wins += byWhite ? whiteWinProb : (1.0 - whiteWinProb);
}

// ─── Utilitaires ────────────────────────────────────────────────────────────

static std::vector<Move64_t> get_legal_moves(const Board64_t& board, bool isWhite) {
    uint64_t l = isWhite ? board.white_left() : board.black_left();
    uint64_t f = isWhite ? board.white_forward() : board.black_forward();
    uint64_t r = isWhite ? board.white_right() : board.black_right();
    return Lfr_t(l, f, r).get_moves(isWhite);
}

// Sélection Gibbs depuis la table MAST locale
static Move64_t mast_select(const std::vector<Move64_t>& moves,
                              bool isWhite, const MastTable& tbl) {
    if (moves.size() == 1) return moves[0];
    std::vector<double> w(moves.size());
    double sum = 0.0;
    for (size_t i = 0; i < moves.size(); i++) {
        auto it = tbl.find(mast_key(moves[i], isWhite));
        double sc = (it != tbl.end()) ? it->second.score() : 0.5;
        w[i] = std::exp(sc / MAST_TEMP);
        sum += w[i];
    }
    double r = (double)(tl_rand() & 0xFFFF) / 65535.0 * sum;
    double cum = 0.0;
    for (size_t i = 0; i < moves.size(); i++) {
        cum += w[i];
        if (r <= cum) return moves[i];
    }
    return moves.back();
}

// Playout guidé MAST (utilise table locale)
static double mast_playout(Board64_t board, bool cur,
                             std::vector<std::pair<Move64_t,bool>>& played,
                             const MastTable& tbl) {
    for (int d = 0; d < 200; d++) {
        if (board.white_win()) return 1.0;
        if (board.black_win()) return 0.0;
        std::vector<Move64_t> moves = get_legal_moves(board, cur);
        if (moves.empty()) return cur ? 0.0 : 1.0;
        Move64_t chosen = mast_select(moves, cur, tbl);
        played.push_back({chosen, cur});
        board.apply_move(chosen, cur);
        cur = !cur;
    }
    return __builtin_popcountll(board.white) >= __builtin_popcountll(board.black) ? 1.0 : 0.0;
}

// ─── Nœud UCT/MAST ──────────────────────────────────────────────────────────

struct MASTNode {
    Board64_t board;
    bool      isWhiteToPlay;
    Move64_t  moveFromParent;
    MASTNode* parent;

    std::vector<MASTNode*> children;
    std::vector<Move64_t>  untriedMoves;
    int    visits;
    double wins;

    MASTNode(const Board64_t& b, bool w, MASTNode* p, Move64_t m,
             const MastTable& tbl)
        : board(b), isWhiteToPlay(w), moveFromParent(m),
          parent(p), visits(0), wins(0.0)
    {
        untriedMoves = get_legal_moves(b, w);
        // Tri par score MAST décroissant (pop_back = meilleur)
        std::sort(untriedMoves.begin(), untriedMoves.end(),
            [&](const Move64_t& a, const Move64_t& b_) {
                auto ia = tbl.find(mast_key(a, w));
                auto ib = tbl.find(mast_key(b_, w));
                double sa = ia != tbl.end() ? ia->second.score() : 0.5;
                double sb = ib != tbl.end() ? ib->second.score() : 0.5;
                return sa < sb;
            });
    }

    ~MASTNode() { for (auto* c : children) delete c; }

    bool isTerminal() const { return board.white_win() || board.black_win(); }

    MASTNode* bestChild() const {
        MASTNode* best = nullptr; double bestV = -1e18;
        for (auto* c : children) {
            double v = c->wins / (c->visits + 1e-9)
                     + UCT_C * std::sqrt(std::log(visits + 1.0) / (c->visits + 1e-9));
            if (v > bestV) { bestV = v; best = c; }
        }
        return best;
    }

    MASTNode* expand(const MastTable& tbl) {
        Move64_t move = untriedMoves.back();
        untriedMoves.pop_back();
        Board64_t nb = board; nb.apply_move(move, isWhiteToPlay);
        MASTNode* child = new MASTNode(nb, !isWhiteToPlay, this, move, tbl);
        children.push_back(child);
        return child;
    }
};

static MASTNode* mast_tree_select(MASTNode* root, const MastTable& tbl) {
    MASTNode* node = root;
    while (!node->isTerminal()) {
        if (!node->untriedMoves.empty()) return node->expand(tbl);
        MASTNode* c = node->bestChild();
        if (!c) break;
        node = c;
    }
    return node;
}

static void mast_backprop(MASTNode* node, double prob,
                            const std::vector<std::pair<Move64_t,bool>>& played,
                            MastTable& tbl) {
    for (const auto& p : played)
        mast_update(tbl, p.first, p.second, prob);
    while (node) {
        node->visits++;
        bool byWhite = node->parent ? node->parent->isWhiteToPlay : false;
        node->wins += byWhite ? prob : (1.0 - prob);
        node = node->parent;
    }
}

// ─── Stats de racine (pour fusion) ──────────────────────────────────────────

struct RootChildStats {
    Move64_t move;
    int      visits;
    double   wins;
};

// ─── Worker : un arbre MAST indépendant par thread ────────────────────────────

static std::vector<RootChildStats>
run_mast_tree(const Board64_t& board, bool isWhite,
              const std::chrono::high_resolution_clock::time_point& start,
              std::atomic<bool>& stop, int tid) {
    tl_seed_init((uint32_t)time(nullptr), tid);

    MastTable tbl;   // table MAST locale à ce thread
    MASTNode* root = new MASTNode(board, isWhite, nullptr, Move64_t(), tbl);
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

        MASTNode* leaf = mast_tree_select(root, tbl);
        if (!leaf) break;

        double result;
        std::vector<std::pair<Move64_t,bool>> played;

        if (leaf->isTerminal())
            result = leaf->board.white_win() ? 1.0 : 0.0;
        else
            result = mast_playout(leaf->board, leaf->isWhiteToPlay, played, tbl);

        mast_backprop(leaf, result, played, tbl);
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
            thread_results[t] = run_mast_tree(board, isWhite, start, stop, t);
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

    fprintf(stderr, "MAST: iters=%d visits=%d winrate=%.4f threads=%d time=%.2fs\n",
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

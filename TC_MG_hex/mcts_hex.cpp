#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include "hexbb.h"

#define BLUE_PLAYER   0
#define RED_PLAYER    1
#define EXPLORATION_C 0.7
#define VIRTUAL_LOSS  3
#define SHALLOW_DEPTH 2
#define PRIOR_WEIGHT  2.0

double g_time_limit_s = 10.0;
static std::atomic<int>      g_total_iters{0};
static std::atomic<uint32_t> g_rand_counter{0};

const int SCORE_WIN  =  100000;
const int SCORE_LOSE = -100000;

static int priority_hex_move(const HexMove& m, bool isBlue) {
    int r = m.pos / HEX_SIZE;
    int c = m.pos % HEX_SIZE;
    int dr = r - 5, dc = c - 5;
    int dist_center = dr * dr + dc * dc;
    int axis_advance = isBlue ? (5 - abs(r - 5)) : (5 - abs(c - 5));
    return dist_center - axis_advance * 3;
}

static void sort_hex_moves(std::vector<HexMove>& moves, bool isBlue) {
    std::stable_sort(moves.begin(), moves.end(),
        [isBlue](const HexMove& a, const HexMove& b) {
            return priority_hex_move(a, isBlue) > priority_hex_move(b, isBlue);
        });
}

static bool is_immediate_win(const HexBoard& board, const HexMove& m, bool isBlue) {
    HexBoard next = board;
    next.apply_move(m, isBlue);
    return next.is_win(isBlue);
}

static bool find_immediate_winning_move(const HexBoard& board, bool isBlue, HexMove& winMove) {
    for (const auto& m : board.get_legal_moves()) {
        if (is_immediate_win(board, m, isBlue)) {
            winMove = m;
            return true;
        }
    }
    return false;
}

static std::vector<HexMove> get_safe_moves(const HexBoard& board, bool isBlue) {
    std::vector<HexMove> safe;
    for (const auto& m : board.get_legal_moves()) {
        HexBoard next = board;
        next.apply_move(m, isBlue);
        if (next.is_win(isBlue)) {
            safe.push_back(m);
            continue;
        }
        HexMove oppWin;
        if (!find_immediate_winning_move(next, !isBlue, oppWin))
            safe.push_back(m);
    }
    return safe;
}

// Nombre max de coups évalués par niveau dans le shallow AB
// Réduit le branching factor pour accélérer l'évaluation des feuilles
static const int AB_TOP_K = 20;

// ─── Ponts hexagonaux (virtual connections) ────────────────────────────────
// Un pont relie deux pions alliés à distance 2 via deux cases vides communes.
// L'adversaire ne peut bloquer qu'une des deux → la connexion est garantie.
// Offsets : (dr_partenaire, dc_partenaire, dr_commun1, dc_commun1, dr_commun2, dc_commun2)
static const int BRIDGE_PATTERNS[6][6] = {
    {-2, +1, -1,  0, -1, +1},
    {-1, +2, -1, +1,  0, +1},
    {+1, +1,  0, +1, +1,  0},
    {+2, -1, +1,  0, +1, -1},
    {+1, -2, +1, -1,  0, -1},
    {-1, -1,  0, -1, -1,  0},
};

// BFS 0-1 avec ponts pour Blue (Nord→Sud)
// Coût 0 = pion Blue ou pont, Coût 1 = vide, bloqué = pion Red
static int shortest_path_bridge_blue(const HexBoard& board) {
    int dist[HEX_CELLS];
    int q[HEX_CELLS * 6];
    for (int i = 0; i < HEX_CELLS; i++) dist[i] = INT_MAX;
    int head = HEX_CELLS * 3, tail = HEX_CELLS * 3;
    uint64_t occ_lo = board.blue_lo | board.red_lo;
    uint64_t occ_hi = board.blue_hi | board.red_hi;

    for (int c = 0; c < HEX_SIZE; c++) {
        int idx = c;
        if (bit_get(board.red_lo, board.red_hi, idx)) continue;
        int cost = bit_get(board.blue_lo, board.blue_hi, idx) ? 0 : 1;
        if (cost < dist[idx]) {
            dist[idx] = cost;
            if (cost == 0) q[--head] = idx;
            else           q[tail++] = idx;
        }
    }

    while (head < tail) {
        int cur = q[head++];
        if (dist[cur] == INT_MAX) continue;
        int r = cur / HEX_SIZE, c = cur % HEX_SIZE;
        if (r == HEX_SIZE - 1) return dist[cur];

        // Voisins hex standards
        for (int d = 0; d < 6; d++) {
            int nr = r + HEX_DR[d], nc = c + HEX_DC[d];
            if (nr < 0 || nr >= HEX_SIZE || nc < 0 || nc >= HEX_SIZE) continue;
            int nidx = nr * HEX_SIZE + nc;
            if (bit_get(board.red_lo, board.red_hi, nidx)) continue;
            int cost = bit_get(board.blue_lo, board.blue_hi, nidx) ? 0 : 1;
            int nd = dist[cur] + cost;
            if (nd < dist[nidx]) {
                dist[nidx] = nd;
                if (cost == 0) { head--; q[head] = nidx; }
                else           { q[tail++] = nidx; }
            }
        }

        // Ponts : si la cellule courante est Blue, vérifier les partenaires de pont
        if (bit_get(board.blue_lo, board.blue_hi, cur)) {
            for (int b = 0; b < 6; b++) {
                int pr = r + BRIDGE_PATTERNS[b][0], pc = c + BRIDGE_PATTERNS[b][1];
                if (pr < 0 || pr >= HEX_SIZE || pc < 0 || pc >= HEX_SIZE) continue;
                int pidx = pr * HEX_SIZE + pc;
                if (!bit_get(board.blue_lo, board.blue_hi, pidx)) continue;
                // Vérifier que les deux cases communes sont vides
                int s1r = r + BRIDGE_PATTERNS[b][2], s1c = c + BRIDGE_PATTERNS[b][3];
                int s2r = r + BRIDGE_PATTERNS[b][4], s2c = c + BRIDGE_PATTERNS[b][5];
                if (s1r < 0 || s1r >= HEX_SIZE || s1c < 0 || s1c >= HEX_SIZE) continue;
                if (s2r < 0 || s2r >= HEX_SIZE || s2c < 0 || s2c >= HEX_SIZE) continue;
                int s1 = s1r * HEX_SIZE + s1c, s2 = s2r * HEX_SIZE + s2c;
                if (!bit_get(occ_lo, occ_hi, s1) && !bit_get(occ_lo, occ_hi, s2)) {
                    if (dist[cur] < dist[pidx]) {
                        dist[pidx] = dist[cur];
                        head--; q[head] = pidx;
                    }
                }
            }
        }
    }
    return INT_MAX;
}

// BFS 0-1 avec ponts pour Red (Ouest→Est)
static int shortest_path_bridge_red(const HexBoard& board) {
    int dist[HEX_CELLS];
    int q[HEX_CELLS * 6];
    for (int i = 0; i < HEX_CELLS; i++) dist[i] = INT_MAX;
    int head = HEX_CELLS * 3, tail = HEX_CELLS * 3;
    uint64_t occ_lo = board.blue_lo | board.red_lo;
    uint64_t occ_hi = board.blue_hi | board.red_hi;

    for (int r = 0; r < HEX_SIZE; r++) {
        int idx = r * HEX_SIZE;
        if (bit_get(board.blue_lo, board.blue_hi, idx)) continue;
        int cost = bit_get(board.red_lo, board.red_hi, idx) ? 0 : 1;
        if (cost < dist[idx]) {
            dist[idx] = cost;
            if (cost == 0) q[--head] = idx;
            else           q[tail++] = idx;
        }
    }

    while (head < tail) {
        int cur = q[head++];
        if (dist[cur] == INT_MAX) continue;
        int r = cur / HEX_SIZE, c = cur % HEX_SIZE;
        if (c == HEX_SIZE - 1) return dist[cur];

        for (int d = 0; d < 6; d++) {
            int nr = r + HEX_DR[d], nc = c + HEX_DC[d];
            if (nr < 0 || nr >= HEX_SIZE || nc < 0 || nc >= HEX_SIZE) continue;
            int nidx = nr * HEX_SIZE + nc;
            if (bit_get(board.blue_lo, board.blue_hi, nidx)) continue;
            int cost = bit_get(board.red_lo, board.red_hi, nidx) ? 0 : 1;
            int nd = dist[cur] + cost;
            if (nd < dist[nidx]) {
                dist[nidx] = nd;
                if (cost == 0) { head--; q[head] = nidx; }
                else           { q[tail++] = nidx; }
            }
        }

        if (bit_get(board.red_lo, board.red_hi, cur)) {
            for (int b = 0; b < 6; b++) {
                int pr = r + BRIDGE_PATTERNS[b][0], pc = c + BRIDGE_PATTERNS[b][1];
                if (pr < 0 || pr >= HEX_SIZE || pc < 0 || pc >= HEX_SIZE) continue;
                int pidx = pr * HEX_SIZE + pc;
                if (!bit_get(board.red_lo, board.red_hi, pidx)) continue;
                int s1r = r + BRIDGE_PATTERNS[b][2], s1c = c + BRIDGE_PATTERNS[b][3];
                int s2r = r + BRIDGE_PATTERNS[b][4], s2c = c + BRIDGE_PATTERNS[b][5];
                if (s1r < 0 || s1r >= HEX_SIZE || s1c < 0 || s1c >= HEX_SIZE) continue;
                if (s2r < 0 || s2r >= HEX_SIZE || s2c < 0 || s2c >= HEX_SIZE) continue;
                int s1 = s1r * HEX_SIZE + s1c, s2 = s2r * HEX_SIZE + s2c;
                if (!bit_get(occ_lo, occ_hi, s1) && !bit_get(occ_lo, occ_hi, s2)) {
                    if (dist[cur] < dist[pidx]) {
                        dist[pidx] = dist[cur];
                        head--; q[head] = pidx;
                    }
                }
            }
        }
    }
    return INT_MAX;
}

// Eval améliorée avec ponts + tempo
static int eval_bridge(const HexBoard& board, bool isBlue) {
    if (board.blue_win()) return isBlue ?  SCORE_WIN : SCORE_LOSE;
    if (board.red_win())  return isBlue ? SCORE_LOSE : SCORE_WIN;

    int pb = shortest_path_bridge_blue(board);
    int pr = shortest_path_bridge_red(board);

    int score;
    if (pb == INT_MAX && pr == INT_MAX) score = 0;
    else if (pb == INT_MAX) score = -SCORE_WIN;
    else if (pr == INT_MAX) score =  SCORE_WIN;
    else score = pr - pb;

    // Bonus tempo pour le joueur au trait
    score += 1;

    return isBlue ? score : -score;
}

// Shallow alpha-beta negamax pour évaluation des feuilles MCTS
// Retourne un score du point de vue du joueur courant (positif = bien)
static int shallow_ab(const HexBoard& board, int depth, int alpha, int beta, bool isBlue) {
    if (board.blue_win()) return isBlue ?  SCORE_WIN + depth : SCORE_LOSE - depth;
    if (board.red_win())  return isBlue ? SCORE_LOSE - depth : SCORE_WIN + depth;
    if (depth == 0) {
        int score = board.eval(isBlue);
        if (score > SCORE_LOSE + 1000 && score < SCORE_WIN - 1000)
            score += 1; // tempo
        return score;
    }

    std::vector<HexMove> moves = board.get_legal_moves();
    if (moves.empty()) return board.eval(isBlue);

    // Tri des coups : meilleurs en premier pour maximiser le pruning
    sort_hex_moves(moves, isBlue);
    std::reverse(moves.begin(), moves.end());

    // Élagage : ne garder que les top-K coups les plus prometteurs
    int limit = std::min((int)moves.size(), AB_TOP_K);

    int best = SCORE_LOSE;
    for (int i = 0; i < limit; i++) {
        HexBoard next = board;
        next.apply_move(moves[i], isBlue);
        int val = -shallow_ab(next, depth - 1, -beta, -alpha, !isBlue);
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
    }
    return best;
}

// Évalue une feuille MCTS : retourne probabilité de victoire Blue dans [0, 1]
// Hybride : shallow alpha-beta (signal stratégique) + 1 playout (diversité)
static double evaluate_leaf(const HexBoard& board, bool isBlueToPlay) {
    if (board.blue_win()) return 1.0;
    if (board.red_win())  return 0.0;

    // Composante stratégique : shallow alpha-beta
    int score = shallow_ab(board, SHALLOW_DEPTH, SCORE_LOSE, SCORE_WIN, isBlueToPlay);

    double ab_prob;
    if (score >= SCORE_WIN - 100) ab_prob = isBlueToPlay ? 1.0 : 0.0;
    else if (score <= SCORE_LOSE + 100) ab_prob = isBlueToPlay ? 0.0 : 1.0;
    else {
        double p = 1.0 / (1.0 + exp(-score / 5.0));
        ab_prob = isBlueToPlay ? p : (1.0 - p);
    }

    // Composante aléatoire : 1 playout pour casser le biais déterministe
    HexBoard b = board;
    b.seed = board.seed ^ g_rand_counter.fetch_add(1, std::memory_order_relaxed);
    double playout_prob = b.random_playout(isBlueToPlay) ? 1.0 : 0.0;

    // Mélange : 70% AB stratégique + 30% playout
    return 0.7 * ab_prob + 0.3 * playout_prob;
}

struct MCTSNode {
    HexBoard board;
    bool isBlueToPlay;
    HexMove moveFromParent;
    MCTSNode* parent;

    std::mutex mtx;
    std::vector<MCTSNode*> children;
    std::vector<HexMove> untriedMoves;

    std::atomic<int> visits;
    double wins; // race acceptée sur x86-64
    double evalPrior; // prior heuristique [0,1] du point de vue du parent

    MCTSNode(const HexBoard& b, bool blueToPlay, MCTSNode* par, HexMove mov)
        : board(b), isBlueToPlay(blueToPlay), moveFromParent(mov),
          parent(par), visits(0), wins(0.0), evalPrior(0.5)
    {
        untriedMoves = board.get_legal_moves();
        // Tri décroissant : pop_back() prend le meilleur coup en premier
        sort_hex_moves(untriedMoves, isBlueToPlay);
    }

    ~MCTSNode() {
        for (auto* c : children) delete c;
    }

    bool isTerminal() const {
        return board.blue_win() || board.red_win();
    }

    // Sélection UCB1
    MCTSNode* selectChild() {
        if (isTerminal()) return nullptr;
        std::lock_guard<std::mutex> lock(mtx);
        if (!untriedMoves.empty()) return nullptr;

        MCTSNode* best = nullptr;
        double bestVal = -1e9;
        int pv = visits.load();

        for (auto* child : children) {
            int cv = child->visits.load();
            double exploit = child->wins / (cv + 1e-6);
            double explore = EXPLORATION_C * std::sqrt(std::log(pv + 1.0) / (cv + 1e-6));
            // Progressive bias : prior heuristique qui décroît avec les visites
            double bias = PRIOR_WEIGHT * child->evalPrior / (cv + 1);
            double ucb = exploit + explore + bias;
            if (ucb > bestVal) { bestVal = ucb; best = child; }
        }
        return best;
    }

    // Expansion thread-safe
    MCTSNode* tryExpand() {
        std::lock_guard<std::mutex> lock(mtx);
        if (untriedMoves.empty()) return nullptr;

        HexMove move = untriedMoves.back();
        untriedMoves.pop_back();

        HexBoard next = board;
        next.apply_move(move, isBlueToPlay);

        MCTSNode* child = new MCTSNode(next, !isBlueToPlay, this, move);
        // Prior heuristique : eval avec ponts du point de vue du parent
        int rawScore = eval_bridge(next, isBlueToPlay);
        child->evalPrior = 1.0 / (1.0 + exp(-rawScore / 5.0));
        children.push_back(child);
        return child;
    }
};

static void backpropagate(MCTSNode* node, double blueWinProb, bool rootIsBlue) {
    while (node != nullptr) {
        node->visits.fetch_add(1);
        bool rewardBlue = (node->parent == nullptr) ? rootIsBlue : !node->isBlueToPlay;
        node->wins += rewardBlue ? blueWinProb : (1.0 - blueWinProb);
        node = node->parent;
    }
}

static std::string get_best_move(HexBoard& board, bool isBlue) {
    std::vector<HexMove> moves = board.get_legal_moves();
    if (moves.empty()) return "resign";

    // 0) Premier coup : toujours jouer au centre (F6 = case 60, optimal en Hex)
    if (moves.size() == HEX_CELLS) {
        return hex_pos_to_str(5 * HEX_SIZE + 5); // F6
    }

    // 1) Coup gagnant immédiat
    HexMove forcedWin;
    if (find_immediate_winning_move(board, isBlue, forcedWin))
        return forcedWin.to_str();

    // 2) Filtre tactique : éviter les coups qui donnent la victoire à l'adversaire
    std::vector<HexMove> safeMoves = get_safe_moves(board, isBlue);

    MCTSNode* root = new MCTSNode(board, isBlue, nullptr, HexMove(-1));

    // Appliquer le filtre tactique si utile
    if (!safeMoves.empty() && safeMoves.size() < root->untriedMoves.size()) {
        sort_hex_moves(safeMoves, isBlue);
        root->untriedMoves = safeMoves;
    }

    if (root->untriedMoves.empty()) {
        delete root;
        return moves[0].to_str();
    }

    auto start = std::chrono::high_resolution_clock::now();
    g_total_iters = 0;

    auto worker = [&](MCTSNode* root_ptr) {
        int iter = 0;
        while (true) {
            if (iter % 64 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(now - start).count() >= g_time_limit_s)
                    break;
            }
            iter++;
            g_total_iters++;

            // Sélection avec virtual loss
            MCTSNode* node = root_ptr;
            std::vector<MCTSNode*> vlPath;
            MCTSNode* child;
            while ((child = node->selectChild()) != nullptr) {
                node->visits.fetch_add(VIRTUAL_LOSS);
                vlPath.push_back(node);
                node = child;
            }

            // Expansion
            if (!node->isTerminal()) {
                MCTSNode* expanded = node->tryExpand();
                if (expanded) node = expanded;
            }

            // Évaluation
            double blueWinProb;
            if (node->isTerminal()) {
                blueWinProb = node->board.blue_win() ? 1.0 : 0.0;
            } else {
                blueWinProb = evaluate_leaf(node->board, node->isBlueToPlay);
            }

            // Backpropagation
            backpropagate(node, blueWinProb, isBlue);

            // Retrait virtual loss
            for (auto* n : vlPath)
                n->visits.fetch_sub(VIRTUAL_LOSS);
        }
    };

    int nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; t++)
        threads.emplace_back(worker, root);
    for (auto& th : threads) th.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();

    if (root->children.empty()) {
        delete root;
        return moves[0].to_str();
    }

    // Meilleur enfant = le plus visité
    MCTSNode* bestNode = nullptr;
    int bestVisits = -1;
    for (auto* ch : root->children) {
        int v = ch->visits.load();
        if (v > bestVisits) { bestVisits = v; bestNode = ch; }
    }

    double bestWinRate = (bestVisits > 0) ? (bestNode->wins / bestVisits) : 0.0;
    HexMove bestMove = bestNode->moveFromParent;
    delete root;

    fprintf(stderr, "ITERS:%d VISITS:%d WINRATE:%.4f TIME:%.2f\n",
            g_total_iters.load(), bestVisits, bestWinRate, elapsed);

    return bestMove.to_str();
}

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));
    if (argc < 3) {
        fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]);
        return 1;
    }
    if (argc >= 4) {
        g_time_limit_s = atof(argv[3]);
        if (g_time_limit_s <= 0) g_time_limit_s = 1.3;
    }

    HexBoard board(argv[1]);
    board.seed = (uint32_t)time(nullptr);
    std::string playerStr(argv[2]);

    bool isBlue = (playerStr == "O");
    printf("%s\n", get_best_move(board, isBlue).c_str());
    return 0;
}

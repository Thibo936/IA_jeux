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
#include <map>
#include "bkbb64.h"

#define WHITE 0
#define BLACK 1
#define EXPLORATION_C 0.6
#define SHALLOW_DEPTH 3
#define VIRTUAL_LOSS 3

// Limite de temps par coup (modifiable via argv[3]).
double g_time_limit_s = 1.3;
// Compteur global d'iterations MCTS (utile pour le debug/perf).
static std::atomic<int> g_total_iters{0};

const int SCORE_WIN = 100000;
const int SCORE_LOSE = -100000;
const int MATERIAL_WEIGHT = 10;
const int ADVANCE_WEIGHT = 5;
const int SAFE_WEIGHT = 3;

// Genere tous les coups legaux pour le joueur au trait via les bitboards L/F/R.
// L/F/R = Left/Forward/Right : trois directions de mouvement dans Breakthrough.
// - Left : diagonal avant-gauche (capture possible)
// - Forward : case directement devant (marche simple)
// - Right : diagonal avant-droite (capture possible)
// Cette fonction decode les bitboards et retourne tous les coups possibles.
static std::vector<Move64_t> get_legal_moves(const Board64_t& board, bool isWhite) {
    uint64_t l = isWhite ? board.white_left() : board.black_left();
    uint64_t f = isWhite ? board.white_forward() : board.black_forward();
    uint64_t r = isWhite ? board.white_right() : board.black_right();
    return Lfr_t(l, f, r).get_moves(isWhite);
}

// Detecte une position terminale (victoire blanche ou noire).
static bool win_position(const Board64_t& board) {
    return board.white_win() || board.black_win();
}

// Heuristique de tri des coups:
// - captures priorisees
// - puis progression vers la ligne de promotion
// Le tri est croissant car on depile ensuite avec pop_back().
static int priority_move(const Move64_t& move, uint64_t enemy, bool isWhite) {
    bool isCapture = (move.pf & enemy) != 0;
    int row = __builtin_ctzll(move.pf) / 8;
    int advance = isWhite ? (7 - row) : row;
    return (isCapture ? 100 : 0) + advance;
}

// Tri stable pour conserver un ordre deterministe entre coups de meme priorite.
static void sort_moves_by_priority(std::vector<Move64_t>& moves, uint64_t enemy, bool isWhite) {
    std::stable_sort(moves.begin(), moves.end(),
        [enemy, isWhite](const Move64_t& lhs, const Move64_t& rhs) {
            return priority_move(lhs, enemy, isWhite) < priority_move(rhs, enemy, isWhite);
        });
}

// Convention: score terminal converti en probabilite de victoire blanche.
static double terminal_white_win_proba(const Board64_t& board) {
    return board.white_win() ? 1.0 : 0.0;
}

// Simule un coup et teste la victoire immediate du joueur courant.
static bool is_immediate_win(const Board64_t& board, const Move64_t& move, bool isWhite) {
    Board64_t nextBoard = board;
    nextBoard.apply_move(move, isWhite);
    return isWhite ? nextBoard.white_win() : nextBoard.black_win();
}

// Cherche un coup qui gagne instantanement (si present).
// Important pour la strategie: si un mat existe, on le joue toujours.
// La fonction iterate sur tous les coups legaux et teste si l'un d'eux atteint la promotion directe.
static bool find_immediate_winning_move(const Board64_t& board, bool isWhite, Move64_t& winningMove) {
    std::vector<Move64_t> moves = get_legal_moves(board, isWhite);
    for (const auto& move : moves) {
        if (is_immediate_win(board, move, isWhite)) {
            winningMove = move;
            return true;
        }
    }
    return false;
}

// Filtre de securite tactique:
// on retire les coups qui laissent une victoire immediate a l'adversaire.
static std::vector<Move64_t> get_safe_moves(const Board64_t& board, bool isWhite) {
    std::vector<Move64_t> safeMoves;
    std::vector<Move64_t> moves = get_legal_moves(board, isWhite);

    for (const auto& move : moves) {
        Board64_t nextBoard = board;
        nextBoard.apply_move(move, isWhite);

        if (win_position(nextBoard)) {
            safeMoves.push_back(move);
            continue;
        }

        Move64_t opponentWin;
        if (!find_immediate_winning_move(nextBoard, !isWhite, opponentWin)) {
            safeMoves.push_back(move);
        }
    }

    return safeMoves;
}

// Évaluation améliorée : positif = blanc avantage
int evaluate_board(const Board64_t& board) {
    // Bornes terminales pour stabiliser les comparaisons alpha-beta.
    if (board.white_win()) return SCORE_WIN;
    if (board.black_win()) return SCORE_LOSE;

    int score = 0;
    int w_most_advanced = 7;
    int b_most_advanced = 0;

    uint64_t w = board.white;
    while (w) {
        // Extraction du pion le moins significatif (bit scan).
        int idx = __builtin_ctzll(w);
        int row = idx / 8;

        // Materiel + progression vers promotion.
        score += MATERIAL_WEIGHT;
        score += (7 - row) * ADVANCE_WEIGHT;

        // Bonus exponentiel près de la promotion
        if (row <= 2) score += (3 - row) * 20;

        if (row < w_most_advanced) w_most_advanced = row;

        w &= (w - 1);
    }

    uint64_t b = board.black;
    while (b) {
        int idx = __builtin_ctzll(b);
        int row = idx / 8;

        // Symetrique noir: soustraction pour rester en perspective blanche.
        score -= MATERIAL_WEIGHT;
        score -= row * ADVANCE_WEIGHT;

        if (row >= 5) score -= (row - 4) * 20;

        if (row > b_most_advanced) b_most_advanced = row;

        b &= (b - 1);
    }

    // Bonus pièce la plus avancée
    score += (7 - w_most_advanced) * 8;
    score -= b_most_advanced * 8;

    // Pieces protegees : bonus si une piece est adjacente a une autre piece.
    // __builtin_popcountll = Population Count (GCC intrinsic) : compte les bits a 1.
    // Exemple: 0b1010 -> 2 bits a 1 -> retourne 2.
    // Shift 7/9 donnent les diagonales avant (pieces alliees potentiellement protectrices).
    uint64_t w_protected = board.white & ((board.white << 7) | (board.white << 9));
    score += __builtin_popcountll(w_protected) * SAFE_WEIGHT;

    uint64_t b_protected = board.black & ((board.black >> 7) | (board.black >> 9));
    score -= __builtin_popcountll(b_protected) * SAFE_WEIGHT;

    return score;
}

// Shallow alpha-beta : recherche peu profonde (SHALLOW_DEPTH=3) pour evaluer les feuilles MCTS.
// Algorithme: minimax avec pruning alpha-beta pour eliminer les branches non pertinentes.
// - Blanc maximise ("white_win" retourne SCORE_WIN)
// - Noir minimise ("black_win" retourne SCORE_LOSE)
// L'idee: voir 3 coups d'avance + captures en priorite = bon compromis rapidite/precision.
static int shallow_ab(const Board64_t& board, int depth, int alpha, int beta, bool isWhite) {
    // Conditions d'arret: terminal ou profondeur max.
    if (board.white_win()) return SCORE_WIN;
    if (board.black_win()) return SCORE_LOSE;
    if (depth == 0) return evaluate_board(board);

    std::vector<Move64_t> moves = get_legal_moves(board, isWhite);
    if (moves.empty()) return isWhite ? SCORE_LOSE : SCORE_WIN;

    // Captures d'abord pour augmenter les chances de coupure alpha-beta.
    uint64_t enemy = isWhite ? board.black : board.white;
    std::partition(moves.begin(), moves.end(), [enemy](const Move64_t& m) {
        return (m.pf & enemy) != 0;
    });

    if (isWhite) {
        // Max player (blanc): maximise le score.
        int best = SCORE_LOSE;
        for (const auto& m : moves) {
            Board64_t next = board;
            next.apply_move(m, true);
            int val = shallow_ab(next, depth - 1, alpha, beta, false);
            if (val > best) best = val;
            if (val > alpha) alpha = val;
            if (beta <= alpha) break;
        }
        return best;
    } else {
        // Min player (noir): minimise le score.
        int best = SCORE_WIN;
        for (const auto& m : moves) {
            Board64_t next = board;
            next.apply_move(m, false);
            int val = shallow_ab(next, depth - 1, alpha, beta, true);
            if (val < best) best = val;
            if (val < beta) beta = val;
            if (beta <= alpha) break;
        }
        return best;
    }
}

// Évalue une feuille : retourne probabilité de victoire BLANC dans [0, 1]
static double evaluation_leaf(const Board64_t& board, bool isWhiteToPlay) {
    // Gestion explicite des feuilles terminales.
    if (board.white_win()) return 1.0;
    if (board.black_win()) return 0.0;

    int score = shallow_ab(board, SHALLOW_DEPTH, SCORE_LOSE, SCORE_WIN, isWhiteToPlay);

    if (score >= SCORE_WIN - 100) return 1.0;
    if (score <= SCORE_LOSE + 100) return 0.0;

    // Conversion score -> proba avec sigmoide pour lisser les valeurs intermediaires.
    return 1.0 / (1.0 + exp(-score / 50.0));
}

struct MCTSNode {
    // Etat de jeu represente par le noeud.
    Board64_t board;
    // Joueur a jouer dans cet etat.
    bool isWhiteToPlay;
    // Coup qui a mene a cet etat depuis parent.
    Move64_t moveFromParent;
    MCTSNode* parent;

    // Mutex local: protege les structures modifiees concurremment.
    std::mutex mtx;  // protege children + untriedMoves
    std::vector<MCTSNode*> children;
    // Coups encore non explores depuis ce noeud.
    std::vector<Move64_t> untriedMoves;

    // Nombre de passages dans le noeud.
    std::atomic<int> visits;
    // Somme des recompenses (du point de vue defini dans backpropagate).
    double wins;  // data race acceptee (writes alignes sur x86-64)

    MCTSNode(const Board64_t& b, bool whiteToPlay, MCTSNode* par, Move64_t mov)
        : board(b), isWhiteToPlay(whiteToPlay), moveFromParent(mov),
          parent(par), visits(0), wins(0.0)
    {
        untriedMoves = get_legal_moves(board, isWhiteToPlay);

        // Tri par priorité croissante (pop_back prend les meilleurs en premier)
        // Priorité : captures + avancement > captures > avancement > reste
        uint64_t enemy = isWhiteToPlay ? board.black : board.white;
        sort_moves_by_priority(untriedMoves, enemy, isWhiteToPlay);
    }

    // Destructeur (~) : appelé automatiquement lorsque l'objet est detruit.
    // Ici, on libere recursivement la memoire allouee pour tous les enfants.
    // Important pour eviter les fuites (l'arbre est cree dynamiquement avec new).
    ~MCTSNode() {
        for (auto* child : children)
            delete child;
    }

    bool isTerminal() const {
        return win_position(board);
    }

    // Selection UCB:
    // - nullptr si noeud non pleinement expanse
    // - sinon meilleur enfant selon exploitation + exploration
    MCTSNode* selectChild() {
        if (isTerminal()) return nullptr;
        std::lock_guard<std::mutex> lock(mtx);
        if (!untriedMoves.empty()) return nullptr;

        MCTSNode* best = nullptr;
        double bestValue = -1e9;
        int parentVisits = visits.load();

        for (auto* child : children) {
            int cv = child->visits.load();
            // UCB1 classique avec epsilon pour eviter division par zero.
            double exploitation = child->wins / (cv + 1e-6);
            double exploration = EXPLORATION_C * std::sqrt(std::log(parentVisits + 1) / (cv + 1e-6));
            double ucb = exploitation + exploration;

            if (ucb > bestValue) {
                bestValue = ucb;
                best = child;
            }
        }
        return best;
    }

    // Expansion thread-safe: consomme un coup non essaye et cree un enfant.
    MCTSNode* tryExpand() {
        std::lock_guard<std::mutex> lock(mtx);
        if (untriedMoves.empty()) return nullptr;

        Move64_t move = untriedMoves.back();
        untriedMoves.pop_back();

        Board64_t nextBoard = board;
        nextBoard.apply_move(move, isWhiteToPlay);

        MCTSNode* child = new MCTSNode(nextBoard, !isWhiteToPlay, this, move);
        children.push_back(child);
        return child;
    }
};

// Backpropagation:
// - incremente les visites sur le chemin noeud -> racine
// - ajoute la recompense du bon point de vue (joueur qui vient de jouer)
// whiteWinProb est toujours exprime en probabilite de victoire blanche.
void backpropagate(MCTSNode* node, double whiteWinProb, bool rootIsWhite) {
    while (node != nullptr) {
        node->visits.fetch_add(1);

        bool rewardWhite = (node->parent == nullptr) ? rootIsWhite : !node->isWhiteToPlay;
        node->wins += rewardWhite ? whiteWinProb : (1.0 - whiteWinProb);

        node = node->parent;
    }
}

std::string get_best_move(Board64_t& board, int color) {
    bool isWhite = (color == WHITE);

    // 1) Priorite absolue: prendre un gain immediat s'il existe.
    Move64_t forcedWin;
    if (find_immediate_winning_move(board, isWhite, forcedWin)) {
        return pos_to_coord(forcedWin.pi) + "-" + pos_to_coord(forcedWin.pf);
    }

    // 2) Filtre tactique: on privilegie les coups qui n'offrent pas de mat direct.
    std::vector<Move64_t> safeMoves = get_safe_moves(board, isWhite);
    uint64_t enemy = isWhite ? board.black : board.white;
    if (!safeMoves.empty()) {
        sort_moves_by_priority(safeMoves, enemy, isWhite);
    }

    // 3) Initialisation de l'arbre MCTS partage entre tous les threads.
    MCTSNode* root = new MCTSNode(board, isWhite, nullptr, Move64_t());
    if (!safeMoves.empty() && safeMoves.size() < root->untriedMoves.size())
        root->untriedMoves = safeMoves;

    if (root->untriedMoves.empty()) {
        delete root;
        return "resign";
    }

    auto start = std::chrono::high_resolution_clock::now();
    g_total_iters = 0;

    // Boucle principale d'un worker MCTS:
    // selection -> expansion -> evaluation -> backpropagation.
    auto worker = [&](MCTSNode* root_ptr) {
        int iter = 0;
        while (true) {
            // Verification periodique de la limite de temps.
            if (iter % 512 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(now - start).count() >= g_time_limit_s)
                    break;
            }
            iter++;
            g_total_iters++;

            // Selection avec virtual loss pour reduire la contention inter-threads.
            MCTSNode* node = root_ptr;
            std::vector<MCTSNode*> vlPath;
            MCTSNode* child;
            while ((child = node->selectChild()) != nullptr) {
                node->visits.fetch_add(VIRTUAL_LOSS);
                vlPath.push_back(node);
                node = child;
            }

            // Expansion d'un nouveau fils si possible.
            if (!node->isTerminal()) {
                MCTSNode* expanded = node->tryExpand();
                if (expanded) {
                    node = expanded;
                }
            }

            // Evaluation de la feuille (terminal direct ou evaluation heuristique).
            double whiteWinProb;
            if (node->isTerminal()) {
                whiteWinProb = terminal_white_win_proba(node->board);
            } else {
                whiteWinProb = evaluation_leaf(node->board, node->isWhiteToPlay);
            }

            // Remontee de l'information de resultat dans l'arbre.
            backpropagate(node, whiteWinProb, isWhite);

            // Retire le virtual loss applique pendant la phase de selection.
            for (auto* n : vlPath) {
                n->visits.fetch_sub(VIRTUAL_LOSS);
            }
        }
    };

    int nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; t++)
        threads.emplace_back(worker, root);
    for (auto& th : threads) th.join();

    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

    // 4) Decision finale: on joue l'enfant le plus visite.
    if (root->children.empty()) {
        delete root;
        return "resign";
    }

    MCTSNode* bestNode = root->children.front();
    int bestVisits = bestNode->visits.load();
    for (size_t i = 1; i < root->children.size(); ++i) {
        MCTSNode* child = root->children[i];
        int visits = child->visits.load();
        if (visits > bestVisits) {
            bestVisits = visits;
            bestNode = child;
        }
    }

    double bestWinRate = (bestVisits > 0) ? (bestNode->wins / bestVisits) : 0.0;
    Move64_t bestMove = bestNode->moveFromParent;

    delete root;

    fprintf(stderr, "ITERS:%d VISITS:%d WINRATE:%.4f TIME:%.2f\n",
            g_total_iters.load(), bestVisits, bestWinRate, elapsed);

    return pos_to_coord(bestMove.pi) + "-" + pos_to_coord(bestMove.pf);
}

int main(int argc, char** argv) {
    srand(time(nullptr));
    // Format attendu:
    // argv[1] = board string
    // argv[2] = "O" (blanc) ou "@" (noir)
    // argv[3] = temps optionnel en secondes
    if (argc < 3) {
        fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]);
        return 0;
    }
    if (argc >= 4) {
        g_time_limit_s = atof(argv[3]);
        if (g_time_limit_s <= 0) g_time_limit_s = 1.3;
    }

    Board64_t B(argv[1]);
    B.seed = (uint32_t)time(nullptr);
    std::string playerStr(argv[2]);

    if (playerStr == "O") {
        printf("%s\n", get_best_move(B, WHITE).c_str());
    } else if (playerStr == "@") {
        printf("%s\n", get_best_move(B, BLACK).c_str());
    }

    return 0;
}

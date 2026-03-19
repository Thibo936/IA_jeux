// hexbb.h — moteur Hex 11x11 avec représentation bitboard 2x64 + 1x32
// Joueur 1 (Blue/O) : connecte Nord (ligne 0) → Sud (ligne 10)
// Joueur 2 (Red/@)  : connecte Ouest (col 0)  → Est  (col 10)
//
// Encodage des cases : case (row, col) → bit index = row*11 + col
//   bits  0..63  dans board_lo
//   bits 64..120 dans board_hi  (bits 0..56 de board_hi)
//
#ifndef HEXBB_H
#define HEXBB_H

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <climits>
#include <inttypes.h>
#include <queue>

// ─── Constantes de taille ───────────────────────────────────────────────────
static const int HEX_SIZE  = 11;          // 11x11
static const int HEX_CELLS = 121;         // 11*11

// ─── Voisins hexagonaux ──────────────────────────────────────────────────────
// Sur un plateau Hex avec la disposition standard (losange aplati) :
// les 6 voisins de (r, c) sont :
//   (r-1, c), (r-1, c+1),
//   (r,   c-1), (r,   c+1),
//   (r+1, c-1), (r+1, c)
static const int HEX_DR[6] = {-1, -1,  0, 0, +1, +1};
static const int HEX_DC[6] = { 0, +1, -1,+1, -1,  0};

// ─── Helpers bitboard 128 bits (représentés par deux uint64_t) ──────────────

// Lire le bit i dans (lo, hi)
static inline bool bit_get(uint64_t lo, uint64_t hi, int i) {
    if (i < 64) return (lo >> i) & 1ULL;
    else        return (hi >> (i - 64)) & 1ULL;
}

// Mettre le bit i à 1
static inline void bit_set(uint64_t& lo, uint64_t& hi, int i) {
    if (i < 64) lo |= (1ULL << i);
    else        hi |= (1ULL << (i - 64));
}

// Mettre le bit i à 0
static inline void bit_clear(uint64_t& lo, uint64_t& hi, int i) {
    if (i < 64) lo &= ~(1ULL << i);
    else        hi &= ~(1ULL << (i - 64));
}

// Nombre de bits à 1 dans (lo, hi) pour les 121 premières positions
static inline int bit_count121(uint64_t lo, uint64_t hi) {
    return __builtin_popcountll(lo) + __builtin_popcountll(hi & 0x1FFFFFFFFFFFFFFULL);
}

// Sélectionner le n-ième bit à 1 (0-indexé) dans (lo, hi)
static inline int bit_select(uint64_t lo, uint64_t hi, int n) {
    int cnt = __builtin_popcountll(lo);
    if (n < cnt) {
        // Dans lo
        for (int i = 0; i < 64; i++) {
            if ((lo >> i) & 1ULL) {
                if (n == 0) return i;
                n--;
            }
        }
    }
    n -= cnt;
    for (int i = 0; i < 57; i++) {
        if ((hi >> i) & 1ULL) {
            if (n == 0) return 64 + i;
            n--;
        }
    }
    return -1; // ne devrait pas arriver
}

// ─── PRNG rapide ─────────────────────────────────────────────────────────────
static inline uint32_t hex_rand_xorshift(uint32_t s) {
    s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
    return s;
}

// ─── Conversion coordonnées ──────────────────────────────────────────────────
// case index → notation "A1".."K11"
static inline std::string hex_pos_to_str(int idx) {
    if (idx < 0 || idx >= HEX_CELLS) return "??";
    int row = idx / HEX_SIZE;
    int col = idx % HEX_SIZE;
    char buf[5];
    snprintf(buf, sizeof(buf), "%c%d", 'A' + col, row + 1);
    return std::string(buf);
}

// notation "A1".."K11" → case index (-1 si invalide)
static inline int hex_str_to_pos(const std::string& s) {
    if (s.size() < 2) return -1;
    int col = s[0] - 'A';
    int row = std::stoi(s.substr(1)) - 1;
    if (col < 0 || col >= HEX_SIZE || row < 0 || row >= HEX_SIZE) return -1;
    return row * HEX_SIZE + col;
}

// ─── Coup Hex : juste une case ───────────────────────────────────────────────
struct HexMove {
    int pos; // 0..120
    HexMove() : pos(-1) {}
    HexMove(int p) : pos(p) {}
    std::string to_str() const { return hex_pos_to_str(pos); }
};

// ─── État du plateau ─────────────────────────────────────────────────────────
struct HexBoard {
    // Pions Blue (joueur 1, 'O') : connecte lignes 0 → 10 (Nord-Sud)
    uint64_t blue_lo, blue_hi;
    // Pions Red  (joueur 2, '@') : connecte cols  0 → 10 (Ouest-Est)
    uint64_t red_lo,  red_hi;
    uint32_t seed;

    HexBoard();
    HexBoard(const std::string& strboard); // 121 chars : '.' ou 'O' ou '@'

    bool operator==(const HexBoard& o) const;

    // Cases vides : (lo, hi) de occupied = blue | red
    void get_empty(uint64_t& elo, uint64_t& ehi) const;

    // Nombre de coups légaux = nombre de cases vides
    int num_legal_moves() const;

    // Générer tous les coups légaux
    std::vector<HexMove> get_legal_moves() const;

    // Appliquer un coup
    void apply_move(const HexMove& m, bool isBlue);

    // Détection de victoire par BFS flood-fill
    bool blue_win() const;   // Blue connecte ligne 0 → ligne 10
    bool red_win()  const;   // Red connecte col  0 → col  10

    bool is_win(bool isBlue) const;

    // Playout aléatoire jusqu'à la fin
    bool random_playout(bool blueToPlay); // retourne true si Blue gagne

    // Coup aléatoire
    HexMove get_rand_move() const;

    // Affichage
    void print(FILE* out = stdout) const;

    // Évaluation heuristique (positive = Blue avantage)
    // Basée sur la différence de longueur du chemin virtuel
    int eval(bool isBlue) const;

private:
    bool flood_blue_win() const;
    bool flood_red_win()  const;
    int  shortest_path_blue() const;
    int  shortest_path_red()  const;
};

// ─── Implémentation ──────────────────────────────────────────────────────────

inline HexBoard::HexBoard()
    : blue_lo(0), blue_hi(0), red_lo(0), red_hi(0), seed(1) {}

inline HexBoard::HexBoard(const std::string& s)
    : blue_lo(0), blue_hi(0), red_lo(0), red_hi(0), seed(1)
{
    for (int i = 0; i < HEX_CELLS && i < (int)s.size(); i++) {
        if (s[i] == 'O') bit_set(blue_lo, blue_hi, i);
        else if (s[i] == '@') bit_set(red_lo, red_hi, i);
    }
}

inline bool HexBoard::operator==(const HexBoard& o) const {
    return blue_lo == o.blue_lo && blue_hi == o.blue_hi
        && red_lo  == o.red_lo  && red_hi  == o.red_hi;
}

inline void HexBoard::get_empty(uint64_t& elo, uint64_t& ehi) const {
    uint64_t occ_lo = blue_lo | red_lo;
    uint64_t occ_hi = blue_hi | red_hi;
    // masque des 121 bits valides : lo = tous les 64, hi = bits 0..56
    elo = ~occ_lo;
    ehi = ~occ_hi & 0x1FFFFFFFFFFFFFFULL;
}

inline int HexBoard::num_legal_moves() const {
    uint64_t elo, ehi;
    get_empty(elo, ehi);
    return bit_count121(elo, ehi);
}

inline std::vector<HexMove> HexBoard::get_legal_moves() const {
    uint64_t elo, ehi;
    get_empty(elo, ehi);
    std::vector<HexMove> moves;
    moves.reserve(bit_count121(elo, ehi));
    for (int i = 0; i < 64; i++)
        if ((elo >> i) & 1ULL) moves.push_back(HexMove(i));
    for (int i = 0; i < 57; i++)
        if ((ehi >> i) & 1ULL) moves.push_back(HexMove(64 + i));
    return moves;
}

inline void HexBoard::apply_move(const HexMove& m, bool isBlue) {
    if (isBlue) bit_set(blue_lo, blue_hi, m.pos);
    else        bit_set(red_lo,  red_hi,  m.pos);
}

// ─── Détection de victoire par BFS ───────────────────────────────────────────

// Blue gagne si elle connecte une cellule de la ligne 0 à une cellule de la ligne 10
// en passant uniquement par des pions Blue
inline bool HexBoard::flood_blue_win() const {
    // Visited bitboard
    uint64_t vis_lo = 0, vis_hi = 0;
    // Queue BFS : on part des pions Blue sur la ligne 0 (indices 0..10)
    int q[HEX_CELLS];
    int head = 0, tail = 0;

    for (int c = 0; c < HEX_SIZE; c++) {
        int idx = c; // ligne 0
        if (bit_get(blue_lo, blue_hi, idx) && !bit_get(vis_lo, vis_hi, idx)) {
            bit_set(vis_lo, vis_hi, idx);
            q[tail++] = idx;
        }
    }

    while (head < tail) {
        int cur = q[head++];
        int r = cur / HEX_SIZE;
        int c = cur % HEX_SIZE;
        if (r == HEX_SIZE - 1) return true; // atteint la ligne 10

        for (int d = 0; d < 6; d++) {
            int nr = r + HEX_DR[d];
            int nc = c + HEX_DC[d];
            if (nr < 0 || nr >= HEX_SIZE || nc < 0 || nc >= HEX_SIZE) continue;
            int nidx = nr * HEX_SIZE + nc;
            if (!bit_get(blue_lo, blue_hi, nidx)) continue;
            if (bit_get(vis_lo, vis_hi, nidx)) continue;
            bit_set(vis_lo, vis_hi, nidx);
            q[tail++] = nidx;
        }
    }
    return false;
}

// Red gagne si elle connecte une cellule de la colonne 0 à une cellule de la colonne 10
inline bool HexBoard::flood_red_win() const {
    uint64_t vis_lo = 0, vis_hi = 0;
    int q[HEX_CELLS];
    int head = 0, tail = 0;

    for (int r = 0; r < HEX_SIZE; r++) {
        int idx = r * HEX_SIZE + 0; // colonne 0
        if (bit_get(red_lo, red_hi, idx) && !bit_get(vis_lo, vis_hi, idx)) {
            bit_set(vis_lo, vis_hi, idx);
            q[tail++] = idx;
        }
    }

    while (head < tail) {
        int cur = q[head++];
        int r = cur / HEX_SIZE;
        int c = cur % HEX_SIZE;
        if (c == HEX_SIZE - 1) return true; // atteint la colonne 10

        for (int d = 0; d < 6; d++) {
            int nr = r + HEX_DR[d];
            int nc = c + HEX_DC[d];
            if (nr < 0 || nr >= HEX_SIZE || nc < 0 || nc >= HEX_SIZE) continue;
            int nidx = nr * HEX_SIZE + nc;
            if (!bit_get(red_lo, red_hi, nidx)) continue;
            if (bit_get(vis_lo, vis_hi, nidx)) continue;
            bit_set(vis_lo, vis_hi, nidx);
            q[tail++] = nidx;
        }
    }
    return false;
}

inline bool HexBoard::blue_win() const { return flood_blue_win(); }
inline bool HexBoard::red_win()  const { return flood_red_win(); }
inline bool HexBoard::is_win(bool isBlue) const {
    return isBlue ? blue_win() : red_win();
}

// ─── Playout aléatoire ───────────────────────────────────────────────────────
inline HexMove HexBoard::get_rand_move() const {
    uint64_t elo, ehi;
    get_empty(elo, ehi);
    int total = bit_count121(elo, ehi);
    if (total == 0) return HexMove(-1);
    // Utilise le seed du board (pas const, mais on fait une copie locale)
    uint32_t s = seed ^ (uint32_t)(blue_lo ^ red_lo ^ (blue_hi << 3));
    s = hex_rand_xorshift(s);
    int idx = (int)(s % (uint32_t)total);
    return HexMove(bit_select(elo, ehi, idx));
}

inline bool HexBoard::random_playout(bool blueToPlay) {
    HexBoard b = *this;
    // Pour un playout rapide, on génère une permutation aléatoire des cases vides
    // et on les attribue alternativement aux deux joueurs.
    // C'est équivalent au playout aléatoire de Hex (propriété de déterminisme de Hex).
    uint64_t elo, ehi;
    b.get_empty(elo, ehi);

    // Construire la liste des cases vides
    int empties[HEX_CELLS];
    int n = 0;
    for (int i = 0; i < 64; i++)
        if ((elo >> i) & 1ULL) empties[n++] = i;
    for (int i = 0; i < 57; i++)
        if ((ehi >> i) & 1ULL) empties[n++] = 64 + i;

    // Mélange Fisher-Yates avec xorshift
    uint32_t s = b.seed ^ (uint32_t)(b.blue_lo ^ b.red_lo);
    for (int i = n - 1; i > 0; i--) {
        s = hex_rand_xorshift(s);
        int j = s % (i + 1);
        int tmp = empties[i]; empties[i] = empties[j]; empties[j] = tmp;
    }

    // Distribuer les cases : blueToPlay commence
    bool cur = blueToPlay;
    for (int i = 0; i < n; i++) {
        b.apply_move(HexMove(empties[i]), cur);
        cur = !cur;
    }
    return b.blue_win();
}

// ─── Évaluation heuristique par plus court chemin ────────────────────────────
// On utilise une recherche BFS sur le graphe "virtuel" où :
// - les cellules occupées par le joueur coûtent 0
// - les cellules vides coûtent 1
// - les cellules adverses sont bloquées (coût infini)
// Résultat : nombre de cellules vides à remplir pour gagner (plus petit = mieux)

inline int HexBoard::shortest_path_blue() const {
    // BFS de la ligne 0 vers la ligne 10 pour Blue
    // Coût 0 = pion Blue, Coût 1 = vide, bloqué = pion Red
    int dist[HEX_CELLS];
    int q[HEX_CELLS * 4]; // deque simulée (0-1 BFS) — assez large pour les deux côtés
    for (int i = 0; i < HEX_CELLS; i++) dist[i] = INT_MAX;

    int head = HEX_CELLS * 2, tail = HEX_CELLS * 2; // deque au milieu du tableau

    // Initialisation : cellules de la ligne 0 accessibles à Blue
    for (int c = 0; c < HEX_SIZE; c++) {
        int idx = c;
        if (bit_get(red_lo, red_hi, idx)) continue; // bloqué
        int cost = bit_get(blue_lo, blue_hi, idx) ? 0 : 1;
        if (cost < dist[idx]) {
            dist[idx] = cost;
            if (cost == 0) q[--head] = idx; // push_front
            else           q[tail++] = idx; // push_back
        }
    }

    while (head < tail) {
        int cur = q[head++];
        int r = cur / HEX_SIZE;
        int c = cur % HEX_SIZE;
        if (r == HEX_SIZE - 1) return dist[cur]; // atteint ligne 10

        for (int d = 0; d < 6; d++) {
            int nr = r + HEX_DR[d];
            int nc = c + HEX_DC[d];
            if (nr < 0 || nr >= HEX_SIZE || nc < 0 || nc >= HEX_SIZE) continue;
            int nidx = nr * HEX_SIZE + nc;
            if (bit_get(red_lo, red_hi, nidx)) continue; // bloqué
            int cost = bit_get(blue_lo, blue_hi, nidx) ? 0 : 1;
            int nd = dist[cur] + cost;
            if (nd < dist[nidx]) {
                dist[nidx] = nd;
                if (cost == 0) { head--; q[head] = nidx; }
                else           { q[tail++] = nidx; }
            }
        }
    }
    return INT_MAX; // pas de chemin (ne devrait pas arriver avant fin de partie)
}

inline int HexBoard::shortest_path_red() const {
    // BFS de la col 0 vers la col 10 pour Red
    int dist[HEX_CELLS];
    int q[HEX_CELLS * 4];
    for (int i = 0; i < HEX_CELLS; i++) dist[i] = INT_MAX;

    int head = HEX_CELLS * 2, tail = HEX_CELLS * 2;

    for (int r = 0; r < HEX_SIZE; r++) {
        int idx = r * HEX_SIZE + 0; // colonne 0
        if (bit_get(blue_lo, blue_hi, idx)) continue;
        int cost = bit_get(red_lo, red_hi, idx) ? 0 : 1;
        if (cost < dist[idx]) {
            dist[idx] = cost;
            if (cost == 0) q[--head] = idx;
            else           q[tail++] = idx;
        }
    }

    while (head < tail) {
        int cur = q[head++];
        int r = cur / HEX_SIZE;
        int c = cur % HEX_SIZE;
        if (c == HEX_SIZE - 1) return dist[cur];

        for (int d = 0; d < 6; d++) {
            int nr = r + HEX_DR[d];
            int nc = c + HEX_DC[d];
            if (nr < 0 || nr >= HEX_SIZE || nc < 0 || nc >= HEX_SIZE) continue;
            int nidx = nr * HEX_SIZE + nc;
            if (bit_get(blue_lo, blue_hi, nidx)) continue;
            int cost = bit_get(red_lo, red_hi, nidx) ? 0 : 1;
            int nd = dist[cur] + cost;
            if (nd < dist[nidx]) {
                dist[nidx] = nd;
                if (cost == 0) { head--; q[head] = nidx; }
                else           { q[tail++] = nidx; }
            }
        }
    }
    return INT_MAX;
}

// eval : positif = Blue avantage, négatif = Red avantage
inline int HexBoard::eval(bool isBlue) const {
    if (blue_win()) return isBlue ?  100000 : -100000;
    if (red_win())  return isBlue ? -100000 :  100000;

    int pb = shortest_path_blue();
    int pr = shortest_path_red();

    // Score : chemin adverse long (bon) - chemin propre long (mauvais)
    // On retourne du point de vue de isBlue
    int score;
    if (pb == INT_MAX && pr == INT_MAX) score = 0;
    else if (pb == INT_MAX) score = -100000;
    else if (pr == INT_MAX) score =  100000;
    else score = pr - pb; // >0 = Blue plus proche

    return isBlue ? score : -score;
}

// ─── Affichage ───────────────────────────────────────────────────────────────
inline void HexBoard::print(FILE* out) const {
    fprintf(out, "    ");
    for (int c = 0; c < HEX_SIZE; c++) fprintf(out, "%c ", 'A' + c);
    fprintf(out, "\n");
    for (int r = 0; r < HEX_SIZE; r++) {
        fprintf(out, "%2d  ", r + 1);
        for (int s = 0; s < r; s++) fprintf(out, " ");
        for (int c = 0; c < HEX_SIZE; c++) {
            int idx = r * HEX_SIZE + c;
            if      (bit_get(blue_lo, blue_hi, idx)) fprintf(out, "O ");
            else if (bit_get(red_lo,  red_hi,  idx)) fprintf(out, "@ ");
            else                                     fprintf(out, ". ");
        }
        fprintf(out, "\n");
    }
}

#endif /* HEXBB_H */

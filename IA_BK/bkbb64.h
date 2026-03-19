// bitboard pour breakthrough 8x8
// avec fonction d'évaluation et playout
#ifndef BKBB64_H
#define BKBB64_H

#include <cstdlib>   
#include <cstdio>
#include <cstdint>
#include <bitset>
#include <chrono>
#include <cstring>
#include <climits>
#include <inttypes.h>
#include <vector>

struct Move64_t {
  uint64_t pi;
  uint64_t pf;

  std::string move_to_str();
};

inline
std::string pos_to_coord(const uint64_t& pos) {
  uint64_t where = 0ULL;
  for(int i = 0; i < 64; i++) {
    where = (1ULL<<i);
    if(where == pos) {
      char buf[3];
      snprintf(buf,3,"%c%c", 'A'+i%8, '0'+8-(i/8));
      return std::string(buf);
    }
  }
  return std::string("ERROR");
}
inline
std::string Move64_t::move_to_str() {
  return pos_to_coord(pi)+"-"+pos_to_coord(pf);
}

struct Lfr_t { 
  uint64_t left;
  uint64_t forward;
  uint64_t right;
  Lfr_t(uint64_t _l, uint64_t _f, uint64_t _r);
  void print();
  std::vector<Move64_t> get_white_moves();
  std::vector<Move64_t> get_black_moves();
  std::vector<Move64_t> get_moves(bool _white);
  Move64_t get_nth_white_move(uint32_t _move_id);
  Move64_t get_nth_black_move(uint32_t _move_id);
  Move64_t get_nth_move(uint32_t _move_id, bool _white);
};
Lfr_t::Lfr_t(uint64_t _l, uint64_t _f, uint64_t _r) {
  left = _l;
  forward = _f;
  right = _r;
}
void Lfr_t::print() {
  std::bitset<64> bl(left);
  std::bitset<64> bf(forward);
  std::bitset<64> br(right);
  int line_id = 0;
  int col_id;
  char lines[8*8];
  for(int i = 0; i < (int) bl.size(); i++) {
    if(i>0 && i%8==0) line_id++;    
    col_id = i%8;
    lines[line_id*8+col_id] = int(bl[i]); // 0 1
    lines[line_id*8+col_id] += 2*int(bf[i]); // 2
    lines[line_id*8+col_id] += 4*int(br[i]); // 4
  }
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) 
      printf("%c", '0'+lines[i*8+j]);
    printf("\n");
  }
  printf("\n");
}

// from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
static inline
uint8_t count8(uint8_t where) {    
  uint8_t c; 
  static const uint8_t S[] = {1, 2, 4}; // Magic Binary Numbers
  static const uint8_t B[] = {0x55, 0x33, 0x0f};
  c = where - ((where >> S[0]) & B[0]);
  c = ((c >> S[1]) & B[1]) + (c & B[1]);
  c = ((c >> S[2]) + c) & B[2];
  return c;
}
static inline
uint16_t count16(uint16_t where) {    
  uint16_t c; 
  static const uint16_t S[] = {1, 2, 4, 8}; // Magic Binary Numbers
  static const uint16_t B[] = {
    0x5555, 0x3333, 
    0x0f0f, 0x00ff}; 
  c = where - ((where >> S[0]) & B[0]);
  c = ((c >> S[1]) & B[1]) + (c & B[1]);
  c = ((c >> S[2]) + c) & B[2];
  c = ((c >> S[3]) + c) & B[3];
  return c;
}
static inline
uint64_t count64(uint64_t where) {    
  uint64_t c; 
  static const uint64_t S[] = {1, 2, 4, 8, 16, 32}; // Magic Binary Numbers
  static const uint64_t B[] = {
    0x5555555555555555, 0x3333333333333333, 
    0x0f0f0f0f0f0f0f0f, 0x00ff00ff00ff00ff, 
    0x0000ffff0000ffff, 0x00000000ffffffff};
  c = where - ((where >> S[0]) & B[0]);
  c = ((c >> S[1]) & B[1]) + (c & B[1]);
  c = ((c >> S[2]) + c) & B[2];
  c = ((c >> S[3]) + c) & B[3];
  c = ((c >> S[4]) + c) & B[4];
  c = ((c >> S[5]) + c) & B[5];
  return c;
}
static inline
uint64_t select_move(const uint64_t& _moves, uint32_t _nb) {
  uint64_t c = 1ULL;
  uint64_t ret = 0ULL;
  for (int i = 0; i < 64; i++) {
    if((c&_moves) == c) {
      if(_nb==0) {
        ret = c;
        break;
      }
      _nb=_nb-1;
    }
    c = c<<1;
  }
  return ret;
}
inline
std::vector<Move64_t> Lfr_t::get_white_moves() {
  uint64_t nb_wl = count64(left);
  uint64_t nb_wf = count64(forward);
  uint64_t nb_wr = count64(right);
  std::vector<Move64_t> ret;
  for(int i = 0; i < int(nb_wf); i++) {
    Move64_t new_move;
    new_move.pf = select_move(forward, i);  
    new_move.pi = new_move.pf<<8;
    ret.push_back(new_move);
  }
  for(int i = 0; i < int(nb_wl); i++) {
    Move64_t new_move;
    new_move.pf = select_move(left, i);  
    new_move.pi = new_move.pf<<9;
    ret.push_back(new_move);
  }
  for(int i = 0; i < int(nb_wr); i++) {
    Move64_t new_move;
    new_move.pf = select_move(right, i);  
    new_move.pi = new_move.pf<<7;
    ret.push_back(new_move);
  }
  return ret;
}
inline
std::vector<Move64_t> Lfr_t::get_black_moves() {
  uint64_t nb_wl = count64(left);
  uint64_t nb_wf = count64(forward);
  uint64_t nb_wr = count64(right);
  std::vector<Move64_t> ret;
  for(int i = 0; i < int(nb_wf); i++) {
    Move64_t new_move;
    new_move.pf = select_move(forward, i);  
    new_move.pi = new_move.pf>>8;
    ret.push_back(new_move);
  } 
  for(int i = 0; i < int(nb_wl); i++) {
    Move64_t new_move;
    new_move.pf = select_move(left, i);  
    new_move.pi = new_move.pf>>7;
    ret.push_back(new_move);
  }
  for(int i = 0; i < int(nb_wr); i++) {
    Move64_t new_move;
    new_move.pf = select_move(right, i);  
    new_move.pi = new_move.pf>>9;
    ret.push_back(new_move);
  }
  return ret;
}
inline 
std::vector<Move64_t> Lfr_t::get_moves(bool _white) {
  if(_white) return get_white_moves();
  else return get_black_moves();
}
inline
Move64_t Lfr_t::get_nth_white_move(uint32_t _move_id) {
  uint64_t nb_wl = count64(left);
  uint64_t nb_wf = count64(forward);
  Move64_t ret;
  if(_move_id < nb_wf) {
    ret.pf = select_move(forward, _move_id);  
    ret.pi = ret.pf<<8;
  } else {
    _move_id -= nb_wf;
    if(_move_id < nb_wl) {
      ret.pf = select_move(left, _move_id);
      ret.pi = ret.pf<<9;
    } else {
      _move_id -= nb_wl;
      ret.pf = select_move(right, _move_id);
      ret.pi = ret.pf<<7;
    }
  }
  return ret;
}
inline
Move64_t Lfr_t::get_nth_black_move(uint32_t _move_id) {
  uint64_t nb_wl = count64(left);
  uint64_t nb_wf = count64(forward);
  Move64_t ret;
  if(_move_id < nb_wf) {
    ret.pf = select_move(forward, _move_id);  
    ret.pi = ret.pf>>8;
  } else {
    _move_id -= nb_wf;
    if(_move_id < nb_wl) {
      ret.pf = select_move(left, _move_id);
      ret.pi = ret.pf>>7;
    } else {
      _move_id -= nb_wl;
      ret.pf = select_move(right, _move_id);
      ret.pi = ret.pf>>9;
    }
  }
  return ret;
}
inline
Move64_t Lfr_t::get_nth_move(uint32_t _move_id, bool _white) {
  if(_white) return get_nth_white_move(_move_id);
  else return get_nth_black_move(_move_id);
}

struct Board64_t {
  uint64_t white;
  uint64_t black;
  uint32_t seed;

  Board64_t();
  Board64_t(const std::string& strboard);
  bool operator== (const Board64_t&) const;
  int eval(const bool _white) const;
  void apply_white_move(const Move64_t& move);
  void apply_black_move(const Move64_t& move);
  void apply_move(const Move64_t& move, bool _white);
  void print_board(FILE* out) const;
  void print_moves(const bool _white) const;
  uint64_t white_forward() const;
  uint64_t white_left() const;
  uint64_t white_right() const;
  uint32_t white_win() const;
  uint64_t black_forward() const;
  uint64_t black_left() const;
  uint64_t black_right() const;
  uint32_t black_win() const;

  uint64_t forward(bool _white) const;
  uint64_t left(bool _white) const;
  uint64_t right(bool _white) const;
  uint32_t win(bool _white) const;
 
  Move64_t get_rand_white_move(const Lfr_t& lfr);
  Move64_t get_rand_white_move();
  void rand_white_move();
  Move64_t get_rand_black_move(const Lfr_t& lfr);
  Move64_t get_rand_black_move();
  void rand_black_move();

  Move64_t get_rand_move(const Lfr_t& lfr, bool _white);
  Move64_t get_rand_move(bool _white);
  void rand_move(bool _white);

  void seq_playout(bool _white);

};

inline
Board64_t::Board64_t() {
  black = 0x000000000000ffff;
  white = 0xffff000000000000;
  seed = 1ULL;
}
inline
Board64_t::Board64_t(const std::string& strboard) {
  black = 0ULL;
  white = 0ULL;
  for(int i = 0; i < (int)strboard.size(); i++) {
    if(strboard[i] == '@') black += (1ULL<<i);
    if(strboard[i] == 'O') white += (1ULL<<i);
  }
  seed = 1ULL;
}
inline
bool Board64_t::operator== (const Board64_t& _o) const {
  if(black == _o.black && white == _o.white) return true;
  return false;
}

inline
int Board64_t::eval(const bool _white) const { // + lignes des pions et - lignes de pions adverses
  int score = 0;
  if(white_win()) {
    if(_white) return INT_MAX;
    return -INT_MAX;
  } 
  if(black_win()) {
    if(_white) return -INT_MAX;
    return INT_MAX;
  } 
  uint64_t wh = white;
  uint64_t bl = black;  
  while (wh) {
    int line = (8-(ffsll(wh) - 1)/8);
    score += line;
    wh &= wh - 1;
  }
  while (bl) {
    int line = (ffsll(bl)-1)/8;
    score -= line;
    bl &= bl - 1;
  }   
  if(_white==false) score=-score;
  return score;
}
inline
void Board64_t::apply_white_move(const Move64_t& move) {
  white = (white^move.pi)|move.pf;
  black = (black|move.pf)^move.pf;  
}
inline
void Board64_t::apply_black_move(const Move64_t& move) {
  black = (black^move.pi)|move.pf;
  white = (white|move.pf)^move.pf;
}
inline
void Board64_t::apply_move(const Move64_t& move, bool _white) {
  if(_white) apply_white_move(move);
  else apply_black_move(move);
}
inline
void Board64_t::print_board(FILE* out=stdout) const {
  std::bitset<64> bbs(black);
  std::bitset<64> wbs(white);
  fprintf(out, "board %d\n", seed);
  fprintf(out, "  A B C D E F G H\n");
  for(int i = 0; i < (int) bbs.size(); i++) {
    if(i>0 && i%8==0) fprintf(out, "%d\n", 8-(i/8-1));
    if(i==0 || i%8==0) fprintf(out, "%d ", 8-(i/8));
    if(int(bbs[i])==1) fprintf(out, "@ ");
    else if(int(wbs[i])==1) fprintf(out, "O ");
    else fprintf(out, ". ");
  }
  fprintf(out, "1\n");
  fprintf(out, "  A B C D E F G H\n");
}

inline
void Board64_t::print_moves(const bool _white) const {  
  if(_white) {
    uint64_t wl = white_left();
    uint64_t wf = white_forward();
    uint64_t wr = white_right();
    std::vector<Move64_t> moves = Lfr_t(wl, wf, wr).get_white_moves();
    for(std::vector<Move64_t>::iterator ii = moves.begin(); ii != moves.end(); ii++) {
      printf("%s ", ii->move_to_str().c_str());
    }
    printf("\n");
  } else {
    uint64_t bl = black_left();
    uint64_t bf = black_forward();
    uint64_t br = black_right();
    std::vector<Move64_t> moves = Lfr_t(bl, bf, br).get_black_moves();
    for(std::vector<Move64_t>::iterator ii = moves.begin(); ii != moves.end(); ii++) {
      printf("%s ", ii->move_to_str().c_str());
    }
    printf("\n");
  }
}
inline
uint64_t Board64_t::white_forward() const {
  uint64_t empty = ~(white | black);
  return (white >> 8) & empty;
}
inline
uint64_t Board64_t::white_left() const {
  return ((white & 0xfefefefefefefefe)>>9)&~white;
}
inline
uint64_t Board64_t::white_right() const {
  return ((white & 0x7f7f7f7f7f7f7f7f)>>7)&~white;
}
inline
uint32_t Board64_t::white_win() const {
  return (black==0ULL) | ((white&0x00000000000000ff)!=0ULL);
}
inline
uint64_t Board64_t::black_forward() const {
  uint64_t empty =~(white | black);
  return (black << 8) & empty;
}
inline
uint64_t Board64_t::black_left() const {
  return ((black & 0xfefefefefefefefe)<<7)&~black;
}
inline
uint64_t Board64_t::black_right() const {
  return ((black & 0x7f7f7f7f7f7f7f7f)<<9)&~black;
}
inline
uint32_t Board64_t::black_win() const {
  return (white==0ULL) |((black&0xff00000000000000)!=0ULL);
}
inline
uint64_t Board64_t::forward(bool _white) const {
  if(_white) return white_forward();
  else return black_forward();
}
inline
uint64_t Board64_t::left(bool _white) const {
  if(_white) return white_left();
  else return black_left();
}
inline
uint64_t Board64_t::right(bool _white) const {
  if(_white) return white_right();
  else return black_right();
}
inline
uint32_t Board64_t::win(bool _white) const {
  if(_white) return white_win();
  else return black_win();

}
void print_state(uint64_t state) {
  std::string pieces = ".1";
  std::bitset<64> bs(state) ;
  for(int i = 0; i < (int) bs.size(); i++) {
    if(i>0 && i%8==0) printf("\n");
    printf("%c", pieces[int(bs[i])]);
  }
  printf("\n");
}
void print_w(uint64_t w) {
  std::string pieces = ".1";
  std::bitset<64> wbs(w);
  int line_id = 0;
  int col_id;
  char lines[8*8];
  for(int i = 0; i < (int) wbs.size(); i++) {
    if(i>0 && i%8==0) line_id++;    
    col_id = i%8;
    lines[line_id*8+col_id] = pieces[int(wbs[i])];
  }
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) 
      printf("%c", lines[i*8+j]);
    printf("\n");
  }
  printf("\n");
}
void print_wb(uint64_t w, uint64_t b) {
  std::string pieces = ".x@";
  std::bitset<64> bbs(b);
  std::bitset<64> wbs(w);
  int line_id = 0;
  int col_id;
  char lines[16*8];
  for(int i = 0; i < (int) bbs.size(); i++) {
    if(i>0 && i%8==0) line_id++;    
    col_id = i%8;
    lines[line_id*16+col_id] = pieces[int(bbs[i])];
    lines[line_id*16+col_id+8] = pieces[int(wbs[i])];
  }
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 16; j++) 
      printf("%c", lines[i*16+j]);
    printf("\n");
  }
  printf("\n");
}

// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
static inline
uint32_t rand_xorshift(uint32_t rng_state) {    
  rng_state ^= (rng_state << 13);
  rng_state ^= (rng_state >> 17);    
  rng_state ^= (rng_state << 5);    
  return rng_state;
}
static inline
uint64_t rand_xorshift64(uint64_t rng_state) {    
  rng_state ^= (rng_state << 13);
  rng_state ^= (rng_state >> 35);    
  rng_state ^= (rng_state << 30);    
  return rng_state;
}

inline
Move64_t Board64_t::get_rand_white_move(const Lfr_t& lfr) { 
  uint64_t nb_wl = count64(lfr.left);
  uint64_t nb_wf = count64(lfr.forward);
  uint64_t nb_wr = count64(lfr.right);
  seed = rand_xorshift(seed);
  uint32_t move_id = seed%uint32_t(nb_wf+nb_wl+nb_wr); 
  Move64_t ret;
  if(move_id < nb_wf) {
    ret.pf = select_move(lfr.forward, move_id); 
    ret.pi = ret.pf<<8;
  } else {
    move_id -= nb_wf;
    if(move_id < nb_wl) {
      ret.pf = select_move(lfr.left, move_id);
      ret.pi = ret.pf<<9;
    } else {
      move_id -= nb_wl;
      ret.pf = select_move(lfr.right, move_id);
      ret.pi = ret.pf<<7;
    }
  }
  return ret;
}
inline
Move64_t Board64_t::get_rand_white_move() {
  uint64_t wl = white_left();
  uint64_t wf = white_forward();
  uint64_t wr = white_right();
  return get_rand_white_move(Lfr_t(wl, wf, wr));
}
inline
void Board64_t::rand_white_move() {
  apply_white_move(get_rand_white_move());
}
inline
Move64_t Board64_t::get_rand_black_move(const Lfr_t& lfr) {
  uint64_t nb_wl = count64(lfr.left);
  uint64_t nb_wf = count64(lfr.forward);
  uint64_t nb_wr = count64(lfr.right);
  seed = rand_xorshift(seed);
  uint32_t move_id = seed%uint32_t(nb_wl+nb_wf+nb_wr); 
  Move64_t ret;
  if(move_id < nb_wf) {
    ret.pf = select_move(lfr.forward, move_id);  
    ret.pi = ret.pf>>8;
  } else {
    move_id -= nb_wf;
    if(move_id < nb_wl) {
      ret.pf = select_move(lfr.left, move_id);
      ret.pi = ret.pf>>7;
    } else {
      move_id -= nb_wl;
      ret.pf = select_move(lfr.right, move_id);
      ret.pi = ret.pf>>9;
    }
  }
  return ret;
}
inline
Move64_t Board64_t::get_rand_black_move() {
  uint64_t wl = black_left();
  uint64_t wf = black_forward();
  uint64_t wr = black_right();  
  return get_rand_black_move(Lfr_t(wl, wf, wr));
}
inline
void Board64_t::rand_black_move() {
  apply_black_move(get_rand_black_move());
}
inline 
Move64_t Board64_t::get_rand_move(const Lfr_t& lfr, bool _white) {
  if(_white) return get_rand_white_move(lfr);
  else return get_rand_black_move(lfr);
}
inline
Move64_t Board64_t::get_rand_move(bool _white) {
  if(_white) return get_rand_white_move();
  else return get_rand_black_move();
}
inline 
void Board64_t::rand_move(bool _white) {
  if(_white) rand_white_move();
  else rand_black_move();
}
inline 
void Board64_t::seq_playout(bool _white) {
  while(1) {
    if(_white) {      
      rand_white_move();
      if(white_win()) break;
    } else {
      rand_black_move();
      if(black_win()) break;
    }
    _white = ! _white;
  }
}

// g++ -std=c++11 -Wall -O3
// Apple M1 Max : 3.119.200.000 per second
// AMD EPYC 7643 48-Core Base Clock 2.3GHz : 1.632.510.000 per second
void print_playout_perf_per_sec(bool _print) {
  Board64_t board;
  Board64_t turn0_board;
  auto begin = std::chrono::steady_clock::now();
  uint64_t nb_playout = 0ULL;
  uint64_t nb_playout_eval_clock = 0ULL;  
  uint32_t seed = 1;
  while(1) {
    board = turn0_board;
    board.seed = seed;
    board.seq_playout(true);
    nb_playout = nb_playout+1ULL;
    nb_playout_eval_clock = nb_playout_eval_clock+1ULL;
    if(nb_playout_eval_clock == 10000ULL) {
      auto end = std::chrono::steady_clock::now();
      auto elapsed = (end - begin).count();
      if(elapsed > 1E9) break;
      nb_playout_eval_clock = 0ULL;
    } 
    seed = seed+1;
  }
  if(_print) fprintf(stderr, "nb_playout %" PRIu64 " per second\n", nb_playout);
}

#endif /* BKBB64_H */
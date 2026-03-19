// random_hex.cpp — joueur aléatoire pour le jeu de Hex 11x11
// Usage : ./TC_MG_random_hex BOARD PLAYER [time_s]

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <string>
#include "hexbb.h"

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));
    if (argc < 3) {
        fprintf(stderr, "usage: %s BOARD PLAYER [time_s]\n", argv[0]);
        return 1;
    }

    HexBoard board(argv[1]);
    HexMove m = board.get_rand_move();
    if (m.pos < 0) {
        printf("resign\n");
        return 1;
    }
    printf("%s\n", m.to_str().c_str());
    return 0;
}

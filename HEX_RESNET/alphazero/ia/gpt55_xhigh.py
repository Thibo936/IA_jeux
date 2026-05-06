# gpt55_xhigh.py -- Joueur heuristique rapide pour Hex 11x11
# Interface CLI : python gpt55_xhigh.py BOARD PLAYER [time_s]

import heapq
import math
import os
import sys
import time


_dir = os.path.dirname(os.path.abspath(__file__))
_train = os.path.join(os.path.dirname(_dir), "train")
for _p in (_dir, _train):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hex_env import HexEnv

try:
    from config import BOARD_SIZE, NUM_CELLS
except ImportError:  # pragma: no cover - fallback pour execution isolee
    BOARD_SIZE = 11
    NUM_CELLS = BOARD_SIZE * BOARD_SIZE


INF_DIST = 10_000.0
WIN_SCORE = 1_000_000.0
TIME_MARGIN = 0.94

# Coordonnees axiales compatibles avec le plateau row-major du moteur.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1))
BRIDGE_OFFSETS = tuple(
    (
        DIRECTIONS[i][0] + DIRECTIONS[(i + 1) % 6][0],
        DIRECTIONS[i][1] + DIRECTIONS[(i + 1) % 6][1],
        DIRECTIONS[i],
        DIRECTIONS[(i + 1) % 6],
    )
    for i in range(6)
)


def _dans_plateau(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _pos(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def _rc(pos: int) -> tuple[int, int]:
    return divmod(pos, BOARD_SIZE)


def _precompute_voisins() -> tuple[tuple[int, ...], ...]:
    voisins: list[tuple[int, ...]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cur = []
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if _dans_plateau(nr, nc):
                    cur.append(_pos(nr, nc))
            voisins.append(tuple(cur))
    return tuple(voisins)


VOISINS = _precompute_voisins()
VOISINS_SET = tuple(set(v) for v in VOISINS)


class _TimeoutRecherche(Exception):
    pass


class GPT55XHighPlayer:
    """
    Joueur sans reseau neuronal : plus court chemin Hex + recherche alpha-beta.

    L'objectif est d'etre stable sous un budget court (~0.5 s/coup). Le score
    principal compare les distances de connexion Nord-Sud et Ouest-Est, puis une
    recherche selective verifie les meilleurs coups tactiques.
    """

    def __init__(self):
        self.last_stats: dict = {}
        self.nodes = 0
        self.deadline = 0.0
        self.root_blue = True
        self.best_depth = 0
        self.best_score = 0.0

    def select_move(self, env: HexEnv, time_s: float = 0.5) -> int:
        debut = time.perf_counter()
        coups = [int(m) for m in env.get_legal_moves()]
        if not coups:
            self.last_stats = {"score": 0, "nodes": 0, "depth": 0, "time": 0.0}
            return -1

        self.nodes = 0
        self.root_blue = bool(env.blue_to_play)
        budget = max(0.03, min(float(time_s), 0.45))
        self.deadline = debut + budget * TIME_MARGIN

        pierres_root = self._nombre_pierres(env, self.root_blue)
        pierres_adv = self._nombre_pierres(env, not self.root_blue)

        gagnants = []
        if pierres_root >= BOARD_SIZE - 1:
            gagnants = [m for m in coups if self._gagne_si_pose(env, m, self.root_blue)]
        if gagnants:
            coup = self._meilleur_coup_simple(env, gagnants, self.root_blue)
            self._maj_stats(debut, WIN_SCORE, 0)
            return coup

        blocs = []
        if pierres_adv >= BOARD_SIZE - 1:
            blocs = [m for m in coups if self._gagne_si_pose(env, m, not self.root_blue)]
        if blocs:
            coup = self._meilleur_coup_simple(env, blocs, self.root_blue)
            self._maj_stats(debut, WIN_SCORE * 0.5, 0)
            return coup

        ordonnes = self._ordonner_coups(
            env,
            coups,
            self.root_blue,
            min(len(coups), 28),
            lourd=True,
        )
        meilleur = ordonnes[0]
        meilleur_score = -math.inf
        profondeur_reussie = 0

        max_depth = self._profondeur_max(len(coups))
        try:
            for profondeur in range(1, max_depth + 1):
                if time.perf_counter() >= self.deadline:
                    break

                limite = self._limite_racine(len(coups), profondeur)
                candidats = ordonnes[:limite]
                score_p = -math.inf
                coup_p = meilleur
                alpha = -WIN_SCORE
                beta = WIN_SCORE
                terminee = False

                for coup in candidats:
                    self._check_time()
                    was_blue = bool(env.blue_to_play)
                    env.apply_move(coup)
                    try:
                        score = self._recherche(env, profondeur - 1, alpha, beta, 1)
                    finally:
                        env.undo_move(coup, was_blue)

                    if score > score_p:
                        score_p = score
                        coup_p = coup
                    if score > alpha:
                        alpha = score
                    terminee = True

                if terminee:
                    meilleur = coup_p
                    meilleur_score = score_p
                    profondeur_reussie = profondeur

        except _TimeoutRecherche:
            pass

        if meilleur_score == -math.inf:
            meilleur_score = self._score_coup_lourd(env, meilleur, self.root_blue)

        self._maj_stats(debut, meilleur_score, profondeur_reussie)
        return int(meilleur)

    def _recherche(
        self,
        env: HexEnv,
        profondeur: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        self.nodes += 1
        if (self.nodes & 31) == 0:
            self._check_time()

        score_terminal = self._score_terminal(env, ply)
        if score_terminal is not None:
            return score_terminal
        if profondeur <= 0:
            return self._evaluer(env)

        coups = [int(m) for m in env.get_legal_moves()]
        if not coups:
            return self._evaluer(env)

        joueur_bleu = bool(env.blue_to_play)
        maximise = joueur_bleu == self.root_blue
        limite = self._limite_noeud(len(coups), profondeur)
        ordonnes = self._ordonner_coups(env, coups, joueur_bleu, limite, lourd=False)

        if maximise:
            valeur = -WIN_SCORE
            for coup in ordonnes:
                was_blue = bool(env.blue_to_play)
                env.apply_move(coup)
                try:
                    score = self._recherche(env, profondeur - 1, alpha, beta, ply + 1)
                finally:
                    env.undo_move(coup, was_blue)

                if score > valeur:
                    valeur = score
                if valeur > alpha:
                    alpha = valeur
                if alpha >= beta:
                    break
            return valeur

        valeur = WIN_SCORE
        for coup in ordonnes:
            was_blue = bool(env.blue_to_play)
            env.apply_move(coup)
            try:
                score = self._recherche(env, profondeur - 1, alpha, beta, ply + 1)
            finally:
                env.undo_move(coup, was_blue)

            if score < valeur:
                valeur = score
            if valeur < beta:
                beta = valeur
            if alpha >= beta:
                break
        return valeur

    def _score_terminal(self, env: HexEnv, ply: int) -> float | None:
        dist_root = self._distance_connexion(env, self.root_blue, ponderee=False)
        if dist_root <= 0.0:
            return WIN_SCORE - ply
        dist_adv = self._distance_connexion(env, not self.root_blue, ponderee=False)
        if dist_adv <= 0.0:
            return -WIN_SCORE + ply
        return None

    def _evaluer(self, env: HexEnv) -> float:
        score = self._score_perspective(env, self.root_blue)
        score += 7.0 if bool(env.blue_to_play) == self.root_blue else -7.0
        return score

    def _score_perspective(self, env: HexEnv, joueur_bleu: bool) -> float:
        ami = min(self._distance_connexion(env, joueur_bleu, ponderee=True), 40.0)
        adv = min(self._distance_connexion(env, not joueur_bleu, ponderee=True), 40.0)

        if ami <= 0.0:
            return WIN_SCORE
        if adv <= 0.0:
            return -WIN_SCORE

        structure = self._bonus_structure(env, joueur_bleu)
        structure -= self._bonus_structure(env, not joueur_bleu)
        return (adv - ami) * 1200.0 + structure * 14.0

    def _ordonner_coups(
        self,
        env: HexEnv,
        coups: list[int],
        joueur_bleu: bool,
        limite: int | None,
        lourd: bool,
    ) -> list[int]:
        if limite is not None:
            pre_limite = min(len(coups), max(limite * 2, limite + 6))
        else:
            pre_limite = len(coups)

        if pre_limite < len(coups):
            pre_scores = [
                (self._score_local(env, coup, joueur_bleu), coup)
                for coup in coups
            ]
            pre_scores.sort(reverse=True)
            coups = [coup for _, coup in pre_scores[:pre_limite]]

        scores: list[tuple[float, int]] = []
        for coup in coups:
            if lourd:
                score = self._score_coup_lourd(env, coup, joueur_bleu)
            else:
                score = self._score_coup_rapide(env, coup, joueur_bleu)
            scores.append((score, coup))

        scores.sort(reverse=True)
        if limite is not None:
            scores = scores[:limite]
        return [coup for _, coup in scores]

    def _score_coup_lourd(self, env: HexEnv, coup: int, joueur_bleu: bool) -> float:
        r, c = _rc(coup)
        plateau = env.blue if joueur_bleu else env.red
        plateau[r, c] = True
        try:
            score = self._score_perspective(env, joueur_bleu)
        finally:
            plateau[r, c] = False
        return score + self._score_local(env, coup, joueur_bleu)

    def _score_coup_rapide(self, env: HexEnv, coup: int, joueur_bleu: bool) -> float:
        score = self._score_local(env, coup, joueur_bleu)
        if self._gagne_si_pose(env, coup, joueur_bleu):
            score += WIN_SCORE
        elif self._gagne_si_pose(env, coup, not joueur_bleu):
            score += WIN_SCORE * 0.7
        return score

    def _score_local(self, env: HexEnv, coup: int, joueur_bleu: bool) -> float:
        r, c = _rc(coup)
        ami = env.blue if joueur_bleu else env.red
        adv = env.red if joueur_bleu else env.blue

        dr = r - (BOARD_SIZE - 1) / 2.0
        dc = c - (BOARD_SIZE - 1) / 2.0
        dist_centre = max(abs(dr), abs(dc), abs(dr + dc))
        score = (BOARD_SIZE - dist_centre) * 3.0

        voisins_amis: list[int] = []
        voisins_adverses: list[int] = []
        for voisin in VOISINS[coup]:
            vr, vc = _rc(voisin)
            if ami[vr, vc]:
                voisins_amis.append(voisin)
            elif adv[vr, vc]:
                voisins_adverses.append(voisin)

        score += len(voisins_amis) * 16.0
        score += len(voisins_adverses) * 8.0
        score += self._paires_non_adjacentes(voisins_amis) * 13.0
        score += self._paires_non_adjacentes(voisins_adverses) * 9.0

        if joueur_bleu:
            if r == 0 or r == BOARD_SIZE - 1:
                score += 5.0
            score += (BOARD_SIZE - 1 - abs((BOARD_SIZE - 1) - 2 * r)) * 0.7
        else:
            if c == 0 or c == BOARD_SIZE - 1:
                score += 5.0
            score += (BOARD_SIZE - 1 - abs((BOARD_SIZE - 1) - 2 * c)) * 0.7

        return score

    def _paires_non_adjacentes(self, cellules: list[int]) -> int:
        total = 0
        for i, a in enumerate(cellules):
            voisins_a = VOISINS_SET[a]
            for b in cellules[i + 1 :]:
                if b not in voisins_a:
                    total += 1
        return total

    def _bonus_structure(self, env: HexEnv, joueur_bleu: bool) -> float:
        ami = env.blue if joueur_bleu else env.red
        adv = env.red if joueur_bleu else env.blue
        bonus = 0.0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if not ami[r, c]:
                    continue

                pos = _pos(r, c)
                for voisin in VOISINS[pos]:
                    if voisin <= pos:
                        continue
                    vr, vc = _rc(voisin)
                    if ami[vr, vc]:
                        bonus += 1.0

                if joueur_bleu:
                    if r == 0 or r == BOARD_SIZE - 1:
                        bonus += 1.3
                elif c == 0 or c == BOARD_SIZE - 1:
                    bonus += 1.3

                for br, bc, d1, d2 in BRIDGE_OFFSETS:
                    nr, nc = r + br, c + bc
                    if not _dans_plateau(nr, nc) or not ami[nr, nc]:
                        continue

                    ar, ac = r + d1[0], c + d1[1]
                    br2, bc2 = r + d2[0], c + d2[1]
                    if not (_dans_plateau(ar, ac) and _dans_plateau(br2, bc2)):
                        continue
                    if not adv[ar, ac] and not adv[br2, bc2]:
                        bonus += 0.7

        return bonus

    def _distance_connexion(self, env: HexEnv, joueur_bleu: bool, ponderee: bool) -> float:
        ami = env.blue if joueur_bleu else env.red
        adv = env.red if joueur_bleu else env.blue
        dist = [INF_DIST] * NUM_CELLS
        tas: list[tuple[float, int]] = []

        if joueur_bleu:
            sources = [_pos(0, c) for c in range(BOARD_SIZE)]
            cibles = {_pos(BOARD_SIZE - 1, c) for c in range(BOARD_SIZE)}
        else:
            sources = [_pos(r, 0) for r in range(BOARD_SIZE)]
            cibles = {_pos(r, BOARD_SIZE - 1) for r in range(BOARD_SIZE)}

        for source in sources:
            r, c = _rc(source)
            if adv[r, c]:
                continue
            w = self._poids_case(env, source, joueur_bleu, ponderee)
            dist[source] = w
            heapq.heappush(tas, (w, source))

        while tas:
            d, pos = heapq.heappop(tas)
            if d != dist[pos]:
                continue
            if pos in cibles:
                return d

            for voisin in VOISINS[pos]:
                vr, vc = _rc(voisin)
                if adv[vr, vc]:
                    continue
                nd = d + self._poids_case(env, voisin, joueur_bleu, ponderee)
                if nd < dist[voisin]:
                    dist[voisin] = nd
                    heapq.heappush(tas, (nd, voisin))

        return INF_DIST

    def _poids_case(self, env: HexEnv, pos: int, joueur_bleu: bool, ponderee: bool) -> float:
        r, c = _rc(pos)
        ami = env.blue if joueur_bleu else env.red
        adv = env.red if joueur_bleu else env.blue
        if ami[r, c]:
            return 0.0
        if adv[r, c]:
            return INF_DIST
        if not ponderee:
            return 1.0

        adj_ami = 0
        adj_adv = 0
        for voisin in VOISINS[pos]:
            vr, vc = _rc(voisin)
            if ami[vr, vc]:
                adj_ami += 1
            elif adv[vr, vc]:
                adj_adv += 1

        poids = 1.0 - min(0.36, adj_ami * 0.09) + min(0.18, adj_adv * 0.04)
        if joueur_bleu and (r == 0 or r == BOARD_SIZE - 1):
            poids -= 0.05
        elif (not joueur_bleu) and (c == 0 or c == BOARD_SIZE - 1):
            poids -= 0.05
        return max(0.42, poids)

    def _gagne_si_pose(self, env: HexEnv, coup: int, joueur_bleu: bool) -> bool:
        r, c = _rc(coup)
        plateau = env.blue if joueur_bleu else env.red
        if self._nombre_pierres(env, joueur_bleu) < BOARD_SIZE - 1:
            return False
        if plateau[r, c]:
            return False
        plateau[r, c] = True
        try:
            return self._distance_connexion(env, joueur_bleu, ponderee=False) <= 0.0
        finally:
            plateau[r, c] = False

    def _nombre_pierres(self, env: HexEnv, joueur_bleu: bool) -> int:
        plateau = env.blue if joueur_bleu else env.red
        return int(plateau.sum())

    def _meilleur_coup_simple(self, env: HexEnv, coups: list[int], joueur_bleu: bool) -> int:
        return max(coups, key=lambda m: self._score_local(env, m, joueur_bleu))

    def _profondeur_max(self, nb_coups: int) -> int:
        if nb_coups > 85:
            return 3
        if nb_coups > 45:
            return 4
        return 5

    def _limite_racine(self, nb_coups: int, profondeur: int) -> int:
        if profondeur <= 1:
            return min(nb_coups, 28)
        if profondeur == 2:
            return min(nb_coups, 22)
        if profondeur == 3:
            return min(nb_coups, 14)
        return min(nb_coups, 10)

    def _limite_noeud(self, nb_coups: int, profondeur: int) -> int:
        if profondeur <= 1:
            return min(nb_coups, 10)
        if profondeur == 2:
            return min(nb_coups, 8)
        return min(nb_coups, 6)

    def _check_time(self) -> None:
        if time.perf_counter() >= self.deadline:
            raise _TimeoutRecherche

    def _maj_stats(self, debut: float, score: float, profondeur: int) -> None:
        elapsed = time.perf_counter() - debut
        self.best_score = float(score)
        self.best_depth = int(profondeur)
        self.last_stats = {
            "score": int(round(max(-WIN_SCORE, min(WIN_SCORE, score)))),
            "nodes": int(self.nodes),
            "depth": int(profondeur),
            "time": float(elapsed),
        }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python gpt55_xhigh.py BOARD PLAYER [time_s]", file=sys.stderr)
        print("  BOARD  : 121 chars ('.' 'O' '@')", file=sys.stderr)
        print("  PLAYER : 'O' (Blue) ou '@' (Red)", file=sys.stderr)
        sys.exit(1)

    _env = HexEnv.from_string(sys.argv[1], sys.argv[2])
    _time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    _player = GPT55XHighPlayer()
    _move = _player.select_move(_env, _time_s)
    stats = _player.last_stats

    print(
        f"SCORE:{stats.get('score', 0)} "
        f"NODES:{stats.get('nodes', 0)} "
        f"DEPTH:{stats.get('depth', 0)} "
        f"TIME:{stats.get('time', 0.0):.3f}",
        file=sys.stderr,
    )
    if _move >= 0:
        print(_env.pos_to_str(_move))
    else:
        print("PASS")

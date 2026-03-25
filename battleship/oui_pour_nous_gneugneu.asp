#const n = 6.

% Grille nxn
r(1..n).
c(1..n).
cell(R,C) :- r(R), c(C).

% eau bateau
{ ship(R,C) } :- cell(R,C).
water(R,C) :- cell(R,C), not ship(R,C).

adj4(R,C,R+1,C) :- cell(R,C), cell(R+1,C).
adj4(R,C,R-1,C) :- cell(R,C), cell(R-1,C).
adj4(R,C,R,C+1) :- cell(R,C), cell(R,C+1).
adj4(R,C,R,C-1) :- cell(R,C), cell(R,C-1).

diag(R,C,R+1,C+1) :- cell(R,C), cell(R+1,C+1).
diag(R,C,R+1,C-1) :- cell(R,C), cell(R+1,C-1).
diag(R,C,R-1,C+1) :- cell(R,C), cell(R-1,C+1).
diag(R,C,R-1,C-1) :- cell(R,C), cell(R-1,C-1).

% pas de diag
:- ship(R1,C1), ship(R2,C2), diag(R1,C1,R2,C2).

deg(R,C,N) :- ship(R,C), N = #count { R2,C2 : adj4(R,C,R2,C2), ship(R2,C2) }.

% max 4
:- deg(R,C,N), N > 2.

left_n(R,C)  :- ship(R,C), ship(R,C-1).
right_n(R,C) :- ship(R,C), ship(R,C+1).
up_n(R,C)    :- ship(R,C), ship(R-1,C).
down_n(R,C)  :- ship(R,C), ship(R+1,C).

hpair(R,C) :- left_n(R,C), right_n(R,C).
vpair(R,C) :- up_n(R,C), down_n(R,C).

% les bateaux plier
:- ship(R,C), deg(R,C,2), not hpair(R,C), not vpair(R,C).

:- r(R), c(C), C <= n-4, ship(R,C), ship(R,C+1), ship(R,C+2), ship(R,C+3), ship(R,C+4).
:- r(R), c(C), R <= n-4, ship(R,C), ship(R+1,C), ship(R+2,C), ship(R+3,C), ship(R+4,C).

% ligne R ou C doit avoir exactement N cases de bateau
:- r(R), row_req(R, N), #count { C : ship(R,C) } != N.
:- c(C), col_req(C, N), #count { R : ship(R,C) } != N.

% tailles des bateaux
% Un bateau de taille 1 (sous-marin) est une case sans voisins orthogonaux
ship_size(R,C,1) :- ship(R,C), deg(R,C,0).

% start ligne col
h_start(R,C) :- ship(R,C), not ship(R,C-1), ship(R,C+1).
v_start(R,C) :- ship(R,C), not ship(R-1,C), ship(R+1,C).

% long ligne 4
ship_size(R,C,2) :- h_start(R,C), not ship(R,C+2).
ship_size(R,C,3) :- h_start(R,C), ship(R,C+2), not ship(R,C+3).
ship_size(R,C,4) :- h_start(R,C), ship(R,C+2), ship(R,C+3), not ship(R,C+4).

% Long col
ship_size(R,C,2) :- v_start(R,C), not ship(R+2,C).
ship_size(R,C,3) :- v_start(R,C), ship(R+2,C), not ship(R+3,C).
ship_size(R,C,4) :- v_start(R,C), ship(R+2,C), ship(R+3,C), not ship(R+4,C).

% Prédicats vérification
has_size(S) :- ship_size(_,_,S).

% au moins 1 bateau de chaque (old)
:- not has_size(1).
:- not has_size(2).
:- not has_size(3).

#maximize { 1,R,C : ship(R,C) }.

#show ship/2.
#show water/2.
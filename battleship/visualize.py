import json
import subprocess
import sys

#asp_file = "oui_pour_nous.asp"
asp_file = "oui_pour_nous_gneugneu.asp"

# n = 4
# rows = [2,1,0,3]
# cols = [1,2,1,2]
n = 6
rows = [4, 1, 3, 0, 2, 0]
cols = [3, 0, 2, 0, 2, 3]
# n = 8
# rows = [3,0,4,2,2,4,2,3]
# cols = [4,1,4,1,4,2,3,1]
# n = 10
# rows = [0,4,2,3,1,2,0,4,0,2]
# cols = [1,3,1,2,1,4,1,0,5,0]
# n = 11 
# rows = [3,1,7,2,3,1,2,2,5,4,5]
# cols = [2,4,5,5,2,1,6,0,6,0,4]
# n = 12
# rows = [1,5,2,0,1,5,1,3,4,0,0,3]
# cols = [1,0,6,2,3,1,1,2,1,1,3,4]
#n = 14
#rows = [0,2,0,1,3,0,4,2,0,1,0,1,2,4]
#cols = [0,2,4,0,0,1,3,3,2,1,1,1,1,1]

# gen ASP
facts = []
for i, count in enumerate(rows):
    facts.append(f"row_req({i+1}, {count}).")
for i, count in enumerate(cols):
    facts.append(f"col_req({i+1}, {count}).")

asp_facts = "\n".join(facts)
print("Config Python :")
print(asp_facts)
print("-" * 30)

print("Clingo :")

try:
    result = subprocess.run(
        ["clingo", asp_file, "-", f"-c", f"n={n}", "--outf=2"],
        input=asp_facts,
        capture_output=True,
        text=True
    )
except FileNotFoundError:
    print("Erreur : Clingo")
    sys.exit(1)

try:
    data = json.loads(result.stdout)
except json.JSONDecodeError:
    print("Erreur decode JSON :")
    print(result.stdout)
    sys.exit(1)

# Cherche les solutions dans les appels Clingo
calls = data.get("Call", [])
if not calls:
    print("Aucun appel exécuté par Clingo")
    sys.exit(1)

witnesses = calls[-1].get("Witnesses", [])
if not witnesses:
    print("Aucune solution trouvée par Clingo pour cette grille")
    sys.exit(1)

# prend la dernière solution.
optimal_solution = witnesses[-1]["Value"]

# init grille vide
grid = [["~ " for _ in range(n)] for _ in range(n)]

# Place les bateaux sur la grille
ships_count = 0
for predicate in optimal_solution:
    if predicate.startswith("ship("):
        # ship(R,C)
        content = predicate[5:-1] 
        r, c = map(int, content.split(","))
        grid[r-1][c-1] = "B "
        ships_count += 1

print(f"\n Meilleur résultat : {ships_count} cases occupées par des bateaux\n")

# Tête des colonnes
col_header = "    " + " ".join(f"{c:2}" for c in cols)
print(col_header)
print("   +" + "-" * (n * 3))

for i, row in enumerate(grid):
    # lignes
    print(f"{rows[i]:2} | " + " ".join(row))
print("\n")

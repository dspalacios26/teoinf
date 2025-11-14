from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import networkx as nx

# ==== Tipos ====
Edge = Tuple[int, int]
Matching = FrozenSet[Edge]
EdgeWeightMap = Dict[Edge, float]


# ==== Utilidades básicas ====
def canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def normalize_matching(raw: Iterable[Iterable[int]]) -> Matching:
    edges: Set[Edge] = set()
    for pair in raw:
        u, v = pair
        if u == v:
            continue
        edges.add(canonical_edge(int(u), int(v)))
    return frozenset(edges)


def fast_max_weight_matching(graph: nx.Graph) -> Matching:
    raw = nx.max_weight_matching(graph, maxcardinality=False, weight="weight")
    return normalize_matching(raw)


# ============================================================
#   NUEVAS FUNCIONES: DISTANCIAS Y VERIFICACIÓN (PAPER)
# ============================================================

def weighted_hamming_distance(M1: Matching, M2: Matching, weights: EdgeWeightMap) -> float:
    dist = 0.0
    all_edges = set(M1) | set(M2)
    for e in all_edges:
        if (e in M1) != (e in M2):
            dist += weights[e]
    return dist


def min_pairwise_distance(matchings: List[Matching], weights: EdgeWeightMap) -> float:
    if len(matchings) < 2:
        return 0.0
    best = float("inf")
    for i in range(len(matchings)):
        for j in range(i + 1, len(matchings)):
            d = weighted_hamming_distance(matchings[i], matchings[j], weights)
            best = min(best, d)
    return best


def estimate_Psi(graph: nx.Graph, weights: EdgeWeightMap,
                 k: int, r: int, delta: float,
                 max_iterations: int, trials: int = 3) -> float:
    """Estimación empírica del valor Ψ."""
    best = 0.0
    import random

    for _ in range(trials):
        seed = random.randint(1, 10**9)
        sols, _ = orchestrate_diverse_matchings(
            graph.copy(), weights, k, r, delta,
            max_iterations, seed, silent=True
        )
        d = min_pairwise_distance(sols, weights)
        best = max(best, d)

    return best


def verify_paper_bound(matchings: List[Matching], weights: EdgeWeightMap,
                       Psi_estimate: float, delta: float, mu: float = 0.5):

    min_dist = min_pairwise_distance(matchings, weights)
    rhs = (mu / 2) * Psi_estimate - delta

    print("\n================ PAPER GUARANTEE CHECK ================")
    print(f"Min pairwise distance: {min_dist:.4f}")
    print(f"Psi estimate:          {Psi_estimate:.4f}")
    print(f"(mu/2)*Psi - δ:        {rhs:.4f}")

    if min_dist >= rhs:
        print("✔ La desigualdad del paper SE CUMPLE.")
    else:
        print("✘ La desigualdad del paper NO se cumple.")
    print("=======================================================\n")


# ============================================================
#      ALGORITMO PRINCIPAL (con SILENT MODE integrado)
# ============================================================

def orchestrate_diverse_matchings(
    graph: nx.Graph,
    weights: EdgeWeightMap,
    k: int = 3,
    r: int = 20,
    delta: float = 0.3,
    max_iterations: int = 300,
    seed: int = 1,
    silent: bool = False,   # <<<<<<<<<< ADDED
) -> Tuple[List[Matching], int]:

    random.seed(seed)

    all_edges = [canonical_edge(u, v) for u, v in graph.edges()]
    dual: Dict[Edge, float] = {e: 1.0 for e in all_edges}

    selected: List[Matching] = []
    seen: Set[Matching] = set()

    eta = max(1e-4, min(0.1, delta * 0.1))

    # Cabecera (solo si NO es silent)
    if not silent:
        print("\n============================================================")
        print("             Running Diverse Matchings (FIXED)              ")
        print("============================================================\n")

    for it in range(1, max_iterations + 1):

        if not silent:
            print(f"[Progress] Attempt {it}/{max_iterations}: generating candidate matching...")

        # Actualizar pesos con ruido suave
        for u, v in graph.edges():
            e = canonical_edge(u, v)
            sigma = delta * 0.01
            noise = random.gauss(0, sigma) * weights[e]
            graph[u][v]["weight"] = weights[e] * dual[e] + noise

        # MATCHING exacto
        M = fast_max_weight_matching(graph)
        if not M:
            continue

        if len(M) > r:
            M = frozenset(sorted(M, key=lambda e: weights[e], reverse=True)[:r])

        if M not in seen:
            idx = len(selected) + 1
            selected.append(M)
            seen.add(M)

            if not silent:
                print(f"[Progress] Selected matching {idx}/{k} (|M|={len(M)})")

            if len(selected) >= k:
                break

        # Actualización dual
        used = set(M)
        for e in all_edges:
            if e in used:
                dual[e] *= (1 - eta * 0.5)
            else:
                dual[e] *= (1 + eta * 0.25)

        # Normalización
        avg = sum(dual.values()) / len(dual)
        inv = 1.0 / avg
        for e in dual:
            dual[e] *= inv

    return selected, len(selected)


# ============================================================
#                CARGA DE DATASET
# ============================================================

def load_dataset(path: Path) -> Tuple[nx.Graph, EdgeWeightMap]:
    if path.is_dir():
        file = path / "edges.csv"
    else:
        file = path

    import csv

    G = nx.Graph()
    w: EdgeWeightMap = {}

    with file.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            try:
                u, v, wt = int(row[0]), int(row[1]), float(row[2])
            except Exception:
                parts = " ".join(row).split()
                if len(parts) < 3:
                    continue
                u, v, wt = int(parts[0]), int(parts[1]), float(parts[2])

            e = canonical_edge(u, v)
            G.add_edge(u, v, weight=wt)
            w[e] = wt

    return G, w


# ============================================================
#                           CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("-r", type=int, default=20)
    p.add_argument("--delta", type=float, default=0.3)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    G, w = load_dataset(Path(a.dataset))

    matchings, _ = orchestrate_diverse_matchings(
        G, w, a.k, a.r, a.delta, a.max_iterations, a.seed, silent=False
    )

    print("\n============================================================")
    print("                           RESULTS                          ")
    print("============================================================\n")

    for i, M in enumerate(matchings, 1):
        total_w = sum(w[canonical_edge(u, v)] for u, v in M)
        print(f"Matching #{i} (|M|={len(M)}; total_weight={total_w:.2f})")
        for u, v in M:
            print(f"  {u} — {v}   (weight={w[canonical_edge(u, v)]:.2f})")
        print("------------------------------------------------------------")

    # === Verificación teórica del paper ===
    Psi_est = estimate_Psi(G, w, a.k, a.r, a.delta, a.max_iterations, trials=3)
    verify_paper_bound(matchings, w, Psi_est, a.delta)


if __name__ == "__main__":
    main()

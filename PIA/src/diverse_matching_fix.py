from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import networkx as nx

# ==== Tipos ====
Edge = Tuple[int, int]
Matching = FrozenSet[Edge]
EdgeWeightMap = Dict[Edge, float]


# ==== Utilidades básicas ====
def canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def normalize_matching(raw: Iterable[Iterable[int]]) -> Matching:
    """Convierte el matching crudo de networkx a un frozenset de aristas canónicas."""
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


# ==== Algoritmo principal, salida estilo 'primera imagen' ====
def orchestrate_diverse_matchings(
    graph: nx.Graph,
    weights: EdgeWeightMap,
    k: int = 3,
    r: int = 20,
    delta: float = 0.7,
    max_iterations: int = 300,
    seed: int = 1,
) -> Tuple[List[Matching], int]:

    random.seed(seed)

    all_edges = [canonical_edge(u, v) for u, v in graph.edges()]
    dual: Dict[Edge, float] = {e: 1.0 for e in all_edges}

    selected: List[Matching] = []
    seen: Set[Matching] = set()

    eta = max(1e-3, min(0.5, delta))

    print()
    print("============================================================")
    print("             Running Diverse Matchings (FIXED)              ")
    print("============================================================")
    print()

    for it in range(1, max_iterations + 1):
        print(f"[Progress] Attempt {it}/{max_iterations}: generating candidate matching...")

        # 1) Actualiza pesos efectivos
        for u, v in graph.edges():
            e = canonical_edge(u, v)
            graph[u][v]["weight"] = weights[e] * dual[e]

        # 2) Matching (exacto primeras iteraciones)
        if it <= 5:
            M = fast_max_weight_matching(graph)
        else:
            M_raw = nx.maximal_matching(graph)
            M = normalize_matching(M_raw)

        if not M:
            continue

        # 3) Recortar a r aristas
        if len(M) > r:
            M = frozenset(sorted(M, key=lambda e: weights[e], reverse=True)[:r])

        # 4) Registrar si es nuevo
        if M not in seen:
            idx = len(selected) + 1
            selected.append(M)
            seen.add(M)
            print(f"[Progress] Selected matching {idx}/{k} (|M|={len(M)})")

            if len(selected) >= k:
                break

        # 5) Actualiza duales
        used = set(M)
        for e in all_edges:
            if e in used:
                dual[e] *= (1 - eta)
            else:
                dual[e] *= (1 + eta)

        # 6) Normalización
        avg = sum(dual.values()) / len(dual)
        inv = 1.0 / avg
        for e in dual:
            dual[e] *= inv

    return selected, len(selected)


# ==== Carga de dataset ====
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


# ==== CLI ====
def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("-r", type=int, default=20)
    p.add_argument("--delta", type=float, default=0.7)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    G, w = load_dataset(Path(a.dataset))
    matchings, _ = orchestrate_diverse_matchings(
        G, w, a.k, a.r, a.delta, a.max_iterations, a.seed
    )

    print()
    print("============================================================")
    print("                           RESULTS                          ")
    print("============================================================")
    print()

    for i, M in enumerate(matchings, 1):
        total_w = sum(w[e] for e in M)
        print(f"Matching #{i} (|M|={len(M)})")
        for u, v in M:
            print(f"  {u} ↔ {v}  (weight={w[(u,v)]:.4f})")
        print(f"Total weight = {total_w:.4f}")
        print("------------------------------------------------------------")


if __name__ == "__main__":
    main()

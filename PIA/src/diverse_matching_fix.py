from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import networkx as nx
from tqdm import tqdm

# ==== Tipos ====
Edge = Tuple[int, int]
Matching = FrozenSet[Edge]
EdgeWeightMap = Dict[Edge, float]


# ==== Utilidades básicas (definidas aquí para no depender de otros módulos) ====
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


# ==== Algoritmo principal, versión optimizada ====
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

    pbar = tqdm(range(1, max_iterations + 1), desc="FIXED", ncols=100)

    for it in pbar:
        # 1) Actualiza pesos efectivos en el grafo
        for u, v in graph.edges():
            e = canonical_edge(u, v)
            graph[u][v]["weight"] = weights[e] * dual[e]

        # 2) Matching:
        #    - primeras iteraciones: exacto (max_weight_matching)
        #    - después: maximal_matching (mucho más rápido)
        if it <= 5:
            M = fast_max_weight_matching(graph)
        else:
            M_raw = nx.maximal_matching(graph)
            M = normalize_matching(M_raw)

        if not M:
            continue

        # 3) Recorta a lo más r aristas por matching
        if len(M) > r:
            M = frozenset(
                sorted(M, key=lambda e: weights[e], reverse=True)[:r]
            )

        # 4) Guarda si es nuevo
        if M not in seen:
            selected.append(M)
            seen.add(M)
            if len(selected) >= k:
                break

        # 5) Actualiza variables duales
        used = set(M)
        for e in all_edges:
            if e in used:
                dual[e] *= (1 - eta)
            else:
                dual[e] *= (1 + eta)

        # 6) Normalización para evitar overflow/underflow
        avg = sum(dual.values()) / len(dual)
        inv_avg = 1.0 / avg
        for e in dual:
            dual[e] *= inv_avg

    pbar.close()
    return selected, len(selected)


# ==== Carga de dataset (igual que en tu versión original) ====
def load_dataset(path: Path) -> Tuple[nx.Graph, EdgeWeightMap]:
    if path.is_dir():
        file = path / "edges_small.csv"
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

    print("\n[FIXED]")
    for i, M in enumerate(matchings, 1):
        total_w = sum(w[e] for e in M)
        print(f"M{i}: {len(M)} edges, total weight={total_w:.2f}")


if __name__ == "__main__":
    main()

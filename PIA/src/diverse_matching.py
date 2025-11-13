# src/diverse_matching.py
from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, FrozenSet

import networkx as nx
from networkx.algorithms.matching import max_weight_matching

import numpy as np

Node = int
Edge = Tuple[Node, Node]
Matching = FrozenSet[Edge]
EdgeWeightMap = Dict[Edge, float]


def canonical_edge(u: Node, v: Node) -> Edge:
    return (u, v) if u <= v else (v, u)


def normalize_matching(raw_matching: Iterable[Iterable[Node]]) -> Matching:
    edges: Set[Edge] = set()
    for pair in raw_matching:
        pair_tuple = tuple(pair)
        if len(pair_tuple) != 2:
            raise ValueError(f"Invalid edge in matching: {pair_tuple}")
        u, v = pair_tuple
        if u == v:
            continue
        edges.add(canonical_edge(int(u), int(v)))
    return frozenset(edges)


def weighted_collaboration_distance(weights: EdgeWeightMap, m1: Matching, m2: Matching) -> float:
    diff = (m1 - m2) | (m2 - m1)
    return sum(weights[edge] for edge in diff)


def min_pairwise_distance(matchings: Sequence[Matching], *, distance_fn) -> float:
    if len(matchings) < 2:
        return 0.0
    min_distance = math.inf
    for i, j in itertools.combinations(range(len(matchings)), 2):
        dist = distance_fn(matchings[i], matchings[j])
        min_distance = min(min_distance, dist)
    return min_distance if min_distance is not math.inf else 0.0


# -------------------------
# Fast data structures
# -------------------------
def _build_edge_index_map(base_weights: EdgeWeightMap):
    edges_list = list(base_weights.keys())
    # stable order
    edges_list.sort()
    idx_of_edge = {edge: i for i, edge in enumerate(edges_list)}
    base_arr = np.array([base_weights[edge] for edge in edges_list], dtype=float)
    return edges_list, idx_of_edge, base_arr


# -------------------------
# Oracles
# -------------------------
def greedy_matching_oracle(
    edges_list: List[Edge],
    weights_arr: np.ndarray,
    max_size: Optional[int] = None,
) -> Matching:
    """Greedy maximal weight matching (fast approximate oracle)."""
    n_edges = len(edges_list)
    idxs = np.argsort(weights_arr)[::-1]  # descending
    used_nodes = set()
    selected = []
    cap = max_size if max_size is not None else n_edges
    for i in idxs:
        if weights_arr[i] <= 0:
            break
        u, v = edges_list[i]
        if u in used_nodes or v in used_nodes:
            continue
        selected.append(canonical_edge(u, v))
        used_nodes.add(u)
        used_nodes.add(v)
        if len(selected) >= cap:
            break
    return frozenset(selected)


def exact_matching_oracle(
    graph: nx.Graph,
) -> Matching:
    raw_matching = max_weight_matching(graph, weight="weight")
    return normalize_matching(raw_matching)


def call_matching_oracle(
    graph: nx.Graph,
    edges_list: List[Edge],
    weights_arr: np.ndarray,
    *,
    max_size: Optional[int] = None,
    reference_weights_arr: Optional[np.ndarray] = None,
    mode: str = "exact",
) -> Matching:
    """
    mode: 'exact' uses networkx Edmonds algorithm on the graph (weights must be pushed to graph attributes),
          'greedy' uses a fast approximate oracle that does not modify the graph.
    """
    if mode == "greedy":
        return greedy_matching_oracle(edges_list, weights_arr, max_size=max_size)

    # exact: update graph edge weights in-place (single pass)
    # Assumes graph already contains all edges in edges_list
    # Use local variables for speed
    g = graph
    # iterate once, update weights
    for (u, v), w in zip(edges_list, weights_arr):
        # Use direct dict access to avoid overhead
        if g.has_edge(u, v):
            g[u][v]["weight"] = float(w)
    matching = exact_matching_oracle(g)

    if max_size is not None and len(matching) > max_size:
        ranking_weights = reference_weights_arr if reference_weights_arr is not None else weights_arr
        ordered = sorted(
            matching,
            key=lambda edge: float(ranking_weights[edges_list.index(edge)]) if edge in edges_list else 0.0,
            reverse=True,
        )
        matching = frozenset(ordered[:max_size])

    return matching


# -------------------------
# Vectorized temporal weights
# -------------------------
def compute_temporal_weights_vector(
    base_arr: np.ndarray,
    edges_list: List[Edge],
    history: Sequence[Matching],
    dual_weights: Sequence[float],
    edge_index: Dict[Edge, int],
    *,
    eta: float,
) -> np.ndarray:
    """
    Vectorized computation:
    weights' = base_weights * exp(-eta * sum_{j: edge in history[j]} dual_weights[j])
    """
    if len(history) != len(dual_weights):
        raise ValueError("History and dual weights must have the same length")

    E = base_arr.shape[0]
    if not history:
        return base_arr.copy()

    H = len(history)
    # Build history presence matrix shape (H, E) as float
    hist_mat = np.zeros((H, E), dtype=float)
    for h_idx, matching in enumerate(history):
        for edge in matching:
            idx = edge_index.get(edge)
            if idx is not None:
                hist_mat[h_idx, idx] = 1.0

    dual = np.array(dual_weights, dtype=float)  # shape (H,)
    # compute penalty per edge: dot(dual, hist_col) for each column -> hist_mat.T @ dual
    penalties = hist_mat.T.dot(dual)  # shape (E,)

    adjusted = base_arr * np.exp(-eta * penalties)
    # clip small negatives (numerical safety)
    adjusted[adjusted < 0] = 0.0
    return adjusted


# -------------------------
# Core search for one matching
# -------------------------
def find_single_diverse_matching(
    graph: nx.Graph,
    base_arr: np.ndarray,
    edges_list: List[Edge],
    edge_index: Dict[Edge, int],
    history: Sequence[Matching],
    * ,
    max_size: int,
    iterations: int,
    eta: float,
    rng: Optional[random.Random] = None,
    oracle_mode: str = "exact",
    jitter_scale: float = 0.01,
) -> Matching:
    if rng is None:
        rng = random.Random()

    if not history:
        # call oracle with base weights (vector)
        return call_matching_oracle(graph, edges_list, base_arr, max_size=max_size, reference_weights_arr=base_arr, mode=oracle_mode)

    # initialize duals
    dual_weights = [1.0 for _ in history]
    best_matching: Optional[Matching] = None
    best_min_distance = -math.inf

    for _ in range(max(1, iterations)):
        # compute temporal weights vectorized
        temporal_arr = compute_temporal_weights_vector(base_arr, edges_list, history, dual_weights, edge_index, eta=eta)

        # jitter (vectorized)
        jitter_factors = 1.0 + jitter_scale * (np.array([rng.uniform(-1.0, 1.0) for _ in range(len(temporal_arr))]))
        jittered = temporal_arr * jitter_factors
        # clamp
        jittered[jittered < 0] = 0.0

        candidate = call_matching_oracle(
            graph,
            edges_list,
            jittered,
            max_size=max_size,
            reference_weights_arr=base_arr,
            mode=oracle_mode,
        )
        if not candidate:
            continue

        if candidate in history:
            # increase duals multiplicatively to encourage shifting away
            dual_weights = [dw * (1.0 + eta) for dw in dual_weights]
            continue

        min_distance = min(
            weighted_collaboration_distance({e: float(base_arr[edge_index[e]]) for e in base_arr_index_to_edge(edge_index)}, candidate, past)
            for past in history
        )

        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_matching = candidate

        # update duals: increase by overlap weight
        for idx, past in enumerate(history):
            # overlap weight between candidate and past (vectorized via indices)
            overlap = 0.0
            for edge in (candidate & past):
                pos = edge_index.get(edge)
                if pos is not None:
                    overlap += float(base_arr[pos])
            dual_weights[idx] *= math.exp(eta * overlap)

        # renormalize duals
        total = sum(dual_weights)
        if total > 0:
            scale = len(dual_weights) / total
            dual_weights = [dw * scale for dw in dual_weights]

    if best_matching is None:
        return call_matching_oracle(graph, edges_list, base_arr, max_size=max_size, reference_weights_arr=base_arr, mode=oracle_mode)

    return best_matching


def base_arr_index_to_edge(edge_index: Dict[Edge, int]):
    # invert mapping
    inv = {i: e for e, i in edge_index.items()}
    # return a generator-like mapping function
    class Mapper(dict):
        pass
    m = {}
    for i, e in inv.items():
        m[e] = i
    return m


# -------------------------
# Top-level orchestrator
# -------------------------
def orchestrate_diverse_matchings(
    graph: nx.Graph,
    base_weights: EdgeWeightMap,
    k: int,
    r: int,
    *,
    delta: float,
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
    oracle_mode: str = "exact",
) -> Tuple[List[Matching], float]:
    if k <= 0:
        raise ValueError("k must be positive")
    if r <= 0:
        raise ValueError("r must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")

    rng = random.Random(seed)
    history: List[Matching] = []
    seen: Set[Matching] = set()

    # Precompute edge arrays for fast vector operations
    edges_list, edge_index, base_arr = _build_edge_index_map(base_weights)

    eta = min(0.5, delta / 2.0)
    base_scale = max(2.0, float(len(base_weights)))
    default_iterations = math.ceil((2.0 * math.log(base_scale)) / (delta ** 2))
    iterations = max_iterations or (k * default_iterations)

    for attempt in range(1, iterations + 1):
        if len(history) >= k:
            break

        print(
            f"[Progress] Attempt {attempt}/{iterations}: generating candidate matching... (oracle={oracle_mode})",
            flush=True,
        )

        candidate = find_single_diverse_matching(
            graph,
            base_arr,
            edges_list,
            edge_index,
            history,
            max_size=r,
            iterations=default_iterations,
            eta=eta,
            rng=rng,
            oracle_mode=oracle_mode,
        )

        if candidate in seen:
            print(f"[Progress] Attempt {attempt}/{iterations}: duplicate matching skipped", flush=True)
            continue
        if len(candidate) > r:
            print(
                f"[Progress] Attempt {attempt}/{iterations}: candidate exceeded size limit (|M|={len(candidate)} > r={r})",
                flush=True,
            )
            continue

        history.append(candidate)
        seen.add(candidate)
        print(
            f"[Progress] Selected matching {len(history)}/{k} after {attempt} attempts (|M|={len(candidate)})",
            flush=True,
        )

    if len(history) < k:
        print(
            f"[Progress] Stopped after {iterations} attempts with {len(history)} of {k} matchings.",
            flush=True,
        )
        raise RuntimeError(
            f"Unable to assemble {k} diverse matchings – only {len(history)} distinct matchings found."
        )

    distance_fn = lambda a, b: weighted_collaboration_distance(base_weights, a, b)
    min_distance = min_pairwise_distance(history, distance_fn=distance_fn)
    return history, min_distance


# -------------------------
# IO and CLI (unchanged semantics)
# -------------------------
def load_weighted_graph(resource: Path, *, edges_filename: Optional[str] = None) -> Tuple[nx.Graph, EdgeWeightMap]:
    resource = Path(resource)
    if not resource.exists():
        raise FileNotFoundError(f"Dataset not found: {resource}")

    if resource.is_dir():
        candidate_names = [edges_filename] if edges_filename else ["edges.csv", "edges.txt"]
        for candidate in candidate_names:
            if candidate and (resource / candidate).exists():
                path = resource / candidate
                break
        else:
            raise FileNotFoundError(
                f"No edges file found in {resource}. Expected one of: {', '.join(name for name in candidate_names if name)}"
            )
    else:
        path = resource

    graph = nx.Graph()
    weights: EdgeWeightMap = {}

    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [segment.strip() for segment in line.replace(",", " ").split() if segment.strip()]
            if len(parts) < 3:
                raise ValueError(
                    f"Expected three columns 'u v weight' on line {line_number} of {path}, found: {raw_line!r}"
                )

            u = int(parts[0])
            v = int(parts[1])
            w = float(parts[2])

            edge = canonical_edge(u, v)
            graph.add_edge(edge[0], edge[1], weight=w)
            weights[edge] = w

    return graph, weights


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the diverse matching solver on a weighted dataset (CSV folder or edge list).",
    )
    parser.add_argument(
        "dataset",
        help="Path to a dataset folder (with edges.csv) or a text file containing 'u v weight' per line",
    )
    parser.add_argument("-k", "--matchings", type=int, required=True, help="Number of matchings to generate")
    parser.add_argument("-r", "--max-size", type=int, required=True, help="Maximum number of edges allowed in each matching")
    parser.add_argument("--delta", type=float, required=True, help="Accuracy parameter from the paper (affects MWU iterations)")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional hard cap on the outer MWU loop iterations",
    )
    parser.add_argument(
        "--edges-filename",
        default=None,
        help="Override the default edges filename inside a dataset directory",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for tie-breaking")
    parser.add_argument(
        "--approx",
        choices=["exact", "greedy"],
        default="greedy",
        help="Oracle mode: 'exact' uses networkx max_weight_matching (slower, exact); 'greedy' uses a fast approximate oracle.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    graph, base_weights = load_weighted_graph(
        Path(args.dataset),
        edges_filename=args.edges_filename,
    )
    matchings, min_distance = orchestrate_diverse_matchings(
        graph,
        base_weights,
        k=args.matchings,
        r=args.max_size,
        delta=args.delta,
        max_iterations=args.max_iterations,
        seed=args.seed,
        oracle_mode=args.approx,
    )

    print(f"Generated {len(matchings)} diverse matchings")
    for idx, matching in enumerate(matchings, start=1):
        print("-" * 54)
        total_weight = sum(base_weights.get(edge, 0.0) for edge in matching)
        print(f"Matching #{idx} (|M|={len(matching)}; total_weight={total_weight:.2f})")
        for edge in sorted(matching):
            weight = base_weights.get(edge, 0.0)
            print(f"  {edge[0]} -- {edge[1]}  (weight={weight:.2f})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Callable

import networkx as nx
from networkx.algorithms.matching import max_weight_matching

Node = int
Edge = Tuple[Node, Node]
Matching = FrozenSet[Edge]
EdgeWeightMap = Dict[Edge, float]
MatroidOracle = Callable[[Set[Edge]], bool]


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


# ============================================================================
# Section 2.4.1: Matching Applications
# ============================================================================

def is_valid_matching(edges: Set[Edge]) -> bool:
    """
    Verify that a set of edges forms a valid matching.
    A valid matching has no two edges sharing a common vertex.
    
    Args:
        edges: Set of edges to verify
        
    Returns:
        True if edges form a valid matching, False otherwise
    """
    covered_nodes: Set[Node] = set()
    for u, v in edges:
        if u in covered_nodes or v in covered_nodes:
            return False
        covered_nodes.add(u)
        covered_nodes.add(v)
    return True


def matching_matroid_oracle(edges: Set[Edge]) -> bool:
    """
    Matroid oracle for the matching matroid.
    A set of edges is independent iff it forms a valid matching.
    
    Args:
        edges: Set of edges to check
        
    Returns:
        True if edges are independent in the matching matroid
    """
    return is_valid_matching(edges)


# ============================================================================
# Section 2.4.2: Matroid Applications
# ============================================================================

def uniform_matroid_oracle(k: int) -> MatroidOracle:
    """
    Create an oracle for a uniform matroid U(k, n).
    A set is independent iff its size is at most k.
    
    Args:
        k: Maximum size of independent sets
        
    Returns:
        Oracle function that checks independence
    """
    def oracle(edges: Set[Edge]) -> bool:
        return len(edges) <= k
    return oracle


def partition_matroid_oracle(partition: Dict[Edge, int], capacities: Dict[int, int]) -> MatroidOracle:
    """
    Create an oracle for a partition matroid.
    Elements are partitioned into groups, and a set is independent iff
    it contains at most capacity[i] elements from group i.
    
    Args:
        partition: Maps each edge to its partition group
        capacities: Maximum number of elements allowed from each group
        
    Returns:
        Oracle function that checks independence
    """
    def oracle(edges: Set[Edge]) -> bool:
        group_counts: Dict[int, int] = {}
        for edge in edges:
            group = partition.get(edge, 0)
            group_counts[group] = group_counts.get(group, 0) + 1
            if group_counts[group] > capacities.get(group, 0):
                return False
        return True
    return oracle


def graphic_matroid_oracle(graph_nodes: Set[Node]) -> MatroidOracle:
    """
    Create an oracle for a graphic matroid (forest matroid).
    A set of edges is independent iff it forms a forest (no cycles).
    
    Args:
        graph_nodes: Set of all nodes in the graph
        
    Returns:
        Oracle function that checks if edges form a forest
    """
    def oracle(edges: Set[Edge]) -> bool:
        if not edges:
            return True
        
        # Build a graph from edges and check for cycles using DFS
        adj: Dict[Node, List[Node]] = {node: [] for node in graph_nodes}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        
        visited: Set[Node] = set()
        
        def has_cycle(node: Node, parent: Optional[Node]) -> bool:
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, node):
                        return True
                elif neighbor != parent:
                    return True
            return False
        
        # Check each connected component
        for node in adj:
            if node not in visited:
                if has_cycle(node, None):
                    return False
        
        return True
    return oracle


def matroid_intersection(
    ground_set: Set[Edge],
    oracle1: MatroidOracle,
    oracle2: MatroidOracle,
    weights: EdgeWeightMap,
) -> Tuple[FrozenSet[Edge], float]:
    """
    Find maximum weight common independent set of two matroids.
    Uses a greedy approximation for matroid intersection.
    
    Args:
        ground_set: Set of all edges to consider
        oracle1: Independence oracle for first matroid
        oracle2: Independence oracle for second matroid
        weights: Weight map for edges
        
    Returns:
        Tuple of (common independent set, total weight)
    """
    # Greedy approach: sort by weight and add if both oracles accept
    sorted_edges = sorted(ground_set, key=lambda e: weights.get(e, 0.0), reverse=True)
    
    independent_set: Set[Edge] = set()
    total_weight = 0.0
    
    for edge in sorted_edges:
        candidate = independent_set | {edge}
        if oracle1(candidate) and oracle2(candidate):
            independent_set.add(edge)
            total_weight += weights.get(edge, 0.0)
    
    return frozenset(independent_set), total_weight


def diverse_matroid_matching(
    graph: nx.Graph,
    base_weights: EdgeWeightMap,
    matroid_oracle: MatroidOracle,
    k: int,
    r: int,
    *,
    delta: float,
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Matching], float]:
    """
    Generate k diverse matchings subject to an additional matroid constraint.
    Each matching must satisfy both:
    1. Be a valid matching (matching matroid)
    2. Satisfy the given matroid constraint
    
    Args:
        graph: Input graph
        base_weights: Edge weights
        matroid_oracle: Additional matroid constraint oracle
        k: Number of matchings to generate
        r: Maximum size of each matching
        delta: Accuracy parameter
        max_iterations: Optional iteration limit
        seed: Random seed
        
    Returns:
        Tuple of (list of matchings, minimum pairwise distance)
    """
    rng = random.Random(seed)
    history: List[Matching] = []
    seen: Set[Matching] = set()
    
    eta = min(0.5, delta / 2.0)
    base_scale = max(2.0, float(len(base_weights)))
    default_iterations = math.ceil((2.0 * math.log(base_scale)) / (delta ** 2))
    iterations = max_iterations or (k * default_iterations)
    
    for attempt in range(1, iterations + 1):
        if len(history) >= k:
            break
        
        print(
            f"[Matroid Progress] Attempt {attempt}/{iterations}: generating candidate...",
            flush=True,
        )
        
        # Generate candidate matching
        candidate = find_single_diverse_matching(
            graph,
            base_weights,
            history,
            max_size=r,
            iterations=default_iterations,
            eta=eta,
            rng=rng,
        )
        
        # Verify matroid constraint
        if not matroid_oracle(set(candidate)):
            print(
                f"[Matroid Progress] Attempt {attempt}: candidate violates matroid constraint",
                flush=True,
            )
            continue
        
        if candidate in seen:
            print(f"[Matroid Progress] Attempt {attempt}: duplicate skipped", flush=True)
            continue
        
        if len(candidate) > r:
            print(
                f"[Matroid Progress] Attempt {attempt}: size limit exceeded",
                flush=True,
            )
            continue
        
        history.append(candidate)
        seen.add(candidate)
        print(
            f"[Matroid Progress] Selected {len(history)}/{k} after {attempt} attempts",
            flush=True,
        )
    
    if len(history) < k:
        raise RuntimeError(
            f"Unable to find {k} matroid-constrained matchings – only {len(history)} found."
        )
    
    distance_fn = lambda a, b: weighted_collaboration_distance(base_weights, a, b)
    min_distance = min_pairwise_distance(history, distance_fn=distance_fn)
    return history, min_distance


def call_matching_oracle(
    graph: nx.Graph,
    weights: EdgeWeightMap,
    *,
    max_size: Optional[int] = None,
    reference_weights: Optional[EdgeWeightMap] = None,
) -> Matching:
    working_graph = nx.Graph()
    working_graph.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        edge = canonical_edge(u, v)
        weight = float(weights.get(edge, 0.0))
        working_graph.add_edge(u, v, weight=weight)
    raw_matching = max_weight_matching(working_graph, weight="weight")
    matching = normalize_matching(raw_matching)

    if max_size is not None and len(matching) > max_size:
        ranking_weights = reference_weights or weights
        ordered = sorted(
            matching,
            key=lambda edge: ranking_weights.get(edge, 0.0),
            reverse=True,
        )
        matching = frozenset(ordered[:max_size])

    return matching


def compute_temporal_weights(
    base_weights: EdgeWeightMap,
    history: Sequence[Matching],
    dual_weights: Sequence[float],
    *,
    eta: float,
) -> EdgeWeightMap:
    if len(history) != len(dual_weights):
        raise ValueError("History and dual weights must have the same length")
    adjusted: EdgeWeightMap = {}
    for edge, weight in base_weights.items():
        penalty = 0.0
        for dual, matching in zip(dual_weights, history):
            if edge in matching:
                penalty += dual
        adjusted_weight = weight * math.exp(-eta * penalty)
        adjusted[edge] = max(adjusted_weight, 0.0)
    return adjusted


def find_single_diverse_matching(
    graph: nx.Graph,
    base_weights: EdgeWeightMap,
    history: Sequence[Matching],
    *,
    max_size: int,
    iterations: int,
    eta: float,
    rng: Optional[random.Random] = None,
) -> Matching:
    if rng is None:
        rng = random.Random()

    if not history:
        return call_matching_oracle(
            graph,
            base_weights,
            max_size=max_size,
            reference_weights=base_weights,
        )

    dual_weights = [1.0 for _ in history]
    best_matching: Optional[Matching] = None
    best_min_distance = -math.inf

    for _ in range(max(1, iterations)):
        temporal_weights = compute_temporal_weights(base_weights, history, dual_weights, eta=eta)
        jittered_weights: EdgeWeightMap = {}
        for edge, weight in temporal_weights.items():
            jitter = 1.0 + 0.01 * rng.uniform(-1.0, 1.0)
            jittered_weights[edge] = max(weight * jitter, 0.0)

        candidate = call_matching_oracle(
            graph,
            jittered_weights,
            max_size=max_size,
            reference_weights=base_weights,
        )
        if not candidate:
            continue

        if candidate in history:
            dual_weights = [dw * (1.0 + eta) for dw in dual_weights]
            continue

        min_distance = min(
            weighted_collaboration_distance(base_weights, candidate, past) for past in history
        )

        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_matching = candidate

        for idx, past in enumerate(history):
            overlap_weight = sum(base_weights[edge] for edge in (candidate & past))
            dual_weights[idx] *= math.exp(eta * overlap_weight)

        total = sum(dual_weights)
        if total > 0:
            scale = len(dual_weights) / total
            dual_weights = [dw * scale for dw in dual_weights]

    if best_matching is None:
        return call_matching_oracle(
            graph,
            base_weights,
            max_size=max_size,
            reference_weights=base_weights,
        )

    return best_matching


def orchestrate_diverse_matchings(
    graph: nx.Graph,
    base_weights: EdgeWeightMap,
    k: int,
    r: int,
    *,
    delta: float,
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
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

    eta = min(0.5, delta / 2.0)
    base_scale = max(2.0, float(len(base_weights)))
    default_iterations = math.ceil((2.0 * math.log(base_scale)) / (delta ** 2))
    iterations = max_iterations or (k * default_iterations)

    for attempt in range(1, iterations + 1):
        if len(history) >= k:
            break

        print(
            f"[Progress] Attempt {attempt}/{iterations}: generating candidate matching...",
            flush=True,
        )

        candidate = find_single_diverse_matching(
            graph,
            base_weights,
            history,
            max_size=r,
            iterations=default_iterations,
            eta=eta,
            rng=rng,
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



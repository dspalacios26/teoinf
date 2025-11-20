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
MaximizationOracle = Callable[[nx.Graph, EdgeWeightMap], FrozenSet[Edge]]


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


def greedy_matroid_maximization(
    graph: nx.Graph,
    weights: EdgeWeightMap,
    matroid_oracle: MatroidOracle,
) -> FrozenSet[Edge]:
    """
    Find maximum weight independent set in a matroid using the Greedy Algorithm.
    This is optimal for matroids (Rado 1957).
    
    Args:
        graph: Input graph (used for edges)
        weights: Edge weights
        matroid_oracle: Oracle to check independence
        
    Returns:
        Maximum weight independent set (basis)
    """
    # Sort all edges by weight descending
    # Note: We consider all edges in the graph, not just those in weights, 
    # but usually weights covers relevant edges.
    all_edges = list(graph.edges())
    # Canonicalize
    canonical_edges = [canonical_edge(u, v) for u, v in all_edges]
    
    # Filter edges that have weights (or assume 0 if not present, but usually we want weighted ones)
    # The paper implies w: E -> R.
    sorted_edges = sorted(
        canonical_edges,
        key=lambda e: weights.get(e, 0.0),
        reverse=True
    )
    
    independent_set: Set[Edge] = set()
    
    for edge in sorted_edges:
        candidate = independent_set | {edge}
        if matroid_oracle(candidate):
            independent_set.add(edge)
            
    return frozenset(independent_set)





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
    oracle_fn: MaximizationOracle,
    *,
    iterations: int,
    eta: float,
    rng: Optional[random.Random] = None,
) -> Matching:
    if rng is None:
        rng = random.Random()

    if not history:
        return oracle_fn(graph, base_weights)

    dual_weights = [1.0 for _ in history]
    best_matching: Optional[Matching] = None
    best_min_distance = -math.inf

    for _ in range(max(1, iterations)):
        temporal_weights = compute_temporal_weights(base_weights, history, dual_weights, eta=eta)
        jittered_weights: EdgeWeightMap = {}
        for edge, weight in temporal_weights.items():
            jitter = 1.0 + 0.01 * rng.uniform(-1.0, 1.0)
            jittered_weights[edge] = max(weight * jitter, 0.0)

        candidate = oracle_fn(graph, jittered_weights)
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
        return oracle_fn(graph, base_weights)

    return best_matching


def orchestrate_diverse_matchings(
    graph: nx.Graph,
    base_weights: EdgeWeightMap,
    oracle_fn: MaximizationOracle,
    k: int,
    *,
    delta: float,
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Matching], float]:
    if k <= 0:
        raise ValueError("k must be positive")
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
            oracle_fn=oracle_fn,
            iterations=default_iterations,
            eta=eta,
            rng=rng,
        )

        if candidate in seen:
            print(f"[Progress] Attempt {attempt}/{iterations}: duplicate matching skipped", flush=True)
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


def export_matchings_to_dot(
    graph: nx.Graph,
    matchings: List[Matching],
    weights: EdgeWeightMap,
    output_path: Path,
    *,
    partition: Optional[Dict[Edge, int]] = None,
    matroid_type: Optional[str] = None,
) -> None:
    """
    Export matchings to DOT format for Graphviz visualization.
    
    Args:
        graph: The input graph
        matchings: List of matchings to visualize
        weights: Edge weights
        output_path: Path to output .dot file
        partition: Optional partition mapping for partition matroid
        matroid_type: Type of matroid constraint used
    """
    with output_path.open("w") as f:
        f.write("digraph DiverseMatchings {\n")
        f.write('  graph [rankdir=LR, splines=true];\n')
        f.write('  node [shape=circle, style=filled, fillcolor=lightgray];\n\n')
        
        # Write matroid info if available
        if matroid_type:
            f.write(f'  label="Diverse Matchings with {matroid_type} Constraint";\n')
            f.write('  labelloc=t;\n')
            f.write('  fontsize=20;\n\n')
        
        # Create a subgraph for each matching
        for idx, matching in enumerate(matchings, start=1):
            f.write(f'  subgraph cluster_{idx} {{\n')
            f.write(f'    label="Matching #{idx} (|M|={len(matching)})";\n')
            f.write('    style=rounded;\n')
            f.write('    color=blue;\n')
            
            # Get nodes in this matching
            nodes_in_matching = set()
            for u, v in matching:
                nodes_in_matching.add(u)
                nodes_in_matching.add(v)
            
            # Add nodes
            for node in sorted(nodes_in_matching):
                f.write(f'    n{idx}_{node} [label="{node}"];\n')
            
            # Add edges with weights and partition info
            for u, v in sorted(matching):
                edge = (u, v)
                weight = weights.get(edge, 0.0)
                
                # Color based on partition if available
                color = "black"
                style = "solid"
                if partition and edge in partition:
                    group = partition[edge]
                    color = "red" if group == 0 else "blue"
                    style = "bold" if group == 0 else "solid"
                
                label = f"{weight:.2f}"
                if partition and edge in partition:
                    label += f"\\nG{partition[edge]}"
                
                f.write(f'    n{idx}_{u} -> n{idx}_{v} [label="{label}", color={color}, style={style}, dir=none];\n')
            
            f.write('  }\n\n')
        
        # Add legend if partition matroid
        if partition and matroid_type == "Partition Matroid":
            f.write('  subgraph cluster_legend {\n')
            f.write('    label="Legend";\n')
            f.write('    style=dashed;\n')
            f.write('    legend_g0 [label="Group 0\\n(High)", shape=box, color=red, style=filled, fillcolor=pink];\n')
            f.write('    legend_g1 [label="Group 1\\n(Low)", shape=box, color=blue, style=filled, fillcolor=lightblue];\n')
            f.write('  }\n')
        
        f.write("}\n")
    
    print(f"📊 Visualization exported to: {output_path}")
    print(f"   To view: dot -Tpng {output_path} -o {output_path.with_suffix('.png')}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the diverse matching solver on a weighted dataset (CSV folder or edge list).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic diverse matching (Problem 3)
  python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3
  
  # With uniform matroid constraint (Problem 4)
  python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 --matroid uniform
  
  # With partition matroid (Problem 4)
  python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 \\
    --matroid partition --partition-threshold 15 \\
    --partition-capacities 3 10 --output matchings.dot
        """,
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
    
    # Matroid constraint options
    matroid_group = parser.add_argument_group("Matroid Constraints (Problem 4)")
    matroid_group.add_argument(
        "--matroid",
        choices=["none", "uniform", "partition", "graphic"],
        default="none",
        help="Type of matroid constraint to apply",
    )
    matroid_group.add_argument(
        "--partition-threshold",
        type=float,
        help="Weight threshold for partition matroid (edges >= threshold go to group 0)",
    )
    matroid_group.add_argument(
        "--partition-capacities",
        type=int,
        nargs=2,
        metavar=("CAP0", "CAP1"),
        help="Capacities for partition matroid groups (group 0, group 1)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        type=str,
        help="Path to output .dot file for visualization (default: no output)",
    )
    output_group.add_argument(
        "--render-png",
        action="store_true",
        help="Automatically render DOT file to PNG using Graphviz",
    )
    
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Entry point. Parses arguments, loads the graph, and runs the diverse matching algorithm.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # Load graph and weights
    graph, base_weights = load_weighted_graph(
        Path(args.dataset),
        edges_filename=args.edges_filename,
    )
    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Validate parameters
    if args.matchings < 1:
        print("Error: Number of matchings must be >= 1")
        return
    if args.max_size < 1:
        print("Error: Maximum matching size must be >= 1")
        return
    if args.delta <= 0 or args.delta >= 1:
        print("Error: Delta must be in range (0, 1)")
        return
    
    # Run appropriate algorithm based on matroid choice
    partition = None
    matroid_name = None
    oracle_fn: MaximizationOracle
    
    if args.matroid == "none":
        print(f"\\n🎯 Running Diverse Matching (Problem 3)")
        print(f"   Parameters: k={args.matchings}, r={args.max_size}, δ={args.delta}")
        
        # Define the Matching Oracle (Problem 3)
        def matching_oracle_wrapper(g: nx.Graph, w: EdgeWeightMap) -> FrozenSet[Edge]:
            return call_matching_oracle(g, w, max_size=args.max_size, reference_weights=base_weights)
            
        oracle_fn = matching_oracle_wrapper
        
    else:
        # Build matroid oracle based on type
        matroid_oracle: MatroidOracle
        
        if args.matroid == "uniform":
            print(f"\\n🎯 Running Diverse Matroid Bases (Problem 4)")
            print(f"   Matroid: Uniform U({args.max_size}, n)")
            matroid_oracle = uniform_matroid_oracle(args.max_size)
            matroid_name = "Uniform Matroid"
            
        elif args.matroid == "partition":
            if args.partition_threshold is None or args.partition_capacities is None:
                print("Error: --partition-threshold and --partition-capacities required for partition matroid")
                return
            
            # Build partition based on threshold
            partition = {}
            for edge in graph.edges():
                weight = base_weights.get(edge, 0.0)
                partition[edge] = 0 if weight >= args.partition_threshold else 1
            
            capacities_list = args.partition_capacities
            capacities = {0: capacities_list[0], 1: capacities_list[1]}
            print(f"\\n🎯 Running Diverse Matroid Bases (Problem 4)")
            print(f"   Matroid: Partition Matroid")
            print(f"   Threshold: {args.partition_threshold}")
            print(f"   Capacities: Group 0 (≥{args.partition_threshold}) → {capacities[0]}, Group 1 (<{args.partition_threshold}) → {capacities[1]}")
            
            # Count edges in each partition
            count_g0 = sum(1 for g in partition.values() if g == 0)
            count_g1 = sum(1 for g in partition.values() if g == 1)
            print(f"   Partition: Group 0 has {count_g0} edges, Group 1 has {count_g1} edges")
            
            matroid_oracle = partition_matroid_oracle(partition, capacities)
            matroid_name = "Partition Matroid"
            
        elif args.matroid == "graphic":
            nodes = set(graph.nodes())
            print(f"\\n🎯 Running Diverse Matroid Bases (Problem 4)")
            print(f"   Matroid: Graphic Matroid on {len(nodes)} nodes")
            matroid_oracle = graphic_matroid_oracle(nodes)
            matroid_name = "Graphic Matroid"
        
        else:
            print(f"Error: Unknown matroid type '{args.matroid}'")
            return
        
        # Define the Greedy Matroid Oracle (Problem 4)
        def matroid_oracle_wrapper(g: nx.Graph, w: EdgeWeightMap) -> FrozenSet[Edge]:
            return greedy_matroid_maximization(g, w, matroid_oracle)
            
        oracle_fn = matroid_oracle_wrapper
        
    print(f"   Parameters: k={args.matchings}, δ={args.delta}")
    
    # Run generic diverse solution algorithm
    matchings, min_distance = orchestrate_diverse_matchings(
        graph,
        base_weights,
        oracle_fn=oracle_fn,
        k=args.matchings,
        delta=args.delta,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )

    # Display results
    print(f"\n📋 Generated {len(matchings)} diverse matchings")
    for idx, matching in enumerate(matchings, start=1):
        print("-" * 54)
        total_weight = sum(base_weights.get(edge, 0.0) for edge in matching)
        print(f"Matching #{idx} (|M|={len(matching)}; total_weight={total_weight:.2f})")
        for edge in sorted(matching):
            weight = base_weights.get(edge, 0.0)
            print(f"  {edge[0]} -- {edge[1]}  (weight={weight:.2f})")
    
    # Export to DOT if requested
    if args.output:
        output_path = Path(args.output)
        export_matchings_to_dot(
            graph=graph,
            matchings=matchings,
            weights=base_weights,
            output_path=output_path,
            partition=partition,
            matroid_type=matroid_name,
        )
        
        # Render to PNG if requested
        if args.render_png:
            png_path = output_path.with_suffix('.png')
            import subprocess
            try:
                subprocess.run(
                    ["dot", "-Tpng", str(output_path), "-o", str(png_path)],
                    check=True,
                    capture_output=True,
                )
                print(f"🖼️  PNG rendered to: {png_path}")
            except FileNotFoundError:
                print("⚠️  Graphviz 'dot' command not found. Install Graphviz to auto-render PNG.")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Error rendering PNG: {e.stderr.decode()}")
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()



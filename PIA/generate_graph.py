#!/usr/bin/env python3
"""
Generate synthetic graphs with arbitrary weights for testing diverse matching algorithms.

Usage:
    python generate_graph.py --nodes 100 --edges 300 --output my_graph
    python generate_graph.py --type complete --nodes 50 --weight-range 1 20 --output complete_50
    python generate_graph.py --type grid --grid-size 10 10 --weight-dist uniform --output grid_10x10
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, cast
import math


def generate_erdos_renyi(n_nodes: int, n_edges: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """Generate random graph using Erdős-Rényi model."""
    rng = random.Random(seed)
    edges = set()
    
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        raise ValueError(f"Cannot create {n_edges} edges with {n_nodes} nodes (max: {max_edges})")
    
    while len(edges) < n_edges:
        u = rng.randint(1, n_nodes)
        v = rng.randint(1, n_nodes)
        if u != v:
            edge = (min(u, v), max(u, v))
            edges.add(edge)
    
    return sorted(edges)


def generate_complete_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """Generate complete graph K_n."""
    edges = []
    for u in range(1, n_nodes + 1):
        for v in range(u + 1, n_nodes + 1):
            edges.append((u, v))
    return edges


def generate_grid_graph(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Generate 2D grid graph."""
    edges = []
    
    def node_id(r: int, c: int) -> int:
        return r * cols + c + 1
    
    for r in range(rows):
        for c in range(cols):
            current = node_id(r, c)
            # Right edge
            if c < cols - 1:
                edges.append((current, node_id(r, c + 1)))
            # Down edge
            if r < rows - 1:
                edges.append((current, node_id(r + 1, c)))
    
    return edges


def generate_cycle_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """Generate cycle graph C_n."""
    edges = []
    for i in range(1, n_nodes + 1):
        next_node = (i % n_nodes) + 1
        edges.append((min(i, next_node), max(i, next_node)))
    return edges


def generate_bipartite_graph(n_left: int, n_right: int, n_edges: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """Generate random bipartite graph."""
    rng = random.Random(seed)
    edges = set()
    
    max_edges = n_left * n_right
    if n_edges > max_edges:
        raise ValueError(f"Cannot create {n_edges} edges in bipartite graph ({n_left}, {n_right}) (max: {max_edges})")
    
    while len(edges) < n_edges:
        u = rng.randint(1, n_left)
        v = rng.randint(n_left + 1, n_left + n_right)
        edges.add((u, v))
    
    return sorted(edges)


def generate_star_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """Generate star graph with center node 1."""
    edges = []
    for i in range(2, n_nodes + 1):
        edges.append((1, i))
    return edges


def generate_3d_matching_hypergraph(n1: int, n2: int, n3: int, n_edges: int, 
                                   seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """
    Generate 3D MATCHING hypergraph (Problem 5 from paper).
    Returns hyperedges (v1, v2, v3) where v1 ∈ V1, v2 ∈ V2, v3 ∈ V3.
    
    Args:
        n1, n2, n3: Sizes of the three partitions
        n_edges: Number of hyperedges to generate
        seed: Random seed
        
    Returns:
        List of 3-tuples (v1, v2, v3)
    """
    rng = random.Random(seed)
    hyperedges = set()
    
    max_edges = n1 * n2 * n3
    if n_edges > max_edges:
        raise ValueError(f"Cannot create {n_edges} hyperedges with partitions ({n1}, {n2}, {n3}) (max: {max_edges})")
    
    # Partition offsets: V1 = [1, n1], V2 = [n1+1, n1+n2], V3 = [n1+n2+1, n1+n2+n3]
    v1_start, v1_end = 1, n1
    v2_start, v2_end = n1 + 1, n1 + n2
    v3_start, v3_end = n1 + n2 + 1, n1 + n2 + n3
    
    while len(hyperedges) < n_edges:
        v1 = rng.randint(v1_start, v1_end)
        v2 = rng.randint(v2_start, v2_end)
        v3 = rng.randint(v3_start, v3_end)
        hyperedges.add((v1, v2, v3))
    
    return sorted(hyperedges)


def generate_complete_3d_hypergraph(n1: int, n2: int, n3: int) -> List[Tuple[int, int, int]]:
    """
    Generate complete 3D hypergraph (all possible 3-tuples).
    Useful for dense 3D MATCHING instances.
    """
    hyperedges = []
    v1_start, v1_end = 1, n1
    v2_start, v2_end = n1 + 1, n1 + n2
    v3_start, v3_end = n1 + n2 + 1, n1 + n2 + n3
    
    for v1 in range(v1_start, v1_end + 1):
        for v2 in range(v2_start, v2_end + 1):
            for v3 in range(v3_start, v3_end + 1):
                hyperedges.append((v1, v2, v3))
    
    return hyperedges


def generate_weights_uniform(n_edges: int, min_weight: float, max_weight: float, 
                            seed: Optional[int] = None) -> List[float]:
    """Generate uniformly distributed weights."""
    rng = random.Random(seed)
    return [rng.uniform(min_weight, max_weight) for _ in range(n_edges)]


def generate_weights_normal(n_edges: int, mean: float, std: float, 
                           min_weight: float = 1.0, seed: Optional[int] = None) -> List[float]:
    """Generate normally distributed weights (clipped to min_weight)."""
    rng = random.Random(seed)
    weights = []
    for _ in range(n_edges):
        w = rng.gauss(mean, std)
        weights.append(max(w, min_weight))
    return weights


def generate_weights_exponential(n_edges: int, lambda_param: float, 
                                 seed: Optional[int] = None) -> List[float]:
    """Generate exponentially distributed weights."""
    rng = random.Random(seed)
    return [rng.expovariate(lambda_param) for _ in range(n_edges)]


def generate_weights_powerlaw(n_edges: int, alpha: float, min_weight: float = 1.0,
                              seed: Optional[int] = None) -> List[float]:
    """Generate power-law distributed weights."""
    rng = random.Random(seed)
    weights = []
    for _ in range(n_edges):
        u = rng.random()
        w = min_weight * (1 - u) ** (-1 / alpha)
        weights.append(w)
    return weights


def generate_weights_bimodal(n_edges: int, low_mean: float, high_mean: float,
                             low_ratio: float = 0.7, seed: Optional[int] = None) -> List[float]:
    """Generate bimodal weight distribution (useful for partition matroids)."""
    rng = random.Random(seed)
    weights = []
    for _ in range(n_edges):
        if rng.random() < low_ratio:
            # Low weight group
            w = rng.uniform(low_mean * 0.5, low_mean * 1.5)
        else:
            # High weight group
            w = rng.uniform(high_mean * 0.5, high_mean * 1.5)
        weights.append(w)
    return weights


def save_graph(edges: List[Tuple[int, int]], weights: List[float], output_dir: Path):
    """Save graph to edges.csv file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_file = output_dir / "edges.csv"
    
    with edges_file.open("w") as f:
        f.write("# source, target, value\n")
        for (u, v), w in zip(edges, weights):
            f.write(f"{u}, {v}, {w:.2f}\n")
    
    print(f"✅ Generated graph: {len(edges)} edges")
    print(f"   Nodes: {max(max(u, v) for u, v in edges)}")
    print(f"   Weight range: [{min(weights):.2f}, {max(weights):.2f}]")
    print(f"   Weight mean: {sum(weights) / len(weights):.2f}")
    print(f"   Saved to: {edges_file}")


def save_3d_hypergraph(hyperedges: List[Tuple[int, int, int]], weights: List[float], output_dir: Path):
    """Save 3D hypergraph to edges.csv file (for 3D MATCHING problem)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_file = output_dir / "edges.csv"
    
    with edges_file.open("w") as f:
        f.write("# v1, v2, v3, weight\n")
        for (v1, v2, v3), w in zip(hyperedges, weights):
            f.write(f"{v1}, {v2}, {v3}, {w:.2f}\n")
    
    # Calculate partition sizes
    v1_nodes = set(v1 for v1, _, _ in hyperedges)
    v2_nodes = set(v2 for _, v2, _ in hyperedges)
    v3_nodes = set(v3 for _, _, v3 in hyperedges)
    
    print(f"✅ Generated 3D hypergraph: {len(hyperedges)} hyperedges")
    print(f"   Partition V1: {len(v1_nodes)} nodes (IDs: {min(v1_nodes)}-{max(v1_nodes)})")
    print(f"   Partition V2: {len(v2_nodes)} nodes (IDs: {min(v2_nodes)}-{max(v2_nodes)})")
    print(f"   Partition V3: {len(v3_nodes)} nodes (IDs: {min(v3_nodes)}-{max(v3_nodes)})")
    print(f"   Weight range: [{min(weights):.2f}, {max(weights):.2f}]")
    print(f"   Weight mean: {sum(weights) / len(weights):.2f}")
    print(f"   Saved to: {edges_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic graphs with arbitrary weights for diverse matching algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random graph with 100 nodes, 300 edges, uniform weights [1, 10]
  python generate_graph.py --nodes 100 --edges 300 --output random_100_300
  
  # Complete graph with 50 nodes, normal weight distribution
  python generate_graph.py --type complete --nodes 50 --weight-dist normal --weight-mean 15 --weight-std 5 --output complete_50
  
  # Grid graph 10x10 with bimodal weights (good for partition matroids)
  python generate_graph.py --type grid --grid-size 10 10 --weight-dist bimodal --weight-low 2 --weight-high 20 --output grid_10x10
  
  # Bipartite graph for matching problems
  python generate_graph.py --type bipartite --bipartite-size 30 40 --edges 100 --output bipartite_30_40
  
  # 3D MATCHING hypergraph (Problem 5 from paper)
  python generate_graph.py --type 3d-matching --partition-sizes 20 20 20 --edges 150 --output 3d_matching_20x20x20
  
  # Complete 3D hypergraph (all possible 3-tuples)
  python generate_graph.py --type 3d-complete --partition-sizes 10 10 10 --output 3d_complete_10x10x10
        """
    )
    
    # Graph structure
    parser.add_argument("--type", choices=["random", "complete", "grid", "cycle", "bipartite", "star", "3d-matching", "3d-complete"],
                       default="random", help="Type of graph structure")
    parser.add_argument("--nodes", type=int, default=100,
                       help="Number of nodes (for random, complete, cycle, star)")
    parser.add_argument("--edges", type=int, default=300,
                       help="Number of edges (for random, bipartite, 3d-matching)")
    parser.add_argument("--grid-size", type=int, nargs=2, metavar=("ROWS", "COLS"),
                       help="Grid dimensions (for grid graph)")
    parser.add_argument("--bipartite-size", type=int, nargs=2, metavar=("LEFT", "RIGHT"),
                       help="Bipartite partition sizes (for bipartite graph)")
    parser.add_argument("--partition-sizes", type=int, nargs=3, metavar=("V1", "V2", "V3"),
                       help="3D hypergraph partition sizes (for 3d-matching, 3d-complete)")
    
    # Weight distribution
    parser.add_argument("--weight-dist", choices=["uniform", "normal", "exponential", "powerlaw", "bimodal"],
                       default="uniform", help="Weight distribution")
    parser.add_argument("--weight-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                       default=[1.0, 10.0], help="Weight range for uniform distribution")
    parser.add_argument("--weight-mean", type=float, default=10.0,
                       help="Mean weight (for normal distribution)")
    parser.add_argument("--weight-std", type=float, default=5.0,
                       help="Standard deviation (for normal distribution)")
    parser.add_argument("--weight-lambda", type=float, default=0.1,
                       help="Lambda parameter (for exponential distribution)")
    parser.add_argument("--weight-alpha", type=float, default=2.0,
                       help="Alpha parameter (for power-law distribution)")
    parser.add_argument("--weight-low", type=float, default=2.0,
                       help="Low weight mean (for bimodal distribution)")
    parser.add_argument("--weight-high", type=float, default=20.0,
                       help="High weight mean (for bimodal distribution)")
    parser.add_argument("--weight-low-ratio", type=float, default=0.7,
                       help="Ratio of low-weight edges (for bimodal distribution)")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory name")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Generate graph structure
    is_3d_hypergraph = args.type in ["3d-matching", "3d-complete"]
    edges: Union[List[Tuple[int, int]], List[Tuple[int, int, int]]]
    
    if args.type == "random":
        edges = generate_erdos_renyi(args.nodes, args.edges, args.seed)
    elif args.type == "complete":
        edges = generate_complete_graph(args.nodes)
    elif args.type == "grid":
        if not args.grid_size:
            parser.error("--grid-size required for grid graph")
        rows, cols = args.grid_size
        edges = generate_grid_graph(rows, cols)
    elif args.type == "cycle":
        edges = generate_cycle_graph(args.nodes)
    elif args.type == "bipartite":
        if not args.bipartite_size:
            parser.error("--bipartite-size required for bipartite graph")
        n_left, n_right = args.bipartite_size
        edges = generate_bipartite_graph(n_left, n_right, args.edges, args.seed)
    elif args.type == "star":
        edges = generate_star_graph(args.nodes)
    elif args.type == "3d-matching":
        if not args.partition_sizes:
            parser.error("--partition-sizes required for 3d-matching")
        n1, n2, n3 = args.partition_sizes
        edges = generate_3d_matching_hypergraph(n1, n2, n3, args.edges, args.seed)
    elif args.type == "3d-complete":
        if not args.partition_sizes:
            parser.error("--partition-sizes required for 3d-complete")
        n1, n2, n3 = args.partition_sizes
        edges = generate_complete_3d_hypergraph(n1, n2, n3)
    else:
        parser.error(f"Unknown graph type: {args.type}")
    
    # Generate weights
    n_edges = len(edges)
    if args.weight_dist == "uniform":
        weights = generate_weights_uniform(n_edges, args.weight_range[0], args.weight_range[1], args.seed)
    elif args.weight_dist == "normal":
        weights = generate_weights_normal(n_edges, args.weight_mean, args.weight_std, 
                                         args.weight_range[0], args.seed)
    elif args.weight_dist == "exponential":
        weights = generate_weights_exponential(n_edges, args.weight_lambda, args.seed)
    elif args.weight_dist == "powerlaw":
        weights = generate_weights_powerlaw(n_edges, args.weight_alpha, args.weight_range[0], args.seed)
    elif args.weight_dist == "bimodal":
        weights = generate_weights_bimodal(n_edges, args.weight_low, args.weight_high,
                                          args.weight_low_ratio, args.seed)
    else:
        parser.error(f"Unknown weight distribution: {args.weight_dist}")
    
    # Save graph
    output_dir = Path(args.output)
    if is_3d_hypergraph:
        save_3d_hypergraph(cast(List[Tuple[int, int, int]], edges), weights, output_dir)
    else:
        save_graph(cast(List[Tuple[int, int]], edges), weights, output_dir)
    
    # Print statistics
    print(f"\n📊 Graph statistics:")
    print(f"   Type: {args.type}")
    print(f"   Weight distribution: {args.weight_dist}")
    if args.seed is not None:
        print(f"   Seed: {args.seed}")


if __name__ == "__main__":
    main()

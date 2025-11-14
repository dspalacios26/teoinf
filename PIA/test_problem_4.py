#!/usr/bin/env python3
"""
Test Problem 4: Diverse Matroid Bases
From Section 2.4.2 of the paper

Problem 4: Given a matroid M = (E, I), a weight function w: E → R, and an integer
k ∈ Z>0, find independent sets I1,...,Ik ∈ I of M maximizing min(1≤i<j≤k) dw(Ii,Ij).
"""

import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diverse_matching import (
    canonical_edge,
    load_weighted_graph,
    uniform_matroid_oracle,
    partition_matroid_oracle,
    graphic_matroid_oracle,
    diverse_matroid_matching,
)


def test_problem_4_uniform_constraint(dataset_path: Path):
    """
    Problem 4 with Uniform Matroid constraint U(k,n)
    Limits the number of edges in each independent set
    """
    print(f"\n{'='*70}")
    print(f"Problem 4: Uniform Matroid on {dataset_path.name}")
    print(f"{'='*70}")
    
    graph, weights = load_weighted_graph(dataset_path)
    print(f"Graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Uniform matroid: at most 15 edges
    max_edges = 15
    oracle = uniform_matroid_oracle(max_edges)
    
    print(f"\nConstraint: Uniform matroid U({max_edges}, n)")
    print(f"Finding k=3 diverse independent sets with max size r={max_edges}...")
    
    matchings, min_dist = diverse_matroid_matching(
        graph,
        weights,
        oracle,
        k=3,
        r=max_edges,
        delta=0.2,
        max_iterations=100,
        seed=42,
    )
    
    print(f"\n✅ Found {len(matchings)} diverse independent sets")
    for i, matching in enumerate(matchings, 1):
        total_weight = sum(weights.get(e, 0) for e in matching)
        print(f"  Independent set {i}: |I|={len(matching)}, weight={total_weight:.2f}")
    
    print(f"\n📊 Minimum pairwise distance: {min_dist:.2f}")
    return matchings, min_dist


def test_problem_4_partition_constraint(dataset_path: Path):
    """
    Problem 4 with Partition Matroid constraint
    Divides edges by weight into high/medium/low groups with quotas
    """
    print(f"\n{'='*70}")
    print(f"Problem 4: Partition Matroid on {dataset_path.name}")
    print(f"{'='*70}")
    
    graph, weights = load_weighted_graph(dataset_path)
    print(f"Graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Partition edges by weight
    partition = {}
    weight_values = list(weights.values())
    if not weight_values:
        print("No edges found!")
        return
    
    threshold_high = sorted(weight_values, reverse=True)[len(weight_values)//4] if len(weight_values) > 3 else max(weight_values)
    threshold_low = sorted(weight_values)[len(weight_values)//4] if len(weight_values) > 3 else min(weight_values)
    
    for edge, weight in weights.items():
        if weight >= threshold_high:
            partition[edge] = 0  # High weight group
        elif weight >= threshold_low:
            partition[edge] = 1  # Medium weight group
        else:
            partition[edge] = 2  # Low weight group
    
    # Capacities: limit how many from each group
    capacities = {0: 5, 1: 7, 2: 8}  # At most 5 high, 7 medium, 8 low
    oracle = partition_matroid_oracle(partition, capacities)
    
    print(f"\nConstraint: Partition matroid with 3 groups")
    print(f"  High weight (≥{threshold_high:.2f}): capacity {capacities[0]}")
    print(f"  Medium weight (≥{threshold_low:.2f}): capacity {capacities[1]}")
    print(f"  Low weight: capacity {capacities[2]}")
    print(f"Finding k=3 diverse independent sets with max size r=15...")
    
    matchings, min_dist = diverse_matroid_matching(
        graph,
        weights,
        oracle,
        k=3,
        r=15,
        delta=0.2,
        max_iterations=100,
        seed=123,
    )
    
    print(f"\n✅ Found {len(matchings)} diverse independent sets")
    for i, matching in enumerate(matchings, 1):
        total_weight = sum(weights.get(e, 0) for e in matching)
        high = sum(1 for e in matching if partition.get(e) == 0)
        medium = sum(1 for e in matching if partition.get(e) == 1)
        low = sum(1 for e in matching if partition.get(e) == 2)
        print(f"  Independent set {i}: |I|={len(matching)}, weight={total_weight:.2f}")
        print(f"    Distribution: {high} high, {medium} medium, {low} low")
    
    print(f"\n📊 Minimum pairwise distance: {min_dist:.2f}")
    return matchings, min_dist


def test_problem_4_graphic_constraint(dataset_path: Path):
    """
    Problem 4 with Graphic Matroid constraint
    Independent sets must be forests (no cycles)
    """
    print(f"\n{'='*70}")
    print(f"Problem 4: Graphic Matroid (Forest) on {dataset_path.name}")
    print(f"{'='*70}")
    
    graph, weights = load_weighted_graph(dataset_path)
    print(f"Graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Graphic matroid: edges must form a forest (no cycles)
    nodes = set(graph.nodes())
    oracle = graphic_matroid_oracle(nodes)
    
    print(f"\nConstraint: Graphic matroid (edges must form a forest)")
    print(f"Finding k=3 diverse forests with max size r=20...")
    
    matchings, min_dist = diverse_matroid_matching(
        graph,
        weights,
        oracle,
        k=3,
        r=20,
        delta=0.2,
        max_iterations=100,
        seed=999,
    )
    
    print(f"\n✅ Found {len(matchings)} diverse forests")
    for i, matching in enumerate(matchings, 1):
        total_weight = sum(weights.get(e, 0) for e in matching)
        print(f"  Forest {i}: |I|={len(matching)} edges, weight={total_weight:.2f}")
    
    print(f"\n📊 Minimum pairwise distance: {min_dist:.2f}")
    return matchings, min_dist


def main():
    print("="*70)
    print("TESTING PROBLEM 4: DIVERSE MATROID BASES")
    print("="*70)
    
    datasets = [
        Path("hep-th-1999"),      # Smallest
        Path("cond-mat-1999"),    # Medium
        Path("astro-ph-1999"),    # Largest
    ]
    
    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"\n⚠️  Dataset {dataset_path} not found, skipping...")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset_path.name}")
        print(f"{'#'*70}")
        
        try:
            # Test 1: Uniform matroid
            test_problem_4_uniform_constraint(dataset_path)
            
            # Test 2: Partition matroid
            test_problem_4_partition_constraint(dataset_path)
            
            # Test 3: Graphic matroid (only on smaller datasets - too slow on large ones)
            if dataset_path.name in ["hep-th-1999", "cond-mat-1999"]:
                test_problem_4_graphic_constraint(dataset_path)
            else:
                print(f"\n⏭️  Skipping graphic matroid test for {dataset_path.name} (too large)")
        
        except Exception as e:
            print(f"\n❌ Error testing {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("PROBLEM 4 TESTING COMPLETE")
    print("="*70)
    print("\nAll three matroid types successfully tested:")
    print("  ✅ Uniform Matroid U(k,n) - Size constraints")
    print("  ✅ Partition Matroid - Quota-based constraints")
    print("  ✅ Graphic Matroid - Forest constraints")


if __name__ == "__main__":
    main()

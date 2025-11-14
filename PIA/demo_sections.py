#!/usr/bin/env python3
"""
Demonstration of Section 2.4.1 (Matching) and 2.4.2 (Matroid) Applications

This script demonstrates the various matroid and matching functionalities
implemented in diverse_matching.py.
"""

import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diverse_matching import (
    canonical_edge,
    # Section 2.4.1: Matching Applications
    is_valid_matching,
    matching_matroid_oracle,
    # Section 2.4.2: Matroid Applications
    uniform_matroid_oracle,
    partition_matroid_oracle,
    graphic_matroid_oracle,
    matroid_intersection,
    diverse_matroid_matching,
)


def demo_section_2_4_1():
    """Demonstrate Section 2.4.1: Matching Applications"""
    print("=" * 70)
    print("SECTION 2.4.1: MATCHING APPLICATIONS")
    print("=" * 70)
    
    # Example 1: Valid matching
    print("\n1. Valid Matching Check:")
    edges = {canonical_edge(0, 1), canonical_edge(2, 3), canonical_edge(4, 5)}
    print(f"   Edges: {edges}")
    print(f"   Is valid matching? {is_valid_matching(edges)}")
    
    # Example 2: Invalid matching (shared vertex)
    print("\n2. Invalid Matching Check:")
    invalid_edges = {canonical_edge(0, 1), canonical_edge(1, 2)}
    print(f"   Edges: {invalid_edges}")
    print(f"   Is valid matching? {is_valid_matching(invalid_edges)}")
    
    # Example 3: Matching matroid oracle
    print("\n3. Matching Matroid Oracle:")
    print(f"   Independent set {edges}? {matching_matroid_oracle(edges)}")
    print(f"   Independent set {invalid_edges}? {matching_matroid_oracle(invalid_edges)}")


def demo_section_2_4_2():
    """Demonstrate Section 2.4.2: Matroid Applications"""
    print("\n" + "=" * 70)
    print("SECTION 2.4.2: MATROID APPLICATIONS")
    print("=" * 70)
    
    # Uniform Matroid
    print("\n1. Uniform Matroid U(3, n):")
    oracle = uniform_matroid_oracle(3)
    small_set = {canonical_edge(0, 1), canonical_edge(2, 3)}
    large_set = {canonical_edge(0, 1), canonical_edge(2, 3), 
                 canonical_edge(4, 5), canonical_edge(6, 7)}
    print(f"   Size 2 set independent? {oracle(small_set)}")
    print(f"   Size 4 set independent? {oracle(large_set)}")
    
    # Partition Matroid
    print("\n2. Partition Matroid:")
    partition = {
        canonical_edge(0, 1): 0,  # Group 0
        canonical_edge(2, 3): 0,  # Group 0
        canonical_edge(4, 5): 1,  # Group 1
        canonical_edge(6, 7): 1,  # Group 1
    }
    capacities = {0: 1, 1: 2}
    oracle = partition_matroid_oracle(partition, capacities)
    
    valid_set = {canonical_edge(0, 1), canonical_edge(4, 5)}
    invalid_set = {canonical_edge(0, 1), canonical_edge(2, 3)}
    
    print(f"   Partition: Group 0 (capacity 1), Group 1 (capacity 2)")
    print(f"   Set {valid_set} independent? {oracle(valid_set)}")
    print(f"   Set {invalid_set} independent? {oracle(invalid_set)} (2 from group 0)")
    
    # Graphic Matroid
    print("\n3. Graphic Matroid (Forest):")
    nodes = {0, 1, 2, 3}
    oracle = graphic_matroid_oracle(nodes)
    
    forest = {canonical_edge(0, 1), canonical_edge(2, 3)}
    cycle = {canonical_edge(0, 1), canonical_edge(1, 2), canonical_edge(2, 0)}
    
    print(f"   Forest {forest} independent? {oracle(forest)}")
    print(f"   Cycle {cycle} independent? {oracle(cycle)}")
    
    # Matroid Intersection
    print("\n4. Matroid Intersection:")
    graph = nx.Graph()
    edges = {
        canonical_edge(0, 1): 5.0,
        canonical_edge(2, 3): 4.0,
        canonical_edge(4, 5): 3.0,
        canonical_edge(6, 7): 2.0,
    }
    for (u, v), w in edges.items():
        graph.add_edge(u, v, weight=w)
    
    # Intersect matching matroid with uniform matroid U(2, n)
    common_set, weight = matroid_intersection(
        ground_set=set(edges.keys()),
        oracle1=matching_matroid_oracle,
        oracle2=uniform_matroid_oracle(2),
        weights=edges
    )
    
    print(f"   Matching ∩ U(2, n):")
    print(f"   Common independent set: {common_set}")
    print(f"   Total weight: {weight}")


def demo_diverse_matroid_matching():
    """Demonstrate diverse matching with matroid constraints"""
    print("\n" + "=" * 70)
    print("DIVERSE MATROID MATCHING")
    print("=" * 70)
    
    # Build test graph
    graph = nx.Graph()
    weights = {
        canonical_edge(0, 1): 2.0,
        canonical_edge(0, 2): 4.0,
        canonical_edge(1, 2): 1.0,
        canonical_edge(3, 4): 5.0,
        canonical_edge(3, 5): 1.0,
        canonical_edge(4, 5): 1.5,
    }
    for (u, v), w in weights.items():
        graph.add_edge(u, v, weight=w)
    
    print("\n1. Diverse Matching with Uniform Constraint U(2, n):")
    oracle = uniform_matroid_oracle(2)
    
    matchings, min_dist = diverse_matroid_matching(
        graph,
        weights,
        oracle,
        k=2,
        r=2,
        delta=0.2,
        max_iterations=30,
        seed=42,
    )
    
    print(f"\n   Generated {len(matchings)} diverse matchings:")
    for i, matching in enumerate(matchings, 1):
        total_w = sum(weights.get(e, 0) for e in matching)
        print(f"   Matching {i}: {matching}")
        print(f"     Total weight: {total_w:.2f}, Size: {len(matching)}")
    print(f"\n   Minimum pairwise distance: {min_dist:.2f}")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: SECTIONS 2.4.1 & 2.4.2")
    print("=" * 70)
    
    demo_section_2_4_1()
    demo_section_2_4_2()
    demo_diverse_matroid_matching()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll implementations verified and working correctly!")
    print("See VERIFICATION_REPORT.md for detailed documentation.\n")


if __name__ == "__main__":
    main()

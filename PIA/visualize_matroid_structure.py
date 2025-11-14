#!/usr/bin/env python3
"""
Proper matroid structure visualization
Shows the matroid ground set, partitions, and independent sets
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.diverse_matching import (
    load_weighted_graph,
    uniform_matroid_oracle,
    partition_matroid_oracle,
    graphic_matroid_oracle,
    matching_matroid_oracle,
    diverse_matroid_matching,
    Edge,
    Matching,
    EdgeWeightMap,
)


def visualize_matroid_structure(
    graph: nx.Graph,
    weights: EdgeWeightMap,
    partition: Dict[Edge, int],
    capacities: Dict[int, int],
    matchings: List[Matching],
    title: str,
    output_file: str,
):
    """
    Visualize the matroid partition structure and how matchings respect it.
    """
    # Create figure with multiple views
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Ground set visualization - show partition
    ax1 = plt.subplot(2, 3, 1)
    visualize_ground_set(graph, weights, partition, ax1)
    
    # 2-4. Show each matching with matroid coloring
    for i, matching in enumerate(matchings[:3]):
        ax = plt.subplot(2, 3, i+2)
        visualize_matching_with_matroid(
            graph, matching, weights, partition,
            f"Independent Set #{i+1}", ax
        )
    
    # 5. Matroid constraint visualization (bar chart)
    ax5 = plt.subplot(2, 3, 5)
    visualize_matroid_constraints(matchings, partition, capacities, ax5)
    
    # 6. Independence property verification
    ax6 = plt.subplot(2, 3, 6)
    visualize_independence_check(matchings, partition, capacities, ax6)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', label='High-weight partition (Group 0)'),
        mpatches.Patch(color='blue', label='Low-weight partition (Group 1)'),
        mpatches.Patch(color='lightgray', label='Not in matching'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=3, fontsize=10, frameon=True)
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved matroid structure visualization to: {output_file}")


def visualize_ground_set(graph, weights, partition, ax):
    """Show the ground set with partition coloring"""
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Separate edges by partition
    group0_edges = [e for e in graph.edges() if partition.get(
        (min(e), max(e)), 1) == 0]
    group1_edges = [e for e in graph.edges() if partition.get(
        (min(e), max(e)), 1) == 1]
    
    # Draw edges by group
    nx.draw_networkx_edges(graph, pos, edgelist=group0_edges,
                          edge_color='red', width=2, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=group1_edges,
                          edge_color='blue', width=1, alpha=0.3, ax=ax)
    
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue',
                          node_size=100, alpha=0.7, ax=ax)
    
    ax.set_title(f"Ground Set E\nRed: {len(group0_edges)} edges (Group 0)\n"
                f"Blue: {len(group1_edges)} edges (Group 1)",
                fontsize=10, fontweight='bold')
    ax.axis('off')


def visualize_matching_with_matroid(graph, matching, weights, partition, title, ax):
    """Visualize a matching with matroid partition colors"""
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Background edges
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray',
                          width=0.3, alpha=0.2, ax=ax)
    
    # Matching edges colored by partition
    for edge in matching:
        group = partition.get(edge, 1)
        color = 'red' if group == 0 else 'blue'
        nx.draw_networkx_edges(graph, pos, edgelist=[edge],
                              edge_color=color, width=3, alpha=0.8, ax=ax)
    
    # Nodes
    matched_nodes = set()
    for u, v in matching:
        matched_nodes.add(u)
        matched_nodes.add(v)
    
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue',
                          node_size=150, alpha=0.8, ax=ax)
    
    # Count by group
    high = sum(1 for e in matching if partition.get(e, 1) == 0)
    low = sum(1 for e in matching if partition.get(e, 1) == 1)
    total_w = sum(weights.get(e, 0) for e in matching)
    
    ax.set_title(f"{title}\n{high} red + {low} blue = {len(matching)} edges\n"
                f"weight={total_w:.1f}",
                fontsize=9, fontweight='bold')
    ax.axis('off')


def visualize_matroid_constraints(matchings, partition, capacities, ax):
    """Bar chart showing constraint satisfaction"""
    n_matchings = len(matchings)
    x = np.arange(n_matchings)
    width = 0.35
    
    high_counts = []
    low_counts = []
    
    for matching in matchings:
        high = sum(1 for e in matching if partition.get(e, 1) == 0)
        low = sum(1 for e in matching if partition.get(e, 1) == 1)
        high_counts.append(high)
        low_counts.append(low)
    
    # Check if we have two groups or just one (uniform matroid)
    has_two_groups = 1 in capacities
    
    if has_two_groups:
        ax.bar(x - width/2, high_counts, width, label='Group 0',
               color='red', alpha=0.7)
        ax.bar(x + width/2, low_counts, width, label='Group 1',
               color='blue', alpha=0.7)
        
        # Add capacity lines
        ax.axhline(y=capacities[0], color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label=f'Capacity 0: {capacities[0]}')
        ax.axhline(y=capacities[1], color='blue', linestyle='--', linewidth=2,
                  alpha=0.5, label=f'Capacity 1: {capacities[1]}')
        title = 'Partition Matroid Constraints\n(bars must stay below lines)'
    else:
        # Uniform matroid - just show total size
        total_counts = [len(m) for m in matchings]
        ax.bar(x, total_counts, width*2, label='Total edges',
               color='red', alpha=0.7)
        ax.axhline(y=capacities[0], color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label=f'Capacity: {capacities[0]}')
        title = 'Uniform Matroid Constraint\n(size must stay at or below line)'
    
    ax.set_xlabel('Independent Set', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'I{i+1}' for i in range(n_matchings)])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def visualize_independence_check(matchings, partition, capacities, ax):
    """Show independence verification for each set"""
    oracle = partition_matroid_oracle(partition, capacities)
    
    has_two_groups = 1 in capacities
    
    data = []
    for i, matching in enumerate(matchings):
        high = sum(1 for e in matching if partition.get(e, 1) == 0)
        low = sum(1 for e in matching if partition.get(e, 1) == 1)
        is_independent = oracle(set(matching))
        
        data.append({
            'set': f'I{i+1}',
            'size': len(matching),
            'high': high,
            'low': low,
            'independent': is_independent
        })
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    if has_two_groups:
        headers = ['Set', 'Size', 'Group 0', 'Group 1', 'Independent?']
        table_data.append(headers)
        
        for d in data:
            row = [
                d['set'],
                str(d['size']),
                f"{d['high']}/{capacities[0]}",
                f"{d['low']}/{capacities[1]}",
                '✅ Yes' if d['independent'] else '❌ No'
            ]
            table_data.append(row)
    else:
        # Uniform matroid - simpler table
        headers = ['Set', 'Size', f'Capacity', 'Independent?']
        table_data.append(headers)
        
        for d in data:
            row = [
                d['set'],
                f"{d['size']}/{capacities[0]}",
                f"≤ {capacities[0]}",
                '✅ Yes' if d['independent'] else '❌ No'
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if '✅' in table_data[i][j]:
                table[(i, j)].set_facecolor('#C8E6C9')
            elif '❌' in table_data[i][j]:
                table[(i, j)].set_facecolor('#FFCDD2')
    
    ax.set_title('Independence Verification\n(Matroid Oracle Check)',
                fontsize=10, fontweight='bold')


def test_partition_matroid_structure():
    """Create comprehensive partition matroid visualization"""
    print("="*70)
    print("CREATING PARTITION MATROID STRUCTURE VISUALIZATION")
    print("="*70)
    
    graph, weights = load_weighted_graph(Path('../lesmis'))
    
    # Define partition
    threshold = 15.0
    partition = {}
    for edge, weight in weights.items():
        partition[edge] = 0 if weight >= threshold else 1
    
    capacities = {0: 8, 1: 10}
    
    print(f"\nPartition:")
    high = sum(1 for g in partition.values() if g == 0)
    low = sum(1 for g in partition.values() if g == 1)
    print(f"  Group 0 (≥{threshold}): {high} edges, capacity {capacities[0]}")
    print(f"  Group 1 (<{threshold}): {low} edges, capacity {capacities[1]}")
    
    # Generate diverse matchings
    oracle = partition_matroid_oracle(partition, capacities)
    matchings, min_dist = diverse_matroid_matching(
        graph, weights, oracle,
        k=3, r=12, delta=0.3,
        max_iterations=150, seed=999
    )
    
    print(f"\n✅ Generated {len(matchings)} diverse independent sets")
    print(f"📊 Minimum pairwise distance: {min_dist:.2f}\n")
    
    # Visualize
    visualize_matroid_structure(
        graph, weights, partition, capacities, matchings,
        "Partition Matroid Structure - Les Misérables Dataset\n"
        f"M = (E, I) where I satisfies: |I ∩ Group_i| ≤ capacity_i",
        "matroid_structure_partition.png"
    )


def test_uniform_matroid_structure():
    """Create uniform matroid visualization"""
    print("\n" + "="*70)
    print("CREATING UNIFORM MATROID STRUCTURE VISUALIZATION")
    print("="*70)
    
    graph, weights = load_weighted_graph(Path('../lesmis'))
    
    k = 10  # Uniform matroid U(10, n)
    
    # Create trivial partition (all in one group) for uniform
    partition = {edge: 0 for edge in weights.keys()}
    capacities = {0: k}
    
    print(f"\nUniform Matroid U({k}, n)")
    print(f"  Independent if |I| ≤ {k}")
    
    oracle = uniform_matroid_oracle(k)
    matchings, min_dist = diverse_matroid_matching(
        graph, weights, oracle,
        k=3, r=k, delta=0.3,
        max_iterations=100, seed=42
    )
    
    print(f"\n✅ Generated {len(matchings)} diverse independent sets")
    print(f"📊 Minimum pairwise distance: {min_dist:.2f}\n")
    
    # For uniform matroid, use single color
    partition_viz = {edge: 0 for edge in weights.keys()}
    capacities_viz = {0: k}
    
    visualize_matroid_structure(
        graph, weights, partition_viz, capacities_viz, matchings,
        f"Uniform Matroid U({k}, n) Structure - Les Misérables Dataset\n"
        f"M = (E, I) where I is independent iff |I| ≤ {k}",
        "matroid_structure_uniform.png"
    )


def main():
    """Generate all matroid structure visualizations"""
    print("="*70)
    print("MATROID STRUCTURE VISUALIZATIONS")
    print("="*70)
    print("\nGenerating comprehensive matroid structure diagrams...\n")
    
    test_partition_matroid_structure()
    test_uniform_matroid_structure()
    
    print("\n" + "="*70)
    print("ALL MATROID STRUCTURE VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 matroid_structure_partition.png - Partition matroid with constraints")
    print("  📊 matroid_structure_uniform.png - Uniform matroid with size limit")
    print("\nThese show:")
    print("  • Ground set E with partition structure")
    print("  • Independent sets (matchings) respecting matroid constraints")
    print("  • Bar charts verifying capacity constraints")
    print("  • Tables showing independence verification")


if __name__ == "__main__":
    main()

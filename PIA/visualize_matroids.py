#!/usr/bin/env python3
"""
Graphical visualization of matroid-constrained diverse matchings
Displays matchings with matroid constraints highlighted
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import List, Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.diverse_matching import (
    load_weighted_graph,
    uniform_matroid_oracle,
    partition_matroid_oracle,
    diverse_matroid_matching,
    Edge,
    Matching,
    EdgeWeightMap,
)


def visualize_matching(
    graph: nx.Graph,
    matching: Matching,
    weights: EdgeWeightMap,
    title: str,
    ax: plt.Axes,
    matroid_info: Dict = None,
):
    """
    Visualize a single matching on a graph.
    
    Args:
        graph: NetworkX graph
        matching: Set of edges in the matching
        weights: Edge weights
        title: Plot title
        ax: Matplotlib axes
        matroid_info: Optional dict with 'partition', 'colors' for highlighting
    """
    # Create layout
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Draw all edges (light gray)
    nx.draw_networkx_edges(
        graph, pos, 
        edgelist=graph.edges(),
        edge_color='lightgray',
        width=0.5,
        alpha=0.3,
        ax=ax
    )
    
    # Draw matched edges with colors based on matroid groups
    if matroid_info and 'partition' in matroid_info:
        partition = matroid_info['partition']
        color_map = matroid_info.get('colors', {0: 'red', 1: 'blue'})
        
        # Group matching edges by partition
        for group, color in color_map.items():
            group_edges = [e for e in matching if partition.get(e) == group]
            if group_edges:
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=group_edges,
                    edge_color=color,
                    width=3.0,
                    alpha=0.8,
                    ax=ax
                )
    else:
        # Draw all matched edges in one color
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=list(matching),
            edge_color='red',
            width=3.0,
            alpha=0.8,
            ax=ax
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color='lightblue',
        node_size=200,
        alpha=0.9,
        ax=ax
    )
    
    # Draw labels (only for nodes in matching)
    matched_nodes = set()
    for u, v in matching:
        matched_nodes.add(u)
        matched_nodes.add(v)
    
    labels = {node: str(node) for node in matched_nodes}
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Add title with statistics
    total_weight = sum(weights.get(e, 0) for e in matching)
    ax.set_title(f"{title}\n|M|={len(matching)}, weight={total_weight:.1f}", 
                 fontsize=10, fontweight='bold')
    ax.axis('off')


def visualize_diverse_matchings(
    graph: nx.Graph,
    matchings: List[Matching],
    weights: EdgeWeightMap,
    title: str,
    matroid_info: Dict = None,
    output_file: str = None,
):
    """
    Visualize multiple diverse matchings in a grid.
    
    Args:
        graph: NetworkX graph
        matchings: List of matchings to visualize
        weights: Edge weights
        title: Overall title
        matroid_info: Optional matroid partition information
        output_file: If provided, save to file instead of showing
    """
    n_matchings = len(matchings)
    
    # Create figure with subplots
    if n_matchings <= 3:
        fig, axes = plt.subplots(1, n_matchings, figsize=(6*n_matchings, 6))
    else:
        n_cols = 3
        n_rows = (n_matchings + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    
    # Ensure axes is iterable
    if n_matchings == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_matchings > 3 else axes
    
    # Plot each matching
    for i, (matching, ax) in enumerate(zip(matchings, axes)):
        visualize_matching(
            graph, matching, weights,
            f"Matching #{i+1}",
            ax,
            matroid_info
        )
    
    # Hide unused subplots
    for i in range(n_matchings, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend if matroid info provided
    if matroid_info and 'partition' in matroid_info:
        color_map = matroid_info.get('colors', {0: 'red', 1: 'blue'})
        labels = matroid_info.get('labels', {0: 'Group 0', 1: 'Group 1'})
        
        legend_elements = [
            mpatches.Patch(color=color, label=labels.get(group, f'Group {group}'))
            for group, color in color_map.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=10, frameon=True)
        plt.subplots_adjust(bottom=0.08)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved visualization to: {output_file}")
    else:
        plt.show()


def test_uniform_matroid_visualization():
    """Test and visualize uniform matroid constraints"""
    print("="*70)
    print("VISUALIZING UNIFORM MATROID")
    print("="*70)
    
    graph, weights = load_weighted_graph(Path('../lesmis'))
    print(f"Dataset: {len(graph.nodes())} nodes, {len(graph.edges())} edges\n")
    
    # Uniform matroid: max 10 edges
    oracle = uniform_matroid_oracle(10)
    
    print("Constraint: Uniform matroid U(10, n) - max 10 edges per matching")
    print("Finding k=3 diverse matchings...\n")
    
    matchings, min_dist = diverse_matroid_matching(
        graph, weights, oracle,
        k=3, r=10, delta=0.3,
        max_iterations=100, seed=42
    )
    
    print(f"✅ Found {len(matchings)} matchings, min distance: {min_dist:.2f}\n")
    
    # Visualize
    visualize_diverse_matchings(
        graph, matchings, weights,
        "Problem 4: Uniform Matroid U(10,n) - Les Misérables",
        output_file='uniform_matroid_lesmis.png'
    )


def test_partition_matroid_visualization():
    """Test and visualize partition matroid constraints"""
    print("\n" + "="*70)
    print("VISUALIZING PARTITION MATROID")
    print("="*70)
    
    graph, weights = load_weighted_graph(Path('../lesmis'))
    print(f"Dataset: {len(graph.nodes())} nodes, {len(graph.edges())} edges\n")
    
    # Partition edges by weight
    threshold = 15.0
    partition = {}
    for edge, weight in weights.items():
        partition[edge] = 0 if weight >= threshold else 1
    
    high_count = sum(1 for g in partition.values() if g == 0)
    low_count = sum(1 for g in partition.values() if g == 1)
    
    capacities = {0: 8, 1: 10}
    
    print(f"Constraint: Partition matroid")
    print(f"  High-weight (≥{threshold}): {high_count} edges, capacity {capacities[0]}")
    print(f"  Low-weight (<{threshold}): {low_count} edges, capacity {capacities[1]}")
    print("Finding k=3 diverse matchings...\n")
    
    oracle = partition_matroid_oracle(partition, capacities)
    matchings, min_dist = diverse_matroid_matching(
        graph, weights, oracle,
        k=3, r=12, delta=0.3,
        max_iterations=150, seed=999
    )
    
    print(f"✅ Found {len(matchings)} matchings, min distance: {min_dist:.2f}\n")
    
    # Print distribution
    for i, matching in enumerate(matchings, 1):
        high = sum(1 for e in matching if partition.get(e) == 0)
        low = sum(1 for e in matching if partition.get(e) == 1)
        total_w = sum(weights.get(e, 0) for e in matching)
        print(f"  Matching {i}: {high} high + {low} low, weight={total_w:.1f}")
    
    # Visualize with color coding
    matroid_info = {
        'partition': partition,
        'colors': {0: 'red', 1: 'blue'},
        'labels': {
            0: f'High-weight (≥{threshold})',
            1: f'Low-weight (<{threshold})'
        }
    }
    
    visualize_diverse_matchings(
        graph, matchings, weights,
        f"Problem 4: Partition Matroid - Les Misérables\n"
        f"Red = High weight (≥{threshold}), Blue = Low weight",
        matroid_info=matroid_info,
        output_file='partition_matroid_lesmis.png'
    )


def test_problem3_visualization():
    """Visualize Problem 3 (diverse matchings without extra matroid constraints)"""
    print("\n" + "="*70)
    print("VISUALIZING PROBLEM 3 (DIVERSE MATCHINGS)")
    print("="*70)
    
    graph, weights = load_weighted_graph(Path('../lesmis'))
    print(f"Dataset: {len(graph.nodes())} nodes, {len(graph.edges())} edges\n")
    
    # Use uniform matroid with large capacity (essentially no extra constraint)
    oracle = uniform_matroid_oracle(15)
    
    print("Finding k=5 diverse matchings (r=15, δ=0.3)...\n")
    
    matchings, min_dist = diverse_matroid_matching(
        graph, weights, oracle,
        k=5, r=15, delta=0.3,
        max_iterations=200, seed=42
    )
    
    print(f"✅ Found {len(matchings)} matchings, min distance: {min_dist:.2f}\n")
    
    for i, matching in enumerate(matchings, 1):
        total_w = sum(weights.get(e, 0) for e in matching)
        print(f"  Matching {i}: |M|={len(matching)}, weight={total_w:.1f}")
    
    # Visualize
    visualize_diverse_matchings(
        graph, matchings, weights,
        "Problem 3: Diverse Matchings - Les Misérables\nk=5, r=15, δ=0.3",
        output_file='problem3_lesmis.png'
    )


def main():
    """Run all visualizations"""
    print("="*70)
    print("MATROID VISUALIZATION FOR LES MISÉRABLES DATASET")
    print("="*70)
    print("\nGenerating visualizations...\n")
    
    # Test Problem 3
    test_problem3_visualization()
    
    # Test Uniform Matroid
    test_uniform_matroid_visualization()
    
    # Test Partition Matroid
    test_partition_matroid_visualization()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 problem3_lesmis.png - Problem 3 diverse matchings")
    print("  📊 uniform_matroid_lesmis.png - Uniform matroid constraints")
    print("  📊 partition_matroid_lesmis.png - Partition matroid constraints")
    print("\nRed edges = High-weight, Blue edges = Low-weight (in partition plot)")


if __name__ == "__main__":
    main()

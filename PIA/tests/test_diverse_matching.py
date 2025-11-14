import sys
from pathlib import Path
import random

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diverse_matching import (
    canonical_edge,
    compute_temporal_weights,
    find_single_diverse_matching,
    orchestrate_diverse_matchings,
    weighted_collaboration_distance,
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


def build_test_graph():
    graph = nx.Graph()
    weights = {
        canonical_edge(0, 1): 2.0,
        canonical_edge(0, 2): 4.0,
        canonical_edge(1, 2): 1.0,
        canonical_edge(3, 4): 5.0,
        canonical_edge(3, 5): 1.0,
        canonical_edge(4, 5): 1.5,
    }
    for (u, v), weight in weights.items():
        graph.add_edge(u, v, weight=weight)
    return graph, weights


def test_weighted_collaboration_distance_symmetric_difference():
    _, weights = build_test_graph()
    m_a = frozenset({canonical_edge(0, 2), canonical_edge(3, 4)})
    m_b = frozenset({canonical_edge(0, 1), canonical_edge(3, 5)})

    distance = weighted_collaboration_distance(weights, m_a, m_b)
    assert pytest.approx(distance, rel=1e-6) == 12.0


def test_compute_temporal_weights_penalises_overlap():
    _, weights = build_test_graph()
    history = [frozenset({canonical_edge(0, 2), canonical_edge(3, 4)})]
    dual_weights = [1.0]
    adjusted = compute_temporal_weights(weights, history, dual_weights, eta=0.4)

    assert adjusted[canonical_edge(0, 2)] < weights[canonical_edge(0, 2)]
    assert adjusted[canonical_edge(3, 4)] < weights[canonical_edge(3, 4)]
    assert pytest.approx(adjusted[canonical_edge(0, 1)], rel=1e-6) == weights[canonical_edge(0, 1)]


def test_find_single_diverse_matching_avoids_history_edges():
    graph, weights = build_test_graph()
    history = [frozenset({canonical_edge(0, 2), canonical_edge(3, 4)})]
    rng = random.Random(123)

    candidate = find_single_diverse_matching(
        graph,
        weights,
        history,
        max_size=2,
        iterations=60,
        eta=0.3,
        rng=rng,
    )

    assert candidate
    assert candidate != history[0]
    overlap_weight = sum(
        weights[edge] for edge in candidate & history[0]
    )
    assert overlap_weight < sum(weights[edge] for edge in history[0])


def test_orchestrate_diverse_matchings_returns_k_unique():
    graph, weights = build_test_graph()
    matchings, min_distance = orchestrate_diverse_matchings(
        graph,
        weights,
        2,
        2,
        delta=0.2,
        seed=2024,
    )

    assert len(matchings) == 2
    assert matchings[0] != matchings[1]
    assert min_distance > 0
    assert all(len(m) <= 2 for m in matchings)


def test_orchestrate_respects_delta_iterations_budget():
    graph, weights = build_test_graph()
    matchings, min_distance = orchestrate_diverse_matchings(
        graph,
        weights,
        2,
        2,
        delta=0.3,
        max_iterations=20,
        seed=99,
    )

    assert len(matchings) == 2
    assert min_distance >= 0


# ============================================================================
# Tests for Section 2.4.1: Matching Applications
# ============================================================================

def test_is_valid_matching_accepts_proper_matching():
    """Test that a valid matching (no shared vertices) is accepted."""
    edges = {canonical_edge(0, 1), canonical_edge(2, 3), canonical_edge(4, 5)}
    assert is_valid_matching(edges) is True


def test_is_valid_matching_rejects_shared_vertex():
    """Test that edges sharing a vertex are rejected."""
    edges = {canonical_edge(0, 1), canonical_edge(1, 2)}
    assert is_valid_matching(edges) is False


def test_is_valid_matching_empty_set():
    """Test that empty set is a valid matching."""
    assert is_valid_matching(set()) is True


def test_matching_matroid_oracle_independent_set():
    """Test matching matroid oracle accepts independent sets."""
    edges = {canonical_edge(0, 1), canonical_edge(2, 3)}
    assert matching_matroid_oracle(edges) is True


def test_matching_matroid_oracle_dependent_set():
    """Test matching matroid oracle rejects dependent sets."""
    edges = {canonical_edge(0, 1), canonical_edge(0, 2)}
    assert matching_matroid_oracle(edges) is False


# ============================================================================
# Tests for Section 2.4.2: Matroid Applications
# ============================================================================

def test_uniform_matroid_accepts_small_sets():
    """Test uniform matroid U(3, n) accepts sets of size <= 3."""
    oracle = uniform_matroid_oracle(3)
    assert oracle({canonical_edge(0, 1)}) is True
    assert oracle({canonical_edge(0, 1), canonical_edge(2, 3)}) is True
    assert oracle({canonical_edge(0, 1), canonical_edge(2, 3), canonical_edge(4, 5)}) is True


def test_uniform_matroid_rejects_large_sets():
    """Test uniform matroid U(3, n) rejects sets of size > 3."""
    oracle = uniform_matroid_oracle(3)
    edges = {
        canonical_edge(0, 1),
        canonical_edge(2, 3),
        canonical_edge(4, 5),
        canonical_edge(6, 7),
    }
    assert oracle(edges) is False


def test_partition_matroid_respects_capacities():
    """Test partition matroid respects group capacities."""
    partition = {
        canonical_edge(0, 1): 0,
        canonical_edge(2, 3): 0,
        canonical_edge(4, 5): 1,
        canonical_edge(6, 7): 1,
        canonical_edge(8, 9): 2,
    }
    capacities = {0: 1, 1: 2, 2: 1}
    oracle = partition_matroid_oracle(partition, capacities)
    
    # Should accept: 1 from group 0, 2 from group 1
    assert oracle({canonical_edge(0, 1), canonical_edge(4, 5), canonical_edge(6, 7)}) is True
    
    # Should reject: 2 from group 0 (capacity is 1)
    assert oracle({canonical_edge(0, 1), canonical_edge(2, 3)}) is False


def test_graphic_matroid_accepts_forest():
    """Test graphic matroid accepts acyclic edge sets (forests)."""
    nodes = {0, 1, 2, 3, 4}
    oracle = graphic_matroid_oracle(nodes)
    
    # Tree edges: no cycle
    forest_edges = {canonical_edge(0, 1), canonical_edge(1, 2), canonical_edge(3, 4)}
    assert oracle(forest_edges) is True


def test_graphic_matroid_rejects_cycle():
    """Test graphic matroid rejects edge sets with cycles."""
    nodes = {0, 1, 2}
    oracle = graphic_matroid_oracle(nodes)
    
    # Triangle: forms a cycle
    cycle_edges = {canonical_edge(0, 1), canonical_edge(1, 2), canonical_edge(2, 0)}
    assert oracle(cycle_edges) is False


def test_graphic_matroid_empty_set():
    """Test graphic matroid accepts empty set."""
    nodes = {0, 1, 2}
    oracle = graphic_matroid_oracle(nodes)
    assert oracle(set()) is True


def test_matroid_intersection_finds_common_independent_set():
    """Test matroid intersection finds common independent set."""
    graph, weights = build_test_graph()
    ground_set = set(weights.keys())
    
    # Intersect matching matroid with uniform matroid U(2, n)
    oracle1 = matching_matroid_oracle
    oracle2 = uniform_matroid_oracle(2)
    
    common_set, total_weight = matroid_intersection(
        ground_set, oracle1, oracle2, weights
    )
    
    assert len(common_set) <= 2
    assert matching_matroid_oracle(set(common_set)) is True
    assert total_weight > 0


def test_matroid_intersection_respects_both_constraints():
    """Test that matroid intersection respects both matroid constraints."""
    edges = {
        canonical_edge(0, 1),
        canonical_edge(2, 3),
        canonical_edge(4, 5),
        canonical_edge(0, 2),
    }
    weights = {edge: 1.0 for edge in edges}
    
    # Matching matroid + uniform matroid U(2, n)
    oracle1 = matching_matroid_oracle
    oracle2 = uniform_matroid_oracle(2)
    
    common_set, _ = matroid_intersection(edges, oracle1, oracle2, weights)
    
    # Result should be valid matching
    assert matching_matroid_oracle(set(common_set)) is True
    # Result should satisfy uniform constraint
    assert len(common_set) <= 2


def test_diverse_matroid_matching_generates_k_matchings():
    """Test that diverse_matroid_matching generates k valid matchings."""
    graph, weights = build_test_graph()
    
    # Use uniform matroid as additional constraint
    oracle = uniform_matroid_oracle(2)
    
    matchings, min_distance = diverse_matroid_matching(
        graph,
        weights,
        oracle,
        k=2,
        r=2,
        delta=0.2,
        max_iterations=50,
        seed=42,
    )
    
    assert len(matchings) == 2
    assert all(len(m) <= 2 for m in matchings)
    assert all(matching_matroid_oracle(set(m)) for m in matchings)
    assert all(oracle(set(m)) for m in matchings)
    assert min_distance >= 0


def test_diverse_matroid_matching_respects_partition_constraint():
    """Test diverse_matroid_matching with partition matroid constraint."""
    graph, weights = build_test_graph()
    
    # Create a more permissive partition: just ensure diversity across weight classes
    partition = {}
    capacities = {}
    for edge, weight in weights.items():
        if weight >= 4.0:
            partition[edge] = 0  # very high weight
        elif weight >= 2.0:
            partition[edge] = 1  # medium weight
        else:
            partition[edge] = 2  # low weight
    
    # Allow up to 2 edges from each partition
    capacities = {0: 2, 1: 2, 2: 2}
    
    oracle = partition_matroid_oracle(partition, capacities)
    
    matchings, _ = diverse_matroid_matching(
        graph,
        weights,
        oracle,
        k=2,
        r=2,
        delta=0.2,
        max_iterations=100,
        seed=123,
    )
    
    assert len(matchings) == 2
    # Verify each matching satisfies partition constraint
    for matching in matchings:
        assert oracle(set(matching)) is True

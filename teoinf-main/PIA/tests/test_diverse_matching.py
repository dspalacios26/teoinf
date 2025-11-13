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
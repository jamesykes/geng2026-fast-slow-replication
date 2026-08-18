from __future__ import annotations

import numpy as np
import pytest

from chu_pair.abm.complete_graph import complete_graph, validate_complete_graph


def test_complete_graph_n2_has_one_edge() -> None:
    graph = complete_graph(2)
    np.testing.assert_array_equal(graph.edge_u, [0])
    np.testing.assert_array_equal(graph.edge_v, [1])
    assert graph.edge_count == 1


def test_complete_graph_n3_has_expected_lexicographic_edges() -> None:
    graph = complete_graph(3)
    np.testing.assert_array_equal(graph.edge_u, [0, 0, 1])
    np.testing.assert_array_equal(graph.edge_v, [1, 2, 2])


@pytest.mark.parametrize("num_agents", [2, 3, 7])
def test_complete_graph_has_no_duplicates_and_degree_n_minus_one(num_agents: int) -> None:
    graph = complete_graph(num_agents)
    edge_u = np.asarray(graph.edge_u)
    edge_v = np.asarray(graph.edge_v)

    assert edge_u.ndim == edge_v.ndim == 1
    assert edge_u.size == edge_v.size == num_agents * (num_agents - 1) // 2
    assert np.all(edge_u < edge_v)
    pairs = set(zip(edge_u.tolist(), edge_v.tolist(), strict=True))
    assert len(pairs) == graph.edge_count
    degrees = np.bincount(np.concatenate((edge_u, edge_v)), minlength=num_agents)
    np.testing.assert_array_equal(degrees, np.full(num_agents, num_agents - 1))
    validate_complete_graph(graph)


@pytest.mark.parametrize("num_agents", [0, 1, -3])
def test_complete_graph_rejects_too_few_agents(num_agents: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        complete_graph(num_agents)


def test_complete_graph_rejects_noninteger_agent_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        complete_graph(3.0)  # type: ignore[arg-type]

"""Packed complete-graph endpoints for the finite-population ABM."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class CompleteGraph:
    """One deterministic upper-triangle entry per undirected edge.

    ``num_agents`` is static pytree metadata. The endpoint arrays are dynamic
    leaves, each with shape ``(E,)`` where ``E=n(n-1)/2``.
    """

    num_agents: int
    edge_u: jax.Array
    edge_v: jax.Array

    @property
    def edge_count(self) -> int:
        return self.num_agents * (self.num_agents - 1) // 2

    def tree_flatten(self):
        return (self.edge_u, self.edge_v), self.num_agents

    @classmethod
    def tree_unflatten(cls, num_agents, children):
        edge_u, edge_v = children
        return cls(num_agents=num_agents, edge_u=edge_u, edge_v=edge_v)


def _validate_num_agents(num_agents: int) -> int:
    if isinstance(num_agents, bool) or not isinstance(num_agents, int):
        raise TypeError("num_agents must be an integer")
    if num_agents < 2:
        raise ValueError("num_agents must be at least 2")
    return num_agents


def complete_graph(num_agents: int) -> CompleteGraph:
    """Build packed lexicographic edges ``(0,1),(0,2),...`` in O(E) memory."""

    n = _validate_num_agents(num_agents)
    edge_count = n * (n - 1) // 2
    repeats = np.arange(n - 1, 0, -1, dtype=np.int32)
    edge_u = np.repeat(np.arange(n - 1, dtype=np.int32), repeats)
    edge_v = np.fromiter(
        (v for u in range(n - 1) for v in range(u + 1, n)),
        dtype=np.int32,
        count=edge_count,
    )
    graph = CompleteGraph(
        num_agents=n,
        edge_u=jnp.asarray(edge_u),
        edge_v=jnp.asarray(edge_v),
    )
    validate_complete_graph(graph)
    return graph


def validate_complete_graph(graph: CompleteGraph) -> None:
    """Host-side structural validation; never called from a compiled step."""

    n = _validate_num_agents(graph.num_agents)
    edge_u = np.asarray(graph.edge_u)
    edge_v = np.asarray(graph.edge_v)
    expected_edges = n * (n - 1) // 2
    if edge_u.shape != (expected_edges,) or edge_v.shape != (expected_edges,):
        raise ValueError(f"complete graph endpoints must each have shape {(expected_edges,)}")
    if np.any(edge_u >= edge_v):
        raise ValueError("complete graph endpoints must satisfy edge_u < edge_v")
    encoded = edge_u.astype(np.int64) * n + edge_v.astype(np.int64)
    if np.unique(encoded).size != expected_edges:
        raise ValueError("complete graph contains duplicate unordered edges")
    degrees = np.bincount(np.concatenate((edge_u, edge_v)), minlength=n)
    if not np.array_equal(degrees, np.full(n, n - 1, dtype=degrees.dtype)):
        raise ValueError("every complete-graph agent must have degree n-1")

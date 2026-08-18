"""Explicit-key ABM initialisation without pair-grid projection after t=0."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..config import validate_abm_seed
from ..initial_conditions import DiscreteQHistogram
from .complete_graph import CompleteGraph
from .simulation import ABMState


@dataclass(frozen=True, slots=True)
class InitializationMetadata:
    mode: str
    num_agents: int
    edge_count: int
    abm_seed: int
    dtype: str
    histogram_seed: int | None = None
    num_runs: int = 1


@dataclass(frozen=True, slots=True)
class InitializedABM:
    state: ABMState
    simulation_key: jax.Array
    metadata: InitializationMetadata


def _validated_dtype(dtype):
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("ABM Q-values must use float32 or float64")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.read("jax_enable_x64"):
        raise ValueError("float64 requires JAX_ENABLE_X64=1 before importing JAX")
    return dtype


def _validate_size(size: int, name: str = "size") -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return size


def _sample_grid_q_arrays(key, points, probabilities, size: int):
    indices = jax.random.choice(
        key,
        points.shape[0],
        shape=(size,),
        replace=True,
        p=probabilities,
    )
    return points[indices]


def sample_grid_matched_q(
    key: jax.Array,
    histogram: DiscreteQHistogram,
    size: int,
    *,
    dtype=jnp.float32,
) -> jax.Array:
    """Sample independent Q-vectors from the exact stored histogram mass."""

    size = _validate_size(size)
    dtype = _validated_dtype(dtype)
    points = jnp.asarray(histogram.grid.flat_q_points, dtype=dtype)
    probabilities = jnp.asarray(histogram.mass.ravel(), dtype=dtype)
    return _sample_grid_q_arrays(key, points, probabilities, size)


def sample_continuous_paper_q(
    key: jax.Array,
    size: int,
    *,
    q_min: float = -0.1,
    q_max: float = 1.2,
    beta_c: tuple[float, float] = (20.0, 80.0),
    beta_d: tuple[float, float] = (80.0, 20.0),
    dtype=jnp.float32,
) -> jax.Array:
    """Sample independent scaled-Beta coordinates from the paper-like law."""

    size = _validate_size(size)
    dtype = _validated_dtype(dtype)
    if not q_max > q_min:
        raise ValueError("q_max must exceed q_min")
    if any(parameter <= 0.0 for parameter in (*beta_c, *beta_d)):
        raise ValueError("Beta shape parameters must be positive")
    key_c, key_d = jax.random.split(key)
    q_c = jax.random.beta(key_c, beta_c[0], beta_c[1], shape=(size,), dtype=dtype)
    q_d = jax.random.beta(key_d, beta_d[0], beta_d[1], shape=(size,), dtype=dtype)
    span = jnp.asarray(q_max - q_min, dtype=dtype)
    lower = jnp.asarray(q_min, dtype=dtype)
    return jnp.stack((lower + span * q_c, lower + span * q_d), axis=-1)


def sample_edge_states(key: jax.Array, edge_count: int) -> jax.Array:
    """Sample independent SH/PD states with probability one half each."""

    edge_count = _validate_size(edge_count, "edge_count")
    return jax.random.bernoulli(key, 0.5, shape=(edge_count,)).astype(jnp.int8)


def initialize_grid_matched(
    graph: CompleteGraph,
    histogram: DiscreteQHistogram,
    *,
    abm_seed: int,
    dtype=jnp.float32,
) -> InitializedABM:
    """Controlled initialisation with distinct histogram and ABM seeds."""

    dtype = _validated_dtype(dtype)
    abm_seed = validate_abm_seed(abm_seed)
    root_key = jax.random.PRNGKey(abm_seed)
    q_key, state_key, simulation_key = jax.random.split(root_key, 3)
    q_values = sample_grid_matched_q(q_key, histogram, graph.num_agents, dtype=dtype)
    edge_states = sample_edge_states(state_key, graph.edge_count)
    metadata = InitializationMetadata(
        mode="grid_matched",
        num_agents=graph.num_agents,
        edge_count=graph.edge_count,
        histogram_seed=histogram.seed,
        abm_seed=abm_seed,
        dtype=str(np.dtype(dtype)),
    )
    return InitializedABM(
        state=ABMState(q_values=q_values, edge_states=edge_states),
        simulation_key=simulation_key,
        metadata=metadata,
    )


def initialize_continuous_paper(
    graph: CompleteGraph,
    *,
    abm_seed: int,
    dtype=jnp.float32,
) -> InitializedABM:
    """Paper-like continuous scaled-Beta Q and independent half-state mode."""

    dtype = _validated_dtype(dtype)
    abm_seed = validate_abm_seed(abm_seed)
    root_key = jax.random.PRNGKey(abm_seed)
    q_key, state_key, simulation_key = jax.random.split(root_key, 3)
    q_values = sample_continuous_paper_q(q_key, graph.num_agents, dtype=dtype)
    edge_states = sample_edge_states(state_key, graph.edge_count)
    metadata = InitializationMetadata(
        mode="continuous_paper",
        num_agents=graph.num_agents,
        edge_count=graph.edge_count,
        histogram_seed=None,
        abm_seed=abm_seed,
        dtype=str(np.dtype(dtype)),
    )
    return InitializedABM(
        state=ABMState(q_values=q_values, edge_states=edge_states),
        simulation_key=simulation_key,
        metadata=metadata,
    )


def initialize_grid_matched_batch(
    graph: CompleteGraph,
    histogram: DiscreteQHistogram,
    *,
    abm_seed: int,
    num_runs: int,
    dtype=jnp.float32,
) -> InitializedABM:
    """Vmap independent controlled initial states from explicit run keys."""

    num_runs = _validate_size(num_runs, "num_runs")
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    dtype = _validated_dtype(dtype)
    abm_seed = validate_abm_seed(abm_seed)
    points = jnp.asarray(histogram.grid.flat_q_points, dtype=dtype)
    probabilities = jnp.asarray(histogram.mass.ravel(), dtype=dtype)
    run_keys = jax.random.split(jax.random.PRNGKey(abm_seed), num_runs)

    def one_run(run_key):
        q_key, state_key, simulation_key = jax.random.split(run_key, 3)
        q_values = _sample_grid_q_arrays(
            q_key, points, probabilities, graph.num_agents
        )
        edge_states = sample_edge_states(state_key, graph.edge_count)
        return ABMState(q_values=q_values, edge_states=edge_states), simulation_key

    states, simulation_keys = jax.vmap(one_run)(run_keys)
    metadata = InitializationMetadata(
        mode="grid_matched",
        num_agents=graph.num_agents,
        edge_count=graph.edge_count,
        histogram_seed=histogram.seed,
        abm_seed=abm_seed,
        dtype=str(np.dtype(dtype)),
        num_runs=num_runs,
    )
    return InitializedABM(state=states, simulation_key=simulation_keys, metadata=metadata)


def initialize_continuous_paper_batch(
    graph: CompleteGraph,
    *,
    abm_seed: int,
    num_runs: int,
    dtype=jnp.float32,
) -> InitializedABM:
    """Vmap independent paper-like continuous initial states."""

    num_runs = _validate_size(num_runs, "num_runs")
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    dtype = _validated_dtype(dtype)
    abm_seed = validate_abm_seed(abm_seed)
    run_keys = jax.random.split(jax.random.PRNGKey(abm_seed), num_runs)

    def one_run(run_key):
        q_key, state_key, simulation_key = jax.random.split(run_key, 3)
        q_values = sample_continuous_paper_q(q_key, graph.num_agents, dtype=dtype)
        edge_states = sample_edge_states(state_key, graph.edge_count)
        return ABMState(q_values=q_values, edge_states=edge_states), simulation_key

    states, simulation_keys = jax.vmap(one_run)(run_keys)
    metadata = InitializationMetadata(
        mode="continuous_paper",
        num_agents=graph.num_agents,
        edge_count=graph.edge_count,
        histogram_seed=None,
        abm_seed=abm_seed,
        dtype=str(np.dtype(dtype)),
        num_runs=num_runs,
    )
    return InitializedABM(state=states, simulation_key=simulation_keys, metadata=metadata)

"""Synchronous deterministic and stochastic JAX ABM evolution."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..model import PAYOFF_TENSOR, TRANSITION_TENSOR, Action, q_learning_velocity
from ..policies import _two_action_boltzmann_probabilities
from .complete_graph import CompleteGraph


# These device constants are derived from, rather than copies of, the shared
# authoritative tensors. A strict x64 run must enable x64 before importing us.
_PAYOFF_TENSOR_JAX = jnp.asarray(PAYOFF_TENSOR)
_TRANSITION_TENSOR_JAX = jnp.asarray(TRANSITION_TENSOR, dtype=jnp.int8)


class ABMState(NamedTuple):
    """Physical state at one labelled time: Q ``(n,2)`` and states ``(E,)``."""

    q_values: jax.Array
    edge_states: jax.Array


class DeterministicStepRecord(NamedTuple):
    """One-step debug record; edge-sized fields are never retained by scan."""

    q_t: jax.Array
    edge_states_t: jax.Array
    actions_t: jax.Array
    edge_actions_u_t: jax.Array
    edge_actions_v_t: jax.Array
    edge_payoffs_u_t: jax.Array
    edge_payoffs_v_t: jax.Array
    payoff_sums_t: jax.Array
    payoff_square_sums_t: jax.Array
    rewards_t: jax.Array
    selected_q_t: jax.Array
    selected_velocities_t: jax.Array
    q_t_plus_1: jax.Array
    edge_states_t_plus_1: jax.Array


class StepRecord(NamedTuple):
    """Lean scan record for the transition from labelled state t to t+1."""

    q_t: jax.Array
    action_probabilities_t: jax.Array
    actions_t: jax.Array
    rewards_t: jax.Array
    selected_velocities_t: jax.Array
    q_t_plus_1: jax.Array
    edge_state_proportions_t: jax.Array
    edge_state_proportions_t_plus_1: jax.Array


class InstrumentedStepRecord(NamedTuple):
    """Lean Phase 3A source-time record with agent-level payoff sums."""

    q_t: jax.Array
    action_probabilities_t: jax.Array
    actions_t: jax.Array
    selected_q_t: jax.Array
    rewards_t: jax.Array
    selected_velocities_t: jax.Array
    payoff_sums_t: jax.Array
    payoff_square_sums_t: jax.Array
    q_t_plus_1: jax.Array
    edge_state_proportions_t: jax.Array
    edge_state_proportions_t_plus_1: jax.Array


class SimulationResult(NamedTuple):
    final_state: ABMState
    final_key: jax.Array
    records: StepRecord | InstrumentedStepRecord


def action_probabilities(q_values: jax.Array, tau) -> jax.Array:
    """Stable JAX policy in shared action order ``(C,D)``."""

    q_array = jnp.asarray(q_values)
    if q_array.ndim < 1 or q_array.shape[-1] != 2:
        raise ValueError("q_values must have final dimension 2")
    tau_array = jnp.asarray(tau, dtype=q_array.dtype)
    return _two_action_boltzmann_probabilities(q_array, tau_array, jnp)


def sample_actions(key: jax.Array, q_values: jax.Array, tau) -> jax.Array:
    """Draw one action per agent; callers reuse this vector on every edge."""

    probabilities = action_probabilities(q_values, tau)
    return jax.random.bernoulli(key, probabilities[..., int(Action.D)]).astype(jnp.int8)


def _validate_step_shapes(
    state: ABMState,
    actions: jax.Array,
    graph: CompleteGraph,
) -> None:
    q_values = state.q_values
    edge_states = state.edge_states
    if q_values.ndim != 2 or q_values.shape[1] != 2:
        raise ValueError("q_values must have shape (n, 2)")
    if q_values.shape[0] != graph.num_agents:
        raise ValueError("q_values and graph disagree on num_agents")
    if edge_states.shape != (graph.edge_count,):
        raise ValueError("edge_states must have shape (E,)")
    if graph.edge_u.shape != (graph.edge_count,) or graph.edge_v.shape != (graph.edge_count,):
        raise ValueError("graph endpoint arrays must have shape (E,)")
    if actions.shape != (graph.num_agents,):
        raise ValueError("actions must have shape (n,)")
    if not jnp.issubdtype(q_values.dtype, jnp.floating):
        raise TypeError("q_values must use a floating dtype")


def step_given_actions(
    state: ABMState,
    actions: jax.Array,
    graph: CompleteGraph,
    alpha,
) -> tuple[ABMState, DeterministicStepRecord]:
    """Apply one synchronous step from supplied one-action-per-agent choices."""

    actions_t = jnp.asarray(actions, dtype=jnp.int8)
    _validate_step_shapes(state, actions_t, graph)
    q_t = state.q_values
    edge_states_t = state.edge_states.astype(jnp.int8)
    edge_u = graph.edge_u
    edge_v = graph.edge_v
    edge_actions_u = actions_t[edge_u]
    edge_actions_v = actions_t[edge_v]

    payoff_tensor = _PAYOFF_TENSOR_JAX.astype(q_t.dtype)
    payoff_u = payoff_tensor[edge_states_t, edge_actions_u, edge_actions_v]
    payoff_v = payoff_tensor[edge_states_t, edge_actions_v, edge_actions_u]

    reward_sums = jnp.zeros((graph.num_agents,), dtype=q_t.dtype)
    reward_sums = reward_sums.at[edge_u].add(payoff_u)
    reward_sums = reward_sums.at[edge_v].add(payoff_v)
    payoff_square_sums = jnp.zeros((graph.num_agents,), dtype=q_t.dtype)
    payoff_square_sums = payoff_square_sums.at[edge_u].add(payoff_u * payoff_u)
    payoff_square_sums = payoff_square_sums.at[edge_v].add(payoff_v * payoff_v)
    rewards = reward_sums / jnp.asarray(graph.num_agents - 1, dtype=q_t.dtype)

    agent_indices = jnp.arange(graph.num_agents, dtype=jnp.int32)
    selected_q = q_t[agent_indices, actions_t]
    alpha_array = jnp.asarray(alpha, dtype=q_t.dtype)
    velocities = q_learning_velocity(selected_q, rewards, alpha_array)
    q_t_plus_1 = q_t.at[agent_indices, actions_t].add(velocities)

    edge_states_t_plus_1 = _TRANSITION_TENSOR_JAX[
        edge_states_t, edge_actions_u, edge_actions_v
    ]
    next_state = ABMState(q_values=q_t_plus_1, edge_states=edge_states_t_plus_1)
    record = DeterministicStepRecord(
        q_t=q_t,
        edge_states_t=edge_states_t,
        actions_t=actions_t,
        edge_actions_u_t=edge_actions_u,
        edge_actions_v_t=edge_actions_v,
        edge_payoffs_u_t=payoff_u,
        edge_payoffs_v_t=payoff_v,
        payoff_sums_t=reward_sums,
        payoff_square_sums_t=payoff_square_sums,
        rewards_t=rewards,
        selected_q_t=selected_q,
        selected_velocities_t=velocities,
        q_t_plus_1=q_t_plus_1,
        edge_states_t_plus_1=edge_states_t_plus_1,
    )
    return next_state, record


def _edge_state_proportions(edge_states: jax.Array, dtype) -> jax.Array:
    counts = jnp.stack(
        (jnp.sum(edge_states == 0), jnp.sum(edge_states == 1)),
        axis=0,
    )
    return counts.astype(dtype) / jnp.asarray(edge_states.shape[0], dtype=dtype)


def stochastic_step(
    state: ABMState,
    key: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
) -> tuple[ABMState, jax.Array, StepRecord]:
    """Sample actions from Q_t, then apply the deterministic old-state core."""

    next_state, next_key, probabilities_t, debug = _stochastic_step_components(
        state, key, graph, alpha, tau
    )
    record = StepRecord(
        q_t=debug.q_t,
        action_probabilities_t=probabilities_t,
        actions_t=debug.actions_t,
        rewards_t=debug.rewards_t,
        selected_velocities_t=debug.selected_velocities_t,
        q_t_plus_1=debug.q_t_plus_1,
        edge_state_proportions_t=_edge_state_proportions(
            debug.edge_states_t, debug.q_t.dtype
        ),
        edge_state_proportions_t_plus_1=_edge_state_proportions(
            debug.edge_states_t_plus_1, debug.q_t.dtype
        ),
    )
    return next_state, next_key, record


def _stochastic_step_components(
    state: ABMState,
    key: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
):
    """Share the exact key/action/dynamics path between record variants."""

    action_key, next_key = jax.random.split(key)
    probabilities_t = action_probabilities(state.q_values, tau)
    actions_t = sample_actions(action_key, state.q_values, tau)
    next_state, debug = step_given_actions(state, actions_t, graph, alpha)
    return next_state, next_key, probabilities_t, debug


def stochastic_step_instrumented(
    state: ABMState,
    key: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
) -> tuple[ABMState, jax.Array, InstrumentedStepRecord]:
    """Run the same stochastic step while retaining Phase 3A agent moments."""

    next_state, next_key, probabilities_t, debug = _stochastic_step_components(
        state, key, graph, alpha, tau
    )
    record = InstrumentedStepRecord(
        q_t=debug.q_t,
        action_probabilities_t=probabilities_t,
        actions_t=debug.actions_t,
        selected_q_t=debug.selected_q_t,
        rewards_t=debug.rewards_t,
        selected_velocities_t=debug.selected_velocities_t,
        payoff_sums_t=debug.payoff_sums_t,
        payoff_square_sums_t=debug.payoff_square_sums_t,
        q_t_plus_1=debug.q_t_plus_1,
        edge_state_proportions_t=_edge_state_proportions(
            debug.edge_states_t, debug.q_t.dtype
        ),
        edge_state_proportions_t_plus_1=_edge_state_proportions(
            debug.edge_states_t_plus_1, debug.q_t.dtype
        ),
    )
    return next_state, next_key, record


def simulate(
    initial_state: ABMState,
    initial_key: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
    *,
    steps: int,
) -> SimulationResult:
    """Run one keyed trajectory with ``lax.scan``.

    ``steps`` is static because it determines output shapes. Values of Q,
    endpoints, keys, alpha, and tau remain ordinary dynamic JAX arguments.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")

    def body(carry, _):
        state, key = carry
        next_state, next_key, record = stochastic_step(state, key, graph, alpha, tau)
        return (next_state, next_key), record

    (final_state, final_key), records = jax.lax.scan(
        body,
        (initial_state, initial_key),
        xs=None,
        length=steps,
    )
    return SimulationResult(final_state=final_state, final_key=final_key, records=records)


simulate_jit = partial(jax.jit, static_argnames=("steps",))(simulate)


def simulate_instrumented(
    initial_state: ABMState,
    initial_key: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
    *,
    steps: int,
) -> SimulationResult:
    """Run a trajectory retaining only agent-level Phase 3A instrumentation."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")

    def body(carry, _):
        state, key = carry
        next_state, next_key, record = stochastic_step_instrumented(
            state, key, graph, alpha, tau
        )
        return (next_state, next_key), record

    (final_state, final_key), records = jax.lax.scan(
        body,
        (initial_state, initial_key),
        xs=None,
        length=steps,
    )
    return SimulationResult(final_state=final_state, final_key=final_key, records=records)


simulate_instrumented_jit = partial(jax.jit, static_argnames=("steps",))(
    simulate_instrumented
)


def simulate_debug(
    initial_state: ABMState,
    initial_key: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
    *,
    steps: int,
) -> SimulationResult:
    """Readable Python-loop driver for a small number of validation steps."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if steps == 0:
        return simulate(initial_state, initial_key, graph, alpha, tau, steps=0)

    state = initial_state
    key = initial_key
    records = []
    for _ in range(steps):
        state, key, record = stochastic_step(state, key, graph, alpha, tau)
        records.append(record)
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)
    return SimulationResult(final_state=state, final_key=key, records=stacked)


def simulate_batch(
    initial_states: ABMState,
    initial_keys: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
    *,
    steps: int,
) -> SimulationResult:
    """Vmap independent runs; leading state/key axes identify runs."""

    run = lambda state, key: simulate(state, key, graph, alpha, tau, steps=steps)
    return jax.vmap(run)(initial_states, initial_keys)


simulate_batch_jit = partial(jax.jit, static_argnames=("steps",))(simulate_batch)


def simulate_instrumented_batch(
    initial_states: ABMState,
    initial_keys: jax.Array,
    graph: CompleteGraph,
    alpha,
    tau,
    *,
    steps: int,
) -> SimulationResult:
    """Vmap independent instrumented trajectories while preserving the run axis."""

    run = lambda state, key: simulate_instrumented(
        state, key, graph, alpha, tau, steps=steps
    )
    return jax.vmap(run)(initial_states, initial_keys)


simulate_instrumented_batch_jit = partial(jax.jit, static_argnames=("steps",))(
    simulate_instrumented_batch
)

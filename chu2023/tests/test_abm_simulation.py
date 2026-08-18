from __future__ import annotations

import ast
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

import chu_pair.abm.simulation as simulation_module
from chu_pair.abm import (
    ABMState,
    action_probabilities,
    complete_graph,
    simulate,
    simulate_batch,
    simulate_debug,
    stochastic_step,
    step_given_actions,
)
from chu_pair.model import PAYOFF_TENSOR, TRANSITION_TENSOR, State


Q_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
ATOL = 1e-12 if Q_DTYPE == jnp.float64 else 3e-7


def small_state() -> tuple:
    graph = complete_graph(3)
    state = ABMState(
        q_values=jnp.asarray(
            [[0.23, 0.71], [0.37, 0.44], [0.61, 0.14]], dtype=Q_DTYPE
        ),
        edge_states=jnp.asarray([State.PD, State.SH, State.SH], dtype=jnp.int8),
    )
    return graph, state


def assert_trees_close(actual, expected) -> None:
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if jnp.issubdtype(actual_leaf.dtype, jnp.inexact):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=0.0, atol=ATOL)
        else:
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_stochastic_record_uses_source_q_policy() -> None:
    graph, state = small_state()
    next_state, _, record = stochastic_step(
        state, jax.random.PRNGKey(8), graph, alpha=0.4, tau=2.0
    )
    expected_cooperate = np.asarray(
        [0.276878194875610, 0.465057054841785, 0.719099657416384]
    )
    np.testing.assert_allclose(
        record.action_probabilities_t[:, 0], expected_cooperate, rtol=0.0, atol=ATOL
    )
    post_probabilities = action_probabilities(next_state.q_values, 2.0)
    assert not np.allclose(record.action_probabilities_t, post_probabilities, atol=1e-4)


def test_eager_and_jitted_stochastic_steps_agree_for_same_key() -> None:
    graph, state = small_state()
    key = jax.random.PRNGKey(9012)
    eager = stochastic_step(state, key, graph, 0.4, 2.0)
    compiled = jax.jit(stochastic_step)(state, key, graph, 0.4, 2.0)
    assert_trees_close(compiled, eager)


def test_scan_matches_debug_repeated_key_threading() -> None:
    graph, state = small_state()
    key = jax.random.PRNGKey(41)
    scanned = simulate(state, key, graph, 0.4, 2.0, steps=6)
    repeated = simulate_debug(state, key, graph, 0.4, 2.0, steps=6)
    assert_trees_close(scanned, repeated)
    expected_final_key = key
    for _ in range(6):
        _, expected_final_key = jax.random.split(expected_final_key)
    np.testing.assert_array_equal(scanned.final_key, expected_final_key)


def test_fixed_key_reproduces_trajectory_and_different_key_changes_stream() -> None:
    graph, state = small_state()
    first = simulate(state, jax.random.PRNGKey(7), graph, 0.4, 2.0, steps=8)
    second = simulate(state, jax.random.PRNGKey(7), graph, 0.4, 2.0, steps=8)
    changed = simulate(state, jax.random.PRNGKey(9), graph, 0.4, 2.0, steps=8)
    assert_trees_close(first, second)
    assert not np.array_equal(first.records.actions_t, changed.records.actions_t)


def test_vmapped_batch_matches_component_runs() -> None:
    graph, state = small_state()
    run_count = 4
    states = ABMState(
        q_values=jnp.broadcast_to(state.q_values, (run_count, *state.q_values.shape)),
        edge_states=jnp.broadcast_to(
            state.edge_states, (run_count, *state.edge_states.shape)
        ),
    )
    keys = jax.random.split(jax.random.PRNGKey(818), run_count)
    batched = simulate_batch(states, keys, graph, 0.4, 2.0, steps=5)

    for run_index in range(run_count):
        component = simulate(
            ABMState(states.q_values[run_index], states.edge_states[run_index]),
            keys[run_index],
            graph,
            0.4,
            2.0,
            steps=5,
        )
        selected = jax.tree_util.tree_map(lambda value: value[run_index], batched)
        assert_trees_close(selected, component)
    assert not np.array_equal(batched.records.actions_t[0], batched.records.actions_t[1])


def test_scan_agrees_with_independent_slow_numpy_dynamics() -> None:
    graph, state = small_state()
    result = simulate(state, jax.random.PRNGKey(333), graph, 0.4, 2.0, steps=5)
    q_values = np.asarray(state.q_values).copy()
    edge_states = np.asarray(state.edge_states).copy()
    edge_u = np.asarray(graph.edge_u)
    edge_v = np.asarray(graph.edge_v)

    for step, actions in enumerate(np.asarray(result.records.actions_t)):
        np.testing.assert_allclose(result.records.q_t[step], q_values, rtol=0.0, atol=ATOL)
        sums = np.zeros(graph.num_agents, dtype=q_values.dtype)
        next_edges = np.empty_like(edge_states)
        for edge, (u, v) in enumerate(zip(edge_u, edge_v, strict=True)):
            old_state = edge_states[edge]
            action_u = actions[u]
            action_v = actions[v]
            sums[u] += PAYOFF_TENSOR[old_state, action_u, action_v]
            sums[v] += PAYOFF_TENSOR[old_state, action_v, action_u]
            next_edges[edge] = TRANSITION_TENSOR[old_state, action_u, action_v]
        rewards = sums / (graph.num_agents - 1)
        selected = q_values[np.arange(graph.num_agents), actions]
        velocities = 0.4 * (rewards - selected)
        next_q = q_values.copy()
        next_q[np.arange(graph.num_agents), actions] += velocities

        np.testing.assert_allclose(result.records.rewards_t[step], rewards, rtol=0.0, atol=ATOL)
        np.testing.assert_allclose(
            result.records.selected_velocities_t[step], velocities, rtol=0.0, atol=ATOL
        )
        np.testing.assert_allclose(result.records.q_t_plus_1[step], next_q, rtol=0.0, atol=ATOL)
        expected_proportions = np.bincount(next_edges, minlength=2) / graph.edge_count
        np.testing.assert_allclose(
            result.records.edge_state_proportions_t_plus_1[step],
            expected_proportions,
            rtol=0.0,
            atol=ATOL,
        )
        q_values = next_q
        edge_states = next_edges

    np.testing.assert_allclose(result.final_state.q_values, q_values, rtol=0.0, atol=ATOL)
    np.testing.assert_array_equal(result.final_state.edge_states, edge_states)


def test_compiled_core_has_no_python_edge_or_agent_loops_or_host_callbacks() -> None:
    graph, state = small_state()
    actions = jnp.asarray([0, 1, 1], dtype=jnp.int8)
    jaxpr = str(jax.make_jaxpr(step_given_actions)(state, actions, graph, 0.4))
    assert "pure_callback" not in jaxpr
    assert "io_callback" not in jaxpr

    source = textwrap.dedent(inspect.getsource(simulation_module.step_given_actions))
    parsed = ast.parse(source)
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(parsed))


def test_default_scan_records_no_edge_sized_history() -> None:
    graph = complete_graph(4)  # E=6 differs from n=4 and steps=3.
    state = ABMState(
        q_values=jnp.zeros((4, 2), dtype=Q_DTYPE),
        edge_states=jnp.zeros((6,), dtype=jnp.int8),
    )
    result = simulate(state, jax.random.PRNGKey(5), graph, 0.4, 2.0, steps=3)

    assert all("edge_payoff" not in field for field in result.records._fields)
    assert all("edge_states" not in field for field in result.records._fields)
    for leaf in jax.tree_util.tree_leaves(result.records):
        assert not (leaf.ndim >= 2 and leaf.shape[:2] == (3, graph.edge_count))

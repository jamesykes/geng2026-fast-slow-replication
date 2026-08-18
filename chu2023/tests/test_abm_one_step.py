from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chu_pair.abm import ABMState, complete_graph, step_given_actions
from chu_pair.model import Action, State


Q_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
ATOL = 1e-12 if Q_DTYPE == jnp.float64 else 2e-7


def test_n2_fixed_pd_step_is_hand_calculated() -> None:
    graph = complete_graph(2)
    state = ABMState(
        q_values=jnp.asarray([[0.23, 0.80], [0.40, 0.67]], dtype=Q_DTYPE),
        edge_states=jnp.asarray([State.PD], dtype=jnp.int8),
    )
    actions = jnp.asarray([Action.C, Action.D], dtype=jnp.int8)

    next_state, record = step_given_actions(state, actions, graph, alpha=0.4)

    np.testing.assert_allclose(record.edge_payoffs_u_t, [-0.1], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(record.edge_payoffs_v_t, [1.2], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(record.rewards_t, [-0.1, 1.2], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(
        record.selected_velocities_t, [-0.132, 0.212], rtol=0.0, atol=ATOL
    )
    np.testing.assert_allclose(
        next_state.q_values,
        [[0.098, 0.80], [0.40, 0.882]],
        rtol=0.0,
        atol=ATOL,
    )
    np.testing.assert_array_equal(next_state.edge_states, [State.PD])
    assert float(next_state.q_values[0, Action.C]) != 0.1
    assert float(next_state.q_values[1, Action.D]) != 0.88


def test_n3_step_has_oriented_old_state_payoffs_and_synchronous_updates() -> None:
    graph = complete_graph(3)
    q_t = jnp.asarray(
        [[0.23, 0.71], [0.37, 0.44], [0.61, 0.14]],
        dtype=Q_DTYPE,
    )
    state = ABMState(
        q_values=q_t,
        edge_states=jnp.asarray([State.PD, State.SH, State.SH], dtype=jnp.int8),
    )
    actions = jnp.asarray([Action.C, Action.D, Action.D], dtype=jnp.int8)

    next_state, record = step_given_actions(state, actions, graph, alpha=0.4)

    # Edges are (0,1), (0,2), (1,2). Expected values are derived directly
    # from the two old-state payoff matrices, not from production helpers.
    np.testing.assert_array_equal(record.edge_actions_u_t, [Action.C, Action.C, Action.D])
    np.testing.assert_array_equal(record.edge_actions_v_t, [Action.D, Action.D, Action.D])
    np.testing.assert_allclose(
        record.edge_payoffs_u_t, [-0.1, 0.0, 0.1], rtol=0.0, atol=ATOL
    )
    np.testing.assert_allclose(
        record.edge_payoffs_v_t, [1.2, 0.1, 0.1], rtol=0.0, atol=ATOL
    )
    # Payoff sums (-.1, 1.3, .2) are divided by n-1=2 exactly once.
    np.testing.assert_allclose(record.rewards_t, [-0.05, 0.65, 0.10], rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(
        record.selected_velocities_t,
        [-0.112, 0.084, -0.016],
        rtol=0.0,
        atol=ATOL,
    )
    expected_q = np.asarray([[0.118, 0.71], [0.37, 0.524], [0.61, 0.124]])
    np.testing.assert_allclose(next_state.q_values, expected_q, rtol=0.0, atol=ATOL)
    np.testing.assert_array_equal(next_state.edge_states, [State.PD, State.SH, State.PD])

    # Every supplied agent action is reused on all incident edges, and only
    # that agent's selected coordinate changes from the common pre-state Q_t.
    np.testing.assert_array_equal(record.edge_actions_u_t, actions[graph.edge_u])
    np.testing.assert_array_equal(record.edge_actions_v_t, actions[graph.edge_v])
    q_post = np.asarray(next_state.q_values)
    q_pre = np.asarray(q_t)
    for agent, action in enumerate(np.asarray(actions)):
        assert q_post[agent, 1 - action] == q_pre[agent, 1 - action]
    assert q_post[0, Action.C] != np.around(q_post[0, Action.C], 2)
    assert q_post[1, Action.D] != np.around(q_post[1, Action.D], 2)
    assert q_post[2, Action.D] != np.around(q_post[2, Action.D], 2)


def test_jitted_and_eager_deterministic_steps_agree() -> None:
    graph = complete_graph(3)
    state = ABMState(
        q_values=jnp.asarray([[0.2, 0.8], [0.7, 0.1], [0.4, 0.3]], dtype=Q_DTYPE),
        edge_states=jnp.asarray([State.SH, State.PD, State.SH], dtype=jnp.int8),
    )
    actions = jnp.asarray([Action.D, Action.C, Action.D], dtype=jnp.int8)

    eager = step_given_actions(state, actions, graph, 0.37)
    compiled = jax.jit(step_given_actions)(state, actions, graph, 0.37)
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        if jnp.issubdtype(eager_leaf.dtype, jnp.inexact):
            np.testing.assert_allclose(compiled_leaf, eager_leaf, rtol=0.0, atol=ATOL)
        else:
            np.testing.assert_array_equal(compiled_leaf, eager_leaf)

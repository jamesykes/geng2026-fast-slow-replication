from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.abm import (
    ABMState,
    QBinSpec,
    aggregate_variance_records,
    complete_graph,
    derive_variance_moments,
    simulate_batch,
    simulate_instrumented,
    simulate_instrumented_batch,
    simulate_instrumented_batch_jit,
    step_given_actions,
)
from chu_pair.model import Action, State


Q_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
ATOL = 1e-12 if Q_DTYPE == jnp.float64 else 3e-7


def _synthetic_records(
    q_values: np.ndarray,
    actions: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    *,
    alpha: float = 0.4,
):
    num_agents = q_values.shape[0]
    selected_q = q_values[np.arange(num_agents), actions]
    rewards = s1 / (num_agents - 1)
    velocities = alpha * (rewards - selected_q)
    leading = (1, 1)
    return SimpleNamespace(
        q_t=q_values.reshape(*leading, num_agents, 2),
        actions_t=actions.reshape(*leading, num_agents),
        selected_q_t=selected_q.reshape(*leading, num_agents),
        rewards_t=rewards.reshape(*leading, num_agents),
        selected_velocities_t=velocities.reshape(*leading, num_agents),
        payoff_sums_t=s1.reshape(*leading, num_agents),
        payoff_square_sums_t=s2.reshape(*leading, num_agents),
    )


def _one_stratum(records, *, num_agents: int, alpha: float = 0.4):
    bins = QBinSpec(np.asarray([-1.0, 2.0]), np.asarray([-1.0, 2.0]))
    statistics = aggregate_variance_records(
        records,
        bins,
        num_agents=num_agents,
        alpha=alpha,
    )
    estimates = derive_variance_moments(statistics)
    return statistics, estimates, (0, 0, 0, 0, 0)


def test_fixed_n3_instrumentation_has_hand_calculated_s1_s2_and_source_values() -> None:
    graph = complete_graph(3)
    q_t = jnp.asarray(
        [[0.23, 0.71], [0.37, 0.44], [0.61, 0.14]], dtype=Q_DTYPE
    )
    state = ABMState(
        q_values=q_t,
        edge_states=jnp.asarray([State.PD, State.SH, State.SH], dtype=jnp.int8),
    )
    actions = jnp.asarray([Action.C, Action.D, Action.D], dtype=jnp.int8)

    next_state, record = step_given_actions(state, actions, graph, alpha=0.4)

    np.testing.assert_allclose(record.payoff_sums_t, [-0.1, 1.3, 0.2], rtol=0, atol=ATOL)
    np.testing.assert_allclose(
        record.payoff_square_sums_t, [0.01, 1.45, 0.02], rtol=0, atol=ATOL
    )
    np.testing.assert_allclose(record.rewards_t, [-0.05, 0.65, 0.1], rtol=0, atol=ATOL)
    np.testing.assert_allclose(record.selected_q_t, [0.23, 0.44, 0.14], rtol=0, atol=ATOL)
    np.testing.assert_allclose(
        record.selected_velocities_t, [-0.112, 0.084, -0.016], rtol=0, atol=ATOL
    )
    np.testing.assert_allclose(
        next_state.q_values,
        [[0.118, 0.71], [0.37, 0.524], [0.61, 0.124]],
        rtol=0,
        atol=ATOL,
    )


def test_direct_ordered_opponent_enumeration_equals_s1_squared_minus_s2() -> None:
    graph = complete_graph(3)
    state = ABMState(
        q_values=jnp.asarray([[0.2, 0.8], [0.3, 0.4], [0.6, 0.1]], dtype=Q_DTYPE),
        edge_states=jnp.asarray([State.PD, State.SH, State.SH], dtype=jnp.int8),
    )
    actions = jnp.asarray([Action.C, Action.D, Action.D], dtype=jnp.int8)
    _, record = step_given_actions(state, actions, graph, alpha=0.4)

    incident = [[] for _ in range(3)]
    for edge, (u, v) in enumerate(zip(np.asarray(graph.edge_u), np.asarray(graph.edge_v))):
        incident[u].append(float(record.edge_payoffs_u_t[edge]))
        incident[v].append(float(record.edge_payoffs_v_t[edge]))
    directly_enumerated = []
    for payoffs in incident:
        directly_enumerated.append(
            sum(
                payoffs[h] * payoffs[k]
                for h in range(len(payoffs))
                for k in range(len(payoffs))
                if h != k
            )
        )
    algebraic = np.asarray(record.payoff_sums_t) ** 2 - np.asarray(
        record.payoff_square_sums_t
    )
    np.testing.assert_allclose(algebraic, directly_enumerated, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    ("edge_one_counts", "expected_covariance"),
    [([1.0, 1.0, 1.0, 3.0], 0.0), ([0.0, 0.0, 3.0, 3.0], 0.25)],
)
def test_reward_decomposition_has_known_zero_and_nonzero_covariance(
    edge_one_counts,
    expected_covariance: float,
) -> None:
    s1 = np.asarray(edge_one_counts)
    s2 = s1.copy()  # Binary edge payoffs satisfy Y**2=Y.
    q = np.asarray([[0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6]])
    records = _synthetic_records(q, np.zeros(4, dtype=np.int8), s1, s2)
    _, estimates, index = _one_stratum(records, num_agents=4)

    assert estimates.mu[index] == pytest.approx(0.5, abs=1e-15)
    assert estimates.sigma2[index] == pytest.approx(0.25, abs=1e-15)
    assert estimates.covariance[index] == pytest.approx(expected_covariance, abs=1e-15)
    assert estimates.decomposed_reward_variance[index] == pytest.approx(
        estimates.direct_reward_variance[index], abs=1e-15
    )


def test_finite_bin_velocity_identity_includes_q_variance_and_reward_q_covariance() -> None:
    s1 = np.asarray([0.0, 1.0, 2.0, 3.0])
    s2 = np.asarray([0.0, 1.0, 2.0, 3.0])
    q = np.asarray([[0.05, 0.9], [0.25, 0.9], [0.2, 0.9], [0.7, 0.9]])
    alpha = 0.4
    records = _synthetic_records(q, np.zeros(4, dtype=np.int8), s1, s2, alpha=alpha)
    _, estimates, index = _one_stratum(records, num_agents=4, alpha=alpha)

    rewards = s1 / 3.0
    selected_q = q[:, 0]
    velocities = alpha * (rewards - selected_q)
    population_variance = lambda values: np.mean(values**2) - np.mean(values) ** 2
    covariance = np.mean(rewards * selected_q) - np.mean(rewards) * np.mean(selected_q)
    expected = alpha**2 * (
        population_variance(rewards)
        + population_variance(selected_q)
        - 2.0 * covariance
    )
    assert estimates.finite_bin_velocity_variance[index] == pytest.approx(expected, abs=1e-15)
    assert estimates.direct_velocity_variance[index] == pytest.approx(expected, abs=1e-15)
    assert not np.isclose(
        expected,
        alpha**2 * estimates.direct_reward_variance[index],
        rtol=0,
        atol=1e-4,
    )


def test_binning_conditions_on_both_q_coordinates_and_selected_action() -> None:
    q = np.asarray([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    actions = np.asarray([0, 0, 1, 1], dtype=np.int8)
    records = _synthetic_records(q, actions, np.ones(4), np.ones(4))
    bins = QBinSpec(np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.5, 1.0]))
    statistics = aggregate_variance_records(
        records, bins, num_agents=4, alpha=0.4
    )

    expected_nonzero = {
        (0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 0, 1, 1, 1),
    }
    actual_nonzero = set(map(tuple, np.argwhere(statistics.counts == 1)))
    assert actual_nonzero == expected_nonzero
    assert int(statistics.counts.sum()) == 4


def test_bin_boundaries_empty_single_and_out_of_range_are_explicit() -> None:
    q = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    records = _synthetic_records(q, np.asarray([0, 1, 1]), np.ones(3), np.ones(3))
    bins = QBinSpec(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0]))
    statistics = aggregate_variance_records(records, bins, num_agents=3, alpha=0.4)
    estimates = derive_variance_moments(statistics)

    assert statistics.counts[0, 0, 0, 0, 0] == 1
    assert statistics.counts[0, 0, 1, 1, 1] == 2
    single = (0, 0, 0, 0, 0)
    empty = (0, 0, 0, 0, 1)
    assert estimates.underpopulated[single]
    assert estimates.direct_velocity_variance[single] == pytest.approx(0.0)
    assert not estimates.has_observations[empty]
    assert np.isnan(estimates.direct_velocity_variance[empty])

    outside_q = q.copy()
    outside_q[0, 0] = -0.01
    outside = _synthetic_records(
        outside_q, np.asarray([0, 1, 1]), np.ones(3), np.ones(3)
    )
    with pytest.raises(ValueError, match="outside bin range"):
        aggregate_variance_records(outside, bins, num_agents=3, alpha=0.4)


def test_float32_bin_endpoints_use_effective_comparison_edges() -> None:
    bins = QBinSpec([-0.1, 0.5, 1.2], [-0.1, 0.5, 1.2])
    lower = np.float32(-0.1)
    interior = np.float32(0.5)
    upper = np.float32(1.2)
    q = np.asarray(
        [[lower, lower], [interior, interior], [upper, upper]],
        dtype=np.float32,
    )

    q_c_bin, q_d_bin = bins.assign(q)
    np.testing.assert_array_equal(q_c_bin, [0, 1, 1])
    np.testing.assert_array_equal(q_d_bin, [0, 1, 1])
    assert bins.q_c_edges[0] == np.float64(-0.1)
    assert bins.q_c_edges[-1] == np.float64(1.2)

    below = np.nextafter(lower, np.float32(-np.inf))
    above = np.nextafter(upper, np.float32(np.inf))
    with pytest.raises(ValueError, match="outside bin range"):
        bins.assign(np.asarray([[below, lower]], dtype=np.float32))
    with pytest.raises(ValueError, match="outside bin range"):
        bins.assign(np.asarray([[upper, above]], dtype=np.float32))


def test_edges_that_collapse_in_float32_are_rejected() -> None:
    adjacent_float64 = np.nextafter(np.float64(1.0), np.float64(2.0))
    bins = QBinSpec([0.0, 1.0, adjacent_float64, 2.0], [0.0, 2.0])

    with pytest.raises(ValueError, match=r"float32.*collapsed"):
        bins.assign(np.asarray([[0.5, 0.5]], dtype=np.float32))


def test_float64_bin_endpoints_and_nextafter_remain_exact() -> None:
    bins = QBinSpec([-0.1, 0.5, 1.2], [-0.1, 0.5, 1.2])
    q_c_bin, q_d_bin = bins.assign(
        np.asarray([[-0.1, -0.1], [0.5, 0.5], [1.2, 1.2]], dtype=np.float64)
    )
    np.testing.assert_array_equal(q_c_bin, [0, 1, 1])
    np.testing.assert_array_equal(q_d_bin, [0, 1, 1])

    with pytest.raises(ValueError, match="outside bin range"):
        bins.assign(
            np.asarray(
                [[np.nextafter(-0.1, -np.inf), -0.1]],
                dtype=np.float64,
            )
        )
    with pytest.raises(ValueError, match="outside bin range"):
        bins.assign(
            np.asarray(
                [[1.2, np.nextafter(1.2, np.inf)]],
                dtype=np.float64,
            )
        )


def test_n2_marks_distinct_covariance_undefined_without_division_by_zero() -> None:
    q = np.asarray([[0.2, 0.8], [0.4, 0.6]])
    s1 = np.asarray([0.25, 0.75])
    records = _synthetic_records(q, np.zeros(2, dtype=np.int8), s1, s1**2)
    _, estimates, index = _one_stratum(records, num_agents=2)

    assert not estimates.distinct_covariance_defined[index]
    assert np.isnan(estimates.m11[index])
    assert np.isnan(estimates.covariance[index])
    assert estimates.decomposed_reward_variance[index] == pytest.approx(
        estimates.direct_reward_variance[index], abs=1e-15
    )


def _batched_initial_state(run_count: int = 2):
    graph = complete_graph(4)
    one_q = jnp.asarray(
        [[0.2, 0.8], [0.7, 0.1], [0.4, 0.3], [0.1, 0.6]], dtype=Q_DTYPE
    )
    one_edges = jnp.asarray(
        [State.SH, State.PD, State.SH, State.PD, State.SH, State.PD], dtype=jnp.int8
    )
    states = ABMState(
        q_values=jnp.broadcast_to(one_q, (run_count, *one_q.shape)),
        edge_states=jnp.broadcast_to(one_edges, (run_count, *one_edges.shape)),
    )
    keys = jax.random.split(jax.random.PRNGKey(1234), run_count)
    return graph, states, keys


def test_batched_statistics_match_independently_processed_runs() -> None:
    graph, states, keys = _batched_initial_state()
    result = simulate_instrumented_batch(states, keys, graph, 0.4, 2.0, steps=3)
    bins = QBinSpec(np.asarray([-0.1, 1.2]), np.asarray([-0.1, 1.2]))
    batched = aggregate_variance_records(
        result.records, bins, num_agents=4, alpha=0.4
    )

    for run in range(2):
        component_records = jax.tree_util.tree_map(lambda value: value[run], result.records)
        component = aggregate_variance_records(
            component_records, bins, num_agents=4, alpha=0.4
        )
        for field in (
            "counts",
            "sum_s1",
            "sum_s2",
            "sum_distinct_products",
            "sum_reward",
            "sum_reward_squared",
            "sum_selected_q",
            "sum_selected_q_squared",
            "sum_reward_selected_q",
            "sum_velocity",
            "sum_velocity_squared",
        ):
            np.testing.assert_allclose(
                getattr(batched, field)[run], getattr(component, field)[0], rtol=0, atol=0
            )


def test_instrumentation_does_not_change_trajectory_actions_or_rng() -> None:
    graph, states, keys = _batched_initial_state(run_count=1)
    standard = simulate_batch(states, keys, graph, 0.4, 2.0, steps=5)
    instrumented = simulate_instrumented_batch_jit(
        states, keys, graph, 0.4, 2.0, steps=5
    )

    np.testing.assert_array_equal(standard.final_key, instrumented.final_key)
    np.testing.assert_array_equal(
        standard.final_state.edge_states, instrumented.final_state.edge_states
    )
    np.testing.assert_allclose(
        standard.final_state.q_values, instrumented.final_state.q_values, rtol=0, atol=0
    )
    for field in (
        "q_t",
        "action_probabilities_t",
        "actions_t",
        "rewards_t",
        "selected_velocities_t",
        "q_t_plus_1",
        "edge_state_proportions_t",
        "edge_state_proportions_t_plus_1",
    ):
        np.testing.assert_array_equal(
            getattr(standard.records, field), getattr(instrumented.records, field)
        )


def test_instrumented_records_are_source_timed_and_have_no_edge_history() -> None:
    graph, states, keys = _batched_initial_state(run_count=1)
    result = simulate_instrumented(
        ABMState(states.q_values[0], states.edge_states[0]),
        keys[0],
        graph,
        0.4,
        2.0,
        steps=3,
    )

    np.testing.assert_array_equal(result.records.q_t[0], states.q_values[0])
    first_actions = np.asarray(result.records.actions_t[0])
    expected_selected_q = np.asarray(states.q_values[0])[np.arange(4), first_actions]
    np.testing.assert_allclose(
        result.records.selected_q_t[0], expected_selected_q, rtol=0, atol=ATOL
    )
    np.testing.assert_allclose(
        result.records.q_t[1], result.records.q_t_plus_1[0], rtol=0, atol=ATOL
    )
    assert all("edge_payoff" not in field for field in result.records._fields)
    for leaf in jax.tree_util.tree_leaves(result.records):
        assert not (leaf.ndim >= 2 and leaf.shape[:2] == (3, graph.edge_count))

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.abm import (
    action_probabilities,
    complete_graph,
    initialize_continuous_paper,
    initialize_continuous_paper_batch,
    initialize_grid_matched,
    initialize_grid_matched_batch,
    sample_actions,
    sample_continuous_paper_q,
    sample_edge_states,
    sample_grid_matched_q,
)
from chu_pair.config import ABMConfig
from chu_pair.grids import QGrid
from chu_pair.initial_conditions import DiscreteQHistogram, seeded_legacy_histogram
from chu_pair.policies import boltzmann_probabilities


def off_diagonal_histogram() -> DiscreteQHistogram:
    grid = QGrid(q_min=-1.0, q_max=1.0, spacing=1.0)
    mass = np.zeros((3, 3), dtype=np.float64)
    mass[0, 2] = 0.25
    mass[2, 0] = 0.75
    return DiscreteQHistogram(grid=grid, mass=mass, seed=41, mode="test_off_diagonal")


def test_jax_policy_matches_authoritative_numpy_policy() -> None:
    q = np.asarray([[-0.4, 0.7], [0.2, -0.3], [1.0, 1.0]])
    expected = boltzmann_probabilities(q, tau=1.7)
    np.testing.assert_allclose(action_probabilities(jnp.asarray(q), 1.7), expected, rtol=1e-6)


def test_action_frequencies_match_two_nonuniform_boltzmann_policies() -> None:
    per_type = 10_000
    q_values = jnp.concatenate(
        (
            jnp.broadcast_to(jnp.asarray([-0.5, 0.5]), (per_type, 2)),
            jnp.broadcast_to(jnp.asarray([0.5, -0.5]), (per_type, 2)),
        ),
        axis=0,
    )
    actions = np.asarray(sample_actions(jax.random.PRNGKey(901), q_values, jnp.log(3.0)))

    assert abs(np.mean(actions[:per_type] == 0) - 0.25) < 0.02
    assert abs(np.mean(actions[per_type:] == 0) - 0.75) < 0.02
    np.testing.assert_array_equal(
        sample_actions(jax.random.PRNGKey(901), q_values, jnp.log(3.0)), actions
    )
    assert not np.array_equal(
        sample_actions(jax.random.PRNGKey(902), q_values, jnp.log(3.0)), actions
    )


def test_two_agent_joint_action_frequencies_match_independent_branch_weights() -> None:
    q_values = jnp.asarray([[-0.5, 0.5], [0.5, -0.5]])
    keys = jax.random.split(jax.random.PRNGKey(612), 20_000)
    actions = np.asarray(
        jax.vmap(lambda key: sample_actions(key, q_values, jnp.log(3.0)))(keys)
    )
    branch = 2 * actions[:, 0] + actions[:, 1]
    frequencies = np.bincount(branch, minlength=4) / branch.size

    # Codes are CC, CD, DC, DD for p_0(C)=1/4 and p_1(C)=3/4.
    np.testing.assert_allclose(
        frequencies,
        [3 / 16, 1 / 16, 9 / 16, 3 / 16],
        rtol=0.0,
        atol=0.025,
    )


def test_grid_matched_q_samples_follow_supplied_joint_histogram() -> None:
    samples = np.asarray(
        sample_grid_matched_q(
            jax.random.PRNGKey(17), off_diagonal_histogram(), 20_000
        )
    )
    first = np.all(samples == (-1.0, 1.0), axis=1)
    second = np.all(samples == (1.0, -1.0), axis=1)
    assert np.all(first | second)
    assert abs(first.mean() - 0.25) < 0.02
    assert abs(second.mean() - 0.75) < 0.02


def test_continuous_paper_q_has_expected_means_and_independent_coordinates() -> None:
    samples = np.asarray(sample_continuous_paper_q(jax.random.PRNGKey(112), 20_000))
    np.testing.assert_allclose(samples.mean(axis=0), [0.16, 0.94], rtol=0.0, atol=0.005)
    covariance = np.cov(samples.T, ddof=0)[0, 1]
    assert abs(covariance) < 1.2e-4
    assert np.all(samples > -0.1)
    assert np.all(samples < 1.2)


def test_initial_edge_states_are_half_half_at_large_sample() -> None:
    states = np.asarray(sample_edge_states(jax.random.PRNGKey(808), 20_000))
    assert set(np.unique(states)).issubset({0, 1})
    assert abs(np.mean(states == 0) - 0.5) < 0.02
    adjacent_pairs = 2 * states[0::2] + states[1::2]
    pair_frequencies = np.bincount(adjacent_pairs, minlength=4) / adjacent_pairs.size
    np.testing.assert_allclose(pair_frequencies, np.full(4, 0.25), rtol=0.0, atol=0.025)


def test_named_initialization_modes_return_shapes_and_separate_metadata() -> None:
    graph = complete_graph(8)
    histogram = off_diagonal_histogram()
    controlled = initialize_grid_matched(graph, histogram, abm_seed=55)
    continuous = initialize_continuous_paper(graph, abm_seed=55)

    assert controlled.state.q_values.shape == continuous.state.q_values.shape == (8, 2)
    assert controlled.state.edge_states.shape == continuous.state.edge_states.shape == (28,)
    assert controlled.metadata.mode == "grid_matched"
    assert controlled.metadata.histogram_seed == 41
    assert controlled.metadata.abm_seed == 55
    assert continuous.metadata.mode == "continuous_paper"
    assert continuous.metadata.histogram_seed is None
    np.testing.assert_array_equal(
        initialize_grid_matched(graph, histogram, abm_seed=55).state.q_values,
        controlled.state.q_values,
    )


def test_batched_initialization_modes_have_independent_run_axes() -> None:
    graph = complete_graph(5)
    histogram = off_diagonal_histogram()
    controlled = initialize_grid_matched_batch(
        graph, histogram, abm_seed=81, num_runs=3
    )
    continuous = initialize_continuous_paper_batch(graph, abm_seed=82, num_runs=3)

    assert controlled.state.q_values.shape == continuous.state.q_values.shape == (3, 5, 2)
    assert controlled.state.edge_states.shape == continuous.state.edge_states.shape == (3, 10)
    assert controlled.simulation_key.shape == continuous.simulation_key.shape == (3, 2)
    assert controlled.metadata.num_runs == continuous.metadata.num_runs == 3
    assert not np.array_equal(controlled.simulation_key[0], controlled.simulation_key[1])


def test_abm_seeds_must_fit_the_jax_prng_key_without_aliasing() -> None:
    with pytest.raises(ValueError, match="lie in"):
        ABMConfig(abm_seed=2**32)
    with pytest.raises(ValueError, match="lie in"):
        initialize_continuous_paper(complete_graph(2), abm_seed=2**32)


def test_histogram_and_abm_seeds_affect_only_their_own_stages() -> None:
    graph = complete_graph(32)
    grid = QGrid(q_min=-0.1, q_max=1.2, spacing=0.1)
    histogram_a = seeded_legacy_histogram(grid, seed=101, sample_count=500)
    histogram_b = seeded_legacy_histogram(grid, seed=202, sample_count=500)
    assert not np.array_equal(histogram_a.mass, histogram_b.mass)

    same_abm_a = initialize_grid_matched(graph, histogram_a, abm_seed=77)
    same_abm_b = initialize_grid_matched(graph, histogram_b, abm_seed=77)
    np.testing.assert_array_equal(same_abm_a.state.edge_states, same_abm_b.state.edge_states)
    np.testing.assert_array_equal(same_abm_a.simulation_key, same_abm_b.simulation_key)
    assert not np.array_equal(same_abm_a.state.q_values, same_abm_b.state.q_values)

    changed_abm = initialize_grid_matched(graph, histogram_a, abm_seed=78)
    np.testing.assert_array_equal(histogram_a.mass, histogram_a.mass.copy())
    assert not np.array_equal(same_abm_a.simulation_key, changed_abm.simulation_key)
    assert not np.array_equal(same_abm_a.state.edge_states, changed_abm.state.edge_states)


def test_more_repetitions_reduce_monte_carlo_standard_error() -> None:
    keys = jax.random.split(jax.random.PRNGKey(5150), 64)

    def estimates(sample_count):
        q_values = jnp.broadcast_to(
            jnp.asarray([jnp.log(0.3 / 0.7), 0.0]),
            (sample_count, 2),
        )
        return jax.vmap(
            lambda key: jnp.mean(sample_actions(key, q_values, 1.0) == 0)
        )(keys)

    small = np.asarray(estimates(256))
    large = np.asarray(estimates(1024))
    ratio = small.std(ddof=1) / large.std(ddof=1)
    assert 1.4 < ratio < 2.8

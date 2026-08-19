from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.abm import uncertainty
from chu_pair.abm import (
    NamedBinScheme,
    QBinSpec,
    aggregate_variance_records,
    anchor_bin_index,
    assert_child_reconstructs_parent,
    bootstrap_run_weights,
    cluster_bootstrap_intervals,
    complete_graph,
    initialize_continuous_paper_batch,
    pool_sufficient_statistics,
    pooled_point_estimands,
    simulate_instrumented_batch_jit,
    validate_nested_schemes,
)
from chu_pair.abm.statistics import derive_variance_moments


def _statistics_from_run_values(run_values, *, dtype=np.float64, alpha=0.4):
    runs = len(run_values)
    agents = max(2, max(len(values) for values in run_values))
    q = np.zeros((runs, 1, agents, 2), dtype=dtype)
    actions = np.ones((runs, 1, agents), dtype=np.int8)
    s1 = np.zeros((runs, 1, agents), dtype=dtype)
    for run, values in enumerate(run_values):
        count = len(values)
        actions[run, 0, :count] = 0
        s1[run, 0, :count] = values
    selected_q = np.take_along_axis(q, actions[..., None], axis=-1)[..., 0]
    rewards = s1
    records = SimpleNamespace(
        q_t=q,
        actions_t=actions,
        selected_q_t=selected_q,
        rewards_t=rewards,
        selected_velocities_t=alpha * (rewards - selected_q),
        payoff_sums_t=s1,
        payoff_square_sums_t=s1 * s1,
    )
    return aggregate_variance_records(
        records,
        QBinSpec([-1.0, 1.0], [-1.0, 1.0]),
        num_agents=2,
        alpha=alpha,
        min_count=1,
    )


def test_pooled_sums_are_observation_weighted_not_mean_run_variances() -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0]])
    count, contributing, point = pooled_point_estimands(statistics)
    index = (0, 0, 0, 0)

    assert count[index] == 3
    assert contributing[index] == 2
    assert point["direct_reward_variance"][index] == pytest.approx(56.0 / 3.0)
    assert point["direct_reward_variance"][index] != pytest.approx((1.0 + 0.0) / 2.0)
    pooled = pool_sufficient_statistics(statistics)
    assert pooled.counts.shape[0] == 1
    assert pooled.sum_reward[(0, *index)] == pytest.approx(12.0)


def test_direct_pooling_matches_explicit_ones_without_allocating_run_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0], [4.0]])
    weighted = pool_sufficient_statistics(
        statistics,
        np.ones(3, dtype=np.int64),
    )

    def forbidden_ones(*args, **kwargs):
        raise AssertionError("unweighted pooling allocated a run-weight vector")

    monkeypatch.setattr(uncertainty.np, "ones", forbidden_ones)
    direct = pool_sufficient_statistics(statistics)

    np.testing.assert_array_equal(direct.counts, weighted.counts)
    assert direct.counts.dtype == np.int64
    for name in uncertainty.SUFFICIENT_SUM_FIELDS:
        np.testing.assert_array_equal(getattr(direct, name), getattr(weighted, name))
        assert getattr(direct, name).dtype == np.float64


def test_fixed_cluster_weights_give_hand_calculated_percentile_interval() -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0]])
    weights = np.asarray([[2, 0], [0, 2], [1, 1]], dtype=np.int32)
    summary = cluster_bootstrap_intervals(
        statistics,
        weights,
        confidence_level=0.5,
        stratum_chunk_size=1,
    )
    index = (0, 0, 0, 0)

    # Replicate variances are [1, 0, 56/3]. Linear 25/75 percentiles are
    # 0.5 and (1 + 56/3)/2 = 59/6.
    assert summary.lower["direct_reward_variance"][index] == pytest.approx(0.5)
    assert summary.upper["direct_reward_variance"][index] == pytest.approx(59.0 / 6.0)
    assert summary.valid_replicates["direct_reward_variance"][index] == 3
    assert summary.interval_valid["direct_reward_variance"][index]


def test_run_multiplicities_apply_to_complete_correlated_clusters() -> None:
    statistics = _statistics_from_run_values([[0.0, 0.0], [10.0, 10.0]])
    weights = np.asarray([[2, 0], [0, 2]], dtype=np.int32)
    summary = cluster_bootstrap_intervals(
        statistics,
        weights,
        confidence_level=0.5,
        stratum_chunk_size=2,
    )
    index = (0, 0, 0, 0)

    assert summary.point["direct_reward_variance"][index] == pytest.approx(25.0)
    assert summary.lower["direct_reward_variance"][index] == pytest.approx(0.0)
    assert summary.upper["direct_reward_variance"][index] == pytest.approx(0.0)
    # An invalid observation-level resample can retain two zeros and two tens,
    # giving variance 25 instead of either whole-run replicate's variance 0.
    assert np.var([0.0, 0.0, 10.0, 10.0]) == pytest.approx(25.0)


def test_empty_replicates_and_few_contributing_runs_invalidate_intervals() -> None:
    statistics = _statistics_from_run_values([[2.0], [], []])
    weights = np.asarray(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1]],
        dtype=np.int32,
    )
    summary = cluster_bootstrap_intervals(
        statistics,
        weights,
        confidence_level=0.95,
        stratum_chunk_size=1,
    )
    index = (0, 0, 0, 0)

    assert summary.contributing_runs[index] == 1
    assert summary.valid_replicates["mu"][index] == 2
    assert summary.invalid_replicates["mu"][index] == 2
    assert not summary.interval_valid["mu"][index]
    assert np.isnan(summary.lower["mu"][index])
    assert summary.valid_replicates["m11"][index] == 0


def test_bootstrap_seed_is_reproducible_and_does_not_change_point_estimates() -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0], [4.0]])
    weights_a = bootstrap_run_weights(3, 32, 91)
    weights_b = bootstrap_run_weights(3, 32, 91)
    weights_c = bootstrap_run_weights(3, 32, 92)
    np.testing.assert_array_equal(weights_a, weights_b)
    assert not np.array_equal(weights_a, weights_c)
    assert np.all(weights_a.sum(axis=1) == 3)

    summary_a = cluster_bootstrap_intervals(
        statistics, weights_a, confidence_level=0.9, stratum_chunk_size=1
    )
    summary_c = cluster_bootstrap_intervals(
        statistics, weights_c, confidence_level=0.9, stratum_chunk_size=1
    )
    for name in summary_a.point:
        np.testing.assert_allclose(summary_a.point[name], summary_c.point[name], equal_nan=True)


def test_bootstrap_chunk_sizes_are_identical_and_do_not_change_statistics() -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0], [4.0]])
    weights = bootstrap_run_weights(3, 25, 7)
    one = cluster_bootstrap_intervals(
        statistics, weights, confidence_level=0.8, stratum_chunk_size=1
    )
    all_cells = cluster_bootstrap_intervals(
        statistics, weights, confidence_level=0.8, stratum_chunk_size=2
    )
    for collection in ("point", "lower", "upper"):
        for name in one.point:
            np.testing.assert_allclose(
                getattr(one, collection)[name],
                getattr(all_cells, collection)[name],
                rtol=0,
                atol=0,
                equal_nan=True,
            )


def test_chunking_never_derives_more_than_the_configured_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0], [4.0]])
    weights = bootstrap_run_weights(3, 7, 3)
    observed_widths = []
    original = uncertainty._estimands_from_sums

    def bounded(counts, sums, **kwargs):
        observed_widths.append(counts.shape[1])
        return original(counts, sums, **kwargs)

    monkeypatch.setattr(uncertainty, "_estimands_from_sums", bounded)
    cluster_bootstrap_intervals(
        statistics, weights, confidence_level=0.8, stratum_chunk_size=1
    )

    assert observed_widths
    assert max(observed_widths) == 1
    assert all(width <= 1 for width in observed_widths)


def test_phase3a_point_estimates_are_recovered_after_run_pooling() -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0], [4.0]])
    pooled = pool_sufficient_statistics(statistics)
    phase3a = derive_variance_moments(pooled)
    _, _, phase3b = pooled_point_estimands(statistics)

    for name in (
        "mu",
        "m2",
        "m11",
        "sigma2",
        "covariance",
        "direct_reward_variance",
        "decomposed_reward_variance",
        "direct_velocity_variance",
        "finite_bin_velocity_variance",
        "selected_q_variance",
        "reward_selected_q_covariance",
    ):
        np.testing.assert_allclose(
            phase3b[name], getattr(phase3a, name)[0], rtol=0, atol=0, equal_nan=True
        )


def test_bootstrap_does_not_mutate_per_run_statistics() -> None:
    statistics = _statistics_from_run_values([[0.0, 2.0], [10.0], [4.0]])
    before = {
        name: np.array(getattr(statistics, name), copy=True)
        for name in ("counts", *uncertainty.SUFFICIENT_SUM_FIELDS)
    }
    weights = bootstrap_run_weights(3, 12, 45)
    weights_before = weights.copy()
    cluster_bootstrap_intervals(
        statistics, weights, confidence_level=0.8, stratum_chunk_size=1
    )

    np.testing.assert_array_equal(weights, weights_before)
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(statistics, name), expected)


def test_float32_and_float64_bootstrap_results_agree() -> None:
    weights = bootstrap_run_weights(3, 20, 5)
    f32 = cluster_bootstrap_intervals(
        _statistics_from_run_values([[0.1, 0.2], [0.3], [0.4]], dtype=np.float32),
        weights,
        confidence_level=0.8,
        stratum_chunk_size=1,
    )
    f64 = cluster_bootstrap_intervals(
        _statistics_from_run_values([[0.1, 0.2], [0.3], [0.4]], dtype=np.float64),
        weights,
        confidence_level=0.8,
        stratum_chunk_size=1,
    )
    for name in f32.point:
        np.testing.assert_allclose(f32.point[name], f64.point[name], atol=2e-8, rtol=1e-6, equal_nan=True)


def _nested_schemes(dtype=np.float32):
    schemes = [
        NamedBinScheme("coarse", QBinSpec([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])),
        NamedBinScheme(
            "fine",
            QBinSpec(
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ),
        ),
    ]
    validate_nested_schemes(schemes, dtype)
    return schemes


def test_nested_schemes_validate_configured_and_effective_edges() -> None:
    _nested_schemes(np.float32)
    _nested_schemes(np.float64)
    with pytest.raises(ValueError, match="not nested"):
        validate_nested_schemes(
            [
                NamedBinScheme("coarse", QBinSpec([0, 0.5, 1], [0, 0.5, 1])),
                NamedBinScheme("bad", QBinSpec([0, 0.3, 0.6, 1], [0, 0.3, 0.6, 1])),
            ],
            np.float32,
        )
    with pytest.raises(ValueError, match="different"):
        validate_nested_schemes(
            [
                NamedBinScheme("coarse", QBinSpec([0, 0.5, 1], [0, 0.5, 1])),
                NamedBinScheme("bad", QBinSpec([-1, 0, 0.5, 1], [0, 0.25, 0.5, 1])),
            ],
            np.float32,
        )
    adjacent = np.nextafter(np.float64(0.5), np.float64(1.0))
    with pytest.raises(ValueError, match="collapsed"):
        validate_nested_schemes(
            [
                NamedBinScheme("coarse", QBinSpec([0, 0.5, 1], [0, 0.5, 1])),
                NamedBinScheme(
                    "collapsed",
                    QBinSpec([0, 0.5, adjacent, 0.75, 1], [0, 0.25, 0.5, 0.75, 1]),
                ),
            ],
            np.float32,
        )


def _refinement_records(dtype=np.float64):
    q = np.asarray(
        [[[[0.1, 0.1], [0.3, 0.3], [0.6, 0.6], [0.9, 0.9]]]],
        dtype=dtype,
    )
    actions = np.asarray([[[0, 0, 1, 1]]], dtype=np.int8)
    s1 = np.asarray([[[0.2, 0.4, 0.6, 0.8]]], dtype=dtype)
    selected_q = np.take_along_axis(q, actions[..., None], axis=-1)[..., 0]
    rewards = s1 / 3.0
    return SimpleNamespace(
        q_t=q,
        actions_t=actions,
        selected_q_t=selected_q,
        rewards_t=rewards,
        selected_velocities_t=0.4 * (rewards - selected_q),
        payoff_sums_t=s1,
        payoff_square_sums_t=s1 * s1,
    )


@pytest.mark.parametrize("observation_dtype", [np.float32, np.float64])
def test_child_bins_reconstruct_parent_counts_and_all_sums(
    observation_dtype,
) -> None:
    coarse, fine = _nested_schemes(observation_dtype)
    records = _refinement_records(observation_dtype)
    parent = aggregate_variance_records(records, coarse.bins, num_agents=4, alpha=0.4)
    child = aggregate_variance_records(records, fine.bins, num_agents=4, alpha=0.4)
    diagnostic = assert_child_reconstructs_parent(parent, child)

    assert diagnostic["maximum_parent_observations"] >= 1
    for field in diagnostic["fields"].values():
        assert field["aggregation_dtype"] == "float64"
        assert field["maximum_absolute_difference"] <= field[
            "maximum_allowed_roundoff"
        ]


def test_float32_n2_represented_terms_have_valid_nonzero_distinct_bound() -> None:
    dtype = np.float32
    q = np.asarray([[[[0.15, 0.70], [0.75, 0.20]]]], dtype=dtype)
    actions = np.asarray([[[1, 0]]], dtype=np.int8)
    s1 = np.asarray([[[1.2, -0.1]]], dtype=dtype)
    s2 = s1 * s1
    selected_q = np.take_along_axis(q, actions[..., None], axis=-1)[..., 0]
    rewards = s1
    velocities = np.asarray(0.4, dtype=dtype) * (rewards - selected_q)
    records = SimpleNamespace(
        q_t=q,
        actions_t=actions,
        selected_q_t=selected_q,
        rewards_t=rewards,
        selected_velocities_t=velocities,
        payoff_sums_t=s1,
        payoff_square_sums_t=s2,
    )
    parent = aggregate_variance_records(
        records,
        QBinSpec([-0.2, 1.0], [-0.2, 1.0]),
        num_agents=2,
        alpha=0.4,
        min_count=1,
    )
    child = aggregate_variance_records(
        records,
        QBinSpec([-0.2, 0.5, 1.0], [-0.2, 0.5, 1.0]),
        num_agents=2,
        alpha=0.4,
        min_count=1,
    )
    bounds = uncertainty._absolute_term_bounds(parent, np.dtype(np.float64))
    occupied = parent.counts == 1

    assert float(s1[0, 0, 0]) == 1.2000000476837158
    distinct_values = parent.sum_distinct_products[occupied]
    assert np.max(distinct_values) == pytest.approx(
        5.7220461258111754e-08, rel=0, abs=1e-24
    )
    assert bounds["sum_s1"] >= np.max(np.abs(parent.sum_s1[occupied]))
    assert bounds["sum_distinct_products"] >= np.max(np.abs(distinct_values))
    assert bounds["sum_distinct_products"] > 0.0
    for name in uncertainty.SUFFICIENT_SUM_FIELDS:
        assert bounds[name] >= np.max(np.abs(getattr(parent, name)[occupied]))

    diagnostic = assert_child_reconstructs_parent(parent, child)
    assert diagnostic["observation_dtype"] == "float32"
    assert diagnostic["fields"]["sum_distinct_products"][
        "maximum_allowed_roundoff"
    ] > 0.0


def test_reconstruction_rejects_sum_and_count_corruption() -> None:
    coarse, fine = _nested_schemes(np.float64)
    records = _refinement_records(np.float64)
    parent = aggregate_variance_records(records, coarse.bins, num_agents=4, alpha=0.4)
    child = aggregate_variance_records(records, fine.bins, num_agents=4, alpha=0.4)

    corrupted_sum = child.sum_distinct_products.copy()
    corrupted_sum.flat[np.flatnonzero(child.counts)[0]] += 1e-6
    with pytest.raises(AssertionError, match="sum_distinct_products"):
        assert_child_reconstructs_parent(
            parent,
            replace(child, sum_distinct_products=corrupted_sum),
        )

    corrupted_count = child.counts.copy()
    corrupted_count.flat[np.flatnonzero(child.counts)[0]] += 1
    with pytest.raises(AssertionError):
        assert_child_reconstructs_parent(
            parent,
            replace(child, counts=corrupted_count),
        )

    occupied_index = tuple(np.argwhere(child.counts > 0)[0])
    omitted_sum = child.sum_s1.copy()
    omitted_sum[occupied_index] = 0.0
    with pytest.raises(AssertionError, match="sum_s1"):
        assert_child_reconstructs_parent(parent, replace(child, sum_s1=omitted_sum))

    duplicated_sum = child.sum_s1.copy()
    duplicated_sum[occupied_index] *= 2.0
    with pytest.raises(AssertionError, match="sum_s1"):
        assert_child_reconstructs_parent(
            parent, replace(child, sum_s1=duplicated_sum)
        )


def test_reconstruction_bound_handles_empty_singleton_near_zero_and_cancellation() -> None:
    tiny = 1.0e-200
    parent_values = np.zeros((1, 2, 2), dtype=np.float64)
    parent_counts = np.zeros((1, 2, 2), dtype=np.int64)
    child_values = np.zeros((1, 2, 2, 2, 2), dtype=np.float64)
    child_counts = np.zeros((1, 2, 2, 2, 2), dtype=np.int64)

    child_values[0, 0, 0, 0, 0] = tiny
    child_values[0, 0, 0, 1, 0] = -tiny
    child_counts[0, 0, 0, 0, 0] = 1
    child_counts[0, 0, 0, 1, 0] = 1
    parent_counts[0, 0, 0] = 2
    child_values[0, 0, 1, 1, 1] = np.finfo(np.float64).smallest_subnormal
    child_counts[0, 0, 1, 1, 1] = 1
    parent_values[0, 0, 1] = np.finfo(np.float64).smallest_subnormal
    parent_counts[0, 0, 1] = 1

    reconstructed = child_values.sum(axis=(2, 3))
    allowed = uncertainty._reconstruction_bound(
        parent_values,
        child_values,
        parent_counts,
        child_counts,
        term_bound=1.0,
        dtype=np.dtype(np.float64),
    )

    assert np.all(np.abs(reconstructed - parent_values) <= allowed)
    assert allowed[0, 0, 0] > 0.0
    assert allowed[0, 0, 1] > 0.0
    assert allowed[0, 1, 0] > 0.0
    assert allowed[0, 1, 1] > 0.0


def test_gamma_rejects_invalid_machine_epsilon_denominator() -> None:
    epsilon = np.finfo(np.float32).eps
    with pytest.raises(ValueError, match=r"k\*epsilon >= 1"):
        uncertainty._gamma(math.ceil(1.0 / epsilon), np.dtype(np.float32))

    statistics = _statistics_from_run_values([[0.1]], dtype=np.float32)
    with pytest.raises(ValueError, match="alpha is not finite in float32"):
        uncertainty._absolute_term_bounds(
            replace(statistics, alpha=1.0e300),
            np.dtype(np.float64),
        )


def _actual_abm_reconstruction(dtype):
    graph = complete_graph(32)
    initialization = initialize_continuous_paper_batch(
        graph,
        abm_seed=19,
        num_runs=2,
        dtype=dtype,
    )
    result = simulate_instrumented_batch_jit(
        initialization.state,
        initialization.simulation_key,
        graph,
        0.4,
        2.0,
        steps=4,
    )
    coarse = QBinSpec([-0.1, 0.55, 1.2], [-0.1, 0.55, 1.2])
    fine = QBinSpec(
        [-0.1, 0.2, 0.55, 0.85, 1.2],
        [-0.1, 0.2, 0.55, 0.85, 1.2],
    )
    parent = aggregate_variance_records(
        result.records, coarse, num_agents=32, alpha=0.4, min_count=1
    )
    child = aggregate_variance_records(
        result.records, fine, num_agents=32, alpha=0.4, min_count=1
    )
    return assert_child_reconstructs_parent(parent, child)


def test_actual_float32_abm_reconstruction_uses_represented_value_bounds() -> None:
    diagnostic = _actual_abm_reconstruction(jnp.float32)
    assert diagnostic["observation_dtype"] == "float32"
    for field in diagnostic["fields"].values():
        assert field["maximum_absolute_difference"] <= field[
            "maximum_allowed_roundoff"
        ]


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="requires a fresh CPU+x64 process",
)
def test_actual_x64_abm_reconstruction_uses_scale_aware_roundoff_bound() -> None:
    diagnostic = _actual_abm_reconstruction(jnp.float64)
    distinct = diagnostic["fields"]["sum_distinct_products"]

    assert distinct["maximum_absolute_difference"] == pytest.approx(
        1.8189894035458565e-12,
        rel=0,
        abs=1e-24,
    )
    assert distinct["maximum_allowed_roundoff"] > distinct[
        "maximum_absolute_difference"
    ]
    assert distinct["maximum_allowed_roundoff"] < 1e-8


def test_anchor_mapping_uses_upper_bin_and_inclusive_final_endpoint() -> None:
    coarse = _nested_schemes()[0].bins
    assert anchor_bin_index(coarse, (0.5, 0.5), np.float32) == (1, 1)
    assert anchor_bin_index(coarse, (1.0, 1.0), np.float32) == (1, 1)
    assert anchor_bin_index(coarse, (0.0, 0.0), np.float32) == (0, 0)

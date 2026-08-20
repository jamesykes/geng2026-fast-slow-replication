from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.abm import BinnedSufficientStatistics, QBinSpec, bootstrap_run_weights
from chu_pair.grids import QGrid
from chu_pair.pair_density import build_jax_pair_grid, pair_point_sufficient_jax
from chu_pair.velocity_variance import (
    PairPointSufficientStatistics,
    aggregate_pair_points,
    bootstrap_four_way_intervals,
    coarsen_abm_sufficient,
    coarsen_pair_sufficient,
    compare_four_way,
    derive_pair_binned_moments,
)


def _two_point_pair_statistics() -> PairPointSufficientStatistics:
    # Selected C masses (.25,.75) deliberately differ from focal masses (.5,.5),
    # representing within-bin policy reweighting. Exact-Q means are 0 and 2.
    selected = np.array([[[0.25, 0.0], [0.75, 0.0]]])
    mean = np.array([[[0.0, 0.0], [2.0, 0.0]]])
    second = np.array([[[1.0, 0.0], [5.0, 0.0]]])
    q = np.array([[0.0, 0.0], [1.0, 0.0]])
    q_selected = np.broadcast_to(q[None, :, :], selected.shape)
    return PairPointSufficientStatistics(
        source_times=np.array([0]),
        q_points=q,
        observation_dtype="float64",
        focal_mass=np.array([[0.5, 0.5]]),
        selected_mass=selected,
        sum_y=selected * mean,
        sum_y2=selected * second,
        sum_distinct_y=selected * mean * mean,
        sum_q=selected * q_selected,
        sum_q2=selected * q_selected * q_selected,
        sum_y_q=selected * mean * q_selected,
    )


def _abm_statistics(num_agents: int = 3, runs: int = 1) -> BinnedSufficientStatistics:
    bins = QBinSpec([-0.1, 1.1], [-0.1, 0.1])
    shape = (runs, 1, 1, 1, 2)
    fields = {name: np.zeros(shape) for name in (
        "sum_s1", "sum_s2", "sum_distinct_products", "sum_reward",
        "sum_reward_squared", "sum_selected_q", "sum_selected_q_squared",
        "sum_reward_selected_q", "sum_velocity", "sum_velocity_squared",
    )}
    counts = np.zeros(shape, dtype=np.int64)
    for run in range(runs):
        counts[run, 0, 0, 0, 0] = 2
        if num_agents == 3:
            fields["sum_s1"][run, 0, 0, 0, 0] = 4.0
            fields["sum_s2"][run, 0, 0, 0, 0] = 8.0
            fields["sum_distinct_products"][run, 0, 0, 0, 0] = 6.0
        else:
            fields["sum_s1"][run, 0, 0, 0, 0] = 2.0
            fields["sum_s2"][run, 0, 0, 0, 0] = 4.0
        fields["sum_reward"][run, 0, 0, 0, 0] = 2.0
        fields["sum_reward_squared"][run, 0, 0, 0, 0] = 3.5
        fields["sum_selected_q"][run, 0, 0, 0, 0] = 1.0
        fields["sum_selected_q_squared"][run, 0, 0, 0, 0] = 1.0
        fields["sum_reward_selected_q"][run, 0, 0, 0, 0] = 1.2
        target = 0.128 if num_agents == 3 else 0.2
        fields["sum_velocity_squared"][run, 0, 0, 0, 0] = 2 * target
    return BinnedSufficientStatistics(
        bins=bins,
        num_agents=num_agents,
        alpha=0.4,
        min_count=3,
        observation_dtype="float64",
        effective_q_c_edges=bins.q_c_edges,
        effective_q_d_edges=bins.q_d_edges,
        counts=counts,
        **fields,
    )


def _custom_abm_statistics(*, m2: float, m11: float, q2: float, reward_q: float, velocity_variance: float) -> BinnedSufficientStatistics:
    statistics = _abm_statistics()
    fields = {
        name: np.array(getattr(statistics, name), copy=True)
        for name in (
            "counts", "sum_s1", "sum_s2", "sum_distinct_products",
            "sum_reward", "sum_reward_squared", "sum_selected_q",
            "sum_selected_q_squared", "sum_reward_selected_q", "sum_velocity",
            "sum_velocity_squared",
        )
    }
    index = (0, 0, 0, 0, 0)
    fields["sum_s2"][index] = 4 * m2
    fields["sum_distinct_products"][index] = 4 * m11
    reward_variance = m2 / 2 + m11 / 2 - 1.0
    fields["sum_reward_squared"][index] = 2 * (reward_variance + 1.0)
    fields["sum_selected_q_squared"][index] = 2 * q2
    fields["sum_reward_selected_q"][index] = 2 * reward_q
    fields["sum_velocity_squared"][index] = 2 * velocity_variance
    return BinnedSufficientStatistics(
        bins=statistics.bins,
        num_agents=statistics.num_agents,
        alpha=statistics.alpha,
        min_count=statistics.min_count,
        observation_dtype=statistics.observation_dtype,
        effective_q_c_edges=statistics.effective_q_c_edges,
        effective_q_d_edges=statistics.effective_q_d_edges,
        **fields,
    )


def test_finite_bin_pair_raw_moments_preserve_policy_reweighting_and_q_mixing() -> None:
    sufficient = aggregate_pair_points(
        _two_point_pair_statistics(), QBinSpec([-0.1, 1.1], [-0.1, 0.1])
    )
    moments = derive_pair_binned_moments(sufficient, num_agents=3, alpha=0.4)
    index = (0, 0, 0, 0)

    assert moments.selected_mass[index] == pytest.approx(1.0)
    assert moments.mu[index] == pytest.approx(1.5)
    assert moments.m2[index] == pytest.approx(4.0)
    assert moments.m11[index] == pytest.approx(3.0)
    assert moments.sigma2[index] == pytest.approx(1.75)
    # Exact-Q distinct opponents are independent, yet mixing two Q points gives c=.75.
    assert moments.covariance[index] == pytest.approx(0.75)
    assert moments.mean_q[index] == pytest.approx(0.75)
    assert moments.q_variance[index] == pytest.approx(0.1875)
    assert moments.reward_q_covariance[index] == pytest.approx(0.375)
    assert moments.mean_local_sigma2[index] == pytest.approx(1.0)
    assert moments.velocity_variance[index] == pytest.approx(0.11)


def test_pair_point_payoffs_use_old_state_and_focal_row_orientation() -> None:
    grid = build_jax_pair_grid(QGrid(0.0, 1.0, 1.0), jnp.float32)
    mass = jnp.zeros((2, 4, 4), dtype=jnp.float32).at[1, 0, 0].set(1.0)
    sufficient = pair_point_sufficient_jax(mass, grid, 0.0)

    # PD row payoffs: own C against (C,D)=(1,-.1); own D=(1.2,0).
    np.testing.assert_allclose(np.asarray(sufficient.sum_y[0]), [0.225, 0.3])
    np.testing.assert_allclose(np.asarray(sufficient.sum_y2[0]), [0.2525, 0.36])
    np.testing.assert_allclose(np.asarray(sufficient.selected_mass[0]), [0.5, 0.5])


def test_four_way_exact_reconstruction_pair_and_hybrid_formulas() -> None:
    pair = derive_pair_binned_moments(
        aggregate_pair_points(
            _two_point_pair_statistics(), QBinSpec([-0.1, 1.1], [-0.1, 0.1])
        ),
        num_agents=3,
        alpha=0.4,
    )
    comparison = compare_four_way(_abm_statistics(), pair)
    index = (0, 0, 0, 0)

    assert comparison.abm_sigma2[index] == pytest.approx(1.0)
    assert comparison.abm_covariance[index] == pytest.approx(0.5)
    assert comparison.direct_abm_velocity_variance[index] == pytest.approx(0.128)
    assert comparison.reconstructed_abm_velocity_variance[index] == pytest.approx(0.128)
    assert comparison.direct_minus_reconstructed[index] == pytest.approx(0.0)
    assert comparison.pair_velocity_variance[index] == pytest.approx(0.11)
    # Hybrid replaces pair c=.75 by ABM c=.5, retaining pair sigma/Q terms.
    assert comparison.hybrid_velocity_variance[index] == pytest.approx(0.09)
    assert comparison.pair_minus_direct[index] == pytest.approx(-0.018)
    assert comparison.hybrid_minus_direct[index] == pytest.approx(-0.038)


def test_n_equals_one_omits_cross_opponent_covariance() -> None:
    pair = derive_pair_binned_moments(
        aggregate_pair_points(
            _two_point_pair_statistics(), QBinSpec([-0.1, 1.1], [-0.1, 0.1])
        ),
        num_agents=2,
        alpha=0.4,
    )
    comparison = compare_four_way(_abm_statistics(num_agents=2), pair)
    index = (0, 0, 0, 0)

    assert np.isnan(comparison.abm_covariance[index])
    assert comparison.hybrid_velocity_variance[index] == pytest.approx(
        comparison.pair_velocity_variance[index]
    )
    assert comparison.hybrid_valid[index]


def test_empty_pair_and_abm_strata_remain_explicit() -> None:
    pair = derive_pair_binned_moments(
        aggregate_pair_points(
            _two_point_pair_statistics(), QBinSpec([-0.1, 0.5, 1.1], [-0.1, 0.1])
        ),
        num_agents=3,
        alpha=0.4,
    )
    abm = _abm_statistics()
    comparison = compare_four_way(abm, derive_pair_binned_moments(
        aggregate_pair_points(
            _two_point_pair_statistics(), QBinSpec([-0.1, 1.1], [-0.1, 0.1])
        ), num_agents=3, alpha=0.4
    ))

    assert not pair.has_selected_mass[0, 0, 0, 1]
    assert np.isnan(pair.mu[0, 0, 0, 1])
    assert not comparison.has_abm_observations[0, 0, 0, 1]
    assert np.isnan(comparison.direct_abm_velocity_variance[0, 0, 0, 1])


def test_nested_pair_reconstruction_adds_raw_sums_before_moments() -> None:
    point = _two_point_pair_statistics()
    fine = aggregate_pair_points(
        point, QBinSpec([-0.1, 0.5, 1.1], [-0.1, 0.0, 0.1])
    )
    coarse_bins = QBinSpec([-0.1, 1.1], [-0.1, 0.1])
    reconstructed = coarsen_pair_sufficient(fine, coarse_bins)
    direct = aggregate_pair_points(point, coarse_bins)

    np.testing.assert_array_equal(reconstructed.focal_mass, direct.focal_mass)
    for name in (
        "selected_mass", "sum_y", "sum_y2", "sum_distinct_y",
        "sum_q", "sum_q2", "sum_y_q",
    ):
        np.testing.assert_array_equal(getattr(reconstructed, name), getattr(direct, name))


def test_nested_abm_reconstruction_from_finest_sums_is_exact() -> None:
    coarse = _abm_statistics()
    fine_bins = QBinSpec([-0.1, 0.5, 1.1], [-0.1, 0.1])
    fields = {}
    for name in (
        "counts", "sum_s1", "sum_s2", "sum_distinct_products", "sum_reward",
        "sum_reward_squared", "sum_selected_q", "sum_selected_q_squared",
        "sum_reward_selected_q", "sum_velocity", "sum_velocity_squared",
    ):
        source = getattr(coarse, name)
        target = np.zeros((1, 1, 2, 1, 2), dtype=source.dtype)
        if name == "counts":
            target[:, :, 0] = source // 2
            target[:, :, 1] = source - target[:, :, 0]
        else:
            target[:, :, 0] = source * 0.25
            target[:, :, 1] = source * 0.75
        fields[name] = target
    fine = BinnedSufficientStatistics(
        bins=fine_bins,
        num_agents=coarse.num_agents,
        alpha=coarse.alpha,
        min_count=coarse.min_count,
        observation_dtype=coarse.observation_dtype,
        effective_q_c_edges=fine_bins.q_c_edges,
        effective_q_d_edges=fine_bins.q_d_edges,
        **fields,
    )
    reconstructed = coarsen_abm_sufficient(fine, coarse.bins)
    for name in fields:
        np.testing.assert_array_equal(getattr(reconstructed, name), getattr(coarse, name))


def test_source_time_alignment_rejects_off_by_one() -> None:
    point = _two_point_pair_statistics()
    pair = derive_pair_binned_moments(
        aggregate_pair_points(point, QBinSpec([-0.1, 1.1], [-0.1, 0.1])),
        num_agents=3,
        alpha=0.4,
    )
    compare_four_way(_abm_statistics(), pair, abm_source_times=[0])
    with pytest.raises(ValueError, match="source_times"):
        compare_four_way(_abm_statistics(), pair, abm_source_times=[1])


def test_shared_complete_run_bootstrap_weights_drive_hybrid_intervals() -> None:
    pair = derive_pair_binned_moments(
        aggregate_pair_points(
            _two_point_pair_statistics(), QBinSpec([-0.1, 1.1], [-0.1, 0.1])
        ), num_agents=3, alpha=0.4
    )
    statistics = _abm_statistics(runs=2)
    weights = bootstrap_run_weights(2, 16, 9)
    summary = bootstrap_four_way_intervals(
        statistics, pair, weights, confidence_level=0.95
    )

    assert weights.shape == (16, 2)
    assert np.all(weights.sum(axis=1) == 2)
    index = (0, 0, 0, 0)
    assert summary.interval_valid["direct_abm_velocity_variance"][index]
    assert summary.interval_valid["hybrid_velocity_variance"][index]
    assert "pair_velocity_variance" not in summary.lower


def test_pair_and_abm_can_agree_by_construction() -> None:
    selected = np.array([[[0.5, 0.0], [0.5, 0.0]]])
    means = np.array([[[0.0, 0.0], [2.0, 0.0]]])
    seconds = np.array([[[1.0, 0.0], [5.0, 0.0]]])
    q = np.array([[0.0, 0.0], [1.0, 0.0]])
    q_selected = q[None, :, :]
    point = PairPointSufficientStatistics(
        source_times=np.array([0]),
        q_points=q,
        observation_dtype="float64",
        focal_mass=np.array([[0.5, 0.5]]),
        selected_mass=selected,
        sum_y=selected * means,
        sum_y2=selected * seconds,
        sum_distinct_y=selected * means * means,
        sum_q=selected * q_selected,
        sum_q2=selected * q_selected * q_selected,
        sum_y_q=selected * means * q_selected,
    )
    pair = derive_pair_binned_moments(
        aggregate_pair_points(point, QBinSpec([-0.1, 1.1], [-0.1, 0.1])),
        num_agents=3,
        alpha=0.4,
    )
    abm = _custom_abm_statistics(
        m2=3.0, m11=2.0, q2=0.5, reward_q=1.0, velocity_variance=0.12
    )
    comparison = compare_four_way(abm, pair)
    index = (0, 0, 0, 0)

    assert comparison.direct_abm_velocity_variance[index] == pytest.approx(0.12)
    assert comparison.reconstructed_abm_velocity_variance[index] == pytest.approx(0.12)
    assert comparison.pair_velocity_variance[index] == pytest.approx(0.12)
    assert comparison.hybrid_velocity_variance[index] == pytest.approx(0.12)


def test_nonzero_abm_covariance_moves_hybrid_in_expected_direction() -> None:
    q = np.array([[0.5, 0.0]])
    selected = np.array([[[1.0, 0.0]]])
    means = np.array([[[1.0, 0.0]]])
    point = PairPointSufficientStatistics(
        source_times=np.array([0]),
        q_points=q,
        observation_dtype="float64",
        focal_mass=np.array([[1.0]]),
        selected_mass=selected,
        sum_y=selected * means,
        sum_y2=selected * np.array([[[3.0, 0.0]]]),
        sum_distinct_y=selected * means * means,
        sum_q=selected * q[None, :, :],
        sum_q2=selected * q[None, :, :] ** 2,
        sum_y_q=selected * means * q[None, :, :],
    )
    pair = derive_pair_binned_moments(
        aggregate_pair_points(point, QBinSpec([-0.1, 1.1], [-0.1, 0.1])),
        num_agents=3,
        alpha=0.4,
    )
    abm = _custom_abm_statistics(
        m2=3.0, m11=2.0, q2=0.25, reward_q=0.5, velocity_variance=0.24
    )
    comparison = compare_four_way(abm, pair)
    index = (0, 0, 0, 0)

    assert comparison.abm_covariance[index] == pytest.approx(1.0)
    assert comparison.pair_covariance[index] == pytest.approx(0.0)
    assert comparison.pair_velocity_variance[index] == pytest.approx(0.16)
    assert comparison.hybrid_velocity_variance[index] == pytest.approx(0.24)
    assert abs(comparison.hybrid_minus_direct[index]) < abs(
        comparison.pair_minus_direct[index]
    )

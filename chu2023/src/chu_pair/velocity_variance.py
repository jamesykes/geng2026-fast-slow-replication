"""Matched finite-bin pair-versus-ABM velocity-variance comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import numpy as np

from .abm.statistics import BinnedSufficientStatistics, QBinSpec, derive_variance_moments
from .abm.uncertainty import (
    MIN_CONTRIBUTING_RUNS,
    MIN_VALID_BOOTSTRAP_FRACTION,
    QUANTILE_METHOD,
    pool_sufficient_statistics,
)
from .pair_density.jax_solver import JAXPairGrid, pair_point_sufficient_jax


PAIR_ACTION_SUM_FIELDS = (
    "selected_mass",
    "sum_y",
    "sum_y2",
    "sum_distinct_y",
    "sum_q",
    "sum_q2",
    "sum_y_q",
)
COMPARISON_BOOTSTRAP_ESTIMANDS = (
    "abm_sigma2",
    "abm_covariance",
    "direct_abm_velocity_variance",
    "reconstructed_abm_velocity_variance",
    "hybrid_velocity_variance",
    "pair_minus_direct",
    "hybrid_minus_direct",
    "direct_minus_reconstructed",
)


@dataclass(frozen=True, slots=True)
class PairPointSufficientStatistics:
    """Pair raw sums with axes ``(source_time, focal_point, action)``."""

    source_times: np.ndarray
    q_points: np.ndarray
    observation_dtype: str
    focal_mass: np.ndarray
    selected_mass: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray
    sum_distinct_y: np.ndarray
    sum_q: np.ndarray
    sum_q2: np.ndarray
    sum_y_q: np.ndarray


@dataclass(frozen=True, slots=True)
class PairBinnedSufficientStatistics:
    """Pair raw sums with axes ``(source_time,Q_C-bin,Q_D-bin,action)``."""

    source_times: np.ndarray
    bins: QBinSpec
    observation_dtype: str
    effective_q_c_edges: np.ndarray
    effective_q_d_edges: np.ndarray
    focal_mass: np.ndarray
    selected_mass: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray
    sum_distinct_y: np.ndarray
    sum_q: np.ndarray
    sum_q2: np.ndarray
    sum_y_q: np.ndarray


@dataclass(frozen=True, slots=True)
class PairBinnedMoments:
    source_times: np.ndarray
    focal_mass: np.ndarray
    selected_mass: np.ndarray
    has_focal_mass: np.ndarray
    has_selected_mass: np.ndarray
    mu: np.ndarray
    m2: np.ndarray
    m11: np.ndarray
    sigma2: np.ndarray
    covariance: np.ndarray
    mean_q: np.ndarray
    q_variance: np.ndarray
    reward_q_covariance: np.ndarray
    reward_variance: np.ndarray
    velocity_variance: np.ndarray
    mean_local_sigma2: np.ndarray


@dataclass(frozen=True, slots=True)
class FourWayComparison:
    abm_count: np.ndarray
    contributing_runs: np.ndarray
    pair_focal_mass: np.ndarray
    pair_selected_mass: np.ndarray
    abm_mu: np.ndarray
    abm_m2: np.ndarray
    abm_m11: np.ndarray
    abm_sigma2: np.ndarray
    abm_covariance: np.ndarray
    pair_mu: np.ndarray
    pair_m2: np.ndarray
    pair_m11: np.ndarray
    pair_sigma2: np.ndarray
    pair_covariance: np.ndarray
    abm_mean_q: np.ndarray
    abm_q_variance: np.ndarray
    abm_reward_q_covariance: np.ndarray
    pair_mean_q: np.ndarray
    pair_q_variance: np.ndarray
    pair_reward_q_covariance: np.ndarray
    direct_abm_velocity_variance: np.ndarray
    reconstructed_abm_velocity_variance: np.ndarray
    pair_velocity_variance: np.ndarray
    hybrid_velocity_variance: np.ndarray
    direct_minus_reconstructed: np.ndarray
    pair_minus_direct: np.ndarray
    hybrid_minus_direct: np.ndarray
    pair_to_direct_ratio: np.ndarray
    hybrid_to_direct_ratio: np.ndarray
    pair_mean_local_sigma2: np.ndarray
    has_abm_observations: np.ndarray
    pair_has_focal_mass: np.ndarray
    pair_has_selected_mass: np.ndarray
    pair_valid: np.ndarray
    abm_reconstruction_defined: np.ndarray
    hybrid_valid: np.ndarray
    sparse: np.ndarray


@dataclass(frozen=True, slots=True)
class ComparisonBootstrapSummary:
    lower: dict[str, np.ndarray]
    upper: dict[str, np.ndarray]
    valid_replicates: dict[str, np.ndarray]
    invalid_replicates: dict[str, np.ndarray]
    interval_valid: dict[str, np.ndarray]
    replicates: int
    confidence_level: float


def _safe_divide(total: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.shape(total), np.nan, dtype=np.float64)
    np.divide(total, denominator, out=result, where=denominator > 0)
    return result


def pair_point_sufficient_from_masses(
    pair_masses: list,
    source_times: list[int] | tuple[int, ...],
    grid: JAXPairGrid,
    tau: float,
) -> PairPointSufficientStatistics:
    """Copy bounded exact-grid pair sums from selected source states to host."""

    records = [
        jax.device_get(pair_point_sufficient_jax(mass, grid, tau))
        for mass in pair_masses
    ]
    return pair_point_sufficient_from_records(records, source_times, grid)


def pair_point_sufficient_from_records(
    records: list,
    source_times: list[int] | tuple[int, ...],
    grid: JAXPairGrid,
) -> PairPointSufficientStatistics:
    """Stack already-computed point summaries without retaining pair densities."""

    times = np.asarray(source_times, dtype=np.int64)
    if times.ndim != 1 or len(records) != times.size:
        raise ValueError("records and source_times must be one-dimensional and aligned")
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("source_times must be strictly increasing non-negative integers")
    if not records:
        raise ValueError("at least one pair source state is required")
    q_points = np.asarray(jax.device_get(grid.q_points))
    fields = {
        name: np.stack([np.asarray(getattr(record, name)) for record in records])
        for name in ("focal_mass", *PAIR_ACTION_SUM_FIELDS)
    }
    return PairPointSufficientStatistics(
        source_times=times,
        q_points=q_points,
        observation_dtype=np.dtype(q_points.dtype).name,
        **fields,
    )


def pair_point_sufficient_from_jax_summary(
    summary,
    source_times: list[int] | tuple[int, ...],
    grid: JAXPairGrid,
) -> PairPointSufficientStatistics:
    """Copy an already time-stacked compiled summary without another stack."""

    times = np.asarray(source_times, dtype=np.int64)
    if times.ndim != 1 or times.size < 1 or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("source_times must be strictly increasing non-negative integers")
    q_points = np.asarray(jax.device_get(grid.q_points))
    fields = {
        name: np.asarray(jax.device_get(getattr(summary, name)))
        for name in ("focal_mass", *PAIR_ACTION_SUM_FIELDS)
    }
    if fields["focal_mass"].shape != (times.size, q_points.shape[0]):
        raise ValueError("compiled pair summary shape does not match source_times/grid")
    return PairPointSufficientStatistics(
        source_times=times,
        q_points=q_points,
        observation_dtype=np.dtype(q_points.dtype).name,
        **fields,
    )


def coarsen_abm_sufficient(
    fine: BinnedSufficientStatistics,
    coarse_bins: QBinSpec,
) -> BinnedSufficientStatistics:
    """Reconstruct one nested coarser ABM scheme from authoritative fine sums."""

    coarse_c, coarse_d = coarse_bins.effective_edges(fine.observation_dtype)
    fine_c = fine.effective_q_c_edges
    fine_d = fine.effective_q_d_edges
    if not (
        np.array_equal(fine_c[[0, -1]], coarse_c[[0, -1]])
        and np.array_equal(fine_d[[0, -1]], coarse_d[[0, -1]])
        and np.all(np.isin(coarse_c, fine_c))
        and np.all(np.isin(coarse_d, fine_d))
    ):
        raise ValueError("coarse bins must be nested in the fine effective edges")
    c_map = np.maximum(np.searchsorted(coarse_c, fine_c[:-1], side="right") - 1, 0)
    d_map = np.maximum(np.searchsorted(coarse_d, fine_d[:-1], side="right") - 1, 0)
    output = {}
    for name in (
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
        source = np.asarray(getattr(fine, name))
        target = np.zeros(
            (*source.shape[:2], *coarse_bins.bin_shape, source.shape[-1]),
            dtype=source.dtype,
        )
        for fine_c_index, coarse_c_index in enumerate(c_map):
            for fine_d_index, coarse_d_index in enumerate(d_map):
                target[:, :, coarse_c_index, coarse_d_index, :] += source[
                    :, :, fine_c_index, fine_d_index, :
                ]
        output[name] = target
    return BinnedSufficientStatistics(
        bins=coarse_bins,
        num_agents=fine.num_agents,
        alpha=fine.alpha,
        min_count=fine.min_count,
        observation_dtype=fine.observation_dtype,
        effective_q_c_edges=coarse_c,
        effective_q_d_edges=coarse_d,
        **output,
    )


def aggregate_pair_points(
    statistics: PairPointSufficientStatistics,
    bins: QBinSpec,
) -> PairBinnedSufficientStatistics:
    """Pool exact-grid raw sums into finite Q bins before nonlinear formulas."""

    q_points = np.asarray(statistics.q_points)
    q_c_bin, q_d_bin = bins.assign(q_points)
    effective_c, effective_d = bins.effective_edges(q_points.dtype)
    times = statistics.source_times.size
    shape = (times, *bins.bin_shape, 2)
    focal = np.zeros((times, *bins.bin_shape), dtype=np.float64)
    output = {
        name: np.zeros(shape, dtype=np.float64) for name in PAIR_ACTION_SUM_FIELDS
    }
    for time in range(times):
        np.add.at(focal[time], (q_c_bin, q_d_bin), statistics.focal_mass[time])
        for name in PAIR_ACTION_SUM_FIELDS:
            values = np.asarray(getattr(statistics, name)[time], dtype=np.float64)
            for action in range(2):
                np.add.at(
                    output[name][time, ..., action],
                    (q_c_bin, q_d_bin),
                    values[:, action],
                )
    return PairBinnedSufficientStatistics(
        source_times=statistics.source_times.copy(),
        bins=bins,
        observation_dtype=statistics.observation_dtype,
        effective_q_c_edges=effective_c,
        effective_q_d_edges=effective_d,
        focal_mass=focal,
        **output,
    )


def coarsen_pair_sufficient(
    fine: PairBinnedSufficientStatistics,
    coarse_bins: QBinSpec,
) -> PairBinnedSufficientStatistics:
    """Reconstruct a nested coarser scheme by exact raw-sum addition."""

    coarse_c, coarse_d = coarse_bins.effective_edges(fine.observation_dtype)
    fine_c = fine.effective_q_c_edges
    fine_d = fine.effective_q_d_edges
    if not (
        np.array_equal(fine_c[[0, -1]], coarse_c[[0, -1]])
        and np.array_equal(fine_d[[0, -1]], coarse_d[[0, -1]])
        and np.all(np.isin(coarse_c, fine_c))
        and np.all(np.isin(coarse_d, fine_d))
    ):
        raise ValueError("coarse bins must be nested in the fine effective edges")
    c_map = np.searchsorted(coarse_c, fine_c[:-1], side="right") - 1
    d_map = np.searchsorted(coarse_d, fine_d[:-1], side="right") - 1
    c_map = np.maximum(c_map, 0)
    d_map = np.maximum(d_map, 0)
    times = fine.source_times.size
    focal = np.zeros((times, *coarse_bins.bin_shape), dtype=np.float64)
    output = {
        name: np.zeros((times, *coarse_bins.bin_shape, 2), dtype=np.float64)
        for name in PAIR_ACTION_SUM_FIELDS
    }
    for fine_c_index, coarse_c_index in enumerate(c_map):
        for fine_d_index, coarse_d_index in enumerate(d_map):
            focal[:, coarse_c_index, coarse_d_index] += fine.focal_mass[
                :, fine_c_index, fine_d_index
            ]
            for name in PAIR_ACTION_SUM_FIELDS:
                output[name][:, coarse_c_index, coarse_d_index, :] += getattr(
                    fine, name
                )[:, fine_c_index, fine_d_index, :]
    return PairBinnedSufficientStatistics(
        source_times=fine.source_times.copy(),
        bins=coarse_bins,
        observation_dtype=fine.observation_dtype,
        effective_q_c_edges=coarse_c,
        effective_q_d_edges=coarse_d,
        focal_mass=focal,
        **output,
    )


def derive_pair_binned_moments(
    statistics: PairBinnedSufficientStatistics,
    *,
    num_agents: int,
    alpha: float,
) -> PairBinnedMoments:
    """Derive the finite-bin pair closure from pooled raw selected-action sums."""

    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 2:
        raise ValueError("num_agents must be an integer at least two")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    selected = statistics.selected_mass
    mu = _safe_divide(statistics.sum_y, selected)
    m2 = _safe_divide(statistics.sum_y2, selected)
    m11 = _safe_divide(statistics.sum_distinct_y, selected)
    mean_q = _safe_divide(statistics.sum_q, selected)
    mean_q2 = _safe_divide(statistics.sum_q2, selected)
    mean_y_q = _safe_divide(statistics.sum_y_q, selected)
    sigma2 = m2 - mu * mu
    covariance = m11 - mu * mu
    q_variance = mean_q2 - mean_q * mean_q
    reward_q_covariance = mean_y_q - mu * mean_q
    opponents = num_agents - 1
    reward_variance = sigma2 / opponents
    if opponents > 1:
        reward_variance = reward_variance + (opponents - 1) * covariance / opponents
    velocity_variance = alpha**2 * (
        reward_variance + q_variance - 2.0 * reward_q_covariance
    )
    mean_local_sigma2 = _safe_divide(
        statistics.sum_y2 - statistics.sum_distinct_y,
        selected,
    )
    return PairBinnedMoments(
        source_times=statistics.source_times.copy(),
        focal_mass=statistics.focal_mass,
        selected_mass=selected,
        has_focal_mass=statistics.focal_mass > 0,
        has_selected_mass=selected > 0,
        mu=mu,
        m2=m2,
        m11=m11,
        sigma2=sigma2,
        covariance=covariance,
        mean_q=mean_q,
        q_variance=q_variance,
        reward_q_covariance=reward_q_covariance,
        reward_variance=reward_variance,
        velocity_variance=velocity_variance,
        mean_local_sigma2=mean_local_sigma2,
    )


def select_abm_source_times(
    statistics: BinnedSufficientStatistics,
    source_times: list[int] | tuple[int, ...] | np.ndarray,
) -> BinnedSufficientStatistics:
    """Select explicit ABM source-time records without changing their labels."""

    times = np.asarray(source_times, dtype=np.int64)
    if times.ndim != 1 or times.size < 1 or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("source_times must be non-empty, strictly increasing, and non-negative")
    if times[-1] >= statistics.counts.shape[1]:
        raise ValueError("requested source time is outside the ABM record trajectory")
    fields = {
        name: getattr(statistics, name)[:, times, ...]
        for name in (
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
        )
    }
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


def compare_four_way(
    abm_statistics: BinnedSufficientStatistics,
    pair: PairBinnedMoments,
    *,
    abm_source_times: list[int] | tuple[int, ...] | np.ndarray | None = None,
    ratio_epsilon: float = 1e-15,
) -> FourWayComparison:
    """Combine matched ABM/pair bins into four distinct velocity estimands."""

    if not math.isfinite(ratio_epsilon) or ratio_epsilon < 0:
        raise ValueError("ratio_epsilon must be finite and non-negative")
    pooled = pool_sufficient_statistics(abm_statistics)
    abm = derive_variance_moments(pooled)
    counts = pooled.counts[0]
    abm_arrays = {
        name: np.asarray(getattr(abm, name))[0]
        for name in (
            "mu",
            "m2",
            "m11",
            "sigma2",
            "covariance",
            "mean_selected_q",
            "selected_q_variance",
            "reward_selected_q_covariance",
            "direct_velocity_variance",
            "decomposed_reward_variance",
        )
    }
    expected_shape = counts.shape
    if pair.selected_mass.shape != expected_shape:
        raise ValueError(
            f"pair and ABM stratum shapes differ: {pair.selected_mass.shape} != {expected_shape}"
        )
    expected_times = (
        np.arange(expected_shape[0], dtype=np.int64)
        if abm_source_times is None
        else np.asarray(abm_source_times, dtype=np.int64)
    )
    if expected_times.shape != (expected_shape[0],):
        raise ValueError("abm_source_times must match the selected ABM time axis")
    if not np.array_equal(pair.source_times, expected_times):
        raise ValueError(
            "pair source_times must exactly match ABM record times 0..T-1"
        )
    contributing = np.count_nonzero(abm_statistics.counts, axis=0).astype(np.int64)
    opponents = abm_statistics.num_opponents
    reconstructed = abm_statistics.alpha**2 * (
        abm_arrays["decomposed_reward_variance"]
        + abm_arrays["selected_q_variance"]
        - 2.0 * abm_arrays["reward_selected_q_covariance"]
    )
    if opponents > 1:
        hybrid_reward = (
            pair.sigma2 / opponents
            + (opponents - 1) * abm_arrays["covariance"] / opponents
        )
    else:
        hybrid_reward = pair.sigma2 / opponents
    hybrid = abm_statistics.alpha**2 * (
        hybrid_reward + pair.q_variance - 2.0 * pair.reward_q_covariance
    )
    direct = abm_arrays["direct_velocity_variance"]
    pair_minus = pair.velocity_variance - direct
    hybrid_minus = hybrid - direct
    safe_ratio = np.isfinite(direct) & (np.abs(direct) > ratio_epsilon)
    pair_ratio = np.full(expected_shape, np.nan)
    hybrid_ratio = np.full(expected_shape, np.nan)
    np.divide(pair.velocity_variance, direct, out=pair_ratio, where=safe_ratio)
    np.divide(hybrid, direct, out=hybrid_ratio, where=safe_ratio)
    has_abm = counts > 0
    abm_reconstruction_defined = has_abm & np.isfinite(reconstructed)
    pair_valid = pair.has_selected_mass & np.isfinite(pair.velocity_variance)
    hybrid_valid = pair_valid & has_abm & np.isfinite(hybrid)
    return FourWayComparison(
        abm_count=counts,
        contributing_runs=contributing,
        pair_focal_mass=pair.focal_mass,
        pair_selected_mass=pair.selected_mass,
        abm_mu=abm_arrays["mu"],
        abm_m2=abm_arrays["m2"],
        abm_m11=abm_arrays["m11"],
        abm_sigma2=abm_arrays["sigma2"],
        abm_covariance=abm_arrays["covariance"],
        pair_mu=pair.mu,
        pair_m2=pair.m2,
        pair_m11=pair.m11,
        pair_sigma2=pair.sigma2,
        pair_covariance=pair.covariance,
        abm_mean_q=abm_arrays["mean_selected_q"],
        abm_q_variance=abm_arrays["selected_q_variance"],
        abm_reward_q_covariance=abm_arrays["reward_selected_q_covariance"],
        pair_mean_q=pair.mean_q,
        pair_q_variance=pair.q_variance,
        pair_reward_q_covariance=pair.reward_q_covariance,
        direct_abm_velocity_variance=direct,
        reconstructed_abm_velocity_variance=reconstructed,
        pair_velocity_variance=pair.velocity_variance,
        hybrid_velocity_variance=hybrid,
        direct_minus_reconstructed=direct - reconstructed,
        pair_minus_direct=pair_minus,
        hybrid_minus_direct=hybrid_minus,
        pair_to_direct_ratio=pair_ratio,
        hybrid_to_direct_ratio=hybrid_ratio,
        pair_mean_local_sigma2=pair.mean_local_sigma2,
        has_abm_observations=has_abm,
        pair_has_focal_mass=np.broadcast_to(
            pair.has_focal_mass[..., None], expected_shape
        ),
        pair_has_selected_mass=pair.has_selected_mass,
        pair_valid=pair_valid,
        abm_reconstruction_defined=abm_reconstruction_defined,
        hybrid_valid=hybrid_valid,
        sparse=has_abm & (counts < abm_statistics.min_count),
    )


def bootstrap_four_way_intervals(
    abm_statistics: BinnedSufficientStatistics,
    pair: PairBinnedMoments,
    run_weights: np.ndarray,
    *,
    confidence_level: float,
) -> ComparisonBootstrapSummary:
    """Complete-run intervals for ABM-dependent comparison quantities only."""

    weights = np.asarray(run_weights)
    runs = abm_statistics.counts.shape[0]
    if weights.ndim != 2 or weights.shape[1] != runs:
        raise ValueError(f"run_weights must have shape (B,{runs})")
    if not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0):
        raise ValueError("run_weights must contain non-negative integers")
    if np.any(weights.sum(axis=1) != runs):
        raise ValueError("each bootstrap row must contain exactly R run draws")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    replicates = weights.shape[0]
    shape = pair.selected_mass.shape
    values = {
        name: np.full((replicates, *shape), np.nan, dtype=np.float64)
        for name in COMPARISON_BOOTSTRAP_ESTIMANDS
    }
    for replicate, run_weight in enumerate(weights):
        pooled = pool_sufficient_statistics(abm_statistics, run_weight)
        comparison = compare_four_way(
            pooled,
            pair,
            abm_source_times=pair.source_times,
        )
        for name in COMPARISON_BOOTSTRAP_ESTIMANDS:
            values[name][replicate] = getattr(comparison, name)
    contributing = np.count_nonzero(abm_statistics.counts, axis=0)
    required = max(2, math.ceil(MIN_VALID_BOOTSTRAP_FRACTION * replicates))
    alpha_tail = (1.0 - confidence_level) / 2.0
    lower = {name: np.full(shape, np.nan) for name in values}
    upper = {name: np.full(shape, np.nan) for name in values}
    valid_replicates = {name: np.zeros(shape, dtype=np.int32) for name in values}
    invalid_replicates = {name: np.zeros(shape, dtype=np.int32) for name in values}
    interval_valid = {name: np.zeros(shape, dtype=np.bool_) for name in values}
    for name, array in values.items():
        for index in np.ndindex(shape):
            sample = array[(slice(None), *index)]
            finite = np.isfinite(sample)
            valid_count = int(np.count_nonzero(finite))
            valid_replicates[name][index] = valid_count
            invalid_replicates[name][index] = replicates - valid_count
            valid = contributing[index] >= MIN_CONTRIBUTING_RUNS and valid_count >= required
            interval_valid[name][index] = valid
            if valid:
                lower[name][index], upper[name][index] = np.quantile(
                    sample[finite],
                    [alpha_tail, 1.0 - alpha_tail],
                    method=QUANTILE_METHOD,
                )
    return ComparisonBootstrapSummary(
        lower=lower,
        upper=upper,
        valid_replicates=valid_replicates,
        invalid_replicates=invalid_replicates,
        interval_valid=interval_valid,
        replicates=int(replicates),
        confidence_level=float(confidence_level),
    )

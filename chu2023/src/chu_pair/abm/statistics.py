"""Population-moment estimators for Phase 3A ABM source-time records."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class QBinSpec:
    """Two-dimensional Q bins, left-closed and right-open except at the top."""

    q_c_edges: FloatArray
    q_d_edges: FloatArray

    def __post_init__(self) -> None:
        for name in ("q_c_edges", "q_d_edges"):
            edges = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError(f"{name} must be a one-dimensional array with at least two edges")
            if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
                raise ValueError(f"{name} must be finite and strictly increasing")
            edges.setflags(write=False)
            object.__setattr__(self, name, edges)

    @property
    def num_q_c_bins(self) -> int:
        return int(self.q_c_edges.size - 1)

    @property
    def num_q_d_bins(self) -> int:
        return int(self.q_d_edges.size - 1)

    @property
    def bin_shape(self) -> tuple[int, int]:
        return self.num_q_c_bins, self.num_q_d_bins

    def effective_edges(
        self,
        observation_dtype,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return validated comparison edges in the observation Q dtype."""

        dtype = np.dtype(observation_dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Q observations must use a floating dtype")
        converted = []
        for name in ("q_c_edges", "q_d_edges"):
            edges = np.array(getattr(self, name), dtype=dtype, copy=True)
            if not np.all(np.isfinite(edges)):
                raise ValueError(
                    f"{name} contains a non-finite value after conversion to {dtype.name}"
                )
            if np.any(np.diff(edges) <= 0):
                raise ValueError(
                    f"{name} is not strictly increasing after conversion to "
                    f"observation dtype {dtype.name}; configured edges may have collapsed"
                )
            edges.setflags(write=False)
            converted.append(edges)
        return converted[0], converted[1]

    def assign(self, q_values: ArrayLike) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Assign complete Q-vectors without clipping or discarding observations."""

        q = np.asarray(q_values)
        if q.ndim < 1 or q.shape[-1] != 2:
            raise ValueError("q_values must have final dimension 2")
        if not np.issubdtype(q.dtype, np.floating):
            raise TypeError("q_values must use a floating dtype")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_values must be finite")

        q_c_edges, q_d_edges = self.effective_edges(q.dtype)

        indices = []
        for coordinate, edges, name in (
            (q[..., 0], q_c_edges, "Q(C)"),
            (q[..., 1], q_d_edges, "Q(D)"),
        ):
            outside = (coordinate < edges[0]) | (coordinate > edges[-1])
            if np.any(outside):
                values = coordinate[outside]
                raise ValueError(
                    f"{name} has {values.size} observations outside bin range "
                    f"[{edges[0]}, {edges[-1]}], including {values.flat[0]}"
                )
            index = np.searchsorted(edges, coordinate, side="right") - 1
            index = np.where(coordinate == edges[-1], edges.size - 2, index)
            indices.append(index.astype(np.intp, copy=False))
        return indices[0], indices[1]


@dataclass(frozen=True, slots=True)
class BinnedSufficientStatistics:
    """Sums with axes ``(run, source_time, q_C_bin, q_D_bin, action)``."""

    bins: QBinSpec
    num_agents: int
    alpha: float
    min_count: int
    observation_dtype: str
    effective_q_c_edges: np.ndarray
    effective_q_d_edges: np.ndarray
    counts: IntArray
    sum_s1: FloatArray
    sum_s2: FloatArray
    sum_distinct_products: FloatArray
    sum_reward: FloatArray
    sum_reward_squared: FloatArray
    sum_selected_q: FloatArray
    sum_selected_q_squared: FloatArray
    sum_reward_selected_q: FloatArray
    sum_velocity: FloatArray
    sum_velocity_squared: FloatArray

    @property
    def num_opponents(self) -> int:
        return self.num_agents - 1


@dataclass(frozen=True, slots=True)
class BinnedMomentEstimates:
    """Population moments derived from one common set of sufficient sums."""

    counts: IntArray
    has_observations: BoolArray
    underpopulated: BoolArray
    meets_min_count: BoolArray
    distinct_covariance_defined: BoolArray
    mu: FloatArray
    m2: FloatArray
    m11: FloatArray
    sigma2: FloatArray
    covariance: FloatArray
    mean_reward: FloatArray
    direct_reward_variance: FloatArray
    decomposed_reward_variance: FloatArray
    mean_selected_q: FloatArray
    selected_q_variance: FloatArray
    reward_selected_q_covariance: FloatArray
    mean_velocity: FloatArray
    direct_velocity_variance: FloatArray
    finite_bin_velocity_variance: FloatArray


def _as_batched_records(records) -> dict[str, np.ndarray]:
    q_t = np.asarray(records.q_t)
    if q_t.ndim == 3:
        add_run_axis = True
        q_t = q_t[None, ...]
    elif q_t.ndim == 4:
        add_run_axis = False
    else:
        raise ValueError("instrumented q_t must have shape (T,n,2) or (R,T,n,2)")
    if q_t.shape[-1] != 2:
        raise ValueError("instrumented q_t must have final dimension 2")

    arrays = {"q_t": q_t}
    for name in (
        "actions_t",
        "selected_q_t",
        "rewards_t",
        "selected_velocities_t",
        "payoff_sums_t",
        "payoff_square_sums_t",
    ):
        value = np.asarray(getattr(records, name))
        if add_run_axis:
            value = value[None, ...]
        if value.shape != q_t.shape[:-1]:
            raise ValueError(f"{name} must have shape {q_t.shape[:-1]}, got {value.shape}")
        arrays[name] = value
    return arrays


def aggregate_variance_records(
    records,
    bins: QBinSpec,
    *,
    num_agents: int,
    alpha: float,
    min_count: int = 2,
) -> BinnedSufficientStatistics:
    """Aggregate selected-action observations without pooling independent runs."""

    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 2:
        raise ValueError("num_agents must be an integer at least 2")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer")

    arrays = _as_batched_records(records)
    q_t = arrays["q_t"]
    runs, steps, agents, _ = q_t.shape
    if agents != num_agents:
        raise ValueError(
            f"record agent axis has length {agents}, expected num_agents={num_agents}"
        )
    actions = arrays["actions_t"]
    if np.any((actions != 0) & (actions != 1)):
        raise ValueError("actions_t must contain only action indices 0 and 1")
    q_c_bin, q_d_bin = bins.assign(q_t)
    effective_q_c_edges, effective_q_d_edges = bins.effective_edges(q_t.dtype)

    output_shape = (runs, steps, *bins.bin_shape, 2)
    counts = np.zeros(output_shape, dtype=np.int64)
    sums = {
        name: np.zeros(output_shape, dtype=np.float64)
        for name in (
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

    run_index = np.broadcast_to(
        np.arange(runs, dtype=np.intp)[:, None, None], (runs, steps, agents)
    )
    time_index = np.broadcast_to(
        np.arange(steps, dtype=np.intp)[None, :, None], (runs, steps, agents)
    )
    index = (
        run_index.ravel(),
        time_index.ravel(),
        q_c_bin.ravel(),
        q_d_bin.ravel(),
        actions.astype(np.intp, copy=False).ravel(),
    )
    np.add.at(counts, index, 1)

    s1 = np.asarray(arrays["payoff_sums_t"], dtype=np.float64)
    s2 = np.asarray(arrays["payoff_square_sums_t"], dtype=np.float64)
    reward = np.asarray(arrays["rewards_t"], dtype=np.float64)
    selected_q = np.asarray(arrays["selected_q_t"], dtype=np.float64)
    velocity = np.asarray(arrays["selected_velocities_t"], dtype=np.float64)
    values = {
        "sum_s1": s1,
        "sum_s2": s2,
        "sum_distinct_products": s1 * s1 - s2,
        "sum_reward": reward,
        "sum_reward_squared": reward * reward,
        "sum_selected_q": selected_q,
        "sum_selected_q_squared": selected_q * selected_q,
        "sum_reward_selected_q": reward * selected_q,
        "sum_velocity": velocity,
        "sum_velocity_squared": velocity * velocity,
    }
    for name, value in values.items():
        np.add.at(sums[name], index, value.ravel())

    return BinnedSufficientStatistics(
        bins=bins,
        num_agents=num_agents,
        alpha=float(alpha),
        min_count=min_count,
        observation_dtype=np.dtype(q_t.dtype).name,
        effective_q_c_edges=effective_q_c_edges,
        effective_q_d_edges=effective_q_d_edges,
        counts=counts,
        **sums,
    )


def _safe_mean(total: FloatArray, denominator) -> FloatArray:
    denominator_array = np.asarray(denominator)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, denominator_array, out=result, where=denominator_array > 0)
    return result


def derive_variance_moments(
    statistics: BinnedSufficientStatistics,
) -> BinnedMomentEstimates:
    """Derive population moments, leaving empty or undefined values as NaN."""

    counts = statistics.counts
    has_observations = counts > 0
    underpopulated = has_observations & (counts < statistics.min_count)
    meets_min_count = counts >= statistics.min_count
    opponents = statistics.num_opponents

    mu = _safe_mean(statistics.sum_s1, counts * opponents)
    m2 = _safe_mean(statistics.sum_s2, counts * opponents)
    sigma2 = m2 - mu * mu
    mean_reward = _safe_mean(statistics.sum_reward, counts)
    mean_reward_squared = _safe_mean(statistics.sum_reward_squared, counts)
    direct_reward_variance = mean_reward_squared - mean_reward * mean_reward

    if opponents > 1:
        m11 = _safe_mean(
            statistics.sum_distinct_products,
            counts * opponents * (opponents - 1),
        )
        covariance = m11 - mu * mu
        distinct_covariance_defined = has_observations.copy()
        decomposed_reward_variance = (
            sigma2 / opponents + (opponents - 1) * covariance / opponents
        )
    else:
        m11 = np.full(counts.shape, np.nan, dtype=np.float64)
        covariance = np.full(counts.shape, np.nan, dtype=np.float64)
        distinct_covariance_defined = np.zeros(counts.shape, dtype=np.bool_)
        decomposed_reward_variance = sigma2 / opponents

    mean_selected_q = _safe_mean(statistics.sum_selected_q, counts)
    mean_selected_q_squared = _safe_mean(statistics.sum_selected_q_squared, counts)
    selected_q_variance = mean_selected_q_squared - mean_selected_q * mean_selected_q
    mean_reward_selected_q = _safe_mean(statistics.sum_reward_selected_q, counts)
    reward_selected_q_covariance = (
        mean_reward_selected_q - mean_reward * mean_selected_q
    )
    mean_velocity = _safe_mean(statistics.sum_velocity, counts)
    mean_velocity_squared = _safe_mean(statistics.sum_velocity_squared, counts)
    direct_velocity_variance = mean_velocity_squared - mean_velocity * mean_velocity
    finite_bin_velocity_variance = statistics.alpha**2 * (
        direct_reward_variance
        + selected_q_variance
        - 2.0 * reward_selected_q_covariance
    )

    return BinnedMomentEstimates(
        counts=counts,
        has_observations=has_observations,
        underpopulated=underpopulated,
        meets_min_count=meets_min_count,
        distinct_covariance_defined=distinct_covariance_defined,
        mu=mu,
        m2=m2,
        m11=m11,
        sigma2=sigma2,
        covariance=covariance,
        mean_reward=mean_reward,
        direct_reward_variance=direct_reward_variance,
        decomposed_reward_variance=decomposed_reward_variance,
        mean_selected_q=mean_selected_q,
        selected_q_variance=selected_q_variance,
        reward_selected_q_covariance=reward_selected_q_covariance,
        mean_velocity=mean_velocity,
        direct_velocity_variance=direct_velocity_variance,
        finite_bin_velocity_variance=finite_bin_velocity_variance,
    )

"""Independent-run bootstrap and nested-bin diagnostics for Phase 3B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np

from ..model import PAYOFF_TENSOR
from .statistics import BinnedSufficientStatistics, QBinSpec, derive_variance_moments


SUFFICIENT_SUM_FIELDS = (
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
BOOTSTRAP_ESTIMANDS = (
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
    "finite_bin_discrepancy",
    "finite_bin_discrepancy_equivalent",
)
MIN_CONTRIBUTING_RUNS = 2
MIN_VALID_BOOTSTRAP_FRACTION = 0.8
QUANTILE_METHOD = "linear"


class NamedBinScheme(NamedTuple):
    name: str
    bins: QBinSpec


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    """Pooled estimates and pointwise run-cluster bootstrap intervals."""

    total_count: np.ndarray
    contributing_runs: np.ndarray
    point: dict[str, np.ndarray]
    lower: dict[str, np.ndarray]
    upper: dict[str, np.ndarray]
    valid_replicates: dict[str, np.ndarray]
    invalid_replicates: dict[str, np.ndarray]
    interval_valid: dict[str, np.ndarray]
    bootstrap_replicates: int
    confidence_level: float


def bootstrap_run_weights(
    num_runs: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Draw run multiplicities; each row represents ``num_runs`` whole runs."""

    for name, value, minimum in (
        ("num_runs", num_runs, 1),
        ("replicates", replicates, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        num_runs,
        size=(replicates, num_runs),
        dtype=np.int32,
    )
    weights = np.zeros((replicates, num_runs), dtype=np.int32)
    rows = np.broadcast_to(
        np.arange(replicates, dtype=np.intp)[:, None], draws.shape
    )
    np.add.at(weights, (rows, draws), 1)
    return weights


def pool_sufficient_statistics(
    statistics: BinnedSufficientStatistics,
    run_weights: np.ndarray | None = None,
) -> BinnedSufficientStatistics:
    """Pool raw sums across runs before applying nonlinear moment formulas."""

    runs = statistics.counts.shape[0]
    if run_weights is None:
        counts = np.sum(statistics.counts, axis=0, dtype=np.int64)[None, ...]
        sums = {
            name: np.sum(
                getattr(statistics, name),
                axis=0,
                dtype=np.float64,
            )[None, ...]
            for name in SUFFICIENT_SUM_FIELDS
        }
    else:
        weights = np.asarray(run_weights)
        if weights.shape != (runs,):
            raise ValueError(f"run_weights must have shape ({runs},)")
        if not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0):
            raise ValueError("run_weights must be non-negative integers")
        integer_weights = weights.astype(np.int64, copy=False)
        float_weights = weights.astype(np.float64, copy=False)
        counts = np.tensordot(
            integer_weights,
            statistics.counts,
            axes=(0, 0),
        )[None, ...]
        sums = {
            name: np.tensordot(
                float_weights,
                getattr(statistics, name),
                axes=(0, 0),
            )[None, ...]
            for name in SUFFICIENT_SUM_FIELDS
        }
    return BinnedSufficientStatistics(
        bins=statistics.bins,
        num_agents=statistics.num_agents,
        alpha=statistics.alpha,
        min_count=statistics.min_count,
        observation_dtype=statistics.observation_dtype,
        effective_q_c_edges=statistics.effective_q_c_edges,
        effective_q_d_edges=statistics.effective_q_d_edges,
        counts=counts,
        **sums,
    )


def _estimands_from_sums(
    counts: np.ndarray,
    sums: dict[str, np.ndarray],
    *,
    num_opponents: int,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Vectorized population moments for arbitrary leading bootstrap axes."""

    def mean(total, denominator):
        result = np.full(total.shape, np.nan, dtype=np.float64)
        np.divide(total, denominator, out=result, where=denominator > 0)
        return result

    count_float = counts.astype(np.float64, copy=False)
    mu = mean(sums["sum_s1"], count_float * num_opponents)
    m2 = mean(sums["sum_s2"], count_float * num_opponents)
    sigma2 = m2 - mu * mu
    mean_reward = mean(sums["sum_reward"], count_float)
    reward_m2 = mean(sums["sum_reward_squared"], count_float)
    direct_reward_variance = reward_m2 - mean_reward * mean_reward
    if num_opponents > 1:
        m11 = mean(
            sums["sum_distinct_products"],
            count_float * num_opponents * (num_opponents - 1),
        )
        covariance = m11 - mu * mu
        decomposed = (
            sigma2 / num_opponents
            + (num_opponents - 1) * covariance / num_opponents
        )
    else:
        m11 = np.full(counts.shape, np.nan, dtype=np.float64)
        covariance = np.full(counts.shape, np.nan, dtype=np.float64)
        decomposed = sigma2 / num_opponents
    mean_q = mean(sums["sum_selected_q"], count_float)
    q_m2 = mean(sums["sum_selected_q_squared"], count_float)
    selected_q_variance = q_m2 - mean_q * mean_q
    reward_q = mean(sums["sum_reward_selected_q"], count_float)
    reward_q_covariance = reward_q - mean_reward * mean_q
    mean_velocity = mean(sums["sum_velocity"], count_float)
    velocity_m2 = mean(sums["sum_velocity_squared"], count_float)
    direct_velocity_variance = velocity_m2 - mean_velocity * mean_velocity
    finite_bin_velocity_variance = alpha**2 * (
        direct_reward_variance
        + selected_q_variance
        - 2.0 * reward_q_covariance
    )
    discrepancy = direct_velocity_variance - alpha**2 * direct_reward_variance
    discrepancy_equivalent = alpha**2 * (
        selected_q_variance - 2.0 * reward_q_covariance
    )
    return {
        "mu": mu,
        "m2": m2,
        "m11": m11,
        "sigma2": sigma2,
        "covariance": covariance,
        "direct_reward_variance": direct_reward_variance,
        "decomposed_reward_variance": decomposed,
        "direct_velocity_variance": direct_velocity_variance,
        "finite_bin_velocity_variance": finite_bin_velocity_variance,
        "selected_q_variance": selected_q_variance,
        "reward_selected_q_covariance": reward_q_covariance,
        "finite_bin_discrepancy": discrepancy,
        "finite_bin_discrepancy_equivalent": discrepancy_equivalent,
    }


def pooled_point_estimands(
    statistics: BinnedSufficientStatistics,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return observation-weighted point estimates with the run axis removed."""

    pooled = pool_sufficient_statistics(statistics)
    estimates = derive_variance_moments(pooled)
    point = {
        name: np.asarray(getattr(estimates, name))[0]
        for name in BOOTSTRAP_ESTIMANDS
        if hasattr(estimates, name)
    }
    point["finite_bin_discrepancy"] = (
        point["direct_velocity_variance"]
        - statistics.alpha**2 * point["direct_reward_variance"]
    )
    point["finite_bin_discrepancy_equivalent"] = statistics.alpha**2 * (
        point["selected_q_variance"]
        - 2.0 * point["reward_selected_q_covariance"]
    )
    contributing = np.count_nonzero(statistics.counts, axis=0).astype(
        np.int64, copy=False
    )
    return pooled.counts[0], contributing, point


def cluster_bootstrap_intervals(
    statistics: BinnedSufficientStatistics,
    run_weights: np.ndarray,
    *,
    confidence_level: float,
    stratum_chunk_size: int,
) -> BootstrapSummary:
    """Pointwise percentile intervals from common complete-run multiplicities."""

    weights = np.asarray(run_weights)
    runs = statistics.counts.shape[0]
    if weights.ndim != 2 or weights.shape[1] != runs:
        raise ValueError(f"run_weights must have shape (B,{runs})")
    if not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0):
        raise ValueError("run_weights must contain non-negative integers")
    if np.any(weights.sum(axis=1) != runs):
        raise ValueError("each bootstrap row must contain exactly R run draws")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if (
        isinstance(stratum_chunk_size, bool)
        or not isinstance(stratum_chunk_size, int)
        or stratum_chunk_size < 1
    ):
        raise ValueError("stratum_chunk_size must be a positive integer")

    total_count, contributing_runs, point = pooled_point_estimands(statistics)
    shape = total_count.shape
    cell_count = int(total_count.size)
    replicates = int(weights.shape[0])
    lower = {name: np.full(shape, np.nan) for name in BOOTSTRAP_ESTIMANDS}
    upper = {name: np.full(shape, np.nan) for name in BOOTSTRAP_ESTIMANDS}
    valid_replicates = {
        name: np.zeros(shape, dtype=np.int32) for name in BOOTSTRAP_ESTIMANDS
    }
    invalid_replicates = {
        name: np.full(shape, replicates, dtype=np.int32)
        for name in BOOTSTRAP_ESTIMANDS
    }
    interval_valid = {
        name: np.zeros(shape, dtype=np.bool_) for name in BOOTSTRAP_ESTIMANDS
    }
    flat_counts = statistics.counts.reshape(runs, cell_count)
    flat_sums = {
        name: getattr(statistics, name).reshape(runs, cell_count)
        for name in SUFFICIENT_SUM_FIELDS
    }
    alpha_tail = (1.0 - confidence_level) / 2.0
    required_valid = max(
        2,
        int(math.ceil(MIN_VALID_BOOTSTRAP_FRACTION * replicates)),
    )
    weights_float = weights.astype(np.float64, copy=False)

    for start in range(0, cell_count, stratum_chunk_size):
        stop = min(start + stratum_chunk_size, cell_count)
        boot_counts = weights @ flat_counts[:, start:stop]
        boot_sums = {
            name: weights_float @ values[:, start:stop]
            for name, values in flat_sums.items()
        }
        values = _estimands_from_sums(
            boot_counts,
            boot_sums,
            num_opponents=statistics.num_opponents,
            alpha=statistics.alpha,
        )
        for name in BOOTSTRAP_ESTIMANDS:
            value = values[name]
            for local in range(stop - start):
                target = start + local
                finite = np.isfinite(value[:, local])
                valid_count = int(np.count_nonzero(finite))
                valid_replicates[name].flat[target] = valid_count
                invalid_replicates[name].flat[target] = replicates - valid_count
                is_valid = (
                    contributing_runs.flat[target] >= MIN_CONTRIBUTING_RUNS
                    and valid_count >= required_valid
                )
                interval_valid[name].flat[target] = is_valid
                if is_valid:
                    quantiles = np.quantile(
                        value[finite, local],
                        [alpha_tail, 1.0 - alpha_tail],
                        method=QUANTILE_METHOD,
                    )
                    lower[name].flat[target] = quantiles[0]
                    upper[name].flat[target] = quantiles[1]

    return BootstrapSummary(
        total_count=total_count,
        contributing_runs=contributing_runs,
        point=point,
        lower=lower,
        upper=upper,
        valid_replicates=valid_replicates,
        invalid_replicates=invalid_replicates,
        interval_valid=interval_valid,
        bootstrap_replicates=replicates,
        confidence_level=float(confidence_level),
    )


def validate_nested_schemes(
    schemes: list[NamedBinScheme],
    observation_dtype,
) -> None:
    """Require successively finer, exactly nested configured/effective edges."""

    if not schemes:
        raise ValueError("at least one bin-refinement scheme is required")
    names = [scheme.name for scheme in schemes]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("bin-refinement scheme names must be non-empty and unique")
    configured_outer = (
        schemes[0].bins.q_c_edges[[0, -1]],
        schemes[0].bins.q_d_edges[[0, -1]],
    )
    effective_outer = tuple(
        edges[[0, -1]]
        for edges in schemes[0].bins.effective_edges(observation_dtype)
    )
    for index, scheme in enumerate(schemes):
        effective = scheme.bins.effective_edges(observation_dtype)
        for axis, label in enumerate(("Q(C)", "Q(D)")):
            if not np.array_equal(
                (scheme.bins.q_c_edges, scheme.bins.q_d_edges)[axis][[0, -1]],
                configured_outer[axis],
            ):
                raise ValueError(f"scheme {scheme.name!r} has different {label} bounds")
            if not np.array_equal(effective[axis][[0, -1]], effective_outer[axis]):
                raise ValueError(
                    f"scheme {scheme.name!r} has different effective {label} bounds"
                )
        if index == 0:
            continue
        parent = schemes[index - 1]
        parent_effective = parent.bins.effective_edges(observation_dtype)
        if not (
            scheme.bins.num_q_c_bins > parent.bins.num_q_c_bins
            and scheme.bins.num_q_d_bins > parent.bins.num_q_d_bins
        ):
            raise ValueError(
                f"scheme {scheme.name!r} must genuinely refine both coordinates"
            )
        for child_edges, parent_edges, child_effective, parent_eff, label in (
            (
                scheme.bins.q_c_edges,
                parent.bins.q_c_edges,
                effective[0],
                parent_effective[0],
                "Q(C)",
            ),
            (
                scheme.bins.q_d_edges,
                parent.bins.q_d_edges,
                effective[1],
                parent_effective[1],
                "Q(D)",
            ),
        ):
            if not np.all(np.isin(parent_edges, child_edges)):
                raise ValueError(
                    f"scheme {scheme.name!r} is not nested in configured {label} edges"
                )
            if not np.all(np.isin(parent_eff, child_effective)):
                raise ValueError(
                    f"scheme {scheme.name!r} is not nested in effective {label} edges"
                )


def _gamma(additions: np.ndarray | int, dtype: np.dtype) -> np.ndarray:
    """Return ``k*eps/(1-k*eps)`` using machine epsilon conservatively."""

    count = np.asarray(additions, dtype=np.float64)
    if np.any(count < 0) or not np.all(np.isfinite(count)):
        raise ValueError("summation addition counts must be finite and non-negative")
    product = count * np.finfo(dtype).eps
    if np.any(product >= 1.0):
        raise ValueError("summation roundoff bound is invalid because k*epsilon >= 1")
    return product / (1.0 - product)


def _outward_nonnegative(value: float, description: str) -> float:
    """Round a finite non-negative binary64 calculation towards ``+inf``."""

    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"non-finite or negative {description} prevents a roundoff bound")
    if result == 0.0:
        return 0.0
    outward = math.nextafter(result, math.inf)
    if not math.isfinite(outward):
        raise ValueError(f"overflow in {description} prevents a roundoff bound")
    return outward


def _outward_sum(description: str, *values: float) -> float:
    return _outward_nonnegative(sum(values), description)


def _outward_product(left: float, right: float, description: str) -> float:
    return _outward_nonnegative(left * right, description)


def _rounded_operation_bound(
    exact_magnitude_bound: float,
    operations: int,
    dtype: np.dtype,
    description: str,
) -> float:
    """Bound represented arithmetic using machine epsilon plus underflow error."""

    magnitude = _outward_nonnegative(exact_magnitude_bound, description)
    gamma = float(_gamma(operations, dtype))
    relative_error = _outward_product(magnitude, gamma, f"{description} error")
    underflow_error = _outward_product(
        float(operations),
        float(np.finfo(dtype).smallest_subnormal),
        f"{description} underflow allowance",
    )
    return _outward_sum(
        description,
        magnitude,
        relative_error,
        underflow_error,
    )


def _represented_scalar(value, dtype: np.dtype, description: str) -> float:
    """Cast one model scalar as the ABM does and reject overflow explicitly."""

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            represented = float(np.asarray(value, dtype=dtype))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{description} is not representable in {dtype.name}") from error
    if not math.isfinite(represented):
        raise ValueError(f"{description} is not finite in {dtype.name}")
    return represented


def _absolute_term_bounds(
    statistics: BinnedSufficientStatistics,
    summation_dtype: np.dtype,
) -> dict[str, float]:
    """Bound actual represented terms entering the sufficient-array sums."""

    opponents = statistics.num_opponents
    observation_dtype = np.dtype(statistics.observation_dtype)
    if observation_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("observation_dtype must be float32 or float64")
    if summation_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("summation dtype must be float32 or float64")
    if opponents < 1:
        raise ValueError("at least one opponent is required for represented bounds")

    represented_payoffs = np.asarray(PAYOFF_TENSOR, dtype=observation_dtype)
    payoff_scale = float(np.max(np.abs(represented_payoffs)))
    q_scale = float(
        max(
            np.max(np.abs(statistics.effective_q_c_edges)),
            np.max(np.abs(statistics.effective_q_d_edges)),
        )
    )
    q_scale = _outward_nonnegative(q_scale, "selected-Q scale")
    alpha_scale = abs(
        _represented_scalar(statistics.alpha, observation_dtype, "alpha")
    )
    alpha_scale = _outward_nonnegative(alpha_scale, "represented alpha")
    represented_opponents = _represented_scalar(
        opponents,
        observation_dtype,
        "opponent count",
    )
    if not math.isfinite(represented_opponents) or represented_opponents <= 0.0:
        raise ValueError("represented opponent count must be finite and positive")

    payoff_square = _rounded_operation_bound(
        _outward_product(payoff_scale, payoff_scale, "payoff square"),
        1,
        observation_dtype,
        "represented payoff square",
    )
    s1_bound = _rounded_operation_bound(
        _outward_product(float(opponents), payoff_scale, "S1 exact magnitude"),
        opponents,
        observation_dtype,
        "represented S1",
    )
    s2_bound = _rounded_operation_bound(
        _outward_product(float(opponents), payoff_square, "S2 exact magnitude"),
        opponents,
        observation_dtype,
        "represented S2",
    )
    reward_bound = _rounded_operation_bound(
        _outward_nonnegative(
            s1_bound / represented_opponents,
            "reward quotient magnitude",
        ),
        1,
        observation_dtype,
        "represented reward",
    )
    reward_minus_q_bound = _rounded_operation_bound(
        _outward_sum("reward-minus-Q magnitude", reward_bound, q_scale),
        1,
        observation_dtype,
        "represented reward-minus-Q",
    )
    velocity_bound = _rounded_operation_bound(
        _outward_product(
            alpha_scale,
            reward_minus_q_bound,
            "velocity product magnitude",
        ),
        1,
        observation_dtype,
        "represented velocity",
    )

    s1_squared = _rounded_operation_bound(
        _outward_product(s1_bound, s1_bound, "host S1 square magnitude"),
        1,
        summation_dtype,
        "host S1 square",
    )
    distinct_product_bound = _rounded_operation_bound(
        _outward_sum(
            "host distinct-product subtraction magnitude",
            s1_squared,
            s2_bound,
        ),
        1,
        summation_dtype,
        "host distinct-product expression",
    )

    def host_product_bound(left: float, right: float, description: str) -> float:
        return _rounded_operation_bound(
            _outward_product(left, right, f"{description} magnitude"),
            1,
            summation_dtype,
            description,
        )

    bounds = {
        "sum_s1": s1_bound,
        "sum_s2": s2_bound,
        "sum_distinct_products": distinct_product_bound,
        "sum_reward": reward_bound,
        "sum_reward_squared": host_product_bound(
            reward_bound, reward_bound, "host reward square"
        ),
        "sum_selected_q": q_scale,
        "sum_selected_q_squared": host_product_bound(
            q_scale, q_scale, "host selected-Q square"
        ),
        "sum_reward_selected_q": host_product_bound(
            reward_bound, q_scale, "host reward-Q product"
        ),
        "sum_velocity": velocity_bound,
        "sum_velocity_squared": host_product_bound(
            velocity_bound, velocity_bound, "host velocity square"
        ),
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in bounds.values()):
        raise ValueError("non-finite model scale prevents a roundoff bound")
    return bounds


def _reconstruction_bound(
    parent_values: np.ndarray,
    child_values: np.ndarray,
    parent_counts: np.ndarray,
    child_counts: np.ndarray,
    *,
    term_bound: float,
    dtype: np.dtype,
) -> np.ndarray:
    """Bound two differently ordered sums of the same represented terms."""

    parent_count_float = parent_counts.astype(np.float64, copy=False)
    child_count_float = child_counts.astype(np.float64, copy=False)
    parent_scale = np.maximum(
        parent_count_float * term_bound,
        np.abs(parent_values),
    )
    child_scale = np.maximum(
        child_count_float * term_bound,
        np.abs(child_values),
    )
    parent_error = _gamma(parent_counts, dtype) * parent_scale
    child_error = np.sum(
        _gamma(child_counts, dtype) * child_scale,
        axis=(2, 3),
        dtype=np.float64,
    )
    child_absolute_sum = np.sum(
        np.abs(child_values),
        axis=(2, 3),
        dtype=np.float64,
    )
    child_bin_count = child_values.shape[2] * child_values.shape[3]
    regroup_error = _gamma(child_bin_count, dtype) * (
        child_absolute_sum + child_error
    )
    operation_count = 2 * parent_count_float + child_bin_count
    underflow_allowance = operation_count * np.finfo(dtype).smallest_subnormal
    return parent_error + child_error + regroup_error + underflow_allowance


def assert_child_reconstructs_parent(
    parent: BinnedSufficientStatistics,
    child: BinnedSufficientStatistics,
) -> dict:
    """Check exact counts and forward-error-bounded floating reconstructions."""

    if parent.counts.shape[:2] != child.counts.shape[:2] or parent.counts.shape[-1] != 2:
        raise ValueError("parent and child run/time/action axes must agree")
    if parent.num_agents != child.num_agents or parent.alpha != child.alpha:
        raise ValueError("parent and child model parameters must agree")
    if parent.observation_dtype != child.observation_dtype:
        raise ValueError("parent and child observation dtypes must agree")
    parent_c = parent.effective_q_c_edges
    parent_d = parent.effective_q_d_edges
    child_c = child.effective_q_c_edges
    child_d = child.effective_q_d_edges
    c_map = np.searchsorted(parent_c, child_c[:-1], side="right") - 1
    d_map = np.searchsorted(parent_d, child_d[:-1], side="right") - 1
    c_map = np.maximum(c_map, 0)
    d_map = np.maximum(d_map, 0)
    diagnostics = {
        "formula": (
            "gamma_k=k*machine_epsilon/(1-k*machine_epsilon), a conservative "
            "substitution for unit roundoff; represented term bounds use the "
            "observation dtype, while parent/child accumulation and regrouping "
            "use the sufficient-array summation dtype"
        ),
        "observation_dtype": np.dtype(parent.observation_dtype).name,
        "maximum_parent_observations": int(np.max(parent.counts, initial=0)),
        "fields": {},
    }
    field_dtypes = {}
    field_bounds = {}
    for name in SUFFICIENT_SUM_FIELDS:
        parent_dtype = np.asarray(getattr(parent, name)).dtype
        child_dtype = np.asarray(getattr(child, name)).dtype
        if parent_dtype != child_dtype or not np.issubdtype(parent_dtype, np.floating):
            raise TypeError(f"{name} parent/child arrays need one common floating dtype")
        field_dtypes[name] = parent_dtype
        field_bounds[name] = _absolute_term_bounds(parent, parent_dtype)[name]
        diagnostics["fields"][name] = {
            "aggregation_dtype": parent_dtype.name,
            "machine_epsilon": float(np.finfo(parent_dtype).eps),
            "maximum_absolute_difference": 0.0,
            "maximum_allowed_roundoff": 0.0,
        }
    for pc in range(parent.bins.num_q_c_bins):
        child_cs = np.flatnonzero(c_map == pc)
        for pd in range(parent.bins.num_q_d_bins):
            child_ds = np.flatnonzero(d_map == pd)
            child_index = np.ix_(
                np.arange(child.counts.shape[0]),
                np.arange(child.counts.shape[1]),
                child_cs,
                child_ds,
                np.arange(2),
            )
            reconstructed_counts = child.counts[child_index].sum(axis=(2, 3))
            np.testing.assert_array_equal(
                reconstructed_counts,
                parent.counts[:, :, pc, pd, :],
            )
            for name in SUFFICIENT_SUM_FIELDS:
                child_values = getattr(child, name)[child_index]
                parent_values = getattr(parent, name)[:, :, pc, pd, :]
                reconstructed = child_values.sum(axis=(2, 3))
                allowed = _reconstruction_bound(
                    parent_values,
                    child_values,
                    parent.counts[:, :, pc, pd, :],
                    child.counts[child_index],
                    term_bound=field_bounds[name],
                    dtype=field_dtypes[name],
                )
                difference = np.abs(reconstructed - parent_values)
                field_diagnostic = diagnostics["fields"][name]
                field_diagnostic["maximum_absolute_difference"] = max(
                    field_diagnostic["maximum_absolute_difference"],
                    float(np.max(difference, initial=0.0)),
                )
                field_diagnostic["maximum_allowed_roundoff"] = max(
                    field_diagnostic["maximum_allowed_roundoff"],
                    float(np.max(allowed, initial=0.0)),
                )
                if np.any(difference > allowed):
                    excess = difference - allowed
                    failing = np.unravel_index(np.argmax(excess), excess.shape)
                    raise AssertionError(
                        f"{name} child reconstruction exceeds its roundoff bound: "
                        f"difference={difference[failing]!r}, "
                        f"allowed={allowed[failing]!r}"
                    )
    return diagnostics


def anchor_bin_index(
    bins: QBinSpec,
    anchor: tuple[float, float],
    observation_dtype,
) -> tuple[int, int]:
    """Map an anchor with the same boundary convention as observations."""

    point = np.asarray(anchor, dtype=np.dtype(observation_dtype))
    if point.shape != (2,):
        raise ValueError("anchor must contain exactly two Q coordinates")
    q_c, q_d = bins.assign(point[None, :])
    return int(q_c[0]), int(q_d[0])

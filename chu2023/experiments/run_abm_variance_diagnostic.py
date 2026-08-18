#!/usr/bin/env python3
"""Run the small Phase 3A ABM variance diagnostic and write binned moments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from chu_pair.abm import (
    QBinSpec,
    aggregate_variance_records,
    complete_graph,
    derive_variance_moments,
    initialize_continuous_paper_batch,
    initialize_grid_matched_batch,
    simulate_instrumented_batch_jit,
)
from chu_pair.config import ABMConfig, LearningConfig
from chu_pair.model import Action

if __package__:
    from experiments import run_abm_baseline as baseline
else:  # Direct ``python experiments/...py`` execution.
    import run_abm_baseline as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "abm_variance_diagnostic_small.toml"
PHASE3A_STATISTICS_ABSOLUTE_LIMITS = {
    "stratum_count": 1_000_000,
    "estimated_peak_statistic_bytes": 256 * 1024**2,
    "output_rows": 250_000,
}
ACTION_COUNT = 2
SUFFICIENT_FLOAT64_ARRAY_COUNT = 10
DERIVED_FLOAT64_ARRAY_COUNT = 14
DERIVED_BOOL_ARRAY_COUNT = 4
DERIVED_INTERMEDIATE_FLOAT64_ARRAY_COUNT = 4
DERIVED_EXPRESSION_WORKSPACE_FLOAT64_ARRAY_COUNT = 3
AGGREGATION_FLOAT64_CONVERSION_ARRAY_COUNT = 5
AGGREGATION_FLOAT64_PRODUCT_ARRAY_COUNT = 5
AGGREGATION_INDEX_ARRAY_COUNT = 5
INSTRUMENTED_OBSERVATION_FLOAT_WIDTH = 7

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
ESTIMATE_FIELDS = (
    "mu",
    "m2",
    "m11",
    "sigma2",
    "covariance",
    "mean_reward",
    "direct_reward_variance",
    "decomposed_reward_variance",
    "mean_selected_q",
    "selected_q_variance",
    "reward_selected_q_covariance",
    "mean_velocity",
    "direct_velocity_variance",
    "finite_bin_velocity_variance",
)


def raw_bin_counts(bin_config: dict) -> tuple[int, int]:
    """Inspect raw parsed edge sequences without constructing NumPy arrays."""

    counts = []
    for name in ("q_c_edges", "q_d_edges"):
        if name not in bin_config:
            raise ValueError(f"bins.{name} is required")
        edges = bin_config[name]
        if isinstance(edges, (str, bytes)):
            raise ValueError(f"bins.{name} must be a sequence of at least two edges")
        try:
            edge_count = len(edges)
        except TypeError as error:
            raise ValueError(
                f"bins.{name} must be a sequence of at least two edges"
            ) from error
        if edge_count < 2:
            raise ValueError(f"bins.{name} must contain at least two edges")
        counts.append(int(edge_count - 1))
    return counts[0], counts[1]


def estimate_statistics_resources(
    abm: ABMConfig,
    *,
    q_c_bins: int,
    q_d_bins: int,
) -> dict:
    """Estimate dense Phase 3A statistic work using allocation-free integers."""

    runs = int(abm.num_runs)
    steps = int(abm.steps)
    agents = int(abm.num_agents)
    for name, value in (("q_c_bins", q_c_bins), ("q_d_bins", q_d_bins)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    stratum_count = runs * steps * q_c_bins * q_d_bins * ACTION_COUNT
    observation_count = runs * steps * agents
    output_rows = stratum_count

    int64_bytes = int(np.dtype(np.int64).itemsize)
    float64_bytes = int(np.dtype(np.float64).itemsize)
    bool_bytes = int(np.dtype(np.bool_).itemsize)
    index_bytes = int(np.dtype(np.intp).itemsize)
    observation_dtype = np.dtype(abm.dtype)
    if observation_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("statistics observations must use float32 or float64")
    observation_float_bytes = int(observation_dtype.itemsize)
    edge_count = q_c_bins + q_d_bins + 2
    configured_edge_bytes = edge_count * float64_bytes
    effective_edge_bytes = edge_count * observation_float_bytes

    sufficient_count_bytes = stratum_count * int64_bytes
    sufficient_sum_bytes = (
        stratum_count * SUFFICIENT_FLOAT64_ARRAY_COUNT * float64_bytes
    )
    retained_sufficient_bytes = sufficient_count_bytes + sufficient_sum_bytes

    # aggregate_variance_records can simultaneously retain the sufficient
    # arrays, host representations of the seven floating/int8 observation
    # fields, five flattened integer index arrays, and five float64 product
    # arrays.  The five base-value conversions allocate only for float32 input;
    # they alias the already counted host observation arrays for float64 input.
    # Broadcast index views and the streaming CSV row are O(1) in strata and
    # require no additional scalable allowance.
    aggregation_observation_bytes = observation_count * (
        INSTRUMENTED_OBSERVATION_FLOAT_WIDTH * observation_float_bytes + 1
    )
    aggregation_index_bytes = (
        observation_count * AGGREGATION_INDEX_ARRAY_COUNT * index_bytes
    )
    aggregation_conversion_bytes = (
        observation_count
        * AGGREGATION_FLOAT64_CONVERSION_ARRAY_COUNT
        * float64_bytes
        if observation_dtype == np.dtype(np.float32)
        else 0
    )
    aggregation_product_bytes = (
        observation_count
        * AGGREGATION_FLOAT64_PRODUCT_ARRAY_COUNT
        * float64_bytes
    )
    aggregation_value_bytes = (
        aggregation_conversion_bytes + aggregation_product_bytes
    )
    aggregation_peak_bytes = (
        configured_edge_bytes
        + effective_edge_bytes
        + retained_sufficient_bytes
        + aggregation_observation_bytes
        + aggregation_index_bytes
        + aggregation_value_bytes
    )

    # derive_variance_moments retains 14 returned float64 arrays, four validity
    # masks, and four non-returned mean-square/cross-moment arrays alongside the
    # sufficient arrays.  Three float64 arrays conservatively cover the largest
    # NumPy expression workspace.  CSV export streams one row at a time, so its
    # stratum-scaled live set (sufficient + returned arrays) is smaller.
    derived_retained_bytes = stratum_count * (
        DERIVED_FLOAT64_ARRAY_COUNT * float64_bytes
        + DERIVED_BOOL_ARRAY_COUNT * bool_bytes
    )
    derived_intermediate_bytes = (
        stratum_count
        * DERIVED_INTERMEDIATE_FLOAT64_ARRAY_COUNT
        * float64_bytes
    )
    derived_expression_workspace_bytes = (
        stratum_count
        * DERIVED_EXPRESSION_WORKSPACE_FLOAT64_ARRAY_COUNT
        * float64_bytes
    )
    derivation_peak_bytes = (
        configured_edge_bytes
        + effective_edge_bytes
        + retained_sufficient_bytes
        + derived_retained_bytes
        + derived_intermediate_bytes
        + derived_expression_workspace_bytes
    )
    estimated_peak_statistic_bytes = max(
        aggregation_peak_bytes, derivation_peak_bytes
    )

    return {
        "num_runs": runs,
        "steps": steps,
        "num_agents": agents,
        "q_c_bins": q_c_bins,
        "q_d_bins": q_d_bins,
        "action_count": ACTION_COUNT,
        "observation_count": observation_count,
        "stratum_count": stratum_count,
        "output_rows": output_rows,
        "estimated_peak_statistic_bytes": estimated_peak_statistic_bytes,
        "components": {
            "configured_float64_edge_bytes": configured_edge_bytes,
            "effective_observation_dtype_edge_bytes": effective_edge_bytes,
            "sufficient_count_int64_bytes": sufficient_count_bytes,
            "sufficient_sum_float64_bytes": sufficient_sum_bytes,
            "aggregation_observation_bytes": aggregation_observation_bytes,
            "aggregation_index_intp_bytes": aggregation_index_bytes,
            "aggregation_float64_conversion_bytes": aggregation_conversion_bytes,
            "aggregation_float64_product_bytes": aggregation_product_bytes,
            "aggregation_value_float64_bytes": aggregation_value_bytes,
            "aggregation_peak_bytes": aggregation_peak_bytes,
            "derived_retained_float64_and_bool_bytes": derived_retained_bytes,
            "derived_intermediate_float64_bytes": derived_intermediate_bytes,
            "derived_expression_workspace_float64_bytes": (
                derived_expression_workspace_bytes
            ),
            "derivation_peak_bytes": derivation_peak_bytes,
            "csv_export_additional_stratum_scaled_bytes": 0,
        },
    }


def validate_statistics_budget(
    abm: ABMConfig,
    *,
    q_c_bins: int,
    q_d_bins: int,
    allow_expensive: bool,
    limits: dict[str, int] | None = None,
) -> dict:
    """Reject dense Phase 3A statistics before simulation or allocation."""

    estimates = estimate_statistics_resources(
        abm,
        q_c_bins=q_c_bins,
        q_d_bins=q_d_bins,
    )
    effective_limits = dict(
        PHASE3A_STATISTICS_ABSOLUTE_LIMITS if limits is None else limits
    )
    if set(effective_limits) != set(PHASE3A_STATISTICS_ABSOLUTE_LIMITS):
        raise ValueError("statistics limits must define all three resource caps")
    for name, value in effective_limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"statistics limit {name} must be a non-negative integer")

    violations = [
        name
        for name in PHASE3A_STATISTICS_ABSOLUTE_LIMITS
        if estimates[name] > effective_limits[name]
    ]
    if violations and not allow_expensive:
        resources = "; ".join(
            f"{name}={estimates[name]:,} (limit {effective_limits[name]:,})"
            for name in PHASE3A_STATISTICS_ABSOLUTE_LIMITS
        )
        raise ValueError(
            "refusing expensive Phase 3A dense statistics before simulation or "
            f"allocation: Bc={estimates['q_c_bins']:,}, "
            f"Bd={estimates['q_d_bins']:,}, A={ACTION_COUNT}, "
            f"R={estimates['num_runs']:,}, T={estimates['steps']:,}; {resources}; "
            f"violations={violations}; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "estimates": estimates,
        "absolute_limits": effective_limits,
        "violations_overridden": violations if allow_expensive else [],
    }


def construct_guarded_bins(
    abm: ABMConfig,
    bin_config: dict,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> tuple[QBinSpec, dict]:
    """Preflight raw bin counts, then allocate and fully validate bin edges."""

    q_c_bins, q_d_bins = raw_bin_counts(bin_config)
    budget = validate_statistics_budget(
        abm,
        q_c_bins=q_c_bins,
        q_d_bins=q_d_bins,
        allow_expensive=allow_expensive,
        limits=limits,
    )
    bins = QBinSpec(bin_config["q_c_edges"], bin_config["q_d_edges"])
    if bins.bin_shape != (q_c_bins, q_d_bins):
        raise RuntimeError(
            "QBinSpec bin counts disagree with the allocation-free preflight"
        )
    bins.effective_edges(np.dtype(abm.dtype))
    return bins, budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="override the conservative Phase 2/3A resource guardrails",
    )
    return parser.parse_args()


def _value_or_missing(array: np.ndarray, index: tuple[int, ...], defined: bool):
    if not defined:
        return ""
    value = array[index]
    return "" if not np.isfinite(value) else float(value)


def write_moment_csv(path: Path, statistics, estimates) -> int:
    """Write every stratum, using blanks rather than numerical undefined values."""

    bins = statistics.bins
    rows_written = 0
    header = [
        "run_id",
        "source_time_t",
        "q_c_bin",
        "q_d_bin",
        "action_index",
        "action",
        "q_c_configured_lower",
        "q_c_effective_lower",
        "q_c_lower_inclusive",
        "q_c_configured_upper",
        "q_c_effective_upper",
        "q_c_upper_inclusive",
        "q_d_configured_lower",
        "q_d_effective_lower",
        "q_d_lower_inclusive",
        "q_d_configured_upper",
        "q_d_effective_upper",
        "q_d_upper_inclusive",
        "count",
        "has_observations",
        "underpopulated",
        "meets_min_count",
        "distinct_covariance_defined",
        *SUFFICIENT_SUM_FIELDS,
        *ESTIMATE_FIELDS,
    ]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        runs, steps = statistics.counts.shape[:2]
        for run in range(runs):
            for source_time in range(steps):
                for q_c_bin in range(bins.num_q_c_bins):
                    for q_d_bin in range(bins.num_q_d_bins):
                        for action in range(2):
                            index = (run, source_time, q_c_bin, q_d_bin, action)
                            has_observations = bool(estimates.has_observations[index])
                            distinct_defined = bool(
                                estimates.distinct_covariance_defined[index]
                            )
                            sums = [
                                _value_or_missing(
                                    np.asarray(getattr(statistics, name)),
                                    index,
                                    has_observations,
                                )
                                for name in SUFFICIENT_SUM_FIELDS
                            ]
                            moment_values = []
                            for name in ESTIMATE_FIELDS:
                                defined = has_observations
                                if name in {"m11", "covariance"}:
                                    defined = distinct_defined
                                moment_values.append(
                                    _value_or_missing(
                                        np.asarray(getattr(estimates, name)), index, defined
                                    )
                                )
                            writer.writerow(
                                [
                                    run,
                                    source_time,
                                    q_c_bin,
                                    q_d_bin,
                                    action,
                                    Action(action).name,
                                    float(bins.q_c_edges[q_c_bin]),
                                    float(statistics.effective_q_c_edges[q_c_bin]),
                                    True,
                                    float(bins.q_c_edges[q_c_bin + 1]),
                                    float(statistics.effective_q_c_edges[q_c_bin + 1]),
                                    q_c_bin == bins.num_q_c_bins - 1,
                                    float(bins.q_d_edges[q_d_bin]),
                                    float(statistics.effective_q_d_edges[q_d_bin]),
                                    True,
                                    float(bins.q_d_edges[q_d_bin + 1]),
                                    float(statistics.effective_q_d_edges[q_d_bin + 1]),
                                    q_d_bin == bins.num_q_d_bins - 1,
                                    int(statistics.counts[index]),
                                    has_observations,
                                    bool(estimates.underpopulated[index]),
                                    bool(estimates.meets_min_count[index]),
                                    distinct_defined,
                                    *sums,
                                    *moment_values,
                                ]
                            )
                            rows_written += 1
    return rows_written


def write_metadata(
    path: Path,
    *,
    config_path: Path,
    config: dict,
    initialization,
    result,
    statistics,
    rows_written: int,
    resource_budget: dict,
) -> None:
    source_hashes = baseline.implementation_source_hashes(config_path)
    source_hashes["experiments/run_abm_variance_diagnostic.py"] = baseline.sha256(
        Path(__file__).resolve()
    )
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "resource_budget": resource_budget,
        "initialization": {
            field: getattr(initialization.metadata, field)
            for field in initialization.metadata.__dataclass_fields__
        },
        "time_convention": (
            "row source_time_t uses Q_t, S_t, the selected action at t, and the "
            "reward/velocity that produce Q_{t+1}; no destination-Q conditioning"
        ),
        "conditioning": "independent run x source time x 2D source-Q bin x selected action",
        "bin_boundary_convention": (
            "[lower, upper) on each coordinate, except the final bin includes its upper edge; "
            "configured edges are retained as float64 provenance while comparisons use edges "
            "converted to the observation Q dtype; out-of-range observations raise instead "
            "of being clipped or discarded"
        ),
        "bin_edges": {
            "observation_dtype": statistics.observation_dtype,
            "configured_float64": {
                "q_c": statistics.bins.q_c_edges.tolist(),
                "q_d": statistics.bins.q_d_edges.tolist(),
            },
            "effective_comparison": {
                "q_c": statistics.effective_q_c_edges.tolist(),
                "q_d": statistics.effective_q_d_edges.tolist(),
            },
        },
        "population_moment_convention": "ddof=0 identities within each stratum",
        "uncertainty_convention": (
            "run is retained as the independent-replicate axis; agents are not uncertainty replicates"
        ),
        "formulas": {
            "mu": "sum(S1)/(count*N)",
            "m2": "sum(S2)/(count*N)",
            "m11": "sum(S1**2-S2)/(count*N*(N-1)) for N>1",
            "sigma2": "m2-mu**2",
            "covariance": "m11-mu**2 for N>1",
            "reward_variance_decomposition": "sigma2/N + (N-1)*covariance/N",
            "finite_bin_velocity_variance": (
                "alpha**2*(Var(reward)+Var(selected_q)-2*Cov(reward,selected_q))"
            ),
        },
        "missing_value_convention": (
            "CSV empty fields mean undefined/empty; validity and count columns distinguish causes"
        ),
        "num_opponents": statistics.num_opponents,
        "rows_written": rows_written,
        "retained_shapes": {
            field: list(getattr(result.records, field).shape)
            for field in result.records._fields
        },
        "backend": jax.default_backend(),
        "devices": baseline.device_metadata(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
        },
        "git": {
            "commit": baseline.git_text("rev-parse", "HEAD"),
            "subproject_status_before_output": baseline.git_text(
                "status", "--short", "--", "."
            ),
        },
        "source_hashes": source_hashes,
    }
    with path.open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = baseline.load_config(config_path)
    learning = LearningConfig(**config["model"])
    abm = ABMConfig(**config["simulation"])
    resource_budget = baseline.validate_resource_budget(
        abm,
        config["safety"],
        args.allow_expensive,
        record_mode=baseline.RESOURCE_MODE_INSTRUMENTED,
    )
    dtype = jnp.float32 if abm.dtype == "float32" else jnp.float64
    bin_config = config["bins"]
    min_count = bin_config["min_count"]
    bins, statistics_budget = construct_guarded_bins(
        abm,
        bin_config,
        args.allow_expensive,
    )
    resource_budget["phase3a_statistics"] = statistics_budget
    graph = complete_graph(abm.num_agents)

    initial = config["initial_condition"]
    if initial["mode"] == "grid_matched":
        histogram, histogram_budget = baseline.construct_grid_matched_histogram(
            initial, args.allow_expensive
        )
        resource_budget["grid_matched_histogram"] = histogram_budget
        initialization = initialize_grid_matched_batch(
            graph,
            histogram,
            abm_seed=abm.abm_seed,
            num_runs=abm.num_runs,
            dtype=dtype,
        )
    elif initial["mode"] == "continuous_paper":
        resource_budget["grid_matched_histogram"] = {"mode": "not_applicable"}
        initialization = initialize_continuous_paper_batch(
            graph,
            abm_seed=abm.abm_seed,
            num_runs=abm.num_runs,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unsupported initial_condition.mode: {initial['mode']!r}")

    result = simulate_instrumented_batch_jit(
        initialization.state,
        initialization.simulation_key,
        graph,
        learning.alpha,
        learning.tau,
        steps=abm.steps,
    )
    result.final_state.q_values.block_until_ready()

    statistics = aggregate_variance_records(
        result.records,
        bins,
        num_agents=abm.num_agents,
        alpha=learning.alpha,
        min_count=min_count,
    )
    estimates = derive_variance_moments(statistics)

    run_name = config["output"]["run_name"]
    if not baseline.RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("output.run_name contains unsupported path characters")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = (PROJECT_ROOT / "outputs" / "abm_variance").resolve()
    run_directory = (output_root / f"{run_name}-{timestamp}").resolve()
    if output_root not in run_directory.parents:
        raise ValueError("output directory must remain beneath outputs/abm_variance")
    run_directory.mkdir(parents=True, exist_ok=False)

    rows_written = write_moment_csv(
        run_directory / "binned_moments.csv", statistics, estimates
    )
    write_metadata(
        run_directory / "metadata.json",
        config_path=config_path,
        config=config,
        initialization=initialization,
        result=result,
        statistics=statistics,
        rows_written=rows_written,
        resource_budget=resource_budget,
    )
    populated = int(np.count_nonzero(statistics.counts))
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        f"wrote {rows_written} strata ({populated} populated) to {run_directory}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the bounded Phase 3B run-cluster bootstrap/refinement smoke diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from chu_pair.abm import (
    BOOTSTRAP_ESTIMANDS,
    MIN_CONTRIBUTING_RUNS,
    MIN_VALID_BOOTSTRAP_FRACTION,
    QUANTILE_METHOD,
    NamedBinScheme,
    QBinSpec,
    aggregate_variance_records,
    anchor_bin_index,
    assert_child_reconstructs_parent,
    bootstrap_run_weights,
    cluster_bootstrap_intervals,
    complete_graph,
    initialize_continuous_paper_batch,
    initialize_grid_matched_batch,
    simulate_instrumented_batch_jit,
    validate_nested_schemes,
)
from chu_pair.config import ABMConfig, LearningConfig
from chu_pair.model import Action

if __package__:
    from experiments import run_abm_baseline as baseline
else:
    import run_abm_baseline as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "abm_uncertainty_smoke.toml"
SCHEME_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PHASE3B_ABSOLUTE_LIMITS = {
    "total_per_run_strata": 1_000_000,
    "estimated_peak_bytes": 256 * 1024**2,
    "pooled_output_rows": 250_000,
    "bootstrap_weight_bytes": 64 * 1024**2,
    "bootstrap_working_bytes": 256 * 1024**2,
    "anchor_output_rows": 100_000,
}
SUFFICIENT_BYTES_PER_STRATUM = 8 + 10 * 8
SUMMARY_BYTES_PER_CELL = 2 * 8 + len(BOOTSTRAP_ESTIMANDS) * (3 * 8 + 2 * 4 + 1)
BOOTSTRAP_WORK_BYTES_PER_REPLICATE_CELL = 280
POOLED_POINT_DERIVATION_BYTES_PER_CELL = 260
RECONSTRUCTION_WORK_BYTES_PER_CHILD_STRATUM = 112
METADATA_PYTHON_BYTES_PER_EFFECTIVE_EDGE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="override fixed Phase 2/3B resource caps",
    )
    return parser.parse_args()


def inspect_raw_refinement_config(config: dict) -> dict:
    """Obtain names/counts/bounds from parsed lists without NumPy edge copies."""

    raw_schemes = config.get("bin_schemes")
    if not isinstance(raw_schemes, list) or not raw_schemes:
        raise ValueError("bin_schemes must be a non-empty array of tables")
    schemes = []
    names = set()
    outer = None
    for raw in raw_schemes:
        if not isinstance(raw, dict):
            raise ValueError("each bin_schemes entry must be a table")
        name = raw.get("name")
        if not isinstance(name, str) or not SCHEME_NAME_PATTERN.fullmatch(name):
            raise ValueError("each bin scheme needs a safe non-empty name")
        if name in names:
            raise ValueError(f"duplicate bin scheme name: {name!r}")
        names.add(name)
        counts = []
        bounds = []
        for key in ("q_c_edges", "q_d_edges"):
            edges = raw.get(key)
            if isinstance(edges, (str, bytes)):
                raise ValueError(f"scheme {name!r} {key} must be an edge sequence")
            try:
                edge_count = len(edges)
            except TypeError as error:
                raise ValueError(
                    f"scheme {name!r} {key} must be an edge sequence"
                ) from error
            if edge_count < 2:
                raise ValueError(f"scheme {name!r} {key} needs at least two edges")
            counts.append(int(edge_count - 1))
            bounds.append((edges[0], edges[-1]))
        if outer is None:
            outer = tuple(bounds)
        elif tuple(bounds) != outer:
            raise ValueError("all raw refinement schemes must have common outer bounds")
        schemes.append({"name": name, "q_c_bins": counts[0], "q_d_bins": counts[1]})

    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("bootstrap table is required")
    for key in ("replicates", "seed", "stratum_chunk_size"):
        value = bootstrap.get(key)
        minimum = 0 if key == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"bootstrap.{key} must be an integer at least {minimum}")
    confidence = bootstrap.get("confidence_level")
    if not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap.confidence_level must lie between zero and one")

    anchors = config.get("anchors", {}).get("points", [])
    if not isinstance(anchors, list):
        raise ValueError("anchors.points must be a list")
    for anchor in anchors:
        if not isinstance(anchor, list) or len(anchor) != 2:
            raise ValueError("each anchor must contain exactly two coordinates")
    return {"schemes": schemes, "bootstrap": bootstrap, "anchor_count": len(anchors)}


def estimate_phase3b_resources(abm: ABMConfig, raw: dict) -> dict:
    """Allocation-free Phase 3B estimate from raw bin counts and scalar config."""

    runs = int(abm.num_runs)
    steps = int(abm.steps)
    agents = int(abm.num_agents)
    dtype = np.dtype(abm.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Phase 3B observations must use float32 or float64")
    item_bytes = int(dtype.itemsize)
    replicates = int(raw["bootstrap"]["replicates"])
    chunk_size = int(raw["bootstrap"]["stratum_chunk_size"])
    observation_count = runs * steps * agents
    pooled_cells = [
        steps * scheme["q_c_bins"] * scheme["q_d_bins"] * 2
        for scheme in raw["schemes"]
    ]
    per_run_strata = [runs * cells for cells in pooled_cells]
    total_per_run_strata = sum(per_run_strata)
    pooled_output_rows = sum(pooled_cells)
    anchor_output_rows = raw["anchor_count"] * steps * 2 * len(raw["schemes"])
    edge_values = sum(
        scheme["q_c_bins"] + scheme["q_d_bins"] + 2
        for scheme in raw["schemes"]
    )
    edge_bytes = edge_values * (8 + item_bytes)
    metadata_effective_edge_python_bytes = (
        edge_values * METADATA_PYTHON_BYTES_PER_EFFECTIVE_EDGE
    )
    total_sufficient_bytes = total_per_run_strata * SUFFICIENT_BYTES_PER_STRATUM
    adjacent_strata = [per_run_strata[0]]
    adjacent_strata.extend(
        per_run_strata[index - 1] + per_run_strata[index]
        for index in range(1, len(per_run_strata))
    )
    peak_sequential_sufficient_bytes = (
        max(adjacent_strata) * SUFFICIENT_BYTES_PER_STRATUM
    )
    reconstruction_working_bytes = max(
        (
            strata * RECONSTRUCTION_WORK_BYTES_PER_CHILD_STRATUM
            for strata in per_run_strata[1:]
        ),
        default=0,
    )
    reconstruction_peak_sufficient_and_work_bytes = max(
        (
            (per_run_strata[index - 1] + per_run_strata[index])
            * SUFFICIENT_BYTES_PER_STRATUM
            + per_run_strata[index]
            * RECONSTRUCTION_WORK_BYTES_PER_CHILD_STRATUM
            for index in range(1, len(per_run_strata))
        ),
        default=per_run_strata[0] * SUFFICIENT_BYTES_PER_STRATUM,
    )

    host_observation_bytes = observation_count * (7 * item_bytes + 1)
    index_bytes = observation_count * 5 * int(np.dtype(np.intp).itemsize)
    conversion_bytes = observation_count * 5 * 8 if item_bytes == 4 else 0
    product_bytes = observation_count * 5 * 8
    aggregation_observation_work_bytes = (
        host_observation_bytes + index_bytes + conversion_bytes + product_bytes
    )
    bootstrap_weight_bytes = replicates * runs * int(np.dtype(np.int32).itemsize)
    bootstrap_weight_serialization_buffer_bytes = min(
        bootstrap_weight_bytes,
        16 * 1024**2,
    )
    bootstrap_weight_float64_bytes = replicates * runs * int(
        np.dtype(np.float64).itemsize
    )
    bootstrap_weight_generation_index_bytes = replicates * int(
        np.dtype(np.intp).itemsize
    )
    bootstrap_weight_generation_peak_bytes = (
        2 * bootstrap_weight_bytes + bootstrap_weight_generation_index_bytes
    )
    bootstrap_weight_processing_bytes = (
        bootstrap_weight_bytes + bootstrap_weight_float64_bytes
    )
    max_chunk_cells = min(chunk_size, max(pooled_cells, default=0))
    bootstrap_working_bytes = (
        replicates
        * max_chunk_cells
        * BOOTSTRAP_WORK_BYTES_PER_REPLICATE_CELL
    )
    retained_summary_bytes = sum(pooled_cells) * SUMMARY_BYTES_PER_CELL
    pooled_point_derivation_bytes = (
        max(pooled_cells, default=0) * POOLED_POINT_DERIVATION_BYTES_PER_CELL
    )
    aggregation_peak = (
        edge_bytes
        + bootstrap_weight_bytes
        + retained_summary_bytes
        + peak_sequential_sufficient_bytes
        + aggregation_observation_work_bytes
    )
    bootstrap_peak = (
        edge_bytes
        + bootstrap_weight_processing_bytes
        + retained_summary_bytes
        + peak_sequential_sufficient_bytes
        + bootstrap_working_bytes
    )
    pooled_point_peak = (
        edge_bytes
        + bootstrap_weight_bytes
        + retained_summary_bytes
        + peak_sequential_sufficient_bytes
        + pooled_point_derivation_bytes
    )
    reconstruction_peak = (
        edge_bytes
        + bootstrap_weight_bytes
        + retained_summary_bytes
        + reconstruction_peak_sufficient_and_work_bytes
    )
    output_peak = (
        edge_bytes
        + metadata_effective_edge_python_bytes
        + bootstrap_weight_bytes
        + bootstrap_weight_serialization_buffer_bytes
        + retained_summary_bytes
        + per_run_strata[-1] * SUFFICIENT_BYTES_PER_STRATUM
    )
    estimated_peak_bytes = max(
        edge_bytes + bootstrap_weight_generation_peak_bytes,
        aggregation_peak,
        pooled_point_peak,
        reconstruction_peak,
        bootstrap_peak,
        output_peak,
    )
    return {
        "num_runs": runs,
        "steps": steps,
        "num_agents": agents,
        "observation_count": observation_count,
        "scheme_count": len(raw["schemes"]),
        "scheme_pooled_cells": pooled_cells,
        "scheme_per_run_strata": per_run_strata,
        "total_per_run_strata": total_per_run_strata,
        "pooled_output_rows": pooled_output_rows,
        "anchor_output_rows": anchor_output_rows,
        "bootstrap_replicates": replicates,
        "stratum_chunk_size": chunk_size,
        "bootstrap_weight_bytes": bootstrap_weight_bytes,
        "bootstrap_working_bytes": bootstrap_working_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "components": {
            "configured_and_effective_edge_bytes": edge_bytes,
            "metadata_effective_edge_python_bytes": (
                metadata_effective_edge_python_bytes
            ),
            "total_sequential_sufficient_bytes": total_sufficient_bytes,
            "peak_sequential_sufficient_bytes": peak_sequential_sufficient_bytes,
            "reconstruction_working_bytes": reconstruction_working_bytes,
            "reconstruction_peak_sufficient_and_work_bytes": (
                reconstruction_peak_sufficient_and_work_bytes
            ),
            "aggregation_observation_work_bytes": aggregation_observation_work_bytes,
            "bootstrap_weight_generation_peak_bytes": (
                bootstrap_weight_generation_peak_bytes
            ),
            "bootstrap_weight_float64_conversion_bytes": (
                bootstrap_weight_float64_bytes
            ),
            "bootstrap_weight_serialization_buffer_bytes": (
                bootstrap_weight_serialization_buffer_bytes
            ),
            "bootstrap_weight_processing_bytes": bootstrap_weight_processing_bytes,
            "retained_summary_bytes": retained_summary_bytes,
            "pooled_point_derivation_bytes": pooled_point_derivation_bytes,
            "pooled_point_run_weight_bytes": 0,
            "aggregation_peak_bytes": aggregation_peak,
            "pooled_point_peak_bytes": pooled_point_peak,
            "reconstruction_peak_bytes": reconstruction_peak,
            "bootstrap_peak_bytes": bootstrap_peak,
            "output_peak_bytes": output_peak,
            "summary_bytes_per_pooled_cell": SUMMARY_BYTES_PER_CELL,
            "bootstrap_work_bytes_per_replicate_chunk_cell": (
                BOOTSTRAP_WORK_BYTES_PER_REPLICATE_CELL
            ),
            "pooled_point_derivation_bytes_per_cell": (
                POOLED_POINT_DERIVATION_BYTES_PER_CELL
            ),
            "reconstruction_work_bytes_per_child_stratum": (
                RECONSTRUCTION_WORK_BYTES_PER_CHILD_STRATUM
            ),
            "metadata_python_bytes_per_effective_edge": (
                METADATA_PYTHON_BYTES_PER_EFFECTIVE_EDGE
            ),
        },
    }


def validate_phase3b_budget(
    abm: ABMConfig,
    raw: dict,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> dict:
    estimates = estimate_phase3b_resources(abm, raw)
    effective_limits = dict(PHASE3B_ABSOLUTE_LIMITS if limits is None else limits)
    if set(effective_limits) != set(PHASE3B_ABSOLUTE_LIMITS):
        raise ValueError("Phase 3B limits must define every fixed resource cap")
    for name, value in effective_limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Phase 3B limit {name} must be a non-negative integer")
    violations = [
        name
        for name in PHASE3B_ABSOLUTE_LIMITS
        if estimates[name] > effective_limits[name]
    ]
    if violations and not allow_expensive:
        detail = "; ".join(
            f"{name}={estimates[name]:,} (limit {effective_limits[name]:,})"
            for name in PHASE3B_ABSOLUTE_LIMITS
        )
        raise ValueError(
            "refusing expensive Phase 3B work before QBinSpec, simulation, "
            f"aggregation, bootstrap, or output construction: {detail}; "
            f"violations={violations}; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "estimates": estimates,
        "absolute_limits": effective_limits,
        "violations_overridden": violations if allow_expensive else [],
    }


def construct_guarded_schemes(
    abm: ABMConfig,
    config: dict,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> tuple[list[NamedBinScheme], list[tuple[float, float]], dict, dict]:
    raw = inspect_raw_refinement_config(config)
    budget = validate_phase3b_budget(abm, raw, allow_expensive, limits=limits)
    schemes = [
        NamedBinScheme(
            scheme["name"],
            QBinSpec(scheme["q_c_edges"], scheme["q_d_edges"]),
        )
        for scheme in config["bin_schemes"]
    ]
    for raw_scheme, scheme in zip(raw["schemes"], schemes):
        if scheme.bins.bin_shape != (
            raw_scheme["q_c_bins"],
            raw_scheme["q_d_bins"],
        ):
            raise RuntimeError("constructed bin counts disagree with Phase 3B preflight")
    validate_nested_schemes(schemes, np.dtype(abm.dtype))
    anchors = [
        tuple(map(float, anchor))
        for anchor in config.get("anchors", {}).get("points", [])
    ]
    for scheme in schemes:
        for anchor in anchors:
            anchor_bin_index(scheme.bins, anchor, np.dtype(abm.dtype))
    return schemes, anchors, raw, budget


def write_pooled_csv(path: Path, results) -> int:
    header = [
        "scheme",
        "source_time_t",
        "q_c_bin",
        "q_d_bin",
        "action_index",
        "action",
        "total_count",
        "contributing_runs",
        "bootstrap_replicates",
    ]
    for name in BOOTSTRAP_ESTIMANDS:
        header.extend(
            [
                f"{name}_point",
                f"{name}_lower",
                f"{name}_upper",
                f"{name}_valid_replicates",
                f"{name}_invalid_replicates",
                f"{name}_interval_valid",
            ]
        )
    rows = 0
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for scheme, summary in results:
            for index in np.ndindex(summary.total_count.shape):
                time, q_c, q_d, action = index
                row = [
                    scheme.name,
                    time,
                    q_c,
                    q_d,
                    action,
                    Action(action).name,
                    int(summary.total_count[index]),
                    int(summary.contributing_runs[index]),
                    summary.bootstrap_replicates,
                ]
                for name in BOOTSTRAP_ESTIMANDS:
                    row.extend(
                        [
                            float(summary.point[name][index]),
                            float(summary.lower[name][index]),
                            float(summary.upper[name][index]),
                            int(summary.valid_replicates[name][index]),
                            int(summary.invalid_replicates[name][index]),
                            bool(summary.interval_valid[name][index]),
                        ]
                    )
                writer.writerow(row)
                rows += 1
    return rows


def write_anchor_csv(path: Path, results, anchors, observation_dtype) -> int:
    header = [
        "scheme",
        "anchor_id",
        "anchor_q_c",
        "anchor_q_d",
        "source_time_t",
        "action_index",
        "action",
        "q_c_bin",
        "q_d_bin",
        "q_c_configured_lower",
        "q_c_configured_upper",
        "q_d_configured_lower",
        "q_d_configured_upper",
        "q_c_configured_width",
        "q_d_configured_width",
        "q_c_effective_lower",
        "q_c_effective_upper",
        "q_d_effective_lower",
        "q_d_effective_upper",
        "q_c_effective_width",
        "q_d_effective_width",
        "total_count",
        "contributing_runs",
        "bootstrap_replicates",
    ]
    for name in BOOTSTRAP_ESTIMANDS:
        header.extend(
            [
                f"{name}_point",
                f"{name}_lower",
                f"{name}_upper",
                f"{name}_valid_replicates",
                f"{name}_invalid_replicates",
                f"{name}_interval_valid",
            ]
        )
    rows = 0
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for scheme, summary in results:
            effective_c, effective_d = scheme.bins.effective_edges(observation_dtype)
            for anchor_id, anchor in enumerate(anchors):
                q_c, q_d = anchor_bin_index(scheme.bins, anchor, observation_dtype)
                for time in range(summary.total_count.shape[0]):
                    for action in range(2):
                        index = (time, q_c, q_d, action)
                        row = [
                            scheme.name,
                            anchor_id,
                            anchor[0],
                            anchor[1],
                            time,
                            action,
                            Action(action).name,
                            q_c,
                            q_d,
                            float(scheme.bins.q_c_edges[q_c]),
                            float(scheme.bins.q_c_edges[q_c + 1]),
                            float(scheme.bins.q_d_edges[q_d]),
                            float(scheme.bins.q_d_edges[q_d + 1]),
                            float(
                                scheme.bins.q_c_edges[q_c + 1]
                                - scheme.bins.q_c_edges[q_c]
                            ),
                            float(
                                scheme.bins.q_d_edges[q_d + 1]
                                - scheme.bins.q_d_edges[q_d]
                            ),
                            float(effective_c[q_c]),
                            float(effective_c[q_c + 1]),
                            float(effective_d[q_d]),
                            float(effective_d[q_d + 1]),
                            float(effective_c[q_c + 1] - effective_c[q_c]),
                            float(effective_d[q_d + 1] - effective_d[q_d]),
                            int(summary.total_count[index]),
                            int(summary.contributing_runs[index]),
                            summary.bootstrap_replicates,
                        ]
                        for name in BOOTSTRAP_ESTIMANDS:
                            row.extend(
                                [
                                    float(summary.point[name][index]),
                                    float(summary.lower[name][index]),
                                    float(summary.upper[name][index]),
                                    int(summary.valid_replicates[name][index]),
                                    int(summary.invalid_replicates[name][index]),
                                    bool(summary.interval_valid[name][index]),
                                ]
                            )
                        writer.writerow(row)
                        rows += 1
    return rows


def write_metadata(
    path: Path,
    *,
    config_path: Path,
    config: dict,
    initialization,
    result,
    schemes,
    anchors,
    weights,
    resource_budget,
    reconstruction_diagnostics,
    pooled_rows: int,
    anchor_rows: int,
) -> None:
    source_hashes = baseline.implementation_source_hashes(config_path)
    source_hashes["experiments/run_abm_uncertainty_diagnostic.py"] = baseline.sha256(
        Path(__file__).resolve()
    )
    scheme_metadata = []
    observation_dtype = np.dtype(config["simulation"]["dtype"])
    for configured, scheme in zip(config["bin_schemes"], schemes):
        effective_q_c, effective_q_d = scheme.bins.effective_edges(
            observation_dtype
        )
        scheme_metadata.append(
            {
                "name": scheme.name,
                "configured_q_c_edges": configured["q_c_edges"],
                "configured_q_d_edges": configured["q_d_edges"],
                "effective_q_c_edges": effective_q_c.tolist(),
                "effective_q_d_edges": effective_q_d.tolist(),
            }
        )
    metadata = {
        "schema_version": 1,
        "milestone": "Phase 3B smoke/validation only; no coverage claim",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "resource_budget": resource_budget,
        "initialization": {
            field: getattr(initialization.metadata, field)
            for field in initialization.metadata.__dataclass_fields__
        },
        "pooled_estimand": (
            "sum per-run sufficient statistics across contributing independent runs, "
            "then apply nonlinear population-moment formulas; observation weighted"
        ),
        "resampling_unit": "one complete independent ABM run with common weights everywhere",
        "bootstrap": {
            "seed": config["bootstrap"]["seed"],
            "replicates": config["bootstrap"]["replicates"],
            "confidence_level": config["bootstrap"]["confidence_level"],
            "quantiles": "pointwise percentile interval",
            "numpy_method": QUANTILE_METHOD,
            "weight_shape": list(weights.shape),
            "weight_dtype": str(weights.dtype),
            "weight_sha256": hashlib.sha256(
                memoryview(weights).cast("B")
            ).hexdigest(),
            "stratum_chunk_size": config["bootstrap"]["stratum_chunk_size"],
        },
        "interval_validity": {
            "minimum_contributing_runs": MIN_CONTRIBUTING_RUNS,
            "minimum_valid_bootstrap_fraction": MIN_VALID_BOOTSTRAP_FRACTION,
            "minimum_valid_bootstrap_replicates": max(
                2,
                int(
                    np.ceil(
                        MIN_VALID_BOOTSTRAP_FRACTION
                        * config["bootstrap"]["replicates"]
                    )
                ),
            ),
            "undefined_endpoints": "NaN in memory and CSV; no fabricated interval",
        },
        "schemes": scheme_metadata,
        "refinement": "each successive scheme strictly refines both coordinates and reconstructs its parent sums",
        "reconstruction_roundoff": reconstruction_diagnostics,
        "anchors": [list(anchor) for anchor in anchors],
        "finite_bin_discrepancy": (
            "Var(v)-alpha^2 Var(reward) = alpha^2[Var(selected Q)-2 Cov(reward,selected Q)]"
        ),
        "pooled_rows": pooled_rows,
        "anchor_rows": anchor_rows,
        "retained_record_shapes": {
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
            "subproject_status_before_output": baseline.git_text("status", "--short", "--", "."),
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
    schemes, anchors, raw, phase3b_budget = construct_guarded_schemes(
        abm, config, args.allow_expensive
    )
    resource_budget["phase3b"] = phase3b_budget
    dtype = jnp.float32 if abm.dtype == "float32" else jnp.float64
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
    weights = bootstrap_run_weights(
        abm.num_runs,
        raw["bootstrap"]["replicates"],
        raw["bootstrap"]["seed"],
    )

    results = []
    previous_statistics = None
    previous_scheme = None
    reconstruction_diagnostics = []
    for scheme in schemes:
        statistics = aggregate_variance_records(
            result.records,
            scheme.bins,
            num_agents=abm.num_agents,
            alpha=learning.alpha,
            min_count=1,
        )
        if previous_statistics is not None:
            diagnostic = assert_child_reconstructs_parent(
                previous_statistics,
                statistics,
            )
            reconstruction_diagnostics.append(
                {
                    "parent_scheme": previous_scheme.name,
                    "child_scheme": scheme.name,
                    **diagnostic,
                }
            )
        summary = cluster_bootstrap_intervals(
            statistics,
            weights,
            confidence_level=raw["bootstrap"]["confidence_level"],
            stratum_chunk_size=raw["bootstrap"]["stratum_chunk_size"],
        )
        results.append((scheme, summary))
        previous_statistics = statistics
        previous_scheme = scheme

    run_name = config["output"]["run_name"]
    if not baseline.RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("output.run_name contains unsupported path characters")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = (PROJECT_ROOT / "outputs" / "abm_uncertainty").resolve()
    run_directory = (output_root / f"{run_name}-{timestamp}").resolve()
    if output_root not in run_directory.parents:
        raise ValueError("output directory must remain beneath outputs/abm_uncertainty")
    run_directory.mkdir(parents=True, exist_ok=False)
    pooled_rows = write_pooled_csv(run_directory / "pooled_intervals.csv", results)
    anchor_rows = write_anchor_csv(
        run_directory / "anchor_refinement.csv",
        results,
        anchors,
        np.dtype(abm.dtype),
    )
    np.savez_compressed(run_directory / "bootstrap_run_weights.npz", weights=weights)
    write_metadata(
        run_directory / "metadata.json",
        config_path=config_path,
        config=config,
        initialization=initialization,
        result=result,
        schemes=schemes,
        anchors=anchors,
        weights=weights,
        resource_budget=resource_budget,
        reconstruction_diagnostics=reconstruction_diagnostics,
        pooled_rows=pooled_rows,
        anchor_rows=anchor_rows,
    )
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        f"wrote {pooled_rows} pooled rows and {anchor_rows} anchor rows to {run_directory}"
    )


if __name__ == "__main__":
    main()

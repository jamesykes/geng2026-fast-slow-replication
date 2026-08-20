#!/usr/bin/env python3
"""Run the bounded Phase 5 matched pair-versus-ABM variance comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from chu_pair.abm import (
    MIN_CONTRIBUTING_RUNS,
    MIN_VALID_BOOTSTRAP_FRACTION,
    QUANTILE_METHOD,
    aggregate_variance_records,
    anchor_bin_index,
    assert_child_reconstructs_parent,
    bootstrap_run_weights,
    complete_graph,
    initialize_grid_matched_batch,
    simulate_instrumented_batch_jit,
)
from chu_pair.config import ABMConfig, LearningConfig
from chu_pair.grids import QGrid
from chu_pair.initial_conditions import seeded_legacy_histogram
from chu_pair.pair_density import (
    build_jax_pair_grid,
    execute_compiled_pair_source_summaries,
    ordered_pair_mass_jax,
    simulate_pair_source_summaries_jit,
    validate_jax_pair_mass,
)
from chu_pair.velocity_variance import (
    COMPARISON_BOOTSTRAP_ESTIMANDS,
    aggregate_pair_points,
    bootstrap_four_way_intervals,
    coarsen_abm_sufficient,
    coarsen_pair_sufficient,
    compare_four_way,
    derive_pair_binned_moments,
    pair_point_sufficient_from_jax_summary,
    select_abm_source_times,
)

from experiments import run_abm_baseline as baseline
from experiments import run_abm_uncertainty_diagnostic as phase3b
from experiments import run_pair_jax_small as phase4


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "velocity_variance_comparison_small.toml"
EXPECTED_SECTIONS = {
    "model",
    "simulation",
    "initial_condition",
    "pair_solver",
    "comparison",
    "bin_schemes",
    "bootstrap",
    "anchors",
    "output",
    "safety",
}
EXPECTED_KEYS = {
    "model": {"alpha", "tau"},
    "simulation": {"num_agents", "steps", "num_runs", "abm_seed", "dtype"},
    "initial_condition": {
        "mode", "histogram_seed", "q_min", "q_max", "spacing",
        "samples_per_grid_cell", "state_probabilities",
    },
    "pair_solver": {"chunk_size", "symmetry_tolerance", "diagnostic_tolerance"},
    "comparison": {"source_times", "minimum_count", "ratio_epsilon"},
    "bootstrap": {"replicates", "confidence_level", "seed", "stratum_chunk_size"},
    "anchors": {"points"},
    "output": {"run_name"},
    "safety": set(baseline.SAFETY_CONFIG_KEYS.values()),
}
PHASE5_ABSOLUTE_LIMITS = {
    "source_times": 64,
    "comparison_rows": 250_000,
    "pair_point_host_bytes": 64 * 1024**2,
    "comparison_peak_bytes": 256 * 1024**2,
}
PAIR_POINT_FLOATS_PER_POINT_ACTION = 7
PAIR_POINT_FOCAL_FLOATS_PER_POINT = 1
PAIR_BINNED_ACTION_BYTES_PER_CELL = 56
PAIR_BINNED_FOCAL_BYTES_PER_Q_CELL = 8
COMPARISON_RETAINED_BYTES_PER_CELL = 512
COMPARISON_BOOTSTRAP_BYTES_PER_REPLICATE_CELL = 64
COMPARISON_METADATA_MAX_CHARS = 256 * 1024
COMPARISON_SERIALIZATION_PEAK_BYTES = (
    3
    * phase4.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
    * COMPARISON_METADATA_MAX_CHARS
    + phase4.SERIALIZATION_FIXED_OVERHEAD_BYTES
)
PHASE5_MAX_LIVE_ROW_BYTES = 16 * 1024
PHASE5_MAX_CSV_WRITE_CHARS = 8 * 1024
PHASE5_EXECUTABLE_ID = "simulate_pair_source_summaries_jit:v1"


@dataclass(frozen=True, slots=True)
class CompiledPairExecutableBundle:
    """Runtime-only guarded executable; the compiled callable is never serialized."""

    compiled_callable: object
    memory_report: dict
    compile_signature: dict
    abstract_arguments: dict
    static_values: dict
    runtime_environment: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-expensive", action="store_true")
    return parser.parse_args()


def _finite(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def inspect_comparison_config(config: dict) -> dict:
    """Validate bounded scalars/lists and derive counts without NumPy allocation."""

    if not isinstance(config, dict) or set(config) != EXPECTED_SECTIONS:
        raise ValueError("configuration sections must exactly match the Phase 5 schema")
    for section, expected in EXPECTED_KEYS.items():
        value = config[section]
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(f"{section} keys must exactly match the Phase 5 schema")
    schemes = config["bin_schemes"]
    if not isinstance(schemes, list) or any(
        not isinstance(scheme, dict)
        or set(scheme) != {"name", "q_c_edges", "q_d_edges"}
        for scheme in schemes
    ):
        raise ValueError("each bin_schemes entry must exactly match the Phase 5 schema")
    learning = LearningConfig(**config["model"])
    abm = ABMConfig(**config["simulation"])
    initial = config["initial_condition"]
    if initial.get("mode") != "grid_matched":
        raise ValueError("Phase 5 requires grid_matched initialization")
    state_probabilities = initial["state_probabilities"]
    if (
        not isinstance(state_probabilities, list)
        or len(state_probabilities) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in state_probabilities)
        or any(not math.isfinite(float(value)) or value < 0 for value in state_probabilities)
        or not math.isclose(sum(state_probabilities), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("state_probabilities must be two non-negative values summing to one")
    if [float(value) for value in state_probabilities] != [0.5, 0.5]:
        raise ValueError("Phase 5 matched initialization requires uniform edge states")
    for key in ("histogram_seed", "samples_per_grid_cell"):
        value = initial[key]
        minimum = 0 if key == "histogram_seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"initial_condition.{key} must be an integer at least {minimum}")

    pair_solver = config["pair_solver"]
    if set(pair_solver) != {"chunk_size", "symmetry_tolerance", "diagnostic_tolerance"}:
        raise ValueError("pair_solver keys must exactly match the Phase 5 schema")
    chunk_size = pair_solver["chunk_size"]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("pair_solver.chunk_size must be a positive integer")
    symmetry_tolerance = _finite(
        pair_solver["symmetry_tolerance"], "pair_solver.symmetry_tolerance", nonnegative=True
    )
    diagnostic_tolerance = _finite(
        pair_solver["diagnostic_tolerance"], "pair_solver.diagnostic_tolerance", nonnegative=True
    )

    comparison = config["comparison"]
    if set(comparison) != {"source_times", "minimum_count", "ratio_epsilon"}:
        raise ValueError("comparison keys must exactly match the Phase 5 schema")
    source_times = comparison["source_times"]
    if (
        not isinstance(source_times, list)
        or not source_times
        or any(isinstance(value, bool) or not isinstance(value, int) for value in source_times)
        or source_times[0] < 0
        or any(right <= left for left, right in zip(source_times, source_times[1:]))
        or source_times[-1] >= abm.steps
    ):
        raise ValueError("comparison.source_times must be increasing ABM source times below steps")
    minimum_count = comparison["minimum_count"]
    if isinstance(minimum_count, bool) or not isinstance(minimum_count, int) or minimum_count < 1:
        raise ValueError("comparison.minimum_count must be a positive integer")
    ratio_epsilon = _finite(
        comparison["ratio_epsilon"], "comparison.ratio_epsilon", nonnegative=True
    )
    run_name = config["output"].get("run_name")
    if (
        set(config["output"]) != {"run_name"}
        or not isinstance(run_name, str)
        or len(run_name) > phase4.MAX_RUN_NAME_LENGTH
        or not phase4.RUN_NAME_PATTERN.fullmatch(run_name)
    ):
        raise ValueError("output.run_name must be bounded safe ASCII")

    phase3_raw = phase3b.inspect_raw_refinement_config(config)
    for scheme in schemes:
        for axis in ("q_c_edges", "q_d_edges"):
            previous = None
            for index, value in enumerate(scheme[axis]):
                edge = _finite(value, f"bin scheme {scheme['name']!r} {axis}[{index}]")
                if previous is not None and edge <= previous:
                    raise ValueError("configured bin edges must be strictly increasing")
                previous = edge
    for index, anchor in enumerate(config["anchors"]["points"]):
        for axis, value in enumerate(anchor):
            _finite(value, f"anchors.points[{index}][{axis}]")
    grid_probe = {
        "model": config["model"],
        "grid": {
            "q_min": initial["q_min"],
            "q_max": initial["q_max"],
            "spacing": initial["spacing"],
        },
        "solver": {
            "steps": source_times[-1],
            "dtype": abm.dtype,
            "chunk_size": chunk_size,
            "diagnostic_stride": 1,
            "symmetry_tolerance": symmetry_tolerance,
            "diagnostic_tolerance": diagnostic_tolerance,
        },
        "initial_condition": {
            "mode": "tiny_two_cell",
            "state_probabilities": state_probabilities,
        },
        "output": {"run_name": run_name},
    }
    pair_raw = phase4.inspect_raw_pair_config(grid_probe)
    return {
        "learning": learning,
        "abm": abm,
        "source_times": source_times,
        "minimum_count": minimum_count,
        "ratio_epsilon": ratio_epsilon,
        "pair_chunk_size": chunk_size,
        "symmetry_tolerance": symmetry_tolerance,
        "diagnostic_tolerance": diagnostic_tolerance,
        "phase3_raw": phase3_raw,
        "pair_raw": pair_raw,
    }


def estimate_phase5_resources(
    raw: dict,
    *,
    base_resources: dict | None = None,
    compiled_report: dict | None = None,
) -> dict:
    """Allocation-free Phase 5 lifetime peaks from arrays actually retained."""

    pair_raw = raw["pair_raw"]
    points = int(pair_raw["agent_grid_points"])
    item_bytes = int(pair_raw["item_bytes"])
    selected_times = len(raw["source_times"])
    cells = [
        selected_times * scheme["q_c_bins"] * scheme["q_d_bins"] * 2
        for scheme in raw["phase3_raw"]["schemes"]
    ]
    pair_point_host = selected_times * points * (
        PAIR_POINT_FOCAL_FLOATS_PER_POINT + 2 * PAIR_POINT_FLOATS_PER_POINT_ACTION
    ) * item_bytes + 2 * points * item_bytes + selected_times * 8
    pair_point_device = selected_times * points * (
        PAIR_POINT_FOCAL_FLOATS_PER_POINT + 2 * PAIR_POINT_FLOATS_PER_POINT_ACTION
    ) * item_bytes
    pair_bytes_by_scheme = [
        cell * PAIR_BINNED_ACTION_BYTES_PER_CELL
        + (cell // 2) * PAIR_BINNED_FOCAL_BYTES_PER_Q_CELL
        for cell in cells
    ]
    pair_binned = pair_bytes_by_scheme[-1] + max(pair_bytes_by_scheme[:-1], default=0)
    steps = int(raw["source_times"][-1])
    diagnostic_row_bytes = 11 * item_bytes + 3
    diagnostic_device = (steps + 1) * diagnostic_row_bytes
    diagnostic_host = diagnostic_device
    destination_validity_device = steps
    destination_validity_host = steps
    retained_comparison = sum(cells) * COMPARISON_RETAINED_BYTES_PER_CELL
    bootstrap_work = (
        raw["phase3_raw"]["bootstrap"]["replicates"]
        * max(cells)
        * COMPARISON_BOOTSTRAP_BYTES_PER_REPLICATE_CELL
    )
    comparison_rows = sum(cells)
    anchor_rows = (
        raw["phase3_raw"]["anchor_count"]
        * selected_times
        * 2
        * len(cells)
    )
    base = {
        "abm_record_bytes": 0,
        "abm_state_working_bytes": 0,
    }
    if base_resources is not None:
        if set(base_resources) != set(base):
            raise ValueError("base_resources must define every Phase 5 base memory term")
        for name, value in base_resources.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        base.update(base_resources)
    phase3 = phase3b.estimate_phase3b_resources(raw["abm"], raw["phase3_raw"])
    phase3_components = phase3["components"]
    phase4_static = phase4.estimate_pair_resources(pair_raw)
    phase4_components = phase4_static["components"]
    density_bytes = int(pair_raw["state_expanded_cells"]) * item_bytes
    grid_device_bytes = (
        int(pair_raw["grid_size"]) * item_bytes
        + points * (2 * item_bytes + 2 * np.dtype(np.int32).itemsize)
    )
    source_slot_device_bytes = (steps + 1) * np.dtype(np.int32).itemsize
    pair_retained_device = (
        2 * density_bytes
        + pair_point_device
        + diagnostic_device
        + destination_validity_device
        + grid_device_bytes
        + source_slot_device_bytes
    )
    pair_transfer_host = (
        density_bytes
        + pair_point_host
        + diagnostic_host
        + destination_validity_host
    )
    phase4_diagnostic_device = int(
        phase4_components["diagnostic_trajectory_bytes"]
    )
    static_pair_kernel_device = (
        int(phase4_static["static_device_bytes"])
        - phase4_diagnostic_device
        + pair_point_device
        + diagnostic_device
        + destination_validity_device
    )
    compiled_device = None
    compiled_host = None
    compiled_available = bool(compiled_report and compiled_report.get("available"))
    if compiled_available:
        compiled_device = int(compiled_report["compiled_device_requirement_bytes"])
        compiled_host = int(compiled_report["compiled_host_requirement_bytes"])
    pair_execution_device = max(
        static_pair_kernel_device,
        0 if compiled_device is None else compiled_device,
    )
    pair_execution_host = 0 if compiled_host is None else compiled_host

    edge_bytes = int(phase3_components["configured_and_effective_edge_bytes"])
    edge_python_bytes = int(phase3_components["metadata_effective_edge_python_bytes"])
    bootstrap_weight_bytes = int(phase3["bootstrap_weight_bytes"])
    bootstrap_weight_float64_bytes = int(
        phase3_components["bootstrap_weight_float64_conversion_bytes"]
    )
    record_and_state = base["abm_record_bytes"] + base["abm_state_working_bytes"]
    persistent_pair_and_abm = (
        record_and_state
        + pair_retained_device
        + pair_point_host
        + bootstrap_weight_bytes
        + edge_bytes
    )
    csv_writer_peak = (
        phase4.SERIALIZATION_FIXED_OVERHEAD_BYTES
        + phase4.SERIALIZATION_IO_BUFFER_BYTES
        + (phase4.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR + 1)
        * PHASE5_MAX_CSV_WRITE_CHARS
    )
    encoded_metadata_retained = (
        phase4.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * COMPARISON_METADATA_MAX_CHARS
    )
    configuration_validation = (
        phase4.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * phase4.MAX_NORMALIZED_CONFIGURATION_JSON_CHARS
        + edge_bytes
        + edge_python_bytes
    )
    compilation = (
        configuration_validation
        + int(phase4_components["grid_construction_host_bytes"])
        + grid_device_bytes
        + pair_execution_host
    )
    abm_simulation = (
        record_and_state + int(phase4_components["histogram_host_bytes"]) + edge_bytes
    )
    pair_execution = (
        record_and_state
        + pair_execution_device
        + pair_execution_host
        + int(phase4_components["histogram_host_bytes"])
        + edge_bytes
    )
    pair_transfer_validation = (
        record_and_state
        + pair_execution_device
        + pair_execution_host
        + pair_transfer_host
        + edge_bytes
    )
    aggregation = (
        persistent_pair_and_abm
        + int(phase3_components["aggregation_observation_work_bytes"])
        + int(phase3["scheme_per_run_strata"][-1]) * 88
        + pair_bytes_by_scheme[-1]
        + retained_comparison
    )
    reconstruction = (
        persistent_pair_and_abm
        + int(phase3_components["reconstruction_peak_sufficient_and_work_bytes"])
        + pair_binned
        + retained_comparison
    )
    pooled_point = (
        persistent_pair_and_abm
        + int(phase3_components["peak_sequential_sufficient_bytes"])
        + int(phase3_components["pooled_point_derivation_bytes"])
        + pair_binned
        + retained_comparison
    )
    bootstrap = (
        persistent_pair_and_abm
        + bootstrap_weight_float64_bytes
        + int(phase3_components["peak_sequential_sufficient_bytes"])
        + pair_binned
        + retained_comparison
        + bootstrap_work
    )
    anchor_accumulation = persistent_pair_and_abm + retained_comparison
    streamed_rows = (
        anchor_accumulation
        + encoded_metadata_retained
        + PHASE5_MAX_LIVE_ROW_BYTES
        + csv_writer_peak
    )
    serialization_component_options = {
        "metadata_encoding": {
            "anchor_accumulation": anchor_accumulation,
            "phase5_json_encoding": COMPARISON_SERIALIZATION_PEAK_BYTES,
        },
        "bootstrap_weight_archive": {
            "anchor_accumulation": anchor_accumulation,
            "encoded_metadata_retained": encoded_metadata_retained,
            "bootstrap_weight_serialization": int(
                phase3_components["bootstrap_weight_serialization_buffer_bytes"]
            ),
        },
        "metadata_chunked_write": {
            "anchor_accumulation": anchor_accumulation,
            "encoded_metadata_retained": encoded_metadata_retained,
            "metadata_text_chunk": (
                phase4.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
                * phase4.SERIALIZATION_CHUNK_CHARS
            ),
            "metadata_ascii_chunk": phase4.SERIALIZATION_CHUNK_CHARS,
            "binary_file_buffer": phase4.SERIALIZATION_IO_BUFFER_BYTES,
            "fixed_serializer_objects": phase4.SERIALIZATION_FIXED_OVERHEAD_BYTES,
        },
    }
    serialization_subpeaks = {
        name: sum(components.values())
        for name, components in serialization_component_options.items()
    }
    serialization = max(serialization_subpeaks.values())
    serialization_determining_subphase = max(
        serialization_subpeaks, key=serialization_subpeaks.get
    )
    phase_peak_components = {
        "configuration_and_scheme_validation": {
            "normalized_configuration_text": phase4.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
            * phase4.MAX_NORMALIZED_CONFIGURATION_JSON_CHARS,
            "configured_and_effective_edges": edge_bytes,
            "effective_edge_python_objects": edge_python_bytes,
        },
        "shape_lowering_and_compilation": {
            "configuration_validation_retained": configuration_validation,
            "grid_construction_host": int(phase4_components["grid_construction_host_bytes"]),
            "grid_device": grid_device_bytes,
            "analyzed_compiled_host": pair_execution_host,
        },
        "abm_simulation": {
            "abm_records_and_working_state": record_and_state,
            "histogram_host": int(phase4_components["histogram_host_bytes"]),
            "configured_and_effective_edges": edge_bytes,
        },
        "pair_execution": {
            "abm_records_and_working_state": record_and_state,
            "pair_executable_device": pair_execution_device,
            "analyzed_compiled_host": pair_execution_host,
            "histogram_host": int(phase4_components["histogram_host_bytes"]),
            "configured_and_effective_edges": edge_bytes,
        },
        "pair_transfer_and_validation": {
            "abm_records_and_working_state": record_and_state,
            "pair_executable_device": pair_execution_device,
            "analyzed_compiled_host": pair_execution_host,
            "pair_transfer_host": pair_transfer_host,
            "configured_and_effective_edges": edge_bytes,
        },
        "finest_plus_one_coarse_reconstruction": {
            "persistent_pair_and_abm": persistent_pair_and_abm,
            "abm_reconstruction_work": int(
                phase3_components["reconstruction_peak_sufficient_and_work_bytes"]
            ),
            "pair_finest_plus_one_coarse": pair_binned,
            "retained_comparison": retained_comparison,
        },
        "aggregation": {
            "persistent_pair_and_abm": persistent_pair_and_abm,
            "abm_observation_work": int(
                phase3_components["aggregation_observation_work_bytes"]
            ),
            "finest_abm_sufficient": int(phase3["scheme_per_run_strata"][-1]) * 88,
            "finest_pair_sufficient": pair_bytes_by_scheme[-1],
            "retained_comparison": retained_comparison,
        },
        "pooled_point_derivation": {
            "persistent_pair_and_abm": persistent_pair_and_abm,
            "peak_sequential_abm_sufficient": int(
                phase3_components["peak_sequential_sufficient_bytes"]
            ),
            "pooled_point_derivation": int(
                phase3_components["pooled_point_derivation_bytes"]
            ),
            "pair_finest_plus_one_coarse": pair_binned,
            "retained_comparison": retained_comparison,
        },
        "bootstrap_chunk_processing": {
            "persistent_pair_and_abm": persistent_pair_and_abm,
            "bootstrap_float64_weight_conversion": bootstrap_weight_float64_bytes,
            "peak_sequential_abm_sufficient": int(
                phase3_components["peak_sequential_sufficient_bytes"]
            ),
            "pair_finest_plus_one_coarse": pair_binned,
            "retained_comparison": retained_comparison,
            "comparison_bootstrap_work": bootstrap_work,
        },
        "bounded_anchor_accumulation": {
            "persistent_pair_and_abm": persistent_pair_and_abm,
            "retained_comparison": retained_comparison,
        },
        "streamed_row_output": {
            "anchor_accumulation": anchor_accumulation,
            "encoded_metadata_retained": encoded_metadata_retained,
            "one_live_python_row": PHASE5_MAX_LIVE_ROW_BYTES,
            "csv_writer_and_buffer": csv_writer_peak,
        },
        "phase5_serialization": serialization_component_options[
            serialization_determining_subphase
        ],
    }
    phase_peaks = {
        name: sum(components.values())
        for name, components in phase_peak_components.items()
    }
    determining_phase = max(phase_peaks, key=phase_peaks.get)
    comparison_peak = phase_peaks[determining_phase]
    return {
        "source_times": selected_times,
        "scheme_cells": cells,
        "comparison_rows": comparison_rows,
        "anchor_rows": anchor_rows,
        "pair_point_host_bytes": pair_point_host,
        "pair_point_device_bytes": pair_point_device,
        "pair_binned_bytes": pair_binned,
        "diagnostic_device_bytes": diagnostic_device,
        "diagnostic_host_bytes": diagnostic_host,
        "destination_validity_device_bytes": destination_validity_device,
        "destination_validity_host_bytes": destination_validity_host,
        "pair_retained_device_bytes": pair_retained_device,
        "pair_transfer_host_bytes": pair_transfer_host,
        "static_pair_kernel_device_bytes": static_pair_kernel_device,
        "analyzed_compiled_device_bytes": compiled_device,
        "analyzed_compiled_host_bytes": compiled_host,
        "retained_comparison_bytes": retained_comparison,
        "comparison_bootstrap_work_bytes": bootstrap_work,
        "maximum_live_python_row_bytes": PHASE5_MAX_LIVE_ROW_BYTES,
        "serialization_live_peak_bytes": COMPARISON_SERIALIZATION_PEAK_BYTES,
        "encoded_metadata_retained_bytes": encoded_metadata_retained,
        "serialization_subpeaks": serialization_subpeaks,
        "serialization_determining_subphase": serialization_determining_subphase,
        "csv_writer_peak_bytes": csv_writer_peak,
        "maximum_csv_write_chars": PHASE5_MAX_CSV_WRITE_CHARS,
        "phase_peak_components": phase_peak_components,
        "phase_peaks": phase_peaks,
        "global_peak_phase": determining_phase,
        "base_resources": base,
        "comparison_peak_bytes": comparison_peak,
        "formula": (
            "global=max(configuration, compilation, ABM, pair execution, pair transfer, "
            "aggregation/reconstruction, pooled, bootstrap, anchor, streamed-row, "
            "serialization lifetime peaks); diagnostics=(T+1)*(11*b+3), "
            "destination validity=T bytes on device and host"
        ),
    }


def validate_phase5_budget(
    raw: dict,
    allow_expensive: bool,
    *,
    base_resources: dict | None = None,
    compiled_report: dict | None = None,
    limits=None,
) -> dict:
    estimates = estimate_phase5_resources(
        raw, base_resources=base_resources, compiled_report=compiled_report
    )
    effective = dict(PHASE5_ABSOLUTE_LIMITS if limits is None else limits)
    if set(effective) != set(PHASE5_ABSOLUTE_LIMITS):
        raise ValueError("Phase 5 limits must define every fixed cap")
    values = {name: estimates[name] for name in PHASE5_ABSOLUTE_LIMITS}
    violations = [name for name in PHASE5_ABSOLUTE_LIMITS if values[name] > effective[name]]
    if violations and not allow_expensive:
        raise ValueError(
            "refusing expensive Phase 5 work before grid, compilation, simulation, "
            f"bootstrap, or output: violations={violations}; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "estimates": estimates,
        "absolute_limits": effective,
        "violations_overridden": violations if allow_expensive else [],
    }


def validate_phase5_pair_static_budget(
    raw: dict,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> dict:
    """Apply Phase 4 fixed caps to Phase 5 kernel arrays, not Phase 4 outputs."""

    pair_raw = raw["pair_raw"]
    phase5 = estimate_phase5_resources(raw)
    effective = dict(
        phase4.PHASE4_ABSOLUTE_LIMITS if limits is None else limits
    )
    if set(effective) != set(phase4.PHASE4_ABSOLUTE_LIMITS):
        raise ValueError("Phase 5 pair limits must define every fixed Phase 4 cap")
    values = {
        "agent_grid_points": int(pair_raw["agent_grid_points"]),
        "ordered_pair_cells": int(pair_raw["ordered_pair_cells"]),
        "state_expanded_cells": int(pair_raw["state_expanded_cells"]),
        "initial_pair_bytes": int(pair_raw["state_expanded_cells"])
        * int(pair_raw["item_bytes"]),
        "combined_peak_bytes": max(
            phase5["phase_peaks"]["pair_execution"],
            phase5["phase_peaks"]["pair_transfer_and_validation"],
        ),
        "retained_full_density_snapshots": 0,
        "diagnostic_output_rows": int(raw["source_times"][-1]) + 1,
    }
    violations = [
        name for name, value in values.items() if value > effective[name]
    ]
    if violations and not allow_expensive:
        raise ValueError(
            "refusing expensive Phase 5 pair kernel before grid construction or "
            f"compilation: violations={violations}; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "static_estimates": {
            "static_device_bytes": phase5["static_pair_kernel_device_bytes"],
            "static_host_bytes": phase5["pair_transfer_host_bytes"],
            "static_combined_peak_bytes": values["combined_peak_bytes"],
            **values,
        },
        "absolute_limits": effective,
        "static_violations": violations,
        "static_violations_overridden": violations if allow_expensive else [],
        "compiled_analysis": None,
        "compiled_violations": [],
        "compiled_violations_overridden": [],
    }


def validate_raw_histogram_budget(
    raw: dict,
    samples_per_grid_cell: int,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> dict:
    """Apply the Phase 2 histogram caps before constructing a ``QGrid``."""

    if (
        isinstance(samples_per_grid_cell, bool)
        or not isinstance(samples_per_grid_cell, int)
        or samples_per_grid_cell < 1
    ):
        raise ValueError("samples_per_grid_cell must be a positive integer")
    grid_size = int(raw["pair_raw"]["grid_size"])
    histogram_cells = grid_size * grid_size
    if histogram_cells != int(raw["pair_raw"]["agent_grid_points"]):
        raise RuntimeError("raw pair and histogram grid sizes disagree")
    estimates = {
        "grid_size": grid_size,
        "histogram_shape": [grid_size, grid_size],
        "histogram_cells": histogram_cells,
        "histogram_count_bytes": histogram_cells * int(np.dtype(np.int64).itemsize),
        "histogram_sample_pairs": histogram_cells * samples_per_grid_cell,
    }
    effective = dict(
        baseline.PHASE2_HISTOGRAM_ABSOLUTE_LIMITS if limits is None else limits
    )
    if set(effective) != set(baseline.PHASE2_HISTOGRAM_ABSOLUTE_LIMITS):
        raise ValueError("histogram limits must define every fixed Phase 2 cap")
    for name, value in effective.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"histogram limit {name} must be a non-negative integer")
    violations = [
        name
        for name in baseline.PHASE2_HISTOGRAM_ABSOLUTE_LIMITS
        if estimates[name] > effective[name]
    ]
    if violations and not allow_expensive:
        raise ValueError(
            "refusing expensive grid-matched histogram before grid construction or "
            f"compilation: violations={violations}; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "estimates": estimates,
        "absolute_limits": effective,
        "violations_overridden": violations if allow_expensive else [],
    }


def source_slot_by_time(source_times: list[int]) -> np.ndarray:
    slots = np.full(source_times[-1] + 1, -1, dtype=np.int32)
    slots[np.asarray(source_times, dtype=np.intp)] = np.arange(
        len(source_times), dtype=np.int32
    )
    return slots


def _runtime_environment_signature() -> dict:
    devices = [
        {
            "id": str(device.id),
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    return {
        "backend": jax.default_backend(),
        "platforms": sorted({device["platform"] for device in devices}),
        "devices": devices,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }


def _grid_argument_spec(grid) -> list[dict]:
    leaves = jax.tree_util.tree_leaves(grid)
    return [
        {"shape": list(leaf.shape), "dtype": np.dtype(leaf.dtype).name}
        for leaf in leaves
    ]


def _scalar_argument_spec(value) -> dict:
    array = jnp.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": np.dtype(array.dtype).name,
        "weak_type": bool(getattr(array, "weak_type", False)),
    }


def phase5_executable_signature(
    raw: dict,
    *,
    mass_shape,
    mass_dtype,
    grid,
    alpha,
    tau,
    slots,
    max_elements,
) -> dict:
    """Describe independently observed arguments and the current runtime."""

    return {
        "executable_id": PHASE5_EXECUTABLE_ID,
        "pair_shape": list(mass_shape),
        "dtype": np.dtype(mass_dtype).name,
        "grid_arguments": _grid_argument_spec(grid),
        "alpha": float(alpha),
        "tau": float(tau),
        "dynamic_scalar_arguments": {
            "alpha": _scalar_argument_spec(alpha),
            "tau": _scalar_argument_spec(tau),
            "diagnostic_tolerance": _scalar_argument_spec(
                raw["diagnostic_tolerance"]
            ),
        },
        "steps": int(raw["source_times"][-1]),
        "summary_count": len(raw["source_times"]),
        "requested_source_times": list(raw["source_times"]),
        "source_slot_by_time": np.asarray(slots, dtype=np.int32).tolist(),
        "source_slot_shape": list(np.shape(slots)),
        "source_slot_dtype": np.dtype(getattr(slots, "dtype", np.int32)).name,
        "chunk_size": int(raw["pair_chunk_size"]),
        "diagnostic_tolerance": float(raw["diagnostic_tolerance"]),
        "symmetry_tolerance": float(raw["symmetry_tolerance"]),
        "validation_max_elements": int(max_elements),
        **_runtime_environment_signature(),
    }


def _bundle_from_compiled(
    compiled,
    raw: dict,
    grid,
    learning,
    abstract_mass,
    slots,
) -> CompiledPairExecutableBundle:
    """Analyze and retain the same callable that guarded execution must invoke."""

    report = phase4.compiled_memory_report(compiled, 0)
    report["executable_id"] = PHASE5_EXECUTABLE_ID
    signature = phase5_executable_signature(
        raw,
        mass_shape=abstract_mass.shape,
        mass_dtype=abstract_mass.dtype,
        grid=grid,
        alpha=learning.alpha,
        tau=learning.tau,
        slots=slots,
        max_elements=raw["pair_raw"]["state_expanded_cells"],
    )
    report["executable_signature"] = signature
    abstract_arguments = {
        "pair_mass": {"shape": list(abstract_mass.shape), "dtype": np.dtype(abstract_mass.dtype).name},
        "grid": _grid_argument_spec(grid),
        "source_slots": {"shape": list(slots.shape), "dtype": np.dtype(slots.dtype).name},
        "dynamic_scalars": signature["dynamic_scalar_arguments"],
    }
    static_values = {
        "steps": raw["source_times"][-1],
        "summary_count": len(raw["source_times"]),
        "chunk_size": raw["pair_chunk_size"],
    }
    return CompiledPairExecutableBundle(
        compiled_callable=compiled,
        memory_report=report,
        compile_signature=signature,
        abstract_arguments=abstract_arguments,
        static_values=static_values,
        runtime_environment={
            key: signature[key]
            for key in ("backend", "platforms", "devices", "jax_enable_x64")
        },
    )


def analyze_compiled_phase5_pair_memory(
    raw: dict, grid, learning
) -> CompiledPairExecutableBundle:
    """Lower, compile, analyze, and retain the exact executable for execution."""

    dtype = jnp.float32 if raw["abm"].dtype == "float32" else jnp.float64
    points = int(raw["pair_raw"]["agent_grid_points"])
    abstract_mass = jax.ShapeDtypeStruct((2, points, points), dtype)
    slots = jnp.asarray(source_slot_by_time(raw["source_times"]))
    lowered = simulate_pair_source_summaries_jit.lower(
        abstract_mass,
        grid,
        learning.alpha,
        learning.tau,
        slots,
        steps=raw["source_times"][-1],
        summary_count=len(raw["source_times"]),
        chunk_size=raw["pair_chunk_size"],
        diagnostic_tolerance=raw["diagnostic_tolerance"],
    )
    return _bundle_from_compiled(
        lowered.compile(), raw, grid, learning, abstract_mass, slots
    )


def _validate_compiled_bundle_contract(bundle: CompiledPairExecutableBundle) -> None:
    """Reject any disagreement among independently retained compile facts."""

    signature = bundle.compile_signature
    expected_static = {
        "steps": signature.get("steps"),
        "summary_count": signature.get("summary_count"),
        "chunk_size": signature.get("chunk_size"),
    }
    expected_abstract = {
        "pair_mass": {
            "shape": signature.get("pair_shape"),
            "dtype": signature.get("dtype"),
        },
        "grid": signature.get("grid_arguments"),
        "source_slots": {
            "shape": signature.get("source_slot_shape"),
            "dtype": signature.get("source_slot_dtype"),
        },
        "dynamic_scalars": signature.get("dynamic_scalar_arguments"),
    }
    expected_environment = {
        key: signature.get(key)
        for key in ("backend", "platforms", "devices", "jax_enable_x64")
    }
    if bundle.memory_report.get("executable_signature") != signature:
        raise ValueError("compiled memory report disagrees with the retained compile signature")
    if bundle.static_values != expected_static:
        raise ValueError("compiled static values disagree with the retained compile signature")
    if bundle.abstract_arguments != expected_abstract:
        raise ValueError("compiled abstract arguments disagree with the retained compile signature")
    if bundle.runtime_environment != expected_environment:
        raise ValueError("compiled runtime environment disagrees with the retained compile signature")


def validate_compiled_phase5_pair_budget(
    pair_budget: dict,
    bundle: CompiledPairExecutableBundle,
    raw: dict,
    allow_expensive: bool,
) -> dict:
    """Fail closed on the exact Phase 5 executable before scientific allocation."""

    report = bundle.memory_report
    violations = []
    try:
        _validate_compiled_bundle_contract(bundle)
    except ValueError:
        violations.append("analyzed_executable_signature_mismatch")
    if report.get("executable_id") != PHASE5_EXECUTABLE_ID:
        violations.append("analyzed_executable_id_mismatch")
    if not report.get("available", False):
        violations.append("compiled_analysis_unavailable")
        report["validation_status"] = "unavailable"
    else:
        static_device = estimate_phase5_resources(raw)[
            "static_pair_kernel_device_bytes"
        ]
        if int(report["compiled_device_requirement_bytes"]) > static_device:
            violations.append("compiled_device_exceeds_phase5_static_bound")
        if int(report["compiled_plus_host_requirement_bytes"]) > int(
            pair_budget["absolute_limits"]["combined_peak_bytes"]
        ):
            violations.append("compiled_combined_peak_bytes")
        report["validation_status"] = "failed" if violations else "passed"
    if violations and not allow_expensive:
        raise ValueError(
            "refusing Phase 5 pair execution after exact shape-only compilation but "
            "before histogram, pair, ABM, or output allocation: "
            f"violations={violations}; pass --allow-expensive to override"
        )
    pair_budget["compiled_analysis"] = report
    pair_budget["compiled_violations"] = violations
    pair_budget["compiled_violations_overridden"] = violations if allow_expensive else []
    return pair_budget


def run_pair_source_summaries(
    bundle: CompiledPairExecutableBundle,
    initial_mass,
    grid,
    learning: LearningConfig,
    source_times: list[int],
    *,
    chunk_size: int,
    symmetry_tolerance: float,
    diagnostic_tolerance: float,
    max_elements: int,
    raw: dict,
):
    """Execute the already analyzed bounded scan and copy only its summaries."""

    _validate_compiled_bundle_contract(bundle)
    slots = jnp.asarray(source_slot_by_time(source_times))
    invocation_raw = {
        **raw,
        "source_times": list(source_times),
        "pair_chunk_size": chunk_size,
        "diagnostic_tolerance": diagnostic_tolerance,
        "symmetry_tolerance": symmetry_tolerance,
    }
    invocation_signature = phase5_executable_signature(
        invocation_raw,
        mass_shape=initial_mass.shape,
        mass_dtype=initial_mass.dtype,
        grid=grid,
        alpha=learning.alpha,
        tau=learning.tau,
        slots=slots,
        max_elements=max_elements,
    )
    if invocation_signature != bundle.compile_signature:
        raise ValueError("compiled pair invocation signature does not match analyzed executable")
    result = execute_compiled_pair_source_summaries(
        bundle.compiled_callable,
        initial_mass,
        grid,
        learning.alpha,
        learning.tau,
        slots,
        steps=source_times[-1],
        summary_count=len(source_times),
        symmetry_tolerance=symmetry_tolerance,
        diagnostic_tolerance=diagnostic_tolerance,
        max_elements=max_elements,
    )
    points = pair_point_sufficient_from_jax_summary(
        result.source_summaries, source_times, grid
    )
    return points, result, invocation_signature


def iter_comparison_rows(results, source_times, observation_dtype):
    actions = ("C", "D")
    for scheme, comparison, intervals in results:
        configured_c = scheme.bins.q_c_edges
        configured_d = scheme.bins.q_d_edges
        effective_c, effective_d = scheme.bins.effective_edges(observation_dtype)
        for local_time, source_time in enumerate(source_times):
            for q_c in range(scheme.bins.num_q_c_bins):
                for q_d in range(scheme.bins.num_q_d_bins):
                    for action in range(2):
                        index = (local_time, q_c, q_d, action)
                        row = {
                            "scheme": scheme.name,
                            "source_time_t": source_time,
                            "q_c_bin": q_c,
                            "q_d_bin": q_d,
                            "configured_q_c_lower": configured_c[q_c],
                            "configured_q_c_upper": configured_c[q_c + 1],
                            "configured_q_d_lower": configured_d[q_d],
                            "configured_q_d_upper": configured_d[q_d + 1],
                            "effective_q_c_lower": effective_c[q_c],
                            "effective_q_c_upper": effective_c[q_c + 1],
                            "effective_q_d_lower": effective_d[q_d],
                            "effective_q_d_upper": effective_d[q_d + 1],
                            "action_index": action,
                            "action": actions[action],
                        }
                        for name in comparison.__dataclass_fields__:
                            value = np.asarray(getattr(comparison, name))
                            value_index = index[:-1] if value.ndim == 3 else index
                            row[name] = value[value_index].item()
                        for name in COMPARISON_BOOTSTRAP_ESTIMANDS:
                            row[f"{name}_lower"] = intervals.lower[name][index]
                            row[f"{name}_upper"] = intervals.upper[name][index]
                            row[f"{name}_interval_valid"] = intervals.interval_valid[name][index]
                            row[f"{name}_valid_replicates"] = intervals.valid_replicates[name][index]
                            row[f"{name}_invalid_replicates"] = intervals.invalid_replicates[name][index]
                        yield row


def iter_anchor_rows(results, anchors, source_times, dtype):
    for scheme, comparison, intervals in results:
        configured_c = scheme.bins.q_c_edges
        configured_d = scheme.bins.q_d_edges
        effective_c, effective_d = scheme.bins.effective_edges(dtype)
        for anchor_index, anchor in enumerate(anchors):
            q_c, q_d = anchor_bin_index(scheme.bins, anchor, dtype)
            for local_time, source_time in enumerate(source_times):
                for action in range(2):
                    index = (local_time, q_c, q_d, action)
                    row = {
                        "scheme": scheme.name,
                        "anchor_index": anchor_index,
                        "anchor_q_c": anchor[0],
                        "anchor_q_d": anchor[1],
                        "source_time_t": source_time,
                        "action_index": action,
                        "q_c_bin": q_c,
                        "q_d_bin": q_d,
                        "configured_q_c_lower": configured_c[q_c],
                        "configured_q_c_upper": configured_c[q_c + 1],
                        "configured_q_d_lower": configured_d[q_d],
                        "configured_q_d_upper": configured_d[q_d + 1],
                        "configured_q_c_width": configured_c[q_c + 1] - configured_c[q_c],
                        "configured_q_d_width": configured_d[q_d + 1] - configured_d[q_d],
                        "effective_q_c_lower": effective_c[q_c],
                        "effective_q_c_upper": effective_c[q_c + 1],
                        "effective_q_d_lower": effective_d[q_d],
                        "effective_q_d_upper": effective_d[q_d + 1],
                        "effective_q_c_width": effective_c[q_c + 1] - effective_c[q_c],
                        "effective_q_d_width": effective_d[q_d + 1] - effective_d[q_d],
                        "abm_count": comparison.abm_count[index],
                        "pair_selected_mass": comparison.pair_selected_mass[index],
                        "direct_abm_velocity_variance": comparison.direct_abm_velocity_variance[index],
                        "reconstructed_abm_velocity_variance": comparison.reconstructed_abm_velocity_variance[index],
                        "pair_velocity_variance": comparison.pair_velocity_variance[index],
                        "hybrid_velocity_variance": comparison.hybrid_velocity_variance[index],
                        "pair_minus_direct": comparison.pair_minus_direct[index],
                        "hybrid_minus_direct": comparison.hybrid_minus_direct[index],
                    }
                    for name in COMPARISON_BOOTSTRAP_ESTIMANDS:
                        row[f"{name}_lower"] = intervals.lower[name][index]
                        row[f"{name}_upper"] = intervals.upper[name][index]
                        row[f"{name}_interval_valid"] = intervals.interval_valid[name][index]
                        row[f"{name}_valid_replicates"] = intervals.valid_replicates[name][index]
                        row[f"{name}_invalid_replicates"] = intervals.invalid_replicates[name][index]
                    yield row


def deep_size_bytes(value) -> int:
    """Identity-aware Python object size used to enforce one live-row bound."""

    seen: set[int] = set()

    def visit(item) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(item)
        if isinstance(item, dict):
            total += sum(visit(key) + visit(child) for key, child in item.items())
        elif isinstance(item, (tuple, list, set, frozenset)):
            total += sum(visit(child) for child in item)
        return total

    return visit(value)


def validate_streamed_rows(row_factory, expected_rows: int, *, observe=None) -> dict:
    """Audit header/rows without retaining serialized text or a row collection."""

    iterator = iter(row_factory())
    try:
        first = next(iterator)
    except StopIteration as error:
        raise RuntimeError("CSV output must contain at least one row") from error
    fieldnames = list(first)
    sink = phase4._BoundedCountingTextSink(PHASE5_MAX_CSV_WRITE_CHARS)
    writer = csv.DictWriter(sink, fieldnames=fieldnames)
    writer.writeheader()
    count = 0
    maximum_live = 0
    maximum_live = deep_size_bytes(first)
    if maximum_live > PHASE5_MAX_LIVE_ROW_BYTES:
        raise RuntimeError("one Phase 5 CSV row exceeds its live-object bound")
    writer.writerow(first)
    if observe is not None:
        observe(first)
    count = 1
    del first
    for row in iterator:
        row_bytes = deep_size_bytes(row)
        maximum_live = max(maximum_live, row_bytes)
        if row_bytes > PHASE5_MAX_LIVE_ROW_BYTES:
            raise RuntimeError("one Phase 5 CSV row exceeds its live-object bound")
        writer.writerow(row)
        if observe is not None:
            observe(row)
        count += 1
    if count != expected_rows:
        raise RuntimeError(f"stream produced {count} rows; expected {expected_rows}")
    return {
        "row_count": count,
        "maximum_live_python_row_bytes": maximum_live,
        "maximum_csv_write_chars": sink.maximum_observed_chars,
    }


def write_streamed_rows(path: Path, row_factory, expected_rows: int) -> None:
    """Write a fresh deterministic row stream with bounded record buffering."""

    iterator = iter(row_factory())
    first = next(iterator)
    count = 0
    with path.open("wb", buffering=phase4.SERIALIZATION_IO_BUFFER_BYTES) as binary_file:
        sink = phase4._BoundedAsciiBinarySink(binary_file, PHASE5_MAX_CSV_WRITE_CHARS)
        writer = csv.DictWriter(sink, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        count = 1
        del first
        for row in iterator:
            writer.writerow(row)
            count += 1
    if count != expected_rows:
        raise RuntimeError(f"written row count {count} disagrees with preflight")


def _weight_hash(weights: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(weights)).cast("B")).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = baseline.load_config(config_path)
    raw = inspect_comparison_config(config)
    learning = raw["learning"]
    abm = raw["abm"]

    # Every allocation-sensitive subsystem is preflighted before QGrid creation.
    abm_budget = baseline.validate_resource_budget(
        abm, config["safety"], args.allow_expensive,
        record_mode=baseline.RESOURCE_MODE_INSTRUMENTED,
    )
    phase3_budget = phase3b.validate_phase3b_budget(
        abm, raw["phase3_raw"], args.allow_expensive
    )
    pair_budget = validate_phase5_pair_static_budget(raw, args.allow_expensive)
    initial = config["initial_condition"]
    histogram_budget = validate_raw_histogram_budget(
        raw, int(initial["samples_per_grid_cell"]), args.allow_expensive
    )
    base_resources = {
        "abm_record_bytes": int(abm_budget["values"]["record_bytes"]),
        "abm_state_working_bytes": int(abm_budget["values"]["state_working_bytes"]),
    }
    phase5_budget = validate_phase5_budget(
        raw, args.allow_expensive, base_resources=base_resources
    )
    normalized_configuration_text = json.dumps(
        config, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    if len(normalized_configuration_text) > phase4.MAX_NORMALIZED_CONFIGURATION_JSON_CHARS:
        raise ValueError("normalized Phase 5 configuration exceeds the bounded size")
    normalized_configuration = json.loads(normalized_configuration_text)
    # Full configured/effective edge, nesting, and anchor validation precedes
    # QGrid construction, JAX lowering, histogram sampling, and simulation.
    schemes, anchors, _, _ = phase3b.construct_guarded_schemes(
        abm, config, args.allow_expensive
    )
    git_commit = phase4._bounded_metadata_text(
        phase4.git_text("rev-parse", "HEAD"), "Git commit", 64
    )
    git_status = phase4._bounded_metadata_text(
        phase4.git_text("status", "--short", "--", "."),
        "Git status",
        phase4.MAX_GIT_STATUS_CHARS,
    )
    source_hashes = {
        **baseline.implementation_source_hashes(config_path),
        "experiments/run_velocity_variance_comparison.py": baseline.sha256(
            Path(__file__).resolve()
        ),
    }

    grid = QGrid(float(initial["q_min"]), float(initial["q_max"]), float(initial["spacing"]))
    dtype = jnp.float32 if abm.dtype == "float32" else jnp.float64
    pair_grid = build_jax_pair_grid(grid, dtype)
    compiled_bundle = analyze_compiled_phase5_pair_memory(raw, pair_grid, learning)
    compiled_report = compiled_bundle.memory_report
    pair_budget = validate_compiled_phase5_pair_budget(
        pair_budget, compiled_bundle, raw, args.allow_expensive
    )
    phase5_budget = validate_phase5_budget(
        raw,
        args.allow_expensive,
        base_resources=base_resources,
        compiled_report=compiled_report,
    )
    constructed_histogram_estimate = baseline.estimate_legacy_histogram_resources(
        grid, int(initial["samples_per_grid_cell"])
    )
    for name in baseline.PHASE2_HISTOGRAM_ABSOLUTE_LIMITS:
        if constructed_histogram_estimate[name] != histogram_budget["estimates"][name]:
            raise RuntimeError("constructed grid disagrees with raw histogram preflight")
    histogram = seeded_legacy_histogram(
        grid,
        seed=int(initial["histogram_seed"]),
        samples_per_grid_cell=int(initial["samples_per_grid_cell"]),
    )
    graph = complete_graph(abm.num_agents)
    initialized = initialize_grid_matched_batch(
        graph, histogram, abm_seed=abm.abm_seed, num_runs=abm.num_runs, dtype=dtype
    )
    abm_result = simulate_instrumented_batch_jit(
        initialized.state,
        initialized.simulation_key,
        graph,
        learning.alpha,
        learning.tau,
        steps=abm.steps,
    )
    abm_result.final_state.q_values.block_until_ready()

    initial_pair = ordered_pair_mass_jax(
        histogram,
        state_probabilities=tuple(initial["state_probabilities"]),
        dtype=dtype,
    )
    validate_jax_pair_mass(
        initial_pair,
        pair_grid,
        symmetry_tolerance=raw["symmetry_tolerance"],
        max_elements=raw["pair_raw"]["state_expanded_cells"],
    )
    pair_points, pair_result, invocation_signature = run_pair_source_summaries(
        compiled_bundle,
        initial_pair,
        pair_grid,
        learning,
        raw["source_times"],
        chunk_size=raw["pair_chunk_size"],
        symmetry_tolerance=raw["symmetry_tolerance"],
        diagnostic_tolerance=raw["diagnostic_tolerance"],
        max_elements=raw["pair_raw"]["state_expanded_cells"],
        raw=raw,
    )
    pair_result.final_mass.block_until_ready()
    weights = bootstrap_run_weights(
        abm.num_runs,
        raw["phase3_raw"]["bootstrap"]["replicates"],
        raw["phase3_raw"]["bootstrap"]["seed"],
    )

    finest_scheme = schemes[-1]
    finest_pair = aggregate_pair_points(pair_points, finest_scheme.bins)
    finest_abm = aggregate_variance_records(
        abm_result.records,
        finest_scheme.bins,
        num_agents=abm.num_agents,
        alpha=learning.alpha,
        min_count=raw["minimum_count"],
    )
    reconstruction_diagnostics = []
    # Only the authoritative finest statistics and one reconstructed coarse
    # scheme are live. Bounded derived comparison/interval summaries are kept
    # for deterministic row streaming and cross-scheme anchors.
    results = []
    for scheme in schemes:
        is_finest = scheme.name == finest_scheme.name
        abm_statistics = (
            finest_abm if is_finest else coarsen_abm_sufficient(finest_abm, scheme.bins)
        )
        pair_sufficient = finest_pair if is_finest else coarsen_pair_sufficient(
            finest_pair, scheme.bins
        )
        if not is_finest:
            diagnostic = assert_child_reconstructs_parent(abm_statistics, finest_abm)
            reconstruction_diagnostics.append(
                {
                    "parent_scheme": scheme.name,
                    "child_scheme": finest_scheme.name,
                    **diagnostic,
                }
            )
        pair_moments = derive_pair_binned_moments(
            pair_sufficient, num_agents=abm.num_agents, alpha=learning.alpha
        )
        selected_abm = select_abm_source_times(abm_statistics, raw["source_times"])
        comparison = compare_four_way(
            selected_abm,
            pair_moments,
            abm_source_times=raw["source_times"],
            ratio_epsilon=raw["ratio_epsilon"],
        )
        intervals = bootstrap_four_way_intervals(
            selected_abm,
            pair_moments,
            weights,
            confidence_level=raw["phase3_raw"]["bootstrap"]["confidence_level"],
        )
        results.append((scheme, comparison, intervals))
        if not is_finest:
            del abm_statistics, pair_sufficient
        del pair_moments, selected_abm

    comparison_factory = lambda: iter_comparison_rows(
        results, raw["source_times"], np.dtype(abm.dtype)
    )
    anchor_factory = lambda: iter_anchor_rows(
        results, anchors, raw["source_times"], np.dtype(abm.dtype)
    )
    descriptive_fields = (
        "direct_abm_velocity_variance",
        "reconstructed_abm_velocity_variance",
        "pair_velocity_variance",
        "hybrid_velocity_variance",
    )
    descriptive = {
        name: {"count": 0, "sum": 0.0, "minimum": math.inf, "maximum": -math.inf}
        for name in descriptive_fields
    }
    maximum_closure_error = float("nan")

    def observe_comparison(row):
        nonlocal maximum_closure_error
        closure = float(row["direct_minus_reconstructed"])
        if math.isfinite(closure):
            absolute = abs(closure)
            maximum_closure_error = (
                absolute
                if not math.isfinite(maximum_closure_error)
                else max(maximum_closure_error, absolute)
            )
        if int(row["abm_count"]) < raw["minimum_count"] or not bool(row["pair_valid"]):
            return
        for name in descriptive_fields:
            value = float(row[name])
            if math.isfinite(value):
                summary = descriptive[name]
                summary["count"] += 1
                summary["sum"] += value
                summary["minimum"] = min(summary["minimum"], value)
                summary["maximum"] = max(summary["maximum"], value)

    comparison_stream = validate_streamed_rows(
        comparison_factory,
        phase5_budget["estimates"]["comparison_rows"],
        observe=observe_comparison,
    )
    anchor_stream = validate_streamed_rows(
        anchor_factory, phase5_budget["estimates"]["anchor_rows"]
    )
    for summary in descriptive.values():
        summary["mean"] = (
            summary["sum"] / summary["count"] if summary["count"] else float("nan")
        )
        if not summary["count"]:
            summary["minimum"] = summary["maximum"] = float("nan")

    diagnostic_host = jax.device_get(pair_result.diagnostics)
    destination_host = np.asarray(jax.device_get(pair_result.destinations_valid))
    metadata = {
        "schema_version": 1,
        "milestone": "Phase 5 bounded matched pair-versus-ABM variance comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": normalized_configuration,
        "source_time_convention": "ABM record t and pair P_t before the update to t+1",
        "formulas": {
            "abm_reconstruction": "alpha^2*(sigma2/N+(N-1)*c/N+Var(q_j)-2*Cov(r,q_j))",
            "pair_closure": "alpha^2*(sigma2_pair/N+(N-1)*c_pair/N+Var_pair(q_j)-2*Cov_pair(r,q_j))",
            "hybrid": "pair formula with c_pair replaced by c_ABM",
            "pair_exact_q_distinct_product": "mu(q,j)^2 before finite-bin pooling",
        },
        "resource_budget": {
            "phase2_abm": abm_budget,
            "phase3b_statistics": phase3_budget,
            "phase5_pair_kernel": pair_budget,
            "phase5_comparison": phase5_budget,
            "histogram": histogram_budget,
        },
        "bootstrap": {
            "unit": "complete independent ABM run",
            "quantile_method": QUANTILE_METHOD,
            "minimum_contributing_runs": MIN_CONTRIBUTING_RUNS,
            "minimum_valid_bootstrap_fraction": MIN_VALID_BOOTSTRAP_FRACTION,
            "minimum_valid_replicates": max(
                2,
                math.ceil(
                    MIN_VALID_BOOTSTRAP_FRACTION
                    * raw["phase3_raw"]["bootstrap"]["replicates"]
                ),
            ),
            "weights_shape": list(weights.shape),
            "weights_sha256": _weight_hash(weights),
            "pair_quantities_resampled": False,
        },
        "nested_abm_reconstruction": reconstruction_diagnostics,
        "pair_execution": {
            "analyzed": compiled_bundle.compile_signature,
            "invoked": invocation_signature,
            "signatures_match": compiled_bundle.compile_signature
            == invocation_signature,
            "abstract_arguments": compiled_bundle.abstract_arguments,
            "static_values": compiled_bundle.static_values,
            "diagnostic_tolerance_used": raw["diagnostic_tolerance"],
            "symmetry_tolerance_used": raw["symmetry_tolerance"],
            "diagnostic_times": list(range(raw["source_times"][-1] + 1)),
            "maximum_conditional_weight_error": float(
                np.max(np.asarray(diagnostic_host.conditional_weight_error))
            ),
            "minimum_conditional_variance": float(
                np.min(np.asarray(diagnostic_host.minimum_conditional_variance))
            ),
            "maximum_mass_error": float(
                np.max(np.abs(np.asarray(diagnostic_host.total_mass) - 1.0))
            ),
            "maximum_symmetry_error": float(
                np.max(np.asarray(diagnostic_host.symmetry_error))
            ),
            "all_destinations_valid": bool(np.all(destination_host)),
        },
        "outputs": {
            "comparison_rows": comparison_stream["row_count"],
            "anchor_rows": anchor_stream["row_count"],
            "comparison_stream": comparison_stream,
            "anchor_stream": anchor_stream,
            "maximum_abm_reconstruction_closure_error": maximum_closure_error,
            "online_descriptive_statistics": descriptive,
        },
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
        },
        "git": {
            "commit": git_commit,
            "subproject_status_before_output": git_status,
        },
        "source_hashes": source_hashes,
        "limitations": [
            "small CPU-scale comparison only",
            "deterministic nearest_legacy pair projection",
            "ABM covariance is empirical and not predicted by pair theory",
            "no production conclusion, full grid, interpolation, or GPU benchmark",
        ],
    }
    encoded_metadata = phase4.encode_bounded_metadata(
        metadata, COMPARISON_METADATA_MAX_CHARS
    )

    run_name = config["output"]["run_name"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = (PROJECT_ROOT / "outputs" / "variance_comparison").resolve()
    run_directory = (output_root / f"{run_name}-{timestamp}").resolve()
    if output_root not in run_directory.parents:
        raise ValueError("output directory must remain beneath outputs/variance_comparison")
    run_directory.mkdir(parents=True, exist_ok=False)
    write_streamed_rows(
        run_directory / "variance_comparison.csv",
        comparison_factory,
        comparison_stream["row_count"],
    )
    write_streamed_rows(
        run_directory / "anchor_bin_refinement.csv",
        anchor_factory,
        anchor_stream["row_count"],
    )
    np.savez_compressed(run_directory / "bootstrap_run_weights.npz", weights=weights)
    phase4.write_bounded_metadata(run_directory / "metadata.json", encoded_metadata)
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        f"wrote {comparison_stream['row_count']} comparison rows and "
        f"{anchor_stream['row_count']} anchor rows "
        f"to {run_directory}; max closure error={maximum_closure_error:.6g}"
    )


if __name__ == "__main__":
    main()

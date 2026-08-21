#!/usr/bin/env python3
"""Run a bounded CPU parity/memory benchmark for flat and separable pair kernels."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

from chu_pair.grids import QGrid
from chu_pair.pair_density import pair_contraction_precision
from chu_pair.initial_conditions import tiny_histogram
from chu_pair.pair_density import (
    CompiledExecutableBundle,
    build_jax_pair_grid,
    estimate_flat_resources,
    estimate_separable_resources,
    full_grid_feasibility,
    make_compiled_executable_bundle,
    production_capacity_preflight,
    simulate_pair_source_summaries_from_histogram_full_jit,
    simulate_pair_source_summaries_from_histogram_jit,
    validate_compiled_executable_bundle,
)
try:
    from experiments.run_pair_jax_small import compiled_memory_report
except ModuleNotFoundError:  # direct ``python experiments/...`` invocation
    from run_pair_jax_small import compiled_memory_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pair_separable_benchmark_small.toml"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXPECTED_SECTIONS = {"model", "benchmark", "cases", "feasibility", "output"}
EXPECTED_KEYS = {
    "model": {"alpha", "tau", "state_probabilities"},
    "benchmark": {"warmup", "repetitions", "safety_margin_fraction"},
    "feasibility": {"grid_size", "representative_steps", "representative_summary_count"},
    "output": {"run_name"},
}
EXPECTED_CASE_KEYS = {
    "label", "q_min", "q_max", "spacing", "dtype", "steps",
    "source_times", "flat_chunk_size", "row_block_size", "column_block_size",
    "diagnostic_tolerance", "symmetry_tolerance",
}
MAX_CONFIG_INTEGER = 2**63 - 1
MAX_RUN_NAME_LENGTH = 64
MAX_LABEL_LENGTH = 32
MAX_CASES = 6
MAX_GRID_SIZE = 17
MAX_AGENT_GRID_POINTS = MAX_GRID_SIZE**2
MAX_STATE_EXPANDED_CELLS = 2 * MAX_AGENT_GRID_POINTS**2
MAX_DENSITY_BYTES = 2 * MAX_AGENT_GRID_POINTS**2 * 8
MAX_STEPS = 4
MAX_SUMMARY_COUNT = 5
MAX_BLOCK_SIZE = MAX_AGENT_GRID_POINTS
MAX_REPETITIONS = 4
MAX_WARMUP = 2
MAX_TIMING_RECORDS = 4 * MAX_CASES
MAX_TIMING_SAMPLES = (MAX_TIMING_RECORDS + 2) * MAX_REPETITIONS
MAX_STATIC_DEVICE_BYTES = 128 * 1024**2
MAX_STATIC_HOST_BYTES = 128 * 1024**2
MAX_STATIC_COMBINED_BYTES = 256 * 1024**2
MAX_COMPILED_DEVICE_BYTES = 128 * 1024**2
MAX_COMPILED_COMBINED_BYTES = 256 * 1024**2
MAX_SERIALIZED_METADATA_BYTES = 256 * 1024
MAX_CSV_RECORD_BYTES = 16 * 1024
MAX_TOTAL_CASE_DENSITY_BYTES = 8 * 1024**2
SOURCE_HASH_BUFFER_BYTES = 1 << 20
MAX_GIT_STATUS_CHARS = 8 * 1024
MAX_DEVICE_COUNT = 16
MAX_DEVICE_TEXT_CHARS = 256
MAX_HLO_TEXT_CHARS = 8 * 1024**2
HLO_HASH_CHUNK_CHARS = 64 * 1024
HLO_LIVE_ALLOWANCE_BYTES = 4 * MAX_HLO_TEXT_CHARS + 2 * HLO_HASH_CHUNK_CHARS
MAX_SAFETY_MARGIN_FRACTION = 1.0
STATIC_REDUCTION_FIXED_BYTES = 4 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help=(
            "override only documented bounded-development static resource caps; "
            "identity, compiled analysis, capacity evidence and science never override"
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _finite(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite{' and non-negative' if nonnegative else ''}")
    return result


def _integer(value, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool) or not isinstance(value, int)
        or value < minimum or value > MAX_CONFIG_INTEGER
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {MAX_CONFIG_INTEGER}]")
    return value


def _safe_text(value, name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum or not RUN_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be bounded safe ASCII text")
    return value


def inspect_raw_config(config: dict) -> dict:
    """Normalize and count every case using allocation-free Python arithmetic."""

    if not isinstance(config, dict) or set(config) != EXPECTED_SECTIONS:
        raise ValueError("configuration sections must match the exact benchmark schema")
    for section, keys in EXPECTED_KEYS.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"configuration keys in {section} must match the exact schema")
    cases = config["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f"cases must contain between 1 and {MAX_CASES} tables")

    model = config["model"]
    alpha = _finite(model["alpha"], "model.alpha", nonnegative=True)
    tau = _finite(model["tau"], "model.tau", nonnegative=True)
    probabilities = model["state_probabilities"]
    if (
        not isinstance(probabilities, list) or len(probabilities) != 2
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in probabilities)
    ):
        raise ValueError("model.state_probabilities must contain two numbers")
    probabilities = [_finite(x, "model.state_probabilities", nonnegative=True) for x in probabilities]
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("model.state_probabilities must sum to one")
    if probabilities != [0.5, 0.5]:
        raise ValueError(
            "the controlled production-oriented benchmark requires uniform state mass"
        )

    benchmark = config["benchmark"]
    warmup = _integer(benchmark["warmup"], "benchmark.warmup")
    repetitions = _integer(benchmark["repetitions"], "benchmark.repetitions", 2)
    if repetitions % 2:
        raise ValueError("benchmark.repetitions must be even so timing order is balanced")
    margin = _finite(
        benchmark["safety_margin_fraction"],
        "benchmark.safety_margin_fraction",
        nonnegative=True,
    )
    if margin > MAX_SAFETY_MARGIN_FRACTION:
        raise ValueError(
            f"benchmark.safety_margin_fraction must not exceed {MAX_SAFETY_MARGIN_FRACTION}"
        )
    feasibility = config["feasibility"]
    feasibility_grid = _integer(feasibility["grid_size"], "feasibility.grid_size", 1)
    if feasibility_grid != 131:
        raise ValueError("the allocation-free feasibility target must be exactly G=131")
    representative_steps = _integer(
        feasibility["representative_steps"], "feasibility.representative_steps"
    )
    representative_summaries = _integer(
        feasibility["representative_summary_count"],
        "feasibility.representative_summary_count",
        1,
    )

    normalized_cases = []
    total_density_bytes = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != EXPECTED_CASE_KEYS:
            raise ValueError(f"case {index} keys must match the exact schema")
        label = _safe_text(case["label"], f"cases[{index}].label", MAX_LABEL_LENGTH)
        q_min = _finite(case["q_min"], f"cases[{index}].q_min")
        q_max = _finite(case["q_max"], f"cases[{index}].q_max")
        spacing = _finite(case["spacing"], f"cases[{index}].spacing")
        if spacing <= 0 or q_max <= q_min:
            raise ValueError("each case requires q_max>q_min and positive spacing")
        intervals = (q_max - q_min) / spacing
        if not math.isfinite(intervals) or not math.isclose(
            intervals, round(intervals), rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError("each grid range must contain an integer number of intervals")
        grid_size = int(round(intervals)) + 1
        points = grid_size * grid_size
        state_cells = 2 * points * points
        dtype = case["dtype"]
        if dtype not in {"float32", "float64"}:
            raise ValueError("case dtype must be float32 or float64")
        item_bytes = 4 if dtype == "float32" else 8
        density_bytes = state_cells * item_bytes
        steps = _integer(case["steps"], f"cases[{index}].steps")
        source_times = case["source_times"]
        if (
            not isinstance(source_times, list)
            or not source_times
            or len(source_times) > MAX_SUMMARY_COUNT
            or any(isinstance(t, bool) or not isinstance(t, int) for t in source_times)
            or sorted(set(source_times)) != source_times
            or source_times[0] < 0
            or source_times[-1] > steps
        ):
            raise ValueError("source_times must be a bounded sorted unique integer list within [0,T]")
        flat_chunk = _integer(case["flat_chunk_size"], "flat_chunk_size", 1)
        row_block = _integer(case["row_block_size"], "row_block_size", 1)
        column_block = _integer(case["column_block_size"], "column_block_size", 1)
        tolerance = _finite(case["diagnostic_tolerance"], "diagnostic_tolerance", nonnegative=True)
        symmetry = _finite(case["symmetry_tolerance"], "symmetry_tolerance", nonnegative=True)
        normalized_cases.append({
            "label": label, "q_min": q_min, "q_max": q_max, "spacing": spacing,
            "grid_size": grid_size, "agent_grid_points": points,
            "state_expanded_cells": state_cells, "density_bytes": density_bytes,
            "dtype": dtype, "item_bytes": item_bytes, "steps": steps,
            "source_times": list(source_times), "summary_count": len(source_times),
            "flat_chunk_size": flat_chunk, "row_block_size": row_block,
            "column_block_size": column_block, "diagnostic_tolerance": tolerance,
            "symmetry_tolerance": symmetry,
        })
        total_density_bytes += density_bytes

    run_name = _safe_text(config["output"]["run_name"], "output.run_name", MAX_RUN_NAME_LENGTH)
    normalized = {
        "model": {"alpha": alpha, "tau": tau, "state_probabilities": probabilities},
        "benchmark": {"warmup": warmup, "repetitions": repetitions, "safety_margin_fraction": margin},
        "cases": normalized_cases,
        "feasibility": {
            "grid_size": 131,
            "representative_steps": representative_steps,
            "representative_summary_count": representative_summaries,
        },
        "output": {"run_name": run_name},
    }
    encoded = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("ascii")) > MAX_SERIALIZED_METADATA_BYTES // 2:
        raise ValueError("normalized configuration exceeds its fixed serialized allowance")
    return {
        "normalized_configuration": normalized,
        "cases": normalized_cases,
        "total_density_bytes": total_density_bytes,
        "timing_record_count": 4 * len(normalized_cases),
        "timing_sample_count": (4 * len(normalized_cases) + 2) * repetitions,
    }


def validate_static_budget(raw: dict, allow_expensive: bool, *, limits: dict | None = None) -> dict:
    fixed = {
        "warmup": MAX_WARMUP, "repetitions": MAX_REPETITIONS,
        "grid_size": MAX_GRID_SIZE, "agent_grid_points": MAX_AGENT_GRID_POINTS,
        "state_expanded_cells": MAX_STATE_EXPANDED_CELLS,
        "density_bytes": MAX_DENSITY_BYTES, "steps": MAX_STEPS,
        "summary_count": MAX_SUMMARY_COUNT, "block_size": MAX_BLOCK_SIZE,
        "timing_records": MAX_TIMING_RECORDS,
        "timing_samples": MAX_TIMING_SAMPLES,
        "total_density_bytes": MAX_TOTAL_CASE_DENSITY_BYTES,
        "static_device_bytes": MAX_STATIC_DEVICE_BYTES,
        "static_host_bytes": MAX_STATIC_HOST_BYTES,
        "static_combined_bytes": MAX_STATIC_COMBINED_BYTES,
    }
    effective = fixed if limits is None else dict(limits)
    if set(effective) != set(fixed):
        raise ValueError("test limits must define the complete immutable cap set")
    config = raw["normalized_configuration"]
    violations = []
    values = {
        "warmup": config["benchmark"]["warmup"],
        "repetitions": config["benchmark"]["repetitions"],
        "timing_records": raw["timing_record_count"],
        "timing_samples": raw["timing_sample_count"],
        "total_density_bytes": raw["total_density_bytes"],
    }
    estimates = []
    for case in raw["cases"]:
        sep_bounded = estimate_separable_resources(
            grid_size=case["grid_size"], dtype_bytes=case["item_bytes"],
            steps=case["steps"], summary_count=case["summary_count"],
            row_block_size=case["row_block_size"],
            column_block_size=case["column_block_size"], return_final_density=False,
        )
        sep_full = estimate_separable_resources(
            grid_size=case["grid_size"], dtype_bytes=case["item_bytes"],
            steps=case["steps"], summary_count=case["summary_count"],
            row_block_size=case["row_block_size"],
            column_block_size=case["column_block_size"], return_final_density=True,
        )
        flat_bounded = estimate_flat_resources(
            grid_size=case["grid_size"], dtype_bytes=case["item_bytes"],
            steps=case["steps"], summary_count=case["summary_count"],
            chunk_size=case["flat_chunk_size"], return_final_density=False,
        )
        flat_full = estimate_flat_resources(
            grid_size=case["grid_size"], dtype_bytes=case["item_bytes"],
            steps=case["steps"], summary_count=case["summary_count"],
            chunk_size=case["flat_chunk_size"], return_final_density=True,
        )
        estimates.append({
            "label": case["label"],
            "separable_bounded": sep_bounded,
            "separable_full_validation": sep_full,
            "flat_bounded": flat_bounded,
            "flat_full_validation": flat_full,
        })
        static_device = max(
            int(value["static_device_bytes"])
            for value in (sep_bounded, sep_full, flat_bounded, flat_full)
        )
        static_host = max(
            int(value["static_host_with_compilation_bytes"])
            for value in (sep_bounded, sep_full, flat_bounded, flat_full)
        ) + HLO_LIVE_ALLOWANCE_BYTES
        case_values = {
            "grid_size": case["grid_size"], "agent_grid_points": case["agent_grid_points"],
            "state_expanded_cells": case["state_expanded_cells"],
            "density_bytes": case["density_bytes"], "steps": case["steps"],
            "summary_count": case["summary_count"],
            "block_size": max(case["row_block_size"], case["column_block_size"]),
            "static_device_bytes": static_device,
            "static_host_bytes": static_host,
            "static_combined_bytes": static_device + static_host,
        }
        for name, value in case_values.items():
            if value > effective[name]:
                violations.append(f"{case['label']}:{name}")
    for name, value in values.items():
        if value > effective[name]:
            violations.append(name)
    if violations and not allow_expensive:
        raise ValueError(
            "refusing benchmark before grid construction or compilation: "
            f"violations={violations}; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "absolute_limits": effective,
        "static_estimates": estimates,
        "violations": violations,
        "violations_overridden": violations if allow_expensive else [],
        "compiled_reports": [],
    }


def source_slots(source_times: list[int], steps: int) -> np.ndarray:
    slots = np.full(steps + 1, -1, dtype=np.int32)
    slots[np.asarray(source_times, dtype=np.int32)] = np.arange(len(source_times), dtype=np.int32)
    return slots


def _block_complete(value):
    """Synchronize every result leaf, not only a conveniently chosen scalar."""

    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        value,
    )


def _device_identity(device, visible_index: int, visible_count: int) -> dict:
    identity = {
        "backend": str(jax.default_backend()),
        "platform": str(device.platform),
        "visible_device_index": int(visible_index),
        "visible_device_count": int(visible_count),
        "id": str(device.id),
        "device_kind": str(device.device_kind),
        "process_index": int(getattr(device, "process_index", 0)),
        "local_hardware_id": int(getattr(device, "local_hardware_id", visible_index)),
    }
    for name in ("uuid", "pci_bus_id"):
        value = getattr(device, name, None)
        if value is not None:
            identity[name] = str(value)
    return identity


def _runtime_environment_signature() -> dict:
    devices = jax.devices()
    identities = [
        _device_identity(device, index, len(devices))
        for index, device in enumerate(devices)
    ]
    if not identities:
        raise RuntimeError("JAX reported no execution devices")
    allocator_environment = {
        name: os.environ[name]
        for name in (
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
        )
        if name in os.environ
    }
    return {
        "backend": str(jax.default_backend()),
        "platform": identities[0]["platform"],
        # Explicit dot-product precision is part of executable identity: the
        # same lowering at a different precision is a different numerical
        # object, so a change must invalidate compiled and prerequisite
        # provenance rather than pass silently.
        "pair_contraction_precision": pair_contraction_precision(),
        "execution_device": identities[0],
        "visible_devices": identities,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_version": str(jax.__version__),
        "jaxlib_version": str(jaxlib.__version__),
        "allocator_environment": allocator_environment,
    }


def _array_spec(value) -> dict:
    abstract = jax.core.get_aval(value)
    return {
        "shape": list(abstract.shape),
        "dtype": np.dtype(abstract.dtype).name,
        "weak_type": bool(getattr(abstract, "weak_type", False)),
    }


def _grid_argument_spec(grid) -> list[dict]:
    return [_array_spec(leaf) for leaf in jax.tree_util.tree_leaves(grid)]


def executable_signature(
    case: dict,
    model: dict,
    *,
    histogram,
    state_probabilities,
    grid,
    slots,
    kernel: str,
    output_mode: str,
    diagnostic_tolerance: float | None = None,
) -> dict:
    """Independently describe compile or invocation facts using actual arguments."""

    tolerance = (
        case["diagnostic_tolerance"]
        if diagnostic_tolerance is None
        else diagnostic_tolerance
    )
    environment = _runtime_environment_signature()
    abstract = {
        "histogram_mass": _array_spec(histogram),
        "state_probabilities": _array_spec(state_probabilities),
        "grid": _grid_argument_spec(grid),
        "alpha": _array_spec(model["alpha"]),
        "tau": _array_spec(model["tau"]),
        "source_slots": _array_spec(slots),
        "diagnostic_tolerance": _array_spec(tolerance),
    }
    static = {
        "kernel": kernel,
        "output_mode": output_mode,
        "steps": int(case["steps"]),
        "summary_count": int(case["summary_count"]),
        "chunk_size": int(case["flat_chunk_size"]),
        "row_block_size": int(case["row_block_size"]),
        "column_block_size": int(case["column_block_size"]),
    }
    return {
        "executable_id": f"pair-source-from-histogram:{output_mode}:v1",
        "kernel": kernel,
        "output_mode": output_mode,
        "grid_size": int(case["grid_size"]),
        "agent_grid_points": int(case["agent_grid_points"]),
        "state_expanded_cells": int(case["state_expanded_cells"]),
        "dtype": np.dtype(histogram.dtype).name,
        "histogram_argument": _array_spec(histogram),
        "state_probability_argument": _array_spec(state_probabilities),
        "grid_arguments": _grid_argument_spec(grid),
        "alpha": float(model["alpha"]),
        "tau": float(model["tau"]),
        "dynamic_scalar_arguments": {
            "alpha": abstract["alpha"],
            "tau": abstract["tau"],
            "diagnostic_tolerance": abstract["diagnostic_tolerance"],
        },
        "steps": int(case["steps"]),
        "summary_count": int(case["summary_count"]),
        "requested_source_times": list(case["source_times"]),
        "source_slots": np.asarray(slots, dtype=np.int32).tolist(),
        "source_slot_argument": _array_spec(slots),
        "chunk_size": int(case["flat_chunk_size"]),
        "row_block_size": int(case["row_block_size"]),
        "column_block_size": int(case["column_block_size"]),
        "diagnostic_tolerance": float(tolerance),
        "symmetry_tolerance": float(case["symmetry_tolerance"]),
        "contract_abstract_arguments": abstract,
        "contract_static_values": static,
        "contract_runtime_environment": environment,
        **environment,
    }


def _bundle_contract_parts(signature: dict) -> tuple[dict, dict, dict]:
    return (
        dict(signature["contract_abstract_arguments"]),
        dict(signature["contract_static_values"]),
        dict(signature["contract_runtime_environment"]),
    )


def _validate_bundle_contract(bundle: CompiledExecutableBundle) -> None:
    validate_compiled_executable_bundle(bundle)


def _bounded_bundle_description(bundle: CompiledExecutableBundle) -> dict:
    """Serialize bounded identity facts while the full contract remains runtime-only."""

    signature = bundle.compile_signature
    fields = (
        "executable_id",
        "kernel",
        "output_mode",
        "grid_size",
        "agent_grid_points",
        "state_expanded_cells",
        "dtype",
        "steps",
        "summary_count",
        "chunk_size",
        "row_block_size",
        "column_block_size",
        "backend",
        "platform",
        "execution_device",
        "jax_enable_x64",
        "jax_version",
        "jaxlib_version",
    )
    return {
        "identity": {name: signature[name] for name in fields if name in signature},
        "signature_sha256": bundle.signature_sha256,
        "bundle_integrity_sha256": bundle.bundle_integrity_sha256,
        "callable_retained_runtime_only": True,
        "complete_contract_serialized": False,
    }


def _bounded_memory_report(bundle: CompiledExecutableBundle) -> dict:
    report = {
        name: value
        for name, value in bundle.memory_report.items()
        if name != "executable_signature"
    }
    report["bundle_integrity_sha256"] = bundle.bundle_integrity_sha256
    return report


def _compile_and_analyze(
    lowered,
    signature: dict,
    *,
    static_host_bytes: int,
) -> tuple[CompiledExecutableBundle, float]:
    """Compile and analyze without invoking the executable."""

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start
    report = compiled_memory_report(compiled, static_host_bytes)
    abstract, static, environment = _bundle_contract_parts(signature)
    bundle = make_compiled_executable_bundle(
        compiled_callable=compiled,
        memory_report=report,
        compile_signature=signature,
        abstract_arguments=abstract,
        static_values=static,
        runtime_environment=environment,
    )
    return bundle, compile_seconds


def _validate_benchmark_bundle(
    bundle: CompiledExecutableBundle,
    static_estimate: dict,
    *,
    label: str,
    allow_expensive: bool,
) -> list[str]:
    """Gate identity and complete analysis before any executable invocation."""

    try:
        _validate_bundle_contract(bundle)
    except (TypeError, ValueError) as error:
        raise ValueError(f"compiled executable identity failed closed: {label}: {error}") from error
    report = bundle.memory_report
    if report.get("available") is not True or report.get("analysis_status") != "complete":
        raise ValueError(f"compiled memory analysis failed closed before execution: {label}")
    fields = (
        "argument_bytes",
        "output_bytes",
        "temporary_bytes",
        "alias_bytes",
        "host_argument_bytes",
        "host_output_bytes",
        "host_temporary_bytes",
        "host_alias_bytes",
        "compiled_device_requirement_bytes",
        "compiled_host_requirement_bytes",
        "static_host_allowance_bytes",
        "compiled_plus_host_requirement_bytes",
    )
    if any(
        isinstance(report.get(name), bool)
        or not isinstance(report.get(name), int)
        or report[name] < 0
        for name in fields
    ):
        raise ValueError(f"compiled memory analysis is incomplete or malformed: {label}")
    device = report["argument_bytes"] + report["output_bytes"] + report[
        "temporary_bytes"
    ] - report["alias_bytes"]
    host = report["host_argument_bytes"] + report["host_output_bytes"] + report[
        "host_temporary_bytes"
    ] - report["host_alias_bytes"]
    if (
        report["alias_bytes"] > report["argument_bytes"] + report["output_bytes"]
        or report["host_alias_bytes"]
        > report["host_argument_bytes"] + report["host_output_bytes"]
        or report["compiled_device_requirement_bytes"] != device
        or report["compiled_host_requirement_bytes"] != host
        or report["compiled_plus_host_requirement_bytes"]
        != device + host + report["static_host_allowance_bytes"]
    ):
        raise ValueError(f"compiled memory analysis is internally inconsistent: {label}")
    violations = []
    if report["compiled_device_requirement_bytes"] > int(
        static_estimate["static_device_bytes"]
    ):
        violations.append(f"{label}:compiled_device_exceeds_static_bound")
    if report["compiled_device_requirement_bytes"] > MAX_COMPILED_DEVICE_BYTES:
        violations.append(f"{label}:compiled_device_bytes")
    if report["compiled_plus_host_requirement_bytes"] > MAX_COMPILED_COMBINED_BYTES:
        violations.append(f"{label}:compiled_combined_bytes")
    if violations:
        raise ValueError(
            "compiled resource analysis rejected before execution and is not "
            f"overridable: {violations}"
        )
    return violations


def _invoke_accepted_bundle(
    bundle: CompiledExecutableBundle,
    arguments: tuple,
    *,
    case: dict,
    model: dict,
    kernel: str,
    output_mode: str,
    diagnostic_tolerance: float,
    production_feasibility: dict | None = None,
    capacity_observation=None,
    allow_expensive: bool = False,
    measure_execution: bool = False,
):
    """Rebuild runtime identity, optionally recheck capacity, invoke, and sync."""

    _validate_bundle_contract(bundle)
    if len(arguments) != 6:
        raise ValueError("pair executable requires exactly six positional arguments")
    histogram, states, grid, alpha, tau, slots = arguments
    actual_model = {**model, "alpha": alpha, "tau": tau}
    invocation_signature = executable_signature(
        case,
        actual_model,
        histogram=histogram,
        state_probabilities=states,
        grid=grid,
        slots=slots,
        kernel=kernel,
        output_mode=output_mode,
        diagnostic_tolerance=diagnostic_tolerance,
    )
    if invocation_signature != bundle.compile_signature:
        raise ValueError("invocation signature does not match the analyzed executable")
    if (production_feasibility is None) != (capacity_observation is None):
        raise ValueError("production feasibility and capacity evidence must be supplied together")
    if production_feasibility is not None:
        # This live recheck is intentionally adjacent to execution so the fixed
        # 60-second evidence age cannot silently expire after compilation/waiting.
        production_capacity_preflight(
            feasibility=production_feasibility,
            bundle=bundle,
            capacity_observation=capacity_observation,
            allow_expensive=allow_expensive,
        )
    start = time.perf_counter() if measure_execution else None
    result = bundle.compiled_callable(
        *arguments, diagnostic_tolerance=diagnostic_tolerance
    )
    result = _block_complete(result)
    if start is not None:
        return result, time.perf_counter() - start
    return result


def _max_tree_difference(left, right) -> float:
    maxima = []
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jnp.inexact):
            maxima.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)), initial=0.0)))
        else:
            maxima.append(0.0 if np.array_equal(np.asarray(a), np.asarray(b)) else math.inf)
    return max(maxima, default=0.0)


def _validate_scientific_result(result, case: dict, *, has_final_mass: bool) -> None:
    tolerance = case["diagnostic_tolerance"]
    symmetry_tolerance = case["symmetry_tolerance"]
    diagnostics = jax.device_get(result.diagnostics)
    if not bool(np.all(np.asarray(result.destinations_valid, dtype=bool))):
        raise ValueError("a projected destination left the configured grid")
    if not bool(np.all(np.asarray(diagnostics.finite, dtype=bool))):
        raise ValueError("pair trajectory contains non-finite mass")
    if not bool(np.all(np.asarray(diagnostics.nonnegative, dtype=bool))):
        raise ValueError("pair trajectory contains negative mass")
    if not bool(np.all(np.asarray(diagnostics.conditional_moments_valid, dtype=bool))):
        raise ValueError("pair trajectory contains invalid conditional moments")
    if np.any(np.abs(np.asarray(diagnostics.total_mass) - 1.0) > tolerance):
        raise ValueError("pair trajectory mass error exceeds diagnostic tolerance")
    if np.any(np.asarray(diagnostics.symmetry_error) > symmetry_tolerance):
        raise ValueError("pair trajectory symmetry error exceeds its tolerance")
    if has_final_mass:
        mass = np.asarray(jax.device_get(result.final_mass))
        if not np.all(np.isfinite(mass)) or np.min(mass, initial=0.0) < -tolerance:
            raise ValueError("final pair mass is non-finite or negative")
        if abs(float(mass.sum()) - 1.0) > tolerance:
            raise ValueError("final pair mass is not normalized")
        symmetry_error = float(
            np.max(np.abs(mass - mass.transpose(0, 2, 1)), initial=0.0)
        )
        if symmetry_error > symmetry_tolerance:
            raise ValueError("final pair mass lost endpoint exchange symmetry")


def _bounded_hlo_facts(hlo_text: str, state_cells: int) -> tuple[str, bool]:
    if len(hlo_text) > MAX_HLO_TEXT_CHARS:
        raise ValueError("lowered HLO text exceeds the immutable host-text cap")
    digest = hashlib.sha256()
    for offset in range(0, len(hlo_text), HLO_HASH_CHUNK_CHARS):
        digest.update(hlo_text[offset : offset + HLO_HASH_CHUNK_CHARS].encode("utf-8"))
    contains_flat = bool(
        re.search(rf"tensor<{state_cells}x4x", hlo_text)
        or re.search(rf"\[{state_cells},4\]", hlo_text)
    )
    return digest.hexdigest(), contains_flat


def _reduction_signature(
    *,
    name: str,
    points: int,
    values,
    destinations,
) -> dict:
    environment = _runtime_environment_signature()
    abstract = {
        "values": _array_spec(values),
        "destinations": _array_spec(destinations),
    }
    static = {
        "kernel": name,
        "output_mode": "reduction_microbenchmark",
        "point_count": int(points),
        "cell_count": int(points * points),
    }
    return {
        "executable_id": f"pair-reduction:{name}:v1",
        "kernel": name,
        "output_mode": "reduction_microbenchmark",
        "dtype": np.dtype(values.dtype).name,
        "point_count": int(points),
        "cell_count": int(points * points),
        "value_argument": abstract["values"],
        "destination_argument": abstract["destinations"],
        "contract_abstract_arguments": abstract,
        "contract_static_values": static,
        "contract_runtime_environment": environment,
        **environment,
    }


def _invoke_reduction_bundle(
    bundle: CompiledExecutableBundle,
    arguments: tuple,
    *,
    name: str,
    points: int,
    measure_execution: bool = False,
):
    validate_compiled_executable_bundle(bundle)
    if len(arguments) != 2:
        raise ValueError("reduction executable requires two positional arguments")
    invocation = _reduction_signature(
        name=name,
        points=points,
        values=arguments[0],
        destinations=arguments[1],
    )
    if invocation != bundle.compile_signature:
        raise ValueError("reduction invocation signature does not match its bundle")
    start = time.perf_counter() if measure_execution else None
    result = _block_complete(bundle.compiled_callable(*arguments))
    if start is not None:
        return result, time.perf_counter() - start
    return result


def investigate_reduction_strategies(points: int, repetitions: int) -> dict:
    """Measure two fully analyzed, identity-bound reduction executables."""

    if repetitions < 2 or repetitions % 2:
        raise ValueError("reduction timing requires an even repetitions value >= 2")
    if not isinstance(points, int) or isinstance(points, bool) or not 1 <= points <= MAX_AGENT_GRID_POINTS:
        raise ValueError("reduction point count exceeds its immutable development cap")

    cell_count = points * points
    # Static Python-integer preflight precedes host inputs, lowering, or device work.
    static_device_bound = cell_count * 128 + STATIC_REDUCTION_FIXED_BYTES
    if static_device_bound > MAX_COMPILED_DEVICE_BYTES:
        raise ValueError("reduction static device estimate exceeds its immutable cap")
    host_values = np.linspace(0.0, 1.0, cell_count, dtype=np.float32)
    host_destinations = (
        np.arange(cell_count, dtype=np.int64) * 17 % cell_count
    ).astype(np.int32)
    abstract_values = jax.ShapeDtypeStruct(host_values.shape, host_values.dtype)
    abstract_destinations = jax.ShapeDtypeStruct(
        host_destinations.shape, host_destinations.dtype
    )

    def scatter(data, indices):
        return jnp.zeros((cell_count,), dtype=data.dtype).at[indices].add(data)

    def sorted_segment(data, indices):
        order = jnp.argsort(indices)
        return jax.ops.segment_sum(
            data[order],
            indices[order],
            cell_count,
            indices_are_sorted=True,
        )

    results = {}
    outputs = {}
    compiled_records = {}
    for name, function in (("scatter", scatter), ("sorted_segment", sorted_segment)):
        lowered = jax.jit(function).lower(
            abstract_values,
            abstract_destinations,
        )
        signature = _reduction_signature(
            name=name,
            points=points,
            values=abstract_values,
            destinations=abstract_destinations,
        )
        bundle, compile_seconds = _compile_and_analyze(
            lowered,
            signature,
            static_host_bytes=host_values.nbytes + host_destinations.nbytes,
        )
        _validate_benchmark_bundle(
            bundle,
            {"static_device_bytes": static_device_bound},
            label=f"reduction:{name}",
            allow_expensive=False,
        )
        compiled_records[name] = {
            "bundle": bundle,
            "compile_seconds": compile_seconds,
            "memory_report": _bounded_memory_report(bundle),
            "timings": [],
            "positions": [],
        }
    # Device inputs are intentionally constructed only after both reports pass.
    values = jnp.asarray(host_values)
    destinations = jnp.asarray(host_destinations)
    for record in compiled_records.values():
        name = record["bundle"].compile_signature["kernel"]
        _invoke_reduction_bundle(
            record["bundle"], (values, destinations), name=name, points=points
        )
    execution_order = []
    for repetition in range(repetitions):
        order = (
            ("scatter", "sorted_segment")
            if repetition % 2 == 0
            else ("sorted_segment", "scatter")
        )
        execution_order.append(list(order))
        for position, name in enumerate(order):
            record = compiled_records[name]
            output, elapsed = _invoke_reduction_bundle(
                record["bundle"],
                (values, destinations),
                name=name,
                points=points,
                measure_execution=True,
            )
            record["timings"].append(elapsed)
            record["positions"].append(position)
            outputs[name] = output
    for name, record in compiled_records.items():
        timings = np.asarray(record["timings"], dtype=np.float64)
        median = float(np.median(timings))
        results[name] = {
            "compile_seconds": record["compile_seconds"],
            "timing_samples_seconds": record["timings"],
            "timing_order_positions": record["positions"],
            "median_execution_seconds": median,
            "minimum_execution_seconds": float(np.min(timings)),
            "maximum_execution_seconds": float(np.max(timings)),
            "mad_execution_seconds": float(np.median(np.abs(timings - median))),
            "memory_report": record["memory_report"],
        }
    results["max_abs_parity_error"] = float(
        np.max(
            np.abs(
                np.asarray(outputs["scatter"])
                - np.asarray(outputs["sorted_segment"])
            ),
            initial=0.0,
        )
    )
    results["adopted"] = False
    results["execution_order"] = execution_order
    results["decision"] = (
        "sorted segment was not adopted because this measured branch reduction "
        "requires an order vector and a larger compiled temporary peak"
    )
    return results


def benchmark_case(
    case: dict,
    model: dict,
    benchmark: dict,
    *,
    allow_expensive: bool = False,
) -> tuple[list[dict], dict]:
    """Compile/gate all exact objects, then warm and counterbalance execution."""

    dtype = jnp.float32 if case["dtype"] == "float32" else jnp.float64
    if dtype == jnp.float64 and not jax.config.read("jax_enable_x64"):
        raise ValueError("float64 benchmark cases require JAX_ENABLE_X64=1 before import")
    grid = QGrid(case["q_min"], case["q_max"], case["spacing"])
    if grid.size != case["grid_size"]:
        raise RuntimeError("QGrid size disagrees with allocation-free preflight")
    jax_grid = build_jax_pair_grid(grid, dtype)
    histogram = tiny_histogram(grid).mass.reshape(-1).astype(np.dtype(dtype))
    histogram_device = jnp.asarray(histogram)
    state_device = jnp.asarray(model["state_probabilities"], dtype=dtype)
    slots = jnp.asarray(source_slots(case["source_times"], case["steps"]))
    warmup = benchmark["warmup"]
    repetitions = benchmark["repetitions"]
    if repetitions < 2 or repetitions % 2:
        raise ValueError("counterbalanced timing requires an even repetitions value >= 2")
    rows = []
    results = {"full_validation": {}, "bounded_from_histogram": {}}
    reports = {}
    compiled_violations = []
    records = {}

    for kernel in ("flat", "separable"):
        resource_function = (
            estimate_flat_resources if kernel == "flat" else estimate_separable_resources
        )
        resource_arguments = dict(
            grid_size=case["grid_size"], dtype_bytes=case["item_bytes"],
            steps=case["steps"], summary_count=case["summary_count"],
        )
        if kernel == "flat":
            resource_arguments["chunk_size"] = case["flat_chunk_size"]
        else:
            resource_arguments["row_block_size"] = case["row_block_size"]
            resource_arguments["column_block_size"] = case["column_block_size"]
        static = dict(
            steps=case["steps"], summary_count=case["summary_count"],
            chunk_size=case["flat_chunk_size"],
            diagnostic_tolerance=case["diagnostic_tolerance"], kernel=kernel,
            row_block_size=case["row_block_size"],
            column_block_size=case["column_block_size"],
        )
        for output_mode, function, return_final in (
            (
                "full_validation",
                simulate_pair_source_summaries_from_histogram_full_jit,
                True,
            ),
            (
                "bounded_from_histogram",
                simulate_pair_source_summaries_from_histogram_jit,
                False,
            ),
        ):
            abstract_histogram = jax.ShapeDtypeStruct(histogram_device.shape, dtype)
            abstract_states = jax.ShapeDtypeStruct(state_device.shape, dtype)
            lowered = function.lower(
                abstract_histogram,
                abstract_states,
                jax_grid,
                model["alpha"],
                model["tau"],
                slots,
                **static,
            )
            hlo_text = lowered.as_text()
            hlo_sha, contains_D4 = _bounded_hlo_facts(
                hlo_text, case["state_expanded_cells"]
            )
            del hlo_text
            static_estimate = resource_function(
                **resource_arguments, return_final_density=return_final
            )
            static_host = (
                int(static_estimate["host_planning_threshold_bytes"])
                + HLO_LIVE_ALLOWANCE_BYTES
            )
            signature = executable_signature(
                case,
                model,
                histogram=abstract_histogram,
                state_probabilities=abstract_states,
                grid=jax_grid,
                slots=slots,
                kernel=kernel,
                output_mode=output_mode,
            )
            bundle, compile_seconds = _compile_and_analyze(
                lowered, signature, static_host_bytes=static_host
            )
            label = f"{case['label']}:{kernel}:{output_mode}"
            violations = _validate_benchmark_bundle(
                bundle,
                static_estimate,
                label=label,
                allow_expensive=allow_expensive,
            )
            compiled_violations.extend(violations)
            key = (output_mode, kernel)
            records[key] = {
                "bundle": bundle,
                "compile_seconds": compile_seconds,
                "static": static_estimate,
                "hlo_sha256": hlo_sha,
                "contains_flat_D_by_4_shape": contains_D4,
                "arguments": (
                    histogram_device,
                    state_device,
                    jax_grid,
                    model["alpha"],
                    model["tau"],
                    slots,
                ),
                "samples": [],
                "positions": [],
            }
            report_key = kernel if output_mode == "full_validation" else f"{kernel}_bounded"
            reports[report_key] = _bounded_memory_report(bundle)

    # All four exact combined initializer/scan objects have now passed identity
    # and compiled-memory gates. Only now may any full pair density be created.
    for record in records.values():
        for _ in range(warmup):
            _invoke_accepted_bundle(
                record["bundle"],
                record["arguments"],
                case=case,
                model=model,
                kernel=record["bundle"].compile_signature["kernel"],
                output_mode=record["bundle"].compile_signature["output_mode"],
                diagnostic_tolerance=case["diagnostic_tolerance"],
            )

    execution_orders = {mode: [] for mode in results}
    for repetition in range(repetitions):
        order = ("flat", "separable") if repetition % 2 == 0 else ("separable", "flat")
        for output_mode in results:
            execution_orders[output_mode].append(list(order))
            for position, kernel in enumerate(order):
                record = records[(output_mode, kernel)]
                result, elapsed = _invoke_accepted_bundle(
                    record["bundle"],
                    record["arguments"],
                    case=case,
                    model=model,
                    kernel=kernel,
                    output_mode=output_mode,
                    diagnostic_tolerance=case["diagnostic_tolerance"],
                    measure_execution=True,
                )
                record["samples"].append(elapsed)
                record["positions"].append(position)
                results[output_mode][kernel] = result

    density = case["density_bytes"]
    for output_mode in ("full_validation", "bounded_from_histogram"):
        for kernel in ("flat", "separable"):
            record = records[(output_mode, kernel)]
            result = results[output_mode][kernel]
            has_final = output_mode == "full_validation"
            _validate_scientific_result(result, case, has_final_mass=has_final)
            if not has_final and hasattr(result, "final_mass"):
                raise RuntimeError("bounded production object returned a full density")
            samples = np.asarray(record["samples"], dtype=np.float64)
            median = float(np.median(samples))
            mad = float(np.median(np.abs(samples - median)))
            report = record["bundle"].memory_report
            rows.append(
                {
                    "case": case["label"],
                    "dtype": case["dtype"],
                    "kernel": kernel,
                    "output_mode": output_mode,
                    "G": case["grid_size"],
                    "M": case["agent_grid_points"],
                    "D": case["state_expanded_cells"],
                    "steps": case["steps"],
                    "summary_count": case["summary_count"],
                    "row_block_size": case["row_block_size"],
                    "column_block_size": case["column_block_size"],
                    "flat_chunk_size": case["flat_chunk_size"],
                    "compile_seconds": record["compile_seconds"],
                    "warmup_count": warmup,
                    "repetitions": repetitions,
                    "timing_samples_seconds": json.dumps(record["samples"]),
                    "timing_order_positions": json.dumps(record["positions"]),
                    "execution_orders": json.dumps(execution_orders[output_mode]),
                    "median_execution_seconds": median,
                    "minimum_execution_seconds": float(np.min(samples)),
                    "maximum_execution_seconds": float(np.max(samples)),
                    "mad_execution_seconds": mad,
                    "execution_seconds_per_step": median / max(case["steps"], 1),
                    "argument_bytes": report["argument_bytes"],
                    "output_bytes": report["output_bytes"],
                    "temporary_bytes": report["temporary_bytes"],
                    "alias_bytes": report["alias_bytes"],
                    "compiled_device_requirement_bytes": report[
                        "compiled_device_requirement_bytes"
                    ],
                    "compiled_bytes_per_Db": report[
                        "compiled_device_requirement_bytes"
                    ]
                    / density,
                    "executable_signature_sha256": record["bundle"].signature_sha256,
                    "hlo_sha256": record["hlo_sha256"],
                    "contains_flat_D_by_4_shape": record[
                        "contains_flat_D_by_4_shape"
                    ],
                }
            )

    for kernel in ("flat", "separable"):
        bounded = results["bounded_from_histogram"][kernel]
        full = results["full_validation"][kernel]
        if (
            _max_tree_difference(bounded.source_summaries, full.source_summaries)
            > case["diagnostic_tolerance"]
        ):
            raise RuntimeError("bounded and full source summaries disagree")

    flat = results["full_validation"]["flat"]
    separable = results["full_validation"]["separable"]
    parity = {
        "final_density_max_abs": float(np.max(np.abs(np.asarray(flat.final_mass) - np.asarray(separable.final_mass)), initial=0.0)),
        "source_summary_max_abs": _max_tree_difference(flat.source_summaries, separable.source_summaries),
        "diagnostic_max_abs": _max_tree_difference(flat.diagnostics, separable.diagnostics),
        "final_mass": float(np.asarray(separable.final_mass).sum()),
        "final_symmetry_error": float(np.max(np.abs(np.asarray(separable.final_mass) - np.asarray(separable.final_mass).transpose(0, 2, 1)), initial=0.0)),
        "final_minimum_mass": float(np.min(np.asarray(separable.final_mass), initial=0.0)),
    }
    if max(parity["final_density_max_abs"], parity["source_summary_max_abs"], parity["diagnostic_max_abs"]) > case["diagnostic_tolerance"]:
        raise RuntimeError(f"flat/separable parity exceeds tolerance: {parity}")
    return rows, {
        "case": case["label"],
        "parity": parity,
        "compiled_reports": reports,
        "compiled_violations": compiled_violations,
        "compiled_violations_overridden": (
            compiled_violations if allow_expensive else []
        ),
        "execution_orders": execution_orders,
        "bundle_descriptions": {
            f"{mode}:{kernel}": _bounded_bundle_description(record["bundle"])
            for (mode, kernel), record in records.items()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(SOURCE_HASH_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _bounded_environment_metadata() -> tuple[str, list[str]]:
    backend = str(jax.default_backend())
    devices = [str(device) for device in jax.devices()]
    if len(backend) > MAX_DEVICE_TEXT_CHARS:
        raise ValueError("backend description exceeds its metadata bound")
    if len(devices) > MAX_DEVICE_COUNT or any(
        len(device) > MAX_DEVICE_TEXT_CHARS for device in devices
    ):
        raise ValueError("device metadata exceeds its fixed count/text bounds")
    return backend, devices


def _write_outputs(directory: Path, rows: list[dict], metadata: dict) -> None:
    metadata_text = json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if len(metadata_text.encode("ascii")) > MAX_SERIALIZED_METADATA_BYTES:
        raise RuntimeError("metadata exceeds fixed serialized bound")
    if len(rows) > MAX_TIMING_RECORDS:
        raise RuntimeError("timing record count exceeds fixed bound")
    sink = []
    for row in rows:
        probe = json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        if len(probe.encode("ascii")) > MAX_CSV_RECORD_BYTES:
            raise RuntimeError("CSV record exceeds fixed serialized bound")
        sink.append(row)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "benchmark.csv").open("w", newline="", encoding="ascii") as file:
        writer = csv.DictWriter(file, fieldnames=list(sink[0]))
        writer.writeheader()
        writer.writerows(sink)
    (directory / "metadata.json").write_text(metadata_text, encoding="ascii")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    raw = inspect_raw_config(load_config(config_path))
    budget = validate_static_budget(raw, args.allow_expensive)
    normalized = raw["normalized_configuration"]
    rows = []
    case_metadata = []
    for case in raw["cases"]:
        case_rows, case_result = benchmark_case(
            case,
            normalized["model"],
            normalized["benchmark"],
            allow_expensive=args.allow_expensive,
        )
        rows.extend(case_rows)
        case_metadata.append(case_result)
    reports = [
        report
        for case in case_metadata
        for report in case["compiled_reports"].values()
    ]
    budget["compiled_reports"] = reports
    separable_ratios = [
        row["compiled_bytes_per_Db"]
        for row in rows
        if row["kernel"] == "separable" and row["output_mode"] == "bounded_from_histogram"
        and row["compiled_bytes_per_Db"] is not None
    ]
    if not separable_ratios:
        raise ValueError(
            "no complete separable compiled-memory analysis is available for the "
            "G=131 validated-bound projection"
        )
    validated_ratio = max(separable_ratios)
    feasibility = {
        dtype: full_grid_feasibility(
            dtype_bytes=4 if dtype == "float32" else 8,
            representative_steps=normalized["feasibility"]["representative_steps"],
            representative_summary_count=normalized["feasibility"]["representative_summary_count"],
            row_block_size=max(case["row_block_size"] for case in raw["cases"]),
            column_block_size=max(case["column_block_size"] for case in raw["cases"]),
            validated_compiled_bytes_per_density_byte=validated_ratio,
            safety_margin_fraction=normalized["benchmark"]["safety_margin_fraction"],
        )
        for dtype in ("float32", "float64")
    }
    reduction_investigation = investigate_reduction_strategies(
        max(case["agent_grid_points"] for case in raw["cases"]),
        normalized["benchmark"]["repetitions"],
    )
    backend, devices = _bounded_environment_metadata()
    git_status = _git("status", "--short", "--", ".")
    if len(git_status) > MAX_GIT_STATUS_CHARS:
        raise ValueError("Git status exceeds its fixed provenance-text bound")
    output_root = (PROJECT_ROOT / "outputs" / "pair_separable_benchmark").resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = (output_root / f"{normalized['output']['run_name']}-{timestamp}").resolve()
    if output_root not in directory.parents:
        raise ValueError("unsafe output path")
    source_paths = {
        "src/chu_pair/pair_density/jax_solver.py": PROJECT_ROOT / "src/chu_pair/pair_density/jax_solver.py",
        "src/chu_pair/pair_density/separable_resources.py": PROJECT_ROOT / "src/chu_pair/pair_density/separable_resources.py",
        "experiments/run_pair_separable_benchmark.py": Path(__file__).resolve(),
        "configuration_file": config_path,
    }
    metadata = {
        "schema_version": 1,
        "milestone": "exact separable JAX pair transport bounded CPU benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": normalized,
        "resource_budget": budget,
        "admission_policy": {
            "overridable_development_limits": [
                "normalized bounded static size caps",
                "bounded repetition/timing-row/timing-sample caps",
                "bounded aggregate development density allowance",
            ],
            "never_overridable": [
                "runtime bundle identity or completeness",
                "live compiled-memory analysis validity",
                "invocation signature mismatch",
                "wrong kernel, backend, or execution device",
                "ambiguous, missing, stale, mismatched, or insufficient capacity evidence",
                "scientific diagnostics",
            ],
        },
        "cases": case_metadata,
        "full_grid_feasibility_only": feasibility,
        "backend": backend,
        "devices": devices,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "jax": jax.__version__, "jaxlib": jaxlib.__version__},
        "git": {"commit": _git("rev-parse", "HEAD"), "subproject_status_before_output": git_status},
        "source_hashes": {label: _sha256(path) for label, path in source_paths.items()},
        "timing_scope": "warm CPU wall time on the reported backend; not a GPU performance claim",
        "accumulation_order": "scatter-add collision order is backend dependent; toleranced, not bitwise, parity",
        "sorted_reduction_investigation": reduction_investigation,
        "limitations": ["no G=131 allocation or compilation", "no GPU validation", "no interpolation", "no production inference"],
    }
    _write_outputs(directory, rows, metadata)
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"wrote {len(rows)} bounded benchmark rows to {directory}")


if __name__ == "__main__":
    main()

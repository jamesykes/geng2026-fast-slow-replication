#!/usr/bin/env python3
"""Run a guarded CPU-safe Phase 4 JAX pair-density smoke calculation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

from chu_pair.config import LearningConfig
from chu_pair.grids import QGrid
from chu_pair.initial_conditions import tiny_histogram
from chu_pair.pair_density.jax_solver import (
    build_jax_pair_grid,
    checked_simulate_pair_density,
    ordered_pair_mass_jax,
    pair_diagnostics_jax,
    simulate_pair_density_jit,
    validate_jax_pair_mass,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pair_jax_small.toml"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_RUN_NAME_LENGTH = 64
MAX_CONFIG_INTEGER = 2**63 - 1
MAX_NORMALIZED_CONFIGURATION_JSON_CHARS = 4 * 1024
MAX_GIT_STATUS_CHARS = 2 * 1024
MAX_DEVICE_COUNT = 8
MAX_DEVICE_FIELD_CHARS = 128
MAX_COMPILED_REASON_CHARS = 1024
MAX_JSON_BYTES_PER_CHARACTER = 12
FIXED_METADATA_JSON_CHARS = 32 * 1024
MAX_CSV_WRITE_CHARS = 4 * 1024
SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR = 8
SERIALIZATION_CHUNK_CHARS = 4 * 1024
SERIALIZATION_IO_BUFFER_BYTES = 8 * 1024
SERIALIZATION_FIXED_OVERHEAD_BYTES = 64 * 1024
MAX_SERIALIZATION_LIVE_PEAK_BYTES = 3 * 1024**2
PHASE4_ABSOLUTE_LIMITS = {
    "agent_grid_points": 4_096,
    "ordered_pair_cells": 4_000_000,
    "state_expanded_cells": 8_000_000,
    "initial_pair_bytes": 64 * 1024**2,
    "combined_peak_bytes": 256 * 1024**2,
    "retained_full_density_snapshots": 0,
    "diagnostic_output_rows": 10_000,
}
DIAGNOSTIC_FLOAT_SCALARS = 11
DIAGNOSTIC_BOOL_SCALARS = 3
STATIC_FULL_DENSITY_DEVICE_COPIES = 8
PYTHON_DIAGNOSTIC_ROW_BYTES = 4_096
SOURCE_HASH_BUFFER_BYTES = 1 << 20

EXPECTED_CONFIG_KEYS = {
    "model": {"alpha", "tau"},
    "grid": {"q_min", "q_max", "spacing"},
    "solver": {
        "steps",
        "dtype",
        "chunk_size",
        "diagnostic_stride",
        "symmetry_tolerance",
        "diagnostic_tolerance",
    },
    "initial_condition": {"mode", "state_probabilities"},
    "output": {"run_name"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="override fixed Phase 4 resource caps while recording violations",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _require_exact_config_schema(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a table")
    if set(config) != set(EXPECTED_CONFIG_KEYS):
        unknown = sorted(set(config) - set(EXPECTED_CONFIG_KEYS))
        missing = sorted(set(EXPECTED_CONFIG_KEYS) - set(config))
        raise ValueError(
            f"configuration sections must match the Phase 4 schema; "
            f"unknown={unknown}, missing={missing}"
        )
    for section, expected_keys in EXPECTED_CONFIG_KEYS.items():
        table = config[section]
        if not isinstance(table, dict):
            raise ValueError(f"configuration section {section} must be a table")
        if set(table) != expected_keys:
            unknown = sorted(set(table) - expected_keys)
            missing = sorted(expected_keys - set(table))
            raise ValueError(
                f"configuration keys in {section} must match the Phase 4 schema; "
                f"unknown={unknown}, missing={missing}"
            )


def _finite_number(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or (nonnegative and normalized < 0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _bounded_integer(value, name: str, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > MAX_CONFIG_INTEGER
    ):
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {MAX_CONFIG_INTEGER}]"
        )
    return value


def _bounded_metadata_text(value, name: str, maximum: int) -> str:
    text = str(value)
    if len(text) > maximum:
        raise ValueError(f"{name} exceeds the metadata limit of {maximum} characters")
    return text


def _truncate_compiled_reason(reason: str) -> str:
    if len(reason) <= MAX_COMPILED_REASON_CHARS:
        return reason
    suffix = "... [truncated]"
    return reason[: MAX_COMPILED_REASON_CHARS - len(suffix)] + suffix


def metadata_json_text_bound(normalized_configuration_json_chars: int) -> int:
    """Maximum ASCII characters in the final normalized metadata JSON."""

    return (
        normalized_configuration_json_chars
        + MAX_GIT_STATUS_CHARS * MAX_JSON_BYTES_PER_CHARACTER
        + MAX_COMPILED_REASON_CHARS * MAX_JSON_BYTES_PER_CHARACTER
        + MAX_DEVICE_COUNT
        * 3
        * MAX_DEVICE_FIELD_CHARS
        * MAX_JSON_BYTES_PER_CHARACTER
        + FIXED_METADATA_JSON_CHARS
    )


def serialization_payload_bound(normalized_configuration_json_chars: int) -> int:
    """Maximum metadata JSON plus one streamed CSV header or record payload."""

    return (
        metadata_json_text_bound(normalized_configuration_json_chars)
        + MAX_CSV_WRITE_CHARS
    )


def serialization_live_peak_components(
    normalized_configuration_json_chars: int,
) -> dict[str, int]:
    """Conservative live-memory peaks for bounded JSON/CSV serialization.

    Eight bytes per source character avoids relying on CPython's compact ASCII
    storage.  JSON encoding allows the bounded input object/string storage, a
    complete set of escaped text fragments, and the joined result to coexist.
    During output, the complete metadata text coexists with at most one bounded
    CSV record or metadata chunk as text, its one-byte ASCII encoding, the
    explicitly sized binary file buffer, and fixed serializer/object overhead.
    The stages are sequential, so their peaks are alternatives rather than
    additive.
    """

    metadata_chars = metadata_json_text_bound(normalized_configuration_json_chars)
    metadata_encoding_peak = (
        3 * SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR * metadata_chars
        + SERIALIZATION_FIXED_OVERHEAD_BYTES
    )
    metadata_write_peak = (
        SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR * metadata_chars
        + SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR * SERIALIZATION_CHUNK_CHARS
        + SERIALIZATION_CHUNK_CHARS
        + SERIALIZATION_IO_BUFFER_BYTES
        + SERIALIZATION_FIXED_OVERHEAD_BYTES
    )
    csv_write_peak = (
        SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR * metadata_chars
        + SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR * MAX_CSV_WRITE_CHARS
        + MAX_CSV_WRITE_CHARS
        + SERIALIZATION_IO_BUFFER_BYTES
        + SERIALIZATION_FIXED_OVERHEAD_BYTES
    )
    return {
        "metadata_encoding_peak_bytes": metadata_encoding_peak,
        "metadata_write_peak_bytes": metadata_write_peak,
        "csv_write_peak_bytes": csv_write_peak,
    }


def serialization_live_peak_bound(normalized_configuration_json_chars: int) -> int:
    """Maximum of the audited sequential serialization-stage peaks."""

    return max(
        serialization_live_peak_components(
            normalized_configuration_json_chars
        ).values()
    )


def inspect_raw_pair_config(config: dict) -> dict:
    """Validate scalar structure and derive grid counts without array allocation."""

    _require_exact_config_schema(config)
    model = config["model"]
    grid = config["grid"]
    solver = config["solver"]
    initial = config["initial_condition"]
    output = config["output"]

    alpha = _finite_number(model["alpha"], "model.alpha", nonnegative=True)
    tau = _finite_number(model["tau"], "model.tau", nonnegative=True)
    q_min = _finite_number(grid["q_min"], "grid.q_min")
    q_max = _finite_number(grid["q_max"], "grid.q_max")
    spacing = _finite_number(grid["spacing"], "grid.spacing")
    if spacing <= 0 or q_max <= q_min:
        raise ValueError("grid requires q_max>q_min and positive spacing")
    intervals = (q_max - q_min) / spacing
    if not math.isfinite(intervals):
        raise ValueError("grid interval count must be finite")
    if not math.isclose(intervals, round(intervals), rel_tol=0, abs_tol=1e-10):
        raise ValueError("grid range must contain an integer number of intervals")
    for name, value in (("q_min", q_min), ("q_max", q_max)):
        ratio = value / spacing
        if not math.isfinite(ratio) or not math.isclose(
            ratio, round(ratio), rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"grid.{name} must align with legacy spacing multiples")

    integer_fields = {
        "steps": 0,
        "chunk_size": 1,
        "diagnostic_stride": 1,
    }
    normalized_integers = {}
    for name, minimum in integer_fields.items():
        normalized_integers[name] = _bounded_integer(
            solver[name], f"solver.{name}", minimum
        )
    dtype = solver["dtype"]
    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("solver.dtype must be 'float32' or 'float64'")
    symmetry_tolerance = _finite_number(
        solver["symmetry_tolerance"],
        "solver.symmetry_tolerance",
        nonnegative=True,
    )
    diagnostic_tolerance = _finite_number(
        solver["diagnostic_tolerance"],
        "solver.diagnostic_tolerance",
        nonnegative=True,
    )

    if not isinstance(initial["mode"], str) or initial["mode"] != "tiny_two_cell":
        raise ValueError("Phase 4 smoke runner requires tiny_two_cell initialization")
    probabilities = initial["state_probabilities"]
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in probabilities)
        or any(not math.isfinite(float(value)) or value < 0 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("initial state probabilities must be two non-negative values summing to one")

    run_name = output["run_name"]
    if (
        not isinstance(run_name, str)
        or len(run_name) > MAX_RUN_NAME_LENGTH
        or not RUN_NAME_PATTERN.fullmatch(run_name)
        or run_name in {".", ".."}
    ):
        raise ValueError(
            "output.run_name must be 1-64 safe ASCII characters, begin with an "
            "alphanumeric character, and contain only letters, digits, '.', '_', or '-'"
        )

    normalized_configuration = {
        "model": {"alpha": alpha, "tau": tau},
        "grid": {"q_min": q_min, "q_max": q_max, "spacing": spacing},
        "solver": {
            "steps": normalized_integers["steps"],
            "dtype": dtype,
            "chunk_size": normalized_integers["chunk_size"],
            "diagnostic_stride": normalized_integers["diagnostic_stride"],
            "symmetry_tolerance": symmetry_tolerance,
            "diagnostic_tolerance": diagnostic_tolerance,
        },
        "initial_condition": {
            "mode": "tiny_two_cell",
            "state_probabilities": [float(value) for value in probabilities],
        },
        "output": {"run_name": run_name},
    }
    encoded_configuration = json.dumps(
        normalized_configuration,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if not encoded_configuration.isascii():  # pragma: no cover - ensure_ascii contract
        raise RuntimeError("normalized configuration JSON is not ASCII")
    if len(encoded_configuration) > MAX_NORMALIZED_CONFIGURATION_JSON_CHARS:
        raise ValueError("normalized configuration exceeds its fixed JSON limit")
    metadata_text_bound = metadata_json_text_bound(len(encoded_configuration))
    payload_bound = serialization_payload_bound(len(encoded_configuration))
    serialization_peak = serialization_live_peak_bound(len(encoded_configuration))
    if serialization_peak > MAX_SERIALIZATION_LIVE_PEAK_BYTES:
        raise RuntimeError("Phase 4 serialization constants exceed their fixed allowance")

    grid_size = int(round(intervals)) + 1
    agent_points = grid_size * grid_size
    ordered_cells = agent_points * agent_points
    return {
        "q_min": q_min,
        "q_max": q_max,
        "spacing": spacing,
        "grid_size": grid_size,
        "agent_grid_points": agent_points,
        "ordered_pair_cells": ordered_cells,
        "state_expanded_cells": 2 * ordered_cells,
        "steps": normalized_integers["steps"],
        "dtype": dtype,
        "item_bytes": 4 if dtype == "float32" else 8,
        "chunk_size": normalized_integers["chunk_size"],
        "diagnostic_stride": normalized_integers["diagnostic_stride"],
        "symmetry_tolerance": symmetry_tolerance,
        "diagnostic_tolerance": diagnostic_tolerance,
        "state_probabilities": [float(value) for value in probabilities],
        "normalized_configuration": normalized_configuration,
        "normalized_configuration_json_chars": len(encoded_configuration),
        "metadata_json_text_bound_chars": metadata_text_bound,
        "serialization_payload_bound_bytes": payload_bound,
        "serialization_live_peak_bytes": serialization_peak,
    }


def estimate_pair_resources(raw: dict) -> dict:
    """Allocation-free conservative static device/host storage estimate."""

    points = int(raw["agent_grid_points"])
    ordered_cells = int(raw["ordered_pair_cells"])
    state_cells = int(raw["state_expanded_cells"])
    item_bytes = int(raw["item_bytes"])
    steps = int(raw["steps"])
    source_chunk = min(int(raw["chunk_size"]), state_cells)
    stride = int(raw["diagnostic_stride"])
    diagnostic_rows = 1 + steps // stride + int(steps % stride != 0)

    initial_pair_bytes = state_cells * item_bytes
    # Backend-independent allowance for input, scan carry, next/scatter output,
    # returned output, and compiler-created full-density loop/scatter temporaries.
    # A second-stage executable-memory check measures the actual compiled buffers
    # when the backend exposes them; this static allowance bounds compilation first.
    full_density_device_bytes = (
        STATIC_FULL_DENSITY_DEVICE_COPIES * state_cells * item_bytes
    )
    # Grid Q values, focal/policy/weight/moment/velocity arrays, and all pointwise
    # source/projected/destination/safe-destination indices and validity work.
    point_working_bytes = (
        int(raw["grid_size"]) * item_bytes
        + points * (20 * item_bytes + 40)
    )
    # Source mass, two four-branch policy gathers, branch probabilities, and
    # branch masses. Fusion may remove gathers, but preflight does not assume it.
    branch_weight_bytes = source_chunk * 17 * item_bytes
    # Four destination/state/index arrays and scalar source/state/endpoint/index
    # vectors, including the chunk-offset vector, with conservative byte rounding.
    branch_index_bytes = source_chunk * 96
    diagnostic_trajectory_bytes = steps * (
        DIAGNOSTIC_FLOAT_SCALARS * item_bytes + DIAGNOSTIC_BOOL_SCALARS
    )
    static_device_bytes = (
        full_density_device_bytes
        + point_working_bytes
        + branch_weight_bytes
        + branch_index_bytes
        + diagnostic_trajectory_bytes
    )

    # The complete diagnostic trajectory is copied to host and remains live while
    # the final full-density validation copy is checked. The histogram is float64.
    # Grid construction temporarily holds axis values, two int32 index meshes, and
    # the host Q-point table. Selected rows are retained as Python dictionaries;
    # hashing and normalized JSON/CSV serialization use bounded buffers.
    validation_host_copy_bytes = initial_pair_bytes
    diagnostic_host_trajectory_bytes = diagnostic_trajectory_bytes
    histogram_host_bytes = points * 8
    grid_construction_host_bytes = (
        int(raw["grid_size"]) * (16 + item_bytes)
        + points * (8 + 4 * item_bytes)
    )
    diagnostic_row_host_bytes = diagnostic_rows * PYTHON_DIAGNOSTIC_ROW_BYTES
    normalized_configuration_json_chars = int(
        raw.get(
            "normalized_configuration_json_chars",
            MAX_NORMALIZED_CONFIGURATION_JSON_CHARS,
        )
    )
    serialization_peak_bytes = int(
        raw.get(
            "serialization_live_peak_bytes",
            serialization_live_peak_bound(normalized_configuration_json_chars),
        )
    )
    serialization_peaks = serialization_live_peak_components(
        normalized_configuration_json_chars
    )
    output_host_bytes = serialization_peak_bytes + SOURCE_HASH_BUFFER_BYTES
    static_host_bytes = (
        validation_host_copy_bytes
        + diagnostic_host_trajectory_bytes
        + histogram_host_bytes
        + grid_construction_host_bytes
        + diagnostic_row_host_bytes
        + output_host_bytes
    )
    static_combined_bytes = static_device_bytes + static_host_bytes
    return {
        "grid_size": int(raw["grid_size"]),
        "agent_grid_points": points,
        "ordered_pair_cells": ordered_cells,
        "state_expanded_cells": state_cells,
        "dtype": str(raw["dtype"]),
        "item_bytes": item_bytes,
        "initial_pair_bytes": initial_pair_bytes,
        "static_device_bytes": static_device_bytes,
        "static_host_bytes": static_host_bytes,
        "static_combined_peak_bytes": static_combined_bytes,
        "retained_full_density_snapshots": 0,
        "diagnostic_output_rows": diagnostic_rows,
        "components": {
            "static_full_density_device_copies": STATIC_FULL_DENSITY_DEVICE_COPIES,
            "full_density_device_bytes": full_density_device_bytes,
            "point_working_bytes": point_working_bytes,
            "branch_weight_bytes": branch_weight_bytes,
            "branch_index_bytes": branch_index_bytes,
            "diagnostic_trajectory_bytes": diagnostic_trajectory_bytes,
            "effective_source_chunk_cells": source_chunk,
            "validation_host_copy_bytes": validation_host_copy_bytes,
            "diagnostic_host_trajectory_bytes": diagnostic_host_trajectory_bytes,
            "histogram_host_bytes": histogram_host_bytes,
            "grid_construction_host_bytes": grid_construction_host_bytes,
            "diagnostic_row_host_bytes": diagnostic_row_host_bytes,
            "python_diagnostic_row_bytes": PYTHON_DIAGNOSTIC_ROW_BYTES,
            "metadata_json_text_bound_chars": int(
                raw.get(
                    "metadata_json_text_bound_chars",
                    metadata_json_text_bound(normalized_configuration_json_chars),
                )
            ),
            "serialization_payload_bound_bytes": int(
                raw.get(
                    "serialization_payload_bound_bytes",
                    serialization_payload_bound(normalized_configuration_json_chars),
                )
            ),
            "serialization_live_peak_bytes": serialization_peak_bytes,
            **serialization_peaks,
            "normalized_configuration_json_chars": normalized_configuration_json_chars,
            "serialization_text_storage_bytes_per_char": (
                SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
            ),
            "serialization_chunk_chars": SERIALIZATION_CHUNK_CHARS,
            "serialization_io_buffer_bytes": SERIALIZATION_IO_BUFFER_BYTES,
            "serialization_fixed_overhead_bytes": SERIALIZATION_FIXED_OVERHEAD_BYTES,
            "max_csv_write_chars": MAX_CSV_WRITE_CHARS,
            "source_hash_buffer_bytes": SOURCE_HASH_BUFFER_BYTES,
        },
        "formula": {
            "static_device": "8*D*b + (G*b + M*(20*b+40)) + 17*K*b + 96*K + T*(11*b+3)",
            "static_host": "D*b + T*(11*b+3) + 8*M + (G*(16+b) + M*(8+4*b)) + rows*4096 + serialization_peak + 1MiB",
            "static_combined": "static_device + static_host",
        },
        "exclusions": [
            "Python interpreter and imported-library resident memory",
            "compiler executable/code/cache and backend allocator overhead",
            "TOML parser objects already allocated before preflight",
        ],
    }


def _static_limit_values(estimates: dict) -> dict[str, int]:
    return {
        "agent_grid_points": int(estimates["agent_grid_points"]),
        "ordered_pair_cells": int(estimates["ordered_pair_cells"]),
        "state_expanded_cells": int(estimates["state_expanded_cells"]),
        "initial_pair_bytes": int(estimates["initial_pair_bytes"]),
        "combined_peak_bytes": int(estimates["static_combined_peak_bytes"]),
        "retained_full_density_snapshots": int(
            estimates["retained_full_density_snapshots"]
        ),
        "diagnostic_output_rows": int(estimates["diagnostic_output_rows"]),
    }


def validate_pair_budget(
    raw: dict,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> dict:
    estimates = estimate_pair_resources(raw)
    effective_limits = dict(PHASE4_ABSOLUTE_LIMITS if limits is None else limits)
    if set(effective_limits) != set(PHASE4_ABSOLUTE_LIMITS):
        raise ValueError("Phase 4 limits must define every fixed resource cap")
    for name, value in effective_limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Phase 4 limit {name} must be a non-negative integer")
    limit_values = _static_limit_values(estimates)
    violations = [
        name
        for name in PHASE4_ABSOLUTE_LIMITS
        if limit_values[name] > effective_limits[name]
    ]
    if violations and not allow_expensive:
        details = "; ".join(
            f"{name}={limit_values[name]:,} (limit {effective_limits[name]:,})"
            for name in PHASE4_ABSOLUTE_LIMITS
        )
        raise ValueError(
            "refusing expensive Phase 4 pair work before QGrid, JAX arrays, "
            f"execution, or output construction: {details}; violations={violations}; "
            "pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "static_estimates": estimates,
        "absolute_limits": effective_limits,
        "static_violations": violations,
        "static_violations_overridden": violations if allow_expensive else [],
        "compiled_analysis": None,
        "compiled_violations": [],
        "compiled_violations_overridden": [],
    }


def compiled_memory_report(compiled, static_host_bytes: int) -> dict:
    """Read JAX executable memory statistics without claiming unavailable data."""

    base = {
        "performed": True,
        "available": False,
        "backend": jax.default_backend(),
        "argument_bytes": None,
        "output_bytes": None,
        "temporary_bytes": None,
        "alias_bytes": None,
        "host_argument_bytes": None,
        "host_output_bytes": None,
        "host_temporary_bytes": None,
        "host_alias_bytes": None,
        "compiled_device_requirement_bytes": None,
        "compiled_host_requirement_bytes": None,
        "static_host_allowance_bytes": int(static_host_bytes),
        "compiled_plus_host_requirement_bytes": None,
        "formula": (
            "device=(argument+output+temporary-alias); "
            "compiled_host=(host_argument+host_output+host_temporary-host_alias); "
            "combined=device+compiled_host+static_host"
        ),
        "scope": "JAX executable buffers plus the separately estimated runner host allowance",
        "analysis_status": "unavailable",
        "validation_status": "not_checked",
        "unavailable_reason": None,
    }
    try:
        stats = compiled.memory_analysis()
    except Exception as error:
        base["unavailable_reason"] = _truncate_compiled_reason(
            f"{type(error).__name__}: {error}"
        )
        return base
    if stats is None:
        base["unavailable_reason"] = "memory_analysis returned None"
        return base

    names = {
        "argument_bytes": "argument_size_in_bytes",
        "output_bytes": "output_size_in_bytes",
        "temporary_bytes": "temp_size_in_bytes",
        "alias_bytes": "alias_size_in_bytes",
        "host_argument_bytes": "host_argument_size_in_bytes",
        "host_output_bytes": "host_output_size_in_bytes",
        "host_temporary_bytes": "host_temp_size_in_bytes",
        "host_alias_bytes": "host_alias_size_in_bytes",
    }
    values = {}
    for output_name, attribute_name in names.items():
        value = getattr(stats, attribute_name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            base["unavailable_reason"] = _truncate_compiled_reason(
                f"memory_analysis supplied invalid {attribute_name}={value!r}"
            )
            return base
        values[output_name] = value
    if values["alias_bytes"] > values["argument_bytes"] + values["output_bytes"]:
        base["unavailable_reason"] = "alias bytes exceed argument plus output bytes"
        return base
    if (
        values["host_alias_bytes"]
        > values["host_argument_bytes"] + values["host_output_bytes"]
    ):
        base["unavailable_reason"] = (
            "host alias bytes exceed host argument plus host output bytes"
        )
        return base

    device = (
        values["argument_bytes"]
        + values["output_bytes"]
        + values["temporary_bytes"]
        - values["alias_bytes"]
    )
    compiled_host = (
        values["host_argument_bytes"]
        + values["host_output_bytes"]
        + values["host_temporary_bytes"]
        - values["host_alias_bytes"]
    )
    base.update(values)
    base.update(
        {
            "available": True,
            "analysis_status": "complete",
            "compiled_device_requirement_bytes": device,
            "compiled_host_requirement_bytes": compiled_host,
            "compiled_plus_host_requirement_bytes": (
                device + compiled_host + int(static_host_bytes)
            ),
        }
    )
    return base


def analyze_compiled_pair_memory(raw: dict, grid, learning: LearningConfig) -> dict:
    """Compile from an abstract pair shape, then inspect executable buffers."""

    dtype = jnp.float32 if raw["dtype"] == "float32" else jnp.float64
    abstract_mass = jax.ShapeDtypeStruct(
        (2, raw["agent_grid_points"], raw["agent_grid_points"]), dtype
    )
    lowered = simulate_pair_density_jit.lower(
        abstract_mass,
        grid,
        learning.alpha,
        learning.tau,
        steps=raw["steps"],
        chunk_size=raw["chunk_size"],
        diagnostic_tolerance=raw["diagnostic_tolerance"],
    )
    compiled = lowered.compile()
    static_host = estimate_pair_resources(raw)["static_host_bytes"]
    return compiled_memory_report(compiled, static_host)


def validate_compiled_pair_budget(
    budget: dict,
    report: dict,
    allow_expensive: bool,
) -> dict:
    """Apply the fixed combined cap to available executable-memory analysis."""

    budget["compiled_analysis"] = report
    violations = []
    if not report["available"]:
        violations.append("compiled_analysis_unavailable")
        report["validation_status"] = "unavailable"
        if not allow_expensive:
            raise ValueError(
                "refusing Phase 4 pair execution because compiled memory analysis "
                "is unavailable or incomplete before histogram, pair allocation, "
                "execution, or output; reason="
                f"{report.get('unavailable_reason')!r}; pass --allow-expensive to override"
            )
    else:
        compiled_device = int(report["compiled_device_requirement_bytes"])
        static_device = int(budget["static_estimates"]["static_device_bytes"])
        required = int(report["compiled_plus_host_requirement_bytes"])
        limit = int(budget["absolute_limits"]["combined_peak_bytes"])
        if compiled_device > static_device:
            violations.append("compiled_device_exceeds_static_bound")
        if required > limit:
            violations.append("compiled_combined_peak_bytes")
        report["validation_status"] = "failed" if violations else "passed"
        if violations and not allow_expensive:
            raise ValueError(
                "refusing expensive Phase 4 pair execution after shape-only "
                "compilation but before histogram, pair allocation, execution, "
                f"or output: compiled_device={compiled_device:,} "
                f"(static bound {static_device:,}); compiled_plus_host={required:,} "
                f"(limit {limit:,}); violations={violations}; pass "
                "--allow-expensive to override"
            )
    budget["compiled_violations"] = violations
    budget["compiled_violations_overridden"] = violations if allow_expensive else []
    return budget


def selected_times(steps: int, stride: int) -> list[int]:
    times = [0, *range(stride, steps + 1, stride)]
    if times[-1] != steps:
        times.append(steps)
    return times


def diagnostics_rows(initial, trajectory, steps: int, stride: int) -> list[dict]:
    rows = []
    for time in selected_times(steps, stride):
        diagnostics = initial if time == 0 else jax.tree_util.tree_map(
            lambda value: value[time - 1], trajectory
        )
        rows.append(
            {
                "time": time,
                "total_mass": float(np.asarray(diagnostics.total_mass)),
                "state_sh_mass": float(np.asarray(diagnostics.state_masses[0])),
                "state_pd_mass": float(np.asarray(diagnostics.state_masses[1])),
                "mean_q_c": float(np.asarray(diagnostics.mean_q[0])),
                "mean_q_d": float(np.asarray(diagnostics.mean_q[1])),
                "mean_policy_c": float(np.asarray(diagnostics.mean_action_probability[0])),
                "mean_policy_d": float(np.asarray(diagnostics.mean_action_probability[1])),
                "symmetry_error": float(np.asarray(diagnostics.symmetry_error)),
                "minimum_mass": float(np.asarray(diagnostics.minimum_mass)),
                "finite": bool(np.asarray(diagnostics.finite)),
                "nonnegative": bool(np.asarray(diagnostics.nonnegative)),
                "conditional_weight_error": float(np.asarray(diagnostics.conditional_weight_error)),
                "minimum_conditional_variance": float(
                    np.asarray(diagnostics.minimum_conditional_variance)
                ),
                "conditional_moments_valid": bool(
                    np.asarray(diagnostics.conditional_moments_valid)
                ),
            }
        )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def normalized_device_metadata() -> list[dict]:
    devices = jax.devices()
    if len(devices) > MAX_DEVICE_COUNT:
        raise ValueError(
            f"device count {len(devices)} exceeds metadata limit {MAX_DEVICE_COUNT}"
        )
    return [
        {
            "id": _bounded_metadata_text(
                device.id, "device id", MAX_DEVICE_FIELD_CHARS
            ),
            "platform": _bounded_metadata_text(
                device.platform, "device platform", MAX_DEVICE_FIELD_CHARS
            ),
            "device_kind": _bounded_metadata_text(
                device.device_kind, "device kind", MAX_DEVICE_FIELD_CHARS
            ),
        }
        for device in devices
    ]


def encode_bounded_metadata(metadata: dict, maximum_chars: int) -> str:
    """Encode bounded metadata as ASCII JSON without a redundant byte copy."""

    encoded = json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if not encoded.isascii():  # pragma: no cover - ensure_ascii contract
        raise RuntimeError("normalized metadata JSON is not ASCII")
    encoded_chars = len(encoded)
    if encoded_chars > maximum_chars:
        raise RuntimeError(
            f"normalized metadata needs {encoded_chars:,} ASCII characters, above "
            f"the {maximum_chars:,}-character metadata bound"
        )
    return encoded


class _BoundedCountingTextSink:
    """Non-retaining sink used to audit one CSV write at a time."""

    def __init__(self, maximum_chars: int):
        self.maximum_chars = maximum_chars
        self.maximum_observed_chars = 0
        self.write_count = 0

    def write(self, text: str) -> int:
        if not isinstance(text, str) or not text.isascii():
            raise RuntimeError("diagnostic CSV writes must be ASCII text")
        length = len(text)
        if length > self.maximum_chars:
            raise RuntimeError(
                f"diagnostic CSV write needs {length:,} ASCII characters, above "
                f"the {self.maximum_chars:,}-character record allowance"
            )
        self.maximum_observed_chars = max(self.maximum_observed_chars, length)
        self.write_count += 1
        return length


class _BoundedAsciiBinarySink(_BoundedCountingTextSink):
    """Encode and write one already bounded ASCII CSV record at a time."""

    def __init__(self, binary_file, maximum_chars: int):
        super().__init__(maximum_chars)
        self.binary_file = binary_file

    def write(self, text: str) -> int:
        length = super().write(text)
        self.binary_file.write(text.encode("ascii"))
        return length


def validate_csv_record_serialization(rows: list[dict]) -> int:
    """Audit the CSV header and each record without retaining serialized text."""

    fieldnames = list(rows[0])
    sink = _BoundedCountingTextSink(MAX_CSV_WRITE_CHARS)
    writer = csv.DictWriter(sink, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return sink.maximum_observed_chars


def write_bounded_csv(path: Path, rows: list[dict]) -> None:
    """Stream an ASCII CSV through the same one-write bound used in preflight."""

    with path.open("wb", buffering=SERIALIZATION_IO_BUFFER_BYTES) as binary_file:
        sink = _BoundedAsciiBinarySink(binary_file, MAX_CSV_WRITE_CHARS)
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_bounded_metadata(path: Path, encoded_metadata: str) -> None:
    """Write ASCII metadata using bounded text/byte chunks."""

    if not encoded_metadata.isascii():
        raise RuntimeError("normalized metadata JSON is not ASCII")
    with path.open("wb", buffering=SERIALIZATION_IO_BUFFER_BYTES) as binary_file:
        for start in range(0, len(encoded_metadata), SERIALIZATION_CHUNK_CHARS):
            chunk = encoded_metadata[start : start + SERIALIZATION_CHUNK_CHARS]
            binary_file.write(chunk.encode("ascii"))


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    raw = inspect_raw_pair_config(config)
    budget = validate_pair_budget(raw, args.allow_expensive)
    learning = LearningConfig(**raw["normalized_configuration"]["model"])
    git_status = _bounded_metadata_text(
        git_text("status", "--short", "--", "."),
        "subproject git status",
        MAX_GIT_STATUS_CHARS,
    )

    if raw["dtype"] == "float64" and not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "solver.dtype='float64' requires JAX_ENABLE_X64=1 before importing JAX"
        )

    # The fixed guard above precedes every potentially large grid/device allocation.
    grid = QGrid(raw["q_min"], raw["q_max"], raw["spacing"])
    dtype = jnp.float32 if raw["dtype"] == "float32" else jnp.float64
    jax_grid = build_jax_pair_grid(grid, dtype)
    compiled_report = analyze_compiled_pair_memory(raw, jax_grid, learning)
    budget = validate_compiled_pair_budget(
        budget, compiled_report, args.allow_expensive
    )
    device_metadata = normalized_device_metadata()

    # No histogram or full pair state is constructed before both resource gates.
    histogram = tiny_histogram(grid)
    initial_mass = ordered_pair_mass_jax(
        histogram,
        state_probabilities=tuple(raw["state_probabilities"]),
        dtype=dtype,
    )
    validate_jax_pair_mass(
        initial_mass,
        jax_grid,
        symmetry_tolerance=raw["symmetry_tolerance"],
        max_elements=budget["static_estimates"]["state_expanded_cells"],
    )
    initial_diagnostics = pair_diagnostics_jax(
        initial_mass,
        jax_grid,
        learning.tau,
        tolerance=raw["diagnostic_tolerance"],
    )
    result = checked_simulate_pair_density(
        initial_mass,
        jax_grid,
        learning.alpha,
        learning.tau,
        steps=raw["steps"],
        chunk_size=raw["chunk_size"],
        symmetry_tolerance=raw["symmetry_tolerance"],
        diagnostic_tolerance=raw["diagnostic_tolerance"],
        max_elements=budget["static_estimates"]["state_expanded_cells"],
    )
    result.final_mass.block_until_ready()
    rows = diagnostics_rows(
        initial_diagnostics,
        result.diagnostics,
        raw["steps"],
        raw["diagnostic_stride"],
    )
    if len(rows) != budget["static_estimates"]["diagnostic_output_rows"]:
        raise RuntimeError("diagnostic row count disagrees with Phase 4 preflight")

    run_name = raw["normalized_configuration"]["output"]["run_name"]
    output_root = (PROJECT_ROOT / "outputs" / "pair_jax").resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_directory = (output_root / f"{run_name}-{timestamp}").resolve()
    if output_root not in output_directory.parents:
        raise ValueError("output directory must remain beneath outputs/pair_jax")

    source_paths = [
        ("src/chu_pair/model.py", PROJECT_ROOT / "src" / "chu_pair" / "model.py"),
        ("src/chu_pair/grids.py", PROJECT_ROOT / "src" / "chu_pair" / "grids.py"),
        (
            "src/chu_pair/initial_conditions.py",
            PROJECT_ROOT / "src" / "chu_pair" / "initial_conditions.py",
        ),
        ("src/chu_pair/policies.py", PROJECT_ROOT / "src" / "chu_pair" / "policies.py"),
        (
            "src/chu_pair/pair_density/numpy_reference.py",
            PROJECT_ROOT / "src" / "chu_pair" / "pair_density" / "numpy_reference.py",
        ),
        (
            "src/chu_pair/pair_density/jax_solver.py",
            PROJECT_ROOT / "src" / "chu_pair" / "pair_density" / "jax_solver.py",
        ),
        ("experiments/run_pair_jax_small.py", Path(__file__).resolve()),
        ("configuration_file", config_path),
    ]
    metadata = {
        "schema_version": 1,
        "milestone": "Phase 4 JAX nearest_legacy pair solver CPU smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": raw["normalized_configuration"],
        "resource_budget": budget,
        "array_layout": {
            "internal": [2, raw["agent_grid_points"], raw["agent_grid_points"]],
            "canonical": [raw["grid_size"], raw["grid_size"], 2, raw["grid_size"], raw["grid_size"]],
            "mass_semantics": "discrete probability mass; no grid-spacing multiplier",
        },
        "initialization": {
            "mode": "deterministic tiny two-cell histogram",
            "seed": None,
            "state_probabilities": raw["state_probabilities"],
        },
        "diagnostic_selection": {
            "initial_time": 0,
            "stride": raw["diagnostic_stride"],
            "final_time": raw["steps"],
            "row_count": len(rows),
            "final_time_appended_when_off_stride": bool(
                raw["steps"] % raw["diagnostic_stride"]
            ),
        },
        "full_density_snapshots_written": 0,
        "backend": _bounded_metadata_text(
            jax.default_backend(), "backend", MAX_DEVICE_FIELD_CHARS
        ),
        "devices": device_metadata,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "versions": {
            "python": _bounded_metadata_text(
                sys.version.split()[0], "Python version", MAX_DEVICE_FIELD_CHARS
            ),
            "numpy": _bounded_metadata_text(
                np.__version__, "NumPy version", MAX_DEVICE_FIELD_CHARS
            ),
            "jax": _bounded_metadata_text(
                jax.__version__, "JAX version", MAX_DEVICE_FIELD_CHARS
            ),
            "jaxlib": _bounded_metadata_text(
                jaxlib.__version__, "JAXLIB version", MAX_DEVICE_FIELD_CHARS
            ),
        },
        "git": {
            "commit": _bounded_metadata_text(
                git_text("rev-parse", "HEAD"), "Git commit", MAX_DEVICE_FIELD_CHARS
            ),
            "subproject_status_before_output": git_status,
        },
        "source_hashes": {
            label: sha256(path) for label, path in source_paths
        },
        "limitations": [
            "CPU smoke only unless metadata reports another tested backend",
            "nearest_legacy projection only",
            "no pair-derived cross-opponent covariance",
            "no final pair-versus-ABM variance comparison",
        ],
    }
    encoded_metadata = encode_bounded_metadata(
        metadata, raw["metadata_json_text_bound_chars"]
    )
    if len(encoded_metadata) > raw["metadata_json_text_bound_chars"]:
        raise RuntimeError("metadata encoding exceeds the preflight text bound")
    del metadata
    validate_csv_record_serialization(rows)
    output_directory.mkdir(parents=True, exist_ok=False)

    csv_path = output_directory / "diagnostics.csv"
    write_bounded_csv(csv_path, rows)
    write_bounded_metadata(output_directory / "metadata.json", encoded_metadata)

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"wrote {len(rows)} diagnostic rows to {output_directory}")


if __name__ == "__main__":
    main()

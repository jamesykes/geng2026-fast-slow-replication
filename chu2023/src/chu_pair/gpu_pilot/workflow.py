"""Allocation-free planning, immutable gates, and artifacts for GPU pilot stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

FULL_ANALYSIS_CONFIRMATION = "ANALYZE EXACT G131 SEPARABLE"
FULL_EXECUTION_CONFIRMATION = "EXECUTE ONE EXACT G131 STEP"
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_CONFIG_BYTES = 64 * 1024
MAX_GRIDS = 6
MAX_RUN_NAME_CHARS = 64
MAX_STAGE_SECONDS = 6 * 60 * 60
MAX_HOURLY_PRICE_USD = 100.0
MAX_SESSION_COST_USD = 500.0
MAX_PILOT_STATIC_DEVICE_BYTES = 72 * 1024**3
MAX_PILOT_HOST_PLANNING_BYTES = 48 * 1024**3
MAX_PREREQUISITE_AGE_SECONDS = 6 * 60 * 60
MAX_FULL_ANALYSIS_AGE_SECONDS = 10 * 60
REVIEWED_GRID_SPECS = {
    3: (-1.2, 1.2, 1.2),
    5: (-0.4, 1.2, 0.4),
    9: (-0.2, 1.4, 0.2),
    17: (-0.1, 1.5, 0.1),
    33: (-0.1, 1.5, 0.05),
    65: (-0.1, 1.5, 0.025),
    97: (-0.1, 1.82, 0.02),
    131: (-0.1, 1.2, 0.01),
}
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PilotStage(str, Enum):
    DOCTOR = "doctor"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_PILOT = "large-pilot"
    FULL_GRID_ANALYZE = "full-grid-analyze"
    FULL_GRID_ONE_STEP = "full-grid-one-step"


_EXPECTED_SECTIONS = {
    "environment", "model", "initial_condition", "case", "budget", "output"
}
_EXPECTED_KEYS = {
    "environment": {"allocator_policy", "memory_fraction"},
    "model": {"alpha", "tau", "state_probabilities"},
    "initial_condition": {"histogram_seed", "samples_per_grid_cell"},
    "case": {
        "stage", "grids", "dtype", "steps", "source_times",
        "row_block_size", "column_block_size", "diagnostic_tolerance",
        "symmetry_tolerance", "include_g97",
    },
    "budget": {
        "hourly_price_usd", "max_session_cost_usd", "max_stage_seconds",
        "safety_margin_fraction",
    },
    "output": {"run_name"},
}
_EXPECTED_GRIDS = {
    PilotStage.SMALL: (3, 5, 9),
    PilotStage.MEDIUM: (17, 33),
    PilotStage.LARGE_PILOT: (65,),
    PilotStage.FULL_GRID_ANALYZE: (131,),
    PilotStage.FULL_GRID_ONE_STEP: (131,),
}


@dataclass(frozen=True, slots=True)
class PilotConfiguration:
    stage: PilotStage
    allocator_policy: str
    memory_fraction: float
    alpha: float
    tau: float
    state_probabilities: tuple[float, float]
    histogram_seed: int
    samples_per_grid_cell: int
    grids: tuple[int, ...]
    dtype: str
    steps: int
    source_times: tuple[int, ...]
    row_block_size: int
    column_block_size: int
    diagnostic_tolerance: float
    symmetry_tolerance: float
    include_g97: bool
    hourly_price_usd: float
    max_session_cost_usd: float
    max_stage_seconds: int
    safety_margin_fraction: float
    run_name: str
    contraction_precision: str
    normalized_sha256: str


def _finite(value: object, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return result


def _integer(value: object, name: str, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _contraction_precision() -> str:
    """Authoritative explicit precision policy for pair-density contractions."""

    from ..config import PAIR_CONTRACTION_PRECISION

    return PAIR_CONTRACTION_PRECISION


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def load_pilot_configuration(path: Path) -> PilotConfiguration:
    """Strictly normalize one small pilot configuration before JAX import."""

    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("pilot configuration exceeds its fixed byte bound")
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if not isinstance(raw, dict) or set(raw) != _EXPECTED_SECTIONS:
        raise ValueError("pilot configuration sections do not match the exact schema")
    for section, keys in _EXPECTED_KEYS.items():
        if not isinstance(raw[section], dict) or set(raw[section]) != keys:
            raise ValueError(f"pilot configuration keys in {section} do not match the schema")
    try:
        stage = PilotStage(raw["case"]["stage"])
    except (TypeError, ValueError) as error:
        raise ValueError("case.stage is not a recognized pilot stage") from error
    if stage == PilotStage.DOCTOR:
        raise ValueError("doctor does not consume a numerical stage configuration")
    allocator = raw["environment"]["allocator_policy"]
    if allocator not in {"default", "fraction", "no-preallocation"}:
        raise ValueError("environment.allocator_policy is invalid")
    memory_fraction = _finite(
        raw["environment"]["memory_fraction"],
        "environment.memory_fraction", lower=0.05, upper=0.95,
    )
    model = raw["model"]
    alpha = _finite(model["alpha"], "model.alpha", lower=0.0, upper=1.0)
    tau = _finite(model["tau"], "model.tau", lower=0.0, upper=100.0)
    probabilities = model["state_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != 2:
        raise ValueError("model.state_probabilities must contain two values")
    states = tuple(
        _finite(value, "model.state_probabilities", lower=0.0, upper=1.0)
        for value in probabilities
    )
    if not math.isclose(sum(states), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("model.state_probabilities must sum to one")
    if states != (0.5, 0.5):
        raise ValueError("the controlled pilot requires uniform initial edge-state mass")
    initial = raw["initial_condition"]
    histogram_seed = _integer(
        initial["histogram_seed"], "initial_condition.histogram_seed",
        lower=0, upper=2**32 - 1,
    )
    samples_per_cell = _integer(
        initial["samples_per_grid_cell"], "initial_condition.samples_per_grid_cell",
        lower=1, upper=100,
    )
    case = raw["case"]
    grids_raw = case["grids"]
    if (
        not isinstance(grids_raw, list) or not 1 <= len(grids_raw) <= MAX_GRIDS
        or any(isinstance(value, bool) or not isinstance(value, int) for value in grids_raw)
    ):
        raise ValueError("case.grids must be a bounded integer list")
    grids = tuple(grids_raw)
    expected = _EXPECTED_GRIDS[stage]
    include_g97 = case["include_g97"]
    if not isinstance(include_g97, bool):
        raise ValueError("case.include_g97 must be boolean")
    if stage == PilotStage.LARGE_PILOT and include_g97:
        expected = (65, 97)
    elif include_g97:
        raise ValueError("G=97 is optional only in the large-pilot stage")
    if grids != expected:
        raise ValueError(f"{stage.value} grids must be exactly {list(expected)}")
    if any(grid not in REVIEWED_GRID_SPECS for grid in grids):
        raise ValueError("case.grids contains no reviewed legacy-aligned grid")
    dtype = case["dtype"]
    if dtype not in {"float32", "float64"}:
        raise ValueError("case.dtype must be float32 or float64")
    steps = _integer(case["steps"], "case.steps", lower=0, upper=4)
    if stage == PilotStage.FULL_GRID_ANALYZE and steps != 1:
        raise ValueError("full-grid-analyze must analyze the exact one-step executable")
    if stage == PilotStage.FULL_GRID_ONE_STEP and steps != 1:
        raise ValueError("full-grid-one-step is hard-limited to exactly one step")
    times = case["source_times"]
    if (
        not isinstance(times, list) or not 1 <= len(times) <= 5
        or any(isinstance(value, bool) or not isinstance(value, int) for value in times)
        or times != sorted(set(times)) or times[0] < 0 or times[-1] > steps
    ):
        raise ValueError("case.source_times must be bounded, sorted, unique, and within [0,T]")
    row = _integer(case["row_block_size"], "case.row_block_size", lower=1, upper=4096)
    column = _integer(case["column_block_size"], "case.column_block_size", lower=1, upper=4096)
    diagnostic = _finite(
        case["diagnostic_tolerance"], "case.diagnostic_tolerance", lower=0.0, upper=1e-3
    )
    symmetry = _finite(
        case["symmetry_tolerance"], "case.symmetry_tolerance", lower=0.0, upper=1e-3
    )
    budget = raw["budget"]
    price = _finite(
        budget["hourly_price_usd"], "budget.hourly_price_usd",
        lower=0.0, upper=MAX_HOURLY_PRICE_USD,
    )
    session_cost = _finite(
        budget["max_session_cost_usd"], "budget.max_session_cost_usd",
        lower=0.0, upper=MAX_SESSION_COST_USD,
    )
    max_seconds = _integer(
        budget["max_stage_seconds"], "budget.max_stage_seconds",
        lower=1, upper=MAX_STAGE_SECONDS,
    )
    margin = _finite(
        budget["safety_margin_fraction"], "budget.safety_margin_fraction",
        lower=0.25, upper=1.0,
    )
    projected_cost = price * max_seconds / 3600.0
    if projected_cost > session_cost:
        raise ValueError("stage wall-time and hourly-price budget exceed max session cost")
    run_name = raw["output"]["run_name"]
    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise ValueError("output.run_name must be bounded safe ASCII")
    normalized = {
        "stage": stage, "allocator_policy": allocator,
        "memory_fraction": memory_fraction, "alpha": alpha, "tau": tau,
        "state_probabilities": states, "histogram_seed": histogram_seed,
        "samples_per_grid_cell": samples_per_cell,
        "grids": grids, "dtype": dtype,
        "steps": steps, "source_times": tuple(times), "row_block_size": row,
        "column_block_size": column, "diagnostic_tolerance": diagnostic,
        "symmetry_tolerance": symmetry, "include_g97": include_g97,
        "hourly_price_usd": price, "max_session_cost_usd": session_cost,
        "max_stage_seconds": max_seconds, "safety_margin_fraction": margin,
        "run_name": run_name,
        # Fixed numerical policy, deliberately not configurable: a pilot
        # configuration must not be able to restore the platform default and
        # reintroduce TF32 error into the conditional-weight diagnostic.
        "contraction_precision": _contraction_precision(),
    }
    return PilotConfiguration(**normalized, normalized_sha256=_canonical_sha256(normalized))


def estimate_stage_resources(configuration: PilotConfiguration) -> dict[str, object]:
    """Calculate immutable per-grid separable planning estimates with Python ints."""

    # This import initializes the JAX pair package, so callers must first apply
    # the allocator policy. Config parsing itself deliberately remains JAX-free.
    from ..pair_density.separable_resources import (
        estimate_flat_resources,
        estimate_separable_resources,
    )

    item_bytes = 4 if configuration.dtype == "float32" else 8
    cases = []
    for grid in configuration.grids:
        points = int(grid) * int(grid)
        histogram_int64_bytes = points * 8
        histogram_sample_pairs = points * configuration.samples_per_grid_cell
        if points > 5_000_000 or histogram_int64_bytes > 32 * 1024**2 or histogram_sample_pairs > 2_000_000:
            raise ValueError("pilot histogram resource guard rejected before construction")
        estimate = estimate_separable_resources(
            grid_size=int(grid), dtype_bytes=item_bytes, steps=configuration.steps,
            summary_count=len(configuration.source_times),
            row_block_size=configuration.row_block_size,
            column_block_size=configuration.column_block_size,
            return_final_density=False,
        )
        flat = None
        if configuration.stage == PilotStage.SMALL:
            flat = estimate_flat_resources(
                grid_size=int(grid), dtype_bytes=item_bytes, steps=configuration.steps,
                summary_count=len(configuration.source_times),
                chunk_size=min(2 * grid**4, 65_536), return_final_density=False,
            )
        pilot_host_peak = max(
            int(estimate["host_planning_threshold_bytes"]),
            int(estimate["host_grid_histogram_bytes"]) + histogram_int64_bytes,
        )
        cases.append({
            "grid_size": grid, **estimate, "flat_validation": flat,
            "histogram_cells": points,
            "histogram_int64_bytes": histogram_int64_bytes,
            "histogram_sample_pairs": histogram_sample_pairs,
            "pilot_host_planning_threshold_bytes": pilot_host_peak,
        })
    maximum_device = max(
        max(
            int(case["static_device_bytes"]),
            0 if case["flat_validation"] is None else int(case["flat_validation"]["static_device_bytes"]),
        )
        for case in cases
    )
    maximum_host = max(
        max(
            int(case["pilot_host_planning_threshold_bytes"]),
            0 if case["flat_validation"] is None else int(case["flat_validation"]["host_planning_threshold_bytes"]),
        )
        for case in cases
    )
    violations = []
    if maximum_device > MAX_PILOT_STATIC_DEVICE_BYTES:
        violations.append("static_device_bytes")
    if maximum_host > MAX_PILOT_HOST_PLANNING_BYTES:
        violations.append("host_planning_threshold_bytes")
    return {
        "cases": cases,
        "maximum_static_device_bytes": maximum_device,
        "maximum_host_planning_threshold_bytes": maximum_host,
        "absolute_limits": {
            "static_device_bytes": MAX_PILOT_STATIC_DEVICE_BYTES,
            "host_planning_threshold_bytes": MAX_PILOT_HOST_PLANNING_BYTES,
        },
        "violations": violations,
    }


def executable_configuration_sha256(configuration: PilotConfiguration) -> str:
    """Digest only facts that define the compiled/scientific one-step object."""

    payload = {
        "allocator_policy": configuration.allocator_policy,
        "memory_fraction": configuration.memory_fraction,
        "alpha": configuration.alpha,
        "tau": configuration.tau,
        "state_probabilities": configuration.state_probabilities,
        "histogram_seed": configuration.histogram_seed,
        "samples_per_grid_cell": configuration.samples_per_grid_cell,
        "grids": configuration.grids,
        "dtype": configuration.dtype,
        "steps": configuration.steps,
        "source_times": configuration.source_times,
        "row_block_size": configuration.row_block_size,
        "column_block_size": configuration.column_block_size,
        "diagnostic_tolerance": configuration.diagnostic_tolerance,
        "symmetry_tolerance": configuration.symmetry_tolerance,
        "contraction_precision": configuration.contraction_precision,
    }
    return _canonical_sha256(payload)


def stage_invariant_contract(
    configuration: PilotConfiguration,
    *,
    commit: str,
    environment_sha256: str,
    source_hashes: Mapping[str, object],
) -> dict[str, object]:
    """Facts that must not drift between numerical pilot stages.

    Grid size, horizon, blocks, source times, and the small-stage flat oracle
    deliberately vary; all model, initialization, precision, allocator,
    diagnostic, source, and session facts remain bound to the chain.
    """

    relevant_hashes = {
        name: source_hashes.get(name)
        for name in (
            "src/chu_pair/model.py",
            "src/chu_pair/initial_conditions.py",
            "src/chu_pair/pair_density/jax_solver.py",
        )
    }
    if any(
        not isinstance(value, str) or (len(value) != 64 and value != "unavailable")
        for value in relevant_hashes.values()
    ):
        raise ValueError("doctor report lacks bounded model/initialization source hashes")
    return {
        "schema_version": 1,
        "commit": commit,
        "environment_sha256": environment_sha256,
        "source_hashes": relevant_hashes,
        "action_order": ("C", "D"), "state_order": ("SH", "PD"),
        "initialization_law": "seeded-legacy-scaled-beta-joint-histogram",
        "pair_transport": "exact-separable-legacy-projection",
        "contraction_precision": configuration.contraction_precision,
        "alpha": configuration.alpha, "tau": configuration.tau,
        "dtype": configuration.dtype,
        "allocator_policy": configuration.allocator_policy,
        "memory_fraction": configuration.memory_fraction,
        "state_probabilities": configuration.state_probabilities,
        "histogram_seed": configuration.histogram_seed,
        "samples_per_grid_cell": configuration.samples_per_grid_cell,
        "diagnostic_tolerance": configuration.diagnostic_tolerance,
        "symmetry_tolerance": configuration.symmetry_tolerance,
        "hourly_price_usd": configuration.hourly_price_usd,
        "max_session_cost_usd": configuration.max_session_cost_usd,
    }


def stage_invariant_contract_sha256(contract: Mapping[str, object]) -> str:
    return _canonical_sha256(contract)


def expected_allocator_environment(configuration: PilotConfiguration) -> dict[str, str]:
    expected = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": (
            "false" if configuration.allocator_policy == "no-preallocation" else "true"
        )
    }
    if configuration.allocator_policy == "fraction":
        expected["XLA_PYTHON_CLIENT_MEM_FRACTION"] = format(
            configuration.memory_fraction, ".17g"
        )
    return expected


def validate_allocator_identity(
    configuration: PilotConfiguration, actual: object
) -> None:
    if actual != expected_allocator_environment(configuration):
        raise ValueError("compiled allocator identity disagrees with pilot configuration")


def validate_analyzed_signature_match(expected: object, actual: object) -> None:
    if (
        not isinstance(expected, str) or len(expected) != 64
        or not isinstance(actual, str) or len(actual) != 64
        or expected != actual
    ):
        raise ValueError("compiled executable signature differs from full-grid analysis")


def validate_stage_confirmation(stage: PilotStage, phrase: str | None) -> None:
    expected = None
    if stage == PilotStage.FULL_GRID_ANALYZE:
        expected = FULL_ANALYSIS_CONFIRMATION
    elif stage == PilotStage.FULL_GRID_ONE_STEP:
        expected = FULL_EXECUTION_CONFIRMATION
    if expected is not None and phrase != expected:
        raise ValueError(f"{stage.value} requires the exact confirmation phrase: {expected}")


def calculate_cost(elapsed_seconds: float, hourly_price_usd: float) -> dict[str, object]:
    elapsed = _finite(elapsed_seconds, "elapsed_seconds", lower=0.0, upper=MAX_STAGE_SECONDS * 2)
    price = _finite(
        hourly_price_usd, "hourly_price_usd", lower=0.0, upper=MAX_HOURLY_PRICE_USD
    )
    return {
        "elapsed_seconds": elapsed,
        "hourly_price_usd_user_supplied": price,
        "estimated_compute_cost_usd": elapsed * price / 3600.0,
        "claim_scope": "estimate from elapsed wall time and user-supplied instance price; not billing data",
    }


def summarize_stage_cost(
    *,
    elapsed_seconds: float,
    hourly_price_usd: float,
    prior_cumulative_usd: float,
    session_budget_usd: float,
    next_stage_max_seconds: int,
) -> dict[str, object]:
    result = calculate_cost(elapsed_seconds, hourly_price_usd)
    prior = _finite(
        prior_cumulative_usd, "prior_cumulative_usd", lower=0.0, upper=MAX_SESSION_COST_USD
    )
    budget = _finite(
        session_budget_usd, "session_budget_usd", lower=0.0, upper=MAX_SESSION_COST_USD
    )
    next_seconds = _integer(
        next_stage_max_seconds, "next_stage_max_seconds", lower=0, upper=MAX_STAGE_SECONDS
    )
    cumulative = prior + float(result["estimated_compute_cost_usd"])
    projected_next = hourly_price_usd * next_seconds / 3600.0
    result.update(
        configured_session_budget_usd=budget,
        cumulative_stage_estimate_usd=cumulative,
        remaining_session_budget_usd=max(0.0, budget - cumulative),
        projected_next_stage_maximum_usd=projected_next,
        next_stage_would_exceed_session_budget=(cumulative + projected_next > budget),
    )
    return result


def stage_artifact_digest(payload: Mapping[str, object]) -> str:
    return _canonical_sha256(payload)


def write_stage_artifact_atomic(directory: Path, payload: Mapping[str, object]) -> Path:
    """Atomically replace one bounded success-or-failure stage artifact."""

    document = dict(payload)
    document["artifact_sha256"] = stage_artifact_digest(document)
    encoded = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("stage artifact exceeds its fixed serialization bound")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".stage-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    target = directory / "stage.json"
    os.replace(temporary, target)
    return target


def read_prerequisite_artifact(
    path: Path,
    *,
    required_stage: PilotStage,
    commit: str,
    environment_sha256: str,
    now: datetime | None = None,
) -> dict[str, object]:
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("prerequisite artifact exceeds its byte bound")
    document = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(document, dict):
        raise ValueError("prerequisite artifact is not an object")
    digest = document.pop("artifact_sha256", None)
    if digest != stage_artifact_digest(document):
        raise ValueError("prerequisite artifact digest is invalid")
    if document.get("status") != "success" or document.get("stage") != required_stage.value:
        raise ValueError("prerequisite stage did not succeed")
    if document.get("git_commit") != commit or document.get("environment_sha256") != environment_sha256:
        raise ValueError("prerequisite provenance does not match this invocation")
    try:
        created = datetime.fromisoformat(str(document["completed_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prerequisite completion timestamp is invalid") from error
    current = datetime.now(timezone.utc) if now is None else now
    age = (current.astimezone(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
    maximum = (
        MAX_FULL_ANALYSIS_AGE_SECONDS
        if required_stage == PilotStage.FULL_GRID_ANALYZE
        else MAX_PREREQUISITE_AGE_SECONDS
    )
    if not 0.0 <= age <= maximum:
        raise ValueError("prerequisite artifact is stale or from the future")
    document["artifact_sha256"] = digest
    return document

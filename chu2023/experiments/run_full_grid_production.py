#!/usr/bin/env python3
"""Run the human-authorised exact full-grid Phase 5 production variance experiment.

This is the guarded production path for the pre-registered design: exact
``G=131`` separable pair evolution through ``T`` ABM steps, matched independent
ABM runs processed in deterministic chunks, and the validated Phase 5 four-way
velocity-variance comparison with a complete-run clustered bootstrap.

Scientific primitives are reused unchanged from the validated modules. The pair
side uses the same separable from-histogram executable, admission order and
identity checks as the GPU pilot; the comparison uses the same Phase 5
aggregation, moment and bootstrap functions. Nothing here redefines an
estimand, tolerance, projection, chronology or ordering.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

from chu_pair.gpu_pilot.allocator import apply_allocator_policy

CONFIRMATION_PHRASE = "RUN EXACT G131 PRODUCTION VARIANCE"

# Explicit production caps, justified by the pre-registered design rather than
# bypassed with --allow-expensive. They bound this experiment and nothing more.
PRODUCTION_LIMITS = {
    "grid_size": 131,               # exact full grid
    "max_agents": 128,
    "max_steps": 64,                # pre-registered T=32 with headroom
    "max_runs": 4096,               # audited: R=512 primary, R=4096 precision extension
    "max_bootstrap_replicates": 2000,
    "max_source_times": 16,
    "max_bin_schemes": 4,
    "max_comparison_rows": 250_000,
    "max_run_chunk": 128,
    "max_static_device_bytes": 72 * 1024**3,
    "max_host_planning_bytes": 48 * 1024**3,
    "max_per_run_statistic_bytes": 8 * 1024**3,
}
_EXPECTED_SECTIONS = {
    "model", "simulation", "initial_condition", "pair_solver", "comparison",
    "bin_schemes", "bootstrap", "anchors", "output", "budget", "environment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirmation")
    parser.add_argument("--doctor-report", type=Path, required=True)
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--hourly-price-usd", type=float, required=True)
    parser.add_argument("--run-chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibration-runs", type=int, default=0,
                        help="short calibration only; does not write production artifacts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload) -> None:
    _atomic_write(path, (json.dumps(payload, ensure_ascii=True, indent=1, sort_keys=True) + "\n").encode("ascii"))


def load_production_config(path: Path) -> dict:
    """Strictly normalise the production configuration before any JAX import."""

    if path.stat().st_size > 128 * 1024:
        raise ValueError("production configuration exceeds its fixed byte bound")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != _EXPECTED_SECTIONS:
        raise ValueError(f"production configuration sections must be exactly {sorted(_EXPECTED_SECTIONS)}")

    model, sim, init = raw["model"], raw["simulation"], raw["initial_condition"]
    pair, comp, boot = raw["pair_solver"], raw["comparison"], raw["bootstrap"]
    env, budget, out = raw["environment"], raw["budget"], raw["output"]

    def integer(value, name, lower, upper):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        return value

    def number(value, name, lower, upper):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or not lower <= result <= upper:
            raise ValueError(f"{name} must lie in [{lower}, {upper}]")
        return result

    if sim["dtype"] != "float32":
        raise ValueError("the production experiment is pre-registered as float32")
    if env["allocator_policy"] != "fraction":
        raise ValueError("production allocator policy must be 'fraction'")
    memory_fraction = number(env["memory_fraction"], "environment.memory_fraction", 0.05, 0.95)

    spacing = number(init["spacing"], "initial_condition.spacing", 1e-6, 1.0)
    q_min = number(init["q_min"], "initial_condition.q_min", -10.0, 10.0)
    q_max = number(init["q_max"], "initial_condition.q_max", -10.0, 10.0)
    grid_size = int(round((q_max - q_min) / spacing)) + 1
    if grid_size != PRODUCTION_LIMITS["grid_size"]:
        raise ValueError(
            f"production grid must be exactly G={PRODUCTION_LIMITS['grid_size']}, got {grid_size}")

    steps = integer(sim["steps"], "simulation.steps", 1, PRODUCTION_LIMITS["max_steps"])
    num_runs = integer(sim["num_runs"], "simulation.num_runs", 1, PRODUCTION_LIMITS["max_runs"])
    num_agents = integer(sim["num_agents"], "simulation.num_agents", 2, PRODUCTION_LIMITS["max_agents"])
    source_times = comp["source_times"]
    if (not isinstance(source_times, list)
            or not 1 <= len(source_times) <= PRODUCTION_LIMITS["max_source_times"]
            or any(isinstance(t, bool) or not isinstance(t, int) for t in source_times)
            or any(b <= a for a, b in zip(source_times, source_times[1:]))
            or source_times[0] < 0 or source_times[-1] >= steps):
        raise ValueError("comparison.source_times must be increasing, non-negative and below steps")

    schemes = raw["bin_schemes"]
    if not isinstance(schemes, list) or not 1 <= len(schemes) <= PRODUCTION_LIMITS["max_bin_schemes"]:
        raise ValueError("bin_schemes must be a bounded list")
    for scheme in schemes:
        if set(scheme) != {"name", "q_c_edges", "q_d_edges"}:
            raise ValueError("each bin scheme must define exactly name/q_c_edges/q_d_edges")
        for key in ("q_c_edges", "q_d_edges"):
            edges = scheme[key]
            if (not isinstance(edges, list) or len(edges) < 2
                    or any(b <= a for a, b in zip(edges, edges[1:]))):
                raise ValueError(f"{scheme['name']}.{key} must be strictly increasing")

    replicates = integer(boot["replicates"], "bootstrap.replicates", 2,
                         PRODUCTION_LIMITS["max_bootstrap_replicates"])
    normalized = {
        "alpha": number(model["alpha"], "model.alpha", 0.0, 1.0),
        "tau": number(model["tau"], "model.tau", 1e-9, 100.0),
        "num_agents": num_agents, "steps": steps, "num_runs": num_runs,
        "abm_seed": integer(sim["abm_seed"], "simulation.abm_seed", 0, 2**32 - 1),
        "dtype": "float32",
        "q_min": q_min, "q_max": q_max, "spacing": spacing, "grid_size": grid_size,
        "histogram_seed": integer(init["histogram_seed"], "initial_condition.histogram_seed", 0, 2**32 - 1),
        "samples_per_grid_cell": integer(init["samples_per_grid_cell"],
                                         "initial_condition.samples_per_grid_cell", 1, 100),
        "state_probabilities": [number(p, "state_probabilities", 0.0, 1.0)
                                for p in init["state_probabilities"]],
        "row_block_size": integer(pair["row_block_size"], "pair_solver.row_block_size", 1, 4096),
        "column_block_size": integer(pair["column_block_size"], "pair_solver.column_block_size", 1, 4096),
        "diagnostic_tolerance": number(pair["diagnostic_tolerance"], "pair_solver.diagnostic_tolerance", 0.0, 1.0),
        "symmetry_tolerance": number(pair["symmetry_tolerance"], "pair_solver.symmetry_tolerance", 0.0, 1.0),
        "source_times": list(source_times),
        "minimum_count": integer(comp["minimum_count"], "comparison.minimum_count", 1, 1000),
        "ratio_epsilon": number(comp["ratio_epsilon"], "comparison.ratio_epsilon", 0.0, 1.0),
        "bin_schemes": [{"name": s["name"], "q_c_edges": list(map(float, s["q_c_edges"])),
                         "q_d_edges": list(map(float, s["q_d_edges"]))} for s in schemes],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": integer(boot["seed"], "bootstrap.seed", 0, 2**32 - 1),
        "confidence_level": number(boot["confidence_level"], "bootstrap.confidence_level", 0.5, 0.999),
        "anchors": [[number(a, "anchor", -10.0, 10.0) for a in point] for point in raw["anchors"]["points"]],
        "allocator_policy": "fraction", "memory_fraction": memory_fraction,
        "contraction_precision": None,   # filled from the authoritative constant below
        "hourly_price_usd": number(budget["hourly_price_usd"], "budget.hourly_price_usd", 0.0, 100.0),
        "max_session_cost_usd": number(budget["max_session_cost_usd"], "budget.max_session_cost_usd", 0.0, 500.0),
        "max_stage_seconds": integer(budget["max_stage_seconds"], "budget.max_stage_seconds", 1, 24 * 3600),
        "safety_margin_fraction": number(budget["safety_margin_fraction"], "budget.safety_margin_fraction", 0.0, 2.0),
        "run_name": out["run_name"],
    }
    from chu_pair.config import PAIR_CONTRACTION_PRECISION
    normalized["contraction_precision"] = PAIR_CONTRACTION_PRECISION
    if not isinstance(normalized["run_name"], str) or not normalized["run_name"].replace("-", "").replace("_", "").isalnum():
        raise ValueError("output.run_name must be bounded safe ASCII")
    normalized["normalized_sha256"] = _canonical_sha256(normalized)
    return normalized


def estimate_production_resources(config: dict) -> dict:
    """Allocation-free Python-integer estimates for every large lifetime."""

    G = config["grid_size"]
    points = G * G
    item = 4
    ordered_pairs = points * points
    state_cells = 2 * ordered_pairs
    density_bytes = state_cells * item
    summaries = len(config["source_times"])
    # bounded per-source-time point summaries: focal mass + 7 action sums x 2 actions
    summary_bytes = summaries * points * (1 + 7 * 2) * item
    agents = config["num_agents"]
    edges = agents * (agents - 1) // 2
    chunk = min(config["num_runs"], PRODUCTION_LIMITS["max_run_chunk"])
    record_bytes = chunk * (config["steps"] + 1) * agents * 12 * item
    finest = max(len(s["q_c_edges"]) - 1 for s in config["bin_schemes"]) * \
        max(len(s["q_d_edges"]) - 1 for s in config["bin_schemes"])
    per_run_stats = config["num_runs"] * (config["steps"] + 1) * finest * 2 * 11 * 8
    bootstrap_bytes = config["bootstrap_replicates"] * config["num_runs"] * 4
    rows = sum((len(s["q_c_edges"]) - 1) * (len(s["q_d_edges"]) - 1) for s in config["bin_schemes"]) \
        * summaries * 2
    estimate = {
        "grid_size": G, "agent_grid_points": points, "ordered_pair_cells": ordered_pairs,
        "state_expanded_cells": state_cells,
        "one_density_bytes": density_bytes,
        "pair_summary_host_bytes": summary_bytes,
        "abm_edges": edges, "abm_chunk_record_bytes": record_bytes,
        "per_run_sufficient_statistic_bytes": per_run_stats,
        "bootstrap_weight_bytes": bootstrap_bytes,
        "comparison_rows": rows,
        "host_planning_threshold_bytes": summary_bytes + record_bytes + per_run_stats + bootstrap_bytes,
        "violations": [],
    }
    if rows > PRODUCTION_LIMITS["max_comparison_rows"]:
        estimate["violations"].append("comparison_rows_exceed_production_limit")
    if per_run_stats > PRODUCTION_LIMITS["max_per_run_statistic_bytes"]:
        estimate["violations"].append("per_run_statistics_exceed_production_limit")
    if estimate["host_planning_threshold_bytes"] > PRODUCTION_LIMITS["max_host_planning_bytes"]:
        estimate["violations"].append("host_planning_exceeds_production_limit")
    return estimate


def production_run_keys(abm_seed: int, indices):
    """Per-run keys folded in from the GLOBAL run index.

    ``jax.random.split(key, n)`` yields keys that depend on ``n``, so a chunked
    or restarted experiment would not reproduce an unchunked one. Folding the
    global index into the root key makes every run's stream a function of that
    index alone, and therefore invariant to chunk size and restart boundaries.
    """

    import jax
    root = jax.random.PRNGKey(int(abm_seed))
    return [jax.random.fold_in(root, int(index)) for index in indices]


def initialize_runs_by_global_index(graph, histogram, *, abm_seed, indices, dtype):
    """Grid-matched initial states for explicit GLOBAL run indices.

    Uses exactly the validated sampling primitives; only the key derivation
    differs from ``initialize_grid_matched_batch`` so that a run's stream is a
    function of its global index alone.
    """

    import jax
    import jax.numpy as jnp
    from chu_pair.abm import ABMState
    from chu_pair.abm.sampling import _sample_grid_q_arrays, sample_edge_states

    points = jnp.asarray(histogram.grid.flat_q_points, dtype=dtype)
    probabilities = jnp.asarray(histogram.mass.ravel(), dtype=dtype)
    keys = jnp.stack(production_run_keys(abm_seed, indices))

    def one_run(run_key):
        q_key, state_key, simulation_key = jax.random.split(run_key, 3)
        q_values = _sample_grid_q_arrays(q_key, points, probabilities, graph.num_agents)
        edge_states = sample_edge_states(state_key, graph.edge_count)
        return ABMState(q_values=q_values, edge_states=edge_states), simulation_key

    states, simulation_keys = jax.vmap(one_run)(keys)
    return states, simulation_keys


def run_pair_full_grid(config: dict, *, output: Path, telemetry: list,
                       heartbeat=lambda payload: None) -> dict:
    """Exact G=131 separable evolution: admission order identical to the pilot."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    from chu_pair.gpu_pilot import runtime as pilot
    from chu_pair.gpu_pilot.workflow import PilotConfiguration, PilotStage
    from chu_pair.grids import QGrid
    from chu_pair.initial_conditions import seeded_legacy_histogram
    from chu_pair.pair_density import build_jax_pair_grid, validate_pair_source_diagnostics
    from chu_pair.pair_density.separable_resources import (
        production_capacity_preflight, validate_compiled_executable_bundle,
    )
    from experiments import run_pair_separable_benchmark as benchmark

    pilot_configuration = PilotConfiguration(
        stage=PilotStage.FULL_GRID_ONE_STEP,
        allocator_policy=config["allocator_policy"], memory_fraction=config["memory_fraction"],
        alpha=config["alpha"], tau=config["tau"],
        state_probabilities=tuple(config["state_probabilities"]),
        histogram_seed=config["histogram_seed"],
        samples_per_grid_cell=config["samples_per_grid_cell"],
        grids=(config["grid_size"],), dtype="float32", steps=config["steps"],
        source_times=tuple(config["source_times"]),
        row_block_size=config["row_block_size"], column_block_size=config["column_block_size"],
        diagnostic_tolerance=config["diagnostic_tolerance"],
        symmetry_tolerance=config["symmetry_tolerance"], include_g97=False,
        hourly_price_usd=config["hourly_price_usd"],
        max_session_cost_usd=config["max_session_cost_usd"],
        max_stage_seconds=config["max_stage_seconds"],
        safety_margin_fraction=config["safety_margin_fraction"],
        run_name=config["run_name"], contraction_precision=config["contraction_precision"],
        normalized_sha256=config["normalized_sha256"],
    )
    case = pilot._case(pilot_configuration, config["grid_size"])
    model = pilot._model(pilot_configuration)
    host_grid = QGrid(case["q_min"], case["q_max"], case["spacing"])
    if host_grid.size != config["grid_size"]:
        raise RuntimeError("production QGrid disagrees with the normalized grid size")
    dtype = jnp.float32
    abstract_grid = pilot._abstract_grid(host_grid, dtype)
    points = config["grid_size"] ** 2
    abstract_histogram = jax.ShapeDtypeStruct((points,), dtype)
    abstract_states = jax.ShapeDtypeStruct((2,), dtype)
    slots_host = benchmark.source_slots(list(config["source_times"]), config["steps"])
    abstract_slots = jax.ShapeDtypeStruct(slots_host.shape, jnp.int32)
    static = {
        "steps": config["steps"], "summary_count": len(config["source_times"]),
        "chunk_size": case["flat_chunk_size"],
        "diagnostic_tolerance": config["diagnostic_tolerance"], "kernel": "separable",
        "row_block_size": config["row_block_size"], "column_block_size": config["column_block_size"],
    }
    from chu_pair.pair_density import simulate_pair_source_summaries_from_histogram_jit as solver

    lowered = solver.lower(abstract_histogram, abstract_states, abstract_grid,
                           config["alpha"], config["tau"], abstract_slots, **static)
    signature = benchmark.executable_signature(
        case, model, histogram=abstract_histogram, state_probabilities=abstract_states,
        grid=abstract_grid, slots=slots_host, kernel="separable",
        output_mode="bounded_from_histogram")
    heartbeat({"phase": "pair_compilation", "completed": 0, "total": 1, "latest_path": str(output)})
    started = time.perf_counter()
    bundle, compile_seconds = benchmark._compile_and_analyze(
        lowered, signature, static_host_bytes=int(config.get("_static_host_bytes", 1 << 20)))
    report = validate_compiled_executable_bundle(bundle)
    feasibility = {
        "state_expanded_cells": case["state_expanded_cells"],
        "safety_margin_fraction": config["safety_margin_fraction"],
        "minimum_device_memory_bytes": math.ceil(
            int(report["compiled_device_requirement_bytes"]) * (1.0 + config["safety_margin_fraction"])),
    }
    capacity = pilot._capacity_observation(signature, post_initialization=True)
    gate = production_capacity_preflight(feasibility=feasibility, bundle=bundle,
                                         capacity_observation=capacity, allow_expensive=False)
    telemetry.append({"phase": "pair_compile", "seconds": compile_seconds,
                      "compiled_device_requirement_bytes": report["compiled_device_requirement_bytes"],
                      "verified_usable_device_bytes": gate["verified_usable_device_bytes"]})

    # Device inputs exist only after admission. The scan builds P0 on device.
    device_grid = build_jax_pair_grid(host_grid, dtype)
    from chu_pair.initial_conditions import seeded_legacy_histogram as _hist
    histogram = _hist(host_grid, seed=config["histogram_seed"],
                      samples_per_grid_cell=config["samples_per_grid_cell"])
    histogram_device = jnp.asarray(histogram.mass.reshape(-1), dtype=dtype)
    states_device = jnp.asarray(config["state_probabilities"], dtype=dtype)
    slots_device = jnp.asarray(slots_host, dtype=jnp.int32)
    arguments = (histogram_device, states_device, device_grid,
                 config["alpha"], config["tau"], slots_device)
    fresh = pilot._capacity_observation(bundle.compile_signature, post_initialization=True)
    from chu_pair.pair_density.separable_resources import post_initialization_capacity_preflight
    post_gate = post_initialization_capacity_preflight(
        feasibility=feasibility, bundle=bundle, external_capacity=fresh,
        allocator_capacity=pilot._allocator_capacity(pilot_configuration))
    heartbeat({"phase": "pair_execution", "completed": 0, "total": config["steps"],
               "latest_path": str(output)})
    execute_start = time.perf_counter()
    result = benchmark._invoke_accepted_bundle(
        bundle, arguments, case=case, model=model, kernel="separable",
        output_mode="bounded_from_histogram",
        diagnostic_tolerance=config["diagnostic_tolerance"])
    execute_seconds = time.perf_counter() - execute_start
    validate_pair_source_diagnostics(
        result.diagnostics, result.destinations_valid,
        diagnostic_tolerance=config["diagnostic_tolerance"],
        symmetry_tolerance=config["symmetry_tolerance"])
    diagnostics = jax.device_get(result.diagnostics)
    telemetry.append({"phase": "pair_execute", "seconds": execute_seconds,
                      "steps": config["steps"],
                      "seconds_per_step": execute_seconds / max(1, config["steps"])})
    summary = {
        "compile_seconds": compile_seconds, "execute_seconds": execute_seconds,
        "total_seconds": time.perf_counter() - started,
        "executable_signature_sha256": bundle.signature_sha256,
        "compiled_program_sha256": bundle.compiled_program_sha256,
        "compiled_program_evidence": dict(bundle.compiled_program_evidence),
        "bundle_integrity_sha256": bundle.bundle_integrity_sha256,
        "compiled_memory_report": {k: v for k, v in report.items() if k != "executable_signature"},
        "capacity_preflight": gate, "post_initialization_capacity_preflight": post_gate,
        "scientific_diagnostics": pilot._scientific_summary(
            result, configuration=pilot_configuration,
            grid_size=config["grid_size"], kernel="separable"),
    }
    # bounded per-step diagnostics, never a density history
    per_step = []
    total_mass = np.asarray(diagnostics.total_mass)
    for index in range(total_mass.shape[0]):
        per_step.append({
            "step_index": index,
            "total_mass": float(total_mass[index]),
            "mass_error": float(abs(total_mass[index] - 1.0)),
            "symmetry_error": float(np.asarray(diagnostics.symmetry_error)[index]),
            "minimum_mass": float(np.asarray(diagnostics.minimum_mass)[index]),
            "conditional_weight_error": float(np.asarray(diagnostics.conditional_weight_error)[index]),
            "minimum_conditional_variance": float(np.asarray(diagnostics.minimum_conditional_variance)[index]),
            "conditional_moments_valid": bool(np.asarray(diagnostics.conditional_moments_valid)[index]),
            "finite": bool(np.asarray(diagnostics.finite)[index]),
            "nonnegative": bool(np.asarray(diagnostics.nonnegative)[index]),
        })
    return {"summary": summary, "per_step": per_step,
            "source_summaries": result.source_summaries, "grid": device_grid,
            "histogram": histogram, "host_grid": host_grid}


def _concatenate_statistics(parts):
    """Join per-chunk per-run sufficient statistics along the run axis."""

    import numpy as np
    from chu_pair.velocity_variance import BinnedSufficientStatistics

    fields = ("counts", "sum_s1", "sum_s2", "sum_distinct_products", "sum_reward",
              "sum_reward_squared", "sum_selected_q", "sum_selected_q_squared",
              "sum_reward_selected_q", "sum_velocity", "sum_velocity_squared")
    head = parts[0]
    joined = {name: np.concatenate([np.asarray(getattr(p, name)) for p in parts], axis=0)
              for name in fields}
    return BinnedSufficientStatistics(
        bins=head.bins, num_agents=head.num_agents, alpha=head.alpha,
        min_count=head.min_count, observation_dtype=head.observation_dtype,
        effective_q_c_edges=head.effective_q_c_edges,
        effective_q_d_edges=head.effective_q_d_edges, **joined)


def run_abm_chunks(config, histogram, finest_bins, *, chunk_size, checkpoint_dir,
                   resume, telemetry, heartbeat):
    """Deterministic chunked ABM producing only per-run sufficient statistics."""

    import numpy as np
    from chu_pair.abm import complete_graph, simulate_instrumented_batch_jit, aggregate_variance_records
    import jax.numpy as jnp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    graph = complete_graph(config["num_agents"])
    dtype = jnp.float32
    total = config["num_runs"]
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest = {"chunks": [], "config_sha256": config["normalized_sha256"],
                "commit": _git("rev-parse", "HEAD"), "abm_seed": config["abm_seed"],
                "num_runs": total, "chunk_size": chunk_size}
    if resume and manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if (stored.get("config_sha256") != manifest["config_sha256"]
                or stored.get("commit") != manifest["commit"]
                or stored.get("abm_seed") != manifest["abm_seed"]):
            raise ValueError("checkpoint manifest does not match this configuration/commit/seed")
        manifest = stored
    done = {c["start"]: c for c in manifest["chunks"]}
    parts = []
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        indices = list(range(start, stop))
        path = checkpoint_dir / f"runs-{start:06d}-{stop:06d}.npz"
        if start in done and path.exists():
            stored = np.load(path)
            parts.append(_statistics_from_npz(stored, finest_bins, config))
            continue
        chunk_start = time.perf_counter()
        states, keys = initialize_runs_by_global_index(
            graph, histogram, abm_seed=config["abm_seed"], indices=indices, dtype=dtype)
        result = simulate_instrumented_batch_jit(
            states, keys, graph, config["alpha"], config["tau"], steps=config["steps"])
        statistics = aggregate_variance_records(
            result.records, finest_bins, num_agents=config["num_agents"],
            alpha=config["alpha"], min_count=config["minimum_count"])
        del result, states, keys
        arrays = {name: np.asarray(getattr(statistics, name)) for name in (
            "counts", "sum_s1", "sum_s2", "sum_distinct_products", "sum_reward",
            "sum_reward_squared", "sum_selected_q", "sum_selected_q_squared",
            "sum_reward_selected_q", "sum_velocity", "sum_velocity_squared")}
        with tempfile.NamedTemporaryFile(dir=checkpoint_dir, prefix=".ckpt-", suffix=".npz",
                                         delete=False) as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        elapsed = time.perf_counter() - chunk_start
        manifest["chunks"] = [c for c in manifest["chunks"] if c["start"] != start] + [
            {"start": start, "stop": stop, "seconds": elapsed, "sha256": _sha256_file(path)}]
        manifest["chunks"].sort(key=lambda c: c["start"])
        _atomic_json(manifest_path, manifest)
        telemetry.append({"phase": "abm_chunk", "start": start, "stop": stop, "seconds": elapsed,
                          "runs_per_second": (stop - start) / elapsed if elapsed else None})
        heartbeat({"phase": "abm", "completed_runs": stop, "total_runs": total,
                   "seconds_last_chunk": elapsed})
        parts.append(statistics)
    return _concatenate_statistics(parts), manifest


def _statistics_from_npz(stored, bins, config):
    import numpy as np
    from chu_pair.velocity_variance import BinnedSufficientStatistics
    effective_c, effective_d = bins.effective_edges(np.dtype("float32"))
    return BinnedSufficientStatistics(
        bins=bins, num_agents=config["num_agents"], alpha=config["alpha"],
        min_count=config["minimum_count"], observation_dtype="float32",
        effective_q_c_edges=effective_c, effective_q_d_edges=effective_d,
        **{name: stored[name] for name in stored.files})


def _relative_ci_width(rows, estimand="direct_abm_velocity_variance"):
    """Median pointwise relative CI full width over analysable primary strata."""

    import numpy as np
    widths = []
    for row in rows:
        if row.get("sparse") or not row.get("has_abm_observations"):
            continue
        if not row.get(f"{estimand}_interval_valid"):
            continue
        value = row.get(estimand)
        lower, upper = row.get(f"{estimand}_lower"), row.get(f"{estimand}_upper")
        if value is None or lower is None or upper is None:
            continue
        if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper)) or value <= 0:
            continue
        widths.append((upper - lower) / value)
    if not widths:
        return float("nan"), 0, []
    return float(np.median(widths)), len(widths), widths


def main() -> int:
    args = parse_args()
    config = load_production_config(args.config.resolve())
    resource = estimate_production_resources(config)
    if resource["violations"]:
        raise ValueError(f"production resource guard rejected: {resource['violations']}")
    if args.execute and args.confirmation != CONFIRMATION_PHRASE:
        raise ValueError("the exact production confirmation phrase is required to execute")
    if not 1 <= args.run_chunk_size <= PRODUCTION_LIMITS["max_run_chunk"]:
        raise ValueError("run chunk size is outside its production bound")

    apply_allocator_policy(config["allocator_policy"], memory_fraction=config["memory_fraction"])

    commit = _git("rev-parse", "HEAD")
    clean = not bool(_git("status", "--porcelain", "--", "."))
    doctor = json.loads(args.doctor_report.resolve().read_text(encoding="ascii"))
    plan = {
        "configuration": config, "resource_estimate": resource,
        "confirmation_required": True, "dry_run": args.dry_run,
        "would_compile": not args.dry_run, "would_execute": bool(args.execute),
        "run_chunk_size": args.run_chunk_size,
        "production_limits": PRODUCTION_LIMITS,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True, default=str))
        return 0

    if not clean:
        raise ValueError("production execution requires a clean subproject")
    if doctor.get("git", {}).get("commit") != commit or not doctor.get("gpu_ready"):
        raise ValueError("production execution requires a ready doctor at this commit")
    if abs(args.hourly_price_usd - config["hourly_price_usd"]) > 0:
        raise ValueError("--hourly-price-usd must exactly match the reviewed configuration")
    from chu_pair.gpu_pilot.workflow import read_prerequisite_artifact, PilotStage
    prerequisite = read_prerequisite_artifact(
        args.prerequisite.resolve(), required_stage=PilotStage.FULL_GRID_ONE_STEP,
        commit=commit, environment_sha256=str(doctor.get("environment_sha256", "")))

    import numpy as np
    from chu_pair.abm import (anchor_bin_index, assert_child_reconstructs_parent,
                              bootstrap_run_weights)
    from chu_pair.velocity_variance import (QBinSpec, aggregate_pair_points,
                                            bootstrap_four_way_intervals, coarsen_abm_sufficient,
                                            coarsen_pair_sufficient, compare_four_way,
                                            derive_pair_binned_moments,
                                            pair_point_sufficient_from_jax_summary,
                                            select_abm_source_times)
    from experiments import run_velocity_variance_comparison as phase5

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (PROJECT_ROOT / "outputs" / "full_grid_production" /
              f"{config['run_name']}-{timestamp}").resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    telemetry: list = []
    started_utc = datetime.now(timezone.utc)
    start = time.perf_counter()
    heartbeat_path = output / "heartbeat.json"

    def heartbeat(payload):
        _atomic_json(heartbeat_path, {"utc": datetime.now(timezone.utc).isoformat(),
                                      "elapsed_seconds": time.perf_counter() - start, **payload})

    stage = {
        "schema_version": 1, "stage": "full-grid-production", "status": "failed",
        "started_utc": started_utc.isoformat(), "git_commit": commit, "subproject_clean": clean,
        "environment_sha256": doctor.get("environment_sha256"),
        "prerequisite_artifact_sha256": prerequisite["artifact_sha256"],
        "plan": plan, "event_log": [{"event": "production_preflight_accepted",
                                     "utc": started_utc.isoformat()}],
    }
    stage_path = output / "stage.json"

    def write_stage():
        payload = dict(stage)
        payload["artifact_sha256"] = _canonical_sha256(payload)
        _atomic_json(stage_path, payload)

    try:
        heartbeat({"phase": "prerequisite_preparation", "completed": 1, "total": 1,
                   "latest_path": str(args.prerequisite)})
        heartbeat({"phase": "pair_lower_compile_execute", "completed": 0, "total": 1,
                   "latest_path": str(output)})
        pair_stage = run_pair_full_grid(config, output=output, telemetry=telemetry,
                                        heartbeat=heartbeat)
        stage["pair"] = pair_stage["summary"]
        stage["event_log"].append({"event": "pair_full_grid_completed",
                                   "utc": datetime.now(timezone.utc).isoformat()})
        heartbeat({"phase": "pair_complete", "completed": 1, "total": 1,
                   "latest_path": str(output / "stage.json")})
        write_stage()

        schemes = [(s["name"], QBinSpec(np.asarray(s["q_c_edges"], dtype=np.float64),
                                        np.asarray(s["q_d_edges"], dtype=np.float64)))
                   for s in config["bin_schemes"]]
        finest_name, finest_bins = schemes[-1]
        pair_points = pair_point_sufficient_from_jax_summary(
            pair_stage["source_summaries"], config["source_times"], pair_stage["grid"])
        finest_pair = aggregate_pair_points(pair_points, finest_bins)

        abm_statistics, manifest = run_abm_chunks(
            config, pair_stage["histogram"], finest_bins, chunk_size=args.run_chunk_size,
            checkpoint_dir=checkpoints, resume=args.resume, telemetry=telemetry,
            heartbeat=heartbeat)
        stage["checkpoint_manifest"] = {"chunks": len(manifest["chunks"]),
                                        "chunk_size": manifest["chunk_size"]}
        stage["event_log"].append({"event": "abm_runs_completed",
                                   "utc": datetime.now(timezone.utc).isoformat()})
        write_stage()

        heartbeat({"phase": "aggregation", "completed": 0, "total": len(schemes),
                   "latest_path": str(output)})
        results, reconstruction = [], []
        for scheme_index, (name, bins) in enumerate(schemes):
            is_finest = name == finest_name
            abm_scheme = abm_statistics if is_finest else coarsen_abm_sufficient(abm_statistics, bins)
            pair_scheme = finest_pair if is_finest else coarsen_pair_sufficient(finest_pair, bins)
            if not is_finest:
                reconstruction.append({"parent_scheme": name, "child_scheme": finest_name,
                                       **assert_child_reconstructs_parent(abm_scheme, abm_statistics)})
            moments = derive_pair_binned_moments(pair_scheme, num_agents=config["num_agents"],
                                                 alpha=config["alpha"])
            selected = select_abm_source_times(abm_scheme, config["source_times"])
            comparison = compare_four_way(selected, moments,
                                          abm_source_times=config["source_times"],
                                          ratio_epsilon=config["ratio_epsilon"])
            weights = bootstrap_run_weights(config["num_runs"], config["bootstrap_replicates"],
                                            config["bootstrap_seed"])
            heartbeat({"phase": "bootstrap", "scheme": name,
                       "completed": scheme_index, "total": len(schemes),
                       "replicates": config["bootstrap_replicates"],
                       "runs": config["num_runs"], "latest_path": str(output)})
            intervals = bootstrap_four_way_intervals(
                selected, moments, weights, confidence_level=config["confidence_level"])
            heartbeat({"phase": "bootstrap_scheme_complete", "scheme": name,
                       "completed": scheme_index + 1, "total": len(schemes),
                       "latest_path": str(output)})
            results.append((type("S", (), {"name": name, "bins": bins})(), comparison, intervals))
            if is_finest:
                stage["bootstrap_weights_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(weights).tobytes()).hexdigest()
                np.savez_compressed(output / "bootstrap_run_weights.npz", weights=weights)
        stage["nested_reconstruction"] = reconstruction
        stage["event_log"].append({"event": "comparison_completed",
                                   "utc": datetime.now(timezone.utc).isoformat()})

        heartbeat({"phase": "output_serialization", "completed": 0, "total": 4,
                   "latest_path": str(output)})
        rows = list(phase5.iter_comparison_rows(results, config["source_times"], np.dtype("float32")))
        anchor_rows = list(phase5.iter_anchor_rows(results, [tuple(a) for a in config["anchors"]],
                                                   config["source_times"], np.dtype("float32")))
        _write_csv(output / "variance_comparison.csv", rows)
        _write_csv(output / "anchor_bin_refinement.csv", anchor_rows)
        _write_csv(output / "pair_diagnostics.csv", pair_stage["per_step"])
        _write_csv(output / "resource_telemetry.csv", telemetry)

        primary = [r for r in rows if r["scheme"] == finest_name]
        median_width, analysable, widths = _relative_ci_width(primary)
        stage["precision"] = {
            "estimand": "direct_abm_velocity_variance", "scheme": finest_name,
            "median_relative_ci_full_width": median_width,
            "analysable_strata": analysable,
            "target_relative_full_width": 0.20,
            "target_met": bool(median_width <= 0.20) if math.isfinite(median_width) else False,
        }
        heartbeat({"phase": "convergence", "completed": 0, "total": 1,
                   "latest_path": str(output)})
        convergence = _run_count_convergence(
            config, abm_statistics, schemes, finest_pair, finest_name, output,
            heartbeat=heartbeat)
        _write_csv(output / "run_count_convergence.csv", convergence)
        heartbeat({"phase": "final_verification", "completed": 0, "total": 1,
                   "latest_path": str(output / "stage.json")})

        elapsed = time.perf_counter() - start
        stage.update(status="success", completed_utc=datetime.now(timezone.utc).isoformat(),
                     cost={"elapsed_seconds": elapsed,
                           "hourly_price_usd_user_supplied": config["hourly_price_usd"],
                           "estimated_compute_cost_usd": elapsed / 3600.0 * config["hourly_price_usd"],
                           "configured_session_budget_usd": config["max_session_cost_usd"],
                           "claim_scope": "elapsed-time estimate from the user-supplied instance price; not billing data"})
        stage["event_log"].append({"event": "production_succeeded",
                                   "utc": datetime.now(timezone.utc).isoformat()})
        heartbeat({"phase": "complete", "completed": 1, "total": 1,
                   "latest_path": str(stage_path)})
        _atomic_json(output / "metadata.json", _metadata(config, doctor, stage, manifest))
        _atomic_write(output / "production_config.toml", args.config.resolve().read_bytes())
        write_stage()
        print(f"production artifact: {stage_path}", file=sys.stderr)
        print(json.dumps({"status": "success", "output": str(output),
                          "median_relative_ci_full_width": median_width,
                          "analysable_strata": analysable}, sort_keys=True))
        return 0
    except BaseException as error:
        elapsed = time.perf_counter() - start
        stage["error"] = {"type": type(error).__name__, "message": str(error)[:1024]}
        stage["completed_utc"] = datetime.now(timezone.utc).isoformat()
        stage["cost"] = {"status": "failed", "elapsed_seconds": elapsed,
                         "estimated_compute_cost_usd": elapsed / 3600.0 * config["hourly_price_usd"]}
        stage["event_log"].append({"event": "production_failed",
                                   "utc": datetime.now(timezone.utc).isoformat()})
        write_stage()
        raise


def _write_csv(path: Path, rows) -> None:
    import csv
    rows = list(rows)
    if not rows:
        _atomic_write(path, b"")
        return
    columns = sorted({key for row in rows for key in row})
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    _atomic_write(path, buffer.getvalue().encode("ascii", "backslashreplace"))


def _run_count_convergence(config, abm_statistics, schemes, finest_pair, finest_name, output,
                           *, heartbeat=lambda payload: None):
    """Nested global run prefixes; no resampling and no extra simulation."""

    import numpy as np
    from chu_pair.abm import bootstrap_run_weights
    from chu_pair.velocity_variance import (BinnedSufficientStatistics, compare_four_way,
                                            bootstrap_four_way_intervals,
                                            derive_pair_binned_moments, select_abm_source_times)
    from experiments import run_velocity_variance_comparison as phase5
    _, finest_bins = schemes[-1]
    moments = derive_pair_binned_moments(finest_pair, num_agents=config["num_agents"],
                                         alpha=config["alpha"])
    fields = ("counts", "sum_s1", "sum_s2", "sum_distinct_products", "sum_reward",
              "sum_reward_squared", "sum_selected_q", "sum_selected_q_squared",
              "sum_reward_selected_q", "sum_velocity", "sum_velocity_squared")
    total = config["num_runs"]
    prefixes = [r for r in (32, 64, 128, 256, 512, 1024, 2048, 4096) if r <= total]
    if total not in prefixes:
        prefixes.append(total)
    out = []
    for prefix_index, prefix in enumerate(prefixes):
        heartbeat({"phase": "convergence_prefix", "runs": prefix,
                   "completed": prefix_index, "total": len(prefixes),
                   "latest_path": str(output)})
        subset = BinnedSufficientStatistics(
            bins=abm_statistics.bins, num_agents=abm_statistics.num_agents,
            alpha=abm_statistics.alpha, min_count=abm_statistics.min_count,
            observation_dtype=abm_statistics.observation_dtype,
            effective_q_c_edges=abm_statistics.effective_q_c_edges,
            effective_q_d_edges=abm_statistics.effective_q_d_edges,
            **{name: np.asarray(getattr(abm_statistics, name))[:prefix] for name in fields})
        selected = select_abm_source_times(subset, config["source_times"])
        comparison = compare_four_way(selected, moments, abm_source_times=config["source_times"],
                                      ratio_epsilon=config["ratio_epsilon"])
        weights = bootstrap_run_weights(prefix, config["bootstrap_replicates"],
                                        config["bootstrap_seed"])
        intervals = bootstrap_four_way_intervals(selected, moments, weights,
                                                 confidence_level=config["confidence_level"])
        scheme_obj = type("S", (), {"name": finest_name, "bins": finest_bins})()
        rows = list(phase5.iter_comparison_rows([(scheme_obj, comparison, intervals)],
                                                config["source_times"], np.dtype("float32")))
        width, analysable, widths = _relative_ci_width(rows)
        usable = [r for r in rows if r.get("has_abm_observations") and not r.get("sparse")
                  and r.get("pair_valid")]
        def med(key):
            values = [r[key] for r in usable
                      if r.get(key) is not None and np.isfinite(r[key])]
            return float(np.median(values)) if values else float("nan")
        valid_fraction = [r["direct_abm_velocity_variance_valid_replicates"] /
                          config["bootstrap_replicates"] for r in usable
                          if r.get("direct_abm_velocity_variance_valid_replicates") is not None]
        out.append({
            "runs": prefix, "analysable_strata": analysable,
            "median_relative_ci_full_width": width,
            "p10_relative_ci_full_width": float(np.percentile(widths, 10)) if widths else float("nan"),
            "p90_relative_ci_full_width": float(np.percentile(widths, 90)) if widths else float("nan"),
            "median_direct_minus_reconstructed": med("direct_minus_reconstructed"),
            "max_abs_direct_minus_reconstructed": float(np.max(np.abs([
                r["direct_minus_reconstructed"] for r in usable
                if r.get("direct_minus_reconstructed") is not None]))) if usable else float("nan"),
            "median_pair_to_direct_ratio": med("pair_to_direct_ratio"),
            "median_hybrid_to_direct_ratio": med("hybrid_to_direct_ratio"),
            "median_valid_bootstrap_fraction": float(np.median(valid_fraction)) if valid_fraction else float("nan"),
            "median_abm_covariance": med("abm_covariance"),
            "median_pair_covariance": med("pair_covariance"),
        })
    return out


def _metadata(config, doctor, stage, manifest):
    import jax
    return {
        "schema_version": 1,
        "milestone": "human-authorised exact full-grid Phase 5 production variance experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": stage["git_commit"], "subproject_clean": stage["subproject_clean"]},
        "configuration": config,
        "environment_sha256": doctor.get("environment_sha256"),
        "pair_contraction_precision": doctor.get("pair_contraction_precision"),
        "backend": doctor.get("backend"), "devices": doctor.get("devices"),
        "versions": doctor.get("versions"),
        "allocator_environment": doctor.get("allocator_environment"),
        "capacity": doctor.get("capacity"),
        "source_hashes": doctor.get("source_hashes"),
        "pair": stage.get("pair"),
        "checkpoint_manifest": manifest,
        "bootstrap": {"unit": "complete independent ABM run",
                      "replicates": config["bootstrap_replicates"],
                      "seed": config["bootstrap_seed"],
                      "confidence_level": config["confidence_level"],
                      "quantile_method": "linear",
                      "inference_scope": "pointwise descriptive intervals; no multiplicity-adjusted claim"},
        "source_time_convention": "ABM record t and pair P_t before the update to t+1",
        "limitations": [
            "finite population n and finite bins",
            "pair theory does not supply the empirical cross-opponent covariance",
            "pointwise descriptive intervals only; no multiplicity-adjusted hypothesis test",
            "largest_free_block_bytes is unsupported by this backend; fragmentation is not measured",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())

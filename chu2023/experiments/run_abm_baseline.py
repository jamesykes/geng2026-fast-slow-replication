#!/usr/bin/env python3
"""Run a guarded, CPU-safe Phase 2 ABM baseline and write ignored outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
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
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from chu_pair.abm import (
    action_probabilities,
    complete_graph,
    initialize_continuous_paper_batch,
    initialize_grid_matched_batch,
    simulate_batch_jit,
)
from chu_pair.config import ABMConfig, LearningConfig
from chu_pair.grids import QGrid
from chu_pair.initial_conditions import seeded_legacy_histogram


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "abm_baseline_small.toml"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PHASE2_ABSOLUTE_LIMITS = {
    "num_agents": 128,
    "steps": 500,
    "num_runs": 32,
    "run_step_edges": 20_000_000,
    "record_bytes": 256 * 1024**2,
    "state_working_bytes": 256 * 1024**2,
}
# Operational caps for the host-side legacy histogram builder.  They are not
# scientific parameters and cannot be raised by the TOML configuration.  The
# 32 MiB limit applies to the int64 count array itself; subsequent float64 mass
# arrays make a conservative low cap preferable.  Two million Python-loop
# sample pairs (two Beta variates per pair) admits the full 131-point legacy
# grid while rejecting accidentally extreme initialization grids.
PHASE2_HISTOGRAM_ABSOLUTE_LIMITS = {
    "histogram_cells": 5_000_000,
    "histogram_count_bytes": 32 * 1024**2,
    "histogram_sample_pairs": 2_000_000,
}
HISTOGRAM_COUNT_DTYPE = np.dtype(np.int64)
SAFETY_CONFIG_KEYS = {
    "num_agents": "max_agents",
    "steps": "max_steps",
    "num_runs": "max_runs",
    "run_step_edges": "max_run_step_edges",
    "record_bytes": "max_record_bytes",
    "state_working_bytes": "max_state_working_bytes",
}
RESOURCE_MODE_BASELINE = "baseline"
RESOURCE_MODE_INSTRUMENTED = "instrumented"
BASELINE_RECORD_AGENT_FLOAT_FIELDS = {
    "q_t": 2,
    "action_probabilities_t": 2,
    "rewards_t": 1,
    "selected_velocities_t": 1,
    "q_t_plus_1": 2,
}
BASELINE_WORKING_AGENT_FLOAT_FIELDS = {
    "q_state_buffers": 4,
    "action_probabilities": 2,
    "rewards": 1,
    "selected_velocities": 1,
}
INSTRUMENTED_RECORD_AGENT_FLOAT_FIELDS = {
    "selected_q_t": 1,
    "payoff_sums_t": 1,
    "payoff_square_sums_t": 1,
}
INSTRUMENTED_WORKING_AGENT_FLOAT_FIELDS = {
    "payoff_sums_t": 1,
    "payoff_square_sums_t": 1,
}
RESOURCE_MODES = {
    RESOURCE_MODE_BASELINE: {
        "record_agent_float_fields": BASELINE_RECORD_AGENT_FLOAT_FIELDS,
        # The committed Phase 2 guard included one conservative float-width
        # allowance beyond the eight explicitly retained floating fields.
        "committed_record_float_allowance": 1,
        "working_agent_float_fields": BASELINE_WORKING_AGENT_FLOAT_FIELDS,
        "additional_record_fields": [],
        "additional_working_fields": [],
    },
    RESOURCE_MODE_INSTRUMENTED: {
        # InstrumentedStepRecord adds three (R,T,n) floating arrays:
        # selected_q_t, payoff_sums_t, and payoff_square_sums_t.  Computing
        # S1/S2 also keeps two additional agent-sized floating accumulators live.
        "record_agent_float_fields": {
            **BASELINE_RECORD_AGENT_FLOAT_FIELDS,
            **INSTRUMENTED_RECORD_AGENT_FLOAT_FIELDS,
        },
        "committed_record_float_allowance": 1,
        "working_agent_float_fields": {
            **BASELINE_WORKING_AGENT_FLOAT_FIELDS,
            **INSTRUMENTED_WORKING_AGENT_FLOAT_FIELDS,
        },
        "additional_record_fields": [
            "selected_q_t",
            "payoff_sums_t",
            "payoff_square_sums_t",
        ],
        "additional_working_fields": ["payoff_sums_t", "payoff_square_sums_t"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="override the conservative Phase 2 resource guardrails",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _configured_limits(safety: dict) -> dict[str, int]:
    configured = {}
    for name in PHASE2_ABSOLUTE_LIMITS:
        key = SAFETY_CONFIG_KEYS[name]
        value = safety[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"safety.{key} must be a non-negative integer")
        configured[name] = value
    return configured


def validate_resource_budget(
    abm: ABMConfig,
    safety: dict,
    allow_expensive: bool,
    *,
    record_mode: str = RESOURCE_MODE_BASELINE,
) -> dict:
    """Reject expensive work before graph or batched state allocation."""

    if not isinstance(record_mode, str) or record_mode not in RESOURCE_MODES:
        valid = ", ".join(sorted(RESOURCE_MODES))
        raise ValueError(f"record_mode must be one of: {valid}")
    layout = RESOURCE_MODES[record_mode]
    record_agent_float_scalars = sum(layout["record_agent_float_fields"].values())
    record_agent_float_scalars += layout["committed_record_float_allowance"]
    working_agent_float_scalars = sum(
        layout["working_agent_float_fields"].values()
    )

    edge_count = abm.edge_count
    run_step_edges = abm.num_runs * abm.steps * edge_count
    # The baseline widths are the committed Phase 2 guard semantics.  The
    # instrumented mode adds the three explicit Phase 3A record fields above.
    item_bytes = np.dtype(abm.dtype).itemsize
    record_bytes = abm.num_runs * abm.steps * (
        record_agent_float_scalars * abm.num_agents * item_bytes
        + abm.num_agents
        + 4 * item_bytes
    )
    # Conservative live-state/one-step estimate, including shared graph endpoints,
    # two state buffers, edge actions/payoffs, and agent-sized temporaries.
    state_working_bytes = 2 * edge_count * np.dtype(np.int32).itemsize
    state_working_bytes += abm.num_runs * (
        working_agent_float_scalars * abm.num_agents * item_bytes
        + 2 * abm.num_agents
        + edge_count * (4 * item_bytes + 8)
    )
    values = {
        "num_agents": abm.num_agents,
        "steps": abm.steps,
        "num_runs": abm.num_runs,
        "run_step_edges": run_step_edges,
        "record_bytes": record_bytes,
        "state_working_bytes": state_working_bytes,
    }
    configured = _configured_limits(safety)
    effective = {
        name: min(configured[name], PHASE2_ABSOLUTE_LIMITS[name])
        for name in PHASE2_ABSOLUTE_LIMITS
    }
    violations = [
        f"{name}={value} exceeds {effective[name]}"
        for name, value in values.items()
        if value > effective[name]
    ]
    if abm.num_agents >= 1_000:
        violations.append("n>=1000 is outside the bounded Phase 2 milestone")
    if violations and not allow_expensive:
        raise ValueError(
            "refusing expensive baseline configuration; pass --allow-expensive to override: "
            + "; ".join(violations)
        )
    return {
        "allow_expensive": allow_expensive,
        "record_mode": record_mode,
        "record_layout": {
            "agent_float_fields_per_run_step": layout[
                "record_agent_float_fields"
            ],
            "committed_agent_float_allowance_per_run_step": layout[
                "committed_record_float_allowance"
            ],
            "agent_float_scalars_per_run_step": record_agent_float_scalars,
            "agent_int8_scalars_per_run_step": 1,
            "state_summary_float_scalars_per_run_step": 4,
            "additional_record_fields": layout["additional_record_fields"],
            "working_agent_float_fields_per_run": layout[
                "working_agent_float_fields"
            ],
            "working_agent_float_scalars_per_run": working_agent_float_scalars,
            "additional_working_fields": layout["additional_working_fields"],
        },
        "values": values,
        "configured_limits": configured,
        "absolute_limits": PHASE2_ABSOLUTE_LIMITS,
        "effective_limits": effective,
        "violations_overridden": violations if allow_expensive else [],
    }


def estimate_legacy_histogram_resources(
    grid: QGrid,
    samples_per_grid_cell: int,
) -> dict[str, int | float | list[int]]:
    """Estimate the exact host count allocation and sampling loop with Python ints."""

    if (
        isinstance(samples_per_grid_cell, bool)
        or not isinstance(samples_per_grid_cell, int)
        or samples_per_grid_cell <= 0
    ):
        raise ValueError("samples_per_grid_cell must be a positive integer")

    # seeded_legacy_histogram allocates this exact shape and, when sample_count
    # is omitted, iterates agent_point_count * samples_per_grid_cell times.
    histogram_shape = (int(grid.size), int(grid.size))
    histogram_cells = histogram_shape[0] * histogram_shape[1]
    agent_point_count = int(grid.agent_point_count)
    if histogram_cells != agent_point_count:
        raise RuntimeError(
            "QGrid.agent_point_count disagrees with the legacy histogram shape"
        )
    histogram_count_bytes = histogram_cells * int(HISTOGRAM_COUNT_DTYPE.itemsize)
    histogram_sample_pairs = agent_point_count * samples_per_grid_cell
    return {
        "grid_q_min": float(grid.q_min),
        "grid_q_max": float(grid.q_max),
        "grid_spacing": float(grid.spacing),
        "grid_size": histogram_shape[0],
        "histogram_shape": list(histogram_shape),
        "histogram_cells": histogram_cells,
        "histogram_count_bytes": histogram_count_bytes,
        "histogram_sample_pairs": histogram_sample_pairs,
    }


def validate_legacy_histogram_budget(
    grid: QGrid,
    samples_per_grid_cell: int,
    allow_expensive: bool,
    *,
    limits: dict[str, int] | None = None,
) -> dict:
    """Reject an oversized legacy histogram before allocation or sampling."""

    estimates = estimate_legacy_histogram_resources(grid, samples_per_grid_cell)
    effective_limits = dict(
        PHASE2_HISTOGRAM_ABSOLUTE_LIMITS if limits is None else limits
    )
    if set(effective_limits) != set(PHASE2_HISTOGRAM_ABSOLUTE_LIMITS):
        raise ValueError("histogram limits must define all three resource caps")
    for name, value in effective_limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"histogram limit {name} must be a non-negative integer")

    violations = [
        name
        for name in PHASE2_HISTOGRAM_ABSOLUTE_LIMITS
        if estimates[name] > effective_limits[name]
    ]
    if violations and not allow_expensive:
        resources = "; ".join(
            f"{name}={estimates[name]:,} (limit {effective_limits[name]:,})"
            for name in PHASE2_HISTOGRAM_ABSOLUTE_LIMITS
        )
        grid_description = (
            f"q_min={grid.q_min}, q_max={grid.q_max}, spacing={grid.spacing}, "
            f"size={grid.size}"
        )
        raise ValueError(
            "refusing expensive grid-matched histogram before allocation or sampling: "
            f"{resources}; grid({grid_description}); each sample pair draws two Beta "
            "variates; pass --allow-expensive to override"
        )
    return {
        "allow_expensive": allow_expensive,
        "estimates": estimates,
        "absolute_limits": effective_limits,
        "violations_overridden": violations if allow_expensive else [],
    }


def construct_grid_matched_histogram(
    initial: dict,
    allow_expensive: bool,
):
    """Preflight, then construct the runner's sole configurable histogram path."""

    grid = QGrid(
        q_min=float(initial["q_min"]),
        q_max=float(initial["q_max"]),
        spacing=float(initial["spacing"]),
    )
    samples_per_grid_cell = int(initial["samples_per_grid_cell"])
    budget = validate_legacy_histogram_budget(
        grid,
        samples_per_grid_cell,
        allow_expensive,
    )
    histogram = seeded_legacy_histogram(
        grid,
        seed=int(initial["histogram_seed"]),
        samples_per_grid_cell=samples_per_grid_cell,
    )
    return histogram, budget


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_source_hashes(config_path: Path) -> dict[str, str]:
    """Hash the dirty-tree implementation as well as immutable provenance."""

    paths = [
        PROJECT_ROOT / "case2_1.py",
        PROJECT_ROOT / "pair-approx_multi-agent_stochastic_games.pdf",
        PROJECT_ROOT / "MODEL_SPEC.md",
        PROJECT_ROOT / "PLAN.md",
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        *sorted((PROJECT_ROOT / "src" / "chu_pair").rglob("*.py")),
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256(path)
        for path in paths
    }
    hashes["input_config"] = sha256(config_path)
    return hashes


def git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def device_metadata() -> list[dict]:
    return [
        {
            "id": device.id,
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]


def state_trajectory(result, initial_state, tau: float) -> tuple[np.ndarray, ...]:
    records = result.records
    steps = records.actions_t.shape[1]
    if steps == 0:
        q = np.asarray(initial_state.q_values)[:, None, :, :]
        policies = np.asarray(action_probabilities(initial_state.q_values, tau))[:, None, :, :]
        edge_states = np.asarray(initial_state.edge_states)
        proportions = np.stack(
            (
                np.mean(edge_states == 0, axis=1),
                np.mean(edge_states == 1, axis=1),
            ),
            axis=-1,
        )[:, None, :]
        return q, policies, proportions

    q = np.concatenate(
        (np.asarray(records.q_t)[:, :1], np.asarray(records.q_t_plus_1)),
        axis=1,
    )
    policies = np.concatenate(
        (
            np.asarray(records.action_probabilities_t)[:, :1],
            np.asarray(action_probabilities(records.q_t_plus_1, tau)),
        ),
        axis=1,
    )
    proportions = np.concatenate(
        (
            np.asarray(records.edge_state_proportions_t)[:, :1],
            np.asarray(records.edge_state_proportions_t_plus_1),
        ),
        axis=1,
    )
    return q, policies, proportions


def chronology_label(state_t: int, steps: int) -> str:
    if steps == 0:
        return "Q_0,S_0 initial/final state; no steps executed"
    if state_t < steps:
        return f"Q_{state_t},S_{state_t} before step {state_t}"
    return f"Q_{state_t},S_{state_t} after final step {state_t - 1}"


def write_outputs(
    run_directory: Path,
    config_path: Path,
    config: dict,
    abm: ABMConfig,
    learning: LearningConfig,
    initialization,
    result,
    resource_budget: dict,
) -> None:
    q, policies, proportions = state_trajectory(result, initialization.state, learning.tau)
    mean_q = q.mean(axis=(0, 2))
    mean_policy = policies.mean(axis=(0, 2))
    mean_state = proportions.mean(axis=0)

    with (run_directory / "state_trajectory.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "state_t",
                "chronology",
                "mean_q_c",
                "mean_q_d",
                "mean_policy_c",
                "mean_policy_d",
                "state_sh_proportion",
                "state_pd_proportion",
            ]
        )
        for state_t in range(abm.steps + 1):
            writer.writerow(
                [
                    state_t,
                    chronology_label(state_t, abm.steps),
                    *mean_q[state_t].tolist(),
                    *mean_policy[state_t].tolist(),
                    *mean_state[state_t].tolist(),
                ]
            )

    records = result.records
    np.savez_compressed(
        run_directory / "step_records.npz",
        step_t=np.arange(abm.steps, dtype=np.int32),
        actions_t=np.asarray(records.actions_t),
        rewards_t=np.asarray(records.rewards_t),
        selected_velocities_t=np.asarray(records.selected_velocities_t),
    )

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "resource_budget": resource_budget,
        "initialization": initialization.metadata.__dict__
        if hasattr(initialization.metadata, "__dict__")
        else {
            field: getattr(initialization.metadata, field)
            for field in initialization.metadata.__dataclass_fields__
        },
        "key_convention": (
            "ABM seed -> independent run roots -> Q, edge-state, and dynamics keys; "
            "each dynamics step splits action_key then continuation_key"
        ),
        "backend": jax.default_backend(),
        "devices": device_metadata(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
        },
        "git": {
            "commit": git_text("rev-parse", "HEAD"),
            "subproject_status_before_output": git_text("status", "--short", "--", "."),
        },
        "source_hashes": implementation_source_hashes(config_path),
        "array_shapes": {
            "final_q": list(result.final_state.q_values.shape),
            "final_edge_states": list(result.final_state.edge_states.shape),
            "actions_t": list(result.records.actions_t.shape),
            "rewards_t": list(result.records.rewards_t.shape),
            "selected_velocities_t": list(result.records.selected_velocities_t.shape),
        },
    }
    with (run_directory / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        "initial state: "
        f"mean_Q={mean_q[0].tolist()} mean_policy={mean_policy[0].tolist()} "
        f"state_proportions={mean_state[0].tolist()}"
    )
    print(
        "final state:   "
        f"mean_Q={mean_q[-1].tolist()} mean_policy={mean_policy[-1].tolist()} "
        f"state_proportions={mean_state[-1].tolist()}"
    )
    print(f"wrote {run_directory}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    learning = LearningConfig(**config["model"])
    abm = ABMConfig(**config["simulation"])
    resource_budget = validate_resource_budget(
        abm, config["safety"], args.allow_expensive
    )
    dtype = jnp.float32 if abm.dtype == "float32" else jnp.float64
    graph = complete_graph(abm.num_agents)

    initial = config["initial_condition"]
    mode = initial["mode"]
    if mode == "grid_matched":
        histogram, histogram_budget = construct_grid_matched_histogram(
            initial,
            args.allow_expensive,
        )
        resource_budget["grid_matched_histogram"] = histogram_budget
        initialization = initialize_grid_matched_batch(
            graph,
            histogram,
            abm_seed=abm.abm_seed,
            num_runs=abm.num_runs,
            dtype=dtype,
        )
    elif mode == "continuous_paper":
        resource_budget["grid_matched_histogram"] = {"mode": "not_applicable"}
        initialization = initialize_continuous_paper_batch(
            graph,
            abm_seed=abm.abm_seed,
            num_runs=abm.num_runs,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unsupported initial_condition.mode: {mode!r}")

    result = simulate_batch_jit(
        initialization.state,
        initialization.simulation_key,
        graph,
        learning.alpha,
        learning.tau,
        steps=abm.steps,
    )
    result.final_state.q_values.block_until_ready()

    run_name = config["output"]["run_name"]
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("output.run_name contains unsupported path characters")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = (PROJECT_ROOT / "outputs" / "abm").resolve()
    run_directory = (output_root / f"{run_name}-{timestamp}").resolve()
    if output_root not in run_directory.parents:
        raise ValueError("output directory must remain beneath outputs/abm")
    run_directory.mkdir(parents=True, exist_ok=False)
    write_outputs(
        run_directory,
        config_path,
        config,
        abm,
        learning,
        initialization,
        result,
        resource_budget,
    )


if __name__ == "__main__":
    main()

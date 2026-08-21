"""JAX-importing runtime for analyzed exact separable pilot execution."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
import subprocess
import threading
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from chu_pair.grids import QGrid
from chu_pair.initial_conditions import seeded_legacy_histogram
from chu_pair.pair_density import (
    JAXPairGrid,
    allocator_capacity_observation,
    build_jax_pair_grid,
    discover_nvidia_device_capacity,
    flat_validation_capacity_preflight,
    production_capacity_preflight,
    post_initialization_capacity_preflight,
    simulate_pair_source_summaries_from_histogram_jit,
    validate_compiled_executable_bundle,
    validate_pair_source_diagnostics,
)
from chu_pair.gpu_pilot.cuda_identity import CudaDriverIdentityProvider
from chu_pair.gpu_pilot.workflow import (
    PilotConfiguration,
    REVIEWED_GRID_SPECS,
    validate_analyzed_signature_match,
    validate_allocator_identity,
)
from experiments import run_pair_separable_benchmark as benchmark


MAX_TELEMETRY_OUTPUT_BYTES = 16 * 1024
TELEMETRY_INTERVAL_SECONDS = 0.1
_REPETITIONS = {"small": 3, "medium": 2, "large-pilot": 1, "full-grid-one-step": 1}


def _abstract_grid(grid: QGrid, dtype) -> JAXPairGrid:
    factor = 10**grid.decimal_places
    spacing_ticks = int(np.around(grid.spacing * factor))
    q_min_ticks = int(np.around(grid.q_min * factor))
    points = grid.size * grid.size
    return JAXPairGrid(
        size=grid.size,
        decimal_factor=factor,
        spacing_ticks=spacing_ticks,
        q_min_ticks=q_min_ticks,
        values=jax.ShapeDtypeStruct((grid.size,), dtype),
        q_points=jax.ShapeDtypeStruct((points, 2), dtype),
        q_c_indices=jax.ShapeDtypeStruct((points,), jnp.int32),
        q_d_indices=jax.ShapeDtypeStruct((points,), jnp.int32),
    )


def _case(configuration: PilotConfiguration, grid_size: int) -> dict[str, object]:
    points = grid_size * grid_size
    try:
        q_min, q_max, spacing = REVIEWED_GRID_SPECS[grid_size]
    except KeyError as error:
        raise ValueError("pilot grid size has no reviewed legacy-aligned grid") from error
    return {
        "label": f"gpu-{configuration.stage.value}-g{grid_size}",
        "q_min": q_min, "q_max": q_max,
        "spacing": spacing,
        "grid_size": grid_size, "agent_grid_points": points,
        "state_expanded_cells": 2 * points * points,
        "density_bytes": 2 * points * points * (4 if configuration.dtype == "float32" else 8),
        "dtype": configuration.dtype,
        "item_bytes": 4 if configuration.dtype == "float32" else 8,
        "steps": configuration.steps,
        "source_times": list(configuration.source_times),
        "summary_count": len(configuration.source_times),
        "flat_chunk_size": min(2 * points * points, 65_536),
        "row_block_size": configuration.row_block_size,
        "column_block_size": configuration.column_block_size,
        "diagnostic_tolerance": configuration.diagnostic_tolerance,
        "symmetry_tolerance": configuration.symmetry_tolerance,
    }


def _model(configuration: PilotConfiguration) -> dict[str, object]:
    return {
        "alpha": configuration.alpha,
        "tau": configuration.tau,
        "state_probabilities": list(configuration.state_probabilities),
    }


def _device_memory_statistics():
    """Read the exact JAX device's allocator statistics, or None."""

    device = jax.devices()[0]
    reader = getattr(device, "memory_stats", None)
    try:
        return reader() if callable(reader) else None
    except Exception:
        return None


def _capacity_observation(signature: dict[str, object], *, post_initialization: bool = False):
    provider = CudaDriverIdentityProvider.from_system()
    return discover_nvidia_device_capacity(
        signature["execution_device"],
        cuda_visible_ordinal_mapper=provider.map_visible_ordinal,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        preallocate_setting=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        memory_fraction_setting=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        post_initialization=post_initialization,
        allocator_statistics=_device_memory_statistics() if post_initialization else None,
    )


def _allocator_capacity(configuration: PilotConfiguration):
    """Read only supported allocator facts after device inputs exist."""

    device = jax.devices()[0]
    reader = getattr(device, "memory_stats", None)
    try:
        statistics = reader() if callable(reader) else None
    except Exception:
        statistics = None
    return allocator_capacity_observation(
        statistics, policy=configuration.allocator_policy
    )


class _GpuMemorySampler:
    def __init__(self, *, runner: Callable[..., object] = subprocess.run) -> None:
        self._runner = runner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes: int | None = None
        self.error: str | None = None

    def _sample(self) -> None:
        command = [
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                completed = self._runner(
                    command, check=True, capture_output=True, text=True, timeout=2,
                )
                output = completed.stdout
                if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_TELEMETRY_OUTPUT_BYTES:
                    raise ValueError("telemetry output missing or over bound")
                for line in output.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and int(fields[0]) == os.getpid():
                        used = int(fields[1]) * 1024**2
                        self.peak_bytes = used if self.peak_bytes is None else max(self.peak_bytes, used)
            except Exception as error:  # telemetry is diagnostic, never an admission substitute
                self.error = f"{type(error).__name__}: {error}"[:512]
                return
            self._stop.wait(TELEMETRY_INTERVAL_SECONDS)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *unused) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def _scientific_summary(result) -> dict[str, object]:
    diagnostics = jax.device_get(result.diagnostics)
    return {
        "total_mass": np.asarray(diagnostics.total_mass).tolist(),
        "state_masses": np.asarray(diagnostics.state_masses).tolist(),
        "mean_q": np.asarray(diagnostics.mean_q).tolist(),
        "mean_action_probability": np.asarray(diagnostics.mean_action_probability).tolist(),
        "symmetry_error": np.asarray(diagnostics.symmetry_error).tolist(),
        "minimum_mass": np.asarray(diagnostics.minimum_mass).tolist(),
        "all_finite": bool(np.all(np.asarray(diagnostics.finite))),
        "all_nonnegative": bool(np.all(np.asarray(diagnostics.nonnegative))),
        "all_destinations_valid": bool(np.all(np.asarray(result.destinations_valid))),
    }


def compile_and_maybe_execute_case(
    configuration: PilotConfiguration,
    *,
    grid_size: int,
    static_estimate: dict[str, object],
    execute: bool,
    expected_signature_sha256: str | None = None,
) -> dict[str, object]:
    """Lower, compile, completely analyze, capacity-gate, then optionally invoke."""

    if jax.default_backend() != "gpu":
        raise RuntimeError("pilot numerical stages require a JAX GPU backend")
    dtype = jnp.float32 if configuration.dtype == "float32" else jnp.float64
    if dtype == jnp.float64 and not jax.config.read("jax_enable_x64"):
        raise RuntimeError("float64 pilot requires JAX_ENABLE_X64=1 before import")
    case = _case(configuration, grid_size)
    model = _model(configuration)
    host_grid = QGrid(case["q_min"], case["q_max"], case["spacing"])
    if host_grid.size != grid_size:
        raise RuntimeError("pilot QGrid disagrees with allocation-free grid count")
    abstract_grid = _abstract_grid(host_grid, dtype)
    points = grid_size * grid_size
    abstract_histogram = jax.ShapeDtypeStruct((points,), dtype)
    abstract_states = jax.ShapeDtypeStruct((2,), dtype)
    slots_host = benchmark.source_slots(list(configuration.source_times), configuration.steps)
    abstract_slots = jax.ShapeDtypeStruct(slots_host.shape, jnp.int32)
    kernels = ("flat", "separable") if configuration.stage.value == "small" else ("separable",)
    compiled = {}
    for kernel in kernels:
        static = {
            "steps": configuration.steps,
            "summary_count": len(configuration.source_times),
            "chunk_size": case["flat_chunk_size"],
            "diagnostic_tolerance": configuration.diagnostic_tolerance,
            "kernel": kernel,
            "row_block_size": configuration.row_block_size,
            "column_block_size": configuration.column_block_size,
        }
        lowered = simulate_pair_source_summaries_from_histogram_jit.lower(
            abstract_histogram, abstract_states, abstract_grid,
            configuration.alpha, configuration.tau, abstract_slots, **static,
        )
        signature = benchmark.executable_signature(
            case, model, histogram=abstract_histogram,
            state_probabilities=abstract_states, grid=abstract_grid,
            slots=slots_host, kernel=kernel, output_mode="bounded_from_histogram",
        )
        validate_allocator_identity(configuration, signature["allocator_environment"])
        kernel_estimate = (
            static_estimate["flat_validation"] if kernel == "flat" else static_estimate
        )
        static_host = int(kernel_estimate["host_planning_threshold_bytes"])
        bundle, compile_seconds = benchmark._compile_and_analyze(
            lowered, signature, static_host_bytes=static_host,
        )
        if (
            expected_signature_sha256 is not None
            and kernel == "separable"
        ):
            validate_analyzed_signature_match(
                expected_signature_sha256, bundle.signature_sha256
            )
        report = validate_compiled_executable_bundle(bundle)
        feasibility = {
            "state_expanded_cells": case["state_expanded_cells"],
            "safety_margin_fraction": configuration.safety_margin_fraction,
            "minimum_device_memory_bytes": math.ceil(
                max(int(kernel_estimate["static_device_bytes"]), int(report["compiled_device_requirement_bytes"]))
                * (1.0 + configuration.safety_margin_fraction)
            ),
        }
        # ``lowered.compile()`` above already reserved the configured
        # preallocation pool, so this admission point is past allocator
        # initialization.  Device inputs still do not exist, so the identity
        # gate keeps charging the full analysed requirement.
        capacity = _capacity_observation(signature, post_initialization=True)
        if kernel == "separable":
            capacity_gate = production_capacity_preflight(
                feasibility=feasibility, bundle=bundle,
                capacity_observation=capacity, allow_expensive=False,
            )
        else:
            capacity_gate = flat_validation_capacity_preflight(
                feasibility=feasibility, bundle=bundle,
                capacity_observation=capacity,
            )
        compiled[kernel] = {
            "bundle": bundle, "signature": signature, "feasibility": feasibility,
            "record": {
                "compile_seconds": compile_seconds,
                "compiled_memory_report": {
                    key: value for key, value in report.items() if key != "executable_signature"
                },
                "executable_signature_sha256": bundle.signature_sha256,
                "capacity_preflight": capacity_gate,
                "invoked": False,
            },
        }
    record: dict[str, object] = {
        "grid_size": grid_size,
        "steps": configuration.steps,
        "kernels": {kernel: entry["record"] for kernel, entry in compiled.items()},
        "invoked": False,
    }
    if not execute:
        return record

    # Only after exact compiled analysis and capacity admission do any device
    # initialization inputs exist. The combined executable constructs P0 itself.
    device_grid = build_jax_pair_grid(host_grid, dtype)
    histogram_host = seeded_legacy_histogram(
        host_grid, seed=configuration.histogram_seed,
        samples_per_grid_cell=configuration.samples_per_grid_cell,
    ).mass.reshape(-1)
    histogram = jnp.asarray(histogram_host, dtype=dtype)
    states = jnp.asarray(configuration.state_probabilities, dtype=dtype)
    slots = jnp.asarray(slots_host, dtype=jnp.int32)
    arguments = (
        histogram, states, device_grid, configuration.alpha, configuration.tau, slots,
    )
    repetitions = _REPETITIONS[configuration.stage.value]
    outputs = {}
    for repetition in range(repetitions):
        order = kernels if repetition % 2 == 0 else tuple(reversed(kernels))
        for kernel in order:
            entry = compiled[kernel]
            fresh_capacity = _capacity_observation(entry["signature"], post_initialization=True)
            post_capacity = post_initialization_capacity_preflight(
                feasibility=entry["feasibility"], bundle=entry["bundle"],
                external_capacity=fresh_capacity,
                allocator_capacity=_allocator_capacity(configuration),
            )
            start = time.perf_counter()
            with _GpuMemorySampler() as sampler:
                invoke_keywords = {}
                result = benchmark._invoke_accepted_bundle(
                    entry["bundle"], arguments, case=case, model=model, kernel=kernel,
                    output_mode="bounded_from_histogram",
                    diagnostic_tolerance=configuration.diagnostic_tolerance,
                    **invoke_keywords,
                )
            elapsed = time.perf_counter() - start
            outputs[kernel] = result
            kernel_record = entry["record"]
            kernel_record.setdefault("post_initialization_capacity_preflight", []).append(post_capacity)
            kernel_record.setdefault("execution_seconds", []).append(elapsed)
            if sampler.peak_bytes is not None:
                current = kernel_record.get("peak_gpu_process_memory_bytes")
                kernel_record["peak_gpu_process_memory_bytes"] = (
                    sampler.peak_bytes if current is None else max(current, sampler.peak_bytes)
                )
            if sampler.error:
                kernel_record.setdefault("gpu_memory_telemetry_errors", []).append(sampler.error)
    for kernel, result in outputs.items():
        validate_pair_source_diagnostics(
            result.diagnostics, result.destinations_valid,
            diagnostic_tolerance=configuration.diagnostic_tolerance,
            symmetry_tolerance=configuration.symmetry_tolerance,
        )
        kernel_record = compiled[kernel]["record"]
        samples = kernel_record["execution_seconds"]
        sample_array = np.asarray(samples, dtype=np.float64)
        median = float(np.median(sample_array))
        kernel_record.update(
            invoked=True, execution_repetitions=repetitions,
            first_execution_seconds=samples[0],
            subsequent_execution_seconds=samples[1:],
            median_execution_seconds=median,
            minimum_execution_seconds=float(np.min(sample_array)),
            maximum_execution_seconds=float(np.max(sample_array)),
            mad_execution_seconds=float(np.median(np.abs(sample_array - median))),
            scientific_diagnostics=_scientific_summary(result),
        )
    parity = None
    if kernels == ("flat", "separable"):
        parity = {
            "source_summary_max_abs": benchmark._max_tree_difference(
                outputs["flat"].source_summaries, outputs["separable"].source_summaries
            ),
            "diagnostic_max_abs": benchmark._max_tree_difference(
                outputs["flat"].diagnostics, outputs["separable"].diagnostics
            ),
        }
        if max(parity.values()) > configuration.diagnostic_tolerance:
            raise RuntimeError(f"small GPU flat/separable parity failed: {parity}")
    record.update(invoked=True, parity=parity)
    return record

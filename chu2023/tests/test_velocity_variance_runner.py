from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import io
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.abm import NamedBinScheme, QBinSpec
from chu_pair.config import LearningConfig
from chu_pair.grids import QGrid
from chu_pair.pair_density import (
    JAXPairDiagnostics,
    build_jax_pair_grid,
    checked_pair_mass_step,
    pair_point_sufficient_jax,
    validate_pair_source_diagnostics,
)
from chu_pair.velocity_variance import ComparisonBootstrapSummary, FourWayComparison
from experiments import run_velocity_variance_comparison as runner


def _raw() -> dict:
    return runner.inspect_comparison_config(
        runner.baseline.load_config(runner.DEFAULT_CONFIG)
    )


def _small_pair_case(source_times=(0, 1), dtype="float32"):
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["simulation"].update(
        steps=max(source_times) + 1, dtype=dtype, num_agents=3, num_runs=2
    )
    config["initial_condition"].update(
        q_min=-0.5, q_max=1.0, spacing=0.5, samples_per_grid_cell=2
    )
    config["comparison"]["source_times"] = list(source_times)
    config["pair_solver"]["chunk_size"] = 8
    config["bin_schemes"] = [
        {"name": "coarse", "q_c_edges": [-0.5, 0.5, 1.0], "q_d_edges": [-0.5, 0.5, 1.0]},
        {"name": "fine", "q_c_edges": [-0.5, 0.0, 0.5, 1.0], "q_d_edges": [-0.5, 0.0, 0.5, 1.0]},
    ]
    config["anchors"]["points"] = [[0.0, 0.0], [1.0, 1.0]]
    raw = runner.inspect_comparison_config(config)
    scalar_dtype = jnp.float32 if dtype == "float32" else jnp.float64
    grid = build_jax_pair_grid(QGrid(-0.5, 1.0, 0.5), scalar_dtype)
    initial = jnp.zeros((2, 16, 16), dtype=scalar_dtype)
    initial = initial.at[0, 1, 1].set(0.2)
    initial = initial.at[1, 6, 6].set(0.3)
    initial = initial.at[0, 1, 6].set(0.1)
    initial = initial.at[0, 6, 1].set(0.1)
    initial = initial.at[1, 2, 9].set(0.15)
    initial = initial.at[1, 9, 2].set(0.15)
    learning = LearningConfig(alpha=0.6, tau=1.3)
    return raw, grid, initial, learning


@lru_cache(maxsize=None)
def _compile_small_case(source_times=(0, 1), dtype="float32"):
    raw, grid, initial, learning = _small_pair_case(source_times, dtype)
    bundle = runner.analyze_compiled_phase5_pair_memory(raw, grid, learning)
    return raw, grid, initial, learning, bundle


def test_comparison_resource_estimate_matches_independent_hand_calculation() -> None:
    estimate = runner.estimate_phase5_resources(_raw())

    assert estimate["source_times"] == 2
    assert estimate["scheme_cells"] == [16, 64]
    assert estimate["comparison_rows"] == 80
    assert estimate["anchor_rows"] == 24
    assert estimate["pair_point_host_bytes"] == (
        2 * 196 * 15 * 4 + 2 * 196 * 4 + 2 * 8
    )
    assert estimate["pair_point_device_bytes"] == 2 * 196 * 15 * 4
    assert estimate["pair_binned_bytes"] == (
        64 * 56 + 32 * 8 + 16 * 56 + 8 * 8
    )
    assert estimate["retained_comparison_bytes"] == 80 * 512
    assert estimate["comparison_bootstrap_work_bytes"] == 64 * 64 * 64
    assert estimate["serialization_live_peak_bytes"] == (
        3 * 8 * (256 * 1024) + 64 * 1024
    )
    assert estimate["maximum_live_python_row_bytes"] == 16 * 1024
    assert estimate["diagnostic_device_bytes"] == 2 * (11 * 4 + 3)
    assert estimate["diagnostic_host_bytes"] == 2 * (11 * 4 + 3)
    assert estimate["destination_validity_device_bytes"] == 1
    assert estimate["destination_validity_host_bytes"] == 1
    assert estimate["pair_retained_device_bytes"] == 641_471
    assert estimate["pair_transfer_host_bytes"] == 332_527
    assert estimate["static_pair_kernel_device_bytes"] == 3_177_559
    assert estimate["encoded_metadata_retained_bytes"] == 8 * 256 * 1024
    assert estimate["serialization_subpeaks"] == {
        "metadata_encoding": 7_066_255,
        "bootstrap_weight_archive": 2_807_951,
        "metadata_chunked_write": 2_917_007,
    }
    assert estimate["comparison_peak_bytes"] == 7_066_255
    assert estimate["global_peak_phase"] == "phase5_serialization"
    for name, components in estimate["phase_peak_components"].items():
        assert sum(components.values()) == estimate["phase_peaks"][name]


def test_phase5_limit_accepts_exact_boundary_and_rejects_one_below() -> None:
    raw = _raw()
    estimate = runner.estimate_phase5_resources(raw)
    limits = {
        name: estimate[name] for name in runner.PHASE5_ABSOLUTE_LIMITS
    }
    runner.validate_phase5_budget(raw, False, limits=limits)
    limits["comparison_peak_bytes"] -= 1
    with pytest.raises(ValueError, match="comparison_peak_bytes"):
        runner.validate_phase5_budget(raw, False, limits=limits)


def test_float64_phase5_resource_terms_are_hand_calculated() -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["simulation"]["dtype"] = "float64"
    estimate = runner.estimate_phase5_resources(
        runner.inspect_comparison_config(config)
    )
    assert estimate["pair_point_host_bytes"] == 2 * 196 * 15 * 8 + 2 * 196 * 8 + 16
    assert estimate["pair_point_device_bytes"] == 2 * 196 * 15 * 8
    assert estimate["diagnostic_device_bytes"] == 2 * (11 * 8 + 3)
    assert estimate["diagnostic_host_bytes"] == 2 * (11 * 8 + 3)
    assert estimate["pair_retained_device_bytes"] == 1_281_359
    assert estimate["pair_transfer_host_bytes"] == 665_031
    assert estimate["static_pair_kernel_device_bytes"] == 5_954_055
    assert estimate["comparison_peak_bytes"] == 7_731_295


@pytest.mark.parametrize(("dtype", "item_bytes"), [("float32", 4), ("float64", 8)])
def test_t0_and_requested_summary_resource_terms(dtype, item_bytes) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["simulation"].update(steps=1, dtype=dtype)
    config["initial_condition"].update(q_min=-0.5, q_max=1.0, spacing=0.5)
    config["comparison"]["source_times"] = [0]
    raw = runner.inspect_comparison_config(config)
    estimate = runner.estimate_phase5_resources(raw)
    points = 16
    assert estimate["diagnostic_device_bytes"] == 11 * item_bytes + 3
    assert estimate["diagnostic_host_bytes"] == 11 * item_bytes + 3
    assert estimate["destination_validity_device_bytes"] == 0
    assert estimate["destination_validity_host_bytes"] == 0
    assert estimate["pair_point_device_bytes"] == points * 15 * item_bytes
    assert estimate["pair_point_host_bytes"] == (
        points * 15 * item_bytes + 2 * points * item_bytes + 8
    )


def test_chunk_and_requested_time_counts_enter_only_their_actual_lifetimes() -> None:
    one, _, _, _ = _small_pair_case((0,), "float32")
    three, _, _, _ = _small_pair_case((0, 1, 3), "float32")
    three["pair_chunk_size"] = 3
    three["pair_raw"]["chunk_size"] = 3
    one_estimate = runner.estimate_phase5_resources(one)
    three_estimate = runner.estimate_phase5_resources(three)

    assert one_estimate["source_times"] == 1
    assert three_estimate["source_times"] == 3
    assert three_estimate["diagnostic_device_bytes"] == 4 * 47
    assert three_estimate["destination_validity_device_bytes"] == 3
    assert three_estimate["pair_point_device_bytes"] == 3 * 16 * 15 * 4
    phase4_static = runner.phase4.estimate_pair_resources(three["pair_raw"])
    expected_kernel = (
        phase4_static["static_device_bytes"]
        - phase4_static["components"]["diagnostic_trajectory_bytes"]
        + three_estimate["pair_point_device_bytes"]
        + 4 * 47
        + 3
    )
    assert three_estimate["static_pair_kernel_device_bytes"] == expected_kernel


def test_phase5_lifetime_model_accepts_cap_below_unused_phase4_runner_buffers() -> None:
    raw = _raw()
    estimate = runner.estimate_phase5_resources(raw)
    phase4_estimate = runner.phase4.estimate_pair_resources(raw["pair_raw"])
    unused_phase4_runner = (
        phase4_estimate["components"]["diagnostic_row_host_bytes"]
        + phase4_estimate["components"]["serialization_live_peak_bytes"]
        + phase4_estimate["components"]["source_hash_buffer_bytes"]
    )
    cap = estimate["comparison_peak_bytes"]
    assert estimate["comparison_peak_bytes"] + unused_phase4_runner > cap
    limits = {name: estimate[name] for name in runner.PHASE5_ABSOLUTE_LIMITS}
    runner.validate_phase5_budget(raw, False, limits=limits)


def test_destination_validity_output_can_cross_the_phase5_cap() -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["simulation"]["steps"] = 33
    config["comparison"]["source_times"] = [0, 32]
    raw = runner.inspect_comparison_config(config)
    estimate = runner.estimate_phase5_resources(raw)
    assert estimate["destination_validity_device_bytes"] == 32
    assert estimate["global_peak_phase"] == "phase5_serialization"
    limits = {name: estimate[name] for name in runner.PHASE5_ABSOLUTE_LIMITS}
    limits["comparison_peak_bytes"] -= estimate["destination_validity_device_bytes"]
    with pytest.raises(ValueError, match="comparison_peak_bytes"):
        runner.validate_phase5_budget(raw, False, limits=limits)


def test_maximum_configured_source_time_counts_every_diagnostic_and_destination() -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["simulation"]["steps"] = 500
    config["comparison"]["source_times"] = [0, 499]
    estimate = runner.estimate_phase5_resources(
        runner.inspect_comparison_config(config)
    )
    assert estimate["diagnostic_device_bytes"] == 500 * 47
    assert estimate["diagnostic_host_bytes"] == 500 * 47
    assert estimate["destination_validity_device_bytes"] == 499
    assert estimate["destination_validity_host_bytes"] == 499


def test_phase_peaks_add_only_the_base_arrays_live_in_that_phase() -> None:
    raw = _raw()
    without = runner.estimate_phase5_resources(raw)
    base = {"abm_record_bytes": 11, "abm_state_working_bytes": 13}
    with_base = runner.estimate_phase5_resources(raw, base_resources=base)
    retained = sum(base.values())
    assert with_base["phase_peaks"]["configuration_and_scheme_validation"] == without[
        "phase_peaks"
    ]["configuration_and_scheme_validation"]
    assert with_base["phase_peaks"]["shape_lowering_and_compilation"] == without[
        "phase_peaks"
    ]["shape_lowering_and_compilation"]
    for name in set(with_base["phase_peaks"]) - {
        "configuration_and_scheme_validation", "shape_lowering_and_compilation"
    }:
        assert with_base["phase_peaks"][name] == without["phase_peaks"][name] + retained


def test_pair_source_records_label_p0_then_p1_without_off_by_one() -> None:
    raw, pair_grid, initial, learning, bundle = _compile_small_case()
    summaries, result, invoked = runner.run_pair_source_summaries(
        bundle,
        initial,
        pair_grid,
        learning,
        [0, 1],
        chunk_size=8,
        symmetry_tolerance=2e-6,
        diagnostic_tolerance=2e-6,
        max_elements=512,
        raw=raw,
    )

    expected_p0 = np.asarray(initial).sum(axis=(0, 2))
    expected_p1 = np.asarray(result.final_mass).sum(axis=(0, 2))
    np.testing.assert_allclose(summaries.focal_mass[0], expected_p0, atol=2e-7)
    np.testing.assert_allclose(summaries.focal_mass[1], expected_p1, atol=2e-7)
    assert not np.allclose(expected_p0, expected_p1)
    assert invoked == bundle.compile_signature


def test_pair_source_scan_with_only_p0_has_no_transport_or_destination_record() -> None:
    raw, pair_grid, initial, learning, bundle = _compile_small_case((0,))
    summaries, result, _ = runner.run_pair_source_summaries(
        bundle,
        initial,
        pair_grid,
        learning,
        [0],
        chunk_size=8,
        symmetry_tolerance=2e-6,
        diagnostic_tolerance=2e-6,
        max_elements=512,
        raw=raw,
    )
    np.testing.assert_array_equal(np.asarray(result.final_mass), np.asarray(initial))
    assert np.asarray(result.destinations_valid).shape == (0,)
    assert np.asarray(result.diagnostics.total_mass).shape == (1,)
    assert summaries.source_times.tolist() == [0]


def test_exact_analyzed_compiled_object_is_invoked_once_and_jitted_wrapper_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, grid, initial, learning, original = _compile_small_case()

    class SpyCompiled:
        def __init__(self, delegate):
            self.delegate = delegate
            self.analysis_calls = 0
            self.invocations = 0

        def memory_analysis(self):
            self.analysis_calls += 1
            return self.delegate.memory_analysis()

        def __call__(self, *args, **kwargs):
            self.invocations += 1
            return self.delegate(*args, **kwargs)

    spy = SpyCompiled(original.compiled_callable)
    abstract = jax.ShapeDtypeStruct(initial.shape, initial.dtype)
    slots = jnp.asarray(runner.source_slot_by_time(raw["source_times"]))
    bundle = runner._bundle_from_compiled(spy, raw, grid, learning, abstract, slots)
    pair_budget = runner.validate_phase5_pair_static_budget(raw, False)
    runner.validate_compiled_phase5_pair_budget(pair_budget, bundle, raw, False)

    monkeypatch.setattr(
        runner,
        "simulate_pair_source_summaries_jit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the original jitted wrapper was called after analysis")
        ),
    )
    runner.run_pair_source_summaries(
        bundle,
        initial,
        grid,
        learning,
        [0, 1],
        chunk_size=8,
        symmetry_tolerance=2e-6,
        diagnostic_tolerance=2e-6,
        max_elements=512,
        raw=raw,
    )

    assert spy.analysis_calls == 1
    assert spy.invocations == 1


def test_phase5_compiled_analysis_failure_is_fail_closed_with_only_explicit_override() -> None:
    raw, _, _, _, bundle = _compile_small_case()
    report = deepcopy(bundle.memory_report)
    report.update(
        available=False,
        analysis_status="unavailable",
        unavailable_reason="injected failure",
        compiled_device_requirement_bytes=None,
        compiled_host_requirement_bytes=None,
        compiled_plus_host_requirement_bytes=None,
    )
    unavailable = replace(bundle, memory_report=report)
    budget = runner.validate_phase5_pair_static_budget(raw, False)
    with pytest.raises(ValueError, match="exact shape-only compilation"):
        runner.validate_compiled_phase5_pair_budget(budget, unavailable, raw, False)
    overridden = runner.validate_compiled_phase5_pair_budget(
        runner.validate_phase5_pair_static_budget(raw, True), unavailable, raw, True
    )
    assert overridden["compiled_violations_overridden"] == [
        "compiled_analysis_unavailable"
    ]
    assert overridden["compiled_analysis"]["validation_status"] == "unavailable"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dtype", "float64"),
        ("pair_shape", [2, 15, 15]),
        ("steps", 2),
        ("chunk_size", 9),
        ("requested_source_times", [0]),
        ("validation_max_elements", 511),
        ("backend", "not-the-current-backend"),
        ("devices", [{"id": "99", "platform": "cpu", "device_kind": "fake"}]),
        ("jax_enable_x64", not bool(jax.config.read("jax_enable_x64"))),
    ],
)
def test_every_execution_signature_mismatch_fails_before_compiled_invocation(
    field, replacement
) -> None:
    raw, grid, initial, learning, bundle = _compile_small_case()

    class ForbiddenCompiled:
        def __call__(self, *args, **kwargs):
            raise AssertionError("mismatched executable was invoked")

    signature = deepcopy(bundle.compile_signature)
    signature[field] = replacement
    report = deepcopy(bundle.memory_report)
    report["executable_signature"] = signature
    mismatched = replace(
        bundle,
        compiled_callable=ForbiddenCompiled(),
        memory_report=report,
        compile_signature=signature,
    )
    with pytest.raises(ValueError, match="compiled"):
        runner.run_pair_source_summaries(
            mismatched,
            initial,
            grid,
            learning,
            [0, 1],
            chunk_size=8,
            symmetry_tolerance=2e-6,
            diagnostic_tolerance=2e-6,
            max_elements=512,
            raw=raw,
        )


@pytest.mark.parametrize(
    ("dtype", "atol"), [("float32", 3e-6), ("float64", 1e-12)]
)
def test_combined_scan_matches_repeated_one_step_reference_at_all_requested_times(
    dtype, atol
) -> None:
    if dtype == "float64" and not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 parity requires JAX_ENABLE_X64=1 before import")
    source_times = (0, 2, 3)
    raw, grid, initial, learning, bundle = _compile_small_case(source_times, dtype)
    _, result, _ = runner.run_pair_source_summaries(
        bundle,
        initial,
        grid,
        learning,
        list(source_times),
        chunk_size=8,
        symmetry_tolerance=raw["symmetry_tolerance"],
        diagnostic_tolerance=raw["diagnostic_tolerance"],
        max_elements=512,
        raw=raw,
    )

    mass = initial
    independent = []
    for time_index in range(source_times[-1] + 1):
        if time_index in source_times:
            independent.append(jax.device_get(pair_point_sufficient_jax(mass, grid, learning.tau)))
        if time_index < source_times[-1]:
            mass = checked_pair_mass_step(
                mass,
                grid,
                learning.alpha,
                learning.tau,
                chunk_size=8,
                symmetry_tolerance=2e-6 if dtype == "float32" else 1e-12,
                max_elements=512,
            ).mass

    for field in result.source_summaries._fields:
        expected = np.stack([np.asarray(getattr(point, field)) for point in independent])
        np.testing.assert_allclose(
            np.asarray(getattr(result.source_summaries, field)),
            expected,
            rtol=0,
            atol=atol,
        )
    np.testing.assert_allclose(
        np.asarray(result.final_mass), np.asarray(mass), rtol=0, atol=atol
    )


def test_resource_rejection_precedes_grid_compilation_simulation_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner.baseline, "load_config", lambda path: deepcopy(config))
    original = runner.validate_phase5_budget
    monkeypatch.setattr(
        runner,
        "validate_phase5_budget",
        lambda raw, allow, **kwargs: original(
            raw,
            allow,
            base_resources=kwargs.get("base_resources"),
            limits={name: 0 for name in runner.PHASE5_ABSOLUTE_LIMITS},
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric or output work entered before Phase 5 rejection")

    for name in (
        "QGrid",
        "build_jax_pair_grid",
        "simulate_instrumented_batch_jit",
        "run_pair_source_summaries",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="before grid, compilation, simulation"):
        runner.main()


def test_histogram_sampling_guard_precedes_grid_and_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["initial_condition"]["samples_per_grid_cell"] = 20_000
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner.baseline, "load_config", lambda path: deepcopy(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("grid or compilation entered before histogram rejection")

    monkeypatch.setattr(runner, "QGrid", forbidden)
    monkeypatch.setattr(runner, "build_jax_pair_grid", forbidden)
    monkeypatch.setattr(runner.phase4, "analyze_compiled_pair_memory", forbidden)

    with pytest.raises(ValueError, match="histogram before grid construction"):
        runner.main()


def test_exact_executed_scan_memory_rejection_precedes_scientific_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner.baseline, "load_config", lambda path: deepcopy(config))

    def excessive(raw, grid, learning):
        dtype = jnp.float32 if raw["abm"].dtype == "float32" else jnp.float64
        abstract = jax.ShapeDtypeStruct(
            (2, raw["pair_raw"]["agent_grid_points"], raw["pair_raw"]["agent_grid_points"]),
            dtype,
        )
        slots = jnp.asarray(runner.source_slot_by_time(raw["source_times"]))
        signature = runner.phase5_executable_signature(
            raw,
            mass_shape=abstract.shape,
            mass_dtype=abstract.dtype,
            grid=grid,
            alpha=learning.alpha,
            tau=learning.tau,
            slots=slots,
            max_elements=raw["pair_raw"]["state_expanded_cells"],
        )
        report = {
            "available": True,
            "compiled_device_requirement_bytes": runner.estimate_phase5_resources(raw)[
                "static_pair_kernel_device_bytes"
            ] + 1,
            "compiled_host_requirement_bytes": 0,
            "compiled_plus_host_requirement_bytes": 0,
            "executable_id": runner.PHASE5_EXECUTABLE_ID,
            "executable_signature": signature,
        }
        return runner.CompiledPairExecutableBundle(
            compiled_callable=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("rejected executable was invoked")
            ),
            memory_report=report,
            compile_signature=signature,
            abstract_arguments={
                "pair_mass": {"shape": list(abstract.shape), "dtype": np.dtype(abstract.dtype).name},
                "grid": runner._grid_argument_spec(grid),
                "source_slots": {"shape": list(slots.shape), "dtype": np.dtype(slots.dtype).name},
                "dynamic_scalars": signature["dynamic_scalar_arguments"],
            },
            static_values={"steps": 1, "summary_count": 2, "chunk_size": 4096},
            runtime_environment={
                key: signature[key]
                for key in ("backend", "platforms", "devices", "jax_enable_x64")
            },
        )

    monkeypatch.setattr(runner, "analyze_compiled_phase5_pair_memory", excessive)

    def forbidden(*args, **kwargs):
        raise AssertionError("scientific allocation entered after compiled rejection")

    monkeypatch.setattr(runner.phase4, "analyze_compiled_pair_memory", forbidden)
    for name in (
        "seeded_legacy_histogram",
        "complete_graph",
        "initialize_grid_matched_batch",
        "simulate_instrumented_batch_jit",
        "ordered_pair_mass_jax",
        "run_pair_source_summaries",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    with pytest.raises(ValueError, match="exact shape-only compilation"):
        runner.main()


def _assert_bin_validation_precedes_numeric_work(config, match, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner.baseline, "load_config", lambda path: deepcopy(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric work entered before scientific bin validation")

    for name in (
        "QGrid",
        "analyze_compiled_phase5_pair_memory",
        "seeded_legacy_histogram",
        "simulate_instrumented_batch_jit",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    with pytest.raises(ValueError, match=match):
        runner.main()


def test_float32_edge_collapse_is_rejected_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["bin_schemes"][1]["q_c_edges"] = [
        -0.1, 0.2, 0.55, 0.55000001, 0.85, 1.2
    ]
    _assert_bin_validation_precedes_numeric_work(config, "collapsed", monkeypatch)


def test_invalid_nesting_is_rejected_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["bin_schemes"][1]["q_c_edges"] = [-0.1, 0.2, 0.6, 0.85, 1.2]
    _assert_bin_validation_precedes_numeric_work(config, "nested", monkeypatch)


def test_invalid_anchor_is_rejected_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["anchors"]["points"] = [[2.0, 0.2]]
    _assert_bin_validation_precedes_numeric_work(config, "outside", monkeypatch)


def test_fourteen_nested_levels_use_sequential_not_total_sufficient_peak() -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    interior = [-0.01 + 0.09 * index for index in range(13)]
    config["bin_schemes"] = [
        {
            "name": f"level_{bins:02d}",
            "q_c_edges": [-0.1, *interior[: bins - 1], 1.2],
            "q_d_edges": [-0.1, *interior[: bins - 1], 1.2],
        }
        for bins in range(1, 15)
    ]
    raw = runner.inspect_comparison_config(config)
    schemes, _, _, budget = runner.phase3b.construct_guarded_schemes(
        raw["abm"], config, False
    )
    assert len(schemes) == 14
    estimate = budget["estimates"]
    simultaneous = estimate["total_per_run_strata"] * 88
    sequential = estimate["components"]["peak_sequential_sufficient_bytes"]
    assert simultaneous - sequential >= 1_349_538
    phase5 = runner.estimate_phase5_resources(raw)
    pair_bytes = [cell * 56 + (cell // 2) * 8 for cell in phase5["scheme_cells"]]
    assert phase5["pair_binned_bytes"] == pair_bytes[-1] + max(pair_bytes[:-1])


def _diagnostics(*, weight_error=0.0, minimum_variance=0.0):
    return JAXPairDiagnostics(
        total_mass=jnp.array([1.0, 1.0]),
        state_masses=jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        mean_q=jnp.zeros((2, 2)),
        mean_action_probability=jnp.full((2, 2), 0.5),
        symmetry_error=jnp.zeros(2),
        minimum_mass=jnp.zeros(2),
        finite=jnp.ones(2, dtype=bool),
        nonnegative=jnp.ones(2, dtype=bool),
        conditional_weight_error=jnp.full(2, weight_error),
        minimum_conditional_variance=jnp.full(2, minimum_variance),
        conditional_moments_valid=jnp.ones(2, dtype=bool),
    )


def test_pair_diagnostic_tolerance_controls_weight_and_variance_checks() -> None:
    destinations = jnp.array([True])
    validate_pair_source_diagnostics(
        _diagnostics(weight_error=5e-7, minimum_variance=-5e-7),
        destinations,
        diagnostic_tolerance=1e-6,
        symmetry_tolerance=1e-6,
    )
    with pytest.raises(ValueError, match="conditional-weight"):
        validate_pair_source_diagnostics(
            _diagnostics(weight_error=5e-7),
            destinations,
            diagnostic_tolerance=1e-7,
            symmetry_tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="conditional variance"):
        validate_pair_source_diagnostics(
            _diagnostics(minimum_variance=-5e-7),
            destinations,
            diagnostic_tolerance=1e-7,
            symmetry_tolerance=1e-6,
        )


def test_runner_rejects_off_by_one_source_time_configuration() -> None:
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["comparison"]["source_times"] = [0, config["simulation"]["steps"]]
    with pytest.raises(ValueError, match="below steps"):
        runner.inspect_comparison_config(config)


def _maximum_name_config_and_rows():
    config = runner.baseline.load_config(runner.DEFAULT_CONFIG)
    config["simulation"]["steps"] = 500
    config["simulation"].update(num_agents=128, num_runs=32)
    config["output"]["run_name"] = "R" * runner.phase4.MAX_RUN_NAME_LENGTH
    config["comparison"]["source_times"] = [499]
    config["bin_schemes"] = [
        {
            "name": "S",
            "q_c_edges": [-0.1, 0.20000000000000004, 1.2],
            "q_d_edges": [-0.1, 0.20000000000000004, 1.2],
        }
    ]
    config["anchors"]["points"] = [[1.2, 1.2]]
    config["bootstrap"]["replicates"] = 100_000
    compact = lambda: json.dumps(
        config, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    remaining = runner.phase4.MAX_NORMALIZED_CONFIGURATION_JSON_CHARS - len(compact())
    assert remaining > 0
    config["bin_schemes"][0]["name"] += "S" * remaining
    assert len(compact()) == runner.phase4.MAX_NORMALIZED_CONFIGURATION_JSON_CHARS
    runner.inspect_comparison_config(config)

    bins = QBinSpec(
        config["bin_schemes"][0]["q_c_edges"],
        config["bin_schemes"][0]["q_d_edges"],
    )
    scheme = NamedBinScheme(config["bin_schemes"][0]["name"], bins)
    action_shape = (1, 2, 2, 2)
    focal_shape = (1, 2, 2)
    float_values = np.full(action_shape, np.finfo(np.float64).max)
    float_values[..., 0] *= -1
    float_values[0, 0, 0, 0] = np.nan
    boolean_fields = {
        "has_abm_observations",
        "pair_has_focal_mass",
        "pair_has_selected_mass",
        "pair_valid",
        "abm_reconstruction_defined",
        "hybrid_valid",
        "sparse",
    }
    integer_fields = {"abm_count", "contributing_runs"}
    values = {}
    float_index = 0
    for name in FourWayComparison.__dataclass_fields__:
        shape = focal_shape if name == "pair_focal_mass" else action_shape
        if name in boolean_fields:
            values[name] = np.ones(shape, dtype=bool)
        elif name in integer_fields:
            maximum = 128 * 32 if name == "abm_count" else 32
            values[name] = np.full(shape, maximum, dtype=np.int64)
        else:
            values[name] = np.broadcast_to(float_values, action_shape).copy()
            selected_value = (
                np.nan
                if "ratio" in name
                else (-np.finfo(np.float64).max if float_index % 2 else np.finfo(np.float64).max)
            )
            values[name][0, 1, 1, 1] = selected_value
            float_index += 1
            if shape == focal_shape:
                values[name] = values[name][..., 0]
    comparison = FourWayComparison(**values)
    lower = {
        name: np.full(action_shape, -np.finfo(np.float64).max)
        for name in runner.COMPARISON_BOOTSTRAP_ESTIMANDS
    }
    upper = {
        name: np.full(action_shape, np.finfo(np.float64).max)
        for name in runner.COMPARISON_BOOTSTRAP_ESTIMANDS
    }
    valid = {}
    invalid = {}
    for index, name in enumerate(runner.COMPARISON_BOOTSTRAP_ESTIMANDS):
        valid_count = 100_000 if index % 2 == 0 else 0
        valid[name] = np.full(action_shape, valid_count, dtype=np.int64)
        invalid[name] = np.full(action_shape, 100_000 - valid_count, dtype=np.int64)
    interval_valid = {
        name: np.ones(action_shape, dtype=bool)
        for name in runner.COMPARISON_BOOTSTRAP_ESTIMANDS
    }
    intervals = ComparisonBootstrapSummary(
        lower=lower,
        upper=upper,
        valid_replicates=valid,
        invalid_replicates=invalid,
        interval_valid=interval_valid,
        replicates=100_000,
        confidence_level=np.nextafter(1.0, 0.0),
    )
    results = [(scheme, comparison, intervals)]
    comparison_row = None
    for row in runner.iter_comparison_rows(results, [499], np.dtype(np.float64)):
        comparison_row = row
    anchor_row = None
    for row in runner.iter_anchor_rows(
        results, [(1.2, 1.2)], [499], np.dtype(np.float64)
    ):
        anchor_row = row
    assert comparison_row is not None and anchor_row is not None
    return comparison_row, anchor_row


@pytest.mark.parametrize("kind", ["comparison", "anchor"])
def test_real_maximum_schema_row_fits_exact_live_and_csv_bounds(
    kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison_row, anchor_row = _maximum_name_config_and_rows()
    row = comparison_row if kind == "comparison" else anchor_row
    assert len(comparison_row) == 91
    live = runner.deep_size_bytes(row)

    independent_sink = io.StringIO(newline="")
    independent_writer = csv.DictWriter(independent_sink, fieldnames=list(row))
    independent_writer.writeheader()
    header_chars = len(independent_sink.getvalue().encode("ascii"))
    independent_sink.seek(0)
    independent_sink.truncate(0)
    independent_writer.writerow(row)
    row_chars = len(independent_sink.getvalue().encode("ascii"))
    independent_csv_chars = max(header_chars, row_chars)

    monkeypatch.setattr(runner, "PHASE5_MAX_LIVE_ROW_BYTES", live)
    generous = runner.validate_streamed_rows(lambda: iter((row,)), 1)
    csv_chars = generous["maximum_csv_write_chars"]
    assert csv_chars == independent_csv_chars
    assert live <= 16 * 1024
    assert csv_chars <= 8 * 1024
    estimate = runner.estimate_phase5_resources(_raw())
    assert estimate["csv_writer_peak_bytes"] == 64 * 1024 + 8 * 1024 + 9 * 8 * 1024

    monkeypatch.setattr(runner, "PHASE5_MAX_CSV_WRITE_CHARS", csv_chars)
    exact = runner.validate_streamed_rows(lambda: iter((row,)), 1)
    assert exact["maximum_live_python_row_bytes"] == live
    assert exact["maximum_csv_write_chars"] == csv_chars

    monkeypatch.setattr(runner, "PHASE5_MAX_LIVE_ROW_BYTES", live - 1)
    with pytest.raises(RuntimeError, match="live-object bound"):
        runner.validate_streamed_rows(lambda: iter((row,)), 1)
    monkeypatch.setattr(runner, "PHASE5_MAX_LIVE_ROW_BYTES", live)
    monkeypatch.setattr(runner, "PHASE5_MAX_CSV_WRITE_CHARS", csv_chars - 1)
    with pytest.raises(RuntimeError, match="character record allowance"):
        runner.validate_streamed_rows(lambda: iter((row,)), 1)


def test_runner_executes_one_abm_and_one_pair_trajectory_for_all_schemes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = {"abm": 0, "pair": 0}
    original_abm = runner.simulate_instrumented_batch_jit
    original_pair = runner.run_pair_source_summaries

    def counted_abm(*args, **kwargs):
        calls["abm"] += 1
        return original_abm(*args, **kwargs)

    def counted_pair(*args, **kwargs):
        calls["pair"] += 1
        return original_pair(*args, **kwargs)

    monkeypatch.setattr(runner, "simulate_instrumented_batch_jit", counted_abm)
    monkeypatch.setattr(runner, "run_pair_source_summaries", counted_pair)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )

    runner.main()

    assert calls == {"abm": 1, "pair": 1}
    run_directories = list((tmp_path / "outputs" / "variance_comparison").iterdir())
    assert len(run_directories) == 1
    output = run_directories[0]
    with (output / "variance_comparison.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 80
    assert {int(row["source_time_t"]) for row in rows} == {0, 1}
    assert {row["scheme"] for row in rows} == {"coarse", "fine"}
    assert (output / "anchor_bin_refinement.csv").exists()
    assert (output / "bootstrap_run_weights.npz").exists()
    assert (output / "metadata.json").exists()
    assert not any("density" in path.name for path in output.iterdir())
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["pair_execution"]["signatures_match"]
    assert metadata["pair_execution"]["diagnostic_tolerance_used"] == 2e-6
    assert metadata["resource_budget"]["phase5_comparison"]["estimates"][
        "global_peak_phase"
    ] == "phase5_serialization"
    assert "phase5_pair_kernel" in metadata["resource_budget"]
    assert "phase4_pair" not in metadata["resource_budget"]


def test_row_prevalidation_failure_leaves_no_output_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "PHASE5_MAX_LIVE_ROW_BYTES", 0)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )

    with pytest.raises(RuntimeError, match="live-object bound"):
        runner.main()

    assert not (tmp_path / "outputs").exists()

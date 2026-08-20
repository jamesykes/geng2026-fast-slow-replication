from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments import run_pair_separable_benchmark as runner
import chu_pair.pair_density.separable_resources as resources
from chu_pair.pair_density import (
    estimate_flat_resources,
    estimate_separable_resources,
    discover_nvidia_device_capacity,
    full_grid_feasibility,
    make_compiled_executable_bundle,
    make_device_capacity_observation,
    production_capacity_preflight,
)


def _raw() -> dict:
    return runner.inspect_raw_config(runner.load_config(runner.DEFAULT_CONFIG))


def test_raw_preflight_counts_shapes_without_constructing_grid(monkeypatch) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["cases"][0].update(q_min=-1.0, q_max=1.0, spacing=2 / 130)
    monkeypatch.setattr(runner, "QGrid", lambda *args, **kwargs: pytest.fail("QGrid entered"))

    raw = runner.inspect_raw_config(config)

    assert raw["cases"][0]["grid_size"] == 131
    assert raw["cases"][0]["agent_grid_points"] == 17_161
    assert raw["cases"][0]["state_expanded_cells"] == 588_999_842
    with pytest.raises(ValueError, match="before grid construction"):
        runner.validate_static_budget(raw, False)


def test_configuration_cannot_define_or_raise_caps() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["safety"] = {"grid_size": 131}
    with pytest.raises(ValueError, match="sections"):
        runner.inspect_raw_config(config)


def test_timing_sample_preflight_counts_case_and_reduction_samples() -> None:
    raw = _raw()
    repetitions = raw["normalized_configuration"]["benchmark"]["repetitions"]
    assert raw["timing_record_count"] == 4 * len(raw["cases"])
    assert raw["timing_sample_count"] == (
        raw["timing_record_count"] + 2
    ) * repetitions
    assert raw["timing_sample_count"] <= runner.MAX_TIMING_SAMPLES


def test_controlled_benchmark_rejects_nonuniform_initial_state_law() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["model"]["state_probabilities"] = [0.4, 0.6]
    with pytest.raises(ValueError, match="uniform state mass"):
        runner.inspect_raw_config(config)


def test_explicit_override_records_static_violations_with_injected_small_limits() -> None:
    raw = _raw()
    limits = {
        "warmup": 0,
        "repetitions": 1,
        "grid_size": 2,
        "agent_grid_points": 4,
        "state_expanded_cells": 32,
        "density_bytes": 1,
        "steps": 0,
        "summary_count": 1,
        "block_size": 1,
        "timing_records": 1,
        "timing_samples": 1,
        "total_density_bytes": 1,
        "static_device_bytes": 1,
        "static_host_bytes": 1,
        "static_combined_bytes": 1,
    }
    with pytest.raises(ValueError, match="before grid construction"):
        runner.validate_static_budget(raw, False, limits=limits)
    budget = runner.validate_static_budget(raw, True, limits=limits)
    assert budget["violations"]
    assert budget["violations_overridden"] == budget["violations"]


def test_static_separable_formula_reconstructs_every_named_term() -> None:
    estimate = estimate_separable_resources(
        grid_size=5,
        dtype_bytes=4,
        steps=2,
        summary_count=3,
        row_block_size=7,
        column_block_size=6,
        return_final_density=False,
    )
    G = 5
    M = 25
    D = 1_250
    b = 4
    density = D * b
    summaries = 15 * 3 * M * b
    diagnostics = 3 * (11 * b + 3) + 2
    grid_device = G * b + 2 * M * b + 2 * M * 4
    initialization = M * b + 2 * b
    tables = M * (20 * b + 40)
    tile = 2 * (7 * 6) * b + 7 * 6 + 2 * 7 * 4 + 7 + 2 * 6 * 4 + 6
    named = 2 * density + grid_device + initialization + tables + tile + summaries + diagnostics

    assert estimate["one_density_bytes"] == density
    assert estimate["source_summary_bytes"] == summaries
    assert estimate["diagnostic_device_bytes"] == diagnostics
    assert estimate["grid_device_bytes"] == grid_device
    assert estimate["initialization_device_bytes"] == initialization
    assert estimate["policy_moment_destination_bytes"] == tables
    assert estimate["block_workspace_bytes"] == tile
    assert estimate["transport_named_bytes"] == named
    assert estimate["scatter_lowering_allowance_bytes"] == 2 * density
    assert estimate["static_device_fixed_bytes"] == 4 * 1024
    assert estimate["static_device_bytes"] == named + 2 * density + 4 * 1024
    modeled_host = estimate["static_host_bytes"]
    assert estimate["modeled_coexisting_host_numerical_bytes"] == modeled_host
    assert estimate["heuristic_host_staging_reserve_bytes"] == 2 * density
    assert estimate["host_planning_threshold_bytes"] == modeled_host + 2 * density


def test_flat_formula_retains_full_branch_workspace_and_density_allowance() -> None:
    estimate = estimate_flat_resources(
        grid_size=5,
        dtype_bytes=4,
        steps=2,
        summary_count=3,
        chunk_size=1_250,
        return_final_density=False,
    )
    assert estimate["chunk_workspace_bytes"] == 1_250 * (17 * 4 + 96)
    assert estimate["static_device_bytes"] > 8 * 5_000


def _production_bundle(projection: dict, *, signature_updates: dict | None = None):
    called = []

    device = {
        "backend": "gpu",
        "platform": "gpu",
        "visible_device_index": 0,
        "visible_device_count": 1,
        "id": "0",
        "device_kind": "NVIDIA Test GPU",
        "process_index": 0,
        "local_hardware_id": 0,
        "uuid": "GPU-test",
    }
    environment = {
        "backend": "gpu",
        "platform": "gpu",
        "execution_device": device,
        "visible_devices": [device],
        "jax_enable_x64": False,
        "jax_version": "test-jax",
        "jaxlib_version": "test-jaxlib",
    }
    abstract = {
        "histogram_mass": {"shape": [17_161], "dtype": "float32", "weak_type": False},
        "state_probabilities": {"shape": [2], "dtype": "float32", "weak_type": False},
        "grid": [{"shape": [131], "dtype": "float32", "weak_type": False}],
        "alpha": {"shape": [], "dtype": "float32", "weak_type": True},
        "tau": {"shape": [], "dtype": "float32", "weak_type": True},
        "source_slots": {"shape": [101], "dtype": "int32", "weak_type": False},
        "diagnostic_tolerance": {"shape": [], "dtype": "float32", "weak_type": True},
    }
    static = {
        "kernel": "separable",
        "output_mode": "bounded_from_histogram",
        "steps": 100,
        "summary_count": 11,
        "chunk_size": 1,
        "row_block_size": 81,
        "column_block_size": 100,
    }
    signature = {
        "executable_id": "pair-source-from-histogram:bounded_from_histogram:v1",
        "kernel": "separable",
        "output_mode": "bounded_from_histogram",
        "backend": "gpu",
        "platform": "gpu",
        "execution_device": device,
        "visible_devices": [device],
        "state_expanded_cells": projection["state_expanded_cells"],
        "grid_size": 131,
        "agent_grid_points": 17_161,
        "dtype": "float32",
        "histogram_argument": abstract["histogram_mass"],
        "state_probability_argument": abstract["state_probabilities"],
        "grid_arguments": abstract["grid"],
        "alpha": 0.4,
        "tau": 1.3,
        "dynamic_scalar_arguments": {
            "alpha": abstract["alpha"],
            "tau": abstract["tau"],
            "diagnostic_tolerance": abstract["diagnostic_tolerance"],
        },
        "steps": 100,
        "summary_count": 11,
        "requested_source_times": list(range(11)),
        "source_slots": list(range(11)) + [-1] * 90,
        "source_slot_argument": abstract["source_slots"],
        "row_block_size": 81,
        "column_block_size": 100,
        "chunk_size": 1,
        "diagnostic_tolerance": 1e-6,
        "symmetry_tolerance": 1e-6,
        "jax_enable_x64": False,
        "jax_version": "test-jax",
        "jaxlib_version": "test-jaxlib",
        "contract_abstract_arguments": abstract,
        "contract_static_values": static,
        "contract_runtime_environment": environment,
    }
    signature.update(signature_updates or {})
    report = {
        "available": True,
        "analysis_status": "complete",
        "argument_bytes": 10,
        "output_bytes": 20,
        "temporary_bytes": 30,
        "alias_bytes": 5,
        "host_argument_bytes": 0,
        "host_output_bytes": 0,
        "host_temporary_bytes": 0,
        "host_alias_bytes": 0,
        "compiled_device_requirement_bytes": 55,
        "compiled_host_requirement_bytes": 0,
        "static_host_allowance_bytes": 0,
        "compiled_plus_host_requirement_bytes": 55,
        "backend": "gpu",
    }
    statistics = SimpleNamespace(
        argument_size_in_bytes=10,
        output_size_in_bytes=20,
        temp_size_in_bytes=30,
        alias_size_in_bytes=5,
        host_argument_size_in_bytes=0,
        host_output_size_in_bytes=0,
        host_temp_size_in_bytes=0,
        host_alias_size_in_bytes=0,
    )

    class Compiled:
        def __call__(self, *args, **kwargs):
            called.append((args, kwargs))
            return None

        def memory_analysis(self):
            return statistics

    compiled = Compiled()
    bundle = make_compiled_executable_bundle(
        compiled_callable=compiled,
        memory_report=report,
        compile_signature=signature,
        abstract_arguments=abstract,
        static_values=static,
        runtime_environment=environment,
    )
    return bundle, called


def _capacity(bundle, *, free: int, observed_at: datetime | None = None, **updates):
    values = {
        "source": "injected-test-provider",
        "observed_at_utc": (observed_at or datetime.now(timezone.utc)).isoformat(),
        "backend": "gpu",
        "platform": "gpu",
        "visible_device_index": 0,
        "execution_device_identity": bundle.compile_signature["execution_device"],
        "stable_device_identity": {"uuid": "GPU-test"},
        "device_name": "NVIDIA Test GPU",
        "total_physical_bytes": free + 1024,
        "free_bytes": free,
        "used_bytes": 1024,
    }
    values.update(updates)
    return make_device_capacity_observation(**values)


def test_full_grid_projection_is_exact_allocation_free_and_capacity_fails_closed() -> None:
    projection = full_grid_feasibility(
        dtype_bytes=4,
        representative_steps=100,
        representative_summary_count=11,
        row_block_size=81,
        column_block_size=100,
        validated_compiled_bytes_per_density_byte=5.0,
        safety_margin_fraction=0.25,
    )
    assert projection["agent_grid_points"] == 17_161
    assert projection["state_expanded_cells"] == 588_999_842
    assert projection["one_density_bytes"] == 2_355_999_368
    assert projection["ratio_projected_compiled_bytes"] == 11_779_996_840
    assert projection["minimum_device_memory_bytes"] >= 14_724_996_050
    assert not projection["execution_permitted"]
    assert projection["compiled_projection_kind"] == "empirical small-CPU planning projection"
    assert not projection["compiled_projection_is_formal_bound"]
    assert projection["host_requirement_bytes"] == (
        projection["modeled_coexisting_host_numerical_bytes"]
        + projection["heuristic_host_staging_reserve_bytes"]
    )

    bundle, _ = _production_bundle(projection)
    enough = projection["minimum_device_memory_bytes"] + 1
    capacity = _capacity(bundle, free=enough)
    passed = production_capacity_preflight(
        feasibility=projection,
        bundle=bundle,
        capacity_observation=capacity,
        allow_expensive=False,
    )
    assert passed["passed"]
    assert passed["verified_usable_device_bytes"] == enough

    cpu_bundle, _ = _production_bundle(projection, signature_updates={"backend": "cpu", "platform": "cpu"})
    with pytest.raises(ValueError, match="not overridable"):
        production_capacity_preflight(
            feasibility=projection,
            bundle=cpu_bundle,
            capacity_observation=capacity,
            allow_expensive=True,
        )

    synthetic = make_compiled_executable_bundle(
        compiled_callable=lambda *args, **kwargs: None,
        memory_report=bundle.memory_report,
        compile_signature=bundle.compile_signature,
        abstract_arguments={},
        static_values={},
        runtime_environment={},
    )
    with pytest.raises(ValueError, match="not overridable"):
        production_capacity_preflight(
            feasibility=projection,
            bundle=synthetic,
            capacity_observation=capacity,
            allow_expensive=True,
        )
    with pytest.raises(ValueError, match="not overridable"):
        production_capacity_preflight(
            feasibility=projection,
            bundle={"available": True},
            capacity_observation=capacity,
            allow_expensive=True,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"steps": 99},
        {"row_block_size": 80},
        {"column_block_size": 99},
        {"dtype": "float64"},
        {"state_expanded_cells": 123},
        {"backend": "cpu"},
    ],
)
def test_production_bundle_mutations_are_non_overridable(mutation) -> None:
    projection = full_grid_feasibility(
        dtype_bytes=4,
        representative_steps=100,
        representative_summary_count=11,
        row_block_size=81,
        column_block_size=100,
        validated_compiled_bytes_per_density_byte=5.0,
        safety_margin_fraction=0.25,
    )
    bundle, _ = _production_bundle(projection)
    capacity = _capacity(bundle, free=projection["minimum_device_memory_bytes"] + 1)
    mutated = replace(bundle, compile_signature={**bundle.compile_signature, **mutation})
    with pytest.raises(ValueError, match="not overridable"):
        production_capacity_preflight(
            feasibility=projection,
            bundle=mutated,
            capacity_observation=capacity,
            allow_expensive=True,
        )


def test_capacity_evidence_is_fresh_matched_usable_and_non_overridable(monkeypatch) -> None:
    projection = full_grid_feasibility(
        dtype_bytes=4,
        representative_steps=100,
        representative_summary_count=11,
        row_block_size=81,
        column_block_size=100,
        validated_compiled_bytes_per_density_byte=5.0,
        safety_margin_fraction=0.25,
    )
    bundle, _ = _production_bundle(projection)
    required = projection["minimum_device_memory_bytes"]
    now = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(resources, "_utc_now", lambda: now)
    cases = [
        _capacity(
            bundle,
            free=required - 1,
            observed_at=now,
            total_physical_bytes=2 * required,
            used_bytes=required + 1,
        ),
        _capacity(bundle, free=required + 1, observed_at=now - timedelta(seconds=61)),
        replace(
            _capacity(bundle, free=required + 1, observed_at=now),
            execution_device_identity={**bundle.compile_signature["execution_device"], "id": "1"},
        ),
        replace(
            _capacity(bundle, free=required + 1, observed_at=now),
            free_bytes=None,
            usable_device_bytes=None,
        ),
        replace(
            _capacity(bundle, free=required + 1, observed_at=now),
            stable_device_identity={},
        ),
        _capacity(bundle, free=required + 1, observed_at=now - timedelta(days=7)),
        _capacity(bundle, free=required + 1, observed_at=now + timedelta(seconds=2)),
        replace(
            _capacity(bundle, free=required + 1, observed_at=now),
            observed_at_utc="2026-08-20T12:00:00",
        ),
        replace(
            _capacity(bundle, free=required + 1, observed_at=now),
            observed_at_utc="",
        ),
    ]
    for observation in cases:
        with pytest.raises(ValueError, match="not overridable"):
            production_capacity_preflight(
                feasibility=projection,
                bundle=bundle,
                capacity_observation=observation,
                allow_expensive=True,
            )

    with pytest.raises(TypeError):
        production_capacity_preflight(
            feasibility=projection,
            bundle=bundle,
            capacity_observation=_capacity(
                bundle, free=required + 1, observed_at=now - timedelta(days=7)
            ),
            allow_expensive=True,
            maximum_evidence_age_seconds=10**9,
        )

    allocator_limited = _capacity(
        bundle,
        free=required + 10_000,
        observed_at=now,
        allocator_reserved_bytes=20_000,
        allocator_available_bytes=required - 1,
    )
    assert allocator_limited.usable_device_bytes == required - 1
    with pytest.raises(ValueError, match="not overridable"):
        production_capacity_preflight(
            feasibility=projection,
            bundle=bundle,
            capacity_observation=allocator_limited,
            allow_expensive=True,
        )


def test_nvidia_capacity_provider_matches_stable_identity_and_fails_bounded() -> None:
    identity = {
        "backend": "gpu",
        "platform": "gpu",
        "visible_device_index": 0,
        "visible_device_count": 1,
        "device_kind": "NVIDIA Test GPU",
        "uuid": "GPU-test",
    }

    def successful(*args, **kwargs):
        return SimpleNamespace(stdout="0, GPU-test, 0000:01:00.0, Test GPU, 100, 75, 25\n")

    observation = discover_nvidia_device_capacity(
        identity, command_runner=successful, preallocate_setting="false"
    )
    assert observation.available
    assert observation.total_physical_bytes == 100 * 1024**2
    assert observation.free_bytes == observation.usable_device_bytes == 75 * 1024**2
    assert observation.stable_device_identity["uuid"] == "GPU-test"
    ambiguous_allocator = discover_nvidia_device_capacity(
        identity,
        command_runner=successful,
        preallocate_setting="true",
        memory_fraction_setting=None,
    )
    assert not ambiguous_allocator.available
    assert "allocator" in ambiguous_allocator.unavailable_reason

    def failed(*args, **kwargs):
        raise OSError("nvidia-smi unavailable")

    unavailable = discover_nvidia_device_capacity(
        identity, command_runner=failed, preallocate_setting="false"
    )
    assert not unavailable.available
    assert "OSError" in unavailable.unavailable_reason
    wrong = discover_nvidia_device_capacity(
        {**identity, "uuid": "GPU-other"},
        command_runner=successful,
        preallocate_setting="false",
    )
    assert not wrong.available


def test_nvidia_mapping_uses_only_stable_cuda_to_nvml_identity() -> None:
    output = (
        "0, GPU-AAAA1111-0000, 00000000:65:00.0, GPU A, 100, 70, 30\n"
        "1, GPU-BBBB2222-0000, 00000000:17:00.0, GPU B, 200, 150, 50\n"
    )

    def query(*args, **kwargs):
        return SimpleNamespace(stdout=output)

    logical = {
        "backend": "gpu",
        "platform": "gpu",
        "visible_device_index": 0,
        "visible_device_count": 2,
        "device_kind": "NVIDIA Test GPU",
    }
    mapper_calls = []

    def reordered_mapper(**kwargs):
        mapper_calls.append(kwargs)
        return {"uuid": "GPU-BBBB2222-0000"}

    reordered = discover_nvidia_device_capacity(
        logical,
        command_runner=query,
        cuda_visible_devices="1,0",
        cuda_device_order="PCI_BUS_ID",
        cuda_visible_ordinal_mapper=reordered_mapper,
        preallocate_setting="false",
    )
    assert reordered.available
    assert reordered.device_name == "GPU B"
    assert mapper_calls == [{
        "visible_index": 0,
        "cuda_visible_devices": "1,0",
        "cuda_device_order": "PCI_BUS_ID",
    }]

    missing_mapping = discover_nvidia_device_capacity(
        logical,
        command_runner=query,
        cuda_visible_devices="1,0",
        preallocate_setting="false",
    )
    assert not missing_mapping.available
    uuid_visible = discover_nvidia_device_capacity(
        logical,
        command_runner=query,
        cuda_visible_devices="GPU-AAAA1111",
        preallocate_setting="false",
    )
    assert uuid_visible.available and uuid_visible.device_name == "GPU A"
    pci_identity = discover_nvidia_device_capacity(
        {**logical, "pci_bus_id": "0000:17:00.0"},
        command_runner=query,
        cuda_visible_devices="0,1",
        preallocate_setting="false",
    )
    assert pci_identity.available and pci_identity.device_name == "GPU B"
    wrong = discover_nvidia_device_capacity(
        logical,
        command_runner=query,
        cuda_visible_ordinal_mapper=lambda **kwargs: {"uuid": "GPU-WRONG"},
        preallocate_setting="false",
    )
    assert not wrong.available

    def ambiguous_query(*args, **kwargs):
        return SimpleNamespace(
            stdout=(
                "0, GPU-ABCDEF11, 0000:01:00.0, A, 100, 80, 20\n"
                "1, GPU-ABCDEF22, 0000:02:00.0, B, 100, 80, 20\n"
            )
        )

    ambiguous = discover_nvidia_device_capacity(
        logical,
        command_runner=ambiguous_query,
        cuda_visible_devices="GPU-ABCDEF",
        preallocate_setting="false",
    )
    assert not ambiguous.available

    def mig_query(*args, **kwargs):
        return SimpleNamespace(
            stdout="0, MIG-GPU-AAAA1111/1/2, 0000:65:00.0, MIG slice, 20, 15, 5\n"
        )

    mig = discover_nvidia_device_capacity(
        {**logical, "visible_device_count": 1, "mig_uuid": "MIG-GPU-AAAA1111/1/2"},
        command_runner=mig_query,
        preallocate_setting="false",
    )
    assert mig.available


def test_small_exact_executables_show_parity_and_separable_memory_reduction() -> None:
    raw = _raw()
    case = raw["cases"][1]
    rows, result = runner.benchmark_case(
        case,
        raw["normalized_configuration"]["model"],
        {"warmup": 0, "repetitions": 2},
    )
    by_key = {(row["kernel"], row["output_mode"]): row for row in rows}

    assert result["parity"]["final_density_max_abs"] <= case["diagnostic_tolerance"]
    assert result["parity"]["source_summary_max_abs"] <= case["diagnostic_tolerance"]
    assert by_key[("separable", "bounded_from_histogram")]["compiled_device_requirement_bytes"] < by_key[("flat", "bounded_from_histogram")]["compiled_device_requirement_bytes"]
    assert not by_key[("separable", "bounded_from_histogram")]["contains_flat_D_by_4_shape"]
    assert by_key[("flat", "bounded_from_histogram")]["contains_flat_D_by_4_shape"]
    assert all(report["available"] for report in result["compiled_reports"].values())
    for row in rows:
        samples = np.asarray(json.loads(row["timing_samples_seconds"]))
        positions = json.loads(row["timing_order_positions"])
        assert samples.shape == (2,)
        assert positions == [0, 1] or positions == [1, 0]
        assert set(positions) == {0, 1}
        assert row["median_execution_seconds"] == pytest.approx(np.median(samples))
        assert row["minimum_execution_seconds"] == pytest.approx(np.min(samples))
        assert row["maximum_execution_seconds"] == pytest.approx(np.max(samples))
        assert row["mad_execution_seconds"] == pytest.approx(
            np.median(np.abs(samples - np.median(samples)))
        )
    expected_orders = [["flat", "separable"], ["separable", "flat"]]
    assert result["execution_orders"]["full_validation"] == expected_orders
    assert result["execution_orders"]["bounded_from_histogram"] == expected_orders


def _benchmark_fake_bundle(signature: dict, report_updates: dict | None = None):
    calls = []

    report = {
        "available": True,
        "analysis_status": "complete",
        "argument_bytes": 10,
        "output_bytes": 20,
        "temporary_bytes": 30,
        "alias_bytes": 5,
        "host_argument_bytes": 0,
        "host_output_bytes": 0,
        "host_temporary_bytes": 0,
        "host_alias_bytes": 0,
        "compiled_device_requirement_bytes": 55,
        "compiled_host_requirement_bytes": 0,
        "static_host_allowance_bytes": 0,
        "compiled_plus_host_requirement_bytes": 55,
        "backend": signature["backend"],
    }
    report.update(report_updates or {})
    statistics = SimpleNamespace(
        argument_size_in_bytes=report.get("argument_bytes"),
        output_size_in_bytes=report.get("output_bytes"),
        temp_size_in_bytes=report.get("temporary_bytes"),
        alias_size_in_bytes=report.get("alias_bytes"),
        host_argument_size_in_bytes=report.get("host_argument_bytes"),
        host_output_size_in_bytes=report.get("host_output_bytes"),
        host_temp_size_in_bytes=report.get("host_temporary_bytes"),
        host_alias_size_in_bytes=report.get("host_alias_bytes"),
    )

    class Compiled:
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return jnp.asarray(1.0)

        def memory_analysis(self):
            return statistics

    compiled = Compiled()
    abstract, static, environment = runner._bundle_contract_parts(signature)
    return (
        make_compiled_executable_bundle(
            compiled_callable=compiled,
            memory_report=report,
            compile_signature=signature,
            abstract_arguments=abstract,
            static_values=static,
            runtime_environment=environment,
        ),
        calls,
    )


def _small_benchmark_signature():
    raw = _raw()
    case = raw["cases"][0]
    model = raw["normalized_configuration"]["model"]
    grid = runner.build_jax_pair_grid(
        runner.QGrid(case["q_min"], case["q_max"], case["spacing"]), jnp.float32
    )
    histogram = jax.ShapeDtypeStruct((case["agent_grid_points"],), jnp.float32)
    states = jax.ShapeDtypeStruct((2,), jnp.float32)
    slots = jnp.asarray(runner.source_slots(case["source_times"], case["steps"]))
    signature = runner.executable_signature(
        case,
        model,
        histogram=histogram,
        state_probabilities=states,
        grid=grid,
        slots=slots,
        kernel="separable",
        output_mode="bounded_from_histogram",
    )
    return raw, case, signature


@pytest.mark.parametrize(
    "report_updates",
    [
        {"available": False, "analysis_status": "unavailable"},
        {"temporary_bytes": None},
        {"compiled_device_requirement_bytes": 54},
    ],
)
def test_compiled_analysis_rejects_before_any_invocation(report_updates) -> None:
    _, case, signature = _small_benchmark_signature()
    bundle, calls = _benchmark_fake_bundle(signature, report_updates)
    estimate = estimate_separable_resources(
        grid_size=case["grid_size"],
        dtype_bytes=case["item_bytes"],
        steps=case["steps"],
        summary_count=case["summary_count"],
        row_block_size=case["row_block_size"],
        column_block_size=case["column_block_size"],
        return_final_density=False,
    )
    with pytest.raises(ValueError, match="failed closed|before execution|malformed|inconsistent"):
        runner._validate_benchmark_bundle(
            bundle, estimate, label="injected", allow_expensive=True
        )
    assert calls == []


def test_excessive_compiled_report_rejects_before_invocation() -> None:
    _, case, signature = _small_benchmark_signature()
    temporary = runner.MAX_COMPILED_DEVICE_BYTES + 1
    claimed = 10 + 20 + temporary - 5
    bundle, calls = _benchmark_fake_bundle(
        signature,
        {
            "temporary_bytes": temporary,
            "compiled_device_requirement_bytes": claimed,
            "compiled_plus_host_requirement_bytes": claimed,
        },
    )
    estimate = estimate_separable_resources(
        grid_size=case["grid_size"], dtype_bytes=4, steps=case["steps"],
        summary_count=case["summary_count"], row_block_size=1,
        column_block_size=1, return_final_density=False,
    )
    with pytest.raises(ValueError, match="before execution"):
        runner._validate_benchmark_bundle(
            bundle, estimate, label="excessive", allow_expensive=True
        )
    assert calls == []


def test_live_memory_analysis_must_match_the_stored_report() -> None:
    _, case, signature = _small_benchmark_signature()
    original, _ = _benchmark_fake_bundle(signature)

    class DifferentLiveExecutable:
        def __call__(self, *args, **kwargs):
            pytest.fail("mismatched live executable was invoked")

        def memory_analysis(self):
            return SimpleNamespace(
                argument_size_in_bytes=10,
                output_size_in_bytes=20,
                temp_size_in_bytes=31,
                alias_size_in_bytes=5,
                host_argument_size_in_bytes=0,
                host_output_size_in_bytes=0,
                host_temp_size_in_bytes=0,
                host_alias_size_in_bytes=0,
            )

    bundle = make_compiled_executable_bundle(
        compiled_callable=DifferentLiveExecutable(),
        memory_report=original.memory_report,
        compile_signature=signature,
        abstract_arguments=original.abstract_arguments,
        static_values=original.static_values,
        runtime_environment=original.runtime_environment,
    )
    estimate = estimate_separable_resources(
        grid_size=case["grid_size"], dtype_bytes=4, steps=case["steps"],
        summary_count=case["summary_count"], row_block_size=1,
        column_block_size=1, return_final_density=False,
    )
    with pytest.raises(ValueError, match="live temporary_bytes"):
        runner._validate_benchmark_bundle(
            bundle, estimate, label="live-mismatch", allow_expensive=True
        )


def test_exact_bundle_callable_is_invoked_once_and_identity_is_non_overridable() -> None:
    raw, case, signature = _small_benchmark_signature()
    model = raw["normalized_configuration"]["model"]
    grid = runner.build_jax_pair_grid(
        runner.QGrid(case["q_min"], case["q_max"], case["spacing"]), jnp.float32
    )
    slots = jnp.asarray(runner.source_slots(case["source_times"], case["steps"]))
    arguments = (
        jnp.zeros((case["agent_grid_points"],), dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        grid,
        model["alpha"],
        model["tau"],
        slots,
    )
    bundle, calls = _benchmark_fake_bundle(signature)
    result = runner._invoke_accepted_bundle(
        bundle,
        arguments,
        case=case,
        model=model,
        kernel="separable",
        output_mode="bounded_from_histogram",
        diagnostic_tolerance=signature["diagnostic_tolerance"],
    )
    assert float(np.asarray(result)) == 1.0
    assert len(calls) == 1
    description = runner._bounded_bundle_description(bundle)
    serialized_report = runner._bounded_memory_report(bundle)
    assert "compile_signature" not in description
    assert "executable_signature" not in serialized_report
    assert description["bundle_integrity_sha256"] == bundle.bundle_integrity_sha256
    invalid_calls = [
        (arguments, case, 999.0),
        ((jnp.zeros((case["agent_grid_points"] + 1,), dtype=jnp.float32), *arguments[1:]), case, signature["diagnostic_tolerance"]),
        ((jnp.zeros((case["agent_grid_points"],), dtype=jnp.int32), *arguments[1:]), case, signature["diagnostic_tolerance"]),
        ((arguments[0], arguments[1], arguments[2], jnp.asarray(model["alpha"], dtype=jnp.float32), arguments[4], arguments[5]), case, signature["diagnostic_tolerance"]),
        (arguments, {**case, "steps": case["steps"] + 1}, signature["diagnostic_tolerance"]),
        (arguments, {**case, "row_block_size": case["row_block_size"] + 1}, signature["diagnostic_tolerance"]),
        ((*arguments[:-1], arguments[-1].at[0].set(-1)), case, signature["diagnostic_tolerance"]),
    ]
    for actual_arguments, actual_case, tolerance in invalid_calls:
        with pytest.raises(ValueError, match="signature"):
            runner._invoke_accepted_bundle(
                bundle,
                actual_arguments,
                case=actual_case,
                model=model,
                kernel="separable",
                output_mode="bounded_from_histogram",
                diagnostic_tolerance=tolerance,
            )
    assert len(calls) == 1
    mismatched_callable = replace(bundle, compiled_callable=lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="identity"):
        runner._invoke_accepted_bundle(
            mismatched_callable,
            arguments,
            case=case,
            model=model,
            kernel="separable",
            output_mode="bounded_from_histogram",
            diagnostic_tolerance=signature["diagnostic_tolerance"],
        )


@pytest.mark.parametrize("runtime_field", ["backend", "execution_device", "jax_enable_x64"])
def test_invocation_rebuild_rejects_runtime_mutation(monkeypatch, runtime_field) -> None:
    raw, case, signature = _small_benchmark_signature()
    model = raw["normalized_configuration"]["model"]
    grid = runner.build_jax_pair_grid(
        runner.QGrid(case["q_min"], case["q_max"], case["spacing"]), jnp.float32
    )
    arguments = (
        jnp.zeros((case["agent_grid_points"],), dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        grid,
        model["alpha"],
        model["tau"],
        jnp.asarray(runner.source_slots(case["source_times"], case["steps"])),
    )
    bundle, calls = _benchmark_fake_bundle(signature)
    environment = deepcopy(signature["contract_runtime_environment"])
    if runtime_field == "backend":
        environment["backend"] = "gpu"
    elif runtime_field == "execution_device":
        environment["execution_device"] = {
            **environment["execution_device"], "id": "other"
        }
    else:
        environment["jax_enable_x64"] = not environment["jax_enable_x64"]
    monkeypatch.setattr(runner, "_runtime_environment_signature", lambda: environment)
    with pytest.raises(ValueError, match="signature"):
        runner._invoke_accepted_bundle(
            bundle,
            arguments,
            case=case,
            model=model,
            kernel="separable",
            output_mode="bounded_from_histogram",
            diagnostic_tolerance=case["diagnostic_tolerance"],
        )
    assert calls == []


def test_production_capacity_is_rechecked_immediately_before_invocation(monkeypatch) -> None:
    raw, case, signature = _small_benchmark_signature()
    model = raw["normalized_configuration"]["model"]
    grid = runner.build_jax_pair_grid(
        runner.QGrid(case["q_min"], case["q_max"], case["spacing"]), jnp.float32
    )
    arguments = (
        jnp.zeros((case["agent_grid_points"],), dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        grid,
        model["alpha"],
        model["tau"],
        jnp.asarray(runner.source_slots(case["source_times"], case["steps"])),
    )
    bundle, calls = _benchmark_fake_bundle(signature)
    admission_calls = []

    def admission(**kwargs):
        assert calls == []
        admission_calls.append(kwargs)
        return {"passed": True}

    monkeypatch.setattr(runner, "production_capacity_preflight", admission)
    runner._invoke_accepted_bundle(
        bundle,
        arguments,
        case=case,
        model=model,
        kernel="separable",
        output_mode="bounded_from_histogram",
        diagnostic_tolerance=case["diagnostic_tolerance"],
        production_feasibility={"test": True},
        capacity_observation=object(),
        allow_expensive=True,
    )
    assert len(admission_calls) == 1
    assert len(calls) == 1


def test_benchmark_memory_failure_precedes_initializer_warmup_timing_and_output(
    monkeypatch,
) -> None:
    def unavailable(lowered, signature, *, static_host_bytes):
        return _benchmark_fake_bundle(
            signature, {"available": False, "analysis_status": "unavailable"}
        )[0], 0.0

    monkeypatch.setattr(runner, "_compile_and_analyze", unavailable)
    monkeypatch.setattr(
        runner,
        "_invoke_accepted_bundle",
        lambda *args, **kwargs: pytest.fail("executable entered before memory rejection"),
    )
    monkeypatch.setattr(
        runner,
        "_write_outputs",
        lambda *args, **kwargs: pytest.fail("output entered before memory rejection"),
    )
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: SimpleNamespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    assert not hasattr(runner, "ordered_pair_mass_from_histogram_jit")
    with pytest.raises(ValueError, match="failed closed|before execution"):
        runner.main()


def test_sorted_reduction_investigation_measures_cost_and_is_not_adopted() -> None:
    result = runner.investigate_reduction_strategies(points=9, repetitions=2)
    assert result["max_abs_parity_error"] <= 1e-6
    assert result["sorted_segment"]["memory_report"]["temporary_bytes"] > result["scatter"]["memory_report"]["temporary_bytes"]
    assert result["execution_order"] == [
        ["scatter", "sorted_segment"],
        ["sorted_segment", "scatter"],
    ]
    assert result["scatter"]["timing_order_positions"] == [0, 1]
    assert result["sorted_segment"]["timing_order_positions"] == [1, 0]
    assert not result["adopted"]


@pytest.mark.parametrize("invalid_kernel", ["scatter", "sorted_segment"])
@pytest.mark.parametrize(
    "report_updates",
    [
        {"available": False, "analysis_status": "unavailable"},
        {"temporary_bytes": None},
        {
            "temporary_bytes": runner.MAX_COMPILED_DEVICE_BYTES + 1,
            "compiled_device_requirement_bytes": (
                runner.MAX_COMPILED_DEVICE_BYTES + 1 + 10 + 20 - 5
            ),
            "compiled_plus_host_requirement_bytes": (
                runner.MAX_COMPILED_DEVICE_BYTES + 1 + 10 + 20 - 5
            ),
        },
    ],
)
def test_reduction_report_rejects_before_device_inputs_or_execution(
    monkeypatch, invalid_kernel, report_updates
) -> None:
    compile_order = []

    def injected(lowered, signature, *, static_host_bytes):
        name = signature["kernel"]
        compile_order.append(name)
        updates = report_updates if name == invalid_kernel else None
        return _benchmark_fake_bundle(signature, updates)[0], 0.0

    monkeypatch.setattr(runner, "_compile_and_analyze", injected)
    monkeypatch.setattr(
        runner.jnp,
        "asarray",
        lambda *args, **kwargs: pytest.fail("device inputs constructed before rejection"),
    )
    monkeypatch.setattr(
        runner,
        "_invoke_reduction_bundle",
        lambda *args, **kwargs: pytest.fail("reduction executable entered before rejection"),
    )
    with pytest.raises(ValueError, match="failed closed|exceeds|malformed"):
        runner.investigate_reduction_strategies(points=3, repetitions=2)
    assert invalid_kernel in compile_order


def test_reduction_identity_and_actual_arguments_gate_exact_callable() -> None:
    values = jnp.arange(9, dtype=jnp.float32)
    destinations = jnp.arange(9, dtype=jnp.int32)
    signature = runner._reduction_signature(
        name="scatter", points=3, values=values, destinations=destinations
    )
    bundle, calls = _benchmark_fake_bundle(signature)
    runner._invoke_reduction_bundle(
        bundle, (values, destinations), name="scatter", points=3
    )
    assert len(calls) == 1
    for invalid_arguments in (
        (jnp.arange(8, dtype=jnp.float32), destinations),
        (jnp.arange(9, dtype=jnp.float16), destinations),
        (values, jnp.arange(9, dtype=jnp.int16)),
    ):
        with pytest.raises(ValueError, match="signature"):
            runner._invoke_reduction_bundle(
                bundle, invalid_arguments, name="scatter", points=3
            )
    mismatched = replace(bundle, compiled_callable=lambda *args: None)
    with pytest.raises(ValueError, match="identity"):
        runner._invoke_reduction_bundle(
            mismatched, (values, destinations), name="scatter", points=3
        )
    assert len(calls) == 1


def test_complete_result_synchronization_waits_for_every_leaf() -> None:
    calls = []

    class Leaf:
        def __init__(self, label):
            self.label = label

        def block_until_ready(self):
            calls.append(self.label)
            return self

    runner._block_complete((Leaf("first"), Leaf("second")))
    assert calls == ["first", "second"]


def test_resource_override_cannot_bypass_scientific_result_validation() -> None:
    case = _raw()["cases"][0]
    diagnostics = SimpleNamespace(
        finite=jnp.asarray([False]),
        nonnegative=jnp.asarray([True]),
        conditional_moments_valid=jnp.asarray([True]),
        total_mass=jnp.asarray([1.0]),
        symmetry_error=jnp.asarray([0.0]),
    )
    result = SimpleNamespace(
        diagnostics=diagnostics,
        destinations_valid=jnp.asarray([True]),
        final_mass=jnp.asarray(np.ones((2, 9, 9)) / 162, dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="non-finite"):
        runner._validate_scientific_result(result, case, has_final_mass=True)


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="requires a fresh CPU+x64 process",
)
def test_x64_static_and_compiled_case_is_supported() -> None:
    raw = _raw()
    case = deepcopy(raw["cases"][0])
    case["dtype"] = "float64"
    case["item_bytes"] = 8
    case["density_bytes"] *= 2
    rows, result = runner.benchmark_case(
        case,
        raw["normalized_configuration"]["model"],
        {"warmup": 0, "repetitions": 2},
    )
    assert all(row["dtype"] == "float64" for row in rows)
    assert result["parity"]["final_density_max_abs"] <= 1e-12

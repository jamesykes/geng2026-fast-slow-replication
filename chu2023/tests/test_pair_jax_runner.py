from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from experiments import run_pair_jax_small as runner
from chu_pair.config import LearningConfig
from chu_pair.grids import QGrid
from chu_pair.pair_density.jax_solver import build_jax_pair_grid


def _default_raw(dtype: str = "float32") -> dict:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["solver"]["dtype"] = dtype
    return runner.inspect_raw_pair_config(config)


@pytest.mark.parametrize(
    ("section", "key"),
    [(None, "unknown_section"), ("solver", "unknown_key")],
)
def test_configuration_schema_rejects_unknown_sections_and_keys(
    section: str | None, key: str
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    target = config if section is None else config[section]
    target[key] = "x" * 10_000

    with pytest.raises(ValueError, match="unknown"):
        runner.inspect_raw_pair_config(config)


@pytest.mark.parametrize(
    "run_name",
    [
        "A" * (runner.MAX_RUN_NAME_LENGTH + 1),
        "../escape",
        "dir/name",
        "dir\\name",
        "bad\nname",
        "snowman-☃",
    ],
)
def test_run_name_rejects_unbounded_escaped_and_unsafe_strings(run_name: str) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["output"]["run_name"] = run_name

    with pytest.raises(ValueError, match="output.run_name"):
        runner.inspect_raw_pair_config(config)


def test_maximum_valid_normalized_configuration_has_a_proven_serialization_bound() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["output"]["run_name"] = "A" + "-" * (runner.MAX_RUN_NAME_LENGTH - 1)
    config["solver"].update(
        steps=runner.MAX_CONFIG_INTEGER,
        chunk_size=runner.MAX_CONFIG_INTEGER,
        diagnostic_stride=runner.MAX_CONFIG_INTEGER,
    )
    raw = runner.inspect_raw_pair_config(config)
    independent_metadata_chars = (
        raw["normalized_configuration_json_chars"]
        + runner.MAX_GIT_STATUS_CHARS * runner.MAX_JSON_BYTES_PER_CHARACTER
        + runner.MAX_COMPILED_REASON_CHARS * runner.MAX_JSON_BYTES_PER_CHARACTER
        + runner.MAX_DEVICE_COUNT
        * 3
        * runner.MAX_DEVICE_FIELD_CHARS
        * runner.MAX_JSON_BYTES_PER_CHARACTER
        + runner.FIXED_METADATA_JSON_CHARS
    )
    independent_payload = independent_metadata_chars + runner.MAX_CSV_WRITE_CHARS
    independent_metadata_encoding_peak = (
        3
        * runner.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * independent_metadata_chars
        + runner.SERIALIZATION_FIXED_OVERHEAD_BYTES
    )
    independent_metadata_write_peak = (
        runner.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * independent_metadata_chars
        + runner.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * runner.SERIALIZATION_CHUNK_CHARS
        + runner.SERIALIZATION_CHUNK_CHARS
        + runner.SERIALIZATION_IO_BUFFER_BYTES
        + runner.SERIALIZATION_FIXED_OVERHEAD_BYTES
    )
    independent_csv_write_peak = (
        runner.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * independent_metadata_chars
        + runner.SERIALIZATION_TEXT_STORAGE_BYTES_PER_CHAR
        * runner.MAX_CSV_WRITE_CHARS
        + runner.MAX_CSV_WRITE_CHARS
        + runner.SERIALIZATION_IO_BUFFER_BYTES
        + runner.SERIALIZATION_FIXED_OVERHEAD_BYTES
    )
    independent_peak = max(
        independent_metadata_encoding_peak,
        independent_metadata_write_peak,
        independent_csv_write_peak,
    )

    assert set(raw["normalized_configuration"]) == set(runner.EXPECTED_CONFIG_KEYS)
    assert raw["normalized_configuration"]["output"]["run_name"] == config["output"][
        "run_name"
    ]
    assert raw["metadata_json_text_bound_chars"] == independent_metadata_chars
    assert raw["serialization_payload_bound_bytes"] == independent_payload
    assert raw["serialization_live_peak_bytes"] == independent_peak
    assert runner.serialization_live_peak_components(
        raw["normalized_configuration_json_chars"]
    ) == {
        "metadata_encoding_peak_bytes": independent_metadata_encoding_peak,
        "metadata_write_peak_bytes": independent_metadata_write_peak,
        "csv_write_peak_bytes": independent_csv_write_peak,
    }
    assert independent_peak <= runner.MAX_SERIALIZATION_LIVE_PEAK_BYTES


def test_serialized_configuration_contains_only_normalized_recognized_fields() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    raw = runner.inspect_raw_pair_config(config)
    encoded = runner.encode_bounded_metadata(
        {"configuration": raw["normalized_configuration"]},
        raw["metadata_json_text_bound_chars"],
    )
    decoded = json.loads(encoded)

    assert decoded["configuration"] == raw["normalized_configuration"]
    assert set(decoded["configuration"]) == set(runner.EXPECTED_CONFIG_KEYS)
    assert encoded.isascii()
    assert len(encoded) <= raw["metadata_json_text_bound_chars"]


def test_serialization_bound_covers_worst_case_json_escaping() -> None:
    escaped = "\U0010ffff"
    probe = {
        "git_status": escaped * runner.MAX_GIT_STATUS_CHARS,
        "compiled_reason": escaped * runner.MAX_COMPILED_REASON_CHARS,
        "devices": [
            {
                name: escaped * runner.MAX_DEVICE_FIELD_CHARS
                for name in ("id", "platform", "device_kind")
            }
            for _ in range(runner.MAX_DEVICE_COUNT)
        ],
    }
    escaped_json = json.dumps(probe, ensure_ascii=True, separators=(",", ":"))
    actual_variable_chars = len(escaped_json)
    accounted_variable_chars = (
        runner.MAX_GIT_STATUS_CHARS * runner.MAX_JSON_BYTES_PER_CHARACTER
        + runner.MAX_COMPILED_REASON_CHARS * runner.MAX_JSON_BYTES_PER_CHARACTER
        + runner.MAX_DEVICE_COUNT
        * 3
        * runner.MAX_DEVICE_FIELD_CHARS
        * runner.MAX_JSON_BYTES_PER_CHARACTER
        + runner.FIXED_METADATA_JSON_CHARS
    )

    assert escaped_json.isascii()
    assert actual_variable_chars <= accounted_variable_chars
    assert (
        runner.serialization_live_peak_bound(
            runner.MAX_NORMALIZED_CONFIGURATION_JSON_CHARS
        )
        <= runner.MAX_SERIALIZATION_LIVE_PEAK_BYTES
    )


def test_metadata_length_validation_does_not_create_an_ascii_byte_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AsciiTextThatMustNotBeEncoded(str):
        def __add__(self, other):
            return AsciiTextThatMustNotBeEncoded(super().__add__(other))

        def encode(self, *args, **kwargs):
            raise AssertionError("full metadata text was redundantly encoded")

    monkeypatch.setattr(
        runner.json,
        "dumps",
        lambda *args, **kwargs: AsciiTextThatMustNotBeEncoded('{"ok":true}'),
    )

    encoded = runner.encode_bounded_metadata({}, maximum_chars=32)

    assert encoded == '{"ok":true}\n'


@pytest.mark.parametrize("kind", ["record", "header"])
def test_csv_accepts_exact_character_boundary_and_rejects_one_over(kind: str) -> None:
    maximum = runner.MAX_CSV_WRITE_CHARS
    if kind == "record":
        accepted = [{"field": "x" * (maximum - 2)}]
        rejected = [{"field": "x" * (maximum - 1)}]
    else:
        accepted = [{"h" * (maximum - 2): ""}]
        rejected = [{"h" * (maximum - 1): ""}]

    assert runner.validate_csv_record_serialization(accepted) == maximum
    with pytest.raises(RuntimeError, match="record allowance"):
        runner.validate_csv_record_serialization(rejected)


def test_csv_validation_streams_one_write_at_a_time_without_stringio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = []
    original_sink = runner._BoundedCountingTextSink

    class AuditedSink(original_sink):
        def write(self, text: str) -> int:
            writes.append(text)
            return super().write(text)

    monkeypatch.setattr(runner, "_BoundedCountingTextSink", AuditedSink)
    rows = [{"field": index} for index in range(3)]

    runner.validate_csv_record_serialization(rows)

    assert writes == ["field\r\n", "0\r\n", "1\r\n", "2\r\n"]
    source = inspect.getsource(runner.validate_csv_record_serialization)
    assert "StringIO" not in source
    assert "getvalue" not in source


def test_bounded_writers_preserve_existing_ascii_json_and_csv_bytes(tmp_path) -> None:
    rows = [{"time": 0, "finite": True}, {"time": 1, "finite": False}]
    metadata = {"configuration": {"dtype": "float32"}, "rows": 2}
    encoded = runner.encode_bounded_metadata(metadata, maximum_chars=1_024)
    metadata_path = tmp_path / "metadata.json"
    csv_path = tmp_path / "diagnostics.csv"

    runner.write_bounded_metadata(metadata_path, encoded)
    runner.write_bounded_csv(csv_path, rows)
    expected_csv = io.StringIO(newline="")
    writer = csv.DictWriter(expected_csv, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    assert metadata_path.read_bytes() == encoded.encode("ascii")
    assert csv_path.read_bytes() == expected_csv.getvalue().encode("ascii")


def test_invalid_metadata_configuration_is_rejected_before_numeric_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["output"]["run_name"] = "../escape"
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric work entered before configuration rejection")

    for name in (
        "git_text",
        "validate_pair_budget",
        "QGrid",
        "analyze_compiled_pair_memory",
        "tiny_histogram",
        "checked_simulate_pair_density",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="output.run_name"):
        runner.main()


def test_final_metadata_bound_failure_precedes_output_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(runner, "git_text", lambda *args: "")
    monkeypatch.setattr(runner, "build_jax_pair_grid", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "analyze_compiled_pair_memory",
        lambda *args: {
            "available": True,
            "compiled_device_requirement_bytes": 1,
            "compiled_plus_host_requirement_bytes": 1,
        },
    )
    monkeypatch.setattr(runner, "normalized_device_metadata", lambda: [])
    monkeypatch.setattr(runner, "tiny_histogram", lambda *args: object())
    monkeypatch.setattr(runner, "ordered_pair_mass_jax", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "validate_jax_pair_mass", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "pair_diagnostics_jax", lambda *args, **kwargs: object())
    final_mass = SimpleNamespace(block_until_ready=lambda: None)
    monkeypatch.setattr(
        runner,
        "checked_simulate_pair_density",
        lambda *args, **kwargs: SimpleNamespace(
            final_mass=final_mass, diagnostics=object()
        ),
    )
    monkeypatch.setattr(
        runner,
        "diagnostics_rows",
        lambda *args: [
            {"time": time, "total_mass": 1.0} for time in range(5)
        ],
    )
    monkeypatch.setattr(
        runner,
        "encode_bounded_metadata",
        lambda metadata, maximum_chars: "x" * (maximum_chars + 1),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("output creation entered after failed serialization check")

    monkeypatch.setattr(runner, "validate_csv_record_serialization", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)

    with pytest.raises(RuntimeError, match="preflight text bound"):
        runner.main()


@pytest.mark.parametrize(
    (
        "dtype",
        "expected_device",
        "expected_host",
        "expected_combined",
    ),
    [
        ("float32", 53_704, 3_704_768, 3_758_472),
        ("float64", 100_252, 3_710_364, 3_810_616),
    ],
)
def test_static_resource_components_match_independent_hand_calculation(
    dtype: str,
    expected_device: int,
    expected_host: int,
    expected_combined: int,
) -> None:
    estimate = runner.estimate_pair_resources(_default_raw(dtype))
    item_bytes = 4 if dtype == "float32" else 8

    assert estimate["agent_grid_points"] == 25
    assert estimate["ordered_pair_cells"] == 625
    assert estimate["state_expanded_cells"] == 1_250
    assert estimate["initial_pair_bytes"] == 1_250 * item_bytes
    assert estimate["diagnostic_output_rows"] == 5
    components = estimate["components"]
    assert components["static_full_density_device_copies"] == 8
    assert components["full_density_device_bytes"] == 8 * 1_250 * item_bytes
    assert components["point_working_bytes"] == 5 * item_bytes + 25 * (
        20 * item_bytes + 40
    )
    assert components["branch_weight_bytes"] == 64 * 17 * item_bytes
    assert components["branch_index_bytes"] == 64 * 96
    assert components["diagnostic_trajectory_bytes"] == 4 * (
        11 * item_bytes + 3
    )
    assert components["validation_host_copy_bytes"] == 1_250 * item_bytes
    assert components["diagnostic_host_trajectory_bytes"] == 4 * (
        11 * item_bytes + 3
    )
    assert components["histogram_host_bytes"] == 25 * 8
    assert components["grid_construction_host_bytes"] == (
        5 * (16 + item_bytes) + 25 * (8 + 4 * item_bytes)
    )
    assert components["diagnostic_row_host_bytes"] == 5 * 4_096
    assert components["metadata_json_text_bound_chars"] == 106_837
    assert components["serialization_payload_bound_bytes"] == 110_933
    assert components["serialization_live_peak_bytes"] == 2_629_624
    assert components["metadata_encoding_peak_bytes"] == 2_629_624
    assert components["metadata_write_peak_bytes"] == 965_288
    assert components["csv_write_peak_bytes"] == 965_288
    assert components["serialization_text_storage_bytes_per_char"] == 8
    assert components["serialization_chunk_chars"] == 4_096
    assert components["serialization_io_buffer_bytes"] == 8_192
    assert components["serialization_fixed_overhead_bytes"] == 65_536
    assert components["max_csv_write_chars"] == 4_096
    assert components["source_hash_buffer_bytes"] == 1 << 20
    assert estimate["static_device_bytes"] == expected_device
    assert estimate["static_host_bytes"] == expected_host
    assert estimate["static_combined_peak_bytes"] == expected_combined
    assert expected_combined == expected_device + expected_host
    assert estimate["retained_full_density_snapshots"] == 0


def test_corrected_serialization_peak_is_accepted_at_limit_and_rejected_one_below() -> None:
    raw = _default_raw()
    estimate = runner.estimate_pair_resources(raw)
    limits = runner._static_limit_values(estimate)

    accepted = runner.validate_pair_budget(raw, False, limits=limits)
    assert accepted["static_estimates"]["static_combined_peak_bytes"] == limits[
        "combined_peak_bytes"
    ]

    limits["combined_peak_bytes"] -= 1
    with pytest.raises(ValueError, match="combined_peak_bytes"):
        runner.validate_pair_budget(raw, False, limits=limits)


@pytest.mark.parametrize(("steps", "chunk_size"), [(0, 1), (1, 7), (4, 64), (4, 1_250)])
def test_static_estimate_handles_zero_steps_rows_and_chunk_sizes(
    steps: int, chunk_size: int
) -> None:
    raw = _default_raw()
    raw["steps"] = steps
    raw["chunk_size"] = chunk_size
    estimate = runner.estimate_pair_resources(raw)
    effective_chunk = min(chunk_size, 1_250)
    rows = 1 + steps // raw["diagnostic_stride"] + int(
        steps % raw["diagnostic_stride"] != 0
    )

    assert estimate["diagnostic_output_rows"] == rows
    assert estimate["components"]["effective_source_chunk_cells"] == effective_chunk
    assert estimate["components"]["branch_weight_bytes"] == effective_chunk * 17 * 4
    assert estimate["components"]["branch_index_bytes"] == effective_chunk * 96
    assert estimate["components"]["diagnostic_trajectory_bytes"] == steps * 47
    assert estimate["components"]["diagnostic_host_trajectory_bytes"] == steps * 47
    assert estimate["components"]["diagnostic_row_host_bytes"] == rows * 4_096


@pytest.mark.parametrize(
    "resource_name",
    [
        name
        for name in runner.PHASE4_ABSOLUTE_LIMITS
        if name != "retained_full_density_snapshots"
    ],
)
def test_every_fixed_phase4_limit_is_enforced(resource_name: str) -> None:
    raw = _default_raw()
    estimate = runner.estimate_pair_resources(raw)
    limits = runner._static_limit_values(estimate)
    limits[resource_name] -= 1
    with pytest.raises(ValueError, match=resource_name):
        runner.validate_pair_budget(raw, False, limits=limits)


def test_runner_has_no_configurable_full_density_snapshot_path() -> None:
    raw = _default_raw()
    budget = runner.validate_pair_budget(raw, False)

    assert budget["static_estimates"]["retained_full_density_snapshots"] == 0
    assert budget["absolute_limits"]["retained_full_density_snapshots"] == 0


def test_near_limit_float64_case_rejected_by_new_full_density_and_host_bound() -> None:
    points = 43 * 43
    raw = {
        "grid_size": 43,
        "agent_grid_points": points,
        "ordered_pair_cells": points * points,
        "state_expanded_cells": 2 * points * points,
        "dtype": "float64",
        "item_bytes": 8,
        "steps": 4,
        "chunk_size": 64,
        "diagnostic_stride": 1,
    }
    estimate = runner.estimate_pair_resources(raw)
    components = estimate["components"]
    old_device = (
        3 * raw["state_expanded_cells"] * raw["item_bytes"]
        + components["point_working_bytes"]
        + components["branch_weight_bytes"]
        + components["branch_index_bytes"]
        + components["diagnostic_trajectory_bytes"]
    )

    assert old_device < runner.PHASE4_ABSOLUTE_LIMITS["combined_peak_bytes"]
    assert (
        estimate["static_combined_peak_bytes"]
        > runner.PHASE4_ABSOLUTE_LIMITS["combined_peak_bytes"]
    )
    with pytest.raises(ValueError, match="combined_peak_bytes"):
        runner.validate_pair_budget(raw, False)


def test_allow_expensive_is_only_bypass_and_records_all_positive_violations() -> None:
    raw = _default_raw()
    budget = runner.validate_pair_budget(
        raw,
        True,
        limits={name: 0 for name in runner.PHASE4_ABSOLUTE_LIMITS},
    )
    expected = [
        name
        for name in runner.PHASE4_ABSOLUTE_LIMITS
        if runner._static_limit_values(budget["static_estimates"])[name] > 0
    ]

    assert budget["allow_expensive"] is True
    assert budget["static_violations_overridden"] == expected


def test_configuration_cannot_add_safety_section_to_raise_fixed_caps() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["safety"] = {
        f"max_{name}": 10**30 for name in runner.PHASE4_ABSOLUTE_LIMITS
    }
    with pytest.raises(ValueError, match="unknown=.*safety"):
        runner.inspect_raw_pair_config(config)


def test_full_host_diagnostic_trajectory_rejects_reviewed_high_t_case() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["solver"].update(
        steps=3_000_000, diagnostic_stride=3_000_000, chunk_size=1
    )
    raw = runner.inspect_raw_pair_config(config)
    estimate = runner.estimate_pair_resources(raw)
    host_diagnostics = 3_000_000 * (11 * 4 + 3)

    assert host_diagnostics == 141_000_000
    assert estimate["components"]["diagnostic_host_trajectory_bytes"] == host_diagnostics
    assert estimate["components"]["serialization_live_peak_bytes"] == 2_629_888
    assert estimate["static_combined_peak_bytes"] == 285_735_740
    assert estimate["static_combined_peak_bytes"] - host_diagnostics == 144_735_740
    with pytest.raises(ValueError, match="combined_peak_bytes"):
        runner.validate_pair_budget(raw, False)


def test_high_t_host_diagnostic_rejection_precedes_every_numeric_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["solver"].update(
        steps=3_000_000, diagnostic_stride=3_000_000, chunk_size=1
    )
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(runner, "git_text", lambda *args: "")

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric or output stage entered before static rejection")

    for name in (
        "QGrid",
        "build_jax_pair_grid",
        "analyze_compiled_pair_memory",
        "tiny_histogram",
        "ordered_pair_mass_jax",
        "checked_simulate_pair_density",
        "encode_bounded_metadata",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="combined_peak_bytes"):
        runner.main()


def test_large_grid_rejection_precedes_grid_jax_execution_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["grid"] = {"q_min": -1.0, "q_max": 1.0, "spacing": 0.01}
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation or execution entered before Phase 4 rejection")

    for name in (
        "QGrid",
        "build_jax_pair_grid",
        "analyze_compiled_pair_memory",
        "tiny_histogram",
        "ordered_pair_mass_jax",
        "checked_simulate_pair_density",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="before QGrid, JAX arrays, execution, or output"):
        runner.main()


def test_compiled_memory_accounting_subtracts_aliases_and_adds_host_allowance() -> None:
    stats = SimpleNamespace(
        argument_size_in_bytes=100,
        output_size_in_bytes=80,
        temp_size_in_bytes=40,
        alias_size_in_bytes=30,
        host_argument_size_in_bytes=20,
        host_output_size_in_bytes=10,
        host_temp_size_in_bytes=8,
        host_alias_size_in_bytes=5,
    )
    report = runner.compiled_memory_report(
        SimpleNamespace(memory_analysis=lambda: stats), static_host_bytes=1_000
    )

    assert report["available"] is True
    assert report["analysis_status"] == "complete"
    assert report["compiled_device_requirement_bytes"] == 100 + 80 + 40 - 30
    assert report["compiled_host_requirement_bytes"] == 20 + 10 + 8 - 5
    assert report["compiled_plus_host_requirement_bytes"] == 1_223


@pytest.mark.parametrize(
    "memory_result",
    [
        None,
        SimpleNamespace(argument_size_in_bytes=None),
        SimpleNamespace(
            argument_size_in_bytes=-1,
            output_size_in_bytes=0,
            temp_size_in_bytes=0,
            alias_size_in_bytes=0,
            host_argument_size_in_bytes=0,
            host_output_size_in_bytes=0,
            host_temp_size_in_bytes=0,
            host_alias_size_in_bytes=0,
        ),
        SimpleNamespace(
            argument_size_in_bytes=1,
            output_size_in_bytes=1,
            temp_size_in_bytes=0,
            alias_size_in_bytes=3,
            host_argument_size_in_bytes=0,
            host_output_size_in_bytes=0,
            host_temp_size_in_bytes=0,
            host_alias_size_in_bytes=0,
        ),
    ],
)
def test_unavailable_or_inconsistent_compiled_analysis_fails_closed(memory_result) -> None:
    report = runner.compiled_memory_report(
        SimpleNamespace(memory_analysis=lambda: memory_result),
        static_host_bytes=123,
    )

    assert report["performed"] is True
    assert report["available"] is False
    assert report["compiled_device_requirement_bytes"] is None
    assert report["compiled_plus_host_requirement_bytes"] is None
    assert report["unavailable_reason"]
    budget = runner.validate_pair_budget(_default_raw(), False)
    with pytest.raises(ValueError, match="compiled memory analysis is unavailable"):
        runner.validate_compiled_pair_budget(budget, report, False)

    budget = runner.validate_pair_budget(_default_raw(), True)
    budget = runner.validate_compiled_pair_budget(budget, report, True)
    assert budget["compiled_analysis"] == report
    assert report["validation_status"] == "unavailable"
    assert budget["compiled_violations"] == ["compiled_analysis_unavailable"]
    assert budget["compiled_violations_overridden"] == budget["compiled_violations"]


def test_compiled_memory_analysis_exception_is_recorded_as_unavailable() -> None:
    def unavailable():
        raise ValueError("backend does not report executable memory")

    report = runner.compiled_memory_report(
        SimpleNamespace(memory_analysis=unavailable), static_host_bytes=123
    )

    assert report["available"] is False
    assert report["analysis_status"] == "unavailable"
    assert report["unavailable_reason"] == (
        "ValueError: backend does not report executable memory"
    )
    budget = runner.validate_pair_budget(_default_raw(), False)
    with pytest.raises(ValueError, match="compiled memory analysis is unavailable"):
        runner.validate_compiled_pair_budget(budget, report, False)


def test_compiled_validation_status_distinguishes_pass_and_failure() -> None:
    passing_budget = runner.validate_pair_budget(_default_raw(), False)
    passing_report = {
        "available": True,
        "compiled_device_requirement_bytes": 1,
        "compiled_plus_host_requirement_bytes": 1,
    }
    runner.validate_compiled_pair_budget(passing_budget, passing_report, False)
    assert passing_report["validation_status"] == "passed"

    failing_budget = runner.validate_pair_budget(_default_raw(), True)
    failing_report = {
        "available": True,
        "compiled_device_requirement_bytes": failing_budget["static_estimates"][
            "static_device_bytes"
        ]
        + 1,
        "compiled_plus_host_requirement_bytes": 1,
    }
    runner.validate_compiled_pair_budget(failing_budget, failing_report, True)
    assert failing_report["validation_status"] == "failed"
    assert failing_budget["compiled_violations_overridden"] == [
        "compiled_device_exceeds_static_bound"
    ]


def test_unavailable_analysis_rejects_main_before_pair_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(runner, "git_text", lambda *args: "")
    report = {
        "available": False,
        "unavailable_reason": "injected unavailable analysis",
        "validation_status": "not_checked",
    }
    monkeypatch.setattr(runner, "analyze_compiled_pair_memory", lambda *args: report)

    def forbidden(*args, **kwargs):
        raise AssertionError("pair allocation, execution, or output entered")

    for name in (
        "normalized_device_metadata",
        "tiny_histogram",
        "ordered_pair_mass_jax",
        "checked_simulate_pair_density",
        "diagnostics_rows",
        "encode_bounded_metadata",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="compiled memory analysis is unavailable"):
        runner.main()


def test_injected_compiled_overage_is_rejected_before_pair_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))
    limit = runner.PHASE4_ABSOLUTE_LIMITS["combined_peak_bytes"]
    report = {
        "available": True,
        "compiled_device_requirement_bytes": 1,
        "compiled_plus_host_requirement_bytes": limit + 1,
    }
    monkeypatch.setattr(runner, "analyze_compiled_pair_memory", lambda *args: report)

    def forbidden(*args, **kwargs):
        raise AssertionError("pair allocation, execution, or output entered")

    for name in (
        "tiny_histogram",
        "ordered_pair_mass_jax",
        "checked_simulate_pair_density",
        "diagnostics_rows",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="before histogram, pair allocation, execution"):
        runner.main()


def test_allow_expensive_records_static_and_compiled_violations() -> None:
    raw = _default_raw()
    budget = runner.validate_pair_budget(
        raw,
        True,
        limits={name: 0 for name in runner.PHASE4_ABSOLUTE_LIMITS},
    )
    report = {
        "available": True,
        "compiled_device_requirement_bytes": 1,
        "compiled_plus_host_requirement_bytes": 1,
    }
    budget = runner.validate_compiled_pair_budget(budget, report, True)

    assert budget["static_violations"]
    assert budget["static_violations_overridden"] == budget["static_violations"]
    assert budget["compiled_violations"] == ["compiled_combined_peak_bytes"]
    assert budget["compiled_violations_overridden"] == budget["compiled_violations"]
    assert report["validation_status"] == "failed"


def test_compiled_device_requirement_must_not_exceed_static_device_bound() -> None:
    budget = runner.validate_pair_budget(_default_raw(), False)
    static = budget["static_estimates"]
    report = {
        "available": True,
        "compiled_device_requirement_bytes": static["static_device_bytes"] + 1,
        "compiled_plus_host_requirement_bytes": static[
            "static_combined_peak_bytes"
        ]
        + 1,
    }

    with pytest.raises(ValueError, match="compiled_device_exceeds_static_bound"):
        runner.validate_compiled_pair_budget(budget, report, False)


def test_allow_expensive_does_not_bypass_scientific_configuration_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["initial_condition"]["state_probabilities"] = [0.4, 0.4]
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=True),
    )
    monkeypatch.setattr(runner, "load_config", lambda path: deepcopy(config))

    with pytest.raises(ValueError, match="state probabilities"):
        runner.main()


def test_real_shape_only_compilation_is_below_new_static_bound_and_exposes_old_gap() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["grid"] = {"q_min": -1.0, "q_max": 1.0, "spacing": 0.25}
    raw = runner.inspect_raw_pair_config(config)
    estimate = runner.estimate_pair_resources(raw)
    grid = build_jax_pair_grid(QGrid(-1.0, 1.0, 0.25), jnp.float32)
    report = runner.analyze_compiled_pair_memory(
        raw, grid, LearningConfig(**config["model"])
    )

    assert report["available"] is True
    assert report["compiled_device_requirement_bytes"] <= estimate["static_device_bytes"]
    assert (
        report["compiled_plus_host_requirement_bytes"]
        <= estimate["static_combined_peak_bytes"]
    )
    old_device_formula = (
        3 * raw["state_expanded_cells"] * raw["item_bytes"]
        + estimate["components"]["point_working_bytes"]
        + estimate["components"]["branch_weight_bytes"]
        + estimate["components"]["branch_index_bytes"]
        + estimate["components"]["diagnostic_trajectory_bytes"]
    )
    assert report["compiled_device_requirement_bytes"] > old_device_formula


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="requires a fresh CPU+x64 process",
)
def test_real_float64_compiled_memory_is_below_static_bound() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config["grid"] = {"q_min": -1.0, "q_max": 1.0, "spacing": 0.25}
    config["solver"]["dtype"] = "float64"
    raw = runner.inspect_raw_pair_config(config)
    estimate = runner.estimate_pair_resources(raw)
    grid = build_jax_pair_grid(QGrid(-1.0, 1.0, 0.25), jnp.float64)
    report = runner.analyze_compiled_pair_memory(
        raw, grid, LearningConfig(**config["model"])
    )

    assert report["available"] is True
    assert report["compiled_device_requirement_bytes"] <= estimate["static_device_bytes"]
    assert report["compiled_plus_host_requirement_bytes"] <= estimate[
        "static_combined_peak_bytes"
    ]


def test_diagnostic_time_selection_includes_initial_stride_and_final() -> None:
    assert runner.selected_times(0, 3) == [0]
    assert runner.selected_times(6, 3) == [0, 3, 6]
    assert runner.selected_times(7, 3) == [0, 3, 6, 7]

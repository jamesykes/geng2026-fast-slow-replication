from __future__ import annotations

import argparse
import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from chu_pair.abm import NamedBinScheme, QBinSpec
from chu_pair.config import ABMConfig
from experiments import run_abm_baseline as baseline
from experiments import run_abm_uncertainty_diagnostic as runner


def _raw_fixture() -> dict:
    return {
        "schemes": [
            {"name": "coarse", "q_c_bins": 1, "q_d_bins": 1},
            {"name": "fine", "q_c_bins": 2, "q_d_bins": 2},
        ],
        "bootstrap": {
            "replicates": 5,
            "seed": 17,
            "stratum_chunk_size": 3,
            "confidence_level": 0.8,
        },
        "anchor_count": 2,
    }


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (
            "float32",
            {
                "configured_and_effective_edge_bytes": 120,
                "metadata_effective_edge_python_bytes": 400,
                "aggregation_observation_work_bytes": 3_576,
                "aggregation_peak_bytes": 22_366,
                "pooled_point_peak_bytes": 25_030,
                "reconstruction_peak_bytes": 24_166,
                "bootstrap_peak_bytes": 23_070,
                "output_peak_bytes": 18_174,
                "estimated_peak_bytes": 25_030,
            },
        ),
        (
            "float64",
            {
                "configured_and_effective_edge_bytes": 160,
                "metadata_effective_edge_python_bytes": 400,
                "aggregation_observation_work_bytes": 3_288,
                "aggregation_peak_bytes": 22_118,
                "pooled_point_peak_bytes": 25_070,
                "reconstruction_peak_bytes": 24_206,
                "bootstrap_peak_bytes": 23_110,
                "output_peak_bytes": 18_214,
                "estimated_peak_bytes": 25_070,
            },
        ),
    ],
)
def test_phase3b_resource_estimate_matches_independent_hand_calculation(
    dtype: str,
    expected: dict[str, int],
) -> None:
    abm = ABMConfig(num_agents=4, steps=3, num_runs=2, dtype=dtype)
    estimate = runner.estimate_phase3b_resources(abm, _raw_fixture())
    components = estimate["components"]

    assert estimate["observation_count"] == 24
    assert estimate["scheme_pooled_cells"] == [6, 24]
    assert estimate["scheme_per_run_strata"] == [12, 48]
    assert estimate["total_per_run_strata"] == 60
    assert estimate["pooled_output_rows"] == 30
    assert estimate["anchor_output_rows"] == 24
    assert estimate["bootstrap_weight_bytes"] == 40
    assert estimate["bootstrap_working_bytes"] == 5 * 3 * 280
    assert components["total_sequential_sufficient_bytes"] == 60 * 88
    assert components["peak_sequential_sufficient_bytes"] == 60 * 88
    assert components["reconstruction_working_bytes"] == 48 * 112
    assert components["reconstruction_peak_sufficient_and_work_bytes"] == (
        60 * 88 + 48 * 112
    )
    assert components["bootstrap_weight_generation_peak_bytes"] == 120
    assert components["bootstrap_weight_float64_conversion_bytes"] == 80
    assert components["bootstrap_weight_serialization_buffer_bytes"] == 40
    assert components["bootstrap_weight_processing_bytes"] == 120
    assert components["retained_summary_bytes"] == 30 * 445
    assert components["pooled_point_derivation_bytes"] == 24 * 260
    for name, value in expected.items():
        if name == "estimated_peak_bytes":
            assert estimate[name] == value
        else:
            assert components[name] == value


def test_zero_step_keeps_edges_weights_and_anchor_preflight_visible() -> None:
    abm = ABMConfig(num_agents=2, steps=0, num_runs=2, dtype="float32")
    estimate = runner.estimate_phase3b_resources(abm, _raw_fixture())

    assert estimate["observation_count"] == 0
    assert estimate["total_per_run_strata"] == 0
    assert estimate["pooled_output_rows"] == 0
    assert estimate["anchor_output_rows"] == 0
    assert estimate["components"]["configured_and_effective_edge_bytes"] == 120
    assert estimate["bootstrap_weight_bytes"] == 40
    assert estimate["components"]["metadata_effective_edge_python_bytes"] == 400
    assert estimate["estimated_peak_bytes"] == 600


@pytest.mark.parametrize(
    ("dtype", "expected_peak"),
    [("float32", 464), ("float64", 480)],
)
def test_zero_step_r32_direct_pool_peak_has_no_run_weight_temporaries(
    dtype: str,
    expected_peak: int,
) -> None:
    abm = ABMConfig(num_agents=2, steps=0, num_runs=32, dtype=dtype)
    raw = {
        "schemes": [{"name": "only", "q_c_bins": 1, "q_d_bins": 1}],
        "bootstrap": {
            "replicates": 1,
            "seed": 0,
            "stratum_chunk_size": 1,
            "confidence_level": 0.95,
        },
        "anchor_count": 0,
    }
    estimate = runner.estimate_phase3b_resources(abm, raw)

    assert estimate["components"]["pooled_point_run_weight_bytes"] == 0
    assert estimate["components"]["pooled_point_derivation_bytes"] == 0
    assert estimate["components"]["pooled_point_peak_bytes"] == (
        176 if dtype == "float32" else 192
    )
    assert estimate["estimated_peak_bytes"] == expected_peak

    limits = {
        name: int(estimate[name]) for name in runner.PHASE3B_ABSOLUTE_LIMITS
    }
    limits["estimated_peak_bytes"] = expected_peak - 1
    with pytest.raises(ValueError, match="estimated_peak_bytes"):
        runner.validate_phase3b_budget(abm, raw, False, limits=limits)


@pytest.mark.parametrize("resource_name", list(runner.PHASE3B_ABSOLUTE_LIMITS))
def test_every_phase3b_cap_is_enforced_independently(resource_name: str) -> None:
    abm = ABMConfig(num_agents=4, steps=3, num_runs=2, dtype="float32")
    raw = _raw_fixture()
    estimate = runner.estimate_phase3b_resources(abm, raw)
    limits = {
        name: int(estimate[name]) for name in runner.PHASE3B_ABSOLUTE_LIMITS
    }
    limits[resource_name] -= 1

    with pytest.raises(ValueError, match=resource_name):
        runner.validate_phase3b_budget(
            abm, raw, False, limits=limits
        )


def test_allow_expensive_is_the_only_recorded_cap_bypass() -> None:
    abm = ABMConfig(num_agents=4, steps=3, num_runs=2, dtype="float32")
    budget = runner.validate_phase3b_budget(
        abm,
        _raw_fixture(),
        True,
        limits={name: 0 for name in runner.PHASE3B_ABSOLUTE_LIMITS},
    )

    assert budget["allow_expensive"] is True
    assert budget["violations_overridden"] == list(runner.PHASE3B_ABSOLUTE_LIMITS)


def test_guard_precedes_qbinspec_and_constructed_counts_must_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = baseline.load_config(runner.DEFAULT_CONFIG)
    abm = ABMConfig(**config["simulation"])
    events: list[str] = []
    original_guard = runner.validate_phase3b_budget
    original_qbinspec = runner.QBinSpec

    def guarded(*args, **kwargs):
        events.append("guard")
        return original_guard(*args, **kwargs)

    def constructed(*args, **kwargs):
        events.append("QBinSpec")
        return original_qbinspec(*args, **kwargs)

    monkeypatch.setattr(runner, "validate_phase3b_budget", guarded)
    monkeypatch.setattr(runner, "QBinSpec", constructed)
    schemes, _, raw, budget = runner.construct_guarded_schemes(
        abm, config, False
    )

    assert events == ["guard", "QBinSpec", "QBinSpec"]
    assert [scheme.bins.bin_shape for scheme in schemes] == [
        (entry["q_c_bins"], entry["q_d_bins"]) for entry in raw["schemes"]
    ]
    assert budget["violations_overridden"] == []


def test_low_limit_rejection_precedes_qbinspec_with_small_raw_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = baseline.load_config(runner.DEFAULT_CONFIG)
    abm = ABMConfig(num_agents=2, steps=0, num_runs=1, dtype="float32")

    def forbidden(*args, **kwargs):
        raise AssertionError("QBinSpec constructed before Phase 3B rejection")

    monkeypatch.setattr(runner, "QBinSpec", forbidden)
    limits = dict(runner.PHASE3B_ABSOLUTE_LIMITS)
    limits["estimated_peak_bytes"] = 1
    with pytest.raises(ValueError, match="estimated_peak_bytes"):
        runner.construct_guarded_schemes(
            abm, config, False, limits=limits
        )


def test_main_rejects_before_qbinspec_simulation_aggregation_bootstrap_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = baseline.load_config(runner.DEFAULT_CONFIG)
    config["bin_schemes"][1]["q_c_edges"] = list(range(1_002))
    config["bin_schemes"][1]["q_d_edges"] = list(range(1_002))
    config["bin_schemes"][0]["q_c_edges"] = [0, 1_001]
    config["bin_schemes"][0]["q_d_edges"] = [0, 1_001]
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(baseline, "load_config", lambda path: config)
    monkeypatch.setattr(baseline, "validate_resource_budget", lambda *a, **k: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("expensive stage entered before Phase 3B rejection")

    for name in (
        "QBinSpec",
        "complete_graph",
        "simulate_instrumented_batch_jit",
        "aggregate_variance_records",
        "bootstrap_run_weights",
        "write_pooled_csv",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    with pytest.raises(ValueError, match="refusing expensive Phase 3B"):
        runner.main()


def test_runner_simulates_once_and_reuses_identical_bootstrap_weights(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "simulate": 0,
        "aggregate": 0,
        "record_ids": [],
        "weight_ids": [],
    }
    original_simulate = runner.simulate_instrumented_batch_jit
    original_aggregate = runner.aggregate_variance_records
    original_bootstrap = runner.cluster_bootstrap_intervals

    def simulate_once(*args, **kwargs):
        calls["simulate"] += 1
        return original_simulate(*args, **kwargs)

    def aggregate_each(*args, **kwargs):
        calls["aggregate"] += 1
        calls["record_ids"].append(id(args[0]))
        return original_aggregate(*args, **kwargs)

    def record_weights(statistics, weights, **kwargs):
        calls["weight_ids"].append(id(weights))
        return original_bootstrap(statistics, weights, **kwargs)

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "simulate_instrumented_batch_jit", simulate_once)
    monkeypatch.setattr(runner, "aggregate_variance_records", aggregate_each)
    monkeypatch.setattr(runner, "cluster_bootstrap_intervals", record_weights)
    monkeypatch.setattr(runner, "write_metadata", lambda *args, **kwargs: None)
    runner.main()

    assert calls["simulate"] == 1
    assert calls["aggregate"] == 2
    assert len(set(calls["record_ids"])) == 1
    assert len(calls["weight_ids"]) == 2
    assert len(set(calls["weight_ids"])) == 1


def test_anchor_csv_reports_configured_and_effective_widths(tmp_path) -> None:
    summary = SimpleNamespace(
        total_count=np.ones((1, 1, 1, 2), dtype=np.int64),
        contributing_runs=np.ones((1, 1, 1, 2), dtype=np.int64),
        bootstrap_replicates=3,
        point={name: np.zeros((1, 1, 1, 2)) for name in runner.BOOTSTRAP_ESTIMANDS},
        lower={name: np.zeros((1, 1, 1, 2)) for name in runner.BOOTSTRAP_ESTIMANDS},
        upper={name: np.ones((1, 1, 1, 2)) for name in runner.BOOTSTRAP_ESTIMANDS},
        valid_replicates={
            name: np.full((1, 1, 1, 2), 3, dtype=np.int32)
            for name in runner.BOOTSTRAP_ESTIMANDS
        },
        invalid_replicates={
            name: np.zeros((1, 1, 1, 2), dtype=np.int32)
            for name in runner.BOOTSTRAP_ESTIMANDS
        },
        interval_valid={
            name: np.ones((1, 1, 1, 2), dtype=bool)
            for name in runner.BOOTSTRAP_ESTIMANDS
        },
    )
    scheme = NamedBinScheme("only", QBinSpec([-0.1, 1.2], [-0.1, 1.2]))
    path = tmp_path / "anchors.csv"
    assert runner.write_anchor_csv(
        path, [(scheme, summary)], [(1.2, 1.2)], np.float32
    ) == 2
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["q_c_configured_width"] == "1.3"
    assert float(rows[0]["q_c_effective_width"]) == pytest.approx(
        float(np.float32(1.2) - np.float32(-0.1))
    )
    assert rows[0]["bootstrap_replicates"] == "3"

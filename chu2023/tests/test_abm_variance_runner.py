from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from chu_pair.abm import QBinSpec, aggregate_variance_records, derive_variance_moments
from chu_pair.config import ABMConfig
from experiments import run_abm_baseline as baseline
from experiments import run_abm_variance_diagnostic as runner


def test_diagnostic_csv_marks_empty_and_n2_undefined_values(tmp_path) -> None:
    q = np.asarray([[[[0.25, 0.25], [0.25, 0.75]]]], dtype=np.float32)
    actions = np.asarray([[[0, 1]]], dtype=np.int8)
    selected_q = np.asarray([[[0.25, 0.75]]])
    s1 = np.asarray([[[0.2, 0.8]]])
    rewards = s1.copy()  # N=1
    velocity = 0.4 * (rewards - selected_q)
    records = SimpleNamespace(
        q_t=q,
        actions_t=actions,
        selected_q_t=selected_q,
        rewards_t=rewards,
        selected_velocities_t=velocity,
        payoff_sums_t=s1,
        payoff_square_sums_t=s1**2,
    )
    bins = QBinSpec(np.asarray([-0.1, 0.5, 1.2]), np.asarray([-0.1, 1.2]))
    statistics = aggregate_variance_records(
        records, bins, num_agents=2, alpha=0.4, min_count=2
    )
    estimates = derive_variance_moments(statistics)

    output = tmp_path / "moments.csv"
    rows_written = runner.write_moment_csv(output, statistics, estimates)
    with output.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert rows_written == 4
    assert len(rows) == 4
    assert fieldnames is not None
    assert len(fieldnames) == len(set(fieldnames))
    populated = [row for row in rows if row["has_observations"] == "True"]
    empty = [row for row in rows if row["has_observations"] == "False"]
    assert len(populated) == 2
    assert len(empty) == 2
    assert all(row["underpopulated"] == "True" for row in populated)
    assert all(row["covariance"] == "" and row["m11"] == "" for row in populated)
    assert all(row["mean_reward"] != "" for row in populated)
    assert all(row["sum_s1"] == "" and row["mean_reward"] == "" for row in empty)
    assert rows[0]["q_c_configured_lower"] == "-0.1"
    assert rows[0]["q_c_effective_lower"] == str(float(np.float32(-0.1)))
    assert (
        rows[0]["q_c_configured_lower"]
        != rows[0]["q_c_effective_lower"]
    )


def test_metadata_records_corrected_float64_statistics_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = np.asarray([[[[0.25, 0.25], [0.75, 0.75]]]], dtype=np.float64)
    actions = np.asarray([[[0, 1]]], dtype=np.int8)
    selected_q = np.asarray([[[0.25, 0.75]]], dtype=np.float64)
    s1 = np.asarray([[[0.2, 0.8]]], dtype=np.float64)
    rewards = s1.copy()
    records = SimpleNamespace(
        q_t=q,
        actions_t=actions,
        selected_q_t=selected_q,
        rewards_t=rewards,
        selected_velocities_t=0.4 * (rewards - selected_q),
        payoff_sums_t=s1,
        payoff_square_sums_t=s1**2,
    )
    bins = QBinSpec([0.0, 1.0], [0.0, 1.0])
    statistics = aggregate_variance_records(
        records,
        bins,
        num_agents=2,
        alpha=0.4,
    )
    abm = ABMConfig(num_agents=2, steps=1, num_runs=1, dtype="float64")
    budget = runner.validate_statistics_budget(
        abm,
        q_c_bins=1,
        q_d_bins=1,
        allow_expensive=False,
    )

    @dataclass
    class Metadata:
        mode: str = "test"

    Records = namedtuple("Records", ["q_t"])
    initialization = SimpleNamespace(metadata=Metadata())
    result = SimpleNamespace(records=Records(q_t=q))
    monkeypatch.setattr(baseline, "implementation_source_hashes", lambda path: {})
    monkeypatch.setattr(baseline, "sha256", lambda path: "0" * 64)
    monkeypatch.setattr(baseline, "git_text", lambda *args: "test")
    monkeypatch.setattr(baseline, "device_metadata", lambda: [])
    output = tmp_path / "metadata.json"
    runner.write_metadata(
        output,
        config_path=tmp_path / "config.toml",
        config={"simulation": {"dtype": "float64"}},
        initialization=initialization,
        result=result,
        statistics=statistics,
        rows_written=2,
        resource_budget={"phase3a_statistics": budget},
    )

    with output.open() as file:
        metadata = json.load(file)
    recorded = metadata["resource_budget"]["phase3a_statistics"]["estimates"]
    assert recorded["components"]["aggregation_float64_conversion_bytes"] == 0
    assert recorded["components"]["aggregation_float64_product_bytes"] == 80
    assert recorded["components"]["aggregation_peak_bytes"] == 514
    assert recorded["estimated_peak_statistic_bytes"] == 584


def _small_resource_fixture() -> tuple[ABMConfig, QBinSpec]:
    abm = ABMConfig(num_agents=3, steps=4, num_runs=2, dtype="float32")
    bins = QBinSpec([0.0, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0])
    return abm, bins


def test_statistics_resource_estimate_matches_hand_calculation() -> None:
    abm, bins = _small_resource_fixture()
    estimate = runner.estimate_statistics_resources(
        abm,
        q_c_bins=bins.num_q_c_bins,
        q_d_bins=bins.num_q_d_bins,
    )

    assert estimate["observation_count"] == 24
    assert estimate["stratum_count"] == 96
    assert estimate["output_rows"] == 96
    assert estimate["components"] == {
        "configured_float64_edge_bytes": 56,
        "effective_observation_dtype_edge_bytes": 28,
        "sufficient_count_int64_bytes": 768,
        "sufficient_sum_float64_bytes": 7_680,
        "aggregation_observation_bytes": 696,
        "aggregation_index_intp_bytes": 960,
        "aggregation_float64_conversion_bytes": 960,
        "aggregation_float64_product_bytes": 960,
        "aggregation_value_float64_bytes": 1_920,
        "aggregation_peak_bytes": 12_108,
        "derived_retained_float64_and_bool_bytes": 11_136,
        "derived_intermediate_float64_bytes": 3_072,
        "derived_expression_workspace_float64_bytes": 2_304,
        "derivation_peak_bytes": 25_044,
        "csv_export_additional_stratum_scaled_bytes": 0,
    }
    assert estimate["estimated_peak_statistic_bytes"] == 25_044


def test_float64_statistics_resource_estimate_avoids_conversion_double_count() -> None:
    abm = ABMConfig(num_agents=3, steps=4, num_runs=2, dtype="float64")
    estimate = runner.estimate_statistics_resources(
        abm,
        q_c_bins=2,
        q_d_bins=3,
    )

    components = estimate["components"]
    assert components["configured_float64_edge_bytes"] == 56
    assert components["effective_observation_dtype_edge_bytes"] == 56
    assert components["sufficient_count_int64_bytes"] == 768
    assert components["sufficient_sum_float64_bytes"] == 7_680
    assert components["aggregation_observation_bytes"] == 1_368
    assert components["aggregation_index_intp_bytes"] == 960
    assert components["aggregation_float64_conversion_bytes"] == 0
    assert components["aggregation_float64_product_bytes"] == 960
    assert components["aggregation_value_float64_bytes"] == 960
    assert components["aggregation_peak_bytes"] == 11_848
    assert components["derivation_peak_bytes"] == 25_072
    assert estimate["estimated_peak_statistic_bytes"] == 25_072


@pytest.mark.parametrize(
    "resource_name",
    ["stratum_count", "estimated_peak_statistic_bytes", "output_rows"],
)
def test_each_statistics_resource_cap_is_enforced_independently(
    resource_name: str,
) -> None:
    abm, bins = _small_resource_fixture()
    estimate = runner.estimate_statistics_resources(
        abm,
        q_c_bins=bins.num_q_c_bins,
        q_d_bins=bins.num_q_d_bins,
    )
    limits = {
        name: int(estimate[name])
        for name in runner.PHASE3A_STATISTICS_ABSOLUTE_LIMITS
    }
    limits[resource_name] -= 1

    with pytest.raises(ValueError, match=resource_name):
        runner.validate_statistics_budget(
            abm,
            q_c_bins=bins.num_q_c_bins,
            q_d_bins=bins.num_q_d_bins,
            allow_expensive=False,
            limits=limits,
        )


def test_statistics_allow_expensive_bypasses_all_caps_without_allocation() -> None:
    abm, bins = _small_resource_fixture()
    budget = runner.validate_statistics_budget(
        abm,
        q_c_bins=bins.num_q_c_bins,
        q_d_bins=bins.num_q_d_bins,
        allow_expensive=True,
        limits={name: 0 for name in runner.PHASE3A_STATISTICS_ABSOLUTE_LIMITS},
    )

    assert budget["violations_overridden"] == list(
        runner.PHASE3A_STATISTICS_ABSOLUTE_LIMITS
    )
    assert budget["allow_expensive"] is True


def test_small_diagnostic_statistics_budget_is_accepted() -> None:
    config = baseline.load_config(runner.DEFAULT_CONFIG)
    abm = ABMConfig(**config["simulation"])
    bins = QBinSpec(
        config["bins"]["q_c_edges"],
        config["bins"]["q_d_edges"],
    )
    budget = runner.validate_statistics_budget(
        abm,
        q_c_bins=bins.num_q_c_bins,
        q_d_bins=bins.num_q_d_bins,
        allow_expensive=False,
    )

    assert budget["violations_overridden"] == []
    assert budget["estimates"]["stratum_count"] == 216
    assert budget["estimates"]["output_rows"] == 216


def test_zero_step_edge_memory_rejection_precedes_qbinspec_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abm = ABMConfig(num_agents=3, steps=0, num_runs=1, dtype="float32")
    bin_config = {
        "q_c_edges": [0.0, 1.0],
        "q_d_edges": [0.0, 1.0],
    }

    def forbidden_qbinspec(*args, **kwargs):
        raise AssertionError("QBinSpec allocation began before resource rejection")

    monkeypatch.setattr(runner, "QBinSpec", forbidden_qbinspec)
    limits = {
        "stratum_count": 0,
        "estimated_peak_statistic_bytes": 47,
        "output_rows": 0,
    }

    with pytest.raises(ValueError, match="estimated_peak_statistic_bytes"):
        runner.construct_guarded_bins(
            abm,
            bin_config,
            allow_expensive=False,
            limits=limits,
        )


def test_accepted_raw_counts_are_guarded_before_qbinspec_and_must_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abm = ABMConfig(num_agents=3, steps=1, num_runs=1, dtype="float32")
    bin_config = {
        "q_c_edges": [0.0, 0.5, 1.0],
        "q_d_edges": [0.0, 0.25, 0.5, 1.0],
    }
    events = []
    original_validate = runner.validate_statistics_budget
    original_qbinspec = runner.QBinSpec

    def guarded(*args, **kwargs):
        events.append("guard")
        return original_validate(*args, **kwargs)

    def constructed(*args, **kwargs):
        events.append("QBinSpec")
        return original_qbinspec(*args, **kwargs)

    monkeypatch.setattr(runner, "validate_statistics_budget", guarded)
    monkeypatch.setattr(runner, "QBinSpec", constructed)
    bins, budget = runner.construct_guarded_bins(
        abm,
        bin_config,
        allow_expensive=False,
    )

    assert events == ["guard", "QBinSpec"]
    assert runner.raw_bin_counts(bin_config) == (2, 3)
    assert bins.bin_shape == (2, 3)
    assert (
        budget["estimates"]["q_c_bins"],
        budget["estimates"]["q_d_bins"],
    ) == bins.bin_shape


def test_float64_near_limit_is_not_rejected_by_removed_conversion_double_count() -> None:
    abm = ABMConfig(num_agents=100, steps=1, num_runs=1, dtype="float64")
    estimate = runner.estimate_statistics_resources(
        abm,
        q_c_bins=1,
        q_d_bins=1,
    )
    old_double_counted_peak = estimate["components"]["aggregation_peak_bytes"]
    old_double_counted_peak += 40 * estimate["observation_count"]
    limit = 15_000

    assert estimate["components"]["aggregation_peak_bytes"] == 13_940
    assert estimate["estimated_peak_statistic_bytes"] == 13_940
    assert estimate["estimated_peak_statistic_bytes"] < limit
    assert old_double_counted_peak > limit
    budget = runner.validate_statistics_budget(
        abm,
        q_c_bins=1,
        q_d_bins=1,
        allow_expensive=False,
        limits={
            "stratum_count": 2,
            "estimated_peak_statistic_bytes": limit,
            "output_rows": 2,
        },
    )
    assert budget["violations_overridden"] == []


def test_large_bin_configuration_is_rejected_before_simulation_or_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(baseline.load_config(runner.DEFAULT_CONFIG))
    config["bins"]["q_c_edges"] = np.linspace(-0.1, 1.2, 1001).tolist()
    config["bins"]["q_d_edges"] = np.linspace(-0.1, 1.2, 1001).tolist()

    def forbidden(*args, **kwargs):
        raise AssertionError("simulation or statistic allocation began")

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(baseline, "load_config", lambda path: config)
    monkeypatch.setattr(runner, "QBinSpec", forbidden)
    monkeypatch.setattr(runner, "simulate_instrumented_batch_jit", forbidden)
    monkeypatch.setattr(runner, "aggregate_variance_records", forbidden)

    with pytest.raises(ValueError, match=r"before simulation or allocation") as error:
        runner.main()

    message = str(error.value)
    assert "Bc=1,000" in message
    assert "Bd=1,000" in message
    assert "R=3" in message
    assert "T=4" in message
    assert "--allow-expensive" in message


def test_phase3a_runner_explicitly_requests_instrumented_resource_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAfterResourceCheck(Exception):
        pass

    def capture_mode(abm, safety, allow_expensive, *, record_mode):
        assert record_mode == baseline.RESOURCE_MODE_INSTRUMENTED
        raise StopAfterResourceCheck

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(config=runner.DEFAULT_CONFIG, allow_expensive=False),
    )
    monkeypatch.setattr(baseline, "validate_resource_budget", capture_mode)

    with pytest.raises(StopAfterResourceCheck):
        runner.main()

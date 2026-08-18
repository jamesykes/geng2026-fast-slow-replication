from __future__ import annotations

import csv
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.abm import (
    ABMState,
    InitializedABM,
    InitializationMetadata,
    complete_graph,
    simulate_batch_jit,
)
from chu_pair.config import ABMConfig, LearningConfig
from chu_pair.grids import QGrid
from chu_pair.model import State
from chu_pair.policies import boltzmann_probabilities
from experiments import run_abm_baseline as runner


def permissive_declared_limits() -> dict[str, int]:
    return {
        config_key: 10**30
        for config_key in runner.SAFETY_CONFIG_KEYS.values()
    }


def test_declared_safety_limits_cannot_weaken_absolute_phase2_caps() -> None:
    abm = ABMConfig(num_agents=129, steps=0, num_runs=1)
    with pytest.raises(ValueError, match="num_agents=129 exceeds 128"):
        runner.validate_resource_budget(abm, permissive_declared_limits(), False)


def test_resource_budget_counts_zero_step_state_and_records_overrides() -> None:
    accepted = runner.validate_resource_budget(
        ABMConfig(num_agents=2, steps=0, num_runs=1),
        permissive_declared_limits(),
        False,
    )
    assert accepted["values"]["run_step_edges"] == 0
    assert accepted["values"]["record_bytes"] == 0
    assert accepted["values"]["state_working_bytes"] > 0

    overridden = runner.validate_resource_budget(
        ABMConfig(num_agents=129, steps=0, num_runs=1),
        permissive_declared_limits(),
        True,
    )
    assert overridden["allow_expensive"] is True
    assert overridden["violations_overridden"]


def test_default_resource_estimate_matches_committed_phase2_values() -> None:
    abm = ABMConfig(num_agents=3, steps=2, num_runs=4, dtype="float32")
    budget = runner.validate_resource_budget(
        abm,
        permissive_declared_limits(),
        False,
    )

    assert budget["record_mode"] == runner.RESOURCE_MODE_BASELINE
    assert budget["values"]["record_bytes"] == 1_016
    assert budget["values"]["state_working_bytes"] == 720
    assert sum(
        budget["record_layout"]["agent_float_fields_per_run_step"].values()
    ) == 8
    assert (
        budget["record_layout"]["committed_agent_float_allowance_per_run_step"]
        == 1
    )
    assert budget["record_layout"]["additional_record_fields"] == []
    assert budget["record_layout"]["additional_working_fields"] == []


def test_instrumented_resource_estimate_adds_only_phase3a_agent_fields() -> None:
    abm = ABMConfig(num_agents=3, steps=2, num_runs=4, dtype="float32")
    baseline = runner.validate_resource_budget(
        abm,
        permissive_declared_limits(),
        False,
    )
    instrumented = runner.validate_resource_budget(
        abm,
        permissive_declared_limits(),
        False,
        record_mode=runner.RESOURCE_MODE_INSTRUMENTED,
    )

    assert instrumented["values"]["record_bytes"] == 1_304
    assert instrumented["values"]["state_working_bytes"] == 816
    assert instrumented["values"]["record_bytes"] - baseline["values"][
        "record_bytes"
    ] == 4 * 2 * 3 * 3 * np.dtype(np.float32).itemsize
    assert instrumented["values"]["state_working_bytes"] - baseline["values"][
        "state_working_bytes"
    ] == 4 * 2 * 3 * np.dtype(np.float32).itemsize
    assert instrumented["record_layout"]["additional_record_fields"] == [
        "selected_q_t",
        "payoff_sums_t",
        "payoff_square_sums_t",
    ]
    assert instrumented["record_layout"]["agent_float_fields_per_run_step"][
        "selected_q_t"
    ] == 1


@pytest.mark.parametrize("record_mode", ["ambiguous", None, ["baseline"]])
def test_invalid_resource_mode_is_rejected(record_mode) -> None:
    with pytest.raises(ValueError, match="record_mode"):
        runner.validate_resource_budget(
            ABMConfig(num_agents=3, steps=2, num_runs=1),
            permissive_declared_limits(),
            False,
            record_mode=record_mode,
        )


def test_committed_phase2_sized_limit_still_accepts_baseline() -> None:
    abm = ABMConfig(num_agents=3, steps=2, num_runs=4, dtype="float32")
    safety = permissive_declared_limits()
    safety["max_record_bytes"] = 1_016
    safety["max_state_working_bytes"] = 720

    budget = runner.validate_resource_budget(abm, safety, False)

    assert budget["violations_overridden"] == []
    assert budget["record_mode"] == runner.RESOURCE_MODE_BASELINE


def test_zero_step_chronology_has_no_negative_final_step() -> None:
    label = runner.chronology_label(state_t=0, steps=0)
    assert label == "Q_0,S_0 initial/final state; no steps executed"
    assert "-1" not in label


def test_dirty_tree_provenance_hashes_include_phase2_implementation() -> None:
    hashes = runner.implementation_source_hashes(runner.DEFAULT_CONFIG)
    assert "src/chu_pair/abm/simulation.py" in hashes
    assert "src/chu_pair/model.py" in hashes
    assert "experiments/run_abm_baseline.py" in hashes
    assert "input_config" in hashes
    assert all(len(digest) == 64 for digest in hashes.values())


def test_dangerous_grid_is_rejected_before_histogram_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("histogram construction or sampling began")

    monkeypatch.setattr(runner, "seeded_legacy_histogram", forbidden_constructor)
    initial = {
        "q_min": -0.1,
        "q_max": 1.2,
        "spacing": 0.0001,
        "histogram_seed": 7,
        "samples_per_grid_cell": 10,
    }

    with pytest.raises(ValueError, match=r"before allocation or sampling") as error:
        runner.construct_grid_matched_histogram(initial, allow_expensive=False)

    message = str(error.value)
    assert "histogram_cells=169,026,001" in message
    assert "histogram_count_bytes=1,352,208,008" in message
    assert "histogram_sample_pairs=1,690,260,010" in message
    assert "spacing=0.0001" in message
    assert "--allow-expensive" in message


def test_histogram_estimator_matches_small_exact_allocation_and_loop() -> None:
    estimates = runner.estimate_legacy_histogram_resources(
        QGrid(q_min=-0.1, q_max=0.1, spacing=0.1),
        samples_per_grid_cell=4,
    )
    assert estimates["histogram_shape"] == [3, 3]
    assert estimates["histogram_cells"] == 9
    assert estimates["histogram_count_bytes"] == 9 * np.dtype(np.int64).itemsize
    assert estimates["histogram_sample_pairs"] == 36


@pytest.mark.parametrize(
    "resource_name",
    ["histogram_cells", "histogram_count_bytes", "histogram_sample_pairs"],
)
def test_each_histogram_resource_limit_is_enforced(resource_name: str) -> None:
    grid = QGrid(q_min=-0.1, q_max=0.1, spacing=0.1)
    estimates = runner.estimate_legacy_histogram_resources(grid, 4)
    limits = {
        name: int(estimates[name])
        for name in runner.PHASE2_HISTOGRAM_ABSOLUTE_LIMITS
    }
    limits[resource_name] -= 1

    with pytest.raises(ValueError, match=resource_name):
        runner.validate_legacy_histogram_budget(
            grid,
            4,
            allow_expensive=False,
            limits=limits,
        )


def test_allow_expensive_bypasses_guard_without_large_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls = []

    def fake_constructor(grid, **kwargs):
        calls.append((grid, kwargs))
        return sentinel

    monkeypatch.setattr(runner, "seeded_legacy_histogram", fake_constructor)
    initial = {
        "q_min": -0.1,
        "q_max": 1.2,
        "spacing": 0.0001,
        "histogram_seed": 7,
        "samples_per_grid_cell": 10,
    }
    histogram, budget = runner.construct_grid_matched_histogram(
        initial,
        allow_expensive=True,
    )

    assert histogram is sentinel
    assert len(calls) == 1
    assert budget["violations_overridden"] == list(
        runner.PHASE2_HISTOGRAM_ABSOLUTE_LIMITS
    )


def test_small_baseline_histogram_budget_is_accepted() -> None:
    initial = runner.load_config(runner.DEFAULT_CONFIG)["initial_condition"]
    grid = QGrid(
        q_min=float(initial["q_min"]),
        q_max=float(initial["q_max"]),
        spacing=float(initial["spacing"]),
    )
    budget = runner.validate_legacy_histogram_budget(
        grid,
        int(initial["samples_per_grid_cell"]),
        allow_expensive=False,
    )

    assert budget["violations_overridden"] == []
    assert budget["estimates"]["histogram_cells"] == 196
    assert budget["estimates"]["histogram_count_bytes"] == 1_568
    assert budget["estimates"]["histogram_sample_pairs"] == 1_960


def _run_timing_fixture(steps: int):
    graph = complete_graph(3)
    q_values = jnp.asarray(
        [[[0.23, 0.71], [0.37, 0.44], [0.61, 0.14]]],
        dtype=jnp.float32,
    )
    edge_states = jnp.full((1, graph.edge_count), int(State.PD), dtype=jnp.int8)
    initial_state = ABMState(q_values=q_values, edge_states=edge_states)
    keys = jnp.stack((jax.random.PRNGKey(8),))
    initialization = InitializedABM(
        state=initial_state,
        simulation_key=keys,
        metadata=InitializationMetadata(
            mode="runner_timing_test",
            num_agents=3,
            edge_count=graph.edge_count,
            abm_seed=8,
            dtype="float32",
            num_runs=1,
        ),
    )
    result = simulate_batch_jit(
        initial_state,
        keys,
        graph,
        alpha=0.4,
        tau=2.0,
        steps=steps,
    )
    return initialization, result


def _direct_state_observables(state, tau: float) -> tuple[np.ndarray, ...]:
    final_q = np.asarray(state.q_values)
    final_edges = np.asarray(state.edge_states)
    mean_q = final_q.mean(axis=(0, 1))
    mean_policy = boltzmann_probabilities(final_q, tau).mean(axis=(0, 1))
    state_proportions = np.asarray(
        [
            np.mean(final_edges == int(State.SH)),
            np.mean(final_edges == int(State.PD)),
        ]
    )
    return mean_q, mean_policy, state_proportions


def _write_timing_outputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    steps: int,
):
    initialization, result = _run_timing_fixture(steps)
    abm = ABMConfig(num_agents=3, steps=steps, num_runs=1, abm_seed=8)
    learning = LearningConfig(alpha=0.4, tau=2.0)
    run_directory = tmp_path / f"timing-{steps}"
    run_directory.mkdir()
    monkeypatch.setattr(runner, "implementation_source_hashes", lambda path: {})
    monkeypatch.setattr(runner, "git_text", lambda *args: "test")
    resource_budget = runner.validate_resource_budget(
        abm,
        permissive_declared_limits(),
        False,
    )
    runner.write_outputs(
        run_directory,
        runner.DEFAULT_CONFIG,
        {"test": {"steps": steps}},
        abm,
        learning,
        initialization,
        result,
        resource_budget,
    )
    with (run_directory / "state_trajectory.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    trajectory = runner.state_trajectory(result, initialization.state, learning.tau)
    return initialization, result, rows, trajectory


def _assert_reported_final_matches_state(result, row: dict, trajectory) -> None:
    expected_q, expected_policy, expected_states = _direct_state_observables(
        result.final_state,
        tau=2.0,
    )
    q, policies, proportions = trajectory
    np.testing.assert_allclose(q[:, -1].mean(axis=(0, 1)), expected_q, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        policies[:, -1].mean(axis=(0, 1)), expected_policy, rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        proportions[:, -1].mean(axis=0), expected_states, rtol=0, atol=1e-7
    )
    actual_row = np.asarray(
        [
            float(row["mean_q_c"]),
            float(row["mean_q_d"]),
            float(row["mean_policy_c"]),
            float(row["mean_policy_d"]),
            float(row["state_sh_proportion"]),
            float(row["state_pd_proportion"]),
        ]
    )
    np.testing.assert_allclose(
        actual_row,
        np.concatenate((expected_q, expected_policy, expected_states)),
        rtol=0,
        atol=1e-7,
    )


def test_zero_step_runner_reports_initialized_state_without_phantom_update(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialization, result, rows, trajectory = _write_timing_outputs(
        tmp_path,
        monkeypatch,
        steps=0,
    )

    np.testing.assert_array_equal(
        np.asarray(result.final_state.q_values),
        np.asarray(initialization.state.q_values),
    )
    np.testing.assert_array_equal(
        np.asarray(result.final_state.edge_states),
        np.asarray(initialization.state.edge_states),
    )
    np.testing.assert_array_equal(
        np.asarray(result.final_key),
        np.asarray(initialization.simulation_key),
    )
    assert result.records.actions_t.shape == (1, 0, 3)
    assert len(rows) == 1
    assert rows[0]["chronology"] == "Q_0,S_0 initial/final state; no steps executed"
    _assert_reported_final_matches_state(result, rows[0], trajectory)

    with (tmp_path / "timing-0" / "metadata.json").open() as file:
        metadata = json.load(file)
    recorded_budget = metadata["resource_budget"]
    assert recorded_budget["record_mode"] == runner.RESOURCE_MODE_BASELINE
    assert recorded_budget["record_layout"]["additional_record_fields"] == []
    assert "payoff_sums_t" not in metadata["array_shapes"]


def test_one_step_runner_reports_post_step_final_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialization, result, rows, trajectory = _write_timing_outputs(
        tmp_path,
        monkeypatch,
        steps=1,
    )

    initial_q, initial_policy, initial_states = _direct_state_observables(
        initialization.state,
        tau=2.0,
    )
    final_q, final_policy, final_states = _direct_state_observables(
        result.final_state,
        tau=2.0,
    )
    assert not np.allclose(final_q, initial_q)
    assert not np.allclose(final_policy, initial_policy)
    assert not np.allclose(final_states, initial_states)
    assert len(rows) == 2
    assert rows[-1]["chronology"] == "Q_1,S_1 after final step 0"
    _assert_reported_final_matches_state(result, rows[-1], trajectory)

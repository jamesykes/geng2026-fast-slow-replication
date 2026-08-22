from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments import run_full_grid_production as production

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = PROJECT_ROOT / "outputs" / "full_grid_production" / "production.toml"


def _tiny_config(**overrides):
    config = {
        "alpha": 0.4, "tau": 2.0, "num_agents": 4, "steps": 3, "num_runs": 4,
        "abm_seed": 20230819, "dtype": "float32",
        "q_min": -0.1, "q_max": 1.2, "spacing": 0.01, "grid_size": 131,
        "histogram_seed": 20230818, "samples_per_grid_cell": 10,
        "state_probabilities": [0.5, 0.5],
        "row_block_size": 256, "column_block_size": 256,
        "diagnostic_tolerance": 1e-4, "symmetry_tolerance": 1e-4,
        "source_times": [0, 1], "minimum_count": 2, "ratio_epsilon": 1e-15,
        "bin_schemes": [{"name": "coarse", "q_c_edges": [-0.1, 0.55, 1.2],
                         "q_d_edges": [-0.1, 0.55, 1.2]}],
        "bootstrap_replicates": 8, "bootstrap_seed": 31032023, "confidence_level": 0.95,
        "anchors": [[0.55, 0.55]], "allocator_policy": "fraction", "memory_fraction": 0.85,
        "contraction_precision": "highest", "hourly_price_usd": 3.29,
        "max_session_cost_usd": 100.0, "max_stage_seconds": 600,
        "safety_margin_fraction": 0.4, "run_name": "test",
        "normalized_sha256": "0" * 64,
    }
    config.update(overrides)
    return config


def test_run_keys_depend_only_on_global_index() -> None:
    """Chunk boundaries must not change any run's random stream."""

    import jax.numpy as jnp
    everything = production.production_run_keys(20230819, range(8))
    first_half = production.production_run_keys(20230819, range(0, 4))
    second_half = production.production_run_keys(20230819, range(4, 8))
    chunked = first_half + second_half
    for whole, part in zip(everything, chunked):
        assert jnp.array_equal(whole, part)
    # A different seed must give a different stream.
    other = production.production_run_keys(20230820, range(8))
    assert not jnp.array_equal(everything[0], other[0])


def _abm_statistics(config, chunk_size, tmp_path, *, resume=False, stop_after=None):
    import jax.numpy as jnp
    from chu_pair.grids import QGrid
    from chu_pair.initial_conditions import seeded_legacy_histogram
    from chu_pair.velocity_variance import QBinSpec

    grid = QGrid(-0.1, 1.2, 0.1)
    histogram = seeded_legacy_histogram(grid, seed=config["histogram_seed"],
                                        samples_per_grid_cell=config["samples_per_grid_cell"])
    bins = QBinSpec(np.asarray([-0.1, 0.55, 1.2]), np.asarray([-0.1, 0.55, 1.2]))
    calls = {"n": 0}

    def heartbeat(payload):
        calls["n"] += 1
        if stop_after is not None and calls["n"] >= stop_after:
            raise KeyboardInterrupt("simulated interruption")

    return production.run_abm_chunks(
        config, histogram, bins, chunk_size=chunk_size, checkpoint_dir=tmp_path,
        resume=resume, telemetry=[], heartbeat=heartbeat)


def test_chunked_abm_equals_unchunked(tmp_path) -> None:
    """Identical global run indices must give identical sufficient statistics."""

    config = _tiny_config()
    whole, _ = _abm_statistics(config, 4, tmp_path / "whole")
    chunked, _ = _abm_statistics(config, 2, tmp_path / "chunked")
    for name in ("counts", "sum_s1", "sum_s2", "sum_reward", "sum_velocity",
                 "sum_velocity_squared", "sum_selected_q", "sum_reward_selected_q"):
        np.testing.assert_array_equal(np.asarray(getattr(whole, name)),
                                      np.asarray(getattr(chunked, name)))
    assert np.asarray(whole.counts).shape[0] == config["num_runs"]


def test_interrupted_run_resumes_exactly(tmp_path) -> None:
    """A resumed experiment must equal an uninterrupted one, with no gaps."""

    config = _tiny_config()
    reference, _ = _abm_statistics(config, 2, tmp_path / "reference")
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    with pytest.raises(KeyboardInterrupt):
        _abm_statistics(config, 2, partial_dir, stop_after=1)
    manifest = json.loads((partial_dir / "checkpoint_manifest.json").read_text())
    assert len(manifest["chunks"]) == 1          # one chunk committed atomically
    resumed, resumed_manifest = _abm_statistics(config, 2, partial_dir, resume=True)
    assert len(resumed_manifest["chunks"]) == 2
    for name in ("counts", "sum_velocity", "sum_velocity_squared", "sum_reward"):
        np.testing.assert_array_equal(np.asarray(getattr(reference, name)),
                                      np.asarray(getattr(resumed, name)))


def test_resume_rejects_foreign_checkpoints(tmp_path) -> None:
    """A checkpoint from another configuration, commit or seed must fail closed."""

    config = _tiny_config()
    _abm_statistics(config, 2, tmp_path)
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["abm_seed"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        _abm_statistics(config, 2, tmp_path, resume=True)


def test_production_config_is_the_pre_registered_design() -> None:
    """The shipped production configuration must match the pre-registration."""

    config = production.load_production_config(PRODUCTION_CONFIG)
    assert config["grid_size"] == 131 and config["spacing"] == 0.01
    assert (config["q_min"], config["q_max"]) == (-0.1, 1.2)
    assert config["alpha"] == 0.4 and config["tau"] == 2.0
    assert config["num_agents"] == 128 and config["steps"] == 32 and config["num_runs"] == 512
    assert config["abm_seed"] == 20230819 and config["histogram_seed"] == 20230818
    assert config["source_times"] == [0, 1, 2, 4, 8, 16, 24, 31]
    assert config["bootstrap_replicates"] == 2000 and config["bootstrap_seed"] == 31032023
    assert config["confidence_level"] == 0.95
    assert config["dtype"] == "float32" and config["contraction_precision"] == "highest"
    assert config["allocator_policy"] == "fraction" and config["memory_fraction"] == 0.85
    assert config["max_session_cost_usd"] == 100.0 and config["hourly_price_usd"] == 3.29
    assert [s["name"] for s in config["bin_schemes"]] == ["coarse", "fine", "refined"]


def test_refined_scheme_is_exact_midpoint_subdivision() -> None:
    """The third scheme must subdivide every fine interval at its exact midpoint."""

    config = production.load_production_config(PRODUCTION_CONFIG)
    fine = [s for s in config["bin_schemes"] if s["name"] == "fine"][0]
    refined = [s for s in config["bin_schemes"] if s["name"] == "refined"][0]
    for key in ("q_c_edges", "q_d_edges"):
        expected = []
        for lower, upper in zip(fine[key], fine[key][1:]):
            expected += [lower, (lower + upper) / 2.0]
        expected.append(fine[key][-1])
        assert len(refined[key]) == 2 * (len(fine[key]) - 1) + 1
        np.testing.assert_allclose(refined[key], expected, rtol=0, atol=1e-12)


def test_configuration_rejects_designs_outside_production_limits(tmp_path) -> None:
    """Over-limit configurations must fail during normalization, before allocation."""

    text = PRODUCTION_CONFIG.read_text()
    cases = [
        ("num_runs = 512", "num_runs = 8192", "simulation.num_runs"),
        ("num_agents = 128", "num_agents = 4096", "simulation.num_agents"),
        ("spacing = 0.01", "spacing = 0.1", "G=131"),
        ('dtype = "float32"', 'dtype = "float64"', "float32"),
        ("replicates = 2000", "replicates = 999999", "bootstrap.replicates"),
        ('allocator_policy = "fraction"', 'allocator_policy = "default"', "fraction"),
        ("source_times = [0, 1, 2, 4, 8, 16, 24, 31]", "source_times = [0, 1, 99]", "source_times"),
    ]
    for old, new, message in cases:
        path = tmp_path / "bad.toml"
        path.write_text(text.replace(old, new))
        with pytest.raises(ValueError, match=message):
            production.load_production_config(path)


def test_resource_estimates_are_allocation_free_and_bounded(monkeypatch) -> None:
    """Estimates are Python integers computed without touching a device."""

    monkeypatch.setattr("chu_pair.grids.QGrid", lambda *a, **k: pytest.fail("QGrid constructed"))
    config = production.load_production_config(PRODUCTION_CONFIG)
    estimate = production.estimate_production_resources(config)
    assert estimate["violations"] == []
    assert estimate["grid_size"] == 131
    assert estimate["agent_grid_points"] == 17161
    assert estimate["state_expanded_cells"] == 2 * 17161 ** 2
    assert estimate["comparison_rows"] == (4 + 16 + 64) * 8 * 2
    assert all(isinstance(estimate[k], int) for k in
               ("one_density_bytes", "pair_summary_host_bytes", "comparison_rows"))


def test_execution_requires_the_exact_confirmation_phrase(tmp_path, monkeypatch) -> None:
    """The production phrase is mandatory and exact."""

    from types import SimpleNamespace
    for phrase in (None, "", "run exact g131 production variance", "ANALYZE EXACT G131 SEPARABLE"):
        monkeypatch.setattr(production, "parse_args", lambda phrase=phrase: SimpleNamespace(
            config=PRODUCTION_CONFIG, confirmation=phrase,
            doctor_report=tmp_path / "d.json", prerequisite=tmp_path / "p.json",
            hourly_price_usd=3.29, run_chunk_size=32, resume=False, calibration_runs=0,
            execute=True, dry_run=False))
        with pytest.raises(ValueError, match="confirmation phrase"):
            production.main()
    assert production.CONFIRMATION_PHRASE == "RUN EXACT G131 PRODUCTION VARIANCE"


def test_relative_ci_width_ignores_sparse_and_invalid_strata() -> None:
    """Precision is measured only over analysable strata with valid intervals."""

    rows = [
        {"sparse": False, "has_abm_observations": True, "direct_abm_velocity_variance": 1.0,
         "direct_abm_velocity_variance_lower": 0.9, "direct_abm_velocity_variance_upper": 1.1,
         "direct_abm_velocity_variance_interval_valid": True},
        {"sparse": True, "has_abm_observations": True, "direct_abm_velocity_variance": 1.0,
         "direct_abm_velocity_variance_lower": 0.0, "direct_abm_velocity_variance_upper": 9.0,
         "direct_abm_velocity_variance_interval_valid": True},
        {"sparse": False, "has_abm_observations": True, "direct_abm_velocity_variance": 1.0,
         "direct_abm_velocity_variance_lower": 0.0, "direct_abm_velocity_variance_upper": 9.0,
         "direct_abm_velocity_variance_interval_valid": False},
        {"sparse": False, "has_abm_observations": False, "direct_abm_velocity_variance": 1.0,
         "direct_abm_velocity_variance_lower": 0.0, "direct_abm_velocity_variance_upper": 9.0,
         "direct_abm_velocity_variance_interval_valid": True},
    ]
    median, count, widths = production._relative_ci_width(rows)
    assert count == 1 and median == pytest.approx(0.2) and widths == [pytest.approx(0.2)]


def test_pair_source_time_alignment_uses_requested_times() -> None:
    """Requested source times must map to strictly increasing scan slots."""

    from experiments import run_pair_separable_benchmark as benchmark
    config = production.load_production_config(PRODUCTION_CONFIG)
    slots = np.asarray(benchmark.source_slots(list(config["source_times"]), config["steps"]))
    # The scan writes summary slot k exactly at requested source time k and
    # nowhere else, so an ABM record at t is never paired with another P_t.
    for slot, source_time in enumerate(config["source_times"]):
        assert slots[source_time] == slot
    unused = [t for t in range(len(slots)) if t not in config["source_times"]]
    assert all(slots[t] == -1 for t in unused)
    assert max(config["source_times"]) < config["steps"]


def test_production_limit_admits_the_audited_r4096_extension() -> None:
    """The R guard is raised explicitly and audited, never bypassed."""

    assert production.PRODUCTION_LIMITS["max_runs"] == 4096
    text = (PROJECT_ROOT / "outputs" / "full_grid_production" / "production.toml").read_text()
    for runs, ok in ((512, True), (4096, True), (4097, False), (8192, False)):
        path = PROJECT_ROOT / "outputs" / "full_grid_production" / f".limit-{runs}.toml"
        path.write_text(text.replace("num_runs = 512", f"num_runs = {runs}"))
        try:
            if ok:
                config = production.load_production_config(path)
                assert config["num_runs"] == runs
                assert production.estimate_production_resources(config)["violations"] == []
            else:
                with pytest.raises(ValueError, match="simulation.num_runs"):
                    production.load_production_config(path)
        finally:
            path.unlink()


def test_r4096_resource_estimates_stay_inside_audited_limits() -> None:
    """Allocation-free formulas must bound the R=4096 lifetimes."""

    text = (PROJECT_ROOT / "outputs" / "full_grid_production" / "production.toml").read_text()
    path = PROJECT_ROOT / "outputs" / "full_grid_production" / ".r4096-estimate.toml"
    path.write_text(text.replace("num_runs = 512", "num_runs = 4096"))
    try:
        config = production.load_production_config(path)
        estimate = production.estimate_production_resources(config)
    finally:
        path.unlink()
    assert estimate["violations"] == []
    # per-run sufficient statistics scale linearly in R and stay bounded.
    assert estimate["per_run_sufficient_statistic_bytes"] == 4096 * 33 * 64 * 2 * 11 * 8
    assert (estimate["per_run_sufficient_statistic_bytes"]
            <= production.PRODUCTION_LIMITS["max_per_run_statistic_bytes"])
    assert estimate["bootstrap_weight_bytes"] == 2000 * 4096 * 4
    assert estimate["comparison_rows"] == (4 + 16 + 64) * 8 * 2
    assert estimate["grid_size"] == 131 and estimate["agent_grid_points"] == 17161


def test_convergence_prefixes_are_pre_registered_through_4096() -> None:
    """Prefixes are global run prefixes, capped at the total and at 4096."""

    import inspect
    source = inspect.getsource(production._run_count_convergence)
    assert "(32, 64, 128, 256, 512, 1024, 2048, 4096)" in source
    assert "8192" not in source


def test_heartbeat_covers_every_phase() -> None:
    """The stall that made a finished run look hung must not recur."""

    import inspect
    main_source = inspect.getsource(production.main)
    pair_source = inspect.getsource(production.run_pair_full_grid)
    convergence_source = inspect.getsource(production._run_count_convergence)
    combined = main_source + pair_source + convergence_source
    for phase in ("prerequisite_preparation", "pair_compilation", "pair_execution",
                  "aggregation", "bootstrap", "convergence_prefix",
                  "output_serialization", "final_verification", "complete"):
        assert f'"phase": "{phase}"' in combined, phase
    # every heartbeat payload carries time, phase, progress and a latest path
    assert '"latest_path"' in combined
    assert "elapsed_seconds" in inspect.getsource(production.main)


def test_heartbeat_writes_atomically_and_boundedly(tmp_path) -> None:
    """Heartbeats replace atomically and stay small."""

    path = tmp_path / "heartbeat.json"
    for index in range(3):
        production._atomic_json(path, {"phase": "abm", "completed": index, "total": 3})
        payload = json.loads(path.read_text())
        assert payload["completed"] == index
        assert path.stat().st_size < 4096
    assert [p.name for p in tmp_path.iterdir()] == ["heartbeat.json"]

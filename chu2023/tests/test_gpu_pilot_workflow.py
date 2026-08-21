from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import builtins
import os
import subprocess
import sys
import time

import pytest

from chu_pair.gpu_pilot.workflow import (
    FULL_ANALYSIS_CONFIRMATION,
    FULL_EXECUTION_CONFIRMATION,
    PilotStage,
    REVIEWED_GRID_SPECS,
    calculate_cost,
    estimate_stage_resources,
    expected_allocator_environment,
    executable_configuration_sha256,
    load_pilot_configuration,
    read_prerequisite_artifact,
    summarize_stage_cost,
    validate_stage_confirmation,
    validate_analyzed_signature_match,
    validate_allocator_identity,
    write_stage_artifact_atomic,
    stage_invariant_contract,
    stage_invariant_contract_sha256,
)
from chu_pair.grids import QGrid
from scripts import run_gpu_pilot_with_timeout as timeout_runner
from experiments import run_gpu_pilot as pilot_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "stage", "grids"),
    [
        ("gpu_pilot_small.toml", PilotStage.SMALL, (3, 5, 9)),
        ("gpu_pilot_medium.toml", PilotStage.MEDIUM, (17, 33)),
        ("gpu_pilot_large.toml", PilotStage.LARGE_PILOT, (65,)),
        ("gpu_pilot_full_grid_analyze.toml.disabled", PilotStage.FULL_GRID_ANALYZE, (131,)),
        ("gpu_pilot_full_grid_one_step.toml.disabled", PilotStage.FULL_GRID_ONE_STEP, (131,)),
    ],
)
def test_stage_configs_are_strict_and_resource_estimates_are_allocation_free(
    name, stage, grids, monkeypatch
) -> None:
    configuration = load_pilot_configuration(PROJECT_ROOT / "configs" / name)
    assert configuration.stage == stage
    assert configuration.grids == grids
    monkeypatch.setattr(
        "chu_pair.grids.QGrid", lambda *a, **k: pytest.fail("QGrid constructed")
    )
    estimate = estimate_stage_resources(configuration)
    assert [case["grid_size"] for case in estimate["cases"]] == list(grids)


def test_confirmation_phrases_are_exact_and_one_step_is_hard_limited() -> None:
    with pytest.raises(ValueError, match="exact confirmation"):
        validate_stage_confirmation(PilotStage.FULL_GRID_ANALYZE, "analyze")
    validate_stage_confirmation(PilotStage.FULL_GRID_ANALYZE, FULL_ANALYSIS_CONFIRMATION)
    with pytest.raises(ValueError, match="exact confirmation"):
        validate_stage_confirmation(PilotStage.FULL_GRID_ONE_STEP, FULL_ANALYSIS_CONFIRMATION)
    validate_stage_confirmation(PilotStage.FULL_GRID_ONE_STEP, FULL_EXECUTION_CONFIRMATION)
    configuration = load_pilot_configuration(
        PROJECT_ROOT / "configs/gpu_pilot_full_grid_one_step.toml.disabled"
    )
    assert configuration.steps == 1
    analysis = load_pilot_configuration(
        PROJECT_ROOT / "configs/gpu_pilot_full_grid_analyze.toml.disabled"
    )
    assert executable_configuration_sha256(analysis) == executable_configuration_sha256(
        configuration
    )


def test_full_grid_template_cannot_raise_step_count_or_define_caps(tmp_path) -> None:
    source = (
        PROJECT_ROOT / "configs/gpu_pilot_full_grid_one_step.toml.disabled"
    ).read_text(encoding="utf-8")
    too_many_steps = tmp_path / "steps.toml"
    too_many_steps.write_text(source.replace("steps = 1", "steps = 2"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one step"):
        load_pilot_configuration(too_many_steps)
    extra_caps = tmp_path / "caps.toml"
    extra_caps.write_text(source + "\n[safety]\ngrid_size = 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sections"):
        load_pilot_configuration(extra_caps)


def test_every_pilot_grid_is_legacy_aligned_and_contains_the_payoff_range() -> None:
    for expected_size, (q_min, q_max, spacing) in REVIEWED_GRID_SPECS.items():
        grid = QGrid(q_min, q_max, spacing)
        assert grid.size == expected_size
        assert grid.q_min <= -0.1
        assert grid.q_max >= 1.2
    assert REVIEWED_GRID_SPECS[131] == (-0.1, 1.2, 0.01)


def test_float64_resource_plan_uses_eight_byte_pair_scalars() -> None:
    float32 = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_medium.toml")
    estimate32 = estimate_stage_resources(float32)
    estimate64 = estimate_stage_resources(replace(float32, dtype="float64"))
    assert estimate32["cases"][0]["dtype_bytes"] == 4
    assert estimate64["cases"][0]["dtype_bytes"] == 8
    assert estimate64["cases"][0]["one_density_bytes"] == 2 * estimate32["cases"][0]["one_density_bytes"]


def test_cost_is_elapsed_time_times_explicit_price_only() -> None:
    result = calculate_cost(90.0, 4.0)
    assert result["estimated_compute_cost_usd"] == pytest.approx(0.1)
    assert "not billing data" in result["claim_scope"]
    cumulative = summarize_stage_cost(
        elapsed_seconds=90.0, hourly_price_usd=4.0,
        prior_cumulative_usd=0.25, session_budget_usd=0.5,
        next_stage_max_seconds=180,
    )
    assert cumulative["cumulative_stage_estimate_usd"] == pytest.approx(0.35)
    assert cumulative["projected_next_stage_maximum_usd"] == pytest.approx(0.2)
    assert cumulative["next_stage_would_exceed_session_budget"] is True


def test_small_stage_contains_one_and_multiple_step_exact_objects() -> None:
    configuration = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_small.toml")
    variants = pilot_runner._stage_variants(PilotStage.SMALL, configuration)
    assert [(variant.steps, variant.source_times) for variant in variants] == [
        (1, (0, 1)), (2, (0, 1, 2))
    ]
    assert pilot_runner._stage_variants(PilotStage.MEDIUM, configuration) == (configuration,)


def test_altered_full_analysis_executable_signature_is_rejected() -> None:
    valid = "a" * 64
    validate_analyzed_signature_match(valid, valid)
    with pytest.raises(ValueError, match="differs"):
        validate_analyzed_signature_match(valid, "b" * 64)
    with pytest.raises(ValueError, match="differs"):
        validate_analyzed_signature_match("short", valid)


def test_allocator_identity_is_part_of_the_non_overridable_contract() -> None:
    configuration = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_small.toml")
    expected = expected_allocator_environment(configuration)
    validate_allocator_identity(configuration, expected)
    with pytest.raises(ValueError, match="allocator identity"):
        validate_allocator_identity(
            configuration, {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        )


def _success_payload(*, stage: PilotStage, created: datetime) -> dict:
    return {
        "schema_version": 1, "stage": stage.value, "status": "success",
        "completed_utc": created.isoformat(), "git_commit": "abc",
        "environment_sha256": "env",
    }


def test_atomic_artifact_resume_checks_digest_provenance_and_freshness(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    path = write_stage_artifact_atomic(
        tmp_path / "run", _success_payload(stage=PilotStage.SMALL, created=now)
    )
    accepted = read_prerequisite_artifact(
        path, required_stage=PilotStage.SMALL, commit="abc",
        environment_sha256="env", now=now,
    )
    assert accepted["status"] == "success"
    document = json.loads(path.read_text(encoding="ascii"))
    document["status"] = "failed"
    path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(ValueError, match="digest"):
        read_prerequisite_artifact(
            path, required_stage=PilotStage.SMALL, commit="abc",
            environment_sha256="env", now=now,
        )


def test_stale_and_incompatible_prerequisite_artifacts_fail(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    path = write_stage_artifact_atomic(
        tmp_path / "run",
        _success_payload(stage=PilotStage.SMALL, created=now - timedelta(hours=7)),
    )
    with pytest.raises(ValueError, match="stale"):
        read_prerequisite_artifact(
            path, required_stage=PilotStage.SMALL, commit="abc",
            environment_sha256="env", now=now,
        )
    fresh = write_stage_artifact_atomic(
        tmp_path / "fresh", _success_payload(stage=PilotStage.SMALL, created=now)
    )
    with pytest.raises(ValueError, match="provenance"):
        read_prerequisite_artifact(
            fresh, required_stage=PilotStage.SMALL, commit="different",
            environment_sha256="env", now=now,
        )


def test_timeout_wrapper_returns_124_and_warns_capacity_is_stale(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        timeout_runner, "parse_args",
        lambda: SimpleNamespace(timeout_seconds=10, command=["tool"]),
    )
    class TimedOut:
        pid = 123
        def communicate(self, timeout=None):
            if timeout == 10:
                raise timeout_runner.subprocess.TimeoutExpired("tool", 10)
            return b"", b""
    monkeypatch.setattr(timeout_runner.subprocess, "Popen", lambda *a, **k: TimedOut())
    monkeypatch.setattr(timeout_runner.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(timeout_runner.os, "killpg", lambda *a: None)
    assert timeout_runner.main() == 124
    assert "capacity evidence is now stale" in capsys.readouterr().out


def test_full_grid_dry_run_never_imports_numerical_runtime(monkeypatch, capsys) -> None:
    config = PROJECT_ROOT / "configs/gpu_pilot_full_grid_analyze.toml.disabled"
    args = SimpleNamespace(
        stage="full-grid-analyze", config=config, doctor_report=Path("doctor.json"),
        prerequisite=None, confirmation=FULL_ANALYSIS_CONFIRMATION,
        hourly_price_usd=None, allow_expensive=False, enable_g97=False,
        execute=False, dry_run=True,
    )
    doctor = {
        "schema_version": 1, "gpu_ready": False, "environment_sha256": "env",
        "cuda_family": "cuda12", "git": {"commit": "abc"},
    }
    monkeypatch.setattr(pilot_runner, "parse_args", lambda: args)
    monkeypatch.setattr(pilot_runner, "_read_doctor", lambda path: doctor)
    monkeypatch.setattr(pilot_runner, "_git", lambda *a: "abc" if a[0] == "rev-parse" else "dirty")
    monkeypatch.setattr(pilot_runner, "apply_allocator_policy", lambda *a, **k: {})
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "chu_pair.gpu_pilot.runtime":
            pytest.fail("full-grid dry run imported numerical runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert pilot_runner.main() == 0
    assert '"would_compile": false' in capsys.readouterr().out


def test_execute_requires_explicit_price_acknowledgement_before_runtime(monkeypatch) -> None:
    config = PROJECT_ROOT / "configs/gpu_pilot_small.toml"
    args = SimpleNamespace(
        stage="small", config=config, doctor_report=Path("doctor.json"),
        prerequisite=Path("prior.json"), confirmation=None,
        hourly_price_usd=1.0, allow_expensive=False, enable_g97=False,
        execute=True, dry_run=False,
    )
    doctor = {
        "schema_version": 1, "gpu_ready": True, "environment_sha256": "env",
        "cuda_family": "cuda12", "git": {"commit": "abc"},
        "host_memory": {"available_bytes": 10**15},
    }
    monkeypatch.setattr(pilot_runner, "parse_args", lambda: args)
    monkeypatch.setattr(pilot_runner, "_read_doctor", lambda path: doctor)
    monkeypatch.setattr(pilot_runner, "_git", lambda *a: "abc" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(pilot_runner, "apply_allocator_policy", lambda *a, **k: {})
    with pytest.raises(ValueError, match="positive user-supplied"):
        pilot_runner.main()


def test_entrypoint_atomically_records_a_preflight_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pilot_runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(pilot_runner, "_ARTIFACT_WRITTEN", False)
    monkeypatch.setattr(
        pilot_runner, "main", lambda: (_ for _ in ()).throw(ValueError("rejected"))
    )
    monkeypatch.setattr(pilot_runner, "_git", lambda *a: "abc" if a[0] == "rev-parse" else "")
    with pytest.raises(ValueError, match="rejected"):
        pilot_runner.entrypoint()
    artifacts = list(tmp_path.glob("rejected-*/stage.json"))
    assert len(artifacts) == 1
    document = json.loads(artifacts[0].read_text(encoding="ascii"))
    assert document["status"] == "failed"
    assert document["error"] == {"type": "ValueError", "message": "rejected"}


def test_stage_contract_rejects_each_invariant_mutation_and_allows_stage_fields() -> None:
    configuration = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_small.toml")
    hashes = {
        "src/chu_pair/model.py": "a" * 64,
        "src/chu_pair/initial_conditions.py": "b" * 64,
        "src/chu_pair/pair_density/jax_solver.py": "c" * 64,
    }
    base = stage_invariant_contract(configuration, commit="d" * 40, environment_sha256="e" * 64, source_hashes=hashes)
    digest = stage_invariant_contract_sha256(base)
    for replacement in (
        replace(configuration, alpha=0.3), replace(configuration, tau=2.0),
        replace(configuration, dtype="float64"), replace(configuration, allocator_policy="default"),
        replace(configuration, histogram_seed=1), replace(configuration, samples_per_grid_cell=9),
        replace(configuration, state_probabilities=(0.4, 0.6)),
        replace(configuration, diagnostic_tolerance=2e-4),
    ):
        assert stage_invariant_contract_sha256(stage_invariant_contract(replacement, commit="d" * 40, environment_sha256="e" * 64, source_hashes=hashes)) != digest
    assert stage_invariant_contract_sha256(stage_invariant_contract(configuration, commit="f" * 40, environment_sha256="e" * 64, source_hashes=hashes)) != digest
    changed_hashes = {**hashes, "src/chu_pair/model.py": "f" * 64}
    assert stage_invariant_contract_sha256(stage_invariant_contract(configuration, commit="d" * 40, environment_sha256="e" * 64, source_hashes=changed_hashes)) != digest
    for field, value in (
        ("environment_sha256", "f" * 64), ("action_order", ("D", "C")),
        ("state_order", ("PD", "SH")), ("initialization_law", "other"),
        ("hourly_price_usd", 1.0), ("max_session_cost_usd", 201.0),
    ):
        changed = dict(base); changed[field] = value
        assert stage_invariant_contract_sha256(changed) != digest
    # Grid/horizon/block/source-slot fields are deliberately absent: distinct
    # reviewed stages therefore retain the same invariant digest.
    medium = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_medium.toml")
    assert stage_invariant_contract_sha256(stage_invariant_contract(medium, commit="d" * 40, environment_sha256="e" * 64, source_hashes=hashes)) == digest


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_wrapper_kills_forked_descendant_and_writes_failure_artifact(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "descendant.pid"
    artifact = tmp_path / "outputs" / "gpu_pilot" / "timeout"
    artifact.mkdir(parents=True)
    (artifact / "stage.json").write_text('{"status":"success"}')
    outside = tmp_path / "success.json"; outside.write_text('{"status":"success"}')
    code = (
        "import os,time,pathlib; p=os.fork(); "
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid() if p==0 else p)); "
        "time.sleep(30)"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(timeout_runner, "parse_args", lambda: SimpleNamespace(
        timeout_seconds=1, command=[sys.executable, "-c", code], artifact_dir=artifact, stage="small",
    ))
    assert timeout_runner.main() == 124
    payload = json.loads((artifact / "timeout-failure.json").read_text())
    assert payload["status"] == "failed" and payload["capacity_evidence_invalidated"] is True
    assert not (artifact / "stage.json").exists()
    assert outside.exists()
    pid = int(marker.read_text())
    time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_direct_path_invocation_lets_runtime_import_its_benchmark_dependency() -> None:
    """Regression: ``python experiments/run_gpu_pilot.py`` must reach the runtime.

    Executing the runner by path puts ``experiments/`` on ``sys.path[0]`` and
    leaves the project root off the path entirely, so the top-level
    ``from experiments import run_pair_separable_benchmark`` in
    ``chu_pair/gpu_pilot/runtime.py`` previously raised ``ModuleNotFoundError``
    only under the documented command. This reproduces that exact import
    context in a subprocess and requires the runtime to import.
    """

    probe = (
        "import json, pathlib, runpy, sys\n"
        "root = pathlib.Path.cwd().resolve()\n"
        "scripts_dir = root / 'experiments'\n"
        # Reproduce `python experiments/run_gpu_pilot.py`: the script directory
        # leads sys.path and neither '' nor the project root appears on it.
        "sys.path[:] = [str(scripts_dir)] + [\n"
        "    p for p in sys.path[1:] if p not in ('', '.', str(root), str(scripts_dir))\n"
        "]\n"
        # The emulation is only meaningful if `experiments` is unavailable now.
        "import importlib.util\n"
        "assert importlib.util.find_spec('experiments') is None, 'path emulation failed'\n"
        "runpy.run_path(str(scripts_dir / 'run_gpu_pilot.py'), run_name='pilot_under_test')\n"
        "import chu_pair.gpu_pilot.runtime as runtime\n"
        "print(json.dumps({'benchmark': runtime.benchmark.__file__}))\n"
    )
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    environment["JAX_PLATFORM_NAME"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    resolved = json.loads(completed.stdout.strip().splitlines()[-1])["benchmark"]
    assert Path(resolved) == PROJECT_ROOT / "experiments" / "run_pair_separable_benchmark.py"


@pytest.mark.parametrize(
    ("name", "stage"),
    [
        ("gpu_pilot_small.toml", PilotStage.SMALL),
        ("gpu_pilot_medium.toml", PilotStage.MEDIUM),
        ("gpu_pilot_large.toml", PilotStage.LARGE_PILOT),
        ("gpu_pilot_full_grid_one_step.toml.disabled", PilotStage.FULL_GRID_ONE_STEP),
    ],
)
def test_loaded_configuration_keeps_a_real_stage_enum_for_the_runtime(name, stage) -> None:
    """Regression: the loader must not downgrade ``stage`` to a bare ``str``.

    ``PilotStage`` subclasses ``str``, so every ``==`` comparison tolerated the
    downgrade and only the runtime's ``configuration.stage.value`` accesses
    failed, after the live GPU revalidation. This reaches the host-only grid
    description with a configuration from the real loader.
    """

    from chu_pair.gpu_pilot import runtime

    configuration = load_pilot_configuration(PROJECT_ROOT / "configs" / name)
    assert configuration.stage is stage
    assert isinstance(configuration.stage, PilotStage)
    assert configuration.stage.value == stage.value

    described = runtime._case(configuration, grid_size=configuration.grids[0])
    assert described["label"] == f"gpu-{stage.value}-g{configuration.grids[0]}"
    # The two remaining runtime accesses must resolve for this stage as well.
    assert (("flat", "separable") if configuration.stage.value == "small" else ("separable",))
    if stage is not PilotStage.FULL_GRID_ANALYZE:
        assert runtime._REPETITIONS[configuration.stage.value] >= 1


def test_normalized_stage_serializes_as_the_same_external_string() -> None:
    """The enum repair must not move any digest or serialized contract."""

    configuration = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_small.toml")
    payload = asdict(configuration)
    assert payload["stage"] is PilotStage.SMALL
    assert json.dumps({"stage": payload["stage"]}) == '{"stage": "small"}'
    assert json.loads(json.dumps(payload, sort_keys=True))["stage"] == "small"
    # Pinned digests. These moved exactly once, when the explicit float32
    # contraction-precision policy joined the normalized configuration and the
    # executable contract; that change must invalidate older provenance.
    assert configuration.normalized_sha256 == (
        "6d1eb77504f4b56f7c6041b427fe56b61d08390f0bd051e283af93ae1da9b9c7"
    )
    assert executable_configuration_sha256(configuration) == (
        "adff57d9b6ab1267c72b2b25eb7f430d2392229a0346a0d0511e602c52671c83"
    )
    reloaded = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_small.toml")
    assert reloaded.normalized_sha256 == configuration.normalized_sha256
    assert executable_configuration_sha256(reloaded) == executable_configuration_sha256(configuration)


def test_contraction_precision_is_fixed_and_reaches_every_provenance_record() -> None:
    """The explicit precision policy is part of scientific/executable identity."""

    from chu_pair.config import PAIR_CONTRACTION_PRECISION
    from chu_pair.pair_density import pair_contraction_precision

    assert PAIR_CONTRACTION_PRECISION == "highest"
    assert pair_contraction_precision() == PAIR_CONTRACTION_PRECISION

    configuration = load_pilot_configuration(PROJECT_ROOT / "configs/gpu_pilot_small.toml")
    assert configuration.contraction_precision == "highest"

    hashes = {
        name: "a" * 64 for name in (
            "src/chu_pair/model.py", "src/chu_pair/initial_conditions.py",
            "src/chu_pair/pair_density/jax_solver.py",
        )
    }
    contract = stage_invariant_contract(
        configuration, commit="c" * 40, environment_sha256="e" * 64, source_hashes=hashes,
    )
    assert contract["contraction_precision"] == "highest"

    # A different policy must move the normalized, executable and contract
    # digests so incompatible prerequisites cannot be reused.
    other = replace(configuration, contraction_precision="default")
    assert executable_configuration_sha256(other) != executable_configuration_sha256(configuration)
    assert stage_invariant_contract_sha256(
        stage_invariant_contract(
            other, commit="c" * 40, environment_sha256="e" * 64, source_hashes=hashes,
        )
    ) != stage_invariant_contract_sha256(contract)


def test_configuration_parsing_stays_free_of_jax() -> None:
    """The precision policy must not drag JAX into pre-allocator config parsing."""

    probe = (
        "import sys, json\n"
        "from pathlib import Path\n"
        "from chu_pair.gpu_pilot.workflow import load_pilot_configuration\n"
        "c = load_pilot_configuration(Path('configs/gpu_pilot_small.toml'))\n"
        "print(json.dumps({'precision': c.contraction_precision,\n"
        "                  'jax_imported': 'jax' in sys.modules}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["precision"] == "highest"
    assert result["jax_imported"] is False

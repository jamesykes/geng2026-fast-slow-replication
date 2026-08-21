#!/usr/bin/env python3
"""Run one explicitly staged, analyzed exact-separable GPU pilot operation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chu_pair.gpu_pilot.allocator import apply_allocator_policy
from chu_pair.gpu_pilot.workflow import (
    PilotStage,
    executable_configuration_sha256,
    estimate_stage_resources,
    load_pilot_configuration,
    read_prerequisite_artifact,
    stage_invariant_contract,
    stage_invariant_contract_sha256,
    summarize_stage_cost,
    validate_stage_confirmation,
    write_stage_artifact_atomic,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gpu_pilot"
_ARTIFACT_WRITTEN = False
_PREVIOUS = {
    PilotStage.SMALL: PilotStage.DOCTOR,
    PilotStage.MEDIUM: PilotStage.SMALL,
    PilotStage.LARGE_PILOT: PilotStage.MEDIUM,
    PilotStage.FULL_GRID_ANALYZE: PilotStage.LARGE_PILOT,
    PilotStage.FULL_GRID_ONE_STEP: PilotStage.FULL_GRID_ANALYZE,
}
_NEXT_STAGE_MAX_SECONDS = {
    PilotStage.SMALL: 1800,
    PilotStage.MEDIUM: 3600,
    PilotStage.LARGE_PILOT: 7200,
    PilotStage.FULL_GRID_ANALYZE: 7200,
    PilotStage.FULL_GRID_ONE_STEP: 0,
}


def _maximum_stage_charge(*, prior: float, price: float, seconds: int, budget: float) -> tuple[Decimal, Decimal]:
    """Exact decimal preflight for the configured worst-case stage charge."""

    try:
        prior_d, price_d, budget_d = (Decimal(str(value)) for value in (prior, price, budget))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("cost preflight values are invalid") from error
    if any(not value.is_finite() or value < 0 for value in (prior_d, price_d, budget_d)):
        raise ValueError("cost preflight values must be finite and non-negative")
    maximum = prior_d + price_d * Decimal(seconds) / Decimal(3600)
    return maximum, budget_d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(stage.value for stage in PilotStage), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--doctor-report", type=Path, required=True)
    parser.add_argument("--prerequisite", type=Path)
    parser.add_argument("--confirmation")
    parser.add_argument(
        "--hourly-price-usd", type=float,
        help="required execution acknowledgement; must exactly match the normalized config",
    )
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--enable-g97", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()


def _read_doctor(path: Path) -> dict[str, object]:
    if path.stat().st_size > 128 * 1024:
        raise ValueError("doctor report exceeds its byte bound")
    document = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("doctor report schema is invalid")
    from chu_pair.gpu_pilot.doctor import environment_sha256

    if document.get("environment_sha256") != environment_sha256(document):
        raise ValueError("doctor environment digest is invalid")
    return document


def _stage_variants(stage: PilotStage, configuration):
    if stage == PilotStage.SMALL:
        return (
            replace(configuration, steps=1, source_times=(0, 1)),
            configuration,
        )
    return (configuration,)


def main() -> int:
    global _ARTIFACT_WRITTEN
    args = parse_args()
    stage = PilotStage(args.stage)
    doctor = _read_doctor(args.doctor_report.resolve())
    if stage == PilotStage.DOCTOR:
        if args.config is not None or args.prerequisite is not None:
            raise ValueError("doctor stage does not accept config or prerequisite")
        if args.execute and not doctor.get("gpu_ready"):
            raise ValueError("doctor report did not establish a ready GPU environment")
        if args.execute and (
            doctor.get("git", {}).get("commit") != _git("rev-parse", "HEAD")
            or bool(_git("status", "--porcelain", "--", "."))
        ):
            raise ValueError("doctor stage requires the same clean reviewed commit")
        payload = {"stage": stage.value, "dry_run": args.dry_run, "doctor": doctor}
        if args.execute:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            directory = (OUTPUT_ROOT / f"gpu-doctor-{timestamp}").resolve()
            payload.update(
                schema_version=1, status="success",
                completed_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=doctor["git"]["commit"],
                environment_sha256=doctor["environment_sha256"],
            )
            artifact_path = write_stage_artifact_atomic(directory, payload)
            _ARTIFACT_WRITTEN = True
            print(f"stage artifact: {artifact_path}", file=sys.stderr)
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.config is None:
        raise ValueError("numerical stages require --config")
    configuration = load_pilot_configuration(args.config.resolve())
    if configuration.stage != stage:
        raise ValueError("CLI stage and configuration stage disagree")
    allocator_fraction = (
        configuration.memory_fraction if configuration.allocator_policy == "fraction" else None
    )
    apply_allocator_policy(
        configuration.allocator_policy, memory_fraction=allocator_fraction,
    )
    validate_stage_confirmation(stage, args.confirmation)
    if configuration.include_g97 and not args.enable_g97:
        raise ValueError("G=97 requires both include_g97=true and explicit --enable-g97")
    if args.enable_g97 and not configuration.include_g97:
        raise ValueError("--enable-g97 requires a large-pilot configuration containing G=97")
    resource = estimate_stage_resources(configuration)
    if resource["violations"] and not args.allow_expensive:
        raise ValueError(f"static pilot resource guard rejected: {resource['violations']}")
    required_host = math.ceil(
        resource["maximum_host_planning_threshold_bytes"]
        * (1.0 + configuration.safety_margin_fraction)
    )
    doctor_host_available = doctor.get("host_memory", {}).get("available_bytes")
    if args.execute and (
        isinstance(doctor_host_available, bool)
        or not isinstance(doctor_host_available, int)
        or doctor_host_available < required_host
    ):
        raise ValueError("doctor did not establish sufficient available host memory")
    commit = _git("rev-parse", "HEAD")
    clean = not bool(_git("status", "--porcelain", "--", "."))
    environment_sha = str(doctor.get("environment_sha256", ""))
    prerequisite_document = None
    if args.execute:
        if configuration.hourly_price_usd <= 0.0:
            raise ValueError(
                "execution requires a positive user-supplied hourly price in an ignored reviewed config copy"
            )
        if args.hourly_price_usd != configuration.hourly_price_usd:
            raise ValueError(
                "execution requires --hourly-price-usd exactly matching the reviewed configuration"
            )
        if not doctor.get("gpu_ready") or not clean:
            raise ValueError("execution requires a ready doctor report and clean subproject")
        if doctor.get("git", {}).get("commit") != commit:
            raise ValueError("doctor report commit does not match this invocation")
        required = _PREVIOUS[stage]
        if args.prerequisite is None:
            raise ValueError(f"{stage.value} requires a {required.value} prerequisite artifact")
        prerequisite_document = read_prerequisite_artifact(
            args.prerequisite.resolve(), required_stage=required,
            commit=commit, environment_sha256=environment_sha,
        )
    elif args.prerequisite is not None:
        # Dry-run validates a supplied artifact but does not require one on a CPU host.
        prerequisite_document = read_prerequisite_artifact(
            args.prerequisite.resolve(), required_stage=_PREVIOUS[stage],
            commit=commit, environment_sha256=environment_sha,
        )
    executable_config_sha = executable_configuration_sha256(configuration)
    contract = stage_invariant_contract(
        configuration, commit=commit, environment_sha256=environment_sha,
        source_hashes={
            **{
                name: "unavailable" for name in (
                    "src/chu_pair/model.py", "src/chu_pair/initial_conditions.py",
                    "src/chu_pair/pair_density/jax_solver.py",
                )
            },
            **doctor.get("source_hashes", {}),
        },
    )
    contract_sha = stage_invariant_contract_sha256(contract)
    if stage == PilotStage.FULL_GRID_ONE_STEP and prerequisite_document is not None:
        prior_sha = prerequisite_document.get("plan", {}).get(
            "executable_configuration_sha256"
        )
        if prior_sha != executable_config_sha:
            raise ValueError(
                "full-grid analysis artifact does not match the one-step executable configuration"
            )
    prerequisite_digest = (
        None if prerequisite_document is None else prerequisite_document["artifact_sha256"]
    )
    prior_cumulative_cost = 0.0
    if prerequisite_document is not None:
        prior_cost_record = prerequisite_document.get("cost", {})
        raw_prior_cost = prior_cost_record.get(
            "cumulative_stage_estimate_usd", 0.0
        )
        if isinstance(raw_prior_cost, bool) or not isinstance(raw_prior_cost, (int, float)):
            raise ValueError("prerequisite cumulative cost is invalid")
        prior_cumulative_cost = float(raw_prior_cost)
        if not math.isfinite(prior_cumulative_cost) or prior_cumulative_cost < 0.0:
            raise ValueError("prerequisite cumulative cost is invalid")
        if prior_cost_record:
            if (
                prior_cost_record.get("configured_session_budget_usd")
                != configuration.max_session_cost_usd
                or prior_cost_record.get("hourly_price_usd_user_supplied")
                != configuration.hourly_price_usd
            ):
                raise ValueError("prerequisite price/session budget differs from this stage")
        prior_contract = prerequisite_document.get("plan", {}).get("stage_invariant_contract_sha256")
        if _PREVIOUS[stage] != PilotStage.DOCTOR and prior_contract != contract_sha:
            raise ValueError("prerequisite scientific/session contract differs from this stage")
    if args.execute:
        maximum_charge, budget_charge = _maximum_stage_charge(
            prior=prior_cumulative_cost, price=configuration.hourly_price_usd,
            seconds=configuration.max_stage_seconds,
            budget=configuration.max_session_cost_usd,
        )
        if maximum_charge > budget_charge:
            raise ValueError("configured worst-case stage charge exceeds the remaining session budget")
    plan = {
        "stage": stage.value,
        "dry_run": args.dry_run,
        "configuration": asdict(configuration),
        "resource_estimate": resource,
        "resource_violations_overridden": resource["violations"] if args.allow_expensive else [],
        "required_prerequisite": _PREVIOUS[stage].value,
        "confirmation_required": stage in {PilotStage.FULL_GRID_ANALYZE, PilotStage.FULL_GRID_ONE_STEP},
        "would_compile": not args.dry_run,
        "would_execute": bool(args.execute and stage != PilotStage.FULL_GRID_ANALYZE),
        "required_host_bytes_with_margin": required_host,
        "doctor_report_sha256": hashlib.sha256(
            json.dumps(doctor, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "executable_configuration_sha256": executable_config_sha,
        "prerequisite_artifact_sha256": prerequisite_digest,
        "prior_cumulative_stage_estimate_usd": prior_cumulative_cost,
        "stage_invariant_contract": contract,
        "stage_invariant_contract_sha256": contract_sha,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (OUTPUT_ROOT / f"{configuration.run_name}-{timestamp}").resolve()
    if OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("unsafe GPU pilot output directory")
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    payload: dict[str, object] = {
        "schema_version": 1, "stage": stage.value, "status": "failed",
        "started_utc": started.isoformat(), "git_commit": commit,
        "subproject_clean": clean,
        "environment_sha256": environment_sha, "plan": plan,
        "doctor_report": doctor,
        "event_log": [
            {"event": "bounded_preflight_accepted", "utc": started.isoformat()},
        ],
    }
    try:
        from chu_pair.gpu_pilot.runtime import compile_and_maybe_execute_case
        import jax
        import jaxlib
        import numpy as np
        from chu_pair.gpu_pilot.doctor import collect_gpu_doctor_report

        live_doctor = collect_gpu_doctor_report(
            project_root=PROJECT_ROOT, jax_module=jax, numpy_module=np,
            jaxlib_module=jaxlib, cuda_family=str(doctor["cuda_family"]),
        )
        if (
            not live_doctor["gpu_ready"]
            or live_doctor["environment_sha256"] != environment_sha
            or live_doctor["git"]["commit"] != commit
        ):
            raise ValueError("live environment no longer matches the accepted GPU doctor")
        if int(live_doctor["host_memory"]["available_bytes"]) < required_host:
            raise ValueError("live available host memory is below the guarded requirement")
        payload["event_log"].append(
            {"event": "live_environment_revalidated", "utc": datetime.now(timezone.utc).isoformat()}
        )

        records = payload.setdefault("records", [])
        execute_case = stage != PilotStage.FULL_GRID_ANALYZE
        expected_signature = None
        if stage == PilotStage.FULL_GRID_ONE_STEP:
            try:
                expected_signature = prerequisite_document["records"][0]["kernels"][
                    "separable"
                ]["executable_signature_sha256"]
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError("full-grid analysis artifact lacks its exact executable signature") from error
        for case_estimate in resource["cases"]:
            for variant in _stage_variants(stage, configuration):
                records.append(
                    compile_and_maybe_execute_case(
                        variant, grid_size=int(case_estimate["grid_size"]),
                        static_estimate=case_estimate, execute=execute_case,
                        expected_signature_sha256=expected_signature,
                    )
                )
                payload["event_log"].append(
                    {
                        "event": "case_completed", "grid_size": int(case_estimate["grid_size"]),
                        "steps": variant.steps, "utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
        elapsed = time.perf_counter() - start
        if elapsed > configuration.max_stage_seconds:
            raise TimeoutError("stage exceeded its configured in-process wall-time budget")
        cost = summarize_stage_cost(
            elapsed_seconds=elapsed,
            hourly_price_usd=configuration.hourly_price_usd,
            prior_cumulative_usd=prior_cumulative_cost,
            session_budget_usd=configuration.max_session_cost_usd,
            next_stage_max_seconds=_NEXT_STAGE_MAX_SECONDS[stage],
        )
        if cost["cumulative_stage_estimate_usd"] > configuration.max_session_cost_usd:
            raise RuntimeError("observed stage cost estimate exceeded the session budget")
        payload.update(status="success", cost=cost)
        payload["event_log"].append(
            {"event": "stage_succeeded", "utc": datetime.now(timezone.utc).isoformat()}
        )
    except BaseException as error:
        payload["error"] = {"type": type(error).__name__, "message": str(error)[:1024]}
        payload["event_log"].append(
            {"event": "stage_failed", "utc": datetime.now(timezone.utc).isoformat()}
        )
        raise
    finally:
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["artifact_path"] = str(output.relative_to(PROJECT_ROOT) / "stage.json")
        path = write_stage_artifact_atomic(output, payload)
        _ARTIFACT_WRITTEN = True
        print(f"stage artifact: {path}", file=sys.stderr)
    return 0


def entrypoint() -> int:
    """Ensure catchable preflight failures also receive one atomic artifact."""

    global _ARTIFACT_WRITTEN
    try:
        return main()
    except BaseException as error:
        if not _ARTIFACT_WRITTEN:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            directory = (OUTPUT_ROOT / f"rejected-{timestamp}").resolve()
            try:
                commit = _git("rev-parse", "HEAD")
                clean = not bool(_git("status", "--porcelain", "--", "."))
            except Exception:
                commit, clean = "unavailable", False
            payload = {
                "schema_version": 1,
                "stage": "preflight-rejection",
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "subproject_clean": clean,
                "error": {"type": type(error).__name__, "message": str(error)[:1024]},
                "provenance_scope": (
                    "raw rejected configuration and command line are intentionally not serialized"
                ),
            }
            path = write_stage_artifact_atomic(directory, payload)
            _ARTIFACT_WRITTEN = True
            print(f"rejection artifact: {path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())

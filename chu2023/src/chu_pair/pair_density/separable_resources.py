"""Allocation-free resource models for exact separable pair transport."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import subprocess
from typing import Any, Callable, Mapping

import numpy as np


DIAGNOSTIC_FLOAT_SCALARS = 11
DIAGNOSTIC_BOOL_SCALARS = 3
POINT_SUMMARY_FLOATS = 15
STATIC_DEVICE_FIXED_BYTES = 4 * 1024
MAX_CAPACITY_EVIDENCE_AGE_SECONDS = 60
MAX_NVIDIA_SMI_OUTPUT_BYTES = 16 * 1024
NVIDIA_SMI_TIMEOUT_SECONDS = 5
_SHA256_TEXT = re.compile(r"^[0-9a-f]{64}$")
MAX_COMPILED_PROGRAM_EVIDENCE_BYTES = 4096
MAX_CAPACITY_TEXT_CHARS = 512
MAX_STABLE_IDENTITY_FIELDS = 8
MAX_CLOCK_SKEW_SECONDS = 1
_BUNDLE_FACTORY_TOKEN = object()
_MEMORY_ANALYSIS_FIELDS = {
    "argument_bytes": "argument_size_in_bytes",
    "output_bytes": "output_size_in_bytes",
    "temporary_bytes": "temp_size_in_bytes",
    "alias_bytes": "alias_size_in_bytes",
    "host_argument_bytes": "host_argument_size_in_bytes",
    "host_output_bytes": "host_output_size_in_bytes",
    "host_temporary_bytes": "host_temp_size_in_bytes",
    "host_alias_bytes": "host_alias_size_in_bytes",
}


@dataclass(frozen=True)
class CompiledExecutableBundle:
    """Runtime-only identity and memory analysis for one compiled callable."""

    compiled_callable: object
    memory_report: Mapping[str, object]
    compile_signature: Mapping[str, object]
    abstract_arguments: Mapping[str, object]
    static_values: Mapping[str, object]
    runtime_environment: Mapping[str, object]
    callable_identity: int
    signature_sha256: str
    # Digest of the exact bounded compiler IR that produced this callable.
    # The high-level signature records declared policy; two lowerings differing
    # only in dot-product precision share it, so the program digest is what
    # distinguishes the generated programs. It is toolchain-specific and is
    # NOT a cross-version reproducibility claim.
    compiled_program_sha256: str
    compiled_program_evidence: Mapping[str, object]
    bundle_integrity_sha256: str
    _factory_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class DeviceCapacityObservation:
    """Bounded evidence about memory usable by one exact execution device."""

    available: bool
    source: str
    observed_at_utc: str
    backend: str
    platform: str
    visible_device_index: int
    execution_device_identity: Mapping[str, object]
    stable_device_identity: Mapping[str, object]
    device_name: str
    total_physical_bytes: int | None
    free_bytes: int | None
    used_bytes: int | None
    allocator_reserved_bytes: int | None
    allocator_available_bytes: int | None
    allocator_policy: str
    usable_device_bytes: int | None
    usable_bytes_definition: str
    unavailable_reason: str | None = None
    # ``nvidia-smi`` reports free memory after allocations owned by this
    # process.  Keep that fact explicit rather than treating the remaining
    # global free bytes as unavailable to an already-created JAX pool.
    current_process_owned_bytes: int | None = None
    # Pre-initialization evidence describes memory the JAX allocator has not
    # reserved yet; post-initialization evidence describes the pool this
    # process already owns.  The two are never interchangeable.
    post_initialization: bool = False


@dataclass(frozen=True)
class AllocatorCapacityObservation:
    """Post-input allocator evidence for one already initialized JAX device."""

    available: bool
    policy: str
    byte_limit: int | None
    bytes_in_use: int | None
    peak_bytes_in_use: int | None
    largest_free_block_bytes: int | None
    internal_free_bytes: int | None
    source: str
    unavailable_reason: str | None = None
    largest_free_block_status: str = "reported"


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _bundle_integrity_payload(
    *,
    memory_report: Mapping[str, object],
    compile_signature: Mapping[str, object],
    abstract_arguments: Mapping[str, object],
    static_values: Mapping[str, object],
    runtime_environment: Mapping[str, object],
    callable_identity: int,
    compiled_program_sha256: str,
    compiled_program_evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "memory_report": dict(memory_report),
        "compile_signature": dict(compile_signature),
        "abstract_arguments": dict(abstract_arguments),
        "static_values": dict(static_values),
        "runtime_environment": dict(runtime_environment),
        "callable_identity": int(callable_identity),
        "compiled_program_sha256": str(compiled_program_sha256),
        "compiled_program_evidence": dict(compiled_program_evidence),
    }


def make_compiled_executable_bundle(
    *,
    compiled_callable,
    memory_report: Mapping[str, object],
    compile_signature: Mapping[str, object],
    abstract_arguments: Mapping[str, object],
    static_values: Mapping[str, object],
    runtime_environment: Mapping[str, object],
    compiled_program_sha256: str,
    compiled_program_evidence: Mapping[str, object],
) -> CompiledExecutableBundle:
    """Bind one analyzed callable to immutable, independently checkable facts."""

    if not callable(compiled_callable):
        raise TypeError("compiled_callable must be callable")
    if not isinstance(compiled_program_sha256, str) or not _SHA256_TEXT.fullmatch(
        compiled_program_sha256
    ):
        raise ValueError("compiled program digest is missing or malformed")
    if not isinstance(compiled_program_evidence, Mapping):
        raise ValueError("compiled program evidence is missing or malformed")
    program_evidence = {
        key: compiled_program_evidence[key] for key in sorted(compiled_program_evidence)
    }
    for name in ("jax_version", "jaxlib_version", "backend", "device_kind",
                 "jax_enable_x64", "ir_dialect", "ir_bytes"):
        if name not in program_evidence:
            raise ValueError(f"compiled program evidence lacks {name}")
    if len(json.dumps(program_evidence, ensure_ascii=True, sort_keys=True)) > (
        MAX_COMPILED_PROGRAM_EVIDENCE_BYTES
    ):
        raise ValueError("compiled program evidence exceeds its fixed bound")
    signature = dict(compile_signature)
    digest = _canonical_digest(signature)
    report = dict(memory_report)
    report["executable_signature"] = signature
    report["executable_signature_sha256"] = digest
    callable_identity = id(compiled_callable)
    abstract = dict(abstract_arguments)
    static = dict(static_values)
    environment = dict(runtime_environment)
    integrity = _canonical_digest(
        _bundle_integrity_payload(
            memory_report=report,
            compile_signature=signature,
            abstract_arguments=abstract,
            static_values=static,
            runtime_environment=environment,
            callable_identity=callable_identity,
            compiled_program_sha256=compiled_program_sha256,
            compiled_program_evidence=program_evidence,
        )
    )
    return CompiledExecutableBundle(
        compiled_callable=compiled_callable,
        memory_report=report,
        compile_signature=signature,
        abstract_arguments=abstract,
        static_values=static,
        runtime_environment=environment,
        callable_identity=callable_identity,
        signature_sha256=digest,
        compiled_program_sha256=compiled_program_sha256,
        compiled_program_evidence=program_evidence,
        bundle_integrity_sha256=integrity,
        _factory_token=_BUNDLE_FACTORY_TOKEN,
    )


def _validate_array_spec_tree(value: object, *, path: str) -> int:
    """Validate a nested abstract-argument tree and count array-spec leaves."""

    if isinstance(value, Mapping):
        if set(value) == {"shape", "dtype", "weak_type"}:
            shape = value["shape"]
            if (
                not isinstance(shape, list)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in shape)
            ):
                raise ValueError(f"{path}.shape is not a valid dimension list")
            try:
                np.dtype(value["dtype"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}.dtype is invalid") from error
            if not isinstance(value["weak_type"], bool):
                raise ValueError(f"{path}.weak_type must be boolean")
            return 1
        if not value:
            raise ValueError(f"{path} must not be empty")
        return sum(
            _validate_array_spec_tree(child, path=f"{path}.{name}")
            for name, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{path} must not be empty")
        return sum(
            _validate_array_spec_tree(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    raise ValueError(f"{path} contains a non-array-spec leaf")


def _live_memory_values(compiled_callable: object) -> dict[str, int]:
    analysis = getattr(compiled_callable, "memory_analysis", None)
    if not callable(analysis):
        raise ValueError("compiled callable does not provide memory_analysis()")
    try:
        statistics = analysis()
    except Exception as error:
        raise ValueError(
            f"live compiled memory_analysis() failed: {type(error).__name__}: {error}"
        ) from error
    if statistics is None:
        raise ValueError("live compiled memory_analysis() returned None")
    values: dict[str, int] = {}
    for output_name, attribute_name in _MEMORY_ANALYSIS_FIELDS.items():
        value = getattr(statistics, attribute_name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"live compiled memory_analysis() supplied invalid {attribute_name}"
            )
        values[output_name] = value
    return values


def validate_compiled_executable_bundle(
    bundle: CompiledExecutableBundle,
    *,
    expected_compile_signature: Mapping[str, object] | None = None,
) -> dict[str, int]:
    """Validate the complete contract and re-read the retained executable report."""

    if not isinstance(bundle, CompiledExecutableBundle):
        raise ValueError("an exact CompiledExecutableBundle is required")
    if bundle._factory_token is not _BUNDLE_FACTORY_TOKEN:
        raise ValueError("compiled executable bundle was not created by the runtime factory")
    if not callable(bundle.compiled_callable):
        raise ValueError("compiled executable bundle has no callable")
    if id(bundle.compiled_callable) != bundle.callable_identity:
        raise ValueError("compiled callable identity does not match its bundle")
    digest = _canonical_digest(bundle.compile_signature)
    if digest != bundle.signature_sha256:
        raise ValueError("compiled signature digest does not match its bundle")
    if bundle.memory_report.get("executable_signature") != bundle.compile_signature:
        raise ValueError("compiled memory report disagrees with its signature")
    if bundle.memory_report.get("executable_signature_sha256") != digest:
        raise ValueError("compiled memory report has the wrong signature digest")
    if not isinstance(bundle.compiled_program_sha256, str) or not _SHA256_TEXT.fullmatch(
        bundle.compiled_program_sha256
    ):
        raise ValueError("compiled program digest is missing or malformed")
    if not isinstance(bundle.compiled_program_evidence, Mapping) or not bundle.compiled_program_evidence:
        raise ValueError("compiled program evidence is missing or malformed")
    if expected_compile_signature is not None and dict(expected_compile_signature) != dict(
        bundle.compile_signature
    ):
        raise ValueError("expected compile signature does not match its bundle")

    signature = bundle.compile_signature
    required_signature_fields = {
        "executable_id",
        "kernel",
        "output_mode",
        "dtype",
        "contract_abstract_arguments",
        "contract_static_values",
        "contract_runtime_environment",
    }
    if not required_signature_fields.issubset(signature):
        raise ValueError("compiled signature omits required contract fields")
    executable_id = signature["executable_id"]
    if not isinstance(executable_id, str):
        raise ValueError("compiled signature executable_id must be text")
    if executable_id.startswith("pair-source-from-histogram:"):
        required_pair_fields = {
            "grid_size",
            "agent_grid_points",
            "state_expanded_cells",
            "histogram_argument",
            "state_probability_argument",
            "grid_arguments",
            "alpha",
            "tau",
            "dynamic_scalar_arguments",
            "steps",
            "summary_count",
            "requested_source_times",
            "source_slots",
            "source_slot_argument",
            "chunk_size",
            "row_block_size",
            "column_block_size",
            "diagnostic_tolerance",
            "symmetry_tolerance",
        }
        if not required_pair_fields.issubset(signature):
            raise ValueError("pair executable signature is incomplete")
        if (
            signature["agent_grid_points"] != signature["grid_size"] ** 2
            or signature["state_expanded_cells"]
            != 2 * signature["agent_grid_points"] ** 2
            or signature["summary_count"] != len(signature["requested_source_times"])
            or len(signature["source_slots"]) != signature["steps"] + 1
        ):
            raise ValueError("pair executable signature dimensions are inconsistent")
        slots = signature["source_slots"]
        if any(
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot < -1
            or slot >= signature["summary_count"]
            for slot in slots
        ):
            raise ValueError("pair executable source-slot values are invalid")
        selected_times = [index for index, slot in enumerate(slots) if slot >= 0]
        selected_slots = sorted(slot for slot in slots if slot >= 0)
        if (
            selected_times != signature["requested_source_times"]
            or selected_slots != list(range(signature["summary_count"]))
        ):
            raise ValueError("pair executable source-slot contract is inconsistent")
        for name in ("alpha", "tau", "diagnostic_tolerance", "symmetry_tolerance"):
            value = signature[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"pair executable {name} is invalid")
    elif executable_id.startswith("pair-reduction:"):
        required_reduction_fields = {
            "point_count",
            "cell_count",
            "value_argument",
            "destination_argument",
        }
        if not required_reduction_fields.issubset(signature):
            raise ValueError("reduction executable signature is incomplete")
        if signature["cell_count"] != signature["point_count"] ** 2:
            raise ValueError("reduction executable dimensions are inconsistent")
    else:
        raise ValueError("compiled signature executable_id is not recognized")
    if not bundle.abstract_arguments or not bundle.static_values or not bundle.runtime_environment:
        raise ValueError("compiled bundle contract mappings must not be empty")
    if signature["contract_abstract_arguments"] != bundle.abstract_arguments:
        raise ValueError("compiled signature disagrees with abstract arguments")
    if signature["contract_static_values"] != bundle.static_values:
        raise ValueError("compiled signature disagrees with static values")
    if signature["contract_runtime_environment"] != bundle.runtime_environment:
        raise ValueError("compiled signature disagrees with runtime environment")
    if _validate_array_spec_tree(bundle.abstract_arguments, path="abstract_arguments") < 1:
        raise ValueError("compiled bundle has no abstract argument leaves")

    environment = bundle.runtime_environment
    required_environment = {
        "backend",
        "platform",
        "execution_device",
        "visible_devices",
        "jax_enable_x64",
        "jax_version",
        "jaxlib_version",
    }
    if not required_environment.issubset(environment):
        raise ValueError("compiled bundle runtime environment is incomplete")
    if (
        not isinstance(environment["jax_version"], str)
        or not environment["jax_version"]
        or not isinstance(environment["jaxlib_version"], str)
        or not environment["jaxlib_version"]
        or not isinstance(environment["jax_enable_x64"], bool)
        or not isinstance(environment["execution_device"], Mapping)
        or not environment["execution_device"]
        or not isinstance(environment["visible_devices"], list)
        or not environment["visible_devices"]
    ):
        raise ValueError("compiled bundle runtime environment contains invalid values")
    for name in ("backend", "platform", "execution_device", "visible_devices", "jax_enable_x64", "jax_version", "jaxlib_version"):
        if signature.get(name) != environment[name]:
            raise ValueError(f"compiled signature disagrees with runtime {name}")

    report = bundle.memory_report
    if report.get("available") is not True or report.get("analysis_status") != "complete":
        raise ValueError("compiled memory analysis is unavailable or incomplete")
    live_values = _live_memory_values(bundle.compiled_callable)
    for name, value in live_values.items():
        if report.get(name) != value:
            raise ValueError(f"stored compiled memory report disagrees with live {name}")
    argument = live_values["argument_bytes"]
    output = live_values["output_bytes"]
    temporary = live_values["temporary_bytes"]
    alias = live_values["alias_bytes"]
    host_argument = live_values["host_argument_bytes"]
    host_output = live_values["host_output_bytes"]
    host_temporary = live_values["host_temporary_bytes"]
    host_alias = live_values["host_alias_bytes"]
    if alias > argument + output or host_alias > host_argument + host_output:
        raise ValueError("compiled memory aliases exceed argument plus output storage")
    device = argument + output + temporary - alias
    host = host_argument + host_output + host_temporary - host_alias
    static_host = report.get("static_host_allowance_bytes")
    combined = report.get("compiled_plus_host_requirement_bytes")
    if (
        report.get("compiled_device_requirement_bytes") != device
        or report.get("compiled_host_requirement_bytes") != host
        or isinstance(static_host, bool)
        or not isinstance(static_host, int)
        or static_host < 0
        or combined != device + host + static_host
    ):
        raise ValueError("compiled memory report is internally inconsistent")
    if report.get("backend") != environment["backend"]:
        raise ValueError("compiled memory report backend disagrees with runtime environment")

    integrity = _canonical_digest(
        _bundle_integrity_payload(
            memory_report=report,
            compile_signature=signature,
            abstract_arguments=bundle.abstract_arguments,
            static_values=bundle.static_values,
            runtime_environment=environment,
            callable_identity=bundle.callable_identity,
            compiled_program_sha256=bundle.compiled_program_sha256,
            compiled_program_evidence=bundle.compiled_program_evidence,
        )
    )
    if integrity != bundle.bundle_integrity_sha256:
        raise ValueError("compiled bundle integrity digest does not match its contents")
    return {
        **live_values,
        "compiled_device_requirement_bytes": device,
        "compiled_host_requirement_bytes": host,
        "compiled_plus_host_requirement_bytes": combined,
    }


def make_device_capacity_observation(
    *,
    source: str,
    observed_at_utc: str,
    backend: str,
    platform: str,
    visible_device_index: int,
    execution_device_identity: Mapping[str, object],
    stable_device_identity: Mapping[str, object],
    device_name: str,
    total_physical_bytes: int,
    free_bytes: int,
    used_bytes: int,
    allocator_reserved_bytes: int | None = None,
    allocator_available_bytes: int | None = None,
    allocator_policy: str = "injected provider; no additional allocator constraint reported",
    post_initialization: bool = False,
    process_pool_bytes: int | None = None,
    allocator_internal_free_bytes: int | None = None,
) -> DeviceCapacityObservation:
    """Construct verified evidence and derive a conservative usable-byte value.

    Before the JAX allocator reserves its pool, the usable quantity is physical
    free memory, optionally capped by the configured preallocation target.
    Once this process owns the pool, the bytes an executable can actually draw
    on are the pool's own internal free bytes; the external free memory left
    beside the reservation is a different pool and adding or intersecting the
    two would double count this process's own reservation.
    """

    byte_values = {
        "total_physical_bytes": total_physical_bytes,
        "free_bytes": free_bytes,
        "used_bytes": used_bytes,
    }
    for name, value in byte_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if free_bytes > total_physical_bytes or used_bytes > total_physical_bytes:
        raise ValueError("free/used capacity cannot exceed total physical capacity")
    optional = {
        "allocator_reserved_bytes": allocator_reserved_bytes,
        "allocator_available_bytes": allocator_available_bytes,
    }
    for name, value in optional.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    for name, value in {
        "process_pool_bytes": process_pool_bytes,
        "allocator_internal_free_bytes": allocator_internal_free_bytes,
    }.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if post_initialization:
        if allocator_internal_free_bytes is None:
            raise ValueError(
                "post-initialization capacity requires allocator internal free bytes"
            )
        usable = allocator_internal_free_bytes
        allocator_available_bytes = allocator_internal_free_bytes
        definition = (
            "internal free bytes of the allocator pool this process already owns; "
            "external free physical bytes are deliberately excluded so the "
            "process-owned reservation is not counted twice"
        )
    else:
        usable = free_bytes
        definition = "current free physical bytes reported for the matched device"
        if allocator_available_bytes is not None:
            usable = min(usable, allocator_available_bytes)
            definition = (
                "minimum of current free physical bytes and reported allocator-available bytes"
            )
    return DeviceCapacityObservation(
        available=True,
        source=str(source),
        observed_at_utc=str(observed_at_utc),
        backend=str(backend),
        platform=str(platform),
        visible_device_index=int(visible_device_index),
        execution_device_identity=dict(execution_device_identity),
        stable_device_identity=dict(stable_device_identity),
        device_name=str(device_name),
        total_physical_bytes=total_physical_bytes,
        free_bytes=free_bytes,
        used_bytes=used_bytes,
        allocator_reserved_bytes=allocator_reserved_bytes,
        allocator_available_bytes=allocator_available_bytes,
        allocator_policy=str(allocator_policy),
        usable_device_bytes=usable,
        usable_bytes_definition=definition,
        current_process_owned_bytes=process_pool_bytes,
        post_initialization=bool(post_initialization),
    )


def unavailable_device_capacity_observation(
    *,
    source: str,
    execution_device_identity: Mapping[str, object],
    reason: str,
) -> DeviceCapacityObservation:
    """Return bounded unavailable evidence instead of guessing capacity."""

    return DeviceCapacityObservation(
        available=False,
        source=str(source),
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        backend=str(execution_device_identity.get("backend", "unknown")),
        platform=str(execution_device_identity.get("platform", "unknown")),
        visible_device_index=int(execution_device_identity.get("visible_device_index", -1)),
        execution_device_identity=dict(execution_device_identity),
        stable_device_identity={},
        device_name=str(execution_device_identity.get("device_kind", "unknown")),
        total_physical_bytes=None,
        free_bytes=None,
        used_bytes=None,
        allocator_reserved_bytes=None,
        allocator_available_bytes=None,
        allocator_policy="unavailable",
        usable_device_bytes=None,
        usable_bytes_definition="unavailable",
        unavailable_reason=str(reason)[:512],
    )


def allocator_capacity_observation(
    statistics: object,
    *,
    policy: str,
) -> AllocatorCapacityObservation:
    """Normalize supported JAX allocator statistics, failing closed otherwise.

    JAX device ``memory_stats`` is intentionally queried only after device
    inputs exist.  The required fields are conservative: a fixed pool must
    report its limit, current use, and largest free block before it can admit a
    further compiled invocation.
    """

    if policy not in {"default", "fraction", "no-preallocation"}:
        raise ValueError("allocator policy is invalid")
    if not isinstance(statistics, Mapping):
        return AllocatorCapacityObservation(
            False, policy, None, None, None, None, None,
            "jax.device.memory_stats", "allocator statistics are unavailable",
        )

    def integer(name: str) -> int | None:
        value = statistics.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    limit = integer("bytes_limit")
    in_use = integer("bytes_in_use")
    peak = integer("peak_bytes_in_use")
    largest = integer("largest_free_block_bytes")
    if limit is None or in_use is None or in_use > limit:
        return AllocatorCapacityObservation(
            False, policy, limit, in_use, peak, largest, None,
            "jax.device.memory_stats", "allocator limit/use statistics are incomplete",
        )
    internal = limit - in_use
    # A pool that still has internal free bytes cannot simultaneously have a
    # largest free block of zero.  Backends that never populate the field
    # report exactly that impossible pair, so classify it as unreported rather
    # than believing a fragmentation measurement that does not exist.  A real
    # zero (a completely full pool) stays authoritative.
    status = "reported"
    if largest is None:
        status = "not_reported"
    elif largest == 0 and internal > 0:
        largest, status = None, "unsupported_by_backend"
    if largest is not None and largest > internal:
        return AllocatorCapacityObservation(
            False, policy, limit, in_use, peak, largest, internal,
            "jax.device.memory_stats",
            "largest allocator free block exceeds internal free bytes", status,
        )
    return AllocatorCapacityObservation(
        True, policy, limit, in_use, peak, largest, internal,
        "jax.device.memory_stats", None, status,
    )


def post_initialization_capacity_preflight(
    *,
    feasibility: Mapping[str, object],
    bundle: CompiledExecutableBundle,
    external_capacity: DeviceCapacityObservation,
    allocator_capacity: AllocatorCapacityObservation,
    arguments_resident: bool = True,
) -> dict[str, object]:
    """Admit one exact invocation after its arguments already reside on device.

    ``memory_analysis`` reports the full executable requirement including
    arguments.  At this point those argument buffers are already resident, so
    only ``output + temporary - aliases`` is incremental.  Fixed preallocated
    pools are judged from their own free bytes, not from the globally free
    memory left after the pool was reserved.  A growing allocator is judged
    from fresh globally free bytes.  Every unavailable statistic fails closed.
    """

    report = validate_compiled_executable_bundle(bundle)
    fields = ("argument_bytes", "output_bytes", "temporary_bytes", "alias_bytes")
    if any(isinstance(report.get(name), bool) or not isinstance(report.get(name), int)
           or report[name] < 0 for name in fields):
        raise ValueError("post-initialization compiled memory analysis is incomplete")
    argument, output, temporary, aliases = (int(report[name]) for name in fields)
    if aliases > argument + output:
        raise ValueError("post-initialization compiled alias accounting is invalid")
    # Argument buffers are excluded only once they are genuinely resident on
    # the device.  Before device inputs exist they still have to be placed, so
    # the full analysed requirement is charged.
    incremental = max(0, output + temporary - aliases) if arguments_resident else max(
        0, argument + output + temporary - aliases
    )
    margin = feasibility.get("safety_margin_fraction")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(float(margin)) or float(margin) < 0:
        raise ValueError("post-initialization safety margin is invalid")
    required = math.ceil(incremental * (1.0 + float(margin)))
    if not external_capacity.available or external_capacity.free_bytes is None:
        raise ValueError("post-initialization external capacity is unavailable")
    if not allocator_capacity.available:
        raise ValueError(
            "post-initialization allocator capacity is unavailable: "
            f"{allocator_capacity.unavailable_reason}"
        )
    if allocator_capacity.policy == "no-preallocation":
        admitted = external_capacity.free_bytes
        largest = None
        derivation = "fresh external free bytes for growing allocator"
    else:
        if allocator_capacity.internal_free_bytes is None:
            raise ValueError("fixed allocator pool statistics are incomplete")
        largest = allocator_capacity.largest_free_block_bytes
        if largest is None:
            # The backend does not publish a fragmentation measure.  Admit from
            # the pool's own internal free bytes and record that no
            # largest-free-block evidence constrained this decision.
            admitted = allocator_capacity.internal_free_bytes
            derivation = (
                "internal pool free bytes; no largest-free-block measure is "
                f"published by this backend ({allocator_capacity.largest_free_block_status})"
            )
        else:
            admitted = min(allocator_capacity.internal_free_bytes, largest)
            derivation = "minimum internal pool free bytes and largest free allocator block"
    if admitted < required:
        raise ValueError("post-initialization allocator capacity is insufficient")
    return {
        "passed": True,
        "phase": "post-initialization",
        "arguments_resident": bool(arguments_resident),
        "largest_free_block_status": allocator_capacity.largest_free_block_status,
        "allocator_policy": allocator_capacity.policy,
        "external_total_bytes": external_capacity.total_physical_bytes,
        "external_free_bytes": external_capacity.free_bytes,
        "current_process_owned_bytes": external_capacity.current_process_owned_bytes,
        "allocator_byte_limit": allocator_capacity.byte_limit,
        "allocator_bytes_in_use": allocator_capacity.bytes_in_use,
        "allocator_peak_bytes_in_use": allocator_capacity.peak_bytes_in_use,
        "allocator_internal_free_bytes": allocator_capacity.internal_free_bytes,
        "allocator_largest_free_block_bytes": largest,
        "already_resident_argument_bytes": argument if arguments_resident else 0,
        "charged_argument_bytes": 0 if arguments_resident else argument,
        "incremental_output_bytes": output,
        "incremental_temporary_bytes": temporary,
        "alias_bytes": aliases,
        "incremental_executable_requirement_bytes": incremental,
        "safety_margin_fraction": float(margin),
        "required_incremental_bytes_with_margin": required,
        "admitted_usable_bytes": admitted,
        "admission_derivation": derivation,
    }


def _normalize_pci_identifier(value: object) -> str | None:
    text = str(value).strip().lower()
    match = re.fullmatch(
        r"(?:0x)?([0-9a-f]{1,8}):([0-9a-f]{1,2}):([0-9a-f]{1,2})\.([0-7])",
        text,
    )
    if match is None:
        return None
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain & 0xffff:04x}:{bus:02x}:{device:02x}.{function:x}"


def _uuid_identity_matches(candidate: object, requested: object) -> bool:
    left = str(candidate).strip().lower()
    right = str(requested).strip().lower()
    if not right.startswith(("gpu-", "mig-")):
        return False
    # NVIDIA accepts abbreviated UUIDs; require a meaningful stable prefix.
    return len(right) >= 8 and (left.startswith(right) or right.startswith(left))


def discover_nvidia_device_capacity(
    execution_device_identity: Mapping[str, object],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    cuda_visible_ordinal_mapper: Callable[..., Mapping[str, object] | None] | None = None,
    cuda_visible_devices: str | None = None,
    cuda_device_order: str | None = None,
    preallocate_setting: str | None = None,
    memory_fraction_setting: str | None = None,
    post_initialization: bool = False,
    allocator_statistics: Mapping[str, object] | None = None,
) -> DeviceCapacityObservation:
    """Query bounded NVIDIA evidence and fail closed if device matching is ambiguous."""

    source = "nvidia-smi query-gpu"
    if execution_device_identity.get("backend") != "gpu" or execution_device_identity.get(
        "platform"
    ) != "gpu":
        return unavailable_device_capacity_observation(
            source=source,
            execution_device_identity=execution_device_identity,
            reason="execution device is not an NVIDIA GPU backend",
        )
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = command_runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except Exception as error:
        return unavailable_device_capacity_observation(
            source=source,
            execution_device_identity=execution_device_identity,
            reason=f"{type(error).__name__}: {error}",
        )
    output = completed.stdout
    if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_NVIDIA_SMI_OUTPUT_BYTES:
        return unavailable_device_capacity_observation(
            source=source,
            execution_device_identity=execution_device_identity,
            reason="nvidia-smi output is missing or exceeds its fixed bound",
        )
    try:
        rows = list(csv.reader(output.splitlines()))
        parsed = []
        for row in rows:
            if len(row) != 7:
                raise ValueError("unexpected nvidia-smi column count")
            index, uuid, pci, name, total, free, used = (value.strip() for value in row)
            parsed.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "pci_bus_id": pci.lower(),
                    "name": name,
                    "total": int(total) * 1024**2,
                    "free": int(free) * 1024**2,
                    "used": int(used) * 1024**2,
                }
            )
    except (TypeError, ValueError) as error:
        return unavailable_device_capacity_observation(
            source=source,
            execution_device_identity=execution_device_identity,
            reason=f"invalid nvidia-smi output: {error}",
        )
    stable_request: Mapping[str, object] | None = None
    for key in ("uuid", "mig_uuid", "pci_bus_id"):
        if execution_device_identity.get(key):
            stable_request = {key: execution_device_identity[key]}
            break
    visible = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None
        else cuda_visible_devices
    )
    order = (
        os.environ.get("CUDA_DEVICE_ORDER")
        if cuda_device_order is None
        else cuda_device_order
    )
    visible_index = int(execution_device_identity.get("visible_device_index", -1))
    tokens = [] if visible is None else [token.strip() for token in visible.split(",")]
    token = tokens[visible_index] if 0 <= visible_index < len(tokens) else None
    if stable_request is None and token and token.lower().startswith(("gpu-", "mig-")):
        stable_request = {"uuid": token}
    if stable_request is None and cuda_visible_ordinal_mapper is not None:
        try:
            mapped = cuda_visible_ordinal_mapper(
                visible_index=visible_index,
                cuda_visible_devices=visible,
                cuda_device_order=order,
            )
        except Exception as error:
            return unavailable_device_capacity_observation(
                source=source,
                execution_device_identity=execution_device_identity,
                reason=f"CUDA visible-ordinal mapping failed: {type(error).__name__}: {error}",
            )
        if isinstance(mapped, Mapping):
            stable_request = dict(mapped)

    matches = []
    if stable_request:
        requested_uuid = stable_request.get("uuid") or stable_request.get("mig_uuid")
        requested_pci = stable_request.get("pci_bus_id")
        if requested_uuid:
            matches = [
                row for row in parsed if _uuid_identity_matches(row["uuid"], requested_uuid)
            ]
        elif requested_pci:
            normalized = _normalize_pci_identifier(requested_pci)
            matches = (
                []
                if normalized is None
                else [
                    row
                    for row in parsed
                    if _normalize_pci_identifier(row["pci_bus_id"]) == normalized
                ]
            )
    if len(matches) != 1:
        return unavailable_device_capacity_observation(
            source=source,
            execution_device_identity=execution_device_identity,
            reason=(
                "could not uniquely match stable UUID/MIG/PCI evidence to the JAX device; "
                "numeric CUDA ordinals are never compared with nvidia-smi indices"
            ),
        )
    row = matches[0]
    preallocate = (
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        if preallocate_setting is None
        else preallocate_setting
    )
    fraction_text = (
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
        if memory_fraction_setting is None
        else memory_fraction_setting
    )
    allocator_available = None
    if preallocate is not None and preallocate.strip().lower() in {"false", "0"}:
        allocator_policy = "XLA_PYTHON_CLIENT_PREALLOCATE=false; physical free bytes govern"
    elif fraction_text is not None:
        try:
            fraction = float(fraction_text)
        except ValueError:
            fraction = math.nan
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            return unavailable_device_capacity_observation(
                source=source,
                execution_device_identity=execution_device_identity,
                reason="XLA_PYTHON_CLIENT_MEM_FRACTION is not in (0,1]",
            )
        allocator_available = math.floor(row["total"] * fraction)
        # This comparison is only meaningful before the allocator reserves its
        # pool.  Afterwards the reservation itself is what removed those bytes
        # from the external free total, so repeating it would reject the very
        # pool the policy asked for.
        if not post_initialization and allocator_available > row["free"]:
            return unavailable_device_capacity_observation(
                source=source,
                execution_device_identity=execution_device_identity,
                reason="configured JAX preallocation target exceeds current free memory",
            )
        allocator_policy = (
            "explicit XLA_PYTHON_CLIENT_MEM_FRACTION limits the initial allocator pool"
        )
    else:
        return unavailable_device_capacity_observation(
            source=source,
            execution_device_identity=execution_device_identity,
            reason=(
                "JAX allocator/preallocation policy is not explicit; usable capacity "
                "cannot be verified from nominal/free NVIDIA memory alone"
            ),
        )
    pool_bytes = None
    internal_free = None
    if post_initialization:
        # Requirement: never infer process ownership from an nvidia-smi delta
        # when a direct allocator statistic exists.
        allocator = allocator_capacity_observation(
            allocator_statistics,
            policy="fraction" if allocator_available is not None else "no-preallocation",
        )
        if not allocator.available or allocator.internal_free_bytes is None:
            return unavailable_device_capacity_observation(
                source=source,
                execution_device_identity=execution_device_identity,
                reason=(
                    "post-initialization allocator statistics are unavailable: "
                    f"{allocator.unavailable_reason}"
                ),
            )
        internal_free = allocator.internal_free_bytes
        pool_bytes = allocator.byte_limit
    return make_device_capacity_observation(
        source=source,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        backend="gpu",
        platform="gpu",
        visible_device_index=int(execution_device_identity["visible_device_index"]),
        execution_device_identity=execution_device_identity,
        stable_device_identity={
            "nvidia_index": row["index"],
            "uuid": row["uuid"],
            "pci_bus_id": row["pci_bus_id"],
        },
        device_name=row["name"],
        total_physical_bytes=row["total"],
        free_bytes=row["free"],
        used_bytes=row["used"],
        allocator_available_bytes=allocator_available,
        allocator_policy=allocator_policy,
        post_initialization=post_initialization,
        process_pool_bytes=pool_bytes,
        allocator_internal_free_bytes=internal_free,
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def estimate_separable_resources(
    *,
    grid_size: int,
    dtype_bytes: int,
    steps: int,
    summary_count: int,
    row_block_size: int,
    column_block_size: int,
    return_final_density: bool,
) -> dict[str, int | str | dict[str, int]]:
    """Conservative static lifetimes for the implemented tiled kernel.

    The executable report remains authoritative.  This formula names the
    arrays that exist before compilation: two transport densities, a bounded
    source tile and its weighted copy/mask, endpoint policies, moments,
    destination tables, scan outputs, device initialization inputs, and host
    histogram/grid/output copies.  Two extra density-sized allowances cover
    scatter lowering and validation/compilation staging rather than assuming
    fusion.  All arithmetic is Python ``int`` arithmetic.
    """

    grid_size = _positive_integer(grid_size, "grid_size")
    dtype_bytes = _positive_integer(dtype_bytes, "dtype_bytes")
    row_block_size = _positive_integer(row_block_size, "row_block_size")
    column_block_size = _positive_integer(column_block_size, "column_block_size")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    summary_count = _positive_integer(summary_count, "summary_count")

    points = grid_size * grid_size
    pair_cells = points * points
    state_cells = 2 * pair_cells
    density_bytes = state_cells * dtype_bytes
    row_block = min(row_block_size, points)
    column_block = min(column_block_size, points)
    tile_cells = row_block * column_block

    source_summary_bytes = POINT_SUMMARY_FLOATS * summary_count * points * dtype_bytes
    diagnostic_device_bytes = (
        (steps + 1)
        * (DIAGNOSTIC_FLOAT_SCALARS * dtype_bytes + DIAGNOSTIC_BOOL_SCALARS)
        + steps
    )
    grid_device_bytes = (
        grid_size * dtype_bytes
        + 2 * points * dtype_bytes
        + 2 * points * 4
    )
    initialization_device_bytes = points * dtype_bytes + 2 * dtype_bytes
    # Reuse the audited Phase 4 point-working envelope: endpoint policies,
    # conditional numerators/weights/mean/second/variance, focal/occupied arrays,
    # velocities, decimal projection work, destination/safe-destination tables,
    # and diagnostics' pointwise moment work. It is deliberately larger than
    # the returned tables because several calculation temporaries coexist.
    policy_moment_destination_bytes = points * (20 * dtype_bytes + 40)
    block_workspace_bytes = (
        # source and weighted tiles plus a boolean validity tile
        2 * tile_cells * dtype_bytes
        + tile_cells
        # source/destination indices and validity masks for both axes
        + 2 * row_block * 4
        + row_block
        + 2 * column_block * 4
        + column_block
    )
    retained_device_bytes = source_summary_bytes + diagnostic_device_bytes
    returned_density_bytes = density_bytes if return_final_density else 0
    transport_named_bytes = (
        2 * density_bytes
        + grid_device_bytes
        + initialization_device_bytes
        + policy_moment_destination_bytes
        + block_workspace_bytes
        + retained_device_bytes
    )
    scatter_lowering_allowance_bytes = 2 * density_bytes
    static_device_bytes = (
        transport_named_bytes
        + scatter_lowering_allowance_bytes
        + STATIC_DEVICE_FIXED_BYTES
    )

    host_grid_histogram_bytes = (
        points * 8 + 2 * 8 + grid_size * 8 + 2 * points * 8 + 2 * points * 4
    )
    host_bounded_outputs_bytes = retained_device_bytes
    host_final_density_bytes = returned_density_bytes
    static_host_bytes = (
        host_grid_histogram_bytes
        + host_bounded_outputs_bytes
        + host_final_density_bytes
    )
    heuristic_host_staging_reserve_bytes = 2 * density_bytes
    host_planning_threshold_bytes = static_host_bytes + heuristic_host_staging_reserve_bytes
    return {
        "grid_size": grid_size,
        "agent_grid_points": points,
        "ordered_pair_cells": pair_cells,
        "state_expanded_cells": state_cells,
        "dtype_bytes": dtype_bytes,
        "one_density_bytes": density_bytes,
        "source_summary_bytes": source_summary_bytes,
        "diagnostic_device_bytes": diagnostic_device_bytes,
        "grid_device_bytes": grid_device_bytes,
        "initialization_device_bytes": initialization_device_bytes,
        "policy_moment_destination_bytes": policy_moment_destination_bytes,
        "block_workspace_bytes": block_workspace_bytes,
        "retained_device_bytes": retained_device_bytes,
        "returned_density_bytes": returned_density_bytes,
        "transport_named_bytes": transport_named_bytes,
        "scatter_lowering_allowance_bytes": scatter_lowering_allowance_bytes,
        "static_device_fixed_bytes": STATIC_DEVICE_FIXED_BYTES,
        "static_device_bytes": static_device_bytes,
        "host_grid_histogram_bytes": host_grid_histogram_bytes,
        "host_bounded_outputs_bytes": host_bounded_outputs_bytes,
        "host_final_density_bytes": host_final_density_bytes,
        "static_host_bytes": static_host_bytes,
        # Backward-compatible aliases retain the Phase 6 arithmetic while the
        # new names make clear that no two host pair arrays were observed.
        "compilation_validation_host_allowance_bytes": heuristic_host_staging_reserve_bytes,
        "static_host_with_compilation_bytes": host_planning_threshold_bytes,
        "modeled_coexisting_host_numerical_bytes": static_host_bytes,
        "heuristic_host_staging_reserve_bytes": heuristic_host_staging_reserve_bytes,
        "host_planning_threshold_bytes": host_planning_threshold_bytes,
        "excluded_unbounded_host_classes": (
            "compiler RSS/code cache, Python/library RSS, backend allocator overhead"
        ),
        "formula": (
            "device=named(input+output/optional-return+tile+tables+bounded outputs) "
            "+ two-density scatter/lowering allowance+4KiB fixed control/scalars; "
            "host numerical peak=histogram/grid+bounded outputs+optional final; "
            "planning threshold adds a heuristic two-density staging reserve that is "
            "not an observed host pair-array lifetime and excludes compiler/Python RSS"
        ),
    }


def estimate_flat_resources(
    *,
    grid_size: int,
    dtype_bytes: int,
    steps: int,
    summary_count: int,
    chunk_size: int,
    return_final_density: bool,
) -> dict[str, int | str]:
    """Static counterpart for the committed four-branch flat scatter oracle."""

    base = estimate_separable_resources(
        grid_size=grid_size,
        dtype_bytes=dtype_bytes,
        steps=steps,
        summary_count=summary_count,
        row_block_size=1,
        column_block_size=1,
        return_final_density=return_final_density,
    )
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    effective_chunk = min(chunk_size, int(base["state_expanded_cells"]))
    # Retain the committed Phase 4 audited flat envelope: source/policy/branch
    # floating work is 17*K*b and branch/source/destination integer work is 96*K.
    chunk_workspace = effective_chunk * (17 * dtype_bytes + 96)
    static_device = (
        8 * int(base["one_density_bytes"])
        + chunk_workspace
        + int(base["grid_device_bytes"])
        + int(base["policy_moment_destination_bytes"])
        + int(base["retained_device_bytes"])
        + int(base["initialization_device_bytes"])
        + STATIC_DEVICE_FIXED_BYTES
    )
    return {
        **base,
        "chunk_workspace_bytes": chunk_workspace,
        "static_device_bytes": static_device,
        "formula": (
            "eight full-density allowances plus flat D-block four-branch "
            "mass/destination workspace, tables and retained outputs"
        ),
    }


def full_grid_feasibility(
    *,
    dtype_bytes: int,
    representative_steps: int,
    representative_summary_count: int,
    row_block_size: int,
    column_block_size: int,
    validated_compiled_bytes_per_density_byte: float,
    safety_margin_fraction: float,
) -> dict[str, int | float | str]:
    """Project the exact G=131 shapes without constructing any array."""

    ratio = float(validated_compiled_bytes_per_density_byte)
    margin = float(safety_margin_fraction)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("validated compiled ratio must be finite and positive")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("safety margin must be finite and non-negative")
    estimate = estimate_separable_resources(
        grid_size=131,
        dtype_bytes=dtype_bytes,
        steps=representative_steps,
        summary_count=representative_summary_count,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
        return_final_density=False,
    )
    density = int(estimate["one_density_bytes"])
    ratio_projection = math.ceil(ratio * density)
    expected_compiled = max(int(estimate["static_device_bytes"]), ratio_projection)
    minimum_device = math.ceil(expected_compiled * (1.0 + margin))
    host_numerical = int(estimate["modeled_coexisting_host_numerical_bytes"])
    host_reserve = int(estimate["heuristic_host_staging_reserve_bytes"])
    host = int(estimate["host_planning_threshold_bytes"])
    return {
        "grid_size": 131,
        "agent_grid_points": int(estimate["agent_grid_points"]),
        "state_expanded_cells": int(estimate["state_expanded_cells"]),
        "dtype_bytes": dtype_bytes,
        "one_density_bytes": density,
        "static_separable_device_bytes": int(estimate["static_device_bytes"]),
        "validated_compiled_bytes_per_density_byte": ratio,
        "ratio_projected_compiled_bytes": ratio_projection,
        "expected_compiled_device_requirement_bytes": expected_compiled,
        "compiled_projection_kind": "empirical small-CPU planning projection",
        "compiled_projection_is_formal_bound": False,
        "source_summary_bytes": int(estimate["source_summary_bytes"]),
        "modeled_coexisting_host_numerical_bytes": host_numerical,
        "heuristic_host_staging_reserve_bytes": host_reserve,
        "host_requirement_bytes": host,
        "host_requirement_kind": "planning threshold, not measured process RSS",
        "excluded_unbounded_host_classes": estimate["excluded_unbounded_host_classes"],
        "safety_margin_fraction": margin,
        "minimum_device_memory_bytes": minimum_device,
        "execution_permitted": False,
        "scope": "allocation-free feasibility projection; not a full-grid compilation",
    }


def _observation_age_seconds(observed_at_utc: str, now: datetime) -> float:
    try:
        observed = datetime.fromisoformat(observed_at_utc)
    except (TypeError, ValueError) as error:
        raise ValueError("capacity timestamp is not valid ISO-8601") from error
    if observed.tzinfo is None:
        raise ValueError("capacity timestamp must include a timezone")
    if now.tzinfo is None:
        raise ValueError("capacity comparison time must include a timezone")
    age = (now.astimezone(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
    if not math.isfinite(age):
        raise ValueError("capacity evidence age is not finite")
    return age


def _utc_now() -> datetime:
    """Internal clock seam; tests may monkeypatch it but production callers cannot."""

    return datetime.now(timezone.utc)


def _gpu_capacity_preflight(
    *,
    feasibility: dict,
    bundle: CompiledExecutableBundle,
    capacity_observation: DeviceCapacityObservation,
    allow_expensive: bool,
    required_kernel: str,
) -> dict[str, object]:
    """Require exact executable identity and fresh matched usable GPU capacity.

    Every failure here is a production identity/capacity failure and therefore
    remains non-overridable. ``allow_expensive`` is recorded only to demonstrate
    that it cannot convert unknown evidence into a production pass.
    """

    violations: list[str] = []
    try:
        live_report = validate_compiled_executable_bundle(bundle)
    except (TypeError, ValueError):
        violations.append("exact_executable_identity_invalid")
        live_report = {}
    signature = bundle.compile_signature if isinstance(bundle, CompiledExecutableBundle) else {}
    report = bundle.memory_report if isinstance(bundle, CompiledExecutableBundle) else {}
    if signature.get("kernel") != required_kernel:
        violations.append(f"required_kernel_not_{required_kernel}")
    if signature.get("backend") != "gpu" or signature.get("platform") != "gpu":
        violations.append("production_backend_not_gpu")
    if signature.get("state_expanded_cells") != feasibility.get("state_expanded_cells"):
        violations.append("production_grid_shape_mismatch")
    required_fields = (
        "argument_bytes",
        "output_bytes",
        "temporary_bytes",
        "alias_bytes",
        "host_argument_bytes",
        "host_output_bytes",
        "host_temporary_bytes",
        "host_alias_bytes",
        "compiled_device_requirement_bytes",
        "compiled_host_requirement_bytes",
    )
    values = [report.get(name) for name in required_fields]
    valid_values = all(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0
        for value in values
    )
    compiled_requirement = None
    if valid_values:
        (
            argument,
            output,
            temporary,
            alias,
            host_argument,
            host_output,
            host_temporary,
            host_alias,
            claimed,
            claimed_host,
        ) = values
        if (
            alias <= argument + output
            and host_alias <= host_argument + host_output
            and argument + output + temporary - alias == claimed
            and host_argument + host_output + host_temporary - host_alias
            == claimed_host
        ):
            compiled_requirement = claimed
    if (
        report.get("available") is not True
        or report.get("analysis_status") != "complete"
        or compiled_requirement is None
    ):
        violations.append("exact_compiled_analysis_incomplete")

    observation = capacity_observation
    age = None
    usable = None
    if not isinstance(observation, DeviceCapacityObservation) or not observation.available:
        violations.append("verified_device_capacity_unavailable")
    else:
        if (
            len(observation.source) > MAX_CAPACITY_TEXT_CHARS
            or len(observation.device_name) > MAX_CAPACITY_TEXT_CHARS
            or len(observation.allocator_policy) > MAX_CAPACITY_TEXT_CHARS
            or len(observation.usable_bytes_definition) > MAX_CAPACITY_TEXT_CHARS
            or len(observation.stable_device_identity) > MAX_STABLE_IDENTITY_FIELDS
        ):
            violations.append("device_capacity_metadata_unbounded")
        now = _utc_now()
        try:
            age = _observation_age_seconds(observation.observed_at_utc, now)
        except ValueError:
            violations.append("device_capacity_timestamp_invalid")
        else:
            if age < -MAX_CLOCK_SKEW_SECONDS or age > MAX_CAPACITY_EVIDENCE_AGE_SECONDS:
                violations.append("device_capacity_evidence_stale")
        execution_device = signature.get("execution_device")
        if not isinstance(execution_device, Mapping):
            violations.append("production_execution_device_invalid")
            execution_device = {}
        if observation.execution_device_identity != execution_device:
            violations.append("device_capacity_identity_mismatch")
        elif observation.visible_device_index != execution_device.get(
            "visible_device_index"
        ):
            violations.append("device_capacity_visible_index_mismatch")
        if observation.backend != "gpu" or observation.platform != "gpu":
            violations.append("device_capacity_backend_mismatch")
        if not observation.stable_device_identity or not any(
            name in observation.stable_device_identity
            for name in ("uuid", "mig_uuid", "pci_bus_id")
        ):
            violations.append("stable_device_identity_unavailable")
        else:
            execution_uuid = execution_device.get("uuid") or execution_device.get("mig_uuid")
            observed_uuid = observation.stable_device_identity.get("uuid") or observation.stable_device_identity.get("mig_uuid")
            execution_pci = execution_device.get("pci_bus_id")
            observed_pci = observation.stable_device_identity.get("pci_bus_id")
            if execution_uuid and (
                not observed_uuid
                or not _uuid_identity_matches(observed_uuid, execution_uuid)
            ):
                violations.append("stable_device_identity_mismatch")
            if execution_pci and (
                not observed_pci
                or _normalize_pci_identifier(observed_pci)
                != _normalize_pci_identifier(execution_pci)
            ):
                violations.append("stable_device_identity_mismatch")
        if observation.free_bytes is None or observation.usable_device_bytes is None:
            violations.append("usable_device_capacity_unavailable")
        else:
            if observation.post_initialization:
                # The pool this process owns is the only memory the executable
                # can draw on; intersecting it with the external free bytes
                # beside the reservation would double count.
                expected_usable = observation.allocator_available_bytes
            else:
                expected_usable = observation.free_bytes
                if observation.allocator_available_bytes is not None:
                    expected_usable = min(expected_usable, observation.allocator_available_bytes)
            if expected_usable is None:
                violations.append("usable_device_capacity_unavailable")
            elif expected_usable != observation.usable_device_bytes:
                violations.append("usable_device_capacity_derivation_invalid")
            else:
                usable = expected_usable
    exact_with_margin = None
    required_with_margin = int(feasibility["minimum_device_memory_bytes"])
    if compiled_requirement is not None:
        exact_with_margin = math.ceil(
            compiled_requirement
            * (1.0 + float(feasibility["safety_margin_fraction"]))
        )
        required_with_margin = max(required_with_margin, exact_with_margin)
    if usable is not None and usable < required_with_margin:
        violations.append("insufficient_verified_usable_device_memory")
    if violations:
        raise ValueError(
            "production identity/capacity preflight failed closed and is not "
            f"overridable: {violations}"
        )
    return {
        "passed": True,
        "allow_expensive_requested_but_not_applicable": bool(allow_expensive),
        "violations": [],
        "violations_overridden": [],
        "analysis_status": "complete",
        "compiled_device_requirement_bytes": compiled_requirement,
        "exact_compiled_bytes_with_margin": exact_with_margin,
        "planning_minimum_device_bytes": int(feasibility["minimum_device_memory_bytes"]),
        "required_device_bytes_with_margin": required_with_margin,
        "verified_usable_device_bytes": usable,
        "capacity_evidence_age_seconds": age,
        "capacity_observation": asdict(observation),
        "capacity_matches_execution_device": True,
        "live_bundle_validation": {
            "compiled_device_requirement_bytes": live_report.get(
                "compiled_device_requirement_bytes"
            ),
            "bundle_integrity_sha256": bundle.bundle_integrity_sha256,
        },
    }


def production_capacity_preflight(
    *,
    feasibility: dict,
    bundle: CompiledExecutableBundle,
    capacity_observation: DeviceCapacityObservation,
    allow_expensive: bool,
) -> dict[str, object]:
    """Fail-closed capacity gate fixed to the production separable kernel."""

    return _gpu_capacity_preflight(
        feasibility=feasibility,
        bundle=bundle,
        capacity_observation=capacity_observation,
        allow_expensive=allow_expensive,
        required_kernel="separable",
    )


def flat_validation_capacity_preflight(
    *,
    feasibility: dict,
    bundle: CompiledExecutableBundle,
    capacity_observation: DeviceCapacityObservation,
) -> dict[str, object]:
    """Fail-closed capacity gate fixed to the small flat validation oracle."""

    return _gpu_capacity_preflight(
        feasibility=feasibility,
        bundle=bundle,
        capacity_observation=capacity_observation,
        allow_expensive=False,
        required_kernel="flat",
    )

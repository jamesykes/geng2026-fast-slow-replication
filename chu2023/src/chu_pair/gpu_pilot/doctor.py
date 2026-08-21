"""GPU environment and stable-capacity diagnostics with bounded provenance."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable

from .cuda_identity import CudaDriverIdentityProvider, CudaIdentityError


MAX_DOCTOR_REPORT_BYTES = 128 * 1024
MAX_NVIDIA_DRIVER_OUTPUT_BYTES = 4096
SOURCE_HASH_BUFFER_BYTES = 1 << 20
EXPECTED_JAX_VERSION = "0.7.2"
EXPECTED_NUMPY_VERSION = "2.2.4"
_MINIMUM_DRIVER_MAJOR = {"cuda12": 525, "cuda13": 580}
_ALLOWED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "JAX_PLATFORM_NAME",
    "JAX_ENABLE_X64",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
)


def _host_memory() -> dict[str, object]:
    """Read bounded Linux host-memory evidence without another dependency."""

    path = Path("/proc/meminfo")
    values: dict[str, int] = {}
    if path.exists() and path.stat().st_size <= 64 * 1024:
        for line in path.read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0] in {"MemTotal:", "MemAvailable:"} and fields[2] == "kB":
                values[fields[0][:-1]] = int(fields[1]) * 1024
    return {
        "source": "/proc/meminfo" if values else "unavailable",
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
    }


def _nvidia_driver_report(command_runner: Callable[..., object]) -> dict[str, object]:
    try:
        completed = command_runner(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        output = completed.stdout
        if not isinstance(output, str) or len(output.encode("utf-8")) > MAX_NVIDIA_DRIVER_OUTPUT_BYTES:
            raise ValueError("driver query output is missing or over bound")
        versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
        if len(versions) != 1 or len(versions[0]) > 64:
            raise ValueError("driver query did not return one bounded common version")
        return {"available": True, "version": versions[0], "source": "nvidia-smi"}
    except Exception as error:
        return {
            "available": False, "version": None, "source": "nvidia-smi",
            "unavailable_reason": f"{type(error).__name__}: {error}"[:512],
        }


def _driver_compatibility(cuda_family: str, driver: dict[str, object]) -> dict[str, object]:
    """Parse one bounded Linux NVIDIA driver version without guesswork."""

    required = _MINIMUM_DRIVER_MAJOR.get(cuda_family)
    if required is None:
        raise ValueError("cuda family has no reviewed driver minimum")
    text = driver.get("version")
    major = None
    if isinstance(text, str):
        parts = text.strip().split(".")
        if 2 <= len(parts) <= 4 and all(part.isascii() and part.isdecimal() for part in parts):
            major = int(parts[0])
    return {
        "cuda_family": cuda_family,
        "observed_version": text,
        "observed_major": major,
        "required_minimum_major": required,
        "compatible": bool(driver.get("available") and major is not None and major >= required),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(SOURCE_HASH_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=project_root, check=True,
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()


def _device_identity(jax_module, device, index: int, count: int) -> dict[str, object]:
    identity = {
        "backend": str(jax_module.default_backend()),
        "platform": str(device.platform),
        "visible_device_index": int(index),
        "visible_device_count": int(count),
        "id": str(device.id),
        "device_kind": str(device.device_kind)[:256],
        "process_index": int(getattr(device, "process_index", 0)),
        "local_hardware_id": int(getattr(device, "local_hardware_id", index)),
    }
    for name in ("uuid", "pci_bus_id"):
        value = getattr(device, name, None)
        if value is not None:
            identity[name] = str(value)[:256]
    return identity


def environment_sha256(report: dict[str, object]) -> str:
    payload = {
        "python": report["python"],
        "host_total_bytes": report["host_memory"]["total_bytes"],
        "nvidia_driver_version": report["nvidia_driver"]["version"],
        "versions": report["versions"],
        "backend": report["backend"],
        "jax_enable_x64": report["jax_enable_x64"],
        "devices": report["devices"],
        "allocator": report["allocator_environment"],
        "cuda_family": report["cuda_family"],
        "pair_contraction_precision": report["pair_contraction_precision"],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def collect_gpu_doctor_report(
    *,
    project_root: Path,
    jax_module,
    numpy_module,
    jaxlib_module,
    cuda_family: str,
    identity_provider_factory: Callable[[], CudaDriverIdentityProvider] = CudaDriverIdentityProvider.from_system,
    capacity_command_runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Collect a whitelisted report; CPU is represented as expected unavailability."""

    from ..pair_density import pair_contraction_precision
    from ..pair_density.separable_resources import discover_nvidia_device_capacity

    if cuda_family not in {"cuda12", "cuda13"}:
        raise ValueError("cuda_family must be cuda12 or cuda13")
    devices = jax_module.devices()
    if not 1 <= len(devices) <= 16:
        raise RuntimeError("JAX device count is outside the fixed doctor bound")
    identities = [
        _device_identity(jax_module, device, index, len(devices))
        for index, device in enumerate(devices)
    ]
    execution = identities[0]
    mapper = None
    provider_error = None
    if execution["backend"] == "gpu" and execution["platform"] == "gpu":
        try:
            mapper = identity_provider_factory().map_visible_ordinal
        except CudaIdentityError as error:
            provider_error = f"{type(error).__name__}: {error}"[:512]
    capacity = discover_nvidia_device_capacity(
        execution,
        command_runner=capacity_command_runner,
        cuda_visible_ordinal_mapper=mapper,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        preallocate_setting=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        memory_fraction_setting=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )
    driver = (
        _nvidia_driver_report(capacity_command_runner)
        if execution["backend"] == "gpu" and execution["platform"] == "gpu"
        else {"available": False, "version": None, "source": "not queried on non-GPU backend"}
    )
    driver_compatibility = _driver_compatibility(cuda_family, driver)
    commit = _git(project_root, "rev-parse", "HEAD")
    clean = not bool(_git(project_root, "status", "--porcelain", "--", "."))
    relevant = (
        project_root / "src/chu_pair/model.py",
        project_root / "src/chu_pair/initial_conditions.py",
        project_root / "src/chu_pair/pair_density/jax_solver.py",
        project_root / "src/chu_pair/pair_density/separable_resources.py",
        project_root / "src/chu_pair/gpu_pilot/allocator.py",
        project_root / "src/chu_pair/gpu_pilot/cuda_identity.py",
        project_root / "src/chu_pair/gpu_pilot/doctor.py",
        project_root / "src/chu_pair/gpu_pilot/runtime.py",
        project_root / "src/chu_pair/gpu_pilot/workflow.py",
        project_root / "experiments/run_pair_separable_benchmark.py",
        project_root / "experiments/run_gpu_doctor.py",
        project_root / "experiments/run_gpu_pilot.py",
    )
    plugin_distribution = f"jax-{cuda_family}-plugin"
    try:
        plugin_version = metadata.version(plugin_distribution)
    except metadata.PackageNotFoundError:
        plugin_version = None
    report: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "operating_system": platform.system(),
            "architecture": platform.machine(),
        },
        "versions": {
            "jax": str(jax_module.__version__),
            "jaxlib": str(jaxlib_module.__version__),
            "numpy": str(numpy_module.__version__),
            "cuda_plugin_distribution": plugin_distribution,
            "cuda_plugin_version": plugin_version,
        },
        "expected_versions": {
            "jax": EXPECTED_JAX_VERSION,
            "jaxlib": EXPECTED_JAX_VERSION,
            "numpy": EXPECTED_NUMPY_VERSION,
        },
        "cuda_family": cuda_family,
        # Explicit float32 dot-product policy: part of numerical identity, so
        # a change invalidates this doctor and every artifact bound to it.
        "pair_contraction_precision": pair_contraction_precision(),
        "host_memory": _host_memory(),
        "backend": str(jax_module.default_backend()),
        "jax_enable_x64": bool(jax_module.config.read("jax_enable_x64")),
        "devices": identities,
        "allocator_environment": {
            name: os.environ[name][:512]
            for name in _ALLOWED_ENVIRONMENT if name in os.environ
        },
        "capacity": asdict(capacity),
        "capacity_evidence_max_age_seconds": 60,
        "capacity_device_match_available": bool(capacity.available),
        "nvidia_driver": driver,
        "driver_compatibility": driver_compatibility,
        "cuda_identity_provider_error": provider_error,
        "git": {"commit": commit, "subproject_clean": clean},
        "source_hashes": {str(path.relative_to(project_root)): _sha256(path) for path in relevant},
        "limitations": [
            "capacity is a bounded point-in-time estimate, not a guarantee",
            "MIG capacity fails closed until matched slice-level memory evidence is available",
            "reproducibility is limited to the recorded software/backend/device configuration",
        ],
    }
    report["environment_sha256"] = environment_sha256(report)
    report["gpu_ready"] = bool(
        report["backend"] == "gpu"
        and capacity.available
        and driver["available"]
        and driver_compatibility["compatible"]
        and clean
        and report["versions"]["jax"] == EXPECTED_JAX_VERSION
        and report["versions"]["jaxlib"] == EXPECTED_JAX_VERSION
        and report["versions"]["numpy"] == EXPECTED_NUMPY_VERSION
        and plugin_version == EXPECTED_JAX_VERSION
        and report["python"]["operating_system"] == "Linux"
        and report["python"]["architecture"] in {"x86_64", "aarch64"}
        and report["host_memory"]["total_bytes"] is not None
        and report["host_memory"]["available_bytes"] is not None
    )
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True).encode("ascii")
    if len(encoded) > MAX_DOCTOR_REPORT_BYTES:
        raise RuntimeError("GPU doctor report exceeds its fixed serialization bound")
    return report

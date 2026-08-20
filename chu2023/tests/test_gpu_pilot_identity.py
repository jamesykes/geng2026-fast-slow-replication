from __future__ import annotations

from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chu_pair.gpu_pilot.allocator import apply_allocator_policy
from chu_pair.gpu_pilot.cuda_identity import (
    CudaDriverIdentityProvider,
    CudaIdentityError,
)
from chu_pair.pair_density import discover_nvidia_device_capacity
import chu_pair.pair_density.separable_resources as resources
import chu_pair.gpu_pilot.doctor as doctor
from scripts import prepare_gpu_environment as setup
from experiments.run_gpu_doctor import doctor_exit_code


class FakeDriver:
    def __init__(self, identities):
        self.identities = identities
        self.initialized = 0

    def initialize(self):
        self.initialized += 1

    def device_count(self):
        return len(self.identities)

    def stable_identity(self, ordinal):
        return self.identities[ordinal]


def _identity(number: int) -> dict[str, str]:
    return {
        "uuid": f"GPU-00000000-0000-0000-0000-{number:012d}",
        "pci_bus_id": f"0000:{number + 1:02x}:00.0",
    }


def test_allocator_policy_is_explicit_and_must_precede_jax() -> None:
    environment = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.1"}
    report = apply_allocator_policy(
        "fraction", memory_fraction=0.85, environment=environment, loaded_modules={}
    )
    assert environment == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.84999999999999998",
    }
    assert report["name"] == "fraction"
    with pytest.raises(RuntimeError, match="before importing JAX"):
        apply_allocator_policy("default", environment={}, loaded_modules={"jax": object()})


def test_driver_mapper_resolves_numeric_visibility_in_cuda_visible_order() -> None:
    driver = FakeDriver([_identity(2), _identity(0)])
    provider = CudaDriverIdentityProvider(driver)

    mapped = provider.map_visible_ordinal(
        visible_index=0, cuda_visible_devices="2,0", cuda_device_order="PCI_BUS_ID"
    )

    assert mapped == _identity(2)
    assert driver.initialized == 1


def test_driver_mapper_uses_visible_uuid_and_handles_mig_fail_closed() -> None:
    driver = FakeDriver([_identity(3)])
    provider = CudaDriverIdentityProvider(driver)
    assert provider.map_visible_ordinal(
        visible_index=0,
        cuda_visible_devices=_identity(3)["uuid"],
        cuda_device_order=None,
    ) == _identity(3)
    mig = "MIG-12345678-1234-1234-1234-123456789abc"
    assert provider.map_visible_ordinal(
        visible_index=0, cuda_visible_devices=mig, cuda_device_order=None
    ) == {"mig_uuid": mig}


def test_driver_mapper_rejects_malformed_visibility_and_out_of_range() -> None:
    provider = CudaDriverIdentityProvider(FakeDriver([_identity(0)]))
    with pytest.raises(CudaIdentityError, match="unsupported"):
        provider.map_visible_ordinal(
            visible_index=0, cuda_visible_devices="../../bad", cuda_device_order=None
        )
    with pytest.raises(CudaIdentityError, match="outside"):
        provider.map_visible_ordinal(
            visible_index=1, cuda_visible_devices="0", cuda_device_order=None
        )


def test_system_provider_fails_closed_for_missing_library_and_symbols() -> None:
    def missing(path):
        raise OSError("missing")

    with pytest.raises(CudaIdentityError, match="every fixed path"):
        CudaDriverIdentityProvider.from_system(loader=missing)
    with pytest.raises(CudaIdentityError, match="symbol cuInit"):
        CudaDriverIdentityProvider.from_system(loader=lambda path: SimpleNamespace())


def test_numeric_visibility_capacity_matches_by_provider_uuid_not_smi_index() -> None:
    execution = {
        "backend": "gpu", "platform": "gpu", "visible_device_index": 0,
        "visible_device_count": 1, "id": "0", "device_kind": "GPU",
        "process_index": 0, "local_hardware_id": 0,
    }
    provider = CudaDriverIdentityProvider(FakeDriver([_identity(2)]))
    output = (
        "0, GPU-00000000-0000-0000-0000-000000000000, 0000:01:00.0, A, 100, 90, 10\n"
        "7, GPU-00000000-0000-0000-0000-000000000002, 0000:03:00.0, B, 200, 180, 20\n"
    )

    def run(*args, **kwargs):
        return SimpleNamespace(stdout=output)

    observation = discover_nvidia_device_capacity(
        execution, command_runner=run,
        cuda_visible_ordinal_mapper=provider.map_visible_ordinal,
        cuda_visible_devices="2", cuda_device_order="PCI_BUS_ID",
        preallocate_setting="false", memory_fraction_setting=None,
    )

    assert observation.available
    assert observation.stable_device_identity["uuid"].endswith("000000000002")
    assert observation.total_physical_bytes == 200 * 1024**2


def test_mig_visibility_does_not_claim_parent_gpu_capacity() -> None:
    execution = {
        "backend": "gpu", "platform": "gpu", "visible_device_index": 0,
        "visible_device_count": 1, "id": "0", "device_kind": "MIG",
        "process_index": 0, "local_hardware_id": 0,
    }
    output = "0, GPU-00000000-0000-0000-0000-000000000000, 0000:01:00.0, A, 100, 90, 10\n"
    observation = discover_nvidia_device_capacity(
        execution, command_runner=lambda *a, **k: SimpleNamespace(stdout=output),
        cuda_visible_ordinal_mapper=CudaDriverIdentityProvider(
            FakeDriver([_identity(0)])
        ).map_visible_ordinal,
        cuda_visible_devices="MIG-12345678-1234-1234-1234-123456789abc",
        cuda_device_order=None, preallocate_setting="false",
        memory_fraction_setting=None,
    )
    assert not observation.available
    assert "could not uniquely match" in observation.unavailable_reason


@pytest.mark.parametrize(
    ("name", "total_mib"),
    [("NVIDIA H100 PCIe", 81_920), ("NVIDIA RTX A6000", 49_152), ("24GB test GPU", 24_576)],
)
def test_capacity_provider_preserves_portable_gpu_models_and_bytes(name, total_mib) -> None:
    execution = {
        "backend": "gpu", "platform": "gpu", "visible_device_index": 0,
        "visible_device_count": 1, "id": "0", "device_kind": name,
        "process_index": 0, "local_hardware_id": 0,
    }
    stable = _identity(0)
    output = (
        f"4, {stable['uuid']}, {stable['pci_bus_id']}, {name}, "
        f"{total_mib}, {total_mib - 1024}, 1024\n"
    )
    observation = discover_nvidia_device_capacity(
        execution, command_runner=lambda *a, **k: SimpleNamespace(stdout=output),
        cuda_visible_ordinal_mapper=lambda **kwargs: stable,
        cuda_visible_devices="0", cuda_device_order=None,
        preallocate_setting="false", memory_fraction_setting=None,
    )
    assert observation.available
    assert observation.device_name == name
    assert observation.total_physical_bytes == total_mib * 1024**2


def test_pci_only_mapping_normalizes_case_and_wrong_or_ambiguous_devices_fail() -> None:
    execution = {
        "backend": "gpu", "platform": "gpu", "visible_device_index": 0,
        "visible_device_count": 1, "id": "0", "device_kind": "GPU",
        "process_index": 0, "local_hardware_id": 0,
    }
    rows = (
        "0, GPU-00000000-0000-0000-0000-000000000000, 00000000:0A:00.0, A, 100, 90, 10\n"
    )
    accepted = discover_nvidia_device_capacity(
        execution, command_runner=lambda *a, **k: SimpleNamespace(stdout=rows),
        cuda_visible_ordinal_mapper=lambda **kwargs: {"pci_bus_id": "0000:0a:00.0"},
        cuda_visible_devices="0", cuda_device_order=None,
        preallocate_setting="false", memory_fraction_setting=None,
    )
    assert accepted.available
    wrong = discover_nvidia_device_capacity(
        execution, command_runner=lambda *a, **k: SimpleNamespace(stdout=rows),
        cuda_visible_ordinal_mapper=lambda **kwargs: _identity(9),
        cuda_visible_devices="0", cuda_device_order=None,
        preallocate_setting="false", memory_fraction_setting=None,
    )
    assert not wrong.available
    ambiguous_rows = rows + rows.replace("0,", "1,", 1)
    ambiguous = discover_nvidia_device_capacity(
        execution, command_runner=lambda *a, **k: SimpleNamespace(stdout=ambiguous_rows),
        cuda_visible_ordinal_mapper=lambda **kwargs: {"pci_bus_id": "0000:0a:00.0"},
        cuda_visible_devices="0", cuda_device_order=None,
        preallocate_setting="false", memory_fraction_setting=None,
    )
    assert not ambiguous.available


def test_cpu_doctor_is_bounded_and_does_not_serialize_unrelated_environment(
    monkeypatch,
) -> None:
    class Config:
        @staticmethod
        def read(name):
            assert name == "jax_enable_x64"
            return False

    device = SimpleNamespace(
        platform="cpu", id=0, device_kind="cpu", process_index=0,
        local_hardware_id=0,
    )
    fake_jax = SimpleNamespace(
        __version__="0.7.2", config=Config(), devices=lambda: [device],
        default_backend=lambda: "cpu",
    )
    monkeypatch.setenv("PILOT_SECRET_SHOULD_NOT_APPEAR", "sensitive-value")
    monkeypatch.setattr(
        doctor, "_git",
        lambda root, *args: "abc" if args[0] == "rev-parse" else "",
    )
    monkeypatch.setattr(
        doctor.metadata, "version",
        lambda name: (_ for _ in ()).throw(doctor.metadata.PackageNotFoundError()),
    )
    report = doctor.collect_gpu_doctor_report(
        project_root=Path(__file__).resolve().parents[1],
        jax_module=fake_jax,
        numpy_module=SimpleNamespace(__version__="2.2.4"),
        jaxlib_module=SimpleNamespace(__version__="0.7.2"),
        cuda_family="cuda12",
        identity_provider_factory=lambda: pytest.fail("CUDA provider entered on CPU"),
        capacity_command_runner=lambda *a, **k: pytest.fail("nvidia-smi entered on CPU"),
    )
    encoded = json.dumps(report, ensure_ascii=True)
    assert report["backend"] == "cpu"
    assert report["jax_enable_x64"] is False
    assert report["capacity"]["available"] is False
    assert "sensitive-value" not in encoded
    assert doctor_exit_code(gpu_ready=False, expect_gpu=False) == 0
    assert doctor_exit_code(gpu_ready=False, expect_gpu=True) == 2


def test_post_initialization_fraction_pool_uses_internal_free_not_global_free(monkeypatch) -> None:
    gib = 1024**3
    monkeypatch.setattr(resources, "validate_compiled_executable_bundle", lambda bundle: {
        "argument_bytes": 20 * gib, "output_bytes": 10 * gib,
        "temporary_bytes": 5 * gib, "alias_bytes": 2 * gib,
    })
    external = resources.DeviceCapacityObservation(
        True, "test", "2026-01-01T00:00:00+00:00", "gpu", "gpu", 0, {}, {}, "GPU",
        80 * gib, 12 * gib, 68 * gib, None, 68 * gib, "fraction", 12 * gib, "test",
    )
    pool = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * gib, "bytes_in_use": 40 * gib,
         "peak_bytes_in_use": 40 * gib, "largest_free_block_bytes": 24 * gib},
        policy="fraction",
    )
    admitted = resources.post_initialization_capacity_preflight(
        feasibility={"safety_margin_fraction": 0.0}, bundle=object(),
        external_capacity=external, allocator_capacity=pool,
    )
    assert admitted["admitted_usable_bytes"] == 24 * gib
    assert admitted["already_resident_argument_bytes"] == 20 * gib
    assert admitted["incremental_executable_requirement_bytes"] == 13 * gib
    insufficient = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * gib, "bytes_in_use": 60 * gib,
         "peak_bytes_in_use": 60 * gib, "largest_free_block_bytes": 8 * gib},
        policy="fraction",
    )
    with pytest.raises(ValueError, match="insufficient"):
        resources.post_initialization_capacity_preflight(
            feasibility={"safety_margin_fraction": 0.0}, bundle=object(),
            external_capacity=external, allocator_capacity=insufficient,
        )
    unavailable = resources.allocator_capacity_observation({}, policy="fraction")
    with pytest.raises(ValueError, match="unavailable"):
        resources.post_initialization_capacity_preflight(
            feasibility={"safety_margin_fraction": 0.0}, bundle=object(),
            external_capacity=external, allocator_capacity=unavailable,
        )


@pytest.mark.parametrize(("family", "version", "accepted"), [
    ("cuda12", "524.99", False), ("cuda12", "525.60.13", True),
    ("cuda13", "579.1", False), ("cuda13", " 580.2 ", True),
    ("cuda13", "bad", False), ("cuda12", None, False),
])
def test_driver_family_minimums_are_explicit(family, version, accepted) -> None:
    result = doctor._driver_compatibility(family, {"available": version is not None, "version": version})
    assert result["compatible"] is accepted
    assert result["required_minimum_major"] == (525 if family == "cuda12" else 580)
    with pytest.raises(ValueError):
        doctor._driver_compatibility("cuda99", {"available": True, "version": "999.1"})


def test_setup_preview_uses_metadata_without_backend_import(monkeypatch) -> None:
    monkeypatch.setattr(setup.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: pytest.fail("preview spawned child"))
    report = setup._inspect_existing_python()
    assert report["jax"] == "test-version"
    environment = setup._allocator_environment("fraction", 0.85)
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.84999999999999998"


def test_controlled_setup_child_checks_allocator_before_fake_jax_import(tmp_path) -> None:
    (tmp_path / "jax.py").write_text("__version__='0.7.2'\n")
    (tmp_path / "jaxlib.py").write_text("__version__='0.7.2'\n")
    (tmp_path / "numpy.py").write_text("__version__='2.2.4'\n")
    environment = setup._allocator_environment("fraction", 0.85)
    environment["PYTHONPATH"] = str(tmp_path)
    command = [sys.executable, *setup._validation_command("fraction", 0.85)]
    completed = subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=10)
    assert json.loads(completed.stdout)["allocator_policy"] == "fraction"
    bad = dict(environment); bad.pop("XLA_PYTHON_CLIENT_PREALLOCATE")
    failed = subprocess.run(command, env=bad, check=False, capture_output=True, text=True, timeout=10)
    assert failed.returncode != 0 and not failed.stdout
    mismatch = dict(environment); mismatch.pop("XLA_PYTHON_CLIENT_MEM_FRACTION")
    failed = subprocess.run(command, env=mismatch, check=False, capture_output=True, text=True, timeout=10)
    assert failed.returncode != 0 and not failed.stdout

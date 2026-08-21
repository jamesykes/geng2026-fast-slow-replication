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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


GIB = 1024**3


def _smi_row(total_mib: int, free_mib: int, used_mib: int) -> str:
    return (
        f"0, GPU-00000000-0000-0000-0000-000000000000, 0000:01:00.0, H100, "
        f"{total_mib}, {free_mib}, {used_mib}\n"
    )


def _execution_identity() -> dict[str, object]:
    return {
        "backend": "gpu", "platform": "gpu", "visible_device_index": 0,
        "visible_device_count": 1, "id": "0", "device_kind": "H100",
        "process_index": 0, "local_hardware_id": 0,
        "uuid": "GPU-00000000-0000-0000-0000-000000000000",
    }


def _discover(*, free_mib, post_initialization, statistics=None, total_mib=81920):
    return discover_nvidia_device_capacity(
        _execution_identity(),
        command_runner=lambda *a, **k: SimpleNamespace(
            stdout=_smi_row(total_mib, free_mib, total_mib - free_mib)
        ),
        cuda_visible_ordinal_mapper=None,
        cuda_visible_devices=None, cuda_device_order=None,
        preallocate_setting="true", memory_fraction_setting="0.85",
        post_initialization=post_initialization,
        allocator_statistics=statistics,
    )


def test_pre_initialization_checks_fraction_target_against_physical_capacity() -> None:
    """The fraction target is verified while the pool is still unreserved."""

    cold = _discover(free_mib=81400, post_initialization=False)
    assert cold.available and cold.post_initialization is False
    # 0.85 * 80 GiB fits in ~79.5 GiB of genuinely free physical memory.
    assert cold.allocator_available_bytes == int(81920 * 1024**2 * 0.85)
    assert cold.usable_device_bytes == cold.allocator_available_bytes

    # A device already occupied by someone else cannot host the pool at all.
    crowded = _discover(free_mib=20000, post_initialization=False)
    assert not crowded.available
    assert "exceeds current free memory" in crowded.unavailable_reason


def test_warm_pool_low_external_free_is_not_itself_a_rejection() -> None:
    """Regression: the self-defeating pre-initialization comparison must not recur.

    After an 85% pool exists on an 80 GiB-class device, ``nvidia-smi`` reports
    only the sliver beside the reservation. Comparing the pool target against
    that sliver double counts this process's own reservation and previously
    rejected every fraction-preallocation run.
    """

    statistics = {
        "bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
        "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 67 * GIB,
    }
    warm = _discover(free_mib=11000, post_initialization=True, statistics=statistics)
    assert warm.available, warm.unavailable_reason
    assert warm.post_initialization is True
    # Usable bytes come from the owned pool, not from the external remainder,
    # and the two are never intersected or summed.
    assert warm.usable_device_bytes == 67 * GIB
    assert warm.usable_device_bytes != warm.free_bytes
    assert warm.usable_device_bytes > warm.free_bytes
    assert warm.current_process_owned_bytes == 68 * GIB
    assert "not counted twice" in warm.usable_bytes_definition

    # The identical external evidence still fails closed before initialization.
    cold = _discover(free_mib=11000, post_initialization=False)
    assert not cold.available


def test_post_initialization_capacity_requires_allocator_statistics() -> None:
    """Missing or incoherent allocator statistics fail closed, never default."""

    for statistics in (None, {}, {"bytes_limit": 4 * GIB}, {"bytes_limit": 1, "bytes_in_use": 2}):
        observation = _discover(
            free_mib=11000, post_initialization=True, statistics=statistics
        )
        assert not observation.available
        assert "allocator statistics are unavailable" in observation.unavailable_reason


def test_unpopulated_largest_free_block_is_classified_not_believed() -> None:
    """A zero largest free block beside free internal bytes is impossible.

    The CUDA backend never populates this field, so it reports exactly that
    impossible pair. Treat it as unreported and record the classification; a
    genuine zero on a completely full pool stays authoritative.
    """

    unsupported = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
         "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 0},
        policy="fraction",
    )
    assert unsupported.available
    assert unsupported.largest_free_block_bytes is None
    assert unsupported.largest_free_block_status == "unsupported_by_backend"
    assert unsupported.internal_free_bytes == 67 * GIB

    full = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * GIB, "bytes_in_use": 68 * GIB,
         "peak_bytes_in_use": 68 * GIB, "largest_free_block_bytes": 0},
        policy="fraction",
    )
    assert full.available and full.largest_free_block_bytes == 0
    assert full.largest_free_block_status == "reported"

    incoherent = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
         "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 90 * GIB},
        policy="fraction",
    )
    assert not incoherent.available
    assert "exceeds internal free bytes" in incoherent.unavailable_reason


def test_post_initialization_admission_charges_the_right_requirement(monkeypatch) -> None:
    """Resident arguments are excluded once, and only once."""

    monkeypatch.setattr(resources, "validate_compiled_executable_bundle", lambda bundle: {
        "argument_bytes": 20 * GIB, "output_bytes": 10 * GIB,
        "temporary_bytes": 5 * GIB, "alias_bytes": 2 * GIB,
    })
    external = _discover(
        free_mib=11000, post_initialization=True,
        statistics={"bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
                    "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 0},
    )
    pool = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
         "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 0},
        policy="fraction",
    )

    resident = resources.post_initialization_capacity_preflight(
        feasibility={"safety_margin_fraction": 0.25}, bundle=object(),
        external_capacity=external, allocator_capacity=pool, arguments_resident=True,
    )
    assert resident["phase"] == "post-initialization"
    assert resident["arguments_resident"] is True
    assert resident["already_resident_argument_bytes"] == 20 * GIB
    assert resident["charged_argument_bytes"] == 0
    assert resident["incremental_executable_requirement_bytes"] == 13 * GIB
    assert resident["required_incremental_bytes_with_margin"] == int(13 * GIB * 1.25)
    # No largest-free-block measure exists, so the pool's internal free bytes
    # govern and the decision records that explicitly.
    assert resident["admitted_usable_bytes"] == 67 * GIB
    assert resident["largest_free_block_status"] == "unsupported_by_backend"
    assert "no largest-free-block measure" in resident["admission_derivation"]

    # Before device inputs exist the argument bytes must still be charged.
    not_resident = resources.post_initialization_capacity_preflight(
        feasibility={"safety_margin_fraction": 0.25}, bundle=object(),
        external_capacity=external, allocator_capacity=pool, arguments_resident=False,
    )
    assert not_resident["charged_argument_bytes"] == 20 * GIB
    assert not_resident["incremental_executable_requirement_bytes"] == 33 * GIB
    assert not_resident["already_resident_argument_bytes"] == 0


def test_post_initialization_admission_rejects_insufficient_pool(monkeypatch) -> None:
    """Insufficient internal free space and a real small block both reject."""

    monkeypatch.setattr(resources, "validate_compiled_executable_bundle", lambda bundle: {
        "argument_bytes": 20 * GIB, "output_bytes": 10 * GIB,
        "temporary_bytes": 5 * GIB, "alias_bytes": 2 * GIB,
    })
    external = _discover(
        free_mib=11000, post_initialization=True,
        statistics={"bytes_limit": 68 * GIB, "bytes_in_use": 60 * GIB,
                    "peak_bytes_in_use": 60 * GIB, "largest_free_block_bytes": 8 * GIB},
    )
    starved = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * GIB, "bytes_in_use": 60 * GIB,
         "peak_bytes_in_use": 60 * GIB, "largest_free_block_bytes": 8 * GIB},
        policy="fraction",
    )
    with pytest.raises(ValueError, match="insufficient"):
        resources.post_initialization_capacity_preflight(
            feasibility={"safety_margin_fraction": 0.0}, bundle=object(),
            external_capacity=external, allocator_capacity=starved,
        )

    # Plenty of internal free space, but a genuinely reported small block.
    fragmented = resources.allocator_capacity_observation(
        {"bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
         "peak_bytes_in_use": 40 * GIB, "largest_free_block_bytes": 4 * GIB},
        policy="fraction",
    )
    assert fragmented.largest_free_block_status == "reported"
    with pytest.raises(ValueError, match="insufficient"):
        resources.post_initialization_capacity_preflight(
            feasibility={"safety_margin_fraction": 0.0}, bundle=object(),
            external_capacity=external, allocator_capacity=fragmented,
        )


def test_external_and_internal_free_bytes_are_never_summed() -> None:
    """Requirement: the two pools must not be added or intersected."""

    statistics = {
        "bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
        "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 67 * GIB,
    }
    warm = _discover(free_mib=11000, post_initialization=True, statistics=statistics)
    external_free = warm.free_bytes
    internal_free = 67 * GIB
    assert warm.usable_device_bytes == internal_free
    assert warm.usable_device_bytes != external_free + internal_free
    assert warm.usable_device_bytes != min(external_free, internal_free)


def test_post_initialization_evidence_from_another_device_fails_closed() -> None:
    """Device, policy and freshness mismatches remain non-overridable."""

    statistics = {
        "bytes_limit": 68 * GIB, "bytes_in_use": 1 * GIB,
        "peak_bytes_in_use": 1 * GIB, "largest_free_block_bytes": 67 * GIB,
    }
    other = discover_nvidia_device_capacity(
        _execution_identity(),
        command_runner=lambda *a, **k: SimpleNamespace(
            stdout="0, GPU-00000000-0000-0000-0000-0000000000ff, 0000:09:00.0, H100, "
                   "81920, 11000, 70920\n"
        ),
        cuda_visible_ordinal_mapper=None,
        cuda_visible_devices=None, cuda_device_order=None,
        preallocate_setting="true", memory_fraction_setting="0.85",
        post_initialization=True, allocator_statistics=statistics,
    )
    assert not other.available
    assert "uniquely match" in other.unavailable_reason

    with pytest.raises(ValueError):
        resources.allocator_capacity_observation(statistics, policy="not-a-policy")


@pytest.mark.skipif(
    os.environ.get("CHU_PAIR_GPU_CAPACITY_CHECK") != "1",
    reason="opt-in real-GPU capacity check; set CHU_PAIR_GPU_CAPACITY_CHECK=1",
)
def test_real_gpu_warm_pool_admits_after_initialization() -> None:
    """Opt-in: warm a real fraction pool and exercise the real admission path.

    Never runs in ordinary CPU validation. The allocator policy must be applied
    before JAX is imported, so this runs in a fresh subprocess. It verifies on
    real hardware that a warm 85% pool no longer rejects itself and that the
    pool's own statistics, not the external remainder, govern admission.
    """

    probe = (
        "import json, os\n"
        "from chu_pair.gpu_pilot.allocator import apply_allocator_policy\n"
        "apply_allocator_policy('fraction', memory_fraction=0.85)\n"
        "import jax, jax.numpy as jnp\n"
        "if jax.default_backend() != 'gpu':\n"
        "    print(json.dumps({'skipped': 'no gpu backend'})); raise SystemExit(0)\n"
        "from chu_pair.gpu_pilot import runtime as gpu_runtime\n"
        "from chu_pair.pair_density import discover_nvidia_device_capacity\n"
        "from chu_pair.gpu_pilot.cuda_identity import CudaDriverIdentityProvider\n"
        "d = jax.devices()[0]\n"
        "identity = {'backend':'gpu','platform':'gpu','visible_device_index':0,\n"
        "  'visible_device_count':len(jax.devices()),'id':str(d.id),\n"
        "  'device_kind':d.device_kind,'process_index':d.process_index,\n"
        "  'local_hardware_id':getattr(d,'local_hardware_id',0)}\n"
        "jnp.zeros((1024,), jnp.float32).block_until_ready()\n"
        "stats = gpu_runtime._device_memory_statistics()\n"
        "warm = discover_nvidia_device_capacity(identity,\n"
        "  cuda_visible_ordinal_mapper=CudaDriverIdentityProvider.from_system().map_visible_ordinal,\n"
        "  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "  cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'),\n"
        "  preallocate_setting=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),\n"
        "  memory_fraction_setting=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),\n"
        "  post_initialization=True, allocator_statistics=stats)\n"
        "print(json.dumps({'available': warm.available,\n"
        "  'reason': warm.unavailable_reason, 'post': warm.post_initialization,\n"
        "  'usable': warm.usable_device_bytes, 'external_free': warm.free_bytes,\n"
        "  'pool': warm.current_process_owned_bytes,\n"
        "  'bytes_limit': None if stats is None else stats.get('bytes_limit')}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if "skipped" in result:
        pytest.skip(result["skipped"])
    assert result["available"] is True, result["reason"]
    assert result["post"] is True
    # The owned pool dwarfs the external remainder on a preallocated device;
    # the old comparison would have rejected exactly this state.
    assert result["usable"] > result["external_free"]
    assert result["pool"] == result["bytes_limit"]

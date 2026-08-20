"""Bounded CUDA Driver API mapping from visible ordinals to stable identities."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import re
import uuid
from typing import Callable, Protocol


MAX_CUDA_DEVICES = 16
MAX_VISIBLE_TEXT_CHARS = 512
_FIXED_LIBCUDA_PATHS = (
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/local/cuda/compat/libcuda.so.1",
)
_UUID_PATTERN = re.compile(r"^(?:GPU|MIG)-[0-9A-Fa-f-]{8,80}$")


class CudaIdentityError(RuntimeError):
    pass


class _Driver(Protocol):
    def initialize(self) -> None: ...
    def device_count(self) -> int: ...
    def stable_identity(self, ordinal: int) -> dict[str, str]: ...


class _CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class _CtypesCudaDriver:
    def __init__(
        self,
        *,
        loader: Callable[[str], object] = ctypes.CDLL,
        library_paths: tuple[str, ...] = _FIXED_LIBCUDA_PATHS,
    ) -> None:
        if not library_paths or len(library_paths) > len(_FIXED_LIBCUDA_PATHS):
            raise CudaIdentityError("libcuda candidate list is empty or unbounded")
        if any(path not in _FIXED_LIBCUDA_PATHS for path in library_paths):
            raise CudaIdentityError("only the fixed absolute libcuda paths are permitted")
        errors = []
        self._library = None
        for path in library_paths:
            try:
                self._library = loader(path)
                self.library_path = path
                break
            except OSError as error:
                errors.append(f"{path}: {type(error).__name__}")
        if self._library is None:
            raise CudaIdentityError(
                "CUDA driver library was unavailable at every fixed path: "
                + ", ".join(errors)
            )
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        library = self._library
        for name in ("cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetPCIBusId"):
            if not hasattr(library, name):
                raise CudaIdentityError(f"CUDA driver symbol {name} is unavailable")
        uuid_function = getattr(library, "cuDeviceGetUuid_v2", None)
        if uuid_function is None:
            uuid_function = getattr(library, "cuDeviceGetUuid", None)
        if uuid_function is None:
            raise CudaIdentityError("CUDA driver UUID symbol is unavailable")
        self._uuid_function = uuid_function
        library.cuInit.argtypes = [ctypes.c_uint]
        library.cuInit.restype = ctypes.c_int
        library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.cuDeviceGetCount.restype = ctypes.c_int
        library.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        library.cuDeviceGet.restype = ctypes.c_int
        library.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        library.cuDeviceGetPCIBusId.restype = ctypes.c_int
        uuid_function.argtypes = [ctypes.POINTER(_CUuuid), ctypes.c_int]
        uuid_function.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if int(code) != 0:
            raise CudaIdentityError(f"{operation} failed with CUDA status {int(code)}")

    def initialize(self) -> None:
        self._check(self._library.cuInit(0), "cuInit")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self._library.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        return int(count.value)

    def stable_identity(self, ordinal: int) -> dict[str, str]:
        device = ctypes.c_int()
        self._check(self._library.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
        raw_uuid = _CUuuid()
        self._check(self._uuid_function(ctypes.byref(raw_uuid), device), "cuDeviceGetUuid")
        pci = ctypes.create_string_buffer(32)
        self._check(
            self._library.cuDeviceGetPCIBusId(pci, len(pci), device),
            "cuDeviceGetPCIBusId",
        )
        uuid_text = f"GPU-{uuid.UUID(bytes=bytes(raw_uuid.bytes))}"
        pci_text = pci.value.decode("ascii", errors="strict").lower()
        return {"uuid": uuid_text, "pci_bus_id": pci_text}


@dataclass(frozen=True, slots=True)
class CudaDriverIdentityProvider:
    """Actual mapper injected into the existing NVIDIA capacity discovery path."""

    driver: _Driver

    @classmethod
    def from_system(
        cls,
        *,
        loader: Callable[[str], object] = ctypes.CDLL,
        library_paths: tuple[str, ...] = _FIXED_LIBCUDA_PATHS,
    ) -> "CudaDriverIdentityProvider":
        return cls(_CtypesCudaDriver(loader=loader, library_paths=library_paths))

    def map_visible_ordinal(
        self,
        *,
        visible_index: int,
        cuda_visible_devices: str | None,
        cuda_device_order: str | None,
    ) -> dict[str, str]:
        del cuda_device_order  # The initialized driver already exposes CUDA's visible order.
        if isinstance(visible_index, bool) or not isinstance(visible_index, int):
            raise CudaIdentityError("visible CUDA ordinal must be an integer")
        if cuda_visible_devices is not None and len(cuda_visible_devices) > MAX_VISIBLE_TEXT_CHARS:
            raise CudaIdentityError("CUDA_VISIBLE_DEVICES exceeds its fixed text bound")
        tokens = (
            []
            if cuda_visible_devices is None
            else [token.strip() for token in cuda_visible_devices.split(",")]
        )
        if any(not token for token in tokens) or len(tokens) > MAX_CUDA_DEVICES:
            raise CudaIdentityError("CUDA_VISIBLE_DEVICES is empty or exceeds the device bound")
        if tokens and not 0 <= visible_index < len(tokens):
            raise CudaIdentityError("visible CUDA ordinal is outside CUDA_VISIBLE_DEVICES")
        token = tokens[visible_index] if tokens else None
        if token and _UUID_PATTERN.fullmatch(token):
            # MIG tokens are retained explicitly. The current nvidia-smi whole-GPU
            # capacity query will fail closed rather than claim parent capacity.
            key = "mig_uuid" if token.startswith("MIG-") else "uuid"
            if key == "mig_uuid":
                return {key: token}
        elif token is not None:
            try:
                numeric = int(token, 10)
            except ValueError as error:
                raise CudaIdentityError("unsupported CUDA_VISIBLE_DEVICES token") from error
            if numeric < 0:
                raise CudaIdentityError("CUDA_VISIBLE_DEVICES ordinal must be non-negative")
        self.driver.initialize()
        count = self.driver.device_count()
        if not 1 <= count <= MAX_CUDA_DEVICES:
            raise CudaIdentityError("CUDA driver visible-device count is outside the fixed bound")
        if not 0 <= visible_index < count:
            raise CudaIdentityError("visible CUDA ordinal is outside the driver device count")
        identity = self.driver.stable_identity(visible_index)
        if not isinstance(identity, dict) or not identity.get("uuid") or not identity.get("pci_bus_id"):
            raise CudaIdentityError("CUDA driver did not return UUID and PCI identity")
        if not _UUID_PATTERN.fullmatch(identity["uuid"]):
            raise CudaIdentityError("CUDA driver returned a malformed device UUID")
        if len(identity["pci_bus_id"]) > 32:
            raise CudaIdentityError("CUDA driver returned an overlong PCI identity")
        return {"uuid": identity["uuid"], "pci_bus_id": identity["pci_bus_id"]}

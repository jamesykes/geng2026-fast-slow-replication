"""Set one explicit JAX allocator policy before importing JAX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import sys


@dataclass(frozen=True, slots=True)
class AllocatorPolicy:
    name: str
    memory_fraction: float | None
    environment: dict[str, str]


_POLICIES = frozenset({"default", "fraction", "no-preallocation"})


def apply_allocator_policy(
    name: str,
    *,
    memory_fraction: float | None = None,
    environment: dict[str, str] | None = None,
    loaded_modules: object | None = None,
) -> dict[str, object]:
    """Apply and describe a policy; fail if JAX has already been imported."""

    modules = sys.modules if loaded_modules is None else loaded_modules
    if "jax" in modules or "jaxlib" in modules:
        raise RuntimeError("allocator policy must be applied before importing JAX")
    if name not in _POLICIES:
        raise ValueError(f"allocator policy must be one of {sorted(_POLICIES)}")
    target = os.environ if environment is None else environment
    values: dict[str, str]
    fraction: float | None = None
    if name == "default":
        if memory_fraction is not None:
            raise ValueError("default allocator policy does not accept a memory fraction")
        values = {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        }
        target.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        target.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    elif name == "fraction":
        if isinstance(memory_fraction, bool) or not isinstance(memory_fraction, (int, float)):
            raise ValueError("fraction allocator policy requires a finite fraction")
        fraction = float(memory_fraction)
        if not math.isfinite(fraction) or not 0.05 <= fraction <= 0.95:
            raise ValueError("allocator memory fraction must lie in [0.05, 0.95]")
        values = {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": format(fraction, ".17g"),
        }
        target.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    else:
        if memory_fraction is not None:
            raise ValueError("no-preallocation policy does not accept a memory fraction")
        values = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        target.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        target.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    target.update(values)
    return asdict(AllocatorPolicy(name=name, memory_fraction=fraction, environment=values))

#!/usr/bin/env python3
"""Inspect Lambda Stack, then create an isolated exact JAX GPU environment."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys



PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-family", choices=("cuda12", "cuda13"), required=True)
    parser.add_argument("--venv", type=Path, default=PROJECT_ROOT / ".venv-gpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allocator-policy", choices=("default", "fraction", "no-preallocation"), default="fraction")
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    return parser.parse_args()


def _inspect_existing_python() -> dict[str, object]:
    """Inspect installed distributions without importing backend-initializing modules."""

    result: dict[str, object] = {
        "python": platform.python_version(), "architecture": platform.machine(),
    }
    for distribution in ("jax", "jaxlib", "numpy", "jax-cuda12-plugin", "jax-cuda13-plugin"):
        try:
            result[distribution.replace("-", "_")] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution.replace("-", "_")] = None
    return result


def _validation_command(policy: str, fraction: float | None) -> list[str]:
    script = (
        "import json,os; "
        "p=os.environ['PILOT_ALLOCATOR_POLICY']; a=os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']; "
        "assert (p=='fraction' and a=='true' and 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ) or (p=='default' and a=='true' and 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ) or (p=='no-preallocation' and a=='false' and 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ); "
        "import jax,jaxlib,numpy; "
        "print(json.dumps({'jax':jax.__version__,'jaxlib':jaxlib.__version__,"
        "'numpy':numpy.__version__,'allocator_policy':os.environ['PILOT_ALLOCATOR_POLICY'],"
        "'memory_fraction':os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}))"
    )
    return ["-c", script]


def _allocator_environment(policy: str, fraction: float | None) -> dict[str, str]:
    """Build the same bounded policy without importing project/JAX modules."""

    target = dict(os.environ)
    target.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    target.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    if policy == "default":
        if fraction is not None:
            raise ValueError("default policy does not accept a memory fraction")
        target["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    elif policy == "fraction":
        if fraction is None or not 0.05 <= float(fraction) <= 0.95:
            raise ValueError("fraction policy requires a finite fraction in [0.05, 0.95]")
        target["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        target["XLA_PYTHON_CLIENT_MEM_FRACTION"] = format(float(fraction), ".17g")
    else:
        if fraction is not None:
            raise ValueError("no-preallocation policy does not accept a memory fraction")
        target["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    target["PILOT_ALLOCATOR_POLICY"] = policy
    return target


def main() -> int:
    args = parse_args()
    target = args.venv.resolve()
    if PROJECT_ROOT not in target.parents or target == PROJECT_ROOT:
        raise ValueError("GPU virtualenv must be a dedicated directory under this subproject")
    requirement = PROJECT_ROOT / "requirements" / f"gpu-{args.cuda_family}.txt"
    existing = _inspect_existing_python()
    fraction = args.memory_fraction if args.allocator_policy == "fraction" else None
    allocator_environment = _allocator_environment(args.allocator_policy, fraction)
    commands = [
        [sys.executable, "-m", "venv", str(target)],
        [str(target / "bin/python"), "-m", "pip", "install", "--requirement", str(requirement)],
        [str(target / "bin/python"), "-m", "pip", "install", "--no-deps", "--editable", str(PROJECT_ROOT)],
        [str(target / "bin/python"), *_validation_command(args.allocator_policy, fraction)],
    ]
    report = {
        "existing_stack": existing,
        "python_architecture": platform.machine(),
        "cuda_family": args.cuda_family,
        "requirement_file": str(requirement.relative_to(PROJECT_ROOT)),
        "target_virtualenv": str(target),
        "commands": commands,
        "dry_run": args.dry_run,
        "allocator_policy": args.allocator_policy,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("GPU environment creation is supported only on Linux x86_64/aarch64")
    if target.exists():
        raise ValueError("refusing to replace an existing GPU virtualenv")
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        raise RuntimeError("GPU setup parent must not import JAX before controlled validation")
    for index, command in enumerate(commands):
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, check=True,
            capture_output=index == len(commands) - 1, text=True,
            timeout=600, env=allocator_environment,
        )
        if index == len(commands) - 1 and len(completed.stdout.encode("utf-8")) > 4096:
            raise RuntimeError("controlled JAX validation output exceeded its bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

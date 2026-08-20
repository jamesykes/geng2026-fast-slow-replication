#!/usr/bin/env python3
"""Report bounded JAX/CUDA identity and capacity evidence for the GPU pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from chu_pair.gpu_pilot.allocator import apply_allocator_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def doctor_exit_code(*, gpu_ready: bool, expect_gpu: bool) -> int:
    return 0 if (gpu_ready or not expect_gpu) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-family", choices=("cuda12", "cuda13"), required=True)
    parser.add_argument(
        "--allocator-policy", choices=("default", "fraction", "no-preallocation"),
        default="fraction",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--expect-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fraction = args.memory_fraction if args.allocator_policy == "fraction" else None
    apply_allocator_policy(args.allocator_policy, memory_fraction=fraction)
    import jax
    import jaxlib
    import numpy as np

    from chu_pair.gpu_pilot.doctor import collect_gpu_doctor_report

    report = collect_gpu_doctor_report(
        project_root=PROJECT_ROOT, jax_module=jax, numpy_module=np,
        jaxlib_module=jaxlib, cuda_family=args.cuda_family,
    )
    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        allowed = (PROJECT_ROOT / "outputs" / "gpu_pilot").resolve()
        if allowed != output.parent and allowed not in output.parents:
            raise ValueError("doctor output must remain under outputs/gpu_pilot")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".doctor-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    print(text, end="")
    return doctor_exit_code(gpu_ready=bool(report["gpu_ready"]), expect_gpu=args.expect_gpu)


if __name__ == "__main__":
    raise SystemExit(main())

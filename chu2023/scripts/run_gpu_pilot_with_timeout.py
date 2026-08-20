#!/usr/bin/env python3
"""Run one pilot command with a process-level timeout and bounded termination."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile


MAX_ARGUMENTS = 64
MAX_ARGUMENT_CHARS = 4096
GRACE_SECONDS = 10
MAX_CAPTURE_BYTES = 64 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--stage", default="unknown")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _write_timeout_artifact(directory: Path | None, *, stage: str, timeout: int, pid: int, killed: bool) -> None:
    if directory is None:
        return
    root = Path.cwd().resolve() / "outputs" / "gpu_pilot"
    target = directory.resolve()
    if root not in target.parents and target != root:
        raise ValueError("timeout artifact directory must remain under outputs/gpu_pilot")
    payload = {
        "schema_version": 1, "stage": stage[:64], "status": "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": timeout, "child_pid": pid if pid > 0 else None,
        "process_group_terminated": killed, "capacity_evidence_invalidated": True,
    }
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")
    if len(encoded) > 4096:
        raise RuntimeError("timeout artifact exceeded its fixed bound")
    target.mkdir(parents=True, exist_ok=True)
    success = target / "stage.json"
    if success.exists() and success.is_file() and success.stat().st_size <= 256 * 1024:
        try:
            if json.loads(success.read_text(encoding="ascii")).get("status") == "success":
                success.unlink()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    with tempfile.NamedTemporaryFile(dir=target, prefix=".timeout-", delete=False) as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target / "timeout-failure.json")


def main() -> int:
    args = parse_args()
    if not 1 <= args.timeout_seconds <= 6 * 60 * 60:
        raise ValueError("timeout must lie in [1, 21600] seconds")
    if not args.command or args.command[0] == "--":
        command = args.command[1:]
    else:
        command = args.command
    if not command:
        raise ValueError("a command is required")
    if len(command) > MAX_ARGUMENTS or any(
        not isinstance(value, str) or not value or len(value) > MAX_ARGUMENT_CHARS
        for value in command
    ):
        raise ValueError("command argv is outside its fixed safe bounds")
    if os.name != "posix":
        raise RuntimeError("pilot timeout wrapper requires POSIX process groups")
    process = subprocess.Popen(
        command, shell=False, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
            raise RuntimeError("pilot subprocess log output exceeded its fixed bound")
        return process.returncode
    except subprocess.TimeoutExpired:
        # start_new_session=True guarantees this is not our own group; require
        # the direct child to remain its leader before signalling it.
        killed = False
        try:
            is_own_group = process.pid > 0 and os.getpgid(process.pid) == process.pid
        except ProcessLookupError:
            is_own_group = False
        if is_own_group:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                killed = True
            except ProcessLookupError:
                pass
        try:
            stdout, stderr = process.communicate(timeout=GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            if process.pid > 0:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    killed = True
                except ProcessLookupError:
                    pass
            stdout, stderr = process.communicate()
        if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
            print("pilot timed out; child logs exceeded the retained bound")
        _write_timeout_artifact(
            getattr(args, "artifact_dir", None), stage=getattr(args, "stage", "unknown"), timeout=args.timeout_seconds,
            pid=process.pid, killed=killed,
        )
        print(
            "pilot process group timed out and was terminated; capacity evidence is now stale, "
            "so rerun the GPU doctor before any retry"
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())

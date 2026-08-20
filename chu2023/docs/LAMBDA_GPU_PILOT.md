# Lambda Cloud GPU pilot deployment guide

This is a controlled operator runbook for the exact separable JAX pair solver on a Lambda Cloud NVIDIA instance. It is not evidence that this repository has used Lambda Cloud or validated a GPU. The workflow has only been exercised locally in CPU-unavailable and mocked-device modes.

## Provision and secure the instance

1. In the Lambda Cloud console, choose a single-GPU instance with adequate host RAM and local disk and the current Lambda Stack image. The initial target is one H100 PCIe 80 GB; the identity/capacity gates remain applicable to A6000 48 GB, A100 40/80 GB, H100 SXM 80 GB, GH200 96 GB, and the future Warwick cluster. Multi-GPU instances provide no benefit until the solver is explicitly sharded. Record the displayed instance type, GPU model, and hourly price; pricing is user-supplied provenance, not inferred by the code.
2. Add a short-lived SSH public key. Prefer an SSH agent or dedicated key with restrictive permissions. Never paste private keys, API tokens, Git credentials, or cloud secrets into shell history, TOML, metadata, issue text, or logs.
3. Connect with host-key checking enabled and verify the fingerprint through the console.
4. Obtain the repository without credentials in a command-line URL. A repository-limited deploy key is preferable to broad credentials or agent forwarding.
5. Check out the exact reviewed commit. Numerical stages require a clean `chu2023` subproject and record commit and source hashes.
6. Copy only required bounded artifacts, then terminate—not merely stop—the instance. Ending Python or closing SSH does not terminate a billable instance. Confirm termination and billing state in the console. Revoke temporary keys and rotate anything exposed.

The scripts never call the Lambda API, provision, SSH, or terminate an instance. Those remain explicit operator actions.

Do not run `sudo apt full-upgrade` or indiscriminately modify the Lambda Stack image. Instance-local storage is disposable. Preserve artifacts elsewhere before termination. Pilot timings and diagnostics are engineering evidence only; draw no scientific production conclusions from them.

### Paid-session checklist

Before start: confirm the reviewed commit, key scope, instance/GPU, console price, session budget, timeout, artifact destination, and termination responsibility. At session start: note UTC time, run the stack inspection and doctor, and stop immediately on mismatch. Before ending: stop new stages, copy/push artifacts without credentials in commands, verify checksums, terminate the instance in the console, confirm billing has stopped, and revoke temporary access.

## Inspect and isolate the software stack

Before changing the image, record:

```bash
python --version
uname -m
nvidia-smi
python -c 'import jax, jaxlib, numpy; print(jax.__version__, jaxlib.__version__, numpy.__version__); print(jax.default_backend()); print(jax.devices())'
```

Do not assume Lambda Stack JAX matches this project. The pilot pins JAX/JAXLIB through `jax[cuda12]==0.7.2` or `jax[cuda13]==0.7.2`, NumPy 2.2.4, and tests in a project-local virtual environment. CUDA 12 requires Linux driver major version 525 or newer; CUDA 13 requires 580 or newer. The doctor enforces these non-overridable minimums. Do not replace the host driver for this pilot.

Preview and then create the isolated environment:

```bash
python scripts/prepare_gpu_environment.py --cuda-family cuda12 --dry-run
python scripts/prepare_gpu_environment.py --cuda-family cuda12
source .venv-gpu/bin/activate
```

The setup refuses to overwrite an existing environment or target the project root. Separate CUDA 12/13 requirement files prevent installing incompatible plugin families together.

## Set the allocator before JAX import

Every doctor/pilot process applies one explicit policy before importing JAX:

- `default`: explicit preallocation without a fraction override;
- `fraction`: preallocation limited by `XLA_PYTHON_CLIENT_MEM_FRACTION` (checked-in configs use 0.85);
- `no-preallocation`: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, useful diagnostically but potentially slower or fragmented.

The process fails if JAX was imported first. Allocator settings are recorded. Changing them invalidates prerequisite environment provenance and requires a fresh process, doctor, and stage chain.

## Run the GPU doctor

```bash
python experiments/run_gpu_doctor.py \
  --cuda-family cuda12 \
  --allocator-policy fraction \
  --memory-fraction 0.85 \
  --expect-gpu \
  --output outputs/gpu_pilot/doctor.json
```

The bounded report records Python architecture, exact JAX/JAXLIB/NumPy/CUDA-plugin versions, backend/device identities, whitelisted CUDA/JAX allocator variables, Linux `MemTotal`/`MemAvailable`, Git commit/cleanliness, source hashes, and `nvidia-smi` capacity evidence. It never serializes the general environment or command line. Host planning memory plus the configured margin must fit current host availability before compilation; compiler RSS/code cache remain explicitly unmodeled, so the operator must still monitor the instance.

For numeric `CUDA_VISIBLE_DEVICES`, a ctypes CUDA Driver API provider maps the JAX-visible ordinal to driver-visible UUID and PCI identity. It uses only fixed absolute `libcuda.so.1` paths and bounded devices/text; it never equates numeric visibility with an `nvidia-smi` index. UUID-reordered and multi-GPU visibility are supported. MIG tokens are identified, but whole-GPU capacity deliberately fails closed rather than being claimed for a slice.

Capacity is a point-in-time estimate, not a guarantee. Executable admission accepts evidence for at most 60 seconds and recollects it immediately before every invocation.

## Run the staged pilot

Real numerical stages require `--execute`, a matching clean doctor, a successful atomic prerequisite, and `--hourly-price-usd` exactly equal to the reviewed config. Because the checked-in templates deliberately contain `0.0` rather than a possibly stale cloud price, copy each chosen template beneath ignored `outputs/gpu_pilot/`, give the whole stage chain the same console price and session budget, review each copy, and pass the same price on the command line. A changed price/budget requires a new chain. This preserves a clean source tree while normalized config/digest enter the artifact. Cost is only `elapsed_seconds / 3600 * hourly_price_usd`, not billing data.

Create the stage-chain root:

```bash
python experiments/run_gpu_pilot.py \
  --stage doctor --doctor-report outputs/gpu_pilot/doctor.json --execute
```

Use the emitted `stage.json` as the small-stage prerequisite.

### Small: `G=3,5,9`

```bash
python experiments/run_gpu_pilot.py \
  --stage small --config outputs/gpu_pilot/reviewed-small.toml \
  --doctor-report outputs/gpu_pilot/doctor.json \
  --prerequisite outputs/gpu_pilot/gpu-doctor-TIMESTAMP/stage.json \
  --hourly-price-usd PRICE_FROM_CONSOLE --execute
```

Both flat validation and separable bounded executables are compiled and fully analyzed before either runs. Their bounded source summaries/diagnostics must agree; timing order alternates.

### Medium: `G=17,33`

Use `configs/gpu_pilot_medium.toml` and the successful small artifact. Only the separable executable runs.

Float64 configurations additionally require `JAX_ENABLE_X64=1` in the fresh process before the doctor and every stage. The x64 setting is part of environment and executable provenance; a float64 request cannot silently truncate.

For a bounded block-size sweep, create separate ignored config copies and change only `row_block_size`/`column_block_size` within the fixed 4096 limit, running one reviewed case at a time. There is no automatic or unbounded autotuner.

### Large pilot: `G=65`, optional `G=97`

Use `configs/gpu_pilot_large.toml` and the successful medium artifact. The `G=97` variant is disabled by filename and requires both `include_g97=true` and `--enable-g97`; review prior telemetry and capacity before enabling it.

### Full-grid analysis only: `G=131`

The disabled template lowers and compiles the exact one-step bounded separable object but never invokes it. Its executable-configuration digest must match the subsequent one-step stage. It requires a successful large artifact and exact phrase:

```text
ANALYZE EXACT G131 SEPARABLE
```

### One full-grid step

The separate disabled template is hard-limited to `steps=1`, requires a successful full analysis no more than ten minutes old, and exact phrase:

```text
EXECUTE ONE EXACT G131 STEP
```

There is no multi-step full-grid configuration. Do not edit the one-step template into one.

## Admission order and artifacts

Every numerical executable follows this order:

1. strict bounded TOML normalization and allocation-free Python-integer estimates;
2. doctor/prerequisite/commit/clean checks;
3. host-only grid description and abstract JAX shapes;
4. exact lowering and compilation;
5. complete live `memory_analysis()` bound to the retained callable and invocation signature;
6. fresh stable-device-matched capacity plus safety margin;
7. only then device grid/histogram/state inputs;
8. another fresh capacity check adjacent to invocation;
9. execution and mass, finiteness, nonnegativity, symmetry, conditional-moment, and destination-validity checks.

No host pair density is created. After admission, the runner reproduces the configured local-RNG scaled-Beta draw order into one seeded legacy one-agent histogram; sample count and `int64` count storage are preflighted. The analyzed combined executable constructs the independent ordered pair mass on device and returns bounded summaries/diagnostics, not final density. The full `G=131` stage uses the original `[-0.1,1.2]`, `0.01` grid. Reduced performance grids use reviewed decimal-aligned bounds that contain the full payoff range; they are parity/scaling pilots, not a grid-convergence study. Small parity is the only stage also executing the flat oracle.

Each attempted numerical stage atomically replaces one bounded `outputs/gpu_pilot/.../stage.json`, including catchable failures. It records normalized config/digest, predecessor digest, commit/environment digest, package/driver/device provenance, compiled signature/report, capacity, individual timings/dispersion, telemetry, parity and scientific diagnostics, a bounded event log, user price/cumulative cost estimate, status, and error. An OS-killed process cannot promise a final artifact.

Prerequisites are successful, digest-protected, commit/environment/scientific-contract matched, and at most six hours old; full analysis is at most ten minutes old. Never hand-edit an artifact. SHA-256 here provides local integrity/provenance against accidental changes; it is not a signature and cannot defend against an attacker able to rewrite both an artifact and its digest. The workflow assumes a trusted operator, repository, and instance.

## Timeouts, OOMs, and cost

Wrap real stages:

```bash
python scripts/run_gpu_pilot_with_timeout.py --timeout-seconds 1800 -- \
  python experiments/run_gpu_pilot.py ... --execute
```

Configs reject a worst-case stage charge before runtime import, but an in-process timer cannot interrupt a blocked compiler. The wrapper creates a new process group, sends TERM then KILL after a bounded grace period, and returns 124 on timeout. It stops only the local process group: it does not terminate the Lambda instance or stop cloud billing. After timeout, OOM, driver reset, allocator change, or unrelated GPU load, rerun the doctor; old capacity is stale.

The runner records compile time, every synchronized execution time, median/min/max/MAD, bounded process-memory telemetry when `nvidia-smi` supplies it, cumulative elapsed-time cost, remaining session budget, and whether the next stage's fixed maximum duration would exceed that budget. Missing telemetry is recorded and never replaces compiled analysis/capacity. Stop if memory approaches the margin, timings are unstable, diagnostics fail, or cost approaches budget.

Before termination, package the ignored artifacts and retrieve them through the already verified SSH host alias (or push them to an approved private artifact store without credentials in the command):

```bash
tar -C outputs -czf gpu-pilot-artifacts.tgz gpu_pilot
# Run from the trusted local workstation; HOST is an SSH config alias.
scp HOST:/path/to/chu2023/gpu-pilot-artifacts.tgz .
sha256sum gpu-pilot-artifacts.tgz
```

`--allow-expensive` overrides only reported static development-budget violations. It cannot override configuration validity, provenance, executable identity, incomplete analysis, backend/device, stale/insufficient capacity, stage order, phrases, G97 opt-in, one-step limit, or science.

## CPU-only checks and dry runs

Without NVIDIA hardware, this reports expected unavailability without a capacity query:

```bash
python experiments/run_gpu_doctor.py --cuda-family cuda12 --allocator-policy fraction --memory-fraction 0.85
```

Use `--dry-run` for each config. It normalizes and estimates, but never imports the numerical runtime, lowers, compiles, or executes pair code. Full-grid dry runs still require exact phrases. CPU tests mock CUDA identities, reordered visibility, capacity, stale artifacts, timeouts, and admission failures; they never query a real GPU or construct `G=131` arrays.

# Chu et al. pair-approximation numerics

This self-contained subproject preserves the original `case2_1.py` calculation and develops independently tested numerical components for the model in Chu et al. Phases 1 and 2 provide shared model semantics, a readable small-grid NumPy pair-mass reference, and a finite-population JAX agent-based simulation. Pair-JAX transport, covariance instrumentation, Q-bin estimators, and variance experiments remain later phases.

The scientific and numerical conventions are specified in `MODEL_SPEC.md`; staged work and validation gates are in `PLAN.md`. The original paper and `case2_1.py` are provenance artifacts and must not be edited.

## Install and test

From this directory:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The reference tests use tiny grids and must not allocate the original `(131, 131, 2, 131, 131)` pair array. To run without an editable install, pytest also adds `src/` to its import path through `pyproject.toml`.

## Current numerical conventions

- Actions are ordered `(C, D)` and states `(SH, PD)`.
- Pair arrays use axes `(q1_C, q1_D, state, q2_C, q2_D)` and contain probability masses, not pointwise densities.
- The continuous selected-coordinate Q-learning update is separate from pair-grid projection.
- The NumPy pair oracle uses the exact active nearest-grid behaviour of the original `appro()` function and rejects out-of-range destinations.
- Seeded legacy histograms use a local `random.Random` instance and reproduce the original scaled-Beta draw order without mutating global random state.
- Large reference pair allocations are rejected by default.

## JAX ABM

The complete graph is stored as packed upper-triangle endpoint arrays of length `E=n(n-1)/2`; the evolving state contains `Q` with shape `(n,2)` and one edge-state vector with shape `(E,)`. The keyed stochastic step draws one action per agent from `Q_t`, reuses it on all incident edges, calculates rewards from `S_t`, updates only the selected Q-coordinate continuously, and then transitions the same edges to `S_{t+1}`. The default scan retains agent-sized Q/action/reward/velocity histories but no edge-sized history.

Grid-matched initialisation samples from a seeded Phase 1 histogram. Continuous paper-like initialisation samples the documented scaled Beta laws. Both use independent explicit JAX keys for Q, edge states, and subsequent dynamics; histogram and ABM seeds remain separate. ABM seeds are restricted to the unaliased `PRNGKey` range `0..2**32-1`.

Only the number of scan steps is explicitly static. Population size, edge count, run count, dtype, and horizon also specialize compiled array shapes; changing ordinary Q values, endpoint values, seeds, `alpha`, or `tau` does not make them static parameters.

Check the active backend without enabling x64 globally:

```bash
python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

CUDA installation is deliberately deferred until the target HPC environment is known. A strict CPU x64 validation process can be run with:

```bash
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q
```

The Phase 2 validation on 2026-08-18 used Python 3.12.10, NumPy 2.2.4, and JAX/JAXLIB 0.7.2 on `cpu` with `CpuDevice(id=0)`; x64 was disabled by default. Both the default warnings-as-errors run and a fresh CPU+x64 warnings-as-errors run reported `81 passed`. Hand-calculated dynamics use `rtol=0` with `atol=2e-7` or `3e-7` in float32 and `1e-12` in x64. Policy parity uses `rtol=1e-6`; predeclared sampling tolerances are recorded beside each statistical test.

Run the guarded small baseline with:

```bash
python experiments/run_abm_baseline.py --config configs/abm_baseline_small.toml
```

It reports the backend/device and writes configuration, seeds, compact step records, resource estimates, hashes of the current implementation, and clearly labelled state summaries `Q_0,S_0` through final `Q_T,S_T` beneath ignored `outputs/abm/`. Thus `T=0` reports the initialized state once, while the last row for `T>0` is the state after step `T-1`. Hard Phase 2 caps cannot be weakened by the input configuration; exceeding them requires explicit `--allow-expensive`, which is recorded in metadata. The baseline does not claim paper reproduction.

Grid-matched histogram construction is separately preflighted before allocation or sampling. Normal runs allow at most 5,000,000 histogram cells, 32 MiB for the actual `int64` count array, and 2,000,000 sample pairs (two Beta variates per pair). These are conservative operational limits, not model parameters; they are not configurable and only the explicit `--allow-expensive` flag bypasses them.

ABM graph/state memory is `O(n^2)` through the `E` edge states. A single scan retains `O(Tn)` agent histories and `O(T)` summaries, not `O(TE)` edge histories. A batch of `R` runs uses `O(Rn^2)` state/working memory and `O(RTn)` retained records.

Generated experiment data belongs under `outputs/`, which is ignored.

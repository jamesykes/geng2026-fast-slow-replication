# Chu et al. pair-approximation numerics

This self-contained subproject preserves the original `case2_1.py` calculation and develops independently tested numerical components for the model in Chu et al. Phases 1 and 2 provide shared model semantics, a readable small-grid NumPy pair-mass reference, and a finite-population JAX agent-based simulation. Phase 3A adds selected-action ABM variance instrumentation and two-dimensional Q-bin moment estimators. Pair-JAX transport, counterfactual diagnostics, uncertainty intervals, and final variance comparisons remain later phases.

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

## Phase 3A variance diagnostic

The instrumented scan records source-time `Q_t`, the one selected action and Q-coordinate, realised reward and velocity, and per-agent payoff sums `S1=sum_h Y_ih` and `S2=sum_h Y_ih^2`. It uses the same actions, random-key stream, payoffs, Q update, and transitions as the Phase 2 scan. Only `(R,T,n)` agent fields are added; edge payoff histories are not retained.

The host estimator preserves the independent-run axis and aggregates by run, source time, two-dimensional source-Q bin, and realised selected action. It uses population (`ddof=0`) raw moments. Distinct-opponent products come from `S1^2-S2`; `c` is explicitly missing for `n=2`. The exact finite-bin velocity calculation includes reward variance, selected-Q variance, and reward-Q covariance rather than silently treating every Q in a bin as its centre.

Bins are `[lower,upper)` in both coordinates, with the last upper edge included. Configured edges are retained as float64 provenance, while classification uses edges converted to the observation Q dtype; conversion is revalidated and collapsed edges are rejected. Out-of-range observations raise instead of being clipped or discarded. CSV and metadata report configured and effective comparison edges separately. Empty and underpopulated strata are emitted with explicit flags, and undefined estimates are blank in CSV output.

After TOML parsing but before any NumPy edge-array copy, graph construction, initialization, simulation, aggregation, or output creation, the diagnostic obtains `Bc` and `Bd` from the raw sequence lengths and preflights the dense strata `S=R*T*Bc*Bd*2`, their estimated peak statistic memory, and the `S` dense CSV rows. This cannot guard memory already consumed by the TOML parser's Python lists. The memory audit includes the `int64` count, ten retained float64 sums, host observation and index work, five float64 product arrays, five additional float64 conversion arrays only for float32 observations (the conversions alias existing host arrays for float64), fourteen returned float64 moments, four validity masks, four retained derivation intermediates, three conservative NumPy expression work arrays, and both configured/effective NumPy bin-edge arrays. Fixed normal-run caps are 1,000,000 strata, 256 MiB estimated peak statistic memory, and 250,000 output rows. Because every stratum is emitted, the row cap is normally the first stratum-count limit; the other caps remain independent safety checks. Configuration cannot weaken these limits; only `--allow-expensive` bypasses and records violations.

The Phase 2 baseline retains its committed baseline resource accounting. The Phase 3A runner explicitly selects the instrumented accounting mode, which adds exactly the three agent-level record arrays `selected_q_t`, `payoff_sums_t`, and `payoff_square_sums_t`, plus the two live payoff accumulators.

Run the small CPU diagnostic with:

```bash
python experiments/run_abm_variance_diagnostic.py --config configs/abm_variance_diagnostic_small.toml
```

It writes `binned_moments.csv` and reproducibility/formula metadata beneath ignored `outputs/abm_variance/`. The current milestone has 28 focused Phase 3A tests and 115 tests in the complete suite. It does not implement counterfactual unselected-action diagnostics, confidence intervals, the pair-JAX solver, or the final four-way comparison.

Generated experiment data belongs under `outputs/`, which is ignored.

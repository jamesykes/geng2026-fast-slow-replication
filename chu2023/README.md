# Chu et al. pair-approximation numerics

This self-contained subproject preserves the original `case2_1.py` calculation and develops independently tested numerical components for the model in Chu et al. Phases 1 and 2 provide shared model semantics, a readable small-grid NumPy pair-mass reference, and a finite-population JAX agent-based simulation. Phase 3A adds selected-action ABM variance instrumentation and two-dimensional Q-bin moment estimators. Phase 3B adds independent-run bootstrap intervals and nested Q-bin refinement diagnostics. Phase 4 adds a guarded CPU-validated JAX implementation of the exact legacy pair-mass transport. Counterfactual diagnostics, final variance comparisons, full-grid production runs, and GPU benchmarking remain later phases.

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

It writes `binned_moments.csv` and reproducibility/formula metadata beneath ignored `outputs/abm_variance/`. Phase 3A has 28 focused tests. The Phase 3A component does not implement counterfactual unselected-action diagnostics or the final four-way comparison; the separate pair solver is described under Phase 4 below.

## Phase 3B uncertainty and refinement diagnostic

Phase 3B pools the Phase 3A per-run sufficient sums across independent runs before applying the nonlinear moment formulas. This is an observation-weighted conditional estimate, not an unweighted average of run-level variances. Uncertainty is obtained by resampling complete runs: one reproducible `(B,R)` int32 multiplicity matrix is used for every time, bin, action, estimand, and refinement scheme, without consuming a JAX simulation key. Intervals are pointwise percentile intervals using NumPy's `linear` quantile method.

An interval is emitted only when the original stratum contains observations from at least two independent runs and at least `max(2, ceil(0.8 B))` bootstrap estimates are finite. Otherwise its endpoints remain `NaN`, with valid/invalid replicate counts and an explicit validity flag. These checks prevent empty or single-run strata from receiving plausible-looking intervals; they do not establish scientific coverage for the small smoke configuration.

Configured refinement schemes must share outer bounds and each successor must strictly refine both Q coordinates. Nesting is checked in configured float64 edges and in effective observation-dtype edges. Bins retain Phase 3A `[lower,upper)` semantics with the final upper endpoint included. Parent counts reconstruct exactly from child bins. Per-field bounds first follow payoff, scatter, reward, Q, velocity, and host-product arithmetic in the original observation dtype and then apply `gamma_k=k*epsilon/(1-k*epsilon)` to parent/child reconstruction in the actual sufficient-array summation dtype. Here `epsilon=numpy.finfo(dtype).eps` is machine epsilon, used conservatively in place of the smaller round-to-nearest unit roundoff. The actual represented `S1**2-S2` expression receives an additive bound even when `n=2`. This handles cancellation without accepting material corruption; both dtypes, maximum differences, and applied reconstruction allowances are saved in metadata. Anchors use the same assignment rule and report configured/effective bounds and widths at every level. No monotonic trend across bin widths is assumed.

The finite-bin output includes both

```text
Var(v) - alpha^2 Var(reward)
```

and its exact equivalent `alpha^2[Var(selected Q)-2 Cov(reward,selected Q)]`. Thus `alpha^2 Var(reward)` is not labelled as the exact velocity variance of a finite-width bin.

Run the bounded smoke diagnostic with:

```bash
python experiments/run_abm_uncertainty_diagnostic.py --config configs/abm_uncertainty_smoke.toml
```

It runs the ABM once, aggregates schemes sequentially, processes bootstrap strata in bounded chunks, streams CSV rows, and writes pooled intervals, anchor diagnostics, metadata, and the exact bootstrap run weights beneath ignored `outputs/abm_uncertainty/`.

Before `QBinSpec`, graph construction, simulation, aggregation, bootstrap allocation, or output creation, the runner uses raw Python sequence lengths and Python-integer arithmetic to estimate every scheme. If `C_l=T*Bc_l*Bd_l*2`, it accounts for `88R*C_l` bytes of per-run sufficient statistics; configured/effective edge arrays; Phase 3A's dtype-aware observation aggregation work; an int32 `4BR` weight matrix and its generation/float64-processing lifetimes; retained `445*sum_l C_l` point/interval summaries; `260*max_l C_l` pooled-point derivation work; and `280*B*K` bootstrap work for chunk width `K`. Unweighted point pooling directly reduces the run axis in int64/float64, so it retains no `R`-length weights or conversion. Sequential reconstruction retains at most adjacent parent/child sufficient arrays and a conservative `112R*C_l` allowance for indexed values/counts, count conversions, field scales, gamma/error arrays, reduction, and comparison work. Output accounting adds up to 16 MiB for chunked weight serialization and 40 bytes per effective edge converted to JSON-ready Python storage, reuses the parser's configured-edge lists, and hashes weights through a non-copying memory view. The Phase 3B caps cover total per-run strata, pooled and anchor rows, weight bytes, chunk work, and the conservative host-statistics peak; the separate Phase 2 instrumented guard covers JAX records and simulation work. Configuration cannot weaken either set of limits; only recorded `--allow-expensive` can override violations.

Phase 3B has 38 focused tests: 37 pass with one x64-only regression skipped in the default float32 process, and all 38 pass in a CPU+x64 process. Counterfactual unselected-action statistics, focal-agent resampling, the final pair/ABM comparison, and production-scale inference remain future work.

## Phase 4 JAX pair-density solver

The JAX solver stores the canonical probability mass `(G,G,2,G,G)` internally as `(2,M,M)`, where `M=G**2` and the two endpoint axes are ordered. It computes source-time policies and the one-edge conditional arrays `w(s,b|q)`, `mu`, `m2`, and `sigma^2=m2-mu^2` by vectorized contractions. One synchronous step calculates the two action-specific velocity maps from the source mass, applies the active decimal-tick `appro()` projection independently to both endpoint types, and pushes each old-state/source cell through the four joint-action branches using the authoritative transition tensor. A bounded flat scatter is processed in deterministic source chunks; it neither interpolates nor renormalizes.

The compiled multi-step path uses `jax.lax.scan`. It returns the final density and an agent-independent trajectory of eleven floating diagnostics and three validity flags per step, never every full density. `T=0` returns the initial mass and empty diagnostic leaves. Host wrappers validate shape and element count before copying to the host, then check finiteness, nonnegativity, endpoint symmetry, projection bounds, conditional-weight normalization, and moment validity. The construction, host-conversion, and checked-run interfaces accept float64 only when JAX x64 was enabled before import, so the supported runner cannot silently truncate a nominal float64 configuration. Low-level compiled kernels take already-typed JAX arrays.

Run the focused tests and bounded smoke calculation with:

```bash
python -m pytest -q tests/test_pair_jax.py tests/test_pair_jax_runner.py
python experiments/run_pair_jax_small.py --config configs/pair_jax_small.toml
```

The smoke runner uses two resource gates. First, it derives `G`, `M=G**2`, `M**2`, and `D=2M**2` with allocation-free Python-integer arithmetic before constructing a `QGrid` or JAX array. With scalar byte width `b`, effective source chunk `K=min(chunk_size,D)`, `T` steps and selected diagnostic-row count `R_d`, its conservative static device allowance is

```text
8Db
+ [Gb + M(20b+40)]
+ 17Kb
+ 96K
+ T(11b+3).
```

The eight full-density buffers conservatively cover the input, scan carry, scatter destination, returned output and backend-created loop/scatter temporaries. The former `3Db` term was insufficient: on the validated CPU backend at `G=9`, it predicted 177,904 bytes in float32 and 346,412 bytes in float64, while the compiled executables required 280,801 and 546,485 bytes respectively. The remaining static-device terms cover grid/point/moment/destination work, deliberately unfused source/policy/branch temporaries, branch indices and lean scan diagnostics.

The separately recorded static host allowance is

```text
Db + T(11b+3) + 8M + [G(16+b) + M(8+4b)]
+ 4096 R_d + S + 1 MiB.
```

It covers the final device-to-host density validation copy and the simultaneously live complete host diagnostic trajectory, plus the float64 histogram, host grid/index/Q-point construction including gather/stack temporaries, selected Python diagnostic rows, bounded JSON/CSV work and the source-hash read buffer. Static combined storage is the sum of the device and host allowances. Here `S` is the maximum audited live serialization peak, not merely encoded payload length. If `J` is the bounded ASCII metadata length, `C=4096` is the metadata-chunk/CSV-write limit, `U=8` bytes per retained text character is the portable conservative text allowance, `F=8192` is the explicit binary file buffer and `H=65536` covers fixed serializer/object overhead, then

```text
S = max(3 U J + H,
        U J + U C + C + F + H).
```

The first branch permits bounded input-object/string storage, escaped JSON fragments and their joined result to coexist. The input metadata object is released after the checked string is produced. The second covers that retained metadata string together with one streamed metadata chunk or CSV record, its one-byte ASCII encoding and the file buffer. JSON and CSV stages are sequential, so their transient terms are alternatives rather than additive.

After TOML parsing, the runner rejects unknown tables and keys and validates every scalar and string before numerical work. The only serialized configuration is a normalized five-table schema: `model`, `grid`, `solver`, `initial_condition`, and `output`. `run_name` is limited to 64 safe ASCII filename characters. `ensure_ascii=True` makes character count equal to encoded ASCII byte count, so validation uses `len(text)` without allocating a full byte copy. A worst-case JSON-escaping bound covers that configuration, bounded Git status and compiled-analysis reason, at most eight bounded device descriptions, fixed metadata, and one CSV header or record. The separate live-peak formula above covers simultaneous text, escaped fragments, bounded ASCII bytes and file buffers. CSV validation uses a non-retaining counting sink, and JSON/CSV output is written through bounded binary chunks. Exact final checks run before the output directory is created, and the potentially long diagnostic-time list is represented by its bounded selection rule and row count.

After the static gate accepts, the runner constructs only the small grid tables and lowers/compiles from a `ShapeDtypeStruct`, without a full pair allocation. When the backend exposes JAX executable memory statistics, the second gate calculates

```text
compiled device = argument + output + temporary - alias
compiled host = host_argument + host_output + host_temporary - host_alias
compiled combined = compiled device + compiled host + static host.
```

The available compiled device requirement must not exceed the static device allowance, and both static and compiled combined requirements must stay within the fixed 256 MiB normal-run cap. Unavailable, incomplete, invalid or inconsistent executable analysis is itself a safeguard violation: normal execution fails before histogram or pair allocation, while explicit recorded `--allow-expensive` may proceed with status `unavailable` and must not claim a pass. Compilation code/cache, backend allocator overhead, imported-library/interpreter memory and TOML objects already parsed before preflight are explicitly excluded. Static caps bound the shape presented to compilation, but cannot make compiler-process memory itself exactly predictable.

Other fixed caps remain `M<=4096`, `M**2<=4,000,000`, `D<=8,000,000`, initial pair mass at most 64 MiB, zero retained full-density snapshots and at most 10,000 emitted rows. Configuration cannot raise any cap. Only recorded `--allow-expensive` can override static or compiled resource violations; scientific validity checks still run. For the default `G=5`, `M=25`, `D=1250`, `K=64`, `T=4` float32 run, static device, host and combined allowances are 53,704, 3,704,768 and 3,758,472 bytes. The normalized configuration occupies 341 ASCII characters; the maximum encoded metadata-plus-one-CSV-write payload remains 110,933 bytes, while the separately audited live serialization peak is 2,629,624 bytes. The CPU executable reports 5,412 device argument, 5,293 device output, 13,008 device temporary and zero device-alias bytes, with all four compiled-host fields zero. The resulting requirements are 23,713 compiled-device and 3,728,481 compiled-plus-host bytes.

Phase 4 has 69 focused tests: `66 passed, 3 skipped` by default and all 69 pass with CPU+x64 enabled. The complete suite now collects 222 tests: `218 passed, 4 skipped` by default and with warnings treated as errors, and all 222 pass in a fresh CPU+x64 warnings-as-errors process. Validation on 2026-08-19 used Python 3.12.10, NumPy 2.2.4, and JAX/JAXLIB 0.7.2 on the CPU backend. No GPU device was detected or tested.

Phase 4 does not derive cross-opponent covariance from pair data, compare pair theory with ABM variance, use interpolation, run the original `G=131` allocation, or claim GPU validation. Generated experiment data belongs under ignored `outputs/`.

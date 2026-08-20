# Chu et al. pair-approximation numerics

This self-contained subproject preserves the original `case2_1.py` calculation and develops independently tested numerical components for the model in Chu et al. Phases 1 and 2 provide shared model semantics, a readable small-grid NumPy pair-mass reference, and a finite-population JAX agent-based simulation. Phase 3A adds selected-action ABM variance instrumentation and two-dimensional Q-bin moment estimators. Phase 3B adds independent-run bootstrap intervals and nested Q-bin refinement diagnostics. Phase 4 adds a guarded CPU-validated JAX implementation of the exact legacy pair-mass transport. Phase 5 adds the first bounded, source-time- and bin-matched four-way pair-versus-ABM variance comparison. Counterfactual diagnostics, full-grid production runs, interpolation studies, and GPU benchmarking remain later phases.

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

## Phase 5 matched variance comparison

Phase 5 matches ABM source records at step `t` to pair mass `P_t`, before either update to `t+1`. The reduced runner uses the same action/state ordering, tensors, `alpha`, `tau`, population size, seeded legacy histogram, independently sampled pair endpoints, uniform initial state law, configured/effective bin edges, selected action and explicit source-time labels. A mismatch in source-time labels is an error.

At every exact pair-grid point and action, it retains raw sums proportional to `a=p(q) pi_j(q)`, `a mu`, `a m2`, `a mu^2`, `a q_j`, `a q_j^2`, and `a mu q_j`. These are added over a finite bin before nonlinear moments are formed. Therefore the pair finite-bin covariance is `E_B[mu(q,j)^2]-E_B[mu(q,j)]^2`, which need not vanish even though distinct opponents are independent conditional on exact `q`. The four reported estimands are direct realised ABM variance, ABM moment reconstruction, the full finite-bin pair closure, and a hybrid that replaces only the pair finite-bin cross-opponent covariance with the matched ABM estimate. The output also distinguishes pooled one-edge variance from the weighted mean of local exact-grid variances.

The ABM is simulated once and the pair density follows one trajectory. One JAX `lax.scan` begins at `P_0`, records requested point summaries from `P_t`, evaluates the authoritative Phase 4 diagnostics at every retained trajectory state, transports to `P_(t+1)`, and returns the final mass. It retains no density history. Lowering returns a runtime-only bundle containing the exact compiled callable, its memory report, abstract arguments, static values, and backend/device/x64 signature. After all guards pass, that same callable—not the original jitted Python wrapper—is invoked once. An invocation signature rebuilt from the actual mass, grid, scalars, source slots and current runtime must independently match pair shape/dtype, grid leaves, `alpha`, `tau`, step/summary counts, requested times, source-slot map, chunk size, tolerances, backend/platform/device identity and x64 state. The authoritative finest ABM and pair sufficient sums reconstruct one coarser scheme at a time. One reproducible complete-run bootstrap multiplicity matrix is shared across all ABM-dependent rows and schemes. Pair-only quantities are deterministic and have no bootstrap sampling interval.

Run the guarded CPU-scale comparison with:

```bash
python experiments/run_velocity_variance_comparison.py --config configs/velocity_variance_comparison_small.toml
```

It writes `variance_comparison.csv`, `anchor_bin_refinement.csv`, `bootstrap_run_weights.npz`, and bounded provenance/formula/resource metadata beneath ignored `outputs/variance_comparison/`. Configured and effective bin bounds, empty/sparse flags, counts, pair masses, raw-moment derivatives, all four variances, discrepancies, guarded ratios, and valid/invalid complete-run replicate counts are explicit. Rows are generated twice without collection: first through a non-retaining validation/counting sink and online descriptive accumulator, then through the bounded writer after all pre-output checks pass. At most one row object and one ASCII CSV write are live. Worst-normalized-schema tests through the real 91-column comparison and 68-column anchor iterators measured 14,579/11,209 live Python bytes and 4,450/4,022 ASCII characters respectively, below fixed 16 KiB live-object and 8 KiB record bounds.

The allocation-free Phase 5 model uses `L` requested times, `M=G**2` focal points, `C_l=L*Bc_l*Bd_l*2`, pair scalar width `b`, `T` transports and `B` bootstrap replicates. It budgets `15LMb+2Mb+8L` host point storage and `15LMb` device source summaries, the complete `(T+1)(11b+3)` diagnostic output on both device and host, and all `T` destination-validity booleans on both device and host. The static pair-kernel allowance reuses only Phase 4 arrays created by the combined executable; it excludes Phase 4 diagnostic dictionaries, source-hash storage and Phase 4 JSON/CSV buffers. Separate lifetime peaks cover normalized configuration, shape lowering/compilation, ABM simulation, pair execution, pair transfer/validation, finest-plus-one-coarse reconstruction, aggregation, pooled derivation, bootstrap chunks, bounded anchors, one streamed row, and Phase 5 serialization. The guarded global peak is their maximum, with every component and determining phase recorded. JSON encoding, bootstrap-weight archive writing, metadata chunk writing, and CSV writing are modeled as alternative subphases; the already encoded metadata string is retained during the latter three, but unrelated transient buffers are not summed. The comparison lifetimes include `56C_l+8(C_l/2)` pair sufficient storage, `512*sum(C_l)` retained comparisons, `64B*max(C_l)` bootstrap work, the 16 KiB row bound and Phase 5's own 8 KiB CSV writer allowance. Raw list counts and fixed caps precede NumPy bin construction. Configured/effective float edges, float32 collapse, nesting and anchors are then fully validated before grid/JAX construction. Exact executable analysis fails closed before histogram sampling, pair/ABM allocation or simulation, and the global check is repeated with compiled statistics. Configuration cannot raise the caps; only recorded `--allow-expensive` overrides resource failures, never scientific validity.

For the checked-in float32 smoke configuration, the final phase peaks in bytes are: configuration 33,600; compilation 41,776; ABM simulation 12,928; pair execution 3,190,487; pair transfer/validation 3,521,446; reconstruction 810,479; aggregation 772,367; pooled derivation 784,111; bootstrap 1,032,687; anchors 720,431; streamed output 2,981,423; and serialization 7,077,423. Serialization is therefore the global peak. The diagnostic device/host copies are 94 bytes each, destination validity is one byte each, the static pair-kernel allowance is 3,177,559 bytes, and executable analysis reported 1,361,059 device and zero host bytes. On the heterogeneous scan-parity fixture, the maximum source-summary difference was `1.4901161193847656e-08` in float32 and `2.7755575615628914e-17` in x64; final masses were bitwise equal in both modes (test tolerances `3e-6` and `1e-12`).

This milestone is a pipeline and smoke validation, not a production result. It does not run `G=131`, interpolate transport, analytically predict exact-Q cross-opponent covariance, benchmark a GPU, or establish coverage or scientific convergence.

Phase 5 has 49 focused tests: `48 passed, 1 skipped` by default and all 49 pass in CPU+x64. They include independent regressions for exact analyzed/executed callable identity, every execution-signature field, Phase 5 lifetime arithmetic and destination booleans, maximum-schema streamed rows, scan-versus-repeated-step parity, fourteen nested levels, early effective-bin/nesting/anchor rejection, diagnostic-tolerance enforcement and the zero-transport `P_0` case. On 2026-08-20 the complete warnings-as-errors suite reported `266 passed, 5 skipped` by default and `271 passed` in a fresh CPU+x64 process.

## Exact separable pair transport milestone

The production-oriented exact kernel keeps the Phase 4 flat scatter as the unchanged default and explicit validation oracle. For each old state and joint action it applies

```text
P_s(i,k) pi_a(q_i) pi_b(q_k)
  -> P'_{T(s,a,b)}(F_a(i),F_b(k)).
```

The implementation gathers `F_a` only on a static row block and `F_b` only on a static column block, forms one `B_r x B_c` weighted tile, and scatter-adds that tile before proceeding to the next tile and branch. It therefore retains the `(2,M,M)` carry and destination but does not create four full branch matrices, four full destination matrices, a `D x 4` array, a full row-transport intermediate, or a density history. Colliding scatter reductions may be accumulated in a backend-dependent order, so float32/x64 comparisons use declared tolerances rather than promises of bitwise cross-backend reproducibility. A measured sort-plus-segment microbenchmark is recorded; it was not adopted because its explicit order vector increased the compiled temporary peak.

The combined source-summary scan preserves the Phase 5 chronology: summarize `P_t`, diagnose `P_t`, transport to `P_(t+1)`, and retain only requested `15M` point summaries plus `(T+1)` lean diagnostics and `T` destination-validity flags. Both benchmark modes take the authoritative one-agent histogram and controlled `(0.5,0.5)` state law and form the independent ordered pair law inside the analyzed device executable. The bounded production-oriented result omits the final density; a separate combined validation executable returns it only for reduced-grid NumPy/flat/separable parity. No standalone initializer executable runs before analysis.

Run the guarded CPU benchmark with:

```bash
python -m pytest -q tests/test_pair_separable.py tests/test_pair_separable_runner.py
python experiments/run_pair_separable_benchmark.py --config configs/pair_separable_benchmark_small.toml
```

The exact normalized schema admits at most six cases. Before `QGrid`, JAX arrays or compilation, allocation-free Python integers check `G<=17`, `M<=289`, `D<=167042`, one density at most 1,336,336 bytes, `T<=4`, at most five requested summaries, block dimensions at most 289, two or four balanced repetitions/two warmups, 24 timing records, at most 104 case-plus-reduction timing samples, 8 MiB total case densities, 128 MiB static device/host planning limits and a 256 MiB static combined limit. Host preflight includes mode-specific histogram/grid/output arrays and a fixed allowance for one bounded lowered-HLO string plus chunked hashing; HLO text itself is rejected above 8 MiB. Configuration has no limit table and cannot raise these caps. All four combined initializer/scan objects and both reduction microbenchmarks compile before their device inputs, warm-up or timing. One central validator requires a factory-created runtime bundle, the exact retained compiled interface, nonempty abstract/static/environment contracts, versions/device/x64 facts, signature and bundle-integrity digests, complete internally consistent storage fields, and a fresh `memory_analysis()` that agrees with the stored report. Immediately before every call, the invocation signature is rebuilt internally from the actual argument pytree, weak types, tolerance, static context and current runtime; only the retained callable is invoked. Missing/malformed analysis or identity never reaches execution. `--allow-expensive` is limited to documented bounded-development static caps; bundle identity/completeness, live analysis, invocation matching, production capacity and scientific diagnostics remain non-overridable.

For the checked-in CPU float32 matrix (`G=3,5,9`), bounded-from-histogram requirements are 19,183 versus 5,743 bytes, 159,535 versus 25,023 bytes, and 290,383 versus 247,759 bytes for flat versus separable. The combined full-validation objects require 19,135/5,695, 164,543/25,039 and 290,335/247,647 bytes. Maximum final-density and diagnostic parity error under the corrected uniform state fixture was `3.725290298461914e-09`; source summaries agreed exactly. Timing compiles and gates both kernels first, warms both, then alternates flat/separable and separable/flat. Contract validation and compilation are outside the measured interval; every synchronized execution sample, position, order, median, min, max and MAD is retained within fixed caps. Across three isolated CPU runs after the complete live-bundle corrections, the six `G=9` bounded samples were 0.832–1.091 ms flat (median 0.870 ms, MAD 0.030 ms) and 0.512–0.661 ms separable (median 0.524 ms, MAD 0.012 ms). These are CPU observations, not GPU-performance predictions.

The allocation-free `G=131` projection uses `M=17161`, `D=588999842`: one density is 2,355,999,368 bytes (2.194 GiB) in float32 or 4,711,998,736 bytes (4.388 GiB) in float64. Static kernel allowances are 9,437,810,276 and 18,874,782,596 bytes. Multiplying density bytes by the worst deliberately small CPU ratio (`8.862654320987655`) gives empirical planning projections—not formal bounds—of 20,880,407,980 and 41,760,815,959 bytes. The 25% planning thresholds are 26,100,509,975 bytes (24.308 GiB) and 52,201,019,949 bytes (48.616 GiB). Modeled coexisting numerical host arrays are only 11,881,323 and 23,212,027 bytes; separately reported heuristic two-density staging reserves are 4,711,998,736 and 9,423,997,472 bytes, producing the unchanged 4,723,880,059 and 9,447,209,499 planning thresholds. The reserve is not two observed host pair arrays and does not bound compiler RSS/code cache, Python/library RSS or allocator overhead.

A future production decision requires a runtime bundle for the exact full separable executable and fresh capacity evidence matched to its exact JAX device. The optional bounded `nvidia-smi` provider matches GPU UUID, MIG UUID or normalized PCI identity. Numeric `CUDA_VISIBLE_DEVICES` entries require an injected trusted CUDA-runtime mapping from the logical visible ordinal to one of those stable identities; they are never compared with physical `nvidia-smi` indices, including under reordered visibility or `CUDA_DEVICE_ORDER`. The provider records total/free/used bytes and requires an explicit JAX allocator/preallocation policy. Usable bytes are current free bytes, further limited by allocator-available bytes when known. An internal UTC clock enforces the immutable 60-second maximum age (with only one second of future clock skew), and production invocation rechecks admission immediately before execution. Stale, ambiguous, total-only, mismatched, insufficient or unavailable evidence fails closed. Executable identity, GPU backend and verified capacity cannot be overridden by `--allow-expensive`. No `G=131` array was allocated, lowered or compiled and no GPU was present or tested.

The separable milestone adds 52 focused tests: `49 passed, 3 skipped` by default and all 52 pass in CPU+x64. On 2026-08-20 the complete warnings-as-errors suite reported `315 passed, 8 skipped` by default and `323 passed` in a fresh CPU+x64 process. Validation used JAX/JAXLIB 0.7.2 on `CpuDevice(id=0)`; no GPU was present or tested.

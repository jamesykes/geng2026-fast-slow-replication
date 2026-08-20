# Staged implementation and validation plan

## 1. Objective and non-goals

The eventual project will provide two independent numerical routes for the same model:

1. a deterministic pair-mass solver that can reproduce `case2_1.py` and scale to a GPU; and
2. a finite-population JAX agent-based simulation (ABM) that preserves the paper's one-action-per-agent, complete-graph chronology.

Both routes will feed a common statistics layer for mean trajectories and realised Q-update variance, including a deliberate investigation of covariance between rewards from distinct opponents.

The primary statistical object is

```text
D_j(q,t) = Var[v_j^i | Q_t^i=q, A_t^i=a_j]
         = alpha^2 Var[r_t^i(a_j) | Q_t^i=q, A_t^i=a_j].
```

The first version will measure distinct-opponent covariance from the ABM and will treat `c_j=0` only as an explicitly labelled conditional-independence pair closure. It will not implement a triplet closure or substitute an unconditional action-mixture variance for `D_j(q,t)`.

The original paper and `case2_1.py` remain immutable provenance artifacts. Phases 1, 2, the bounded selected-action Phase 3A and independent-run uncertainty Phase 3B milestones, the bounded CPU Phase 4 JAX pair solver, Phase 5 matched comparison, and the exact separable part of Phase 6 are implemented. Counterfactual diagnostics, interpolation/convergence, an actual production GPU run, and full variance experiments remain future work.

## 2. Proposed repository structure

The following structure should be created in a later implementation task:

```text
chu2023/
├── AGENTS.md                         # local constraints and current test commands
├── MODEL_SPEC.md                     # durable model/chronology specification
├── PLAN.md                           # this staged plan
├── pyproject.toml                    # isolated package and test/tool configuration
├── case2_1.py                        # untouched original source
├── pair-approx_multi-agent_stochastic_games.pdf  # untouched paper
├── configs/
│   ├── case2_1_small.toml            # CPU validation configuration
│   ├── variance_small.toml           # small ABM/binning validation configuration
│   └── case2_1_full_f32.toml         # resource-gated full-grid configuration
├── src/
│   └── chu_pair/
│       ├── __init__.py
│       ├── config.py                 # typed numerical/experiment configurations
│       ├── model.py                  # actions, states, payoff and transition tensors
│       ├── policies.py               # stable two-action Boltzmann policy
│       ├── grids.py                  # Q grids, flattening, rounding/interpolation maps
│       ├── initial_conditions.py     # continuous, histogram, and exact-legacy initialisers
│       ├── observables.py            # shared means, marginals, invariant diagnostics
│       ├── pair_density/
│       │   ├── __init__.py
│       │   ├── numpy_reference.py    # small, readable CPU oracle
│       │   ├── payoffs.py            # conditional rewards, velocities, edge moments
│       │   ├── transport.py          # nearest/interpolated JAX pushforwards
│       │   ├── solver.py             # stepping, scans, checkpoints, result API
│       │   └── layouts.py            # (state,M,M) conversion and chunk policies
│       ├── abm/
│       │   ├── __init__.py
│       │   ├── complete_graph.py     # edge construction and synchronous step
│       │   ├── simulation.py         # replicated/batched runs and checkpoints
│       │   ├── sampling.py           # explicit keyed initialisation/actions
│       │   └── instrumentation.py    # S1/S2, realised velocities, cluster IDs
│       └── statistics/
│           ├── __init__.py
│           ├── velocity.py           # selected-coordinate conditional moments
│           ├── reward_moments.py      # edge variance/covariance decomposition
│           ├── conditioning.py        # Q bins, matched weights, occupancy flags
│           ├── pair_predictions.py    # pair one-edge and c=0 predictions
│           └── comparisons.py         # errors, intervals, tabular summaries
├── experiments/
│   ├── reproduce_case2_1.py          # exact legacy-compatible mean trajectory
│   ├── run_abm_baseline.py           # paper-like finite-population runs
│   ├── validate_abm_variance.py       # direct/decomposed ABM estimator checks
│   ├── velocity_variance.py          # four-way theory versus ABM comparison
│   ├── pair_grid_convergence.py      # spacing/interpolation/precision study
│   └── benchmark_pair_gpu.py         # throughput and peak-memory benchmark
├── tests/
│   ├── conftest.py                   # tiny deterministic model fixtures
│   ├── test_model.py                 # payoff and transition truth tables
│   ├── test_grids.py                 # bounds, flattening, rounding/interpolation
│   ├── test_initial_conditions.py    # mass, moments, seeded repeatability
│   ├── test_pair_one_step.py          # hand-computed and NumPy/JAX parity cases
│   ├── test_pair_invariants.py        # mass, non-negativity, symmetry
│   ├── test_pair_reward_moments.py    # pair mu/m2/sigma calculations
│   ├── test_abm_one_step.py           # fixed actions/states exact update
│   ├── test_abm_statistics.py         # seeded Monte Carlo moment checks
│   ├── test_velocity_variance.py      # exact S1/S2 and variance identities
│   ├── test_conditioning.py           # Q weights, dispersion, sparse-bin policy
│   └── test_case2_1_regression.py     # recorded small/full reference observables
└── outputs/                           # ignored/generated, never mixed with sources
    ├── reference/
    ├── experiments/
    └── benchmarks/
```

Shared definitions must own action ordering, state ordering, payoff tensors, transition tensors, Q-update equations, and chronology. The ABM and pair solver must not carry independent hard-coded copies.

## 3. Design rules

- Treat this directory as a self-contained Python subproject without initialising another Git repository.
- Never modify the original script or paper. Record their hashes in regression metadata.
- Make all random operations explicit and seeded. Store seed, configuration, dtype, backend, package versions, and source hash with every result.
- Keep model semantics separate from numerical policy. Transition rules, nearest-grid rounding, interpolation, array layout, and chunk size are separate configuration choices.
- Use one action per agent per timestep in the ABM and reuse it on every incident edge.
- Read payoff and transition outcomes from old edge states, then synchronously commit Q and edge-state updates.
- Keep ABM Q-updates continuous. Grid projection is a pair-solver numerical policy, not an ABM learning rule.
- For the controlled baseline, initialise agents from the same seeded discrete histogram as the pair solver and sample edge states independently with probability one half per state.
- Retain continuous scaled-Beta initialisation as a secondary paper-like ABM mode.
- Treat exact-Q, selected-action conditional variance as the primary estimand. Keep finite-bin approximations and unconditional action mixtures explicitly labelled.
- Use counterfactual focal actions only to improve diagnostic edge-moment estimates; direct velocity validation must use the coordinate actually selected.
- Estimate cross-opponent products in O(n) work per focal agent from `S1` and `S2`, never by enumerating opponent pairs.
- Match Q weights and action-conditioning weights between ABM bins and pair predictions.
- Measure `c_j` from the ABM in the first version. Do not implement a triplet closure.
- Use independent simulation runs as the primary uncertainty sampling units. Retain focal-agent IDs for diagnostics, but never count agents in one complete network as independent replicates.
- Keep a readable NumPy CPU oracle even after the JAX implementation exists.
- Enable JAX 64-bit explicitly in strict CPU parity tests. Treat GPU float32 as a separate validated numerical mode.
- Avoid automatic differentiation; it provides no value here and increases memory pressure.
- Validate scientific quantities and invariants, not only that code executes.

## 4. Phase 0 - Freeze semantics and reference data

Status: core provenance and reproducible small fixtures are implemented. A recorded full/reference trajectory remains future resource-gated work.

Tasks:

1. Preserve the source hashes recorded in `MODEL_SPEC.md`.
2. Treat the committed deletion of the old `case2_1_jax.py` as intentional; do not reconstruct or rely on it as the implementation baseline.
3. Add a tiny deterministic initial pair mass directly as test data; do not depend on random Beta samples for one-step unit tests.
4. Add an exact legacy-initialisation mode using `random.Random(seed)` and the original two-dimensional histogram procedure.
5. Record seeded reduced-grid reference trajectories before attempting the full 131-grid case.
6. Define a trajectory schema containing timestep labels, observables, invariant diagnostics, configuration, seed, dtype/backend, and source hashes.
7. Define the variance schema: action, two-dimensional Q-bin, selected/counterfactual count, Q dispersion, reward/velocity moments, edge moments, distinct-edge product, weighting rule, and run/focal cluster identifiers.

Exit gate:

- A reviewer can reconstruct the chosen reference initial mass, explain every update, and identify the exact statistical conditioning from the specification and fixtures alone.

## 5. Phase 1 - Shared model definitions and small NumPy oracle

Status: **complete (2026-08-18)**.

Implement the common model layer and a deliberately simple loop-based NumPy pair reference.

Tasks:

1. Define `Action(C,D)`, `State(SH,PD)`, payoff tensor shape `(state, own_action, opponent_action)`, and transition tensor shape `(old_state, own_action, opponent_action)`.
2. Define immutable learning and grid configurations for this milestone. Add ABM- and conditioning-specific configuration objects only in their later phases.
3. Implement stable Boltzmann probabilities using `sigmoid(tau*(Q_C-Q_D))`.
4. Implement grid flatten/unflatten maps and an exact compatibility version of `appro()` for the active spacing.
5. Implement conditional payoff/velocity calculation and the four-branch nearest-grid pushforward in clear NumPy.
6. Implement the shared one-edge lookup `y_j(s,b)` and the pair formulas for `mu_j`, `m2_j`, and `sigma_j^2`.
7. Implement shared observables and invariant checks.

CPU tests first:

- payoff matrices and the eight-entry transition truth table;
- action probabilities sum to one and agree with the original formula;
- hand-built one-cell and two-cell pair transports;
- one-step comparison with an independently hand-calculated result;
- mass conservation, non-negativity, and endpoint-exchange symmetry;
- no movement of an unselected Q-coordinate;
- safe handling of zero focal marginal mass;
- exact one-edge first/second moments for a hand-built pair distribution;
- explicit failure or configured handling for out-of-range destinations;

Exit gate:

- **Met.** The authoritative model, continuous update, stable policy, grid/projection utilities, reusable seeded histogram, guarded ordered pair construction, conditional dynamics, one-edge moments, synchronous transport, observables, and diagnostics are implemented.
- The implementation deliberately keeps the Phase 1 payoff/moment/transport oracle together in `pair_density/numpy_reference.py` for readability instead of prematurely splitting it across the later planned JAX-oriented modules.
- Validation command at the Phase 1 boundary: `python -m pytest -q` after installing `.[test]`; the milestone suite reports `37 passed`.
- No scientific deviations from `MODEL_SPEC.md` were introduced. The unknown published histogram was not reconstructed, no full 131-point pair array was allocated, and no JAX or ABM code was started.

## 6. Phase 2 - JAX finite-population ABM

Status: **complete (2026-08-18)**.

Build the ABM immediately after the shared definitions and small reference tests. This is intentionally earlier than the full JAX/GPU pair transport because it is the quickest route to validating the target variance and covariance estimator.

Tasks:

1. Construct upper-triangle edge endpoint arrays for the complete graph.
2. Initialise agent Q-values and edge states with explicit keys. The primary controlled mode samples Q from the corresponding seeded pair histogram and samples edge states independently with probability one half per state. A continuous paper-like Beta Q mode remains secondary.
3. Draw exactly one action per agent each timestep and reuse it on all incident edges.
4. Gather endpoint actions and old edge states; compute both endpoint payoffs on every edge with correct payoff orientation.
5. Aggregate rewards by segment sums, divide by `n-1`, and update only each agent's selected Q-coordinate.
6. Keep those ABM Q updates continuous; do not call pair-grid projection code after initialisation.
7. Transition every edge from its old state and the same endpoint actions.
8. Support small replicated runs first, then batched/chunked replications as memory requires.
9. Record pre-update state at label `t`, realised actions/rewards/velocities for that step, and post-update state at `t+1` without ambiguity.

Deterministic ABM tests:

- `n=2` with fixed Q/actions/state, checked by hand;
- `n=3` with fixed actions and three distinct edge states, verifying endpoint payoff orientation and averaging;
- one agent's action is identical on all incident edges;
- Q updates use old-state rewards and only the selected coordinate;
- transitions use old state plus the same realised actions;
- fixed seeds reproduce actions and trajectories on the same backend.

Statistical ABM tests:

- empirical action frequencies match Boltzmann probabilities;
- empirical initial Beta moments and state proportions match targets;
- increasing independent repetitions shrinks Monte Carlo error at the expected rate;
- small-population mean observables agree with a separate slow reference simulation using fixed seeds.

Exit gate:

- **Met.** Exact `n=2` and heterogeneous `n=3` fixed-action cases establish payoff orientation, `n-1` averaging, old-state rewards, selected continuous updates, shared actions, and synchronous transitions.
- Grid-matched and continuous scaled-Beta modes use explicit independent JAX keys. `lax.scan`, the small Python debug path, JIT, and vmapped independent runs agree on CPU.
- Default scan records retain Q, policies, actions, rewards, and selected velocities but no full edge histories. The guarded baseline remains far below `n=1000`.
- The baseline applies hard pre-allocation Phase 2 caps, records any explicit override, and hashes the current implementation even when the worktree is dirty. Grid-matched initialization is guarded before histogram allocation or sampling by fixed operational caps of 5,000,000 cells, 32 MiB of `int64` counts, and 2,000,000 sample pairs; only `--allow-expensive` bypasses them.
- Runner tests establish that `T=0` reports the initialized state without an update and that the final `Q_T,S_T` row for `T=1` is computed from the post-step final state, including mean Q, policy, and edge-state proportions.
- Validation on 2026-08-18 used JAX/JAXLIB 0.7.2 on `cpu` with `CpuDevice(id=0)`. The 44 focused Phase 2 tests and 37 Phase 1 tests give `81 passed` in both default warnings-as-errors float32 and a fresh CPU+x64 warnings-as-errors process.
- Exact float32 dynamics use `rtol=0` and `atol` between `2e-7` and `3e-7`; the x64 tolerance is `1e-12`. Policy parity uses `rtol=1e-6`, and statistical tolerances are declared beside each sampling test.

## 7. Phase 3 - ABM variance and covariance instrumentation

### Phase 3A - selected-action moments and binning

Status: **complete (2026-08-18)**.

Phase 3A implements the primary realised selected-coordinate sample before the full pair rewrite. For each focal agent with realised action `j=A_t^i`, it streams

```text
S1 = sum_h Y_ih,
S2 = sum_h Y_ih^2,
distinct_sum = S1^2-S2,  N=n-1,
```

from the old-state endpoint payoffs already used by the ABM. It does not enumerate opponent pairs, insert an unselected-coordinate zero, retain edge histories, or consume another random key.

Implemented scope:

1. Retain source `Q_t`, realised selected action, selected `Q_t(j)`, reward, velocity, `S1`, and `S2` with explicit run/time/agent axes.
2. Aggregate by independent run, source time, two-dimensional source-Q bin, and selected action.
3. Store counts and raw sums for `S1`, `S2`, `S1^2-S2`, reward, selected Q, reward-selected-Q products, velocity, and all required squares.
4. Derive population `mu`, `m2`, `m11`, `sigma^2`, `c`, direct/decomposed reward variance, and direct/finite-bin-corrected velocity variance.
5. Report both

   ```text
   Var[alpha(r-Q_j) | Q in B,A=j]
   ```

   and `alpha^2 Var[r | Q in B,A=j]`, plus the exact finite-bin difference from Q dispersion and reward-Q covariance.
6. Use deterministic `[lower,upper)` two-dimensional bins with the final upper edge included. Retain configured float64 edges for provenance, compare against edges converted to the observation dtype, reject conversion-collapsed edges and genuinely out-of-range observations, and retain explicit empty/underpopulated/validity flags.
7. Preserve independent runs as the uncertainty-replicate axis and treat `c` as undefined for `n=2`.
8. Emit a small diagnostic CSV and metadata record without claiming the final pair-theory comparison or inferential intervals.
9. From raw parsed edge-sequence lengths, and before constructing `QBinSpec` or any NumPy edge copy, guard `R*T*Bc*Bd*2` dense strata, dtype-aware audited peak statistic bytes, and the identical number of dense output rows with fixed caps of 1,000,000 strata, 256 MiB, and 250,000 rows. Only the explicit recorded `--allow-expensive` override bypasses them; TOML parser list memory is outside this guard.

Phase 3A tests establish:

- `S1^2-S2` equals an explicit ordered-distinct-pair sum on tiny arrays;
- hand-built edge configurations give exact `mu`, `sigma^2`, `c`, and average-reward variance;
- `n=2` omits `c` and uses only the single-edge term;
- exact synthetic configurations recover known `c=0` and nonzero `c`;
- the raw-moment identity

  ```text
  Var(mean edge payoff)
    = sigma^2/N + (N-1)c/N
  ```

  closes under the same weights;
- a deliberately wide Q-bin demonstrates that raw velocity variance differs from `alpha^2` times reward variance by the recorded Q terms;
- both Q coordinates and the realised action determine the stratum;
- empty, single-observation, boundary, out-of-range, and sparse-bin behavior is deterministic;
- batched and separately processed runs agree, while instrumented and uninstrumented simulations have identical actions, trajectories, final states, and final keys;
- retained histories have no time-by-edge payoff arrays and source-time labels use `Q_t,S_t`.
- float32 and float64 endpoint tests distinguish configured from effective comparison edges, enforce the final inclusive endpoint, reject adjacent outside values, and reject edges collapsed by dtype conversion;
- allocation-free resource tests hand-check every estimate component and prove that a 1,000-by-1,000 bin configuration is rejected before simulation or statistic allocation;
- a zero-step low-limit regression proves configured/effective NumPy edge storage remains visible and rejection precedes `QBinSpec`, while an accepted-order test checks raw and constructed bin counts agree;
- independent float32/float64 calculations prove that only float32 allocates five conversion arrays and that x64 is not rejected by the removed `40O` double count;
- the Phase 2 default resource estimate remains byte-for-byte equal to the committed Phase 2 guard, while Phase 3A explicitly requests the three-field instrumented increment.

Exit gate:

- **Met.** On small CPU runs, the reward variance decomposition closes, direct realised-velocity variance is separately identified, finite-bin Q/covariance terms are visible, independent runs remain separate, and no ordered opponent-pair or time-indexed edge-payoff array is constructed.
- The 28 focused Phase 3A tests and the associated Phase 2 resource regressions give `115 passed` in the complete default suite.

### Phase 3B - independent-run uncertainty and Q-bin refinement

Status: **complete (2026-08-19)**.

Phase 3B reuses the unchanged Phase 3A instrumented trajectories and sufficient sums. Its pooled point estimate sums counts and raw sums across independent runs before applying nonlinear moments, so it is observation weighted rather than an unweighted average of run variances. Its sole uncertainty resampling unit is one complete independent ABM run.

Implemented scope:

1. Generate one reproducible int32 bootstrap multiplicity matrix `(B,R)` from a bootstrap-only NumPy seed; every row contains `R` run draws.
2. Apply the same complete-run weights to every time, Q bin, selected action, estimand, and refinement scheme, without touching JAX keys or trajectories.
3. Recompute all pooled nonlinear Phase 3A moments per replicate and add the two algebraically equivalent finite-bin discrepancy estimates.
4. Form pointwise percentile intervals with `numpy.quantile(method="linear")`. Require at least two contributing runs and at least `max(2,ceil(0.8B))` finite replicates; otherwise retain `NaN` endpoints plus explicit counts and flags.
5. Apply several named, strictly nested two-dimensional bin schemes to one simulation. Check common configured/effective bounds, strict refinement in both axes, conversion collapse, exact count reconstruction, and field-specific gamma-bound reconstruction of differently ordered floating sums.
6. Map configured anchors with the same `[lower,upper)` and final-inclusive rule, reporting configured/effective bounds and widths, counts, point values, intervals, flags, and finite-bin discrepancy by time/action/scheme.
7. Aggregate schemes sequentially and process bootstrap strata in bounded chunks. Never construct `B` by all-strata by all-estimands storage, a time-indexed edge-payoff history, or a stratum-sized CSV collection.
8. Run a separate guarded smoke runner which emits streamed pooled and anchor CSV files, exact bootstrap weights, metadata, resource estimates, seeds, versions, and dirty-tree source hashes under ignored `outputs/abm_uncertainty/`.
9. Before `QBinSpec`, simulation, aggregation, bootstrap allocation, or output, preflight all schemes using allocation-free raw-list lengths and fixed non-configurable normal-run caps. The audit includes sequential sufficient arrays, dtype-aware aggregation work, configured/effective edges at `T=0`, run weights and their conversions, retained summaries, pooled-point derivation, chunked bootstrap work, pooled rows, and anchor rows. Only recorded `--allow-expensive` may bypass violations.

Phase 3B tests establish:

- unequal run counts yield the hand-calculated observation-weighted pooled variance rather than a mean of run variances;
- fixed multiplicities give manual replicate variances and linear percentile endpoints, while a correlated-within-run example differs from invalid observation-level resampling;
- empty bootstrap replicates, one-run strata, valid/invalid replicate counts, and undefined interval endpoints follow the stated policy;
- bootstrap seeds reproduce weights and intervals without changing points, sufficient statistics, simulation randomness, or Phase 3A trajectory guarantees;
- chunk sizes give identical results and the derivation kernel never receives more than the configured number of strata;
- float32 and CPU+x64 results agree within declared tolerances;
- configured/effective nesting rejects non-nested, differently bounded, and float32-collapsed edges;
- child counts and all ten sufficient sums reconstruct their parent, and anchors follow interior-boundary/final-upper conventions;
- float32 represented-value bounds cover all ten actual sufficient terms, including the nonzero `S1**2-S2` rounding residual for `n=2`, and separate observation-dtype arithmetic from float64 reconstruction summation;
- the reviewed heterogeneous `n=32, R=2, T=4, seed=19` trajectories pass in float32 and CPU+x64, while omitted, duplicated, materially corrupted, and count-corrupted child contributions still fail;
- direct run-axis pooling matches explicit all-ones weighting without allocating either run-length weight vector, including hand-checked `T=0, R=32, B=1` float32/float64 peaks;
- one runner simulation is reused for every scheme and the identical weight object reaches every refinement level;
- a single original Phase 3A bin scheme recovers the committed pooled moment formulas exactly;
- independent hand arithmetic covers every float32/float64 resource component, `T=0`, every fixed cap, override recording, guard-before-`QBinSpec` order, and rejection before simulation/aggregation/bootstrap/output;
- streamed anchor output includes configured/effective widths and total/valid/invalid replicate information.

Exit gate:

- **Met.** A bounded CPU smoke run produces pooled selected-action moments, conservative complete-run bootstrap intervals, and nested-bin/anchor diagnostics from one trajectory set without claiming production coverage.
- The 38 focused Phase 3B tests give `37 passed, 1 skipped` in the default float32 process and `38 passed` with CPU+x64 enabled. They bring the complete suite to `152 passed, 1 skipped` in default and warnings-as-errors validation and `153 passed` in CPU+x64 validation.

### Deferred Phase 3 work

- Add separately labelled counterfactual `Y_ih^(C)` and `Y_ih^(D)` diagnostics and their selected-action weighting checks.
- Add focal-agent repeated-measure diagnostics only if a scientifically justified resampling design is specified; complete runs remain the current uncertainty unit.
- Choose production run count, bootstrap count, bin schemes, anchors, sparse-stratum thresholds, and any multiplicity-adjusted inference before scientific use.
- Perform the final pair/ABM four-way comparison in Phase 5, now that the bounded JAX pair solver exists; do not fold it into Phase 4.

## 8. Phase 4 - JAX pair solver on CPU

Status: **complete (2026-08-19) for the bounded exact-scatter CPU milestone**.

Port the deterministic pair solver component by component and compare with the NumPy oracle before using a GPU.

Tasks:

1. Use a `(state, M, M)` JAX pair layout and conversion helpers.
2. Vectorise marginals, observables, payoff matrix-vector contractions, and velocity fields.
3. Compute `f(q,t)`, `w_sb(q,t)`, `mu_j^pair`, `m2_j^pair`, and `sigma_j,pair^2` for both actions.
4. Implement a first correct small-array transport, even if it uses a simple flat scatter.
5. Evaluate separable endpoint transport (`A_a P_s A_b^T`) as a later production optimization. The bounded milestone intentionally retains the first exact flat-scatter formulation so its chronology is transparent and independently comparable with the NumPy oracle.
6. Add optional chunking with a pure functional accumulator and deterministic chunk-order CPU execution.
7. JIT one step, then a short `lax.scan`; keep I/O and metadata writes outside compiled code.

Validation matrix:

| Case | Backend/dtype | Required comparison |
| --- | --- | --- |
| Tiny hand cases | JAX CPU float32 | hand-derived policies, payoffs, velocities, destinations, branch masses, and transitions |
| Reduced grid, one step | NumPy float64/JAX CPU float32 | full array, destinations, and one-edge moments within declared float32 tolerances |
| Reduced grid, many steps | NumPy float64/JAX CPU float32 | full array, observables, moments, and invariants within declared float32 tolerances |
| Reduced grid, many steps | NumPy/JAX CPU float64 | full-array parity with tight x64 tolerances in a fresh process |
| Chunk sizes 1/many/all | JAX CPU | invariant, trajectory, and moment equivalence |

Exit gate:

- **Met.** NumPy and JAX CPU agree over one-step and four-step reduced cases, including exact projected destinations and the Q-resolved one-edge first moment, second moment, and variance needed by the variance project.
- Independent heterogeneous SH and PD fixtures cover nonuniform source policies, asymmetric endpoint payoff orientation, distinct endpoint velocity/projection maps, all four action branches, old-state transitions, complete transported mass, and exchange symmetry.
- The JIT-compatible implementation uses internal `(2,M,M)` mass, bounded source chunks, flat scatter-add, and `lax.scan`. It retains a lean diagnostic trajectory and no full density history. Chunk sizes one, intermediate, and all source cells agree.
- Default float32 and strict CPU+x64 parity use separately justified tolerances. No GPU backend, full-grid run, interpolation, separable production transport, pair-derived `c_j`, or final pair/ABM comparison is claimed by this phase.
- The runner validates an exact bounded configuration schema and serialization allowance before a conservative allocation-free static device-plus-host gate. Static host accounting includes the complete host diagnostic trajectory simultaneously with final-density validation. Serialization accounting distinguishes maximum ASCII payload from the maximum of JSON-encoding and bounded JSON/CSV-writing live peaks; validation makes no redundant full ASCII byte copy, CSV uses a non-retaining counting sink, and output uses bounded binary chunks. Shape-only compilation follows before histogram or pair allocation; complete executable analysis is required for normal execution, while unavailable/incomplete analysis fails closed unless explicit recorded `--allow-expensive` overrides it. Both resource gates share an immutable 256 MiB combined cap and scientific validity checks remain mandatory.
- The 69 focused Phase 4 tests give `66 passed, 3 skipped` by default and `69 passed` with CPU+x64 enabled. The complete suite gives `218 passed, 4 skipped` in default warnings-as-errors validation and `222 passed` in fresh CPU+x64 warnings-as-errors validation.

## 9. Phase 5 - Initial four-way variance comparison

Run the first scientific comparison at modest CPU-manageable scale before full GPU optimisation. This bounded milestone is implemented for reduced grids only.

For selected times, actions, and sufficiently occupied Q-bins, perform four distinct checks:

1. **Direct ABM variance:** estimate `D_j` from realised chosen-coordinate velocities.
2. **ABM decomposition:** estimate `sigma_j,ABM^2` and `c_j,ABM`, and verify

   ```text
   D_j^ABM approximately equals
     alpha^2 [sigma_j,ABM^2/(n-1)
              + (n-2)c_j,ABM/(n-1)].
   ```

   For finite bins, first close the reward-variance identity, then account for Q dispersion/reward-Q covariance when comparing to raw velocity variance.
3. **One-edge pair check:** compare `sigma_j,pair^2` with `sigma_j,ABM^2` under identical bin definitions and selected-action Q weights.
4. **Theory comparison:** compare direct ABM variance with the finite-bin raw-moment pair and hybrid predictions

   ```text
   pure pair:
     alpha^2 [sigma_j,pair^2/N + (N-1)c_j,pair/N
              + Var_pair(q_j) - 2 Cov_pair(r,q_j)]

   hybrid:
     alpha^2 [sigma_j,pair^2/N + (N-1)c_j,ABM/N
              + Var_pair(q_j) - 2 Cov_pair(r,q_j)].
   ```

   Here `c_j,pair=E_B[mu(q,j)^2]-E_B[mu(q,j)]^2`; exact-Q conditional independence does not force this finite-bin mixing covariance to zero. The earlier `alpha^2 sigma^2/N` shorthand applies only at exact `q`.

Tasks:

1. **Met.** Named direct/decomposed/pure-pair/hybrid arrays, discrepancies, guarded ratios and validity flags remain separate.
2. **Met.** Pair exact-grid values are weighted by `p(q) pi_j(q)` and accumulated as raw mass, first/second/distinct, selected-Q and reward-Q sums before nonlinear finite-bin formulas. Both pooled `sigma^2` and the weighted mean of local exact-grid `sigma^2(q,j)` are retained.
3. **Met.** Authoritative finest pair and ABM sufficient statistics reconstruct one nested coarser scheme at a time by raw addition; ABM parent/child sufficient statistics are independently reconstruction-checked and scheme-sized coarse arrays are released sequentially.
4. **Met.** One complete-run multiplicity matrix is shared across every row, scheme and ABM-dependent estimand. Pair-only values are deterministic and receive no sampling interval.
5. **Met.** The runner executes one ABM trajectory batch and the exact compiled `lax.scan` object whose memory report passed, retains no full pair-density history, and labels every requested source `t` against `P_t` before transport. A runtime-only bundle holds that callable, abstract/static compile facts and backend/device/x64 identity; an independently rebuilt invocation signature must match before the one call.
6. **Met.** Immutable Phase 2, 3B, 4-kernel and additional Phase 5 resource gates run before allocation-sensitive work. Complete effective-bin/nesting/anchor validation precedes JAX lowering. Exact executable analysis then fails closed before histogram, pair or ABM allocation. The Phase 5-specific lifetime model reports configuration, compilation, ABM, pair execution, pair transfer/validation, reconstruction, aggregation, pooled, bootstrap, anchor, streamed-row and serialization peaks; it counts `(T+1)` diagnostic rows and `T` destination-validity booleans on device and host without importing Phase 4 runner-only output buffers. Serialization takes the maximum of JSON encoding, CSV, weight-archive and metadata-write subphases while counting the encoded metadata string wherever it remains live.
7. The smoke configuration compares source times `0,1`, two nested bin schemes and both actions with the same model, population, legacy histogram seed and uniform initial state law.

Exit gate:

- **Met for the bounded pipeline, not as a production scientific conclusion.** The four checks are independently named and matched, exact ABM reconstruction closes to floating-point error, and the smoke output reports whether the empirical covariance correction moves the diagnostic hybrid. Full-grid, convergence and inferential conclusions remain later work.
- The real 91-column comparison and 68-column anchor iterators are tested at the maximum normalized configuration size. Their observed maxima (14,579/11,209 live Python bytes and 4,450/4,022 ASCII characters) fit the fixed 16 KiB live-row and 8 KiB record bounds; acceptance at the measured boundary and rejection one byte below are covered independently.
- Repeated one-step and combined-scan summaries agree at source times `0,2,3` on a heterogeneous symmetric mass: maximum differences are `1.4901161193847656e-08` in float32 and `2.7755575615628914e-17` in CPU+x64, with identical final masses.
- The Phase 5 focused suite gives `48 passed, 1 skipped` by default and `49 passed` in CPU+x64. On 2026-08-20 the complete warnings-as-errors suite gives `266 passed, 5 skipped` by default and `271 passed` in a fresh CPU+x64 process.

## 10. Phase 6 - Pair transport policy and CPU grid convergence

Keep exact legacy reproduction and improved numerics as distinct modes.

Status: **exact separable nearest-legacy kernel complete; interpolation and grid-convergence work not started.**

Tasks:

1. **Met for the exact transport implementation.** Preserve the original nearest-legacy round-to-grid behaviour and keep the committed flat kernel as the explicit default/oracle.
2. **Met.** Add an explicit `separable` selector implementing the same eight branch maps with bounded row/column tiles, including partial, unit and at/over-`M` block sizes. It retains neither a full row intermediate nor `D x 4` branch arrays.
3. **Met.** Construct the independent ordered pair density from the one-agent histogram and controlled half/half state law inside both combined compiled device paths. The bounded result has no final density leaf; the reduced validation object returns one only for parity. No standalone initializer executable remains.
4. **Met for reduced CPU cases.** Independently test hand branch weights, endpoint-specific maps, a fixed explicitly enumerated simultaneous row/column collision oracle, both states, all action branches, projection ties, zero support, symmetry, NumPy/flat/separable parity, eager/JIT, multi-step summaries, float32 and CPU+x64.
5. **Met for bounded feasibility, not production execution.** Preflight exact normalized small benchmark shapes before allocation; compile and completely analyze all four combined objects and both reduction executables before their device inputs or execution; bind each exact compiled interface to a factory-created runtime-only contract with signature/integrity digests and fresh live-memory agreement; rebuild invocation identity internally from actual arguments, tolerance, static context and runtime before every synchronized call; retain counterbalanced timing samples/positions/dispersion; and project `G=131` bytes allocation-free. Production capacity requires non-overridable exact identity, complete analysis, GPU backend, immutable 60-second freshness and usable-memory evidence matched by stable UUID/MIG/PCI identity or trusted CUDA-runtime mapping.
6. **Met as an investigation.** Measure a sorted-segment branch reduction against scatter. Do not adopt it because the measured order/sort temporary increased compiled memory.
7. Add conservative linear interpolation only after legacy parity is established.
8. Verify interpolation weights are non-negative, sum to one, and respect boundaries.
9. Run grid-spacing studies on manageable CPU cases, comparing `h`, `h/2` where feasible, nearest versus interpolation, and float32 versus float64.
10. Quantify grid locking and changes in mean trajectories and Q-resolved one-edge moments.
11. Repeat the small four-way variance comparison when numerical policy materially changes `sigma_j,pair^2`.

Exit gate:

- **Partially met.** Exact nearest-legacy flat/separable scientific parity and bounded CPU resource behavior are established. Reports distinguish backend accumulation tolerance from scientific transport changes, and the flat default is unchanged. Interpolation and CPU grid convergence are still required before this phase is complete.
- On the production-oriented bounded `G=9` CPU object, separable compiled device storage is 247,759 bytes versus 290,383 bytes for the flat solver with 64-cell chunks; the separate full-density-return validation object is recorded even where it is not smaller. This is reduced-grid CPU evidence only, not a GPU/full-grid claim.
- The 52 focused tests give `49 passed, 3 skipped` by default and `52 passed` in CPU+x64. The complete warnings-as-errors suite gives `315 passed, 8 skipped` by default and `323 passed` in CPU+x64. No full-grid allocation/lowering/compilation, interpolation or GPU validation was performed.

## 11. Phase 7 - Full GPU pair-density transport

The GPU solver remains a required deliverable, but begins only after ABM variance instrumentation, small pair parity, and the first scientific comparison are working.

Tasks:

1. Benchmark vectorised payoff/edge-moment contractions separately from transport.
2. Measure compiled peak memory for current density, destination density, endpoint maps, and every transport temporary.
3. Sweep endpoint chunk sizes and compare flat scatter, staged segment-sum, and viable sparse representations.
4. Avoid full `(branch, source_cell)` index and weight arrays. Generate branch data per chunk from length-`M` endpoint maps.
5. Test JAX buffer donation and verify whether it reduces actual peak memory.
6. Record compile time, step time, source-cell throughput, device model, JAX/XLA versions, precision, and peak bytes.
7. Add a preflight estimator that rejects configurations whose conservative peak estimate exceeds a configured device budget.
8. Check GPU run-to-run variation from scatter atomic order. Compare invariant, trajectory, and one-edge-moment tolerances rather than bitwise output.

Scaling gates:

- reduced cases reproduce CPU pair arrays and Q-resolved moments;
- no out-of-memory failure occurs at the selected production configuration;
- mass, non-negativity tolerance, symmetry, and moment diagnostics stay within limits;
- full 131-grid runtime and peak memory are reported before larger/finer grids are attempted.

## 12. Phase 8 - Reproduction, scaling, and final comparisons

Run experiments in increasing cost order:

1. Tiny exact cases.
2. Reduced grids and small populations (`n=2,3,10`) on CPU.
3. Short medium ABM/pair cases for the four-way comparison.
4. Figure 1(b)-like pair trajectory with a recorded seed/histogram.
5. ABM `n=1000` with a small number of repetitions as a smoke test.
6. Replication-count convergence, then the paper-like 500 repetitions if resources justify it.
7. Population-size scaling of `D_j`, `sigma_j^2`, and `c_j` with a separate ABM for each `n`.
8. Reuse one pair trajectory to form pure `c=0` predictions for several `n` values when model parameters/initial conditions are otherwise identical.
9. Learning-rate scaling at a fixed snapshot to verify `alpha^2`, followed separately by full new evolutions for each `alpha` because changing learning rate changes later distributions.
10. Grid/precision/transport-policy sensitivity studies on one-edge pair moments and variance predictions.

Primary saved outputs:

- Q-resolved `D_j`, `sigma_j^2`, and `c_j` at selected times;
- sufficiently populated Q-bin tables and heatmaps with counts and Q dispersion;
- occupancy-weighted time series with matched ABM/pair weighting;
- direct/decomposed/pure-pair/hybrid variance series;
- mean Q, action probabilities, state proportions, and pair invariants;
- cluster-respecting uncertainty intervals;
- absolute/relative comparison errors;
- configuration, bin definitions, seed sequence, source hashes, backend/device, dtype, timings, and package versions.

Under `c_j=0`, conditional variance should scale as `1/(n-1)`. At a fixed distributional snapshot, all conditional velocity variances scale as `alpha^2`. These are diagnostics, not licenses to rescale across trajectories generated with different learning rates.

Do not tune conclusions on the same simulations used for final validation. The hybrid uses ABM-measured covariance as a diagnostic correction, not as a standalone pair-theory prediction.

Secondary/future objects must remain outside first-version deliverables:

- unconditional coordinate-update variance without focal-action conditioning;
- covariance between `C` and `D` velocity coordinates;
- a triplet-density closure for `c_j`;
- long-time diffusion/Fokker-Planck approximations;
- kernel or local-regression estimators unless binning diagnostics require them.

## 13. Current and future validation commands

The Phase 1 through Phase 4 commands now exist; later-phase commands remain targets:

```bash
python -m pytest -q
python -m pytest -q tests/test_abm_graph.py tests/test_abm_one_step.py tests/test_abm_sampling.py tests/test_abm_simulation.py tests/test_abm_runner.py
python -m pytest -q tests/test_abm_variance.py tests/test_abm_variance_runner.py
python -m pytest -q tests/test_abm_uncertainty.py tests/test_abm_uncertainty_runner.py
python -m pytest -q tests/test_pair_jax.py tests/test_pair_jax_runner.py
python -m pytest -q tests/test_velocity_variance.py tests/test_velocity_variance_runner.py
python -m pytest -q tests/test_pair_separable.py tests/test_pair_separable_runner.py
PYTHONWARNINGS=error python -m pytest -q
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q
python experiments/run_abm_baseline.py --config configs/abm_baseline_small.toml
python experiments/run_abm_variance_diagnostic.py --config configs/abm_variance_diagnostic_small.toml
python experiments/run_abm_uncertainty_diagnostic.py --config configs/abm_uncertainty_smoke.toml
python experiments/run_pair_jax_small.py --config configs/pair_jax_small.toml
python experiments/run_velocity_variance_comparison.py --config configs/velocity_variance_comparison_small.toml
python experiments/run_pair_separable_benchmark.py --config configs/pair_separable_benchmark_small.toml

# Later phases:
python experiments/validate_abm_variance.py --config configs/variance_small.toml
python experiments/reproduce_case2_1.py --config configs/case2_1_small.toml
python experiments/benchmark_pair_gpu.py --config configs/case2_1_full_f32.toml
```

GPU and full-grid tests should be opt-in markers so ordinary CPU validation never attempts multi-gigabyte allocations.

## 14. Decisions fixed by this specification

- The primary estimand is `D_j(q,t)`, conditional on the exact focal Q-vector and on selecting action `j`.
- `sigma_j^2` is the conditional variance of one incident-edge payoff; `c_j` is covariance between payoffs from two distinct opponents.
- Pair density supplies `mu_j`, `m2_j`, and `sigma_j^2`, but not `c_j`.
- The pure pair prediction sets `c_j=0` as a labelled conditional-independence closure.
- The first version measures `c_j` from the ABM and does not implement a triplet closure.
- The current shared focal action is not described as the direct source of `c_j` after conditioning; persistent correlated state histories remain a possible source.
- The first estimator uses configurable narrow two-dimensional Q-bins, matched action/Q weighting, bin-refinement checks, and sparse-bin flags.
- Counterfactual edge rewards for both focal actions are diagnostics; direct velocity validation uses the realised selected coordinate.
- The ABM and variance instrumentation precede the full GPU pair-density rewrite.
- ABM Q-updates are continuous; only pair-density transport projects to a Q grid.
- The controlled comparison uses the same seeded discrete Q histogram in both routes and independent half/half initial edge states; continuous Beta ABM initialisation is secondary.
- Independent runs, not agents within one complete network, are the primary uncertainty sampling units.
- The old `case2_1_jax.py` deletion is intentional and committed.

## 15. Remaining decisions before later boundaries

| Decision | Needed by |
| --- | --- |
| Exact seed/reference histogram for legacy regression | Full reproduction |
| Production run/bootstrap counts, bin schedule, anchors, and sparse-stratum policy | Reported variance experiment |
| Boundary policy outside the original parameter range | General model support |
| Production nearest/interpolated transport default | Grid convergence completion |
| GPU model, available memory, precision, and runtime budget | Full-grid GPU run |

These choices must be recorded in configurations/results; none changes the already fixed definition of `D_j(q,t)`.

## 16. Definition of completion for the eventual project

The project is scientifically and numerically complete when:

1. the original mean pair-density trajectory can be reproduced from a recorded initial condition without modifying the source script;
2. shared model and small NumPy reference tests establish the exact chronology and one-edge moment formulas;
3. the ABM passes fixed-action chronology tests and emits validated realised-velocity and O(n) `S1`/`S2` statistics;
4. direct ABM variance and the ABM `sigma^2/c` decomposition agree under exact synthetic conditioning and converge under Q-bin refinement;
5. the NumPy and JAX CPU pair solvers agree on small multi-step arrays and Q-resolved one-edge moments;
6. the four-way comparison separates one-edge pair error, cross-opponent covariance, and finite-bin effects;
7. the GPU pair solver reports validated memory, performance, mass, symmetry, precision, and one-edge-moment behaviour;
8. `n` and `alpha` scaling checks respect the difference between fixed-snapshot rescaling and changed trajectories;
9. every published output is reproducible from configuration, conditioning/weighting metadata, seeds, source hashes, and environment metadata.

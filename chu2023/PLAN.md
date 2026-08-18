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

The original paper and `case2_1.py` remain immutable provenance artifacts. Phase 1 is now implemented; JAX, the ABM, GPU transport, Q-bin estimators, bootstrap intervals, and full variance experiments remain future work.

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
- Validation command: `python -m pytest -q` after installing `.[test]`; the milestone suite reports `33 passed`.
- No scientific deviations from `MODEL_SPEC.md` were introduced. The unknown published histogram was not reconstructed, no full 131-point pair array was allocated, and no JAX or ABM code was started.

## 6. Phase 2 - JAX finite-population ABM

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

- Exact small fixed-action tests pass, and seeded small stochastic runs produce stable trajectories and per-agent realised rewards/velocities before `n=1000` or large replication counts are attempted.

## 7. Phase 3 - ABM variance and covariance instrumentation

Implement the primary estimand before the full pair rewrite.

For each focal agent and counterfactual focal action `j`, stream

```text
S1 = sum_h Y_ih^(j),
S2 = sum_h (Y_ih^(j))^2,
distinct_product = (S1^2-S2)/(N(N-1)),  N=n-1,
```

without enumerating opponent pairs. Compute both counterfactual action diagnostics from current edge states and realised opponent actions, but mark them as counterfactual. Direct realised-velocity samples must be restricted to agents with `A_i=j`.

Tasks:

1. Accumulate per-focal `S1`, `S2`, edge mean, edge second moment, and distinct-edge product for both focal actions.
2. Record selected-coordinate realised velocity and the corresponding realised average reward.
3. Implement configurable narrow two-dimensional Q-bins with selected-action counts, counterfactual counts, Q means/covariance/ranges, and sparse-bin flags.
4. Within finite bins, match the selected-action Q distribution. Counterfactual all-agent diagnostics must be restricted or weighted by `x_j(q)` when compared with selected-action targets.
5. Estimate conditional first and second moments rather than relying only on a black-box sample variance.
6. Report both

   ```text
   Var[alpha(r-Q_j) | Q in B,A=j]
   ```

   and `alpha^2 Var[r | Q in B,A=j]`, plus the exact finite-bin difference from Q dispersion and reward-Q covariance.
7. Compute `sigma_j,ABM^2` and `c_j,ABM` from consistently weighted raw moments.
8. Add run and focal-agent identifiers so uncertainty intervals can respect clustering.
9. Add bin-refinement reports and suppress/flag bins below configurable occupancy thresholds.

Estimator tests:

- `S1^2-S2` equals an explicit ordered-distinct-pair sum on tiny arrays;
- hand-built edge configurations give exact `mu`, `sigma^2`, `c`, and average-reward variance;
- `n=2` omits `c` and uses only the single-edge term;
- synthetic conditionally independent edges recover `c=0` within sampling error;
- synthetic shared-latent configurations recover a known nonzero `c`;
- the raw-moment identity

  ```text
  Var(mean edge payoff)
    = sigma^2/N + (N-1)c/N
  ```

  closes under the same weights;
- a deliberately wide Q-bin demonstrates that raw velocity variance differs from `alpha^2` times reward variance by the recorded Q terms;
- selected-action subsampling and `x_j(q)`-weighted counterfactual diagnostics agree under controlled data;
- sparse-bin suppression and bin-edge conventions are deterministic.

Exit gate:

- On small CPU runs, the reward variance decomposition closes, the direct realised-velocity variance is separately identified, finite-bin bias terms are visible, and no ordered opponent-pair array is constructed.

## 8. Phase 4 - JAX pair solver on CPU

Port the deterministic pair solver component by component and compare with the NumPy oracle before using a GPU.

Tasks:

1. Use a `(state, M, M)` JAX pair layout and conversion helpers.
2. Vectorise marginals, observables, payoff matrix-vector contractions, and velocity fields.
3. Compute `f(q,t)`, `w_sb(q,t)`, `mu_j^pair`, `m2_j^pair`, and `sigma_j,pair^2` for both actions.
4. Implement a first correct small-array transport, even if it uses a simple flat scatter.
5. Implement separable endpoint transport (`A_a P_s A_b^T`) with staged segment/scatter operations.
6. Add optional chunking with a pure functional accumulator and deterministic chunk-order CPU execution.
7. JIT one step, then a short `lax.scan`; keep I/O and metadata writes outside compiled code.

Validation matrix:

| Case | Backend/dtype | Required comparison |
| --- | --- | --- |
| Tiny hand case | NumPy/JAX CPU float64 | exact destinations and edge moments |
| Reduced grid, one step | NumPy/JAX CPU float64 | full-array comparison |
| Reduced grid, many steps | NumPy/JAX CPU float64 | full array, observables, moments, invariants |
| Reduced grid float32 | JAX CPU float32 versus float64 | bounded moment and mass error |
| Chunk sizes 1/many/all | JAX CPU | invariant, trajectory, and moment equivalence |

Exit gate:

- NumPy and JAX CPU agree over multi-step reduced cases, including the Q-resolved one-edge first/second moments needed by the variance project.

## 9. Phase 5 - Initial four-way variance comparison

Run the first scientific comparison at modest CPU-manageable scale before full GPU optimisation.

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
4. **Theory comparison:** compare direct ABM variance with

   ```text
   pure pair: alpha^2 sigma_j,pair^2/(n-1)

   hybrid:    alpha^2 [sigma_j,pair^2/(n-1)
                        + (n-2)c_j,ABM/(n-1)].
   ```

Tasks:

1. Produce named, separate direct/decomposed/pure-pair/hybrid series; never overwrite one with another.
2. For pair values in a bin, calculate both the matched pooled edge variance and the occupancy-weighted mean of local grid variances, and label the distinction.
3. Compare multiple bin widths and minimum-count thresholds.
4. Bootstrap or otherwise construct uncertainty intervals by run and focal-agent clusters, not by treating incident edges as independent observations.
5. Diagnose whether discrepancies arise from the one-edge pair law, nonzero `c_j`, finite-bin effects, or more than one source.

Exit gate:

- The four checks can be interpreted independently, use matched conditioning/weights, and answer whether empirical `c_j` materially explains the pair-closure discrepancy in the small baseline case.

## 10. Phase 6 - Pair transport policy and CPU grid convergence

Keep exact legacy reproduction and improved numerics as distinct modes.

Tasks:

1. Name the original mode `nearest_legacy` and preserve its round-to-grid behaviour.
2. Add conservative linear interpolation only after legacy parity is established.
3. Verify interpolation weights are non-negative, sum to one, and respect boundaries.
4. Run grid-spacing studies on manageable CPU cases, comparing `h`, `h/2` where feasible, nearest versus interpolation, and float32 versus float64.
5. Quantify grid locking and changes in mean trajectories and Q-resolved one-edge moments.
6. Repeat the small four-way variance comparison when numerical policy materially changes `sigma_j,pair^2`.

Exit gate:

- Reports distinguish legacy reproduction error, grid discretisation error, floating-point error, and statistical ABM error. No production default changes merely because an alternative looks smoother.

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

## 13. Proposed test commands once the scaffold exists

These commands are targets for the later implementation; they do not exist yet:

```bash
python -m pytest -q
python -m pytest -q tests/test_model.py tests/test_abm_one_step.py
python -m pytest -q tests/test_velocity_variance.py tests/test_conditioning.py
python -m pytest -q tests/test_pair_one_step.py tests/test_pair_reward_moments.py
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q -m "not gpu and not slow"
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
| Minimum bin count, refinement schedule, and cluster interval method | Reported variance experiment |
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

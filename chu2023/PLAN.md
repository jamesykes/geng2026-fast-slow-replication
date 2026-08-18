# Staged implementation and validation plan

## 1. Objective and non-goals

The eventual project will provide two independent numerical routes for the same model:

1. a deterministic pair-mass solver that can reproduce `case2_1.py` and scale to a GPU; and
2. a finite-population JAX agent-based simulation (ABM) that preserves the paper's one-action-per-agent, complete-graph chronology.

Both routes will feed a common statistics layer for mean trajectories and realised Q-update variance, including a deliberate investigation of covariance between rewards from distinct opponents.

This document is a plan only. The original paper and `case2_1.py` remain immutable provenance artifacts. No numerical rewrite, package scaffold, or tests are created in this phase.

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
│   └── case2_1_full_f32.toml         # resource-gated full-grid configuration
├── src/
│   └── chu_pair/
│       ├── __init__.py
│       ├── config.py                 # typed numerical/experiment configurations
│       ├── model.py                  # actions, states, payoff and transition tensors
│       ├── grids.py                  # Q grids, flattening, rounding/interpolation maps
│       ├── initial_conditions.py     # continuous, histogram, and exact-legacy initialisers
│       ├── observables.py            # shared means, marginals, and invariant diagnostics
│       ├── pair_density/
│       │   ├── __init__.py
│       │   ├── numpy_reference.py    # small, readable CPU oracle
│       │   ├── payoffs.py            # conditional rewards and velocity fields
│       │   ├── transport.py          # nearest/interpolated JAX pushforwards
│       │   ├── solver.py             # stepping, scans, checkpoints, result API
│       │   └── layouts.py            # (state,M,M) conversion and chunk policies
│       ├── abm/
│       │   ├── __init__.py
│       │   ├── complete_graph.py     # edge construction and synchronous step
│       │   ├── simulation.py         # replicated/batched runs and checkpoints
│       │   └── sampling.py           # explicit keyed random initialisation/actions
│       └── statistics/
│           ├── __init__.py
│           ├── velocity.py           # realised/counterfactual Q increments
│           ├── reward_moments.py      # edge variances and covariance decomposition
│           ├── pair_predictions.py    # pair-closure theoretical predictions
│           └── comparisons.py         # errors, intervals, tabular summaries
├── experiments/
│   ├── reproduce_case2_1.py          # exact legacy-compatible mean trajectory
│   ├── pair_grid_convergence.py      # spacing/interpolation/precision study
│   ├── run_abm_baseline.py           # paper-like finite-population runs
│   ├── velocity_variance.py          # theory versus ABM variance experiment
│   └── benchmark_pair_gpu.py         # throughput and peak-memory benchmark
├── tests/
│   ├── conftest.py                   # tiny deterministic model fixtures
│   ├── test_model.py                 # payoff and transition truth tables
│   ├── test_grids.py                 # bounds, flattening, rounding/interpolation
│   ├── test_initial_conditions.py    # mass, moments, seeded repeatability
│   ├── test_pair_one_step.py          # hand-computed and NumPy/JAX parity cases
│   ├── test_pair_invariants.py        # mass, non-negativity, symmetry
│   ├── test_abm_one_step.py           # fixed actions/states exact update
│   ├── test_abm_statistics.py         # seeded Monte Carlo moment checks
│   ├── test_velocity_variance.py      # variance/covariance identities
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
- Keep model semantics separate from numerical policy. In particular, transition rules, nearest-grid rounding, interpolation, array layout, and chunk size are separate configuration choices.
- Use one action per agent per timestep in the ABM and reuse it on every incident edge.
- Read payoff and transition outcomes from old edge states, then synchronously commit Q and edge-state updates.
- Keep a readable NumPy CPU oracle even after the JAX implementation exists.
- Enable JAX 64-bit explicitly in strict CPU parity tests. Treat GPU float32 as a separate validated numerical mode.
- Avoid automatic differentiation; it provides no value here and increases memory pressure.
- Validate scientific quantities and invariants, not only that code executes.

## 4. Phase 0 - Freeze semantics and reference data

Status: the prose specification is complete; executable fixtures remain future work.

Tasks:

1. Preserve the source hashes recorded in `MODEL_SPEC.md`.
2. Decide whether the staged deletion of the old `case2_1_jax.py` is intentional before using any prior port as evidence.
3. Add a tiny deterministic initial pair mass directly as test data; do not depend on random Beta samples for one-step unit tests.
4. Add an exact legacy-initialisation mode using Python's `random.Random(seed)` and the original two-dimensional histogram procedure.
5. Record one or more seeded reference trajectories. Start with a reduced grid that runs comfortably on CPU. A full 131-grid trajectory is a later resource-gated regression, not the first test.
6. Define a result schema containing timestep labels, observables, invariant diagnostics, configuration, seed, dtype/backend, and source hashes.

Exit gate:

- A reviewer can reconstruct the chosen reference initial mass and explain every update from the specification and fixture alone.

## 5. Phase 1 - Shared model definitions and small NumPy oracle

Implement the common model layer and a deliberately simple loop-based NumPy reference solver.

Tasks:

1. Define `Action(C,D)`, `State(SH,PD)`, payoff tensor shape `(state, own_action, opponent_action)`, and transition tensor shape `(old_state, own_action, opponent_action)`.
2. Define an immutable configuration for learning rate, selection intensity, grid bounds/spacing, dtype, timestep count, and transport policy.
3. Implement stable Boltzmann probabilities using `sigmoid(tau*(Q_C-Q_D))`.
4. Implement grid flatten/unflatten maps and an exact compatibility version of `appro()` for the active spacing.
5. Implement conditional payoff/velocity calculation and the four-branch nearest-grid pushforward in clear NumPy.
6. Implement shared observables and invariant checks.

CPU tests first:

- payoff matrices and the eight-entry transition truth table;
- action probabilities sum to one and agree with the original formula;
- hand-built one-cell and two-cell mass transports;
- one-step comparison with an independently hand-calculated result;
- mass conservation, non-negativity, and endpoint-exchange symmetry;
- no movement of an unselected Q-coordinate;
- safe handling of zero focal marginal mass;
- explicit failure or configured handling for out-of-range destinations;
- timestep/output convention, including the original script's `0..200` labels and hidden final advance as a compatibility option.

Exit gate:

- All tiny CPU tests pass in float64, and the oracle produces the expected legacy observables from a seeded reduced-grid case.

## 6. Phase 2 - JAX pair solver on CPU

Port one mathematical component at a time and compare it with the NumPy oracle before using a GPU.

Tasks:

1. Use a `(state, M, M)` JAX pair layout and conversion helpers.
2. Vectorise marginals, observables, payoff matrix-vector contractions, and velocity fields.
3. Implement a first correct transport with small arrays, even if it uses a simple flat scatter.
4. Implement the separable endpoint transport (`A_a P_s A_b^T`) using staged segment/scatter operations.
5. Add optional chunking with a pure functional accumulator and deterministic chunk-order CPU execution.
6. JIT one step, then a short `lax.scan`; keep I/O and metadata writes outside compiled code.

Validation matrix:

| Case | Backend/dtype | Required comparison |
| --- | --- | --- |
| Tiny hand case | NumPy/JAX CPU float64 | exact destinations; near-machine-precision masses |
| Reduced grid, one step | NumPy/JAX CPU float64 | full-array comparison |
| Reduced grid, many steps | NumPy/JAX CPU float64 | full array plus all observables/invariants |
| Reduced grid float32 | JAX CPU float32 versus float64 | bounded moment and mass error |
| Chunk sizes 1/many/all | JAX CPU | invariant and observable equivalence |

Exit gate:

- NumPy and JAX CPU agree over multi-step reduced cases within declared tolerances, and chunk size does not change conclusions.

## 7. Phase 3 - Pair transport numerical policy and grid convergence

Keep exact reproduction and improved numerics as distinct modes.

Tasks:

1. Name the original mode `nearest_legacy` and preserve its round-to-grid behaviour.
2. Add conservative linear interpolation only after legacy parity is established.
3. Verify interpolation weights are non-negative, sum to one, and respect boundaries.
4. Run grid-spacing studies on CPU for manageable grids, then GPU: compare `h`, `h/2` where feasible, nearest versus interpolation, and float32 versus float64.
5. Quantify grid locking and changes in Q/state/strategy trajectories.

Exit gate:

- Reports clearly distinguish legacy reproduction error, grid discretisation error, and floating-point error. No production default changes solely because an alternative looks smoother.

## 8. Phase 4 - GPU scaling and memory engineering

Only begin after CPU semantic parity.

Tasks:

1. Benchmark vectorised payoff contractions separately from transport.
2. Measure compiled peak memory for current density, destination density, endpoint maps, and each transport temporary.
3. Sweep endpoint chunk sizes and compare flat scatter, staged segment-sum, and any viable sparse representation.
4. Avoid full `(branch, source_cell)` index and weight arrays. Generate branch data per chunk from length-`M` endpoint maps.
5. Test JAX buffer donation and verify whether it reduces actual peak memory.
6. Record compile time, step time, throughput in source cells/s, device model, JAX/XLA versions, precision, and peak bytes.
7. Add a preflight estimator that rejects configurations whose conservative peak estimate exceeds a configured device budget.
8. Check GPU run-to-run variation caused by scatter atomic order. Compare invariant and trajectory tolerances, not bitwise output.

Scaling gates:

- reduced cases reproduce CPU results;
- no out-of-memory failure at the selected production configuration;
- mass, non-negativity tolerance, and symmetry diagnostics stay within limits for the full trajectory;
- full 131-grid runtime and peak memory are reported before larger/finer grids are attempted.

## 9. Phase 5 - JAX finite-population ABM

Build this independently against shared model definitions, not by translating pair-solver transport.

Tasks:

1. Construct upper-triangle edge endpoint arrays for the complete graph.
2. Initialise agent Q-values and edge states with explicit keys. Support both continuous paper-like Beta initialisation and grid/histogram-matched initialisation.
3. Draw exactly one action per agent each timestep.
4. Gather endpoint actions and old edge states; compute both endpoint payoffs on every edge.
5. Aggregate rewards with segment sums, divide by `n-1`, and update only selected Q-coordinates.
6. Transition every edge from its old state and the same endpoint actions.
7. Support batched replications with controlled memory, chunking either edges or replications when needed.
8. Record both pre-update observables at label `t` and post-update state at `t+1` unambiguously.

Deterministic ABM tests:

- `n=2` with fixed Q/actions/state, checked by hand;
- `n=3` with fixed actions and three distinct edge states, verifying endpoint payoff orientation and averaging;
- same agent action appears on all of its incident edges;
- Q updates use old-state rewards and only the selected coordinate;
- transitions use old state plus the same realised actions;
- fixed seeds reproduce actions and trajectories on the same backend.

Statistical ABM tests:

- empirical action frequencies match Boltzmann probabilities;
- empirical initial Beta moments and state proportions match targets;
- increasing independent repetitions shrinks Monte Carlo error at the expected rate;
- for large `n`/many repetitions, mean trajectories approach pair-solver predictions within predeclared uncertainty bands.

Exit gate:

- Exact small fixed-action tests pass, and seeded small stochastic runs produce stable, interpretable diagnostics before `n=1000` or 500-repetition experiments are attempted.

## 10. Phase 6 - Realised velocity variance and opponent covariance

First fix the estimand in configuration and result metadata.

Recommended primary estimand:

```text
Var[alpha (mean_h Y_ih(a) - Q_i(a)) | time, focal-Q bin, A_i=a]
```

Also report the unconditional selected-coordinate increment separately, because focal action mixing adds another variance term.

Tasks:

1. Instrument the ABM to retain or stream sufficient per-edge contributions for selected focal agents without saving every edge at every timestep.
2. For `K=n-1`, estimate:

   ```text
   sigma^2 = Var(Y_ih),
   c       = Cov(Y_ih, Y_il), h != l,
   Var(mean payoff) = sigma^2/K + (K-1)c/K.
   ```

3. Validate the estimator on synthetic independent-edge data (`c=0`) and synthetic shared-random-effect data with known nonzero covariance.
4. Decompose covariance with the law of total covariance, conditioning successively on focal Q, focal action, and focal incident-state composition.
5. Extend pair calculations to produce single-edge payoff first and second moments.
6. Label the pair-only prediction `c=0` as a conditional-independence closure, not a consequence of pair data.
7. Compare three predictions:

   - independent-edge pair closure;
   - pair variance plus ABM-measured covariance correction;
   - a triplet-closure prediction, only if a justified triplet model is later developed.

8. Use bootstrap or across-run uncertainty intervals that respect clustering by focal agent/run; do not treat all incident edges as independent samples.

Exit gate:

- The variance identity closes numerically in the ABM, the conditioning is explicit, and every theoretical curve states where its covariance term came from.

## 11. Phase 7 - Reproduction and theory-versus-simulation experiments

Run experiments in increasing cost order:

1. Tiny exact cases.
2. Reduced grids and small populations (`n=2,3,10`) on CPU.
3. Short medium cases on CPU and GPU for cross-backend validation.
4. Figure 1(b)-like pair trajectory with recorded seed/histogram.
5. ABM `n=1000` with a small number of repetitions as a smoke test.
6. Replication-count convergence, then the paper-like 500 repetitions if resources justify it.
7. Velocity-variance sweeps over time, population size, learning rate, and selection intensity.
8. Grid/precision/transport-policy sensitivity studies.

For every comparison, save:

- mean Q, mean action probabilities, and state proportions;
- pair-solver invariants;
- ABM across-run standard deviation and confidence intervals;
- realised Q-increment variance under the declared conditioning;
- single-edge payoff variance and distinct-edge covariance;
- absolute/relative errors between theoretical predictions and simulation estimates;
- configuration, seed sequence, source hashes, backend/device, dtype, timings, and package versions.

Do not tune the theory on the same runs used for final validation. If an empirical covariance correction is fitted, reserve independent runs for evaluation.

## 12. Proposed test commands once the scaffold exists

These commands are targets for the later implementation; they do not exist yet:

```bash
python -m pytest -q
python -m pytest -q tests/test_pair_one_step.py tests/test_pair_invariants.py
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q -m "not gpu and not slow"
python experiments/reproduce_case2_1.py --config configs/case2_1_small.toml
python experiments/benchmark_pair_gpu.py --config configs/case2_1_full_f32.toml
```

GPU and full-grid tests should be opt-in markers so ordinary CPU validation never attempts multi-gigabyte allocations.

## 13. Decisions required before implementation reaches each boundary

The following do not block Phase 1 small CPU work, but must be resolved before the named later phases:

| Decision | Needed by |
| --- | --- |
| Exact seed/reference histogram for legacy regression | Full reproduction |
| Initial state sampling probability/independence in ABM | ABM baseline |
| Primary conditioning for "realised velocity" variance | Statistics API |
| Whether covariance is empirical-only or needs a triplet theory | Variance comparison |
| Continuous-Beta versus grid-matched ABM comparison protocol | Theory-versus-ABM experiments |
| Boundary policy outside the original parameter range | General model support |
| Production nearest/interpolated transport default | Grid convergence completion |
| GPU model, available memory, precision, and runtime budget | Full-grid GPU run |
| Intent of the staged deletion of the previous JAX file | Before consulting/reusing prior work |

## 14. Definition of completion for the eventual project

The project is scientifically and numerically complete when:

1. the original mean pair-density trajectory can be reproduced from a recorded initial condition without modifying the original script;
2. the NumPy oracle and JAX CPU solver agree on small multi-step cases;
3. the GPU pair solver reports validated memory, performance, mass, non-negativity, symmetry, and precision behaviour;
4. the ABM passes exact fixed-action chronology tests and reproduces pair-level mean trends within finite-population uncertainty;
5. realised Q-velocity variance is precisely defined and its single-edge variance/cross-edge covariance decomposition closes in simulation;
6. theoretical variance predictions are compared with held-out simulation results with uncertainty and closure assumptions made explicit;
7. every published output is reproducible from a configuration and recorded seed/source/environment metadata.

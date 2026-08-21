# Claude Code handover

## 1. Purpose of this handover

This document gives a new coding agent enough context to work safely without
reconstructing the project's history from chat. It summarizes the scientific
target, implemented milestones, validation boundary, and the first authorized
Lambda-host task. It does not replace the authoritative documents listed in
[Section 12](#12-authoritative-document-map).

The immediate operational objective is deliberately small: on the already
live Lambda instance reported by the human operator, preview and create the
documented isolated environment, run the GPU doctor, report the evidence, and
stop. No numerical GPU stage is authorized in that first session.

The statement that an instance is live and the repository has been cloned is
operator-provided external context. The committed repository cannot validate
that host, its GPU, its checkout, or its capacity. The doctor is the first
evidence-producing step.

## 2. Research aim

The repository studies the stochastic Q-learning model in Chu et al. using two
independent numerical routes:

1. a deterministic pair-probability-mass evolution; and
2. a finite-population JAX agent-based model (ABM).

The central scientific objective is to understand the local one-step
conditional variance of the realized chosen-coordinate Q-learning velocity and
to determine whether discrepancies between pair closure and the ABM arise from
the one-edge law, covariance between distinct opponents, finite-Q-bin mixing,
or a combination of them.

The source paper and `case2_1.py` are immutable provenance artifacts. The
shared payoff and transition tensors in `src/chu_pair/model.py` are the single
implementation authority. Exact model conventions, chronology, array axes,
and equations are in `MODEL_SPEC.md`.

## 3. Essential scientific vocabulary

### 3.1 Chosen-coordinate velocity

For focal agent `i`, action coordinate `j`, and source time `t`,

```text
v_j^i = 1{A_t^i=j} alpha [r_t^i(j)-Q_t^i(j)].
```

The primary exact-Q estimand conditions on both the complete focal Q-vector and
the action actually selected:

```text
D_j(q,t) = Var[v_j^i | Q_t^i=q, A_t^i=j]
         = alpha^2 Var[r_t^i(j) | Q_t^i=q, A_t^i=j].
```

It is not an unconditional coordinate-update variance and not covariance
between the C and D velocity coordinates. Direct ABM validation includes only
the coordinate actually selected and updated.

### 3.2 Single-opponent variance

Let `N=n-1` and let `Y_ih^(j)` be the payoff from one opponent `h` when the
focal action is fixed to `j`. Then

```text
sigma_j^2(q,t) = Var[Y_ih^(j) | Q_t^i=q, A_t^i=j].
```

This is a one-edge quantity. The pair mass determines its conditional one-edge
law and therefore its mean, second moment, and variance.

### 3.3 Cross-opponent covariance

For two distinct opponents `h != k`,

```text
c_j(q,t) = Cov[Y_ih^(j),Y_ik^(j) | Q_t^i=q,A_t^i=j].
```

This is covariance between two incident-edge payoffs sharing a focal agent. It
is not C/D coordinate covariance and is not merely multinomial-count
covariance. The current focal action is fixed by the conditioning, so do not
describe it as the direct contemporaneous source of `c_j`. Persistent shared
history and latent incident-edge configurations can still induce dependence.

The exact reward/velocity decomposition is

```text
D_j(q,t) = alpha^2 [sigma_j^2(q,t)/N + (N-1)c_j(q,t)/N].
```

The one-edge pair density cannot identify `c_j`; the current implementation
measures it from the ABM and does not implement a triplet closure.

### 3.4 Finite-bin effects

The ABM has continuous Q-values, so exact conditioning is estimated with
two-dimensional Q bins. Within a finite bin `B`, Q is not constant:

```text
Var[alpha(r-Q_j) | B,A=j]
  = alpha^2 [Var(r | B,A=j)
             + Var(Q_j | B,A=j)
             - 2 Cov(r,Q_j | B,A=j)].
```

Consequently, `alpha^2 Var(r)` is not generally the finite-bin velocity
variance. Action selection also changes the Q weighting within a bin.

Pair quantities are pooled with selected mass proportional to
`p_t(q) pi_j(q)` before nonlinear moments are formed. Even under exact-Q
conditional independence, the finite-bin pair covariance

```text
c_B,pair = E_B[mu(q,j)^2] - E_B[mu(q,j)]^2
```

need not be zero. Never force exact-Q closure identities onto pooled bins.

### 3.5 The four Phase-5 estimands

Keep these four named quantities distinct:

1. **Direct ABM:** the variance of observed selected-coordinate velocities.
2. **Reconstructed ABM:** the ABM single-edge variance and ABM distinct-edge
   covariance decomposition, plus the finite-bin Q and reward-Q terms.
3. **Pure pair:** the full finite-bin pair raw-moment closure, including its
   finite-bin mixing covariance and Q terms.
4. **Hybrid diagnostic:** the pair single-edge and Q terms, replacing only the
   pair finite-bin distinct-opponent covariance with the matched ABM covariance.

The hybrid is a discrepancy diagnostic, not an analytical pair prediction of
cross-opponent covariance. Pair-only quantities are deterministic; ABM runs are
the uncertainty clusters.

## 4. Model chronology that must not drift

The ABM and pair routes represent the same source-time chronology:

1. evaluate policies from `Q_t`;
2. draw one action per agent and reuse it on every incident edge;
3. compute payoffs using old edge states `S_t`;
4. average ABM rewards over exactly `n-1` opponents;
5. update only the selected Q coordinate;
6. transition edges from `S_t` using those same actions; and
7. synchronously return `Q_(t+1),S_(t+1)`.

ABM Q-updates remain continuous. Nearest-grid projection belongs only to pair
transport. Pair entries are probability masses, not continuous density values,
and sums are not multiplied by grid spacing. Actions are ordered `(C,D)`,
states `(SH,PD)`, and canonical pair axes are
`(q1_C,q1_D,state,q2_C,q2_D)`.

## 5. Implemented milestone map

The Git history and `PLAN.md` give the definitive status. At the current
handover commit, the implemented path is:

| Milestone | Implemented capability | Validation boundary |
| --- | --- | --- |
| Shared model and Phase 1 | Authoritative tensors, grid/projection rules, seeded legacy histogram, observables, and a readable NumPy pair oracle | Small CPU exact and regression cases |
| Phase 2 JAX ABM | Packed complete graph, one action per agent, continuous selected-Q update, keyed initialization and scan/vmap execution | CPU float32 and CPU+x64 |
| Phase 3A instrumentation | Selected-action `S1`, `S2`, velocity and two-dimensional Q-bin sufficient moments without edge histories | CPU diagnostics and estimator identities |
| Phase 3B uncertainty | Complete-independent-run bootstrap, common weights, nested-bin refinement, anchors and reconstruction checks | Bounded CPU smoke cases |
| Phase 4 flat pair solver | Exact legacy nearest-grid JAX transport, conditional one-edge moments, lean `lax.scan`, fail-closed resource analysis | Reduced CPU grids; flat scatter oracle |
| Phase 5 comparison | Source-time/bin/action-matched direct, reconstructed, pure-pair and hybrid estimands | Bounded CPU scientific pipeline, not production inference |
| Phase 6 separable transport | Exact tiled row/column pushforward, device-side pair initialization, analyzed runtime bundles, bounded CPU benchmark | `G<=17` ordinary CPU validation; allocation-free `G=131` projection only |
| Phase 7 pilot workflow | Isolated CUDA environment setup, allocator bootstrap, GPU doctor, stable identity/capacity admission, atomic stage ladder and timeouts | CPU-unavailable and mocked-device tests only |

Phase 6 remains incomplete with respect to interpolation and grid-convergence
studies. Phase 7 is operationally prepared but has no committed real-GPU or
full-grid execution result.

## 6. What CPU evidence establishes—and what it does not

The committed documentation records passing default/warnings-as-errors and
fresh CPU+x64 suites for the implemented milestones using JAX/JAXLIB 0.7.2.
It also records reduced-grid parity among NumPy, flat JAX, and separable JAX
paths, bounded smoke runners, mocked GPU identity/capacity cases, and dry runs.
Check the current `README.md`, `PLAN.md`, Git state, and test output before
repeating any pass count as current fact.

That evidence establishes implementation behavior on CPU and the operation of
mocked/fail-closed safeguards. It does not establish:

- successful execution on an NVIDIA GPU;
- CUDA installation compatibility on the live host;
- the identity or usable capacity of the live GPU;
- GPU numerical tolerances, performance, or memory peaks;
- full-grid compilation or execution; or
- any `G=131` scientific result.

Allocation-free byte projections and small-CPU compiled-memory ratios are
planning evidence only. They are not substitutes for fresh analysis of the
exact retained GPU executable or stable-device-matched capacity evidence.

## 7. Live Lambda boundary and first-session checklist

### 7.1 Authorization boundary

The human operator reports that a Lambda instance is live and this repository
has been cloned there. The next authorized work is only:

1. inspect the checkout and documented host prerequisites;
2. preview and create the isolated project environment as specified;
3. run the GPU doctor with the reviewed allocator policy; and
4. report the result and stop.

Do not start the pilot runner's stage-chain root, small stage, or any later
stage in the same session without new explicit human approval.

### 7.2 Checklist

On the Lambda host:

1. Confirm the current directory is the intended `chu2023` checkout. Run
   `git status --short` and `git rev-parse HEAD`; report any dirty or unexpected
   state before proceeding.
2. Read `AGENTS.md`, `MODEL_SPEC.md`, `PLAN.md`, `README.md`, and
   `docs/LAMBDA_GPU_PILOT.md` from that checkout. Do not rely on copied chat
   commands when the repository guide differs.
3. Confirm with the human that the intended CUDA family and allocator policy
   are still `cuda12`, `fraction`, and `0.85`, or use the explicitly reviewed
   alternatives from the runbook. Do not guess from the GPU name.
4. Perform only the bounded stack inspection listed in
   `docs/LAMBDA_GPU_PILOT.md`. If driver, architecture, checkout, or package
   evidence is inconsistent, stop and report it.
5. Preview environment creation:

   ```bash
   python scripts/prepare_gpu_environment.py --cuda-family cuda12 --dry-run
   ```

   Confirm that preview succeeds and does not install anything.
6. Create and activate the project-local environment exactly as documented:

   ```bash
   python scripts/prepare_gpu_environment.py --cuda-family cuda12
   source .venv-gpu/bin/activate
   ```

   Stop if the tool refuses the destination, installation fails, or the system
   Python/driver would need modification. Do not bypass its checks.
7. Run only the doctor:

   ```bash
   python experiments/run_gpu_doctor.py \
     --cuda-family cuda12 \
     --allocator-policy fraction \
     --memory-fraction 0.85 \
     --expect-gpu \
     --output outputs/gpu_pilot/doctor.json
   ```

8. Record the command exit status. Review the bounded doctor artifact for
   status, exact package versions, backend, visible device, stable identity,
   driver compatibility, allocator policy, Git cleanliness/source hashes,
   host memory, and capacity evidence. Do not paste secrets or unrelated
   environment variables into the report.
9. Report what was observed, what failed or remains uncertain, the artifact
   path relative to the repository, and the exact commands/results.
10. **Stop.** Await explicit human approval before creating the stage-chain
    doctor artifact or running any numerical stage.

The operator—not repository code—remains responsible for monitoring billing,
saving required artifacts, and manually terminating the instance. A process
exit, timeout, closed SSH session, or stopped Python command does not terminate
the instance or stop billing.

## 8. Pilot stage ladder and human gates

After a successful doctor report and only with separate human authorization,
the documented order is:

1. **Doctor stage-chain root:** bind the clean doctor report to an atomic stage
   artifact.
2. **Small:** `G=3,5,9`, including flat/separable parity and timing.
3. **Medium:** `G=17,33`, separable only.
4. **Large:** `G=65`, separable only; optional `G=97` has a separate opt-in.
5. **Disabled full-grid analysis:** exact `G=131` one-step executable analysis
   only, with its exact confirmation phrase and successful large prerequisite.
6. **Disabled full-grid one-step:** at most one exact `G=131` step, with a
   different exact phrase and a fresh matching analysis prerequisite.

Stop after every stage. Give the human its status, provenance, timings, memory
analysis, capacity evidence, diagnostics, estimated cost, and uncertainty.
Advance only after explicit review and authorization of the next named stage.
Success at one rung is not permission for the next; a one-step result is never
permission for a multi-step full-grid run.

The optional `G=97` and both `G=131` templates are disabled deliberately. Do
not rename, edit, copy around, or weaken their enablement and confirmation
requirements. Exact phrases and detailed commands live only in the deployment
guide and should be copied from the checked-out file at execution time.

## 9. Strict resource and executable contract

For every numerical pilot executable, preserve this fail-closed order:

1. normalize and validate the bounded configuration;
2. perform allocation-free resource estimates;
3. validate the clean doctor and predecessor artifacts;
4. build only host descriptions and abstract shapes;
5. lower and compile the exact executable;
6. validate the retained callable's complete live `memory_analysis()` and
   invocation identity;
7. obtain fresh capacity evidence matched to the same physical device;
8. construct device inputs only after admission;
9. recheck capacity adjacent to invocation;
10. invoke the exact retained callable, synchronize, and validate diagnostics.

Unavailable, incomplete, stale, inconsistent, ambiguous, or mismatched evidence
is a failure, not permission to continue. `--allow-expensive` overrides only
specified bounded-development resource limits. It cannot override scientific
validity, provenance, prerequisite order, executable identity, incomplete
analysis, backend/device matching, capacity, confirmation phrases, or the
full-grid one-step limit.

Allocator policy must be applied before JAX import and is part of executable
environment identity. Device identity must use stable UUID/MIG/PCI evidence or
the trusted CUDA Driver API mapping described in the runbook; never equate a
numeric visibility token with an `nvidia-smi` physical index. MIG capacity
fails closed until slice-level evidence can be established.

Capacity evidence is short-lived and current free memory is not guaranteed
future availability. The exact immutable limits and expiry rules live in
`AGENTS.md`, `MODEL_SPEC.md`, and the runner. Never extend them in a handoff
document or command-line convenience wrapper.

## 10. Working style for the next agent

- Inspect first: Git status, exact commit, scoped instructions, configuration,
  predecessor artifacts, and relevant tests before acting.
- Make the smallest justified change. Avoid redesigning established kernels,
  resource accounting, artifact contracts, or statistics without a separately
  reviewed task.
- Use repository commands exactly. Do not transcribe secrets, private host
  details, broad environment dumps, or credential-bearing URLs.
- Treat scientific, numerical, operational, and hardware claims separately.
  A compiled path, a dry run, and a real execution are different evidence.
- Preserve original sources and generated artifacts. Do not hand-edit signed or
  digested prerequisite metadata; create a new valid stage through the runner.
- Before a claim, run focused validation appropriate to the scope and report
  commands, exit statuses, pass/skip counts, artifacts, and remaining gaps.
- Stop on failure. Do not use an override to convert missing proof into a pass.
- Do not commit, push, contact a cloud API, or operate instance lifecycle unless
  the human explicitly authorizes that exact action.

## 11. Do not do yet

Without a new, explicit, bounded task, do not:

- lower, compile, allocate, or execute `G=131`;
- run any full-grid multi-step experiment;
- run unguarded or automatically chained costly GPU stages;
- infer GPU performance or capacity from CPU projections;
- begin interpolation or grid-convergence studies;
- derive a new triplet or other analytical covariance closure;
- add counterfactual unselected-action instrumentation;
- make broad architecture, packaging, JAX, statistics, or resource rewrites;
- start a new scientific branch such as long-time diffusion, Fokker–Planck
  approximations, unconditional velocity variance, or C/D velocity covariance;
- alter the original paper, `case2_1.py`, authoritative tensors, stage artifacts,
  or disabled full-grid templates; or
- mix unrelated parent-repository work into this subproject.

## 12. Authoritative document map

- `AGENTS.md`: binding local scope, scientific invariants, allocation limits,
  GPU restrictions, and current validation commands.
- `MODEL_SPEC.md`: authoritative source chronology, notation, primary variance
  definitions, pair/ABM semantics, finite-bin convention, resource models,
  separable-kernel contract, and GPU operational contract.
- `PLAN.md`: milestone status, tests and exit gates, implemented/deferred scope,
  stage plan, and remaining scientific decisions.
- `README.md`: current package usage, CPU validation record, runner summaries,
  memory formulas, and the guarded-pilot overview.
- `docs/LAMBDA_GPU_PILOT.md`: complete operator commands, environment isolation,
  allocator policies, GPU doctor, stage prerequisites, exact confirmations,
  timeout/OOM handling, artifact retrieval, cost and manual termination duties.
- `src/chu_pair/model.py`: the single authoritative implementation of action
  order, state order, payoff tensor, and transition tensor.
- `case2_1.py` and `pair-approx_multi-agent_stochastic_games.pdf`: immutable
  provenance sources; inspect when required, never edit.
- `configs/`: checked-in bounded smoke/pilot templates. Review and normalize;
  never weaken fixed safeguards or enable disabled work implicitly.
- `tests/`: executable evidence for chronology, moments, binning, pair parity,
  compiled identity, resource admission, device matching, and stage transitions.

When any status, count, command, or limit in this handover appears inconsistent
with those files or current artifacts, verify the repository and stop for human
direction. The authoritative source wins; this handover should be corrected in
a narrow documentation task rather than used to rationalize a mismatch.

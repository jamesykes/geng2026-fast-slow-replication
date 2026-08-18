# Chu et al. pair-approximation model specification

## 1. Scope and source provenance

This document records the behaviour of the original sources currently present in this subproject. It is a specification for later work, not a description of a rewrite.

Sources inspected in full:

- `case2_1.py` (296 lines; SHA-256 `ebeb0d64e57259e940a49437e6ec6a6a3be636dd99b124f3e8858e1dac9db68f`)
- `pair-approx_multi-agent_stochastic_games.pdf` (the eight-page AAAI-23 paper *A Pair-Approximation Method for Modelling the Dynamics of Multi-Agent Stochastic Games*; SHA-256 `6f25ad7c8ea088666ed1e7b8aa6b0f2a100d0817d145a07b4154e0637502eab0`)

The parent repository's `README.md` says that `chu2023/` contains the Chu et al. code and reference paper, separate from later replication work. No local `AGENTS.md` was present when this specification was written.

The Git worktree already contained unrelated changes before this work began: a modified parent `README.md`, an untracked output directory under `../replication/`, and staged deletions of `Dynamics_of_Q-Learning_in_Networked_Stochastic_Games.pdf` and `case2_1_jax.py` in this directory. Those files and changes were not restored, modified, or used as sources here.

## 2. Model targeted by the script

The script is the deterministic pair-distribution calculation corresponding to the heterogeneous-initial-condition SH/PD experiment in Figure 1(b) of the paper.

- Agents have two actions, cooperate (`C`) and defect (`D`).
- Every pair/edge has one of two states. State `0` is the stag-hunt game `s1`; state `1` is the prisoner's-dilemma game `s2`.
- An agent draws one action at the start of a timestep and uses that same action against every opponent, as specified by the paper.
- Rewards are averaged across all opponents. In the infinite-population pair approximation, this becomes a conditional expected reward determined by the focal agent's Q-values.
- Only the Q-value of the selected action is updated.
- The action pair also changes each edge's state at the end of the timestep.

The active constants are:

| Code | Value | Meaning |
| --- | ---: | --- |
| `Tau` | `2` | Boltzmann selection intensity/inverse-temperature parameter, paper `tau` |
| `Eta` | `0.4` | Q-learning rate, paper `alpha` |
| `b` | `1.2` | Temptation payoff in the PD state |
| `r` | `0.1` | SH non-cooperation payoff and magnitude of the PD sucker payoff |
| `space` | `0.01` | Q-grid spacing `h` |
| `t_max` | `200` | Last labelled timestep written; labels run from 0 through 200 inclusive |
| `game_num` | `2` | Number of edge states |

The active row-player payoff matrices are represented by `u` in flattened joint-action order `(CC, CD, DC, DD)`:

```text
u[0] = (1,  0,   0.1, 0.1)   # SH, s1
u[1] = (1, -0.1, 1.2, 0  )   # PD, s2
```

Equivalently,

```text
M_0 = [[1,  0  ],       M_1 = [[1,   -0.1],
       [0.1, 0.1]]              [1.2,  0  ]].
```

`r_p` and `r_d` duplicate these flattened matrices but are not used. `u` is the active payoff data.

## 3. Grid construction and quantisation

`possible_reward = [-0.1, 0, 0.1, 1, 1.2]` supplies the common lower and upper bounds for both Q-values. The grid is

```text
q_l = -0.1 + 0.01 l,   l = 0, ..., 130.
```

Thus:

- `Qc_num = Qd_num = round((1.2 - (-0.1))/0.01) + 1 = 131`;
- both `Q(C)` and `Q(D)` use the same 131-point inclusive grid;
- one agent-Q grid has `M = 131^2 = 17,161` points.

The commented-out code shows that separate action-specific bounds were considered earlier, but the active code uses the global payoff range for both coordinates.

`appro(number)` is intended to find the nearest grid point. For the active `space = 0.01`, it first applies `numpy.around(number, 2)`. Its subsequent divisibility test is modulo `1`, so it succeeds immediately; in this configuration the function is effectively round-to-two-decimal quantisation. Exact half-grid cases inherit NumPy's rounding behaviour. Continuous Beta draws make exact ties probability zero, but later deterministic updates can in principle land on a tie.

No clipping or explicit boundary check is performed. With the supplied payoff bounds, `0 <= Eta <= 1`, initial Q-values inside the bounds, and payoffs inside the same bounds, the unquantised Q update is a convex combination and remains in range. A generalised model must define a boundary policy rather than rely on that invariant.

## 4. Arrays and axes

### 4.1 Main pair mass

The central array is:

```text
p.shape == (131, 131, 2, 131, 131)
```

The axes mean:

| Axis | Index in code | Meaning |
| ---: | --- | --- |
| 0 | `i` | First/focal agent's `Q(C)` grid index |
| 1 | `j` | First/focal agent's `Q(D)` grid index |
| 2 | `g` | Pair/edge state: `0 = s1 = SH`, `1 = s2 = PD` |
| 3 | `k` | Second/opponent agent's `Q(C)` grid index |
| 4 | `m` | Second/opponent agent's `Q(D)` grid index |

Mathematically, `p[i,j,g,k,m]` is a grid-cell **probability mass**, not a pointwise probability-density value:

```text
p_t(q1, s_g, q2) = Pr(Q_first = q1, state = s_g, Q_second = q2).
```

No powers of the grid spacing appear in sums. The total is intended to be one. The pair can be viewed more compactly as an `(M, 2, M)` array with `M = 17,161`; a future GPU layout will probably prefer `(2, M, M)` so each state slice is contiguous.

The paper labels the endpoints `Q_t^1` and `Q_t^2`, calling them focal agent and opponent. Although physical edges are undirected, the stored mathematical pair is ordered. For symmetric initial data and symmetric games/transitions, exchanging the two endpoints leaves the distribution unchanged.

### 4.2 Other arrays and containers

| Name | Shape/type | Meaning |
| --- | --- | --- |
| `possible_reward` | Python list, length 5 | Values used only to obtain the common Q bounds |
| `r_p`, `r_d` | Python lists, length 4 | Unused flattened payoff matrices |
| `u` | Python nested list `(2, 4)` | Active payoff by state and joint action `(CC, CD, DC, DD)` |
| `P` in `init_p()` | dict keyed by `(qc, qd)`, 17,161 keys | Empirical initial joint mass for one agent's two Q-values |
| `p_init` | `(131, 131, 2, 131, 131)` | Initial pair mass, returned as `p` |
| `q_c` | `(131, 131)` | Conditional velocity/increment `v_C(q)`, despite the name suggesting a Q-value |
| `q_d` | `(131, 131)` | Conditional velocity/increment `v_D(q)` |
| `temp_p` | same as `p` | Zeroed destination mass for the next timestep |
| `ave_qc`, `ave_qd` | scalars | Mean focal-agent Q-values |
| `ave_xc`, `ave_xd` | scalars | Mean focal-agent Boltzmann action probabilities |
| `ave_fp` | scalar | Proportion of edges in state `0` (`s1`, SH; called the prosperous state in a comment) |
| `T` | integer scalar | Discrete timestep label |

NumPy creates `p`, `temp_p`, `q_c`, and `q_d` as `float64` because no dtype is supplied.

## 5. Initial distribution

`init_p()` approximates the heterogeneous initial condition in Figure 1(b).

1. It creates the complete two-dimensional Q-grid and a zero count for each `(Q(C), Q(D))` cell.
2. It makes `sample_n = 131^2 * 10 = 171,610` independent calls to each of:

   ```text
   Z_C ~ Beta(20, 80)
   Z_D ~ Beta(80, 20)
   Q(C) = -0.1 + 1.3 Z_C
   Q(D) = -0.1 + 1.3 Z_D.
   ```

3. Each sampled pair is rounded to the nearest Q-grid cell and accumulated in the two-dimensional empirical histogram `P`.
4. Counts are divided by `sample_n`, making `P` sum to one.
5. The ordered endpoints are made independent and identically distributed from that empirical joint histogram, and state is made independent and uniform:

   ```text
   p_0(q1, g, q2) = P(q1) P(q2) / 2,   g in {0, 1}.
   ```

The two coordinates are sampled independently in the underlying continuous construction, but `P` is one finite two-dimensional empirical histogram, so it can contain a small accidental empirical correlation between `Q(C)` and `Q(D)`. The two endpoints are exactly independent conditional on the realised `P` because their pair mass is its outer product.

The standard-library `random` module is used without a seed. Therefore, the evolution after initialisation is deterministic conditional on `p_0`, but running the original script again does not reproduce exactly the same `p_0` or time series. The original calculation should be described as a deterministic mass transport with a stochastic, unrecorded initial histogram.

The theoretical continuous means before quantisation are `E[Q(C)] = 0.16` and `E[Q(D)] = 0.94`.

## 6. Boltzmann action probabilities

For an agent at grid point `q = (q_C, q_D)`, the script computes

```text
x_C(q) = exp(Tau q_C) / (exp(Tau q_C) + exp(Tau q_D)),
x_D(q) = exp(Tau q_D) / (exp(Tau q_C) + exp(Tau q_D)).
```

This is paper Equation (2) with `Tau = tau`. It is also

```text
x_C(q) = sigmoid(Tau (q_C - q_D)),   x_D(q) = 1 - x_C(q).
```

The latter form is preferable in a rewrite because it is numerically stable. The original direct exponentials are safe for the supplied small Q-range, but are not robust to arbitrary bounds or large `Tau`.

The paper calls `tau` a temperature, but both the equation and script use it as a selection intensity or inverse temperature: increasing it makes action selection more greedy, while `tau = 0` gives a uniform distribution.

## 7. Conditional payoffs and Q-learning velocities

For focal grid point `(i,j)`, the code first forms its marginal pair mass

```text
m_t(i,j) = sum_{g,k,m} p_t[i,j,g,k,m].
```

When this is positive, `p_t[i,j,g,k,m] / m_t(i,j)` is the conditional joint distribution of current edge state and opponent Q-values given the focal Q-values.

For a hypothetical focal action `a in {C,D}`, let

```text
r_g(a | q2) = x_C(q2) M_g[a,C] + x_D(q2) M_g[a,D].
```

The expected payoff used by `delta()` is

```text
Rbar_a(i,j) =
    sum_{g,k,m} p_t[i,j,g,k,m] r_g(a | q_{k,m}) / m_t(i,j).
```

The two stored velocities are then

```text
q_c[i,j] = v_C(q_{i,j}) = Eta (Rbar_C(i,j) - q_C(i)),
q_d[i,j] = v_D(q_{i,j}) = Eta (Rbar_D(i,j) - q_D(j)).
```

These implement paper Equations (3)-(5), with the average over infinitely many opponents in Equation (4) replaced by the conditional expectation recovered from the pair distribution. They are counterfactual action-specific increments: `v_C` is the increment if the focal agent selects `C`, and `v_D` is the increment if it selects `D`.

The velocity arrays have no state axis. This implements the paper's argument preceding Equation (12): in the infinite complete population, an agent's averaged reward and hence its velocity are determined by its Q-values and the conditional distribution of all its incident states/opponents, not by the state of the one representative edge currently being transported.

If `m_t(i,j) == 0`, both velocities remain zero. Such a focal cell has no outgoing mass. Correct reuse for a second endpoint relies on pair-exchange symmetry, discussed below.

## 8. State transition rule

The transition is deterministic and symmetric in the two agents. The next state `z(g,a,b)` is:

| Old state | `CC` | `CD` | `DC` | `DD` |
| --- | --- | --- | --- | --- |
| `0 = s1 = SH` | `0` | `0` | `0` | `1` |
| `1 = s2 = PD` | `0` | `1` | `1` | `1` |

Thus only mutual defection sends an SH edge to PD, and only mutual cooperation sends a PD edge back to SH. This is the rule stated for Figure 1 in the paper.

## 9. Probability-mass transport

Define the quantised one-agent maps

```text
F_C(q_C, q_D) = (round_grid(q_C + v_C(q)), q_D),
F_D(q_C, q_D) = (q_C, round_grid(q_D + v_D(q))).
```

`delta_p()` allocates a zero `temp_p`. For every source cell with positive mass and for each of the four joint actions `(a,b)`, it performs the discrete pushforward

```text
temp_p[F_a(q1), z(g,a,b), F_b(q2)] +=
    p_t[q1,g,q2] x_a(q1) x_b(q2).
```

It then replaces `p` by `temp_p`.

This single expression exactly describes the 8 explicit branches in the source (4 action pairs in each of 2 states). Important details are:

- action probabilities are evaluated from the source Q-values;
- payoffs and velocities were also computed from the source distribution;
- only the coordinate of the selected action moves for each endpoint;
- the old state supplies the current payoff and the transition row;
- Q updates and the state transition are committed together into the next pair distribution;
- the four branch weights from a source cell sum to one, so mass is conserved if all destination indices are valid and floating-point accumulation is exact.

The scheme is a synchronous discrete-time, nearest-grid, semi-Lagrangian/finite-volume-style probability-mass pushforward. It is not a literal finite-difference evaluation of the derivatives in the continuous-time PDE of paper Equations (10), (12), and (13).

The same `q_c[k,m]` and `q_d[k,m]` fields are used to update the second endpoint. Those fields were computed from the first-endpoint conditional distribution. This is valid when

```text
p(q1,g,q2) = p(q2,g,q1),
```

which holds for the scripted initial condition and should be preserved by symmetric payoffs and transitions. The script neither enforces nor checks this invariant. It would be incorrect for an asymmetric pair distribution without computing endpoint-specific velocities.

## 10. Observables and saved output

`expected()` computes the first-endpoint marginal observables:

```text
E[Q(C)] = sum p_t(q1,g,q2) q1_C / sum(p_t),
E[Q(D)] = sum p_t(q1,g,q2) q1_D / sum(p_t),
E[x_C]  = sum p_t(q1,g,q2) x_C(q1) / sum(p_t),
E[x_D]  = sum p_t(q1,g,q2) x_D(q1) / sum(p_t),
p(s1)   = sum_{q1,q2} p_t(q1,0,q2) / sum(p_t).
```

These correspond to paper Equations (8) and (14)-(17). Symmetry makes the first-endpoint and second-endpoint averages equal.

The script creates a directory relative to the process working directory named:

```text
Tau=2_Eta=0.4_b=1.2_r=0.1_rebuttal_beta(20,80,80,20)_random
```

It appends to `ts_results.csv` with no header. Each row has six columns:

```text
T, ave_qc, ave_qd, ave_xc, ave_xd, ave_fp
```

The active loop writes 201 rows labelled `T = 0, ..., 200`. The row for `T` describes `p_T`. `delta()` is called before the write but changes only velocity work arrays, not `p`. After writing the `T = 200` row, the script still advances the pair mass once more to `p_201`, increments `T` to 201, and exits without writing that final mass.

The script also prints `Step`, `Q(C)`, and `Q(D)` for each saved time. The distribution snapshot writer is commented out. It does not save:

- the initial histogram or random state;
- model parameters in machine-readable form;
- the full pair distribution;
- marginal Q distributions;
- velocities or payoff moments;
- a CSV header;
- checksums, precision, or software versions.

Because output is opened in append mode, rerunning into an existing directory concatenates another trajectory without warning.

At inspection time no generated result directory or `ts_results.csv` was present in this subproject; only the source script and paper predated these planning documents.

## 11. Precise within-timestep chronology

The state at the beginning of labelled timestep `t` is `p_t`. One iteration is:

1. **Measure `p_t`.** Compute means of Q, mean Boltzmann strategies, and the state-0 mass.
2. **Form source strategies.** Conceptually compute `x_C(q)` and `x_D(q)` from Q-values in `p_t`. The original recomputes these scalar values inside several loops.
3. **Compute conditional expected rewards.** For each occupied focal Q-cell, condition `p_t` on that Q-cell and average the current-state payoff against opponents' source strategies.
4. **Compute action-specific velocities.** Form `v_C(q)` and `v_D(q)` using paper Equation (5).
5. **Save the labelled `t` observables.** No density has yet changed.
6. **Split source mass by joint action.** Each ordered pair cell is weighted by `x_a(q1) x_b(q2)`. At the agent-system level this represents each agent drawing one action and reusing it on all incident edges; the pair equation retains only the marginal for one edge.
7. **Apply the Q update.** On each action branch, move only each endpoint's selected Q-coordinate by its corresponding conditional expected velocity, then round it to the grid.
8. **Apply the edge-state transition.** Use the old state and the same joint action to choose the next state according to the table above.
9. **Accumulate `p_{t+1}` and replace `p_t`.** All updates are synchronous.
10. **Increment the label.** Set `T <- T + 1`.

This is consistent with the paper's narrative chronology: choose actions, play all current-state interactions, average rewards and update the chosen Q-value, then let the same actions drive end-of-step edge-state transitions. The new edge states first affect payoffs at timestep `t+1`.

## 12. Map from code to paper notation and equations

| Script item | Paper notation/concept | Equation or location |
| --- | --- | --- |
| `Eta` | `alpha`, learning rate | (1), (5) |
| `Tau` | `tau`, Boltzmann selection intensity | (2) |
| `u[g]` | row-agent payoff matrix `M_s` | (3), Figure 1 setup |
| `p[i,j,g,k,m]` | joint pair distribution `p(Q_t^1, s, Q_t^2, t)` | (7)-(15) |
| axes `(i,j)` | `Q_t^1 = (Q_t^1(C), Q_t^1(D))` | pair definition preceding (7) |
| axis `g` | edge state `s in S` | (7)-(10) |
| axes `(k,m)` | `Q_t^2 = (Q_t^2(C), Q_t^2(D))` | pair definition preceding (7) |
| `np.sum(p[i][j])` | focal marginal `p(Q_t^1,t)` at a grid point | discrete form of (14) |
| `xc`, `xd` | `x_i(Q)` for actions `C`, `D` | (2) |
| payoff indexing in `delta()` | `e_a^T M_s e_b`, then opponent/action average | (3), (4) |
| `q_c`, `q_d` | `v_C(Q,C)`, `v_D(Q,D)` | (5) |
| `delta_p()` action weights | `x_i(Q_t^1) x_j(Q_t^2)` | (9), (10), (13) |
| hard-coded next `g` | deterministic `z_s(s',a_i,a_j)` | (9), Figure 1 transition rule |
| transport of Q mass | discrete pushforward counterpart of conditional continuity/master equation | (11), (12) |
| joint Q/state transport | discrete-time counterpart of the combined evolution | (7), (10), (13) |
| `ave_fp` | `p(s1,t)` | (8) |
| `ave_qc`, `ave_qd` | `E[Q_t(a_i)]` | (14)-(16) |
| `ave_xc`, `ave_xd` | `E[x_t(a_i)]` | (17) |
| `init_p()` Betas | heterogeneous initial condition | Figure 1(b) caption and experiment text |
| `space`, `appro()` | numerical grid and nearest-grid projection | not specified in the main paper |

The paper often derives a conditional density `p(P,t | s)` and a state mass `p(s,t)` separately. The script stores their joint product directly as `p(P,s,t)` and transports joint mass, which avoids explicitly evaluating the quotient and derivative terms in Equation (13).

## 13. Differences, assumptions, and ambiguities

1. **Discrete map versus continuous-time PDE.** The paper presents differential Equations (10), (12), and (13), whereas the code takes a full synchronous Q-learning step with `alpha = 0.4` and projects the result to a grid. Agreement with the PDE is an empirical numerical claim, not an identity of discretisations.
2. **Mass versus density.** Script entries are cell masses. Calling them a density without noting the missing `h^4` scaling can lead to incorrect integrals in a rewrite.
3. **Random, unreproducible initial quadrature.** The paper specifies continuous Beta laws; the code uses a finite unseeded Monte Carlo histogram. This injects run-to-run variation before otherwise deterministic transport.
4. **Uniform initial state is encoded, not documented in code.** The factor `0.5` means states are independently equiprobable initially. The paper says heterogeneous-case states are determined at random but does not explicitly state the probability in the main text.
5. **Exchange symmetry is required.** Reusing first-endpoint velocity fields for the second endpoint assumes pair symmetry. The intended experiment has it; the code has no assertion and cannot safely support asymmetric games/data.
6. **Only one deterministic transition rule is implemented.** The paper also studies other deterministic and probabilistic transition matrices. There is no general transition tensor in the script.
7. **Nearest-grid projection is undocumented in the paper.** It can cause grid locking when an increment has magnitude below half a cell and introduces quantisation error. A conservative interpolation scheme would define a different numerical method and must not silently replace the legacy mode.
8. **No boundary policy.** The active parameters keep Q-values in range, but changed parameters can produce an out-of-range positive index or a silently wrapping negative NumPy index.
9. **Direct exponentials.** Safe here but less stable than a sigmoid/log-softmax implementation.
10. **Normalisation is diagnostic only.** Observables divide by `sum(p)`, masking small mass drift; transport itself is not renormalised and no mass/non-negativity checks are made.
11. **Final hidden advance.** The saved series ends at `p_200`, but computation advances to unsaved `p_201`.
12. **Output is neither isolated nor self-describing.** It is relative to the launch directory, appends on rerun, and omits seed/configuration metadata.
13. **The script is not an agent-based simulation.** It has no finite population, realised opponent payoffs, repeated simulation runs, or variance calculation. The paper reports separate simulations with `n = 1000` and 500 runs.
14. **Pair information cannot identify cross-opponent covariance.** A one-edge pair marginal determines single-edge reward moments but not the joint law of two different edges incident to the same focal agent.

## 14. Dense memory requirements

The pair array has

```text
N_pair = 131 * 131 * 2 * 131 * 131
       = 2 * 17,161^2
       = 588,999,842 elements.
```

| Precision | One pair array | One pair array | Current + destination | Current + destination |
| --- | ---: | ---: | ---: | ---: |
| `float32` | 2,355,999,368 bytes | 2.356 GB / 2.194 GiB | 4,711,998,736 bytes | 4.712 GB / 4.388 GiB |
| `float64` | 4,711,998,736 bytes | 4.712 GB / 4.388 GiB | 9,423,997,472 bytes | 9.424 GB / 8.777 GiB |

The original NumPy code uses `float64`, so `p` plus `temp_p` alone require about 8.78 GiB during each update. The two velocity grids together are only 274,576 bytes in float64 (137,288 bytes in float32).

These figures are lower bounds. A naive JAX implementation can consume much more through XLA temporaries, broadcast action weights, destination indices, copies caused by layout changes, and allocator preallocation. For perspective, a single full `int32` destination-index vector has the same 2.356 GB byte size as the float32 pair array; four materialised branch-index vectors would add 9.424 GB. Peak-memory design, buffer donation, and chunked/separable transport are therefore requirements, not optional optimisations.

## 15. JAX execution classification

### 15.1 Directly vectorisable and GPU-friendly

- Build the Q-grid and flattened `(M,2)` agent-Q table.
- Precompute `x_C(q)` and `x_D(q)` with a sigmoid.
- Compute total mass, state masses, endpoint marginals, mean Q-values, and mean strategies with reductions/einsums.
- Compute conditional payoff numerators without a five-dimensional broadcast. With a `(2,M,M)` layout, define for state `g` and focal action `a` the length-`M` vector

  ```text
  r_{g,a}(q2) = sum_b x_b(q2) M_g[a,b].
  ```

  Then the numerator for all focal Q-values is `sum_g P_g @ r_{g,a}` and the marginal denominator is `sum_g P_g @ ones`. These dense matrix-vector operations are GPU-friendly and avoid materialising reward tensors.
- Form velocities and quantised destination maps `F_C`, `F_D` for all `M` one-agent grid points.
- Evaluate invariant diagnostics such as mass, minimum mass, symmetry error, and moment ranges.
- Run fixed-length stepping under `jax.lax.scan` once output/checkpoint policy is settled.

### 15.2 Scatter/gather or sparse-segment operations

- The pair-mass pushforward is many-to-one and requires additive accumulation. A naive flat scatter over all 588,999,842 source cells for each action branch is too memory-intensive.
- The endpoint maps are separable: for each source state and action pair, the operator has the form

  ```text
  P'_z += A_a P_g A_b^T,
  ```

  where `A_a[dest,source]` has one nonzero `x_a(source)` per source column in legacy nearest-grid mode. This suggests two staged segment-sums/scatters, precomputed inverse buckets, or carefully benchmarked sparse operators instead of full per-cell destination arrays.
- ABM Q updates need indexed updates to each agent's selected action coordinate.
- ABM reward aggregation over complete-graph edges needs segment sums to both endpoint agents.

### 15.3 Operations needing chunking or redesign

- Chunk pair transport over source rows/columns or endpoint blocks while accumulating into a persistent destination buffer. Chunk sizes must be chosen from measured peak memory, not only the nominal density size.
- Do not materialise `x_a[:,None] * x_b[None,:]` for all branches unless profiling proves it affordable. Fuse weights into chunked transport.
- Avoid storing a full dense destination index for each action pair. Store only the length-`M` endpoint maps and combine them inside a staged operator.
- Consider donating the old pair buffer, but assume that both current and next arrays remain live unless profiling the compiled executable proves otherwise.
- JAX GPU scatter-add may use atomic accumulation with non-deterministic floating-point order. CPU float64 should be the strict reference; GPU validation should use tolerances and invariant/moment comparisons, not bitwise equality.
- For larger or finer grids, benchmark dense, block-sparse/COO, and separable segment implementations. Initial support is sparse, but support can expand, so sparsity must be measured over time rather than assumed.
- The complete-graph ABM has `E = n(n-1)/2` edge states. Store upper-triangle edge lists/state vectors, not an `n x n` duplicated state matrix. At `n = 1000`, `E = 499,500`; for much larger `n`, edge chunks are needed even though agent arrays remain small.

### 15.4 Interpolation policy

Exact legacy reproduction requires nearest-grid transport with one destination per joint-action branch. A scientifically preferable optional scheme could use conservative linear interpolation along the selected coordinate of each endpoint. Each endpoint then contributes to two neighbouring cells, so one joint-action branch contributes to four pair destinations (16 action/interpolation contributions per source cell across the four joint actions). This should be a separately named solver mode with convergence tests; it must not be presented as exact reproduction of `case2_1.py`.

## 16. Finite-population ABM semantics for the later implementation

A faithful finite-population simulator should use:

- `Q`: shape `(n_agents, 2)`;
- `actions`: shape `(n_agents,)`, one Bernoulli/Boltzmann draw per agent per timestep;
- `edge_u`, `edge_v`: fixed endpoint arrays of length `E = n(n-1)/2`;
- `edge_state`: shape `(E,)`, preferably an integer/boolean dtype;
- vectorised payoff lookup on each edge using the old edge state and the two endpoint actions;
- segment sums of the row payoff to `edge_u` and the transposed/second-player payoff to `edge_v`;
- division by `n-1`, then an indexed update of only each agent's chosen Q-coordinate;
- vectorised old-state/joint-action transition for every edge after payoff evaluation;
- explicit PRNG-key threading and recorded seeds.

Drawing one action per edge would define a different model and would erase an important shared-action source of cross-edge dependence. The paper explicitly requires one action per agent, reused against all opponents.

## 17. Variance target and the limit of pair closure

Let `K = n-1`, and let `Y_ih(a)` be focal agent `i`'s payoff from opponent `h` when the focal action is `a`. Conditional on `i` selecting `a`, its realised chosen-coordinate increment is

```text
Delta Q_i(a) = alpha ((1/K) sum_h Y_ih(a) - Q_i(a)).
```

Therefore,

```text
Var[Delta Q_i(a) | conditioning]
  = alpha^2 / K^2 *
    (sum_h Var[Y_ih] + 2 sum_{h<l} Cov[Y_ih, Y_il]).
```

If distinct opponents are exchangeable with per-edge variance `sigma^2` and distinct-edge covariance `c`, this becomes

```text
alpha^2 (sigma^2/K + (K-1)c/K).
```

The pair distribution can supply a focal-Q-conditioned single-edge payoff law and hence a candidate `sigma^2`. It cannot determine `c`, because `c` requires a triplet/two-edge marginal such as

```text
p(Q_i, s_ih, Q_h, s_il, Q_l).
```

Possible later approaches are:

1. impose conditional independence and set `c = 0` as a clearly labelled pair-closure prediction;
2. introduce and validate a triplet closure;
3. estimate `c` directly from the ABM, stratified by time, focal Q, and focal action;
4. use the law of total covariance to separate dependence due to the shared focal action/Q from residual edge-edge dependence.

The unconditional variance of the full vector update additionally contains the mixture variance from whether action `a` was selected at all. It must not be conflated with the conditional variance of the paper's counterfactual velocity `v_a(Q,a)`.

## 18. Questions not answerable from the available files

1. Which random seed and exact empirical Beta histogram, if any, generated the theoretical dots in Figure 1(b)?
2. Was the released script's nearest-grid pushforward the exact numerical method used for the paper, and what convergence checks were performed? The main paper defers derivation details to supplementary material that is not present locally.
3. Is the initial random state probability exactly one half in the agent-based simulations, and are different edge states sampled independently? The script encodes one-half pair mass; the paper only says states are determined at random.
4. In the planned variance study, is "realised Q-learning velocity" meant conditionally on focal Q and selected action, unconditionally for a randomly selected coordinate, or as a population-level sample statistic?
5. Which sources of variance should the theory include: opponent action draws, focal action draws, finite empirical Q/state composition, random initialisation, transition history, and/or variation between simulation runs?
6. Should cross-opponent covariance be predicted with a declared conditional-independence closure, a new triplet approximation, or treated as an empirically measured correction?
7. Should the ABM initialise continuous Q-values exactly from the Beta laws or use the pair solver's discretised empirical histogram for like-for-like comparisons? Both comparisons are useful but answer different questions.
8. What boundary behaviour should apply for future payoff matrices, learning rates, or Q-ranges that violate the current convex-range invariant?
9. After exact legacy parity, should the production pair solver retain nearest-grid rounding or adopt conservative interpolation/convergence refinement?
10. What GPU hardware, memory budget, precision requirement, and reproducibility tolerance will define the target scale?
11. Are the staged deletions of the older PDF and `case2_1_jax.py` intentional and permanent? They were treated as user-owned worktree state and not inspected or changed.

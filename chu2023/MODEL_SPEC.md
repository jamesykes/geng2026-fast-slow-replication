# Chu et al. pair-approximation model specification

## 1. Scope and source provenance

This document records the behaviour of the original sources currently present in this subproject. It is a specification for later work, not a description of a rewrite.

Sources inspected in full:

- `case2_1.py` (296 lines; SHA-256 `ebeb0d64e57259e940a49437e6ec6a6a3be636dd99b124f3e8858e1dac9db68f`)
- `pair-approx_multi-agent_stochastic_games.pdf` (the eight-page AAAI-23 paper *A Pair-Approximation Method for Modelling the Dynamics of Multi-Agent Stochastic Games*; SHA-256 `6f25ad7c8ea088666ed1e7b8aa6b0f2a100d0817d145a07b4154e0637502eab0`)

The parent repository's `README.md` says that `chu2023/` contains the Chu et al. code and reference paper, separate from later replication work. No local `AGENTS.md` was present when this specification was written.

The previously staged deletions of `Dynamics_of_Q-Learning_in_Networked_Stochastic_Games.pdf` and `case2_1_jax.py` were later confirmed to be intentional and committed. Neither deleted file was used as a source for this specification.

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

The finite-population ABM uses continuous Q-values after initialisation. Its selected-coordinate update is the unprojected learning rule

```text
Q_{t+1}^i(a_j) = Q_t^i(a_j) + alpha [r_t^i(a_j) - Q_t^i(a_j)].
```

Nearest-grid projection belongs only to the numerical pair-mass solver. "Grid-matched initialisation" means that initial agent Q-vectors may be sampled from a discrete pair-solver histogram; it never means that subsequent ABM updates are quantised.

For the primary controlled theory-versus-ABM comparison:

- the pair solver constructs a seeded discrete one-agent Q histogram;
- initial ABM Q-vectors are sampled from that same realised histogram;
- initial ABM edge states are sampled independently with probability `1/2` for `SH` and `1/2` for `PD`, matching the factor `0.5` in the scripted initial pair mass;
- continuous scaled-Beta initial Q-values remain available as a secondary, paper-like ABM mode.

Drawing one action per edge would define a different model. The paper explicitly requires one action per agent, reused against all opponents. Conditional on an exact current focal Q-vector and the current focal action, that focal action is fixed and is not itself a contemporaneous source of cross-edge covariance. The rule nevertheless matters for covariance because reusing focal actions over earlier timesteps can correlate the persistent histories of incident edge states.

## 17. Primary variance estimand

For focal agent `i` and action/Q-coordinate `j`, define the realised coordinate velocity at time `t` by

```text
v_j^i
  = Q_{t+1}^i(a_j) - Q_t^i(a_j)
  = 1{A_t^i = a_j} alpha [r_t^i(a_j) - Q_t^i(a_j)].
```

The primary estimand is the local, one-step conditional variance

```text
D_j(q,t) = Var[v_j^i | Q_t^i = q, A_t^i = a_j].
```

Conditional on `A_t^i = a_j`, the indicator is one. Conditional on the exact Q-vector `q`, `q_j` is constant. Therefore

```text
D_j(q,t)
  = alpha^2 Var[r_t^i(a_j) | Q_t^i = q, A_t^i = a_j].
```

This conditioning is part of the scientific definition, not merely an estimation convenience:

- conditioning on the complete Q-vector prevents stochastic reward noise from being mixed with heterogeneity between agents at different Q-values;
- conditioning on the selected action identifies the coordinate that is actually updated;
- under the model, the current focal action is sampled only from the focal Q-vector, so at exact `q` conditioning additionally on `A_t^i = a_j` does not change the current conditional distribution of opponents or incident edge states;
- nevertheless, the action condition is retained to make the realised-update interpretation explicit.

This `D_j(q,t)` is not the covariance between the `C` and `D` velocity coordinates. It is also not an unconditional coordinate-update variance that mixes selected and unselected actions.

## 18. Edge variance, cross-opponent covariance, and exact decomposition

Let `N = n-1`. For opponent `h`, define the counterfactual one-edge payoff for focal action `j` as

```text
Y_ih^(j) = e_j^T M_{s_t^{ih}} e_{A_t^h}.
```

Conditional on the focal action `j`, the realised reward is

```text
r_t^i(a_j) = (1/N) sum_h Y_ih^(j).
```

For exchangeable distinct opponents, define

```text
mu_j(q,t)
  = E[Y_ih^(j) | Q_t^i=q, A_t^i=a_j],

sigma_j^2(q,t)
  = Var[Y_ih^(j) | Q_t^i=q, A_t^i=a_j],

c_j(q,t)
  = Cov[Y_ih^(j), Y_ik^(j) | Q_t^i=q, A_t^i=a_j],  h != k.
```

The exact variance identity is

```text
D_j(q,t)
  = alpha^2 [sigma_j^2(q,t)/N + (N-1)c_j(q,t)/N]
  = alpha^2 [sigma_j^2(q,t)/(n-1) + (n-2)c_j(q,t)/(n-1)].
```

The cross-opponent covariance `c_j` concerns payoffs on two distinct edges sharing a focal agent. It is not the covariance between the `C` and `D` Q-coordinate velocities and is not merely the negative covariance between multinomial category counts.

Because the current focal action is fixed by the conditioning, `c_j` must not be described as simply arising from the current shared focal action. Nonzero covariance can instead survive because earlier one-action-per-agent draws jointly affected all incident edges and thereby correlated their persistent state histories. Conditioning only on the current focal Q-vector also marginalises over latent incident-edge configurations that can induce cross-edge dependence.

## 19. One-edge moments supplied by the pair density

The pair mass `p(q,s,q',t)` determines the conditional law of one randomly chosen incident edge. In the discrete implementation define

```text
f(q,t) = sum_s sum_q' p(q,s,q',t)

w_sb(q,t)
  = [sum_q' p(q,s,q',t) x_b(q')] / f(q,t).
```

Here `w_sb(q,t)` is the probability, conditional on focal `Q=q`, that the sampled incident edge is in state `s` and the opponent selects action `b`. It sums to one over `(s,b)` whenever `f(q,t) > 0`.

For

```text
y_j(s,b) = e_j^T M_s e_b,
```

the pair-derived conditional moments are

```text
mu_j^pair(q,t)
  = sum_s,b w_sb(q,t) y_j(s,b),

m2_j^pair(q,t)
  = sum_s,b w_sb(q,t) y_j(s,b)^2,

sigma_j,pair^2(q,t)
  = m2_j^pair(q,t) - (mu_j^pair(q,t))^2.
```

These are computable by the same state-slice matrix-vector contractions used for the mean payoff, with a second payoff-squared lookup. They should be produced for both focal actions wherever the focal mass is nonzero.

The pair density does not determine `c_j`, because that requires a joint distribution of two edges sharing the same focal agent, for example

```text
p(Q_i, s_ih, Q_h, s_ik, Q_k, t).
```

The pure pair-closure prediction is therefore explicitly

```text
D_j^pair(q,t)
  = alpha^2 sigma_j,pair^2(q,t)/(n-1),
```

with `c_j=0` clearly labelled as a conditional-independence closure. For the first project version, `c_j` will be measured from the ABM; no triplet-density closure will be derived or implemented.

Pair-density evolution is independent of finite population size `n`. Once one pair trajectory has been solved for a model parameter set and initial condition, its one-edge moments can be reused to form `c_j=0` predictions for multiple values of `n`. Separate ABM simulations remain necessary for each `n`, because finite-population trajectories and `c_j` can depend on `n`.

### 19.1 Phase 4 JAX pair-mass representation and chronology

Phase 4 implements the Phase 1 nearest-grid oracle in JAX without changing its scientific law. Let `G` be the number of grid values per Q coordinate and `M=G^2` the number of two-coordinate agent types. The public conversion boundary is the canonical shape

```text
(q1_C, q1_D, state, q2_C, q2_D) = (G,G,2,G,G),
```

while compiled kernels use contiguous state slices

```text
p_flat.shape = (2,M,M).
```

Both endpoint axes remain ordered; endpoint exchange is `p_flat[s,u,v] = p_flat[s,v,u]`. Elements remain discrete probability masses, so no grid-spacing factor enters any sum. The JAX independent initial pair is exactly `pi_s P_u P_v`, with a configured two-state probability vector `pi`; it is an outer product rather than a correlated equal-endpoint construction.

For each source mass, one JAX step performs this chronology:

1. sum over old state and second endpoint to obtain `f_t(q)`;
2. evaluate both endpoint policies from their source `Q_t` values;
3. contract the source pair mass and opponent policy to obtain `w_t(s,b|q)`, then `mu_t`, `m2_t`, and `sigma_t^2` from the authoritative payoff tensor;
4. compute both action-specific conditional increments `alpha[mu_j(q,t)-q_j]` for every occupied focal type;
5. apply the active legacy decimal pre-round, nearest multiple-of-spacing search, and left-tie rule to the selected coordinate only, independently for every endpoint type and action;
6. for every old-state ordered source pair, weight `(C,C)`, `(C,D)`, `(D,C)`, and `(D,D)` by the product of the two source policies, gather the corresponding endpoint-specific projected destinations, transition from the old state with the authoritative transition tensor, and scatter-add the mass;
7. return the complete synchronous destination mass without clipping, interpolation, diffusion, or renormalization.

The grid projection uses integer decimal ticks matching the Phase 1 compatibility implementation. Invalid or non-finite projected values and projected integer ticks outside the supported int32 representation make the destination-valid flag false; checked host entry points raise rather than using the internal safe scatter sentinel as a boundary policy. Float64 grids are rejected unless `JAX_ENABLE_X64=1` was active before JAX import, preventing a nominal float64 configuration from silently becoming float32.

The source domain has `D=2M^2` cells. The implemented baseline traverses this fixed domain in source chunks of `K=min(chunk_size,D)` using `jax.lax.fori_loop`; each chunk emits four branch contributions into one flat scatter accumulator. There is no Python loop over occupied cells inside the kernel. A multi-step `jax.lax.scan` carries only the current `(2,M,M)` mass and a scalar destination-valid flag. It retains per-step state masses, mean Q, mean action probabilities, mass/symmetry/minimum diagnostics, and conditional-moment validity, but no `(T,2,M,M)` density history. `T=0` returns the unchanged initial mass and empty diagnostic arrays.

The checked interfaces validate the input shape and element count before device-to-host conversion, then validate dtype, finiteness, nonnegativity, and endpoint symmetry. After stepping they reject any invalid destination and repeat mass/symmetry validation. No mass normalization occurs. Empty focal types have zero conditional weights, means, second moments, variances, and velocities; occupied types must have weights summing to one within the configured diagnostic tolerance.

The guarded CPU smoke runner derives `G`, `M`, `M^2`, and `D` with Python integers before allocating a grid or JAX array. For floating item size `b`, effective chunk `K`, `T` steps and retained diagnostic-row count `R_d`, its backend-independent static allowances are

```text
device_static
  = 8Db + [Gb + M(20b+40)] + 17Kb + 96K + T(11b+3)

host_static
  = Db + T(11b+3) + 8M + [G(16+b) + M(8+4b)]
    + 4096 R_d + S + 1 MiB

combined_static = device_static + host_static.
```

The old `3Db` device term did not cover the compiled `fori_loop`/scatter buffers and was rejected after executable analysis exceeded it on representative reduced grids. The replacement reserves eight complete device densities before adding pointwise, branch and full device-diagnostic arrays. The checked simulation retains its full `T(11b+3)` host diagnostic transfer while validating the final `Db` density copy, so both appear simultaneously in `host_static`; selected Python/CSV rows are separate. Host storage also includes histogram/grid construction, bounded normalized serialization and hashing. This is a conservative modeled pair-work requirement, not process RSS; interpreter/imported-library memory, compiler code/cache, allocator overhead and already-parsed TOML objects are recorded exclusions.

`S` distinguishes encoded payload size from live serialization memory. Let `J` be the bounded ASCII length of complete normalized metadata, `C=4096` the maximum metadata chunk or CSV header/record, `U=8` bytes per retained text character, `F=8192` the explicit binary file buffer and `H=65536` fixed serializer/object overhead. Then

```text
metadata encoding peak = 3 U J + H
metadata/CSV write peak = U J + U C + C + F + H
S = max(metadata encoding peak, metadata/CSV write peak).
```

The tripled encoding term permits bounded input-object/string storage, temporary escaped fragments and the joined JSON text to coexist. The input metadata object is released after the checked string is produced. The write term permits the retained metadata string, one transient text chunk/record, its ASCII bytes and the file buffer to coexist. JSON and CSV stages are sequential. The factor `U=8` deliberately avoids relying on CPython's compact one-byte ASCII representation.

After raw TOML parsing, the runner requires exactly the documented `model`, `grid`, `solver`, `initial_condition`, and `output` tables and keys, normalizes their bounded scalar values, and rejects unknown fields. `run_name` is at most 64 safe ASCII characters. Only this normalized configuration is serialized. Because `ensure_ascii=True` guarantees ASCII output, character count equals encoded ASCII byte count and exact validation does not create a full byte copy. A worst-case JSON-escaping calculation bounds the normalized configuration, Git status, compiled-analysis reason, at most eight bounded device descriptions, fixed metadata, and one CSV header or record. The separate `S` formula bounds live text, escaped fragments, encoded chunks and file buffers. CSV validation uses a non-retaining counting sink; final JSON and CSV are written in bounded binary chunks. All exact checks finish before output-directory creation.

Only after schema validation and the static combined check accept does the runner build the small `QGrid`/JAX grid tables. It then lowers and compiles the scan from an abstract `(2,M,M)` `ShapeDtypeStruct`, before histogram or pair-density allocation. When `memory_analysis()` is complete, executable device storage is `argument + output + temporary - alias`, and executable host storage is `host_argument + host_output + host_temporary - host_alias`. The device requirement must not exceed `device_static`; adding compiled host storage and `host_static` gives the compiled combined requirement checked against the same fixed 256 MiB cap. Missing, incomplete, invalid or inconsistent backend analysis is a `compiled_analysis_unavailable` safeguard violation. Normal execution fails closed before pair allocation; only recorded `--allow-expensive` may override it, without claiming a compiled pass. Static shape caps limit compilation exposure, but compilation's own code/cache and unreported host-memory usage cannot be measured before compilation.

Fixed ordinary-run limits are `M<=4096`, `M^2<=4,000,000`, `D<=8,000,000`, 64 MiB initial pair mass, 256 MiB static/compiled combined storage, zero full-density snapshots and 10,000 diagnostic rows. Input configuration cannot raise them. A recorded `--allow-expensive` may override resource violations but never shape, numerical or scientific validity checks.

The JAX pair law still determines only the one-edge distribution and its `mu`, `m2`, and `sigma^2`. Phase 4 does not infer `c_j`, perform the final pair/ABM variance comparison, introduce interpolation, run the full legacy grid, or establish GPU performance or precision.

## 20. O(N) ABM moment and covariance estimator

The ABM must not enumerate all ordered pairs of opponents. For one focal agent/configuration and one counterfactual focal action `j`, compute

```text
S1 = sum_h Y_ih^(j),
S2 = sum_h (Y_ih^(j))^2.
```

Then

```text
S1^2 - S2 = sum_{h != k} Y_ih^(j) Y_ik^(j).
```

For `N > 1`, the per-focal sufficient statistics are

```text
edge_mean_i        = S1/N,
edge_second_i      = S2/N,
distinct_product_i = (S1^2-S2)/(N(N-1)).
```

Under common conditional weights, their conditional expectations give

```text
mu_j       = E[edge_mean_i],
sigma_j^2  = E[edge_second_i] - mu_j^2,
c_j        = E[distinct_product_i] - mu_j^2.
```

The corresponding average-reward variance is

```text
sigma_j^2/N + (N-1)c_j/N.
```

Using consistent raw-moment weights makes this decomposition an algebraic estimator check, apart from finite-sample bias conventions and floating-point error. `c_j` is undefined for `n=2` (`N=1`), where only the single-edge variance term exists.

Both `Y_ih^(C)` and `Y_ih^(D)` may be computed from the same current edge states and realised opponent actions, regardless of the focal action actually selected. These are counterfactual diagnostic quantities that improve estimation of `mu_j`, `sigma_j^2`, and `c_j`. Direct validation of realised velocity must still use only agents whose coordinate `j` was actually selected and updated.

## 21. Estimating exact-Q conditional moments from finite ABM data

Exact conditioning on a continuous `Q=q` is unavailable from finite simulations. The first implementation will use configurable narrow two-dimensional Q-bins; kernel smoothing or local regression is deferred unless binning proves inadequate.

For every action, time, and bin, record:

- the selected-action and counterfactual diagnostic sample counts;
- the Q-coordinate means, covariance, ranges, and other dispersion diagnostics;
- conditional first and second moments of reward and realised velocity;
- one-edge first/second moments and distinct-edge products;
- the bin edges and the weighting rule;
- occupancy/quality flags, with sparse bins suppressed or clearly marked.

A finite-width bin is only an approximation to exact-Q conditioning. In particular,

```text
Var[alpha(r-Q_j) | Q in B, A=j]
  = alpha^2 [Var(r | B,A=j)
             + Var(Q_j | B,A=j)
             - 2 Cov(r,Q_j | B,A=j)],
```

so it is not generally equal to `alpha^2 Var[r | Q in B,A=j]`. The direct realised-velocity bin variance and reward-based bin variance must both be reported during validation, together with the Q-dispersion and reward-Q covariance terms. They should converge as bins narrow if the local estimator is behaving properly.

Action conditioning also affects finite-bin weights even though it has no effect at exact `q`: within a bin, `x_j(q)` varies, so the Q distribution among samples with realised `A=j` is proportional to occupancy times `x_j(q)`. Pair predictions must therefore use the same empirical selected-sample Q weights, or theoretical weights proportional to pair occupancy times `x_j(q)`, rather than evaluating only at the bin centre. Counterfactual all-agent ABM diagnostics must likewise be restricted or reweighted when they are compared with the selected-action target.

For pair-theory bin comparisons, retain the distinction between:

- an occupancy-weighted average of local exact-grid variances `sigma_j,pair^2(q,t)`; and
- the variance of the pooled one-edge mixture in a bin, obtained by averaging `m2_j^pair` and `mu_j^pair` before subtracting the squared pooled mean.

The first targets an average of local exact-Q variances; the second matches a pooled finite-bin edge distribution and includes between-Q variation of the conditional mean. Both can be useful, but they must not be interchanged. Initial results must include bin-refinement checks and avoid conclusions driven by poorly populated or overly wide bins.

### 21.1 Phase 3A implemented observation and bin conventions

Phase 3A implements the realised selected-action sample only. At source time `t`, every focal agent contributes exactly one observation to the stratum for its realised `A_t^i`; no zero is inserted for the unselected coordinate. The retained scan fields are `Q_t`, selected action, selected `Q_t(A_t^i)`, reward, selected-coordinate velocity, `S1`, and `S2`. Run and source-time remain explicit array axes, and the focal-agent axis is reduced only when forming the configured strata. Counterfactual diagnostics for the unselected focal action remain a later Phase 3 extension.

The instrumented scan obtains `S1` and `S2` from the same old-state, oriented endpoint payoffs already used for the reward. It retains only agent-sized arrays with shapes `(R,T,n)` (or `(R,T,n,2)` for complete Q/policy arrays), never an `(R,T,E)` payoff history. It uses the same action key and deterministic transition as the uninstrumented scan and consumes no additional randomness.

For a stratum with `K` selected-action observations, Phase 3A stores common population-moment sums and calculates

```text
mu  = sum_i S1_i / (K N),
m2  = sum_i S2_i / (K N),
m11 = sum_i (S1_i^2-S2_i) / (K N(N-1)),
```

with `sigma^2=m2-mu^2` and `c=m11-mu^2`. Empty estimates are missing rather than zero. For `n=2`, `m11` and `c` are explicitly undefined and the decomposed reward variance contains only `sigma^2/N`.

Two-dimensional bins are left-closed and right-open on both coordinates, except that the final bin includes its upper edge. Configured edges are retained in float64 for provenance. Classification converts them to the Q-observation dtype and revalidates finiteness and strict increase; edges that collapse in float32 are a configuration error. A value equal to the effective final upper endpoint is included, while the immediately adjacent representable value outside either effective endpoint raises. Observations are never clipped or silently discarded. CSV and metadata distinguish configured edges from effective comparison edges. Every run/time/bin/action cell retains its count plus explicit empty, underpopulated, minimum-count, and distinct-covariance validity flags. Population moments use `ddof=0`. Independent runs remain separate and are not pooled as though agents within one complete network were independent uncertainty replicates.

The Phase 3A diagnostic derives `Bc` and `Bd` from raw parsed sequence lengths and preflights dense statistic resources before constructing `QBinSpec` or any NumPy edge array, graph, initialization, simulation, aggregation, or output. It does not claim to guard memory already used by the TOML parser's Python lists. With `S=R*T*Bc*Bd*2` strata and `O=R*T*n` agent observations, the estimate includes `88S` bytes for one int64 count plus ten float64 sufficient sums; observation-dtype host fields and five `intp` indices over `O`; five float64 product arrays (`40O`); five float64 conversion arrays (`40O`) for float32 observations only, because float64 conversions alias the host fields; fourteen returned float64 moments, four boolean masks, four non-returned float64 derivation intermediates, three float64 expression-work arrays; and configured/effective NumPy bin-edge storage even when `T=0`. CSV rows are streamed, so export creates no additional stratum-scaled collection and emits exactly `S` rows. Fixed normal-run caps are 1,000,000 strata, 256 MiB estimated peak statistic bytes, and 250,000 output rows; only the explicit recorded `--allow-expensive` flag bypasses them.

The shared Phase 2 resource guard defaults to its committed baseline accounting and does not claim Phase 3A arrays. The Phase 3A diagnostic explicitly selects instrumented accounting, adding exactly `selected_q_t`, `payoff_sums_t`, and `payoff_square_sums_t` to retained agent records and the two live S1/S2 accumulators to working memory.

### 21.2 Phase 3B implemented pooling, uncertainty, and refinement conventions

Let `z=(t,B_c,B_d,j)` denote one source-time, two-dimensional Q-bin, selected-action stratum, and let run `r` retain count `K_r(z)` and the ten Phase 3A sufficient sums. The Phase 3B point estimate first calculates

```text
K(z) = sum_r K_r(z),
U_k(z) = sum_r U_{rk}(z)
```

and only then applies the Phase 3A nonlinear population-moment formulas. It is therefore observation weighted within each conditional stratum. It is not an unweighted mean of separately calculated run-level variances, which would target a different quantity when run counts differ.

For uncertainty, Phase 3B generates one local-NumPy-RNG multiplicity matrix `W` with shape `(B,R)`. Row `b` results from drawing `R` complete run indices with replacement, so every row sums to `R`. Bootstrap replicate `b` replaces every sufficient sum by `sum_r W[b,r] U_r` and recomputes all nonlinear estimands. The exact same `W` is used for all source times, bins, selected actions, estimands, and refinement schemes. Thus dependence among agents, edges, actions, times, and bins within one finite-network trajectory remains inside the resampled cluster. Bootstrap generation has a separate seed and neither consumes nor changes JAX initialization or dynamics keys.

Intervals are pointwise percentile intervals. The lower and upper probabilities are `(1-confidence)/2` and `1-(1-confidence)/2`, evaluated by `numpy.quantile(..., method="linear")` on the finite bootstrap values for that estimand and stratum. An interval is valid only if the original stratum has at least two independently simulated contributing runs and at least `max(2,ceil(0.8B))` finite bootstrap replicates. The output retains total focal-observation count, contributing-run count, point estimate, both endpoints, total/valid/invalid replicate counts, and an explicit validity flag. Invalid endpoints remain `NaN`; no zero or substitute interval is fabricated. This conservative operational policy does not itself guarantee coverage, especially for the bounded smoke run.

The reported pooled estimands are `mu`, `m2`, `m11`, `sigma^2`, `c`, direct and decomposed reward variance, direct and finite-bin-corrected velocity variance, `Var(selected Q)`, and `Cov(reward,selected Q)`. The output also retains both algebraically equivalent finite-bin discrepancy calculations

```text
direct Var(v) - alpha^2 direct Var(reward)

alpha^2 [Var(selected Q) - 2 Cov(reward,selected Q)].
```

Agreement of these two columns is a numerical identity check. Neither column changes the exact-Q definition of `D_j`; it exposes the effect of estimating it with a finite-width bin.

Several named bin schemes may be applied to one unchanged set of source-time ABM records. All schemes have identical configured outer bounds. Each successor must have more bins and contain every parent edge in both coordinates, both in configured float64 provenance and after conversion to the observation dtype. Collapsed, non-finite, ambiguous, non-nested, or differently bounded schemes are rejected. Classification remains left-closed/right-open with the final upper endpoint included. For each adjacent refinement pair, child counts must sum exactly to every parent count. Floating sufficient sums are compared with a field-specific forward-error bound because direct parent aggregation and regrouped child aggregation add the same represented terms in different orders.

Two floating dtypes enter this check and must not be conflated. The observation dtype `o` is the float32 or float64 dtype in which JAX represents payoffs and computes agent-level `S1`, `S2`, rewards, selected Q values, and velocities. The summation dtype `s` is the dtype of the Phase 3A sufficient arrays and their parent/child reconstruction, currently float64. For either dtype `d`, let `epsilon_d=numpy.finfo(d).eps` be machine epsilon, not the smaller round-to-nearest unit roundoff, and define `gamma_k^d=k*epsilon_d/(1-k*epsilon_d)`. Using machine epsilon here is an intentional conservative substitution. A bound is rejected if `k*epsilon_d>=1`.

Let `rho_d(x,k)` be an outward-rounded bound for `k` represented operations applied to a non-negative exact-magnitude bound `x`:

```text
rho_d(x,k) = outward[ x + gamma_k^d x + k eta_d ],
```

where `eta_d` is the smallest positive subnormal of dtype `d`. Every non-negative base sum, product, or quotient used below is also rounded outward in binary64 and overflow/non-finite bounds fail explicitly. Define `p_o` as the largest absolute payoff after casting the authoritative payoff tensor to dtype `o`, `q_o` as the largest absolute effective Q-bin endpoint, `alpha_o` as the represented absolute learning rate, and `N_o` as `N=n-1` represented in dtype `o`. The actual represented per-observation term bounds are:

| Sufficient field | Actual term entering aggregation | Evaluation dtype before sufficient sum | Represented-value bound |
|---|---|---|---|
| `sum_s1` | scatter sum `S1=sum_h y_h` | `o` | `B1=rho_o(N p_o,N)` |
| `sum_s2` | scatter sum `S2=sum_h fl_o(y_h*y_h)` | `o` | `B2=rho_o(N rho_o(p_o^2,1),N)` |
| `sum_distinct_products` | `fl_s(fl_s(S1*S1)-S2)` | `s` after exact promotion to `s` | `rho_s(rho_s(B1^2,1)+B2,1)` |
| `sum_reward` | `r=fl_o(S1/N_o)` | `o` | `Br=rho_o(B1/N_o,1)` |
| `sum_reward_squared` | `fl_s(r*r)` | `s` | `rho_s(Br^2,1)` |
| `sum_selected_q` | represented selected `Q_j` | `o`, then exact promotion | `Bq=q_o` |
| `sum_selected_q_squared` | `fl_s(Q_j*Q_j)` | `s` | `rho_s(Bq^2,1)` |
| `sum_reward_selected_q` | `fl_s(r*Q_j)` | `s` | `rho_s(Br Bq,1)` |
| `sum_velocity` | `fl_o(alpha_o*fl_o(r-Q_j))` | `o` | `Bv=rho_o(alpha_o rho_o(Br+Bq,1),1)` |
| `sum_velocity_squared` | `fl_s(v*v)` | `s` | `rho_s(Bv^2,1)` |

The distinct-product row deliberately bounds the actual stored expression rather than only its ideal ordered-distinct interpretation `N(N-1)p^2`. In particular, when `N=1`, separately rounded `S1*S1` and `S2` need not cancel; the represented bound remains positive and includes this additive rounding residual.

For one parent cell with `m` observations, child counts `m_l`, stored parent sum `P`, stored child sums `C_l`, and `L` child bins, the accepted reconstruction difference uses the summation dtype and is bounded by

```text
gamma_m A
+ sum_l gamma_{m_l} A_l
+ gamma_L [sum_l |C_l| + sum_l gamma_{m_l} A_l]
+ underflow allowance,
```

where `A=max(m*b_f,|P|)` and `A_l=max(m_l*b_f,|C_l|)`, using the table's represented term bound for `b_f`. The final allowance adds `(2m+L) eta_s` for underflow across the parent partial sum, child partial sums, and regrouping. Thus zero or cancellation-sensitive totals use a count-times-represented-value bound rather than an unsafe relative comparison, while large sums also use their actual magnitudes. Counts remain bit-exact. Metadata records both dtypes, machine epsilon, and the maximum observed difference and actually applied allowance for each field and adjacent scheme pair.

Configured Q-space anchors use the same effective-edge assignment as observations. A boundary anchor belongs to the bin on its upper side, except the common final upper endpoint belongs to the final bin. Anchor output records the unique bin, configured and effective bounds and widths, count, contributing runs, point values, bootstrap intervals, validity information, and finite-bin discrepancy at every source time, selected action, and refinement level. Refinement is a bias-variance diagnostic: smaller bins reduce within-bin Q dispersion but commonly reduce counts and increase uncertainty; no monotonicity of the scientific estimates or interval widths is assumed.

Phase 3B preflight occurs after TOML has produced raw Python lists but before `QBinSpec`, any NumPy edge-array copy, graph construction, initialization, simulation, aggregation, bootstrap allocation, or output construction. It does not claim the memory already consumed by those parser lists. For scheme `l`, define `C_l=T*Bc_l*Bd_l*2`, with `S_l=R*C_l`, `O=R*T*n`, bootstrap count `B`, and processing chunk `K=min(configured_chunk,max_l C_l)`. Allocation-free Python-integer arithmetic accounts for:

- `88*sum_l S_l` bytes allocated over the sequential aggregations, with the simultaneous sufficient-statistic peak equal to `88` times the largest first or adjacent parent-plus-child stratum count; gamma-bound parent reconstruction additionally allows a conservative `112S_l` bytes for one indexed child field/count array, float64 count conversions, physical scales, gamma/error arrays, the reduced result, and NumPy comparison workspace;
- configured float64 plus effective observation-dtype edges for all schemes, even at `T=0`;
- dtype-aware Phase 3A aggregation observation/index/conversion/product work over `O`;
- retained int32 bootstrap weights (`4BR`), their int32 draw array and small row-index vector during generation, and the simultaneously live float64 weight conversion during processing;
- `445*sum_l C_l` bytes for retained counts, contributing-run counts, thirteen point arrays, lower/upper arrays, valid/invalid replicate arrays, and validity flags;
- `260*max_l C_l` bytes for the unchunked pooled-point derivation of the largest scheme;
- `280*B*K` bytes for chunked bootstrap pooled counts/sums, estimands, intermediates, and expression/quantile workspace.

The output-stage peak also includes the final scheme's sufficient statistics, retained summaries, up to 16 MiB for NumPy's chunked weight serialization, and a conservative 40 bytes per effective edge converted to JSON-ready Python floats/list slots. The weight hash consumes the existing contiguous array through a byte-cast memory view and does not make a full copy. Configured metadata edges reuse the TOML parser's existing raw lists rather than making another list copy. CSV files are streamed and no all-row table is built.

Unweighted point pooling uses direct `sum(axis=0,dtype=int64)` for counts and `sum(axis=0,dtype=float64)` for each sufficient field. It therefore creates no `R`-length ones vector or float64 weight conversion; metadata records zero pooled-point run-weight bytes. Explicit weighted pooling remains a separate path used for compatibility tests, while bootstrap chunks apply their common multiplicities directly.

The audited Phase 3B host-statistics peak is the maximum of weight generation, aggregation, parent reconstruction, pooled-point derivation, and chunked bootstrap processing with their simultaneously retained arrays. The separate authoritative Phase 2 instrumented guard continues to cover JAX simulation records and simulation working state; metadata records both budgets rather than mislabelling their separate backend/host estimates as one measured device peak. CSV rows are streamed and do not create a stratum-sized Python collection or dataframe. Separate fixed Phase 3B caps cover total per-run strata, pooled rows, anchor rows, retained weight bytes, chunk work, and the overall estimated host-statistics peak. Input configuration cannot weaken them; only the explicit recorded `--allow-expensive` override can bypass violations.

## 22. Required initial variance comparisons

The initial comparison has four separate checks:

1. **Direct ABM variance.** Estimate `D_j(q,t)` from realised chosen-coordinate ABM velocities, using narrow Q-bins as a documented approximation to exact-Q conditioning.
2. **ABM decomposition check.** Estimate `sigma_j,ABM^2` and `c_j,ABM`, and verify

   ```text
   D_j^ABM approximately equals
     alpha^2 [sigma_j,ABM^2/(n-1)
              + (n-2)c_j,ABM/(n-1)].
   ```

   At finite bin width, first verify the reward-variance identity exactly with matched moments, then account for the documented Q-dispersion/reward-Q-covariance difference between reward variance and raw realised-velocity variance.
3. **One-edge pair check.** Compare `sigma_j,pair^2` with `sigma_j,ABM^2` under matched Q-bin and action-conditioning weights. This isolates the accuracy of the pair solver's one-edge payoff law.
4. **Theory comparison.** At exact `q`, compare direct ABM variance with both the pure `c_j=0` pair prediction and the diagnostic hybrid

   ```text
   alpha^2 [sigma_j,pair^2/(n-1)
            + (n-2)c_j,ABM/(n-1)].
   ```

   For a finite bin, use the raw-moment pooled convention in Section 22.1 rather than forcing the bin-level pair covariance or Q-heterogeneity terms to zero.

This separation distinguishes error in the one-edge pair distribution from missing cross-opponent covariance and finite-bin mixing.

### 22.1 Phase 5 finite-bin convention and bounded implementation

The earlier shorthand in Sections 18 and 22 writes the exact-Q closure as `alpha^2 sigma_j,pair^2(q,t)/N`, with `N=n-1`, because distinct opponents are conditionally independent at one exact focal `q`. Phase 5 must compare finite Q bins rather than exact continuous values, so applying that shorthand after pooling would silently discard between-Q mixing. The primary implemented finite-bin convention therefore pools raw pair moments first and then applies the full finite-bin velocity identity. The exact-Q shorthand remains valid only as a local diagnostic, and the occupancy-weighted mean of local `sigma^2(q,j)` is retained separately from the pooled-bin one-edge variance.

At pair source time `t`, exact grid point `q`, and selected focal action `j`, define the selected mass

```text
a_t(q,j) = p_t(q) pi_j(q).
```

The pair solver emits bounded point/action raw sums proportional to

```text
a,
a mu,
a m2,
a mu^2,
a q_j,
a q_j^2,
a mu q_j.
```

The `a mu^2` term is the exact-Q conditional-independence distinct-opponent product. For a finite bin `B`, all seven quantities are summed over its grid points and only then divided by `sum_B a`. Consequently

```text
mu_B       = E_B[mu(q,j)]
m2_B       = E_B[m2(q,j)]
m11_B_pair = E_B[mu(q,j)^2]
sigma2_B_pair = m2_B - mu_B^2
c_B_pair      = m11_B_pair - mu_B^2.
```

Thus `c_B_pair=Var_B(mu(q,j))` can be positive even though the exact-Q conditional covariance is zero. Action conditioning is present in `a=p*pi_j`; it is not an unweighted occupancy average. The remaining pair terms are

```text
Var_B_pair(q_j) = E_B[q_j^2] - E_B[q_j]^2
Cov_B_pair(r,q_j) = E_B[mu(q,j) q_j] - mu_B E_B[q_j].
```

The four implemented finite-bin velocity estimands are

```text
V_direct_ABM = Var_B[observed alpha(r-q_j)]

V_reconstructed_ABM
  = alpha^2 [sigma2_B_ABM/N + (N-1)c_B_ABM/N
             + Var_B_ABM(q_j) - 2 Cov_B_ABM(r,q_j)]

V_pair
  = alpha^2 [sigma2_B_pair/N + (N-1)c_B_pair/N
             + Var_B_pair(q_j) - 2 Cov_B_pair(r,q_j)]

V_hybrid
  = alpha^2 [sigma2_B_pair/N + (N-1)c_B_ABM/N
             + Var_B_pair(q_j) - 2 Cov_B_pair(r,q_j)].
```

For `N=1`, both cross-opponent terms vanish and ABM `m11,c` remain undefined. The hybrid is explicitly diagnostic: it substitutes only the ABM finite-bin cross-opponent covariance and does not turn the pair closure into a standalone analytical covariance theory.

ABM records use source `Q_t`, actions, old edge states, rewards, and selected velocities from step `t`. Pair summaries use `P_t` before the transport to `P_{t+1}`. The bounded runner simulates ABM records through the largest requested time plus one, advances one pair trajectory only through the largest requested source time, and labels the retained point summaries with their explicit source times. Pair and ABM times must match exactly; shifted labels are rejected. Both systems use the same authoritative tensors, `alpha`, `tau`, seeded legacy histogram, independently constructed pair endpoints, uniform initial state law, configured/effective bin edges, and selected-action order.

The runner computes authoritative finest ABM and pair-bin raw sums once, reconstructs one configured coarser nested scheme at a time by exact addition, verifies ABM parent/child reconstruction, derives bounded comparison/bootstrap summaries, and releases that coarse sufficient state before continuing. It runs the ABM once, runs the pair trajectory once, and reuses the same ABM records, pair point summaries, and one `(bootstrap_replicate,run)` multiplicity matrix for every scheme, time, action, bin, and ABM-dependent estimand. Pair-only quantities are deterministic and receive no sampling interval. Direct ABM, reconstructed ABM, hybrid, and their discrepancies are recomputed after complete-run bootstrap pooling. The Phase 3B requirements of at least two contributing runs and at least `max(2,ceil(0.8B))` finite replicates remain unchanged; the quantile method, both thresholds and row-specific invalid replicate counts are explicit outputs. `abm_reconstruction_defined` means the reconstruction is finite for an occupied stratum; it is not a numerical closure-tolerance claim.

The small runner retains no full pair-density history. Lowering and compilation create a runtime-only bundle containing the exact compiled callable, its analyzed memory fields, abstract arguments, static values, and backend/platform/device/x64 signature. After the static and compiled gates accept, that same callable is invoked once; the original jitted Python function is not called again. Immediately before invocation, a signature is independently rebuilt from the actual pair mass, grid leaves, `alpha`, `tau`, source-slot array and current runtime. It must match pair shape/dtype, grid arguments, step and summary counts, requested source times, complete slot map, chunk size, both validation tolerances, backend/platform/device identity and x64 state. For every `P_t` from zero through the largest requested time the scan records the authoritative Phase 4 lean diagnostics, records a `15M` point summary only when requested, then transports synchronously. A separate authoritative host validator checks the returned shapes, every diagnostic, final mass, symmetry and destinations; it never executes an alternative scan or step. The final density is returned; intermediate densities are not.

At `L` requested source times and `M` focal grid points, source-summary storage is `15LMb` on device and `15LMb+2Mb+8L` on host. The combined scan returns the complete `(T+1)(11b+3)` diagnostic trajectory and `T` destination-validity booleans; device and host copies are both counted at transfer/validation and the device copies remain in retained pair outputs where applicable. If `C_l=L*Bc_l*Bd_l*2`, one pair scheme needs `56C_l+8(C_l/2)` bytes; only the authoritative finest plus the largest one-at-a-time coarse reconstruction coexist. Retained final comparisons are conservatively `512*sum_l C_l`, and eight ABM-dependent bootstrap arrays use `64B*max_l C_l`. Comparison and anchor dictionaries are streamed in deterministic order: a first non-retaining pass enforces a 16 KiB live-object and 8 KiB ASCII-record bound and accumulates smoke summaries, then output begins and a second pass writes one row at a time. No row list is retained. Maximum normalized-schema rows produced through the real iterators measured 14,579 live bytes/4,450 characters for the 91-column comparison row and 11,209 bytes/4,022 characters for the 68-column anchor row.

The Phase 5 256 MiB guard is the maximum of explicit simultaneous lifetime peaks, not a sum of unrelated maxima. The phases are normalized configuration/scientific scheme validation, shape lowering/compilation, ABM simulation, pair execution with retained ABM data, pair device-to-host transfer and result validation, finest-plus-one-coarse reconstruction, aggregation, pooled point derivation, bootstrap chunk processing, bounded anchor accumulation, one streamed row, and Phase 5 serialization. Within output, JSON encoding, CSV writing, bootstrap-weight archive writing, and chunked metadata writing are alternative subpeaks. The bounded encoded metadata string remains live during the latter three and is counted there; their mutually exclusive temporary buffers are not added together. Metadata records every component, all serialization subpeaks, the static pair-kernel allowance, analyzed compiled device/host requirements, retained cross-phase arrays, all phase totals and the determining phase. Phase 4 kernel formulas are reused only for arrays created by this executable; Phase 4 runner-only serialization, source hashes and diagnostic dictionaries are excluded. Startup first parses the exact bounded schema, inspects list sizes with Python integers, and enforces raw/static caps. It then constructs `QBinSpec` arrays and validates configured/effective edges, observation-dtype collapse, configured/effective nesting and every anchor. Only after those scientific checks may it create the Q grid and lower the exact Phase 5 scan. Fail-closed compiled analysis and a repeated global check precede histogram sampling, pair/ABM allocation and simulation. The constructed grid must reproduce raw histogram counts. Only recorded `--allow-expensive` overrides resource failures; scientific matching and numerical validity checks remain mandatory. An unavailable or mismatched executable analysis is recorded as such, never as a pass.

### 22.2 Exact separable JAX transport

The production-oriented nearest-grid operator is algebraically identical to the Phase 4 flat scatter. With `M=G^2`, source state `s`, endpoint actions `(a,b)`, policy vectors `u_a(i)=pi_a(q_i)` and `v_b(k)=pi_b(q_k)`, and action-specific maps `F_a,F_b`, one branch is

```text
P'_T(s,a,b)[F_a(i),F_b(k)] += P_s[i,k] u_a(i) v_b(k).
```

The implemented separable kernel loops over the eight old-state/action branches and over static row/column blocks. It gathers `F_a` and `u_a` for one row block, `F_b` and `v_b` for one column block, forms one weighted `B_r x B_c` tile, and scatter-adds it into the persistent `(2,M,M)` output. This is the blocked form of row transport followed by column transport; it avoids a full `M x M` row intermediate as well as flat `D x 4` branch mass/destination arrays. Source policies, conditional rewards, velocities, endpoint-specific maps, transition states and exact legacy projection are unchanged. Branch processing is sequential and stored probability masses are never normalized, clipped, interpolated or multiplied by grid spacing.

For one scalar width `b`, `L` requested summaries and `T` steps, bounded output is `15LMb+(T+1)(11b+3)+T` bytes. The static resource model names input/output densities, grid and histogram initialization arguments, policies/moments/maps, one source and weighted tile plus masks/indices, and bounded outputs. It adds two density-sized device scatter/lowering allowances because static preflight does not assume compiler fusion. The modeled coexisting numerical host peak contains float64 histogram/grid construction and returned summaries/diagnostics; a reduced validation mode additionally returns its final density. A separately named heuristic host staging reserve of two density widths is retained as a conservative planning threshold. It is not two observed host pair arrays and does not bound excluded compiler RSS/code cache, Python/library RSS or backend allocator overhead. Benchmark host preflight also permits one at-most-8-MiB lowered-HLO string at four bytes per character and one bounded encoded hash chunk; the text is chunk-hashed and released before execution.

Device initialization takes the authoritative normalized flat `M`-entry one-agent histogram and controlled probabilities `p=(0.5,0.5)` and constructs `P_0(s,i,k)=p_s h_i h_k` inside each combined compiled device program. The benchmark has no standalone initializer executable. The production-oriented scan returns only bounded summaries, diagnostics and destination validity; the final density remains a device-internal carry. A reduced combined validation object returns the final density solely for direct parity. Tests establish total/state mass, both endpoint marginals, independence and exchange symmetry.

All flat/separable and full/bounded objects, plus both reduction microbenchmarks, are lowered and compiled before their own device input construction or invocation. The authoritative bundle validator re-reads `memory_analysis()` from the exact retained compiled interface and requires agreement with complete internally consistent device/host statistics. Each factory-created runtime-only bundle retains nonempty abstract arguments, all static values, complete signature/environment/device/version/x64 identity, signature digest and whole-bundle integrity digest. Immediately before every warm-up or timed invocation, the signature is rebuilt internally from the actual argument pytree, weak types, actual tolerance, static context and runtime state and must match exactly; execution calls only the retained callable. Missing, malformed, inconsistent or identity-mismatched analysis is non-overridable and precedes pair allocation. Warm-up synchronizes every result leaf. Measured repetitions alternate kernel order and retain bounded individual samples, positions, order, median, min, max and MAD separately from compilation.

Scatter-add reduction order for many-to-one destinations can differ by backend. Scientific equivalence therefore means agreement within dtype/backend tolerances, not bitwise cross-device equality. A bounded linear-branch comparison compiles and times both scatter and sorted segment reduction; sorted segment is not selected because its sort order vector and associated temporary storage increased measured compiled memory. Any later adoption requires new full-operator memory and numerical measurements.

The feasibility target is fixed at `G=131`, `M=17161`, `D=588999842`, so one density is 2,355,999,368 float32 bytes or 4,711,998,736 float64 bytes. The allocation-free record separates exact density arithmetic, the static kernel allowance, an empirical projection formed from the worst measured deliberately small CPU compiled-bytes/density ratio, and the 25% planning threshold. The empirical values are not formal bounds or GPU predictions. No full shape is constructed or lowered.

Production execution is a separate non-overridable decision. It requires the exact separable runtime bundle, an internally rebuilt matching invocation signature, complete live analysis, GPU backend and fresh usable-capacity evidence matched to the execution device. A bounded optional NVIDIA provider queries `nvidia-smi` and matches stable GPU UUID, MIG UUID or normalized PCI identity. A numeric visibility token is accepted only through a trusted injected CUDA-runtime mapping from JAX's logical visible ordinal to stable identity; numeric CUDA and `nvidia-smi` indices are never equated. It also requires an explicit JAX allocator/preallocation policy; nominal total memory alone is insufficient. Usable capacity is current free physical memory further limited by allocator-available bytes where known. The internal UTC clock fixes the maximum age at 60 seconds, rejects evidence more than one second in the future, and admission is rechecked adjacent to execution. Provider failure, ambiguous/mismatched identity, incomplete fields or `required compiled bytes with margin > verified usable bytes` fails closed. `--allow-expensive` can override documented bounded-development static caps only, never bundle identity/completeness, analysis, invocation matching, backend, capacity evidence or scientific diagnostics.

## 23. Primary outputs, uncertainty, and scaling checks

Primary outputs are:

- Q-resolved `D_j`, `sigma_j^2`, and `c_j` estimates at selected times;
- heatmaps or tables restricted to sufficiently populated two-dimensional Q-bins;
- occupancy-weighted time-series summaries with matched theory/ABM Q weighting;
- direct, decomposed, pure-pair, and hybrid predictions kept as distinct named series;
- uncertainty intervals for which independent simulation runs are the primary sampling units;
- retained focal-agent identifiers for repeated-measure diagnostics, without treating agents from one complete-network run as independent replicates;
- scaling checks over population size `n` and learning rate `alpha`.

Under the `c_j=0` closure, variance scales as `1/(n-1)`. At a fixed distributional snapshot all conditional velocity variances scale as `alpha^2`. Changing `alpha` throughout an evolution also changes later Q/state distributions, so a full trajectory comparison across learning rates requires separate ABM simulations and pair evolutions rather than a post-hoc rescaling.

The following are secondary or future objects and are not primary targets for the first version:

- unconditional coordinate-update variance without conditioning on focal action;
- covariance between `C` and `D` velocity coordinates;
- an analytical triplet-density closure for `c_j`;
- long-time diffusion or Fokker-Planck approximations;
- local-regression or kernel estimators unless binning diagnostics show they are needed.

The immediate scientific question is whether `c_j` is negligible and, when it is not, whether adding the empirically measured `c_j` explains the discrepancy between direct ABM variance and the conditional-independence pair prediction.

## 24. Remaining questions not answerable from the available files

1. Which random seed and exact empirical Beta histogram, if any, generated the theoretical dots in Figure 1(b)?
2. Was the released script's nearest-grid pushforward the exact numerical method used for the paper, and what convergence checks were performed? The main paper defers derivation details to supplementary material that is not present locally.
3. Was the paper's own ABM initial state probability exactly one half, and were different initial edge states independent? This remains historically unclear, but it does not block the project: the controlled baseline is fixed to independent `1/2` state draws.
4. What minimum bin count, bin-refinement schedule, and run-level uncertainty procedure will be adequate for the available simulation budget?
5. What boundary behaviour should apply for future payoff matrices, learning rates, or Q-ranges that violate the current convex-range invariant?
6. After exact legacy parity, should the production pair solver retain nearest-grid rounding or adopt conservative interpolation/convergence refinement?
7. What GPU hardware, memory budget, precision requirement, and reproducibility tolerance will define the target scale?
## GPU pilot operational contract

The guarded Lambda Cloud pilot does not change the scientific model, pair chronology, projection, initialization, observables, or variance definitions. It is an execution envelope for the exact bounded separable source-summary executable already validated against the NumPy and flat-scatter oracles.

The numerical stage order is fixed: strict allocation-free configuration/resource inspection; clean matching doctor and predecessor evidence; abstract lowering; exact compilation; live complete executable analysis; fresh CUDA-driver-stable device/capacity admission with configured safety margin; device input construction; a second fresh capacity check adjacent to invocation; then execution and the existing scientific diagnostics. The pair mass is constructed from the one-agent histogram inside the analyzed executable. No host pair mass or full final-density return is permitted.

The pilot one-agent law is a recorded seeded legacy scaled-Beta histogram with the original draw order and local RNG. Its count storage and draw count are guarded before construction. `G=131` uses the original `[-0.1,1.2]` grid at spacing `0.01`; reduced pilot grids use explicit decimal-aligned ranges containing `[-0.1,1.2]` and are performance/parity cases rather than convergence evidence.

The pilot ladder is `G=3,5,9` flat/separable parity, `G=17,33` separable, `G=65` separable, optional independently enabled `G=97`, `G=131` analysis only, and at most one separately confirmed `G=131` step. A full-grid multi-step run is outside this milestone. Full-grid confirmation, prior-artifact identity/freshness, executable identity/completeness, GPU backend and stable device matching, usable capacity, the one-step limit, and scientific validity are non-overridable.

Numeric CUDA visibility is mapped through the initialized CUDA Driver API to UUID and PCI identity; numeric tokens are never treated as `nvidia-smi` indices. MIG visibility is recognized but fails capacity admission until slice-level memory evidence can be matched safely. Capacity evidence is immutable at no more than 60 seconds old and is a conservative admission observation, not a guarantee. Allocator policy is explicit before JAX import and is part of environment provenance.

Pilot costs are estimates derived only from measured elapsed wall time and an explicitly acknowledged user-supplied hourly price. They are not provider billing data. The deployment scripts neither contact Lambda Cloud nor create, access, or terminate instances.

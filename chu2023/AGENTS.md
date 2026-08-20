# Local development instructions

- Work only in this `chu2023` subproject; do not initialise a nested Git repository.
- Treat `MODEL_SPEC.md` and `PLAN.md` as the scientific source of truth.
- Never modify `case2_1.py` or `pair-approx_multi-agent_stochastic_games.pdf`.
- Do not restore the intentionally deleted `case2_1_jax.py` or the other PDF.
- Keep payoff and transition tensors authoritative in `src/chu_pair/model.py`.
- Pair arrays store probability mass with axes `(q1_C, q1_D, state, q2_C, q2_D)`; do not multiply sums by grid spacing.
- ABM Q-updates are continuous. Nearest-grid projection is only for pair-density transport.
- Use small grids in ordinary tests. Never run the original full-grid script as a test.
- The Phase 2 ABM uses one packed state per undirected edge and one action per agent per step.
- Keep scan records agent-sized; edge payoffs may appear only in one-step debug records.
- Phase 3A/3B may retain selected-action `S1`/`S2`, per-run sufficient sums, and bounded bootstrap/refinement summaries only at agent/run/time/bin scale; never retain time-indexed edge-payoff histories.
- Phase 3B uncertainty must resample complete independent runs with one common run-weight matrix across times, bins, actions, estimands, and refinement levels.
- The Phase 4 JAX pair solver uses internal `(state,M,M)` probability-mass arrays, exact legacy projection, bounded source chunks, and lean scan diagnostics; preserve NumPy parity.
- Phase 5 compares pair and ABM quantities only at matched source times, bins, selected actions, model parameters, and seeded legacy initialization. Pool pair raw moments with weights `p(q) pi_j(q)` before nonlinear finite-bin formulas; exact-Q conditional independence does not imply zero finite-bin pair covariance.
- The Phase 5 hybrid replaces only the pair finite-bin distinct-opponent covariance with the matched ABM covariance. Complete independent ABM runs remain the bootstrap units; pair-only quantities are deterministic.
- Phase 5 pair summaries and diagnostics must come from the exact compiled `lax.scan` object whose memory analysis passed; validate an independently rebuilt invocation signature before calling it. Retain the finest sufficient statistics plus at most one coarse reconstruction, and stream wide CSV rows without a full row collection (16 KiB live object and 8 KiB ASCII record bounds).
- Phase 5 resource metadata uses phase-specific lifetime maxima, including `(T+1)` diagnostic rows and `T` destination-validity booleans on device and host; do not import Phase 4 runner-only output buffers.
- The exact separable pair kernel is a tiled row/column-map pushforward. Keep the flat scatter as the default validation oracle, process branches sequentially, and never materialize a `D x 4` branch array or a full density history.
- Production-oriented pair initialization transfers the one-agent histogram and constructs the independent ordered pair mass on device. Its bounded scan may return source summaries and diagnostics, but not a full final density to the host.
- Compile and completely analyze every separable benchmark executable, including reduction microbenchmarks, before invoking any of them. Central validation must re-read `memory_analysis()` from the retained callable; execute it only after a signature rebuilt internally from the actual arguments and tolerance matches.
- Production GPU capacity requires evidence no older than the immutable 60-second cap and matched by UUID, MIG UUID, normalized PCI identity or a trusted CUDA-runtime ordinal mapping. Never compare numeric CUDA visibility tokens with `nvidia-smi` indices. `--allow-expensive` cannot override executable identity, analysis completeness, backend, device matching or capacity evidence.
- Keep ordinary separable tests and benchmarks at `G<=17`. The `G=131` calculation is allocation-free feasibility only until an exact executable analysis, compatible GPU and known device capacity pass the production preflight.
- Do not add counterfactual unselected-action instrumentation, focal-agent resampling, production/full-grid runs, interpolation, production inference, or unvalidated GPU claims yet.

Commands:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pytest -q tests/test_pair_transport.py tests/test_pair_moments.py
python -m pytest -q tests/test_abm_graph.py tests/test_abm_one_step.py tests/test_abm_sampling.py tests/test_abm_simulation.py tests/test_abm_runner.py
python -m pytest -q tests/test_abm_variance.py tests/test_abm_variance_runner.py
python -m pytest -q tests/test_abm_uncertainty.py tests/test_abm_uncertainty_runner.py
python -m pytest -q tests/test_pair_jax.py tests/test_pair_jax_runner.py
python -m pytest -q tests/test_velocity_variance.py tests/test_velocity_variance_runner.py
python -m pytest -q tests/test_pair_separable.py tests/test_pair_separable_runner.py
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q
python experiments/run_abm_baseline.py --config configs/abm_baseline_small.toml
python experiments/run_abm_variance_diagnostic.py --config configs/abm_variance_diagnostic_small.toml
python experiments/run_abm_uncertainty_diagnostic.py --config configs/abm_uncertainty_smoke.toml
python experiments/run_pair_jax_small.py --config configs/pair_jax_small.toml
python experiments/run_velocity_variance_comparison.py --config configs/velocity_variance_comparison_small.toml
python experiments/run_pair_separable_benchmark.py --config configs/pair_separable_benchmark_small.toml
```

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
- Do not add Phase 3 S1/S2 covariance statistics, Q bins, pair-JAX transport, or GPU benchmarks yet.

Commands:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pytest -q tests/test_pair_transport.py tests/test_pair_moments.py
python -m pytest -q tests/test_abm_graph.py tests/test_abm_one_step.py tests/test_abm_sampling.py tests/test_abm_simulation.py tests/test_abm_runner.py
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q
python experiments/run_abm_baseline.py --config configs/abm_baseline_small.toml
```

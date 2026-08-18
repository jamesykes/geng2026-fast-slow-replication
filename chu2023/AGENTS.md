# Local development instructions

- Work only in this `chu2023` subproject; do not initialise a nested Git repository.
- Treat `MODEL_SPEC.md` and `PLAN.md` as the scientific source of truth.
- Never modify `case2_1.py` or `pair-approx_multi-agent_stochastic_games.pdf`.
- Do not restore the intentionally deleted `case2_1_jax.py` or the other PDF.
- Keep payoff and transition tensors authoritative in `src/chu_pair/model.py`.
- Pair arrays store probability mass with axes `(q1_C, q1_D, state, q2_C, q2_D)`; do not multiply sums by grid spacing.
- ABM Q-updates are continuous. Nearest-grid projection is only for pair-density transport.
- Use small grids in ordinary tests. Never run the original full-grid script as a test.
- Do not add JAX/GPU or ABM implementation during the Phase 1 milestone.

Commands:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pytest -q tests/test_pair_transport.py tests/test_pair_moments.py
```


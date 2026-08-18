# Chu et al. pair-approximation numerics

This self-contained subproject preserves the original `case2_1.py` calculation and develops independently tested numerical components for the model in Chu et al. The current milestone contains shared model semantics and a deliberately readable NumPy pair-mass reference for small grids. JAX, the finite-population ABM, GPU transport, Q-bin estimators, and variance experiments are later phases.

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

Generated experiment data belongs under `outputs/`, which is ignored. GPU/JAX installation is intentionally deferred until the target machine is known.


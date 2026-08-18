# Geng 2026 Fast/Slow Replication

Repository layout:

- `replication/`: our replication scripts, reconstructed code, generated outputs, and notes.
- `authors_original/`: place the full original repository supplied by the paper authors here.
- `chu2023/`: Chu et al. 2023 pair-approximation code and reference papers.

The author-supplied code is kept separate from the replication work to make provenance and future comparisons explicit.

Useful entry points:

- `chu2023/case2_1_jax.py`: JAX/GPU port of the Chu et al. case 2.1 script.
- `replication/run_yuan_regular_graph_sweep.py`: complete-to-regular graph degree sweep with state-independent and state-dependent transition rules.
- `replication/run_geng_fast_slow_state_dependent.py`: Geng-style fast-slow comparison with explicitly state-dependent transition probabilities.
- `replication/outputs/local_checks/`: small CPU sanity-check plots and CSV/NPZ outputs.

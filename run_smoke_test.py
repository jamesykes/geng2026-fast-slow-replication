from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL_CODE_DIR = REPO_ROOT / "original_paper_code"

if str(ORIGINAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE_DIR))

from sim_graph import compute_average_s as compute_average_s_sim
from sim_graph import sim_graph
from theory_graph import compute_average_s as compute_average_s_theory
from utils.graph_utils import create_lattice_graph


def main() -> None:
    N = 100
    K = 2
    M = 2
    alpha = 0.001
    beta = 1.0
    gamma = 0.8
    b1 = 5.0
    b2 = 1.2
    c1 = 0.5
    c2 = 0.5
    p1 = 0.8
    p2 = 0.3
    time_steps = 100
    num_reps = 1

    _, A = create_lattice_graph(N=N, dim=2, periodic=True)

    payoff_matrices = jnp.array(
        [
            [[b1 - c1, -c1], [b1, 0.0]],
            [[b2 - c2, -c2], [b2, 0.0]],
        ],
        dtype=jnp.float32,
    )

    T = np.full((K, M, M, K), fill_value=0.0, dtype=np.float32)
    T[:, :, :, 0] = p2
    T[:, :, :, 1] = 1.0 - p2
    T[:, 0, 0, 0] = p1
    T[:, 0, 0, 1] = 1.0 - p1
    T = jnp.array(T)

    Q_history_all, X_history_all, s_history_all = sim_graph(
        N=N,
        K=K,
        M=M,
        time_steps=time_steps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        payoff_matrices=payoff_matrices,
        T=T,
        A=A,
        num_reps=num_reps,
    )

    print(f"Q_history_all shape: {Q_history_all.shape}")
    print(f"X_history_all shape: {X_history_all.shape}")
    print(f"s_history_all shape: {s_history_all.shape}")

    avg_s = compute_average_s_sim(s_history_all, A, K)
    avg_s_theory = compute_average_s_theory(s_history_all, A, K)
    print(f"average_s shape: {avg_s.shape}")
    print(f"average_s first: {avg_s[:, 0, :]}")
    print(f"average_s last: {avg_s[:, -1, :]}")
    print(f"theory_graph.compute_average_s matches: {bool(jnp.allclose(avg_s, avg_s_theory))}")

    mean_coop_state_0_initial = float(X_history_all[:, 0, :, 0, 0].mean())
    mean_coop_state_0_final = float(X_history_all[:, -1, :, 0, 0].mean())
    mean_coop_state_1_initial = float(X_history_all[:, 0, :, 1, 0].mean())
    mean_coop_state_1_final = float(X_history_all[:, -1, :, 1, 0].mean())

    print(f"state 0 cooperation mean, initial: {mean_coop_state_0_initial:.6f}")
    print(f"state 0 cooperation mean, final: {mean_coop_state_0_final:.6f}")
    print(f"state 1 cooperation mean, initial: {mean_coop_state_1_initial:.6f}")
    print(f"state 1 cooperation mean, final: {mean_coop_state_1_final:.6f}")


if __name__ == "__main__":
    main()

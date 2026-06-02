import os
import jax
import jax.numpy as jnp
import numpy as np

from sim import sim_graph
from utils import create_random_regular_graph, create_lattice_graph

os.makedirs("data/fig1", exist_ok=True)
os.makedirs("data/fig2", exist_ok=True)
os.makedirs("data/fig3", exist_ok=True)
os.makedirs("data/fig_SI", exist_ok=True)


def _default_payoff(b1, b2, c, K=2, M=2):
    payoff_matrices = jnp.zeros((K, M, M))
    payoff_matrices = payoff_matrices.at[0].set(jnp.array([[b1 - c, -c], [b1, 0.0]]))
    payoff_matrices = payoff_matrices.at[1].set(jnp.array([[b2 - c, -c], [b2, 0.0]]))
    return payoff_matrices


def _default_T(pc, pr, K=2, M=2):
    T = jnp.zeros((K, M, M, K))
    T = T.at[:, 0, 0, :].set([pr, 1 - pr])
    T = T.at[:, 0, 1, :].set([1 - pc, pc])
    T = T.at[:, 1, 0, :].set([1 - pc, pc])
    T = T.at[:, 1, 1, :].set([1 - pc, pc])
    assert jnp.allclose(jnp.sum(T, axis=-1), 1.0), (
        "The sum over the last dimension of T must be 1"
    )
    return T



def run_fig1():
    N, K, M = 100, 2, 2
    time_steps = 20000
    alpha, beta, gamma = 0.001, 1.0, 0.8
    b1, b2, c = 5.0, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)
    T = _default_T(0.8, 0.7, K, M) # typo: p1 should be 0.7 and p2 should be 0.2 in the caption of Figure 3

    G, A = create_lattice_graph(N, dim=2, periodic=True)

    Q_avg_all, X_avg_all, s_prop_all = sim_graph(
        N, K, M, time_steps, alpha, beta, gamma,
        payoff_matrices, T, A, num_reps=10, init_key=42,
        s_init_prob=jnp.array([0.5, 0.5])
    )

    np.savez_compressed(
        "data/fig1/sim_graph.npz",
        s_mean=jax.device_get(s_prop_all.mean(axis=0)),
        Q_mean=jax.device_get(Q_avg_all.mean(axis=0)),
        X_mean=jax.device_get(X_avg_all.mean(axis=0)),
        s_std=jax.device_get(s_prop_all.std(axis=0)),
        Q_std=jax.device_get(Q_avg_all.std(axis=0)),
        X_std=jax.device_get(X_avg_all.std(axis=0)),
    )



def run_fig2_b1_1_2_gamma():
    N, K, M = 100, 2, 2
    time_steps = 40000
    alpha, beta = 0.001, 1.0
    b1, b2, c = 1.2, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)
    T = _default_T(0.8, 0.7, K, M)

    X_mean_batch_dict = {}
    for gamma in (0.2, 0.5, 0.8):
        G, A = create_lattice_graph(N, dim=2, periodic=True)
        _, X_avg_all, _ = sim_graph(
            N, K, M, time_steps, alpha, beta, gamma=gamma,
            payoff_matrices=payoff_matrices, T=T, A=A,
            num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
        )
        X_mean = jax.device_get(X_avg_all.mean(axis=0))
        X_mean_batch_dict[f"X_mean_{int(gamma*10000)}"] = X_mean

    np.savez_compressed(
        "data/fig2/sim_graph_b1_1_2_gamma.npz",
        **X_mean_batch_dict,
    )



def run_fig2_b1_5_gamma():
    N, K, M = 100, 2, 2
    time_steps = 40000
    alpha, beta = 0.001, 1.0
    b1, b2, c = 5.0, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)
    T = _default_T(0.8, 0.7, K, M)

    X_mean_batch_dict = {}
    for gamma in (0.2, 0.5, 0.8):
        G, A = create_lattice_graph(N, dim=2, periodic=True)
        _, X_avg_all, _ = sim_graph(
            N, K, M, time_steps, alpha, beta, gamma=gamma,
            payoff_matrices=payoff_matrices, T=T, A=A,
            num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
        )
        X_mean = jax.device_get(X_avg_all.mean(axis=0))
        X_mean_batch_dict[f"X_mean_{int(gamma*10000)}"] = X_mean

    np.savez_compressed(
        "data/fig2/sim_graph_b1_5_gamma.npz",
        **X_mean_batch_dict,
    )



def run_fig3_degree():
    N, K, M = 100, 2, 2
    alpha, beta, gamma = 0.001, 1.0, 0.8
    b1, b2, c = 5.0, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)
    T = _default_T(0.8, 0.7, K, M)

    X_mean_batch_dict = {}
    for k in (2, 3, 6, N - 1):
        G, A = create_random_regular_graph(N, k)
        _, X_avg_all, _ = sim_graph(
            N, K, M, time_steps=20000, alpha=alpha, beta=beta, gamma=gamma,
            payoff_matrices=payoff_matrices, T=T, A=A,
            num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
        )
        X_mean = jax.device_get(X_avg_all.mean(axis=0))
        X_mean_batch_dict[f"X_mean_{k}"] = X_mean

    np.savez_compressed(
        "data/fig3/sim_graph_degree.npz",
        **X_mean_batch_dict,
    )



def run_fig3_alpha():
    N, K, M = 100, 2, 2
    beta, gamma = 1.0, 0.8
    b1, b2, c = 5.0, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)
    T = _default_T(0.8, 0.7, K, M)

    X_mean_batch_dict = {}
    for alpha in (0.0002, 0.0005, 0.001):
        G, A = create_lattice_graph(N, dim=2, periodic=True)
        _, X_avg_all, _ = sim_graph(
            N, K, M, time_steps=30000, alpha=alpha, beta=beta, gamma=gamma,
            payoff_matrices=payoff_matrices, T=T, A=A,
            num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
        )
        X_mean = jax.device_get(X_avg_all.mean(axis=0))
        X_mean_batch_dict[f"X_mean_{int(alpha*10000)}"] = X_mean

    np.savez_compressed(
        "data/fig3/sim_graph_alpha.npz",
        **X_mean_batch_dict,
    )



def run_SI_beta():
    N, K, M = 100, 2, 2
    time_steps = 20000
    alpha, gamma = 0.001, 0.8
    b1, b2, c = 5.0, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)
    T = _default_T(0.8, 0.7, K, M)

    X_mean_batch_dict = {}
    for beta in (1.0, 2.0, 5.0):
        G, A = create_lattice_graph(N, dim=2, periodic=True)
        _, X_avg_all, _ = sim_graph(
            N, K, M, time_steps, alpha=alpha, beta=beta, gamma=gamma,
            payoff_matrices=payoff_matrices, T=T, A=A,
            num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
        )
        X_mean = jax.device_get(X_avg_all.mean(axis=0))
        X_mean_batch_dict[f"X_mean_{int(beta)}"] = X_mean

    np.savez_compressed(
        "data/fig_SI/sim_graph_beta.npz",
        **X_mean_batch_dict,
    )



def run_SI_transition():
    N, K, M = 100, 2, 2
    time_steps = 200000
    alpha, beta, gamma = 0.0001, 1.0, 0.8
    b1, b2, c = 5.0, 1.2, 0.5

    payoff_matrices = _default_payoff(b1, b2, c, K, M)

    X_mean_batch_dict = {}
    for pc, pr in ((0.8, 0.7), (0.5, 0.5), (0.2, 0.3)):
        T = _default_T(pc, pr, K, M)
        G, A = create_lattice_graph(N, dim=2, periodic=True)
        _, X_avg_all, _ = sim_graph(
            N, K, M, time_steps, alpha=alpha, beta=beta, gamma=gamma,
            payoff_matrices=payoff_matrices, T=T, A=A,
            num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
        )
        X_mean = jax.device_get(X_avg_all.mean(axis=0))
        X_mean_batch_dict[f"X_mean_{int(pc*10)}_{int(pr*10)}"] = X_mean

    np.savez_compressed(
        "data/fig_SI/sim_graph_transition.npz",
        **X_mean_batch_dict,
    )


if __name__ == "__main__":
    run_fig1(); print("Finished running fig1")
    run_fig2_b1_1_2_gamma(); print("Finished running fig2_b1_1_2_gamma")
    run_fig2_b1_5_gamma(); print("Finished running fig2_b1_5_gamma")
    run_fig3_degree(); print("Finished running fig3_degree")
    run_fig3_alpha(); print("Finished running fig3_alpha")
    run_SI_beta(); print("Finished running SI_beta")
    run_SI_transition(); print("Finished running SI_transition")

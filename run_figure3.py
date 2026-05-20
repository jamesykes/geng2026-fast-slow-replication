from __future__ import annotations

import argparse
import os
import pathlib
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL_CODE_DIR = REPO_ROOT / "original_paper_code"
OUTPUT_DIR = REPO_ROOT / "outputs"
MPLCONFIG_DIR = OUTPUT_DIR / ".mplconfig"

if str(ORIGINAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE_DIR))

from sim_graph import (  # noqa: E402
    choose_actions,
    compute_average_s,
    compute_average_td,
    get_td_errors,
    init_Q_s,
    sample_next_states,
    softmax as sim_softmax,
    update_Q_values,
)
from theory_graph import (  # noqa: E402
    batched_compute_mu_values,
    batched_ij_stationary_dist_func,
    softmax as theory_softmax,
)
from utils.graph_utils import create_lattice_graph  # noqa: E402


def build_figure3_parameters() -> dict[str, float | int]:
    return {
        "N": 100,
        "K": 2,
        "M": 2,
        "alpha": 0.001,
        "beta": 1.0,
        "gamma": 0.8,
        "b1": 5.0,
        "b2": 1.2,
        "c1": 0.5,
        "c2": 0.5,
        "p1": 0.8,
        "p2": 0.3,
    }


def build_payoffs_and_transitions(params: dict[str, float | int]) -> tuple[jnp.ndarray, jnp.ndarray]:
    K = int(params["K"])
    M = int(params["M"])
    b1 = float(params["b1"])
    b2 = float(params["b2"])
    c1 = float(params["c1"])
    c2 = float(params["c2"])
    p1 = float(params["p1"])
    p2 = float(params["p2"])

    # This axis order is the one consistent with the published Figure 3 curves
    # when used with the original update code's payoff_matrices[s, a_i, a_j]
    # indexing.
    payoff_matrices = jnp.array(
        [
            [[b1 - c1, b1], [-c1, 0.0]],
            [[b2 - c2, b2], [-c2, 0.0]],
        ],
        dtype=jnp.float32,
    )

    T = np.zeros((K, M, M, K), dtype=np.float32)
    T[:, :, :, 0] = p2
    T[:, :, :, 1] = 1.0 - p2
    T[:, 0, 0, 0] = p1
    T[:, 0, 0, 1] = 1.0 - p1
    return payoff_matrices, jnp.array(T)


def initialize_replicates(
    num_reps: int,
    init_key: int,
    N: int,
    K: int,
    M: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    keys = []
    Q_values = []
    s_values = []
    for rep in range(num_reps):
        key = jax.random.PRNGKey(init_key + rep)
        key, key_Q, key_s = jax.random.split(key, 3)
        Q, s = init_Q_s(key_Q, key_s, N, K, M, jnp.array([0.5, 0.5]))
        keys.append(key)
        Q_values.append(Q)
        s_values.append(s)
    return jnp.array(keys), jnp.array(Q_values), jnp.array(s_values)


@partial(jax.jit, static_argnames=("K",))
def state_fractions_for_reps(s_values: jnp.ndarray, A: jnp.ndarray, K: int) -> jnp.ndarray:
    connected = A.astype(bool)

    def one_rep(s: jnp.ndarray) -> jnp.ndarray:
        counts = jnp.array([jnp.sum((s == k) & connected) for k in range(K)])
        return counts / jnp.sum(connected)

    return jax.vmap(one_rep)(s_values)


@jax.jit
def policy_cooperation_for_reps(X_values: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.mean(X_values[:, :, 0, 0], axis=1),
            jnp.mean(X_values[:, :, 1, 0], axis=1),
        ],
        axis=-1,
    )


@partial(jax.jit, static_argnames=("time_steps", "K", "M"))
def run_simulation_summary(
    keys: jnp.ndarray,
    Q_values: jnp.ndarray,
    s_values: jnp.ndarray,
    A: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    T: jnp.ndarray,
    time_steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    K: int,
    M: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def one_rep_step(carry):
        key, Q, s = carry
        key, key_a, key_s = jax.random.split(key, 3)
        a = choose_actions(key_a, Q, beta)
        s_next = sample_next_states(key_s, s, a, T)
        td = get_td_errors(payoff_matrices, s, a, Q, s_next, gamma)
        avg_td, den_bool = compute_average_td(td, s, A, K)
        Q_next = update_Q_values(Q, avg_td, den_bool, a, M, alpha)
        return key, Q_next, s_next

    step_reps = jax.vmap(one_rep_step)

    def summarize(Q: jnp.ndarray, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        X = jax.vmap(sim_softmax, in_axes=(0, None))(Q, beta)
        return state_fractions_for_reps(s, A, K), policy_cooperation_for_reps(X)

    initial_summary = summarize(Q_values, s_values)

    def scan_step(carry, _):
        next_carry = step_reps(carry)
        _, Q_next, s_next = next_carry
        return next_carry, summarize(Q_next, s_next)

    _, summaries = jax.lax.scan(scan_step, (keys, Q_values, s_values), None, length=time_steps)
    state_history = jnp.concatenate([initial_summary[0][None, ...], summaries[0]], axis=0)
    policy_history = jnp.concatenate([initial_summary[1][None, ...], summaries[1]], axis=0)
    return state_history, policy_history


@partial(jax.jit, static_argnames=("time_steps", "k"))
def run_theory_summary(
    Q_values: jnp.ndarray,
    A: jnp.ndarray,
    T: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    time_steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    neighbor_idx = jnp.argsort(-A, axis=1)[:, :k]

    def one_rep_summary(Q: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        X = theory_softmax(Q, beta)
        X_neighbors = X[neighbor_idx]
        stationary_dist = batched_ij_stationary_dist_func(X, X_neighbors, T)
        state_mean = jnp.mean(stationary_dist, axis=(0, 1))
        policy_mean = jnp.array([jnp.mean(X[:, 0, 0]), jnp.mean(X[:, 1, 0])])
        return state_mean, policy_mean

    def one_rep_step(Q: jnp.ndarray) -> jnp.ndarray:
        X = theory_softmax(Q, beta)
        X_neighbors = X[neighbor_idx]
        stationary_dist = batched_ij_stationary_dist_func(X, X_neighbors, T)
        p_values = stationary_dist / jnp.sum(stationary_dist, axis=1, keepdims=True)
        nots_values = jnp.mean(1.0 - stationary_dist, axis=1)
        mu_values = batched_compute_mu_values(
            Q,
            X,
            X_neighbors,
            p_values,
            T,
            payoff_matrices,
            alpha,
            gamma,
        )
        return Q + (1.0 - nots_values[..., jnp.newaxis] ** k) * mu_values

    summarize_reps = jax.vmap(one_rep_summary)
    step_reps = jax.vmap(one_rep_step)
    initial_summary = summarize_reps(Q_values)

    def scan_step(Q: jnp.ndarray, _):
        Q_next = step_reps(Q)
        return Q_next, summarize_reps(Q_next)

    _, summaries = jax.lax.scan(scan_step, Q_values, None, length=time_steps)
    state_history = jnp.concatenate([initial_summary[0][None, ...], summaries[0]], axis=0)
    policy_history = jnp.concatenate([initial_summary[1][None, ...], summaries[1]], axis=0)
    return state_history, policy_history


def plot_figure3(
    times: np.ndarray,
    sim_state: np.ndarray,
    sim_policy: np.ndarray,
    theory_state: np.ndarray,
    theory_policy: np.ndarray,
    output_prefix: pathlib.Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"s1": "#6ba37d", "s2": "#f1b666"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)

    axes[0].plot(times, sim_state[:, 0], color=colors["s1"], marker="o", markevery=2, linewidth=1.5, markersize=3, label=r"$s_1$ (Simulation)")
    axes[0].plot(times, theory_state[:, 0], color=colors["s1"], linestyle="--", linewidth=1.2, label=r"$s_1$ (Theory)")
    axes[0].plot(times, sim_state[:, 1], color=colors["s2"], marker="o", markevery=2, linewidth=1.5, markersize=3, label=r"$s_2$ (Simulation)")
    axes[0].plot(times, theory_state[:, 1], color=colors["s2"], linestyle="--", linewidth=1.2, label=r"$s_2$ (Theory)")
    axes[0].set_title("(a) State Distribution", fontsize=10)
    axes[0].set_ylabel("Proportion")
    axes[0].legend(frameon=False, fontsize=7, loc="center right")

    axes[1].plot(times, sim_policy[:, 0], color=colors["s1"], marker="o", markevery=2, linewidth=1.5, markersize=3, label=r"$a_1$ (Simulation)")
    axes[1].plot(times, theory_policy[:, 0], color=colors["s1"], linestyle="--", linewidth=1.2, label=r"$a_1$ (Theory)")
    axes[1].plot(times, 1.0 - sim_policy[:, 0], color=colors["s2"], marker="o", markevery=2, linewidth=1.5, markersize=3, label=r"$a_2$ (Simulation)")
    axes[1].plot(times, 1.0 - theory_policy[:, 0], color=colors["s2"], linestyle="--", linewidth=1.2, label=r"$a_2$ (Theory)")
    axes[1].set_title(r"(b) Policy Distribution in State $s_1$", fontsize=10)
    axes[1].set_ylabel("Average Probability")
    axes[1].legend(frameon=False, fontsize=7, loc="center right")

    axes[2].plot(times, sim_policy[:, 1], color=colors["s1"], marker="o", markevery=2, linewidth=1.5, markersize=3, label=r"$a_1$ (Simulation)")
    axes[2].plot(times, theory_policy[:, 1], color=colors["s1"], linestyle="--", linewidth=1.2, label=r"$a_1$ (Theory)")
    axes[2].plot(times, 1.0 - sim_policy[:, 1], color=colors["s2"], marker="o", markevery=2, linewidth=1.5, markersize=3, label=r"$a_2$ (Simulation)")
    axes[2].plot(times, 1.0 - theory_policy[:, 1], color=colors["s2"], linestyle="--", linewidth=1.2, label=r"$a_2$ (Theory)")
    axes[2].set_title(r"(c) Policy Distribution in State $s_2$", fontsize=10)
    axes[2].set_ylabel("Average Probability")
    axes[2].legend(frameon=False, fontsize=7, loc="center right")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Figure 3 from the original Geng et al. code.")
    parser.add_argument("--time-steps", type=int, default=20_000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--record-stride", type=int, default=1_000)
    parser.add_argument("--init-key", type=int, default=42)
    args = parser.parse_args()

    if args.time_steps % args.record_stride != 0:
        raise ValueError("--time-steps must be divisible by --record-stride")

    OUTPUT_DIR.mkdir(exist_ok=True)
    MPLCONFIG_DIR.mkdir(exist_ok=True)

    params = build_figure3_parameters()
    N = int(params["N"])
    K = int(params["K"])
    M = int(params["M"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])

    _, A = create_lattice_graph(N=N, dim=2, periodic=True)
    k = int(jnp.sum(A[0]).item())
    payoff_matrices, T = build_payoffs_and_transitions(params)
    keys, Q_initial, s_initial = initialize_replicates(args.num_reps, args.init_key, N, K, M)

    print(f"Running Figure 3 simulation: steps={args.time_steps}, reps={args.num_reps}, lattice degree={k}")
    sim_state_all, sim_policy_all = run_simulation_summary(
        keys,
        Q_initial,
        s_initial,
        A,
        payoff_matrices,
        T,
        args.time_steps,
        alpha,
        beta,
        gamma,
        K,
        M,
    )

    print("Running Figure 3 theory trajectory")
    theory_state_all, theory_policy_all = run_theory_summary(
        Q_initial,
        A,
        T,
        payoff_matrices,
        args.time_steps,
        alpha,
        beta,
        gamma,
        k,
    )

    keep = np.arange(0, args.time_steps + 1, args.record_stride)
    sim_state = np.asarray(sim_state_all)[keep].mean(axis=1)
    sim_policy = np.asarray(sim_policy_all)[keep].mean(axis=1)
    theory_state = np.asarray(theory_state_all)[keep].mean(axis=1)
    theory_policy = np.asarray(theory_policy_all)[keep].mean(axis=1)
    times = keep.astype(float)

    output_prefix = OUTPUT_DIR / "figure3_reproduction"
    np.savez(
        output_prefix.with_suffix(".npz"),
        times=times,
        sim_state=sim_state,
        sim_policy=sim_policy,
        theory_state=theory_state,
        theory_policy=theory_policy,
        params=params,
    )
    plot_figure3(times, sim_state, sim_policy, theory_state, theory_policy, output_prefix)

    print(f"Saved {output_prefix.with_suffix('.png')}")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(f"Saved {output_prefix.with_suffix('.npz')}")
    print(f"Final simulation state fractions: {sim_state[-1]}")
    print(f"Final simulation cooperation probabilities: state 0={sim_policy[-1, 0]:.6f}, state 1={sim_policy[-1, 1]:.6f}")

    # Keep an explicit call to the original history-based helper for the initial state shape check.
    initial_average_s = compute_average_s(s_initial[:, None, :, :], A, K)
    print(f"Initial compute_average_s check: {np.asarray(initial_average_s[:, 0, :]).mean(axis=0)}")


if __name__ == "__main__":
    main()

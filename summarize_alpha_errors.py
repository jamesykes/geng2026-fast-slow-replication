from __future__ import annotations

import argparse
import pathlib

import jax.numpy as jnp
import numpy as np

from run_figure3 import (
    build_figure3_parameters,
    build_payoffs_and_transitions,
    create_lattice_graph,
    initialize_replicates,
    run_simulation_summary,
    run_theory_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize simulation-theory errors for alpha sweeps.")
    parser.add_argument("--time-steps", type=int, default=5_000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--record-stride", type=int, default=100)
    parser.add_argument("--init-key", type=int, default=42)
    parser.add_argument("--s-init-prob", type=float, nargs=2, default=(0.45, 0.55))
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--output", default="alpha_error_summary.txt")
    args = parser.parse_args()

    if args.time_steps % args.record_stride != 0:
        raise ValueError("--time-steps must be divisible by --record-stride")

    params = build_figure3_parameters()
    N = int(params["N"])
    K = int(params["K"])
    M = int(params["M"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])

    _, A = create_lattice_graph(N=N, dim=2, periodic=True)
    k = int(jnp.sum(A[0]).item())
    payoff_matrices, T = build_payoffs_and_transitions(params)
    s_init_prob = jnp.array(args.s_init_prob, dtype=jnp.float32)
    marker_idx = np.arange(0, args.time_steps + 1, args.record_stride)

    lines = [
        "Alpha sweep simulation-theory error summary",
        "",
        f"time_steps: {args.time_steps}",
        f"num_reps: {args.num_reps}",
        f"record_stride: {args.record_stride}",
        f"s_init_prob: {tuple(args.s_init_prob)}",
        f"metric_times: {marker_idx.tolist()}",
        "",
        "Definitions:",
        "  s2_coop = mean policy probability X(state=s2, action=cooperate)",
        "  s1_frac = fraction of connected edges in state s1",
        "  s2_frac = fraction of connected edges in state s2",
        "  errors are absolute simulation minus theory at recorded time points",
        "",
        (
            "alpha\t"
            "max_abs_err_s2_coop\tmean_abs_err_s2_coop\tfinal_abs_err_s2_coop\t"
            "max_abs_err_s1_frac\tmean_abs_err_s1_frac\t"
            "max_abs_err_s2_frac\tmean_abs_err_s2_frac\t"
            "final_sim_s2_coop\tfinal_theory_s2_coop"
        ),
    ]

    for alpha in args.alpha_values:
        keys, Q_initial, s_initial = initialize_replicates(args.num_reps, args.init_key, N, K, M, s_init_prob)
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

        sim_state = np.asarray(sim_state_all).mean(axis=1)
        sim_policy = np.asarray(sim_policy_all).mean(axis=1)
        theory_state = np.asarray(theory_state_all).mean(axis=1)
        theory_policy = np.asarray(theory_policy_all).mean(axis=1)

        sim_s2_coop = sim_policy[marker_idx, 1]
        theory_s2_coop = theory_policy[marker_idx, 1]
        s2_coop_error = np.abs(sim_s2_coop - theory_s2_coop)

        s1_frac_error = np.abs(sim_state[marker_idx, 0] - theory_state[marker_idx, 0])
        s2_frac_error = np.abs(sim_state[marker_idx, 1] - theory_state[marker_idx, 1])

        values = [
            f"{alpha:g}",
            f"{np.nanmax(s2_coop_error):.6f}",
            f"{np.nanmean(s2_coop_error):.6f}",
            f"{s2_coop_error[-1]:.6f}",
            f"{np.nanmax(s1_frac_error):.6f}",
            f"{np.nanmean(s1_frac_error):.6f}",
            f"{np.nanmax(s2_frac_error):.6f}",
            f"{np.nanmean(s2_frac_error):.6f}",
            f"{sim_s2_coop[-1]:.6f}",
            f"{theory_s2_coop[-1]:.6f}",
        ]
        lines.append("\t".join(values))

    output_path = pathlib.Path(args.output)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()

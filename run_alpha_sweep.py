from __future__ import annotations

import argparse
import os
import pathlib

import jax.numpy as jnp
import numpy as np

from run_figure3 import (
    MPLCONFIG_DIR,
    OUTPUT_DIR,
    build_figure3_parameters,
    build_payoffs_and_transitions,
    create_lattice_graph,
    initialize_replicates,
    plot_figure3,
    run_simulation_summary,
    run_theory_summary,
)


def format_alpha(alpha: float) -> str:
    return f"{alpha:.0e}".replace("-", "m")


def plot_alpha_sweep(
    results: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    marker_times: np.ndarray,
    marker_idx: np.ndarray,
    output_prefix: pathlib.Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sim": "#6ba37d", "theory": "#2f4858"}
    ncols = 3
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (alpha, _sim_state, sim_policy, _theory_state, theory_policy) in zip(axes, results):
        ax.plot(marker_times, sim_policy[marker_idx, 1], color=colors["sim"], linewidth=1.7, label=r"$a_1$ simulation")
        ax.plot(marker_times, theory_policy[marker_idx, 1], color=colors["theory"], marker="o", linestyle="None", markersize=3, label=r"$a_1$ theory")
        ax.set_title(rf"$\alpha={alpha:g}$", fontsize=10)
        ax.set_xlim(marker_times[0], marker_times[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(results):]:
        ax.axis("off")

    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes[: len(results)]:
        ax.set_xlabel("Time")
        ax.set_ylabel(r"State $s_2$ cooperation")

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep alpha for Figure 3 with fixed initial state proportions.")
    parser.add_argument("--time-steps", type=int, default=5_000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--record-stride", type=int, default=500)
    parser.add_argument("--init-key", type=int, default=42)
    parser.add_argument("--s-init-prob", type=float, nargs=2, default=(0.45, 0.55))
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01])
    parser.add_argument("--output-dir", default="alpha_sweeps")
    parser.add_argument("--output-prefix", default="alpha_sweep_init045")
    args = parser.parse_args()

    if args.time_steps % args.record_stride != 0:
        raise ValueError("--time-steps must be divisible by --record-stride")

    OUTPUT_DIR.mkdir(exist_ok=True)
    MPLCONFIG_DIR.mkdir(exist_ok=True)
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = OUTPUT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_figure3_parameters()
    N = int(params["N"])
    K = int(params["K"])
    M = int(params["M"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])

    _, A = create_lattice_graph(N=N, dim=2, periodic=True)
    k = int(jnp.sum(A[0]).item())
    payoff_matrices, T = build_payoffs_and_transitions(params)
    marker_idx = np.arange(0, args.time_steps + 1, args.record_stride)
    times = np.arange(args.time_steps + 1, dtype=float)
    marker_times = marker_idx.astype(float)
    s_init_prob = jnp.array(args.s_init_prob, dtype=jnp.float32)

    results = []
    print(
        f"Sweeping alpha with s_init_prob={tuple(args.s_init_prob)}, "
        f"steps={args.time_steps}, reps={args.num_reps}"
    )

    for alpha in args.alpha_values:
        params_for_alpha = dict(params)
        params_for_alpha["alpha"] = alpha
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
        results.append((alpha, sim_state, sim_policy, theory_state, theory_policy))

        suffix = format_alpha(alpha)
        figure_prefix = output_dir / f"figure3_alpha_{suffix}_init045"
        plot_figure3(
            times,
            marker_times,
            sim_state,
            sim_policy,
            theory_state,
            theory_policy,
            marker_idx,
            figure_prefix,
            False,
        )

        state2_abs_error = np.abs(sim_policy[marker_idx, 1] - theory_policy[marker_idx, 1])
        print(
            f"alpha={alpha:g} final_sim_s2_coop={sim_policy[-1, 1]:.6f} "
            f"final_theory_s2_coop={theory_policy[-1, 1]:.6f} "
            f"max_marker_abs_error_s2={state2_abs_error.max():.6f} "
            f"saved={figure_prefix.with_suffix('.png')}"
        )

    output_prefix = pathlib.Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = output_dir / output_prefix
    plot_alpha_sweep(results, marker_times, marker_idx, output_prefix)
    print(f"Saved alpha sweep plot {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()

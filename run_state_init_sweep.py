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


def format_prob(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def plot_state_sweep(
    results: list[tuple[float, np.ndarray, np.ndarray]],
    marker_times: np.ndarray,
    marker_idx: np.ndarray,
    output_prefix: pathlib.Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"s1": "#6ba37d", "s2": "#f1b666"}
    ncols = 3
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (s1_init, sim_state, theory_state) in zip(axes, results):
        ax.plot(marker_times, sim_state[marker_idx, 0], color=colors["s1"], linewidth=1.7, label=r"$s_1$ simulation")
        ax.plot(marker_times, sim_state[marker_idx, 1], color=colors["s2"], linewidth=1.7, label=r"$s_2$ simulation")
        ax.plot(marker_times, theory_state[marker_idx, 0], color=colors["s1"], marker="o", linestyle="None", markersize=3, label=r"$s_1$ theory")
        ax.plot(marker_times, theory_state[marker_idx, 1], color=colors["s2"], marker="o", linestyle="None", markersize=3, label=r"$s_2$ theory")
        ax.set_title(rf"$s_1(0)={s1_init:.2f}$, $s_2(0)={1.0 - s1_init:.2f}$", fontsize=10)
        ax.set_xlim(marker_times[0], marker_times[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(results):]:
        ax.axis("off")

    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    for ax in axes[: len(results)]:
        ax.set_xlabel("Time")
        ax.set_ylabel("Proportion")

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep initial state proportions for Figure 3 with alpha fixed at 0.001.")
    parser.add_argument("--time-steps", type=int, default=20_000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--record-stride", type=int, default=1_000)
    parser.add_argument("--init-key", type=int, default=42)
    parser.add_argument("--s1-values", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--output-dir", default="state_init_sweeps")
    parser.add_argument("--output-prefix", default="state_init_sweep_alpha001")
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
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])

    _, A = create_lattice_graph(N=N, dim=2, periodic=True)
    k = int(jnp.sum(A[0]).item())
    payoff_matrices, T = build_payoffs_and_transitions(params)
    marker_idx = np.arange(0, args.time_steps + 1, args.record_stride)
    times = np.arange(args.time_steps + 1, dtype=float)
    marker_times = marker_idx.astype(float)

    results = []
    print(f"Sweeping s_init_prob with alpha={alpha}, steps={args.time_steps}, reps={args.num_reps}")
    theory_state = None
    theory_policy = None

    for s1_init in args.s1_values:
        s_init_prob = jnp.array([s1_init, 1.0 - s1_init], dtype=jnp.float32)
        keys, Q_initial, s_initial = initialize_replicates(args.num_reps, args.init_key, N, K, M, s_init_prob)

        if theory_state is None or theory_policy is None:
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
            theory_state = np.asarray(theory_state_all).mean(axis=1)
            theory_policy = np.asarray(theory_policy_all).mean(axis=1)

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
        results.append((s1_init, sim_state, theory_state))

        suffix = format_prob(s1_init)
        figure_prefix = output_dir / f"figure3_sinit_s1_{suffix}_alpha001"
        plot_figure3(times, marker_times, sim_state, sim_policy, theory_state, theory_policy, marker_idx, figure_prefix, False)
        print(
            f"s_init_prob=({s1_init:.2f}, {1.0 - s1_init:.2f}) "
            f"initial={sim_state[0]} final={sim_state[-1]} saved={figure_prefix.with_suffix('.png')}"
        )

    output_prefix = pathlib.Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = output_dir / output_prefix
    plot_state_sweep(results, marker_times, marker_idx, output_prefix)
    print(f"Saved sweep plot {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()

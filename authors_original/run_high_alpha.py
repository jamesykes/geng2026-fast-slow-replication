import argparse
import csv
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments_sim import _default_payoff as sim_default_payoff
from experiments_sim import _default_T as sim_default_T
from sim import sim_graph
from utils import create_lattice_graph


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ALPHAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def _parse_alphas(values):
    alphas = []
    for value in values:
        alphas.extend(float(part) for part in value.split(",") if part)
    if not alphas:
        raise argparse.ArgumentTypeError("at least one alpha is required")
    return tuple(alphas)


def _alpha_label(alpha):
    return f"{alpha:.0e}".replace("e-0", "e-").replace("e+0", "e")


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def _prepare_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "geng_high_alpha_mpl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    return matplotlib


def _run_sim_alpha(alpha, args, payoff_matrices, transition_tensor, adjacency):
    q_avg_all, x_avg_all, s_prop_all = sim_graph(
        args.n,
        2,
        2,
        args.time_steps,
        alpha,
        args.beta,
        args.gamma,
        payoff_matrices,
        transition_tensor,
        adjacency,
        num_reps=args.num_reps,
        init_key=args.init_key,
        s_init_prob=jnp.array([0.5, 0.5]),
    )
    return (
        jax.device_get(q_avg_all.mean(axis=0)),
        jax.device_get(x_avg_all.mean(axis=0)),
        jax.device_get(s_prop_all.mean(axis=0)),
    )


def _init_q(n, seed=3154):
    q_values = jax.random.normal(jax.random.PRNGKey(seed), shape=[n, 2, 2])
    q_values = (
        (q_values - jnp.mean(q_values, axis=0))
        / jnp.std(q_values, axis=0)
        * 0.1
    )
    q_values = q_values.at[:, 0, 0].add(0.5)
    q_values = q_values.at[:, 0, 1].add(0.0)
    q_values = q_values.at[:, 1, 0].add(0.0)
    q_values = q_values.at[:, 1, 1].add(0.5)
    return q_values


def _run_theory_alpha(alpha, args, payoff_matrices, transition_tensor, adjacency):
    _prepare_matplotlib()
    from theory import simulation_theory_graph

    q_values = _init_q(args.n)
    x_mean, s_mean = simulation_theory_graph(
        q_values,
        args.time_steps,
        alpha,
        args.beta,
        adjacency,
        transition_tensor,
        payoff_matrices,
        args.gamma,
        args.k,
    )
    return jax.device_get(x_mean), jax.device_get(s_mean)


def _tail_std(values, fraction=0.2):
    start = int(values.shape[0] * (1.0 - fraction))
    return float(np.nanstd(values[start:]))


def _rmse(left, right):
    return float(np.sqrt(np.nanmean((left - right) ** 2)))


def _tail_rmse(left, right, fraction=0.2):
    steps = min(left.shape[0], right.shape[0])
    start = int(steps * (1.0 - fraction))
    return _rmse(left[start:steps], right[start:steps])


def _align_state_series(sim_s, theory_s):
    if sim_s is None or theory_s is None or len(theory_s) == 0:
        return None, None
    steps = min(sim_s.shape[0] - 1, theory_s.shape[0])
    if steps <= 0:
        return None, None
    return sim_s[1 : steps + 1], theory_s[:steps]


def _summarize_alpha(alpha, sim_q, sim_x, sim_s, theory_x, theory_s):
    summary = {
        "alpha": alpha,
        "sim_finite": bool(np.isfinite(sim_x).all()) if sim_x is not None else "",
        "theory_finite": bool(np.isfinite(theory_x).all()) if theory_x is not None else "",
    }

    if sim_x is not None:
        summary.update(
            {
                "sim_final_a1_s1": float(sim_x[-1, 0, 0]),
                "sim_final_a1_s2": float(sim_x[-1, 1, 0]),
                "sim_tail_std_a1": _tail_std(sim_x[:, :, 0]),
                "sim_min_policy": float(np.nanmin(sim_x)),
                "sim_max_policy": float(np.nanmax(sim_x)),
            }
        )
    if sim_q is not None:
        summary["sim_max_abs_q_mean"] = float(np.nanmax(np.abs(sim_q)))
    if sim_s is not None:
        summary["sim_final_s1"] = float(sim_s[-1, 0])

    if theory_x is not None:
        summary.update(
            {
                "theory_final_a1_s1": float(theory_x[-1, 0, 0]),
                "theory_final_a1_s2": float(theory_x[-1, 1, 0]),
                "theory_tail_std_a1": _tail_std(theory_x[:, :, 0]),
                "theory_min_policy": float(np.nanmin(theory_x)),
                "theory_max_policy": float(np.nanmax(theory_x)),
            }
        )
    if theory_s is not None and len(theory_s):
        summary["theory_final_s1"] = float(theory_s[-1, 0])

    if sim_x is not None and theory_x is not None:
        steps = min(sim_x.shape[0], theory_x.shape[0])
        summary["sim_theory_rmse_policy"] = _rmse(sim_x[:steps], theory_x[:steps])
        summary["sim_theory_tail_rmse_policy"] = _tail_rmse(
            sim_x[:steps],
            theory_x[:steps],
        )

        summary["final_a1_s1_gap"] = float(
            abs(sim_x[-1, 0, 0] - theory_x[-1, 0, 0])
        )
        summary["final_a1_s2_gap"] = float(
            abs(sim_x[-1, 1, 0] - theory_x[-1, 1, 0])
        )

    sim_s_aligned, theory_s_aligned = _align_state_series(sim_s, theory_s)
    if sim_s_aligned is not None:
        summary["sim_theory_rmse_state"] = _rmse(sim_s_aligned, theory_s_aligned)
        summary["sim_theory_tail_rmse_state"] = _tail_rmse(
            sim_s_aligned,
            theory_s_aligned,
        )
        summary["final_s1_gap"] = float(abs(sim_s[-1, 0] - theory_s[-1, 0]))
        summary["final_s2_gap"] = float(abs(sim_s[-1, 1] - theory_s[-1, 1]))

    return summary


def _save_summary(path, summaries):
    fieldnames = sorted({key for row in summaries for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _summary_float(row, key):
    value = row.get(key, "")
    if value == "":
        return np.nan
    return float(value)


def _load_summary(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _combine_saved_runs(args):
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_q_values = {}
    sim_x_values = {}
    sim_s_values = {}
    theory_x_values = {}
    theory_s_values = {}
    summaries = {}

    for input_dir_arg in args.combine_input_dirs:
        input_dir = _resolve_path(input_dir_arg)
        npz_path = input_dir / "high_alpha_sweep.npz"
        summary_path = input_dir / "high_alpha_summary.csv"

        loaded = np.load(npz_path)
        alphas = loaded["alphas"]
        for idx, alpha_value in enumerate(alphas):
            alpha = float(alpha_value)
            if "sim_Q_mean" in loaded and len(loaded["sim_Q_mean"]):
                sim_q_values[alpha] = loaded["sim_Q_mean"][idx]
            if "sim_X_mean" in loaded and len(loaded["sim_X_mean"]):
                sim_x_values[alpha] = loaded["sim_X_mean"][idx]
            if "sim_s_mean" in loaded and len(loaded["sim_s_mean"]):
                sim_s_values[alpha] = loaded["sim_s_mean"][idx]
            if "theory_X_mean" in loaded and len(loaded["theory_X_mean"]):
                theory_x_values[alpha] = loaded["theory_X_mean"][idx]
            if "theory_s_mean" in loaded and len(loaded["theory_s_mean"]):
                theory_s_values[alpha] = loaded["theory_s_mean"][idx]

        for row in _load_summary(summary_path):
            summaries[float(row["alpha"])] = row

    alphas = tuple(sorted(summaries))
    summary_rows = [summaries[alpha] for alpha in alphas]

    npz_path = output_dir / "high_alpha_sweep.npz"
    np.savez_compressed(
        npz_path,
        alphas=np.array(alphas),
        sim_Q_mean=np.array([sim_q_values[a] for a in alphas if a in sim_q_values]),
        sim_X_mean=np.array([sim_x_values[a] for a in alphas if a in sim_x_values]),
        sim_s_mean=np.array([sim_s_values[a] for a in alphas if a in sim_s_values]),
        theory_X_mean=np.array([theory_x_values[a] for a in alphas if a in theory_x_values]),
        theory_s_mean=np.array([theory_s_values[a] for a in alphas if a in theory_s_values]),
    )

    summary_path = output_dir / "high_alpha_summary.csv"
    _save_summary(summary_path, summary_rows)

    figure_path = None
    selected_figure_path = None
    metrics_figure_path = None
    if not args.no_plot:
        figure_path = _plot_results(args, alphas, sim_x_values, theory_x_values)
        metrics_figure_path = _plot_metric_summary(args, summary_rows)
        if args.selected_plot_alphas:
            selected_alphas = _parse_alphas(args.selected_plot_alphas)
            selected_figure_path = _plot_selected_fig3abc(
                args,
                selected_alphas,
                sim_x_values,
                sim_s_values,
                theory_x_values,
                theory_s_values,
            )

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_path}")
    if figure_path is not None:
        print(f"Wrote {figure_path}")
    if metrics_figure_path is not None:
        print(f"Wrote {metrics_figure_path}")
    if selected_figure_path is not None:
        print(f"Wrote {selected_figure_path}")


def _plot_results(args, alphas, sim_x_values, theory_x_values):
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = _resolve_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(alphas)))

    for state_idx, ax in enumerate(axes):
        for alpha, color in zip(alphas, colors):
            label = rf"$\alpha={_alpha_label(alpha)}$"
            sim_x = sim_x_values.get(alpha)
            theory_x = theory_x_values.get(alpha)
            if sim_x is not None:
                t = np.arange(sim_x.shape[0])
                ax.plot(t, sim_x[:, state_idx, 0], color=color, lw=2, label=label)
            if theory_x is not None:
                t = np.arange(theory_x.shape[0])
                plot_sep = max(1, len(t) // 12)
                ax.plot(
                    t[::plot_sep],
                    theory_x[::plot_sep, state_idx, 0],
                    "o",
                    color=color,
                    markersize=4,
                    alpha=0.8,
                )

        ax.set_title(rf"Average $P(a_1)$ in $s_{state_idx + 1}$")
        ax.set_xlabel("Time")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average probability")
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle("High learning-rate alpha sweep")
    fig.tight_layout()

    figure_path = figure_dir / "high_alpha_sweep.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    return figure_path


def _plot_metric_summary(args, summaries):
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = _resolve_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(summaries, key=lambda row: _summary_float(row, "alpha"))
    alphas = np.array([_summary_float(row, "alpha") for row in summaries])
    policy_rmse = np.array(
        [_summary_float(row, "sim_theory_rmse_policy") for row in summaries]
    )
    state_rmse = np.array(
        [_summary_float(row, "sim_theory_rmse_state") for row in summaries]
    )
    tail_policy_rmse = np.array(
        [_summary_float(row, "sim_theory_tail_rmse_policy") for row in summaries]
    )
    tail_state_rmse = np.array(
        [_summary_float(row, "sim_theory_tail_rmse_state") for row in summaries]
    )
    final_a1_s1_gap = np.array(
        [_summary_float(row, "final_a1_s1_gap") for row in summaries]
    )
    final_a1_s2_gap = np.array(
        [_summary_float(row, "final_a1_s2_gap") for row in summaries]
    )
    final_s1_gap = np.array([_summary_float(row, "final_s1_gap") for row in summaries])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    axes[0].plot(alphas, policy_rmse, "o-", label="policy")
    axes[0].plot(alphas, state_rmse, "s-", label="state")
    axes[0].set_title("(a) Full-horizon RMSE")
    axes[0].set_ylabel("RMSE")

    axes[1].plot(alphas, tail_policy_rmse, "o-", label="policy")
    axes[1].plot(alphas, tail_state_rmse, "s-", label="state")
    axes[1].set_title("(b) Tail RMSE")

    axes[2].plot(alphas, final_a1_s1_gap, "o-", label=r"$P(a_1 \mid s_1)$")
    axes[2].plot(alphas, final_a1_s2_gap, "s-", label=r"$P(a_1 \mid s_2)$")
    axes[2].plot(alphas, final_s1_gap, "^-", label=r"$P(s_1)$")
    axes[2].set_title("(c) Final absolute gaps")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Learning rate $\alpha$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Theory-simulation mismatch by learning rate")
    fig.tight_layout()

    figure_path = figure_dir / "high_alpha_error_metrics.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    return figure_path


def _plot_selected_fig3abc(
    args,
    selected_alphas,
    sim_x_values,
    sim_s_values,
    theory_x_values,
    theory_s_values,
):
    _prepare_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure_dir = _resolve_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    available_alphas = [
        alpha
        for alpha in selected_alphas
        if alpha in sim_x_values or alpha in theory_x_values
    ]
    if not available_alphas:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(available_alphas)))

    for alpha, color in zip(available_alphas, colors):
        label = rf"$\alpha={_alpha_label(alpha)}$"

        sim_s = sim_s_values.get(alpha)
        theory_s = theory_s_values.get(alpha)
        if sim_s is not None:
            t_sim_s = np.arange(sim_s.shape[0])
            axes[0].plot(
                t_sim_s,
                sim_s[:, 0],
                color=color,
                lw=2,
                label=label,
            )
            axes[0].plot(
                t_sim_s,
                sim_s[:, 1],
                color=color,
                lw=2,
                ls=":",
            )
        if theory_s is not None and len(theory_s):
            t_theory_s = np.arange(1, theory_s.shape[0] + 1)
            plot_sep = max(1, len(t_theory_s) // 12)
            axes[0].plot(
                t_theory_s[::plot_sep],
                theory_s[::plot_sep, 0],
                "o",
                color=color,
                markersize=4,
                alpha=0.85,
            )
            axes[0].plot(
                t_theory_s[::plot_sep],
                theory_s[::plot_sep, 1],
                "s",
                color=color,
                markersize=4,
                alpha=0.85,
            )

        sim_x = sim_x_values.get(alpha)
        theory_x = theory_x_values.get(alpha)
        for state_idx, ax in enumerate(axes[1:]):
            if sim_x is not None:
                t_sim_x = np.arange(sim_x.shape[0])
                ax.plot(
                    t_sim_x,
                    sim_x[:, state_idx, 0],
                    color=color,
                    lw=2,
                    label=label,
                )
            if theory_x is not None:
                t_theory_x = np.arange(theory_x.shape[0])
                plot_sep = max(1, len(t_theory_x) // 12)
                ax.plot(
                    t_theory_x[::plot_sep],
                    theory_x[::plot_sep, state_idx, 0],
                    "o",
                    color=color,
                    markersize=4,
                    alpha=0.85,
                )

    axes[0].set_title("(a) State probabilities")
    axes[0].set_ylabel("Probability")
    axes[0].set_xlabel("Time")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].set_title(r"(b) Average $P(a_1 \mid s_1)$")
    axes[1].set_xlabel("Time")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].set_title(r"(c) Average $P(a_1 \mid s_2)$")
    axes[2].set_xlabel("Time")
    axes[2].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    alpha_handles = [
        Line2D([0], [0], color=color, lw=2, label=rf"$\alpha={_alpha_label(alpha)}$")
        for alpha, color in zip(available_alphas, colors)
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2, label="simulation"),
        Line2D([0], [0], color="black", marker="o", ls="", label="theory"),
        Line2D([0], [0], color="black", lw=2, label=r"$s_1$"),
        Line2D([0], [0], color="black", lw=2, ls=":", label=r"$s_2$"),
    ]
    axes[0].legend(handles=style_handles, loc="best", fontsize=9)
    axes[2].legend(handles=alpha_handles, loc="best", fontsize=9)

    fig.suptitle("High learning-rate theory vs simulation")
    fig.tight_layout()

    figure_path = figure_dir / "high_alpha_fig3abc_selected.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    return figure_path


def run(args):
    if args.combine_input_dirs:
        _combine_saved_runs(args)
        return

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = _parse_alphas(args.alphas)

    payoff_matrices = sim_default_payoff(5.0, 1.2, 0.5, 2, 2)
    transition_tensor = sim_default_T(0.8, 0.7, 2, 2)
    _, adjacency = create_lattice_graph(args.n, dim=2, periodic=True)

    sim_q_values = {}
    sim_x_values = {}
    sim_s_values = {}
    theory_x_values = {}
    theory_s_values = {}
    summaries = []

    for alpha in alphas:
        print(f"Running alpha={alpha:g}")
        sim_q = sim_x = sim_s = None
        theory_x = theory_s = None

        if args.mode in ("sim", "both"):
            sim_q, sim_x, sim_s = _run_sim_alpha(
                alpha, args, payoff_matrices, transition_tensor, adjacency
            )
            sim_q_values[alpha] = sim_q
            sim_x_values[alpha] = sim_x
            sim_s_values[alpha] = sim_s

        if args.mode in ("theory", "both"):
            theory_x, theory_s = _run_theory_alpha(
                alpha, args, payoff_matrices, transition_tensor, adjacency
            )
            theory_x_values[alpha] = theory_x
            theory_s_values[alpha] = theory_s

        summaries.append(_summarize_alpha(alpha, sim_q, sim_x, sim_s, theory_x, theory_s))

    npz_path = output_dir / "high_alpha_sweep.npz"
    np.savez_compressed(
        npz_path,
        alphas=np.array(alphas),
        sim_Q_mean=np.array([sim_q_values[a] for a in alphas if a in sim_q_values]),
        sim_X_mean=np.array([sim_x_values[a] for a in alphas if a in sim_x_values]),
        sim_s_mean=np.array([sim_s_values[a] for a in alphas if a in sim_s_values]),
        theory_X_mean=np.array([theory_x_values[a] for a in alphas if a in theory_x_values]),
        theory_s_mean=np.array([theory_s_values[a] for a in alphas if a in theory_s_values]),
    )

    summary_path = output_dir / "high_alpha_summary.csv"
    _save_summary(summary_path, summaries)

    figure_path = None
    selected_figure_path = None
    metrics_figure_path = None
    if not args.no_plot:
        figure_path = _plot_results(args, alphas, sim_x_values, theory_x_values)
        metrics_figure_path = _plot_metric_summary(args, summaries)
        if args.selected_plot_alphas:
            selected_alphas = _parse_alphas(args.selected_plot_alphas)
            selected_figure_path = _plot_selected_fig3abc(
                args,
                selected_alphas,
                sim_x_values,
                sim_s_values,
                theory_x_values,
                theory_s_values,
            )

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_path}")
    if figure_path is not None:
        print(f"Wrote {figure_path}")
    if metrics_figure_path is not None:
        print(f"Wrote {metrics_figure_path}")
    if selected_figure_path is not None:
        print(f"Wrote {selected_figure_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the authors' Figure 3 alpha experiment with higher learning "
            "rates, without overwriting the original reproduction data."
        )
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=[str(alpha) for alpha in DEFAULT_ALPHAS],
        help="Alpha values as space-separated values, or comma-separated chunks.",
    )
    parser.add_argument("--mode", choices=("sim", "theory", "both"), default="both")
    parser.add_argument("--time-steps", type=int, default=30000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--init-key", type=int, default=42)
    parser.add_argument("--output-dir", default="data/high_alpha")
    parser.add_argument("--figure-dir", default="figures/high_alpha")
    parser.add_argument(
        "--combine-input-dirs",
        nargs="+",
        default=[],
        help=(
            "Combine saved high_alpha_sweep.npz/high_alpha_summary.csv files "
            "from these output directories instead of running new experiments."
        ),
    )
    parser.add_argument(
        "--selected-plot-alphas",
        nargs="+",
        default=[],
        help=(
            "Alpha values to include in the Figure 3a-c-style selected plot, "
            "as space-separated values or comma-separated chunks."
        ),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

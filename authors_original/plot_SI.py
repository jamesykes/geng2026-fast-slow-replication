import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs("figures", exist_ok=True)

FS = 20
THREE_COLORS = ['#94B49F', '#ECB390', '#DF7861']


def _style_axes(axes, fs=FS):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(True)
        ax.tick_params(labelsize=fs - 6)
        ax.set_xlabel(ax.get_xlabel(), fontsize=fs - 2)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fs - 2)
        ax.set_title(ax.get_title(), fontsize=fs)


def _plot_scan(axes, t, sim_dict, theory_dict, keys, colors, plot_sep,
               theory_keys=None, theory_color_override=None):
    if theory_keys is None:
        theory_keys = keys
    for state_idx, ax in enumerate(axes):
        for key in keys:
            ax.plot(t, sim_dict[key][:, state_idx, 0],
                    lw=2.5, alpha=0.7, color=colors[key])
        for key in theory_keys:
            marker_color = theory_color_override or colors[key]
            ax.plot(t[::plot_sep], theory_dict[key][::plot_sep, state_idx, 0],
                    'o', color=marker_color, markersize=10, zorder=10)


def plot_SI_beta():
    betas = [1, 2, 5]
    beta_colors = dict(zip(betas, THREE_COLORS))

    theory_data = np.load("data/fig_SI/theory_graph_beta.npz")
    sim_data = np.load("data/fig_SI/sim_graph_beta.npz")

    total_t = 20001
    X_theory = {b: theory_data[f"X_mean_{b}"][:total_t] for b in betas}
    X_sim = {b: sim_data[f"X_mean_{b}"][:total_t] for b in betas}

    t = np.arange(total_t)
    plot_sep = len(t) // 8

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    _plot_scan(axes, t, X_sim, X_theory, betas, beta_colors, plot_sep)

    axes[0].set(title="Policy Distribution of $a_1$ in $s_1$",
                xlabel="Time", ylabel="Average Probability",
                ylim=(0.6, 1.0 + (1 - 0.6) * 0.05))
    axes[0].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].set(title="Policy Distribution of $a_1$ in $s_2$",
                xlabel="Time", ylabel="Average Probability",
                ylim=(-0.05, 1.05))
    axes[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    legend_elements = [
        Line2D([0], [0], color=beta_colors[b], lw=2.5, marker='o',
               markersize=10, label=f'$\\beta={b}$')
        for b in betas
    ]
    _style_axes(axes)
    for ax in axes:
        ax.legend(handles=legend_elements, loc='best', fontsize=FS - 4)

    plt.tight_layout()
    plt.savefig("figures/fig_SI_beta.png", dpi=300)
    plt.show()


def plot_SI_transition():
    pcs = [0.8, 0.5, 0.2]
    prs = [0.7, 0.5, 0.3]
    keys = list(zip(pcs, prs))
    pc_pr_colors = dict(zip(keys, THREE_COLORS))

    theory_data = np.load("data/fig_SI/theory_graph_transition.npz")
    sim_data = np.load("data/fig_SI/sim_graph_transition.npz")
    X_theory = {(pc, pr): theory_data[f"X_mean_{int(pc*10)}_{int(pr*10)}"][::10]
                for pc, pr in keys}
    X_sim = {(pc, pr): sim_data[f"X_mean_{int(pc*10)}_{int(pr*10)}"][::10]
             for pc, pr in keys}

    total_t = 200001
    t = np.arange(0, total_t, 10)
    plot_sep = len(t) // 8

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    _plot_scan(axes, t, X_sim, X_theory, keys, pc_pr_colors, plot_sep)

    axes[0].set(title="Policy Distribution of $a_1$ in $s_1$",
                xlabel="Time", ylabel="Average Probability",
                ylim=(0.3 - (1 - 0.3) * 0.05, 1.0 + (1 - 0.3) * 0.05))
    axes[0].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].set(title="Policy Distribution of $a_1$ in $s_2$",
                xlabel="Time", ylabel="Average Probability",
                ylim=(-0.05, 1.05))
    axes[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    legend_elements = [
        Line2D([0], [0], color=pc_pr_colors[(pc, pr)], lw=2.5, marker='o',
               markersize=10, label=f'$p_1={pc:.1f}, p_2={1-pr:.1f}$')
        for pc, pr in keys
    ]
    _style_axes(axes)
    for ax in axes:
        ax.legend(handles=legend_elements, loc='best', fontsize=FS - 4)

    plt.tight_layout()
    plt.savefig("figures/fig_SI_transition.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_SI_beta()
    plot_SI_transition()

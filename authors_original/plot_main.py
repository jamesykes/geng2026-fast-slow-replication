import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs("figures", exist_ok=True)

FS = 20
THREE_COLORS = ['#94B49F', '#ECB390', '#DF7861']
SUBPLOT_LABELS = ['(a)', '(b)', '(c)', '(d)']


def _style_axes(axes, fs=FS):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(True)
        ax.tick_params(labelsize=fs - 6)
        ax.set_xlabel(ax.get_xlabel(), fontsize=fs - 2)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fs - 2)
        ax.set_title(ax.get_title(), fontsize=fs)


def _add_subplot_labels(axes, fs=FS):
    for label, ax in zip(SUBPLOT_LABELS, np.atleast_1d(axes).flat):
        ax.text(-0.05, 1.15, label, transform=ax.transAxes,
                fontsize=fs, va='top', ha='left',
                fontfamily='Times New Roman')


def _plot_sim_theory(ax, t, sim, theory, color, plot_sep, sim_alpha=0.8,
                     marker_alpha=0.8, lw=2.5, markersize=10):
    ax.plot(t, sim, lw=lw, alpha=sim_alpha, color=color)
    ax.plot(t[::plot_sep], theory[::plot_sep], 'o',
            color=color, markersize=markersize, alpha=marker_alpha)


def plot_fig1():
    theory_data = np.load("data/fig1/theory_graph.npz")
    sim_data = np.load("data/fig1/sim_graph.npz")
    s_mean_theory = theory_data["s_mean"]
    X_mean_theory = theory_data["X_mean"]
    s_mean_sim = sim_data["s_mean"]
    X_mean_sim = sim_data["X_mean"]

    colors = ['#a1d99c', '#ffd59a', '#94B49F', '#ECB390']
    t = np.arange(s_mean_sim.shape[0])
    plot_sep = len(t) // 8

    s_mean_theory[0] = s_mean_sim[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for k in range(2):
        axes[0].plot(t, s_mean_sim[:, k], label=f"$s_{k+1}$ (Simulation)",
                     lw=2.5, alpha=0.4, color=colors[k])
        axes[0].plot(t[0], s_mean_theory[0, k], 'o', color=colors[k], markersize=7)
        axes[0].plot(t[1::plot_sep], s_mean_theory[1::plot_sep, k], 'o',
                     label=f"$s_{k+1}$ (Theory)", color=colors[k], markersize=10)
    axes[0].set(title="State Distribution", xlabel="Time", ylabel="Proportion",
                ylim=(0, 1))

    inset_ax = inset_axes(axes[0], width=1.5, height=1, loc='lower center',
                          bbox_to_anchor=(0.32, 0.08),
                          bbox_transform=axes[0].transAxes)
    for k in range(2):
        inset_ax.plot(t[:5], s_mean_sim[:5, k], lw=2, alpha=0.4, color=colors[k])
        inset_ax.plot(t[:5], s_mean_theory[:5, k], 'o', markersize=6, color=colors[k])
    inset_ax.set_xlim(-0.15, 3.15)
    inset_ax.set_ylim(0.2, 0.8)
    inset_ax.set_xticks([0, 1, 2, 3])
    inset_ax.set_xticklabels(['0', '1', '2', '3'])
    inset_ax.tick_params(labelsize=FS - 8)
    inset_ax.tick_params(left=False, labelleft=False)
    inset_ax.tick_params(axis='x', pad=2)
    inset_ax.grid(False)
    inset_ax.patch.set_alpha(0.8)

    for state_idx, ax in enumerate(axes[1:]):
        for a in range(2):
            ax.plot(t, X_mean_sim[:, state_idx, a],
                    label=f"$a_{a+1}$ (Simulation)",
                    lw=2.5, alpha=0.8, color=colors[a + 2])
            ax.plot(t[::plot_sep], X_mean_theory[::plot_sep, state_idx, a], 'o',
                    label=f"$a_{a+1}$ (Theory)",
                    color=colors[a + 2], markersize=10, alpha=0.8)
        ax.set(title=f"Policy Distribution in State $s_{state_idx+1}$",
               xlabel="Time", ylabel="Average Probability", ylim=(0, 1))

    for ax in axes:
        ax.set_ylim(-0.05, 1.05)
    _style_axes(axes)

    axes[0].legend(loc='upper left', fontsize=FS - 4)
    axes[1].legend(loc='center right', fontsize=FS - 4)
    axes[2].legend(loc='center right', fontsize=FS - 4)

    plt.tight_layout()
    plt.savefig('figures/sim1.png', dpi=300)
    plt.show()


def _plot_gamma_row(axes_row, theory_file, sim_file, gammas, gamma_colors,
                    title_suffix):
    theory_data = np.load(theory_file)
    sim_data = np.load(sim_file)
    X_theory = {g: theory_data[f"X_mean_{g}"] for g in gammas}
    X_sim = {g: sim_data[f"X_mean_{g}"] for g in gammas}

    t = np.arange(X_sim[gammas[0]].shape[0])
    plot_sep = len(t) // 8

    for state_idx, ax in enumerate(axes_row):
        for g in gammas:
            _plot_sim_theory(ax, t,
                             X_sim[g][:, state_idx, 0],
                             X_theory[g][:, state_idx, 0],
                             gamma_colors[g], plot_sep)
        ax.set_title(f"Average Prob of $a_1$ in $s_{state_idx+1}$, {title_suffix}",
                     fontsize=FS)
        ax.set_xlabel("Time", fontsize=FS - 2)
        ax.set_ylabel("Average Probability", fontsize=FS - 2)
        ax.set_ylim(0.3, 1.0 + (1 - 0.3) * 0.05)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True)

    legend_elements = [
        Line2D([0], [0], color=gamma_colors[g], lw=2.5, marker='o',
               markersize=10, label=f'$\\gamma={g/10000}$')
        for g in reversed(gammas)
    ]
    for ax in axes_row:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=FS - 4)


def plot_fig2():
    gammas = [2000, 5000, 8000]
    gamma_colors = dict(zip(gammas, THREE_COLORS))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    _plot_gamma_row(axes[0],
                    "data/fig2/theory_graph_b1_1_2_gamma.npz",
                    "data/fig2/sim_graph_b1_1_2_gamma.npz",
                    gammas, gamma_colors, "($b_1=1.2$)")
    _plot_gamma_row(axes[1],
                    "data/fig2/theory_graph_b1_5_gamma.npz",
                    "data/fig2/sim_graph_b1_5_gamma.npz",
                    gammas, gamma_colors, "($b_1=5$)")

    _style_axes(axes)
    _add_subplot_labels(axes)

    plt.tight_layout()
    plt.savefig("figures/sim2.png", dpi=300)
    plt.show()


def plot_fig3():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    alphas = [2, 5, 10]
    alpha_colors = dict(zip(alphas, THREE_COLORS))
    theory_data = np.load("data/fig3/theory_graph_alpha.npz")
    sim_data = np.load("data/fig3/sim_graph_alpha.npz")
    X_theory_a = {a: theory_data[f"X_mean_{a}"] for a in alphas}
    X_sim_a = {a: sim_data[f"X_mean_{a}"] for a in alphas}

    t_a = np.arange(X_sim_a[alphas[0]].shape[0])
    plot_sep_a = len(t_a) // 8

    for state_idx, ax in enumerate(axes[0]):
        for alpha in alphas:
            _plot_sim_theory(ax, t_a,
                             X_sim_a[alpha][:, state_idx, 0],
                             X_theory_a[alpha][:, state_idx, 0],
                             alpha_colors[alpha], plot_sep_a)
        ax.set_title(f"Average Prob of $a_1$ in $s_{state_idx+1}$", fontsize=FS)
        ax.set_xlabel("Time", fontsize=FS - 2)
        ax.set_ylabel("Average Probability", fontsize=FS - 2)
        ax.set_ylim(0.3, 1.0 + (1 - 0.3) * 0.05)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True)

    alpha_display = {10: '$10^{-3}$', 5: '$5\\times 10^{-4}$', 2: '$2\\times 10^{-4}$'}
    legend_alpha = [
        Line2D([0], [0], color=alpha_colors[a], lw=2.5, marker='o', markersize=10,
               label='$\\alpha=$' + alpha_display[a])
        for a in alphas
    ]
    axes[0][0].legend(handles=legend_alpha, loc='lower right', fontsize=FS - 4)
    axes[0][1].legend(handles=legend_alpha, loc='upper left', fontsize=FS - 4)

    ks = [3, 6, 99]
    k_colors = dict(zip(ks, THREE_COLORS))
    theory_data = np.load("data/fig3/theory_graph_degree.npz")
    sim_data = np.load("data/fig3/sim_graph_degree.npz")
    total_t_k = 20001
    X_theory_k = {k: theory_data[f"X_mean_{k}"][:total_t_k] for k in ks}
    X_sim_k = {k: sim_data[f"X_mean_{k}"][:total_t_k] for k in ks}

    t_k = np.arange(total_t_k)
    plot_sep_k = len(t_k) // 8

    ylims_k = [(0.6, 1.0 + (1 - 0.6) * 0.05), (0.3, 1.0 + (1 - 0.3) * 0.05)]
    yticks_k = [[0.6, 0.7, 0.8, 0.9, 1.0],
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    for state_idx, ax in enumerate(axes[1]):
        for k in ks:
            _plot_sim_theory(ax, t_k,
                             X_sim_k[k][:, state_idx, 0],
                             X_theory_k[k][:, state_idx, 0],
                             k_colors[k], plot_sep_k,
                             sim_alpha=0.7, marker_alpha=1.0)
        ax.set(title=f"Average Prob of $a_1$ in $s_{state_idx+1}$",
               xlabel="Time", ylabel="Average Probability",
               ylim=ylims_k[state_idx])
        ax.set_yticks(yticks_k[state_idx])
        ax.grid(True)

    legend_k = []
    for k in reversed(ks):
        label = 'complete graph' if k == 99 else f'$k={k}$'
        legend_k.append(Line2D([0], [0], color=k_colors[k], lw=2.5, marker='o',
                               markersize=10, label=label))
    axes[1][0].legend(handles=legend_k, loc='lower right', fontsize=FS - 4)
    axes[1][1].legend(handles=legend_k, loc='lower right', fontsize=FS - 4)

    _style_axes(axes)
    _add_subplot_labels(axes)

    plt.tight_layout()
    plt.savefig("figures/sim3.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()

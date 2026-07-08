from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL_CODE_DIR = REPO_ROOT / "original_paper_code"
OUTPUT_DIR = REPO_ROOT / "outputs"
MPLCONFIG_DIR = pathlib.Path("/private/tmp/geng2026_fast_slow_matplotlib_cache")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ORIGINAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE_DIR))

from run_figure3 import run_simulation_summary, run_theory_summary  # noqa: E402
from utils.graph_utils import create_random_regular_graph  # noqa: E402


def build_sh_pd_payoffs(b: float, r: float) -> jnp.ndarray:
    return jnp.array(
        [
            [[1.0, 0.0], [r, r]],
            [[1.0, -r], [b, 0.0]],
        ],
        dtype=jnp.float32,
    )


def build_transition_tensor(rule: str, p_cc: float, p_other: float) -> jnp.ndarray:
    if rule == "state_independent":
        t = np.zeros((2, 2, 2, 2), dtype=np.float32)
        t[:, :, :, 0] = p_other
        t[:, :, :, 1] = 1.0 - p_other
        t[:, 0, 0, 0] = p_cc
        t[:, 0, 0, 1] = 1.0 - p_cc
        return jnp.array(t)

    if rule == "state_dependent":
        t = np.zeros((2, 2, 2, 2), dtype=np.float32)
        # Cooperate/cooperate moves to, or keeps, the SH-like state.
        t[0, 0, 0, 0] = 1.0
        t[1, 0, 0, 0] = 1.0
        # Defect/defect moves to, or keeps, the PD-like state.
        t[0, 1, 1, 1] = 1.0
        t[1, 1, 1, 1] = 1.0
        # Mixed outcomes keep the current state. This is the state-dependent
        # contrast to the action-only rule above.
        t[0, 0, 1, 0] = 1.0
        t[0, 1, 0, 0] = 1.0
        t[1, 0, 1, 1] = 1.0
        t[1, 1, 0, 1] = 1.0
        return jnp.array(t)

    raise ValueError(f"Unknown transition rule: {rule}")


def scaled_beta(
    key: jax.Array,
    alpha: float,
    beta: float,
    low: float,
    high: float,
    shape: tuple[int, ...],
) -> jnp.ndarray:
    return low + (high - low) * jax.random.beta(key, alpha, beta, shape=shape)


def initialize_beta_replicates(
    num_reps: int,
    init_key: int,
    n: int,
    s_init_prob: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    keys = []
    q_values = []
    s_values = []
    for rep in range(num_reps):
        key = jax.random.PRNGKey(init_key + rep)
        key, key_qc, key_qd, key_s = jax.random.split(key, 4)
        qc = scaled_beta(key_qc, 60.0, 60.0, -0.1, 1.2, (n, 2))
        qd = scaled_beta(key_qd, 5.0, 5.0, -0.1, 1.2, (n, 2))
        q = jnp.stack([qc, qd], axis=-1).astype(jnp.float32)
        s_upper = jax.random.choice(key_s, jnp.array([0, 1]), shape=(n, n), p=s_init_prob)
        s = jnp.triu(s_upper) + jnp.triu(s_upper, k=1).T
        keys.append(key)
        q_values.append(q)
        s_values.append(s)
    return jnp.array(keys), jnp.array(q_values), jnp.array(s_values)


def plot_degree_sweep(rows: list[dict[str, float | int | str]], output_prefix: pathlib.Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for rule, color in (("state_independent", "#536d8a"), ("state_dependent", "#b85c38")):
        subset = [row for row in rows if row["rule"] == rule]
        if not subset:
            continue
        degrees = np.array([row["degree"] for row in subset], dtype=float)
        sim = np.array([row["sim_final_coop"] for row in subset], dtype=float)
        theory = np.array([row["theory_final_coop"] for row in subset], dtype=float)
        order = np.argsort(degrees)
        label = rule.replace("_", " ")
        ax.plot(degrees[order], sim[order], color=color, linewidth=1.7, label=f"{label} simulation")
        ax.plot(degrees[order], theory[order], color=color, marker="o", linestyle="None", markersize=4, label=f"{label} theory")

    ax.set_xlabel("Regular graph degree")
    ax.set_ylabel("Final cooperation probability")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Yuan-style complete-to-regular graph degree sweep.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--time-steps", type=int, default=20_000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 6, 99])
    parser.add_argument("--rules", choices=("state_independent", "state_dependent"), nargs="+", default=["state_independent", "state_dependent"])
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=1.2)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--p-cc", type=float, default=0.8)
    parser.add_argument("--p-other", type=float, default=0.3)
    parser.add_argument("--s-init-prob", type=float, nargs=2, default=(0.5, 0.5))
    parser.add_argument("--init-key", type=int, default=42)
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="yuan_regular_graph_sweeps")
    parser.add_argument("--output-prefix", default="yuan_regular_graph_sweep")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = OUTPUT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    MPLCONFIG_DIR.mkdir(exist_ok=True)

    payoff_matrices = build_sh_pd_payoffs(args.b, args.r)
    s_init_prob = jnp.array(args.s_init_prob, dtype=jnp.float32)
    rows: list[dict[str, float | int | str]] = []

    print(f"JAX devices: {jax.devices()}")
    for rule in args.rules:
        transition = build_transition_tensor(rule, args.p_cc, args.p_other)
        for degree in args.degrees:
            if degree >= args.n:
                raise ValueError(f"Degree {degree} must be less than n={args.n}")
            _, adjacency = create_random_regular_graph(args.n, degree, seed=args.graph_seed + degree)
            keys, q_initial, s_initial = initialize_beta_replicates(
                args.num_reps,
                args.init_key,
                args.n,
                s_init_prob,
            )
            theory_state, theory_policy = run_theory_summary(
                q_initial,
                adjacency,
                transition,
                payoff_matrices,
                args.time_steps,
                args.alpha,
                args.beta,
                args.gamma,
                degree,
            )
            sim_state, sim_policy = run_simulation_summary(
                keys,
                q_initial,
                s_initial,
                adjacency,
                payoff_matrices,
                transition,
                args.time_steps,
                args.alpha,
                args.beta,
                args.gamma,
                2,
                2,
            )
            sim_policy_mean = np.asarray(sim_policy).mean(axis=1)
            theory_policy_mean = np.asarray(theory_policy).mean(axis=1)
            sim_state_mean = np.asarray(sim_state).mean(axis=1)
            theory_state_mean = np.asarray(theory_state).mean(axis=1)
            sim_final_coop = float(sim_policy_mean[-1].mean())
            theory_final_coop = float(theory_policy_mean[-1].mean())
            row = {
                "rule": rule,
                "degree": degree,
                "sim_final_coop": sim_final_coop,
                "theory_final_coop": theory_final_coop,
                "sim_final_state0": float(sim_state_mean[-1, 0]),
                "theory_final_state0": float(theory_state_mean[-1, 0]),
            }
            rows.append(row)
            print(
                f"rule={rule} degree={degree} "
                f"sim_final_coop={sim_final_coop:.6f} "
                f"theory_final_coop={theory_final_coop:.6f}"
            )

    csv_path = output_dir / f"{args.output_prefix}.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["rule", "degree", "sim_final_coop", "theory_final_coop", "sim_final_state0", "theory_final_state0"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_prefix = output_dir / args.output_prefix
    plot_degree_sweep(rows, output_prefix)
    print(f"Saved {csv_path}")
    print(f"Saved {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()

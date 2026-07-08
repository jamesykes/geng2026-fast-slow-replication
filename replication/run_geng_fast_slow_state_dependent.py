from __future__ import annotations

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from run_figure3 import (
    OUTPUT_DIR,
    build_figure3_parameters,
    initialize_replicates,
    plot_figure3,
    run_simulation_summary,
    run_theory_summary,
)
from utils.graph_utils import create_lattice_graph


def build_state_dependent_transitions(
    p_cc_by_state: tuple[float, float],
    p_other_by_state: tuple[float, float],
) -> jnp.ndarray:
    t = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for state in range(2):
        t[state, :, :, 0] = p_other_by_state[state]
        t[state, :, :, 1] = 1.0 - p_other_by_state[state]
        t[state, 0, 0, 0] = p_cc_by_state[state]
        t[state, 0, 0, 1] = 1.0 - p_cc_by_state[state]
    return jnp.array(t)


def build_pd_payoffs(params: dict[str, float | int]) -> jnp.ndarray:
    b1 = float(params["b1"])
    b2 = float(params["b2"])
    c1 = float(params["c1"])
    c2 = float(params["c2"])
    return jnp.array(
        [
            [[b1 - c1, b1], [-c1, 0.0]],
            [[b2 - c2, b2], [-c2, 0.0]],
        ],
        dtype=jnp.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Geng fast-slow run with explicit state-dependent transitions.")
    parser.add_argument("--time-steps", type=int, default=20_000)
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--record-stride", type=int, default=1_000)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--p-cc-by-state", type=float, nargs=2, default=(0.85, 0.65))
    parser.add_argument("--p-other-by-state", type=float, nargs=2, default=(0.40, 0.20))
    parser.add_argument("--s-init-prob", type=float, nargs=2, default=(0.45, 0.55))
    parser.add_argument("--init-key", type=int, default=42)
    parser.add_argument("--output-dir", default="geng_state_dependent")
    parser.add_argument("--output-prefix", default="geng_fast_slow_state_dependent")
    args = parser.parse_args()

    if args.time_steps % args.record_stride != 0:
        raise ValueError("--time-steps must be divisible by --record-stride")

    params = build_figure3_parameters()
    n = int(params["N"])
    k_states = int(params["K"])
    m_actions = int(params["M"])
    alpha = float(params["alpha"] if args.alpha is None else args.alpha)
    beta = float(params["beta"])
    gamma = float(params["gamma"])

    _, adjacency = create_lattice_graph(n, dim=2, periodic=True)
    degree = int(jnp.sum(adjacency[0]).item())
    payoffs = build_pd_payoffs(params)
    transitions = build_state_dependent_transitions(
        tuple(args.p_cc_by_state),
        tuple(args.p_other_by_state),
    )
    s_init_prob = jnp.array(args.s_init_prob, dtype=jnp.float32)
    keys, q_initial, s_initial = initialize_replicates(args.num_reps, args.init_key, n, k_states, m_actions, s_init_prob)

    print(f"JAX devices: {jax.devices()}")
    print(
        "Running state-dependent fast-slow comparison: "
        f"steps={args.time_steps}, reps={args.num_reps}, alpha={alpha}, degree={degree}"
    )

    sim_state_all, sim_policy_all = run_simulation_summary(
        keys,
        q_initial,
        s_initial,
        adjacency,
        payoffs,
        transitions,
        args.time_steps,
        alpha,
        beta,
        gamma,
        k_states,
        m_actions,
    )
    theory_state_all, theory_policy_all = run_theory_summary(
        q_initial,
        adjacency,
        transitions,
        payoffs,
        args.time_steps,
        alpha,
        beta,
        gamma,
        degree,
    )

    marker_idx = np.arange(0, args.time_steps + 1, args.record_stride)
    times = np.arange(args.time_steps + 1, dtype=float)
    marker_times = marker_idx.astype(float)
    sim_state = np.asarray(sim_state_all).mean(axis=1)
    sim_policy = np.asarray(sim_policy_all).mean(axis=1)
    theory_state = np.asarray(theory_state_all).mean(axis=1)
    theory_policy = np.asarray(theory_policy_all).mean(axis=1)

    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = OUTPUT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / args.output_prefix

    plot_figure3(
        times,
        marker_times,
        sim_state,
        sim_policy,
        theory_state,
        theory_policy,
        marker_idx,
        output_prefix,
        False,
    )
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        sim_state=sim_state,
        sim_policy=sim_policy,
        theory_state=theory_state,
        theory_policy=theory_policy,
        transitions=np.asarray(transitions),
    )
    print(f"Saved {output_prefix.with_suffix('.png')}")
    print(f"Saved {output_prefix.with_suffix('.npz')}")
    print(f"Final simulation state fractions: {sim_state[-1]}")
    print(f"Final theory state fractions: {theory_state[-1]}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import pathlib
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def grid_size(possible_rewards: tuple[float, ...], space: float) -> int:
    return round((max(possible_rewards) - min(possible_rewards)) / space) + 1


def quantize_to_grid(values: jnp.ndarray, r_min: float, space: float, n: int) -> jnp.ndarray:
    return jnp.clip(jnp.rint((values - r_min) / space), 0, n - 1).astype(jnp.int32)


def build_initial_distribution(
    key: jax.Array,
    n: int,
    r_min: float,
    r_max: float,
    space: float,
    samples_multiplier: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    key_qc, key_qd = jax.random.split(key)
    sample_n = n * n * samples_multiplier
    qc = r_min + (r_max - r_min) * jax.random.beta(key_qc, 20.0, 80.0, shape=(sample_n,))
    qd = r_min + (r_max - r_min) * jax.random.beta(key_qd, 80.0, 20.0, shape=(sample_n,))
    qc_idx = quantize_to_grid(qc, r_min, space, n)
    qd_idx = quantize_to_grid(qd, r_min, space, n)
    bins = qc_idx * n + qd_idx
    p_single = jnp.bincount(bins, length=n * n).reshape((n, n)).astype(dtype)
    p_single = p_single / p_single.sum()
    pair_mass = p_single[:, :, None, None] * p_single[None, None, :, :]
    return jnp.repeat(pair_mass[:, :, None, :, :] * jnp.asarray(0.5, dtype=dtype), repeats=2, axis=2)


def build_index_arrays(n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jnp.indices((n, n, 2, n, n), dtype=jnp.int32)


@partial(jax.jit, static_argnames=("n",))
def summarize_distribution(
    p: jnp.ndarray,
    q_axis: jnp.ndarray,
    xc: jnp.ndarray,
    n: int,
) -> jnp.ndarray:
    qc_mesh = q_axis[:, None]
    qd_mesh = q_axis[None, :]
    marginal = jnp.sum(p, axis=(2, 3, 4))
    total = jnp.sum(marginal)
    ave_qc = jnp.sum(qc_mesh * marginal) / total
    ave_qd = jnp.sum(qd_mesh * marginal) / total
    ave_xc = jnp.sum(xc * marginal) / total
    ave_xd = 1.0 - ave_xc
    ave_fp = jnp.sum(p[:, :, 0, :, :]) / total
    return jnp.array([ave_qc, ave_qd, ave_xc, ave_xd, ave_fp])


@partial(jax.jit, static_argnames=("n",))
def compute_q_increments(
    p: jnp.ndarray,
    q_axis: jnp.ndarray,
    xc: jnp.ndarray,
    eta: float,
    payoffs: jnp.ndarray,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    q_mesh_c = q_axis[:, None]
    q_mesh_d = q_axis[None, :]
    xd = 1.0 - xc
    marginal = jnp.sum(p, axis=(2, 3, 4))

    payoff_c_by_neighbor = jnp.stack(
        [
            xc * payoffs[0, 0, 0] + xd * payoffs[0, 0, 1],
            xc * payoffs[1, 0, 0] + xd * payoffs[1, 0, 1],
        ],
        axis=0,
    )
    payoff_d_by_neighbor = jnp.stack(
        [
            xc * payoffs[0, 1, 0] + xd * payoffs[0, 1, 1],
            xc * payoffs[1, 1, 0] + xd * payoffs[1, 1, 1],
        ],
        axis=0,
    )
    payoff_c_num = jnp.einsum("ijgkm,gkm->ij", p, payoff_c_by_neighbor, optimize="optimal")
    payoff_d_num = jnp.einsum("ijgkm,gkm->ij", p, payoff_d_by_neighbor, optimize="optimal")
    payoff_c = jnp.where(marginal > 0.0, payoff_c_num / marginal, 0.0)
    payoff_d = jnp.where(marginal > 0.0, payoff_d_num / marginal, 0.0)
    return eta * (payoff_c - q_mesh_c), eta * (payoff_d - q_mesh_d)


@partial(jax.jit, static_argnames=("n",))
def update_distribution(
    p: jnp.ndarray,
    q_axis: jnp.ndarray,
    xc: jnp.ndarray,
    q_c: jnp.ndarray,
    q_d: jnp.ndarray,
    indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    r_min: float,
    space: float,
    n: int,
) -> jnp.ndarray:
    old_i, old_j, old_g, old_k, old_m = indices
    xd = 1.0 - xc
    new_i_c = quantize_to_grid(q_axis[:, None] + q_c, r_min, space, n)
    new_j_d = quantize_to_grid(q_axis[None, :] + q_d, r_min, space, n)

    p_cc = p * xc[old_i, old_j] * xc[old_k, old_m]
    p_cd = p * xc[old_i, old_j] * xd[old_k, old_m]
    p_dc = p * xd[old_i, old_j] * xc[old_k, old_m]
    p_dd = p * xd[old_i, old_j] * xd[old_k, old_m]

    next_p = jnp.zeros_like(p)
    next_p = next_p.at[
        new_i_c[old_i, old_j],
        old_j,
        jnp.zeros_like(old_g),
        new_i_c[old_k, old_m],
        old_m,
    ].add(p_cc)
    next_p = next_p.at[
        new_i_c[old_i, old_j],
        old_j,
        old_g,
        old_k,
        new_j_d[old_k, old_m],
    ].add(p_cd)
    next_p = next_p.at[
        old_i,
        new_j_d[old_i, old_j],
        old_g,
        new_i_c[old_k, old_m],
        old_m,
    ].add(p_dc)
    next_p = next_p.at[
        old_i,
        new_j_d[old_i, old_j],
        jnp.ones_like(old_g),
        old_k,
        new_j_d[old_k, old_m],
    ].add(p_dd)
    return next_p


@partial(jax.jit, static_argnames=("n",))
def one_step(
    p: jnp.ndarray,
    q_axis: jnp.ndarray,
    xc: jnp.ndarray,
    eta: float,
    payoffs: jnp.ndarray,
    indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    r_min: float,
    space: float,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    summary = summarize_distribution(p, q_axis, xc, n)
    q_c, q_d = compute_q_increments(p, q_axis, xc, eta, payoffs, n)
    next_p = update_distribution(p, q_axis, xc, q_c, q_d, indices, r_min, space, n)
    return next_p, summary


def run(args: argparse.Namespace) -> pathlib.Path:
    possible_rewards = (-args.r, 0.0, args.r, 1.0, args.b)
    r_min = min(possible_rewards)
    r_max = max(possible_rewards)
    n = grid_size(possible_rewards, args.space)
    dense_entries = n * n * 2 * n * n
    if dense_entries > args.max_dense_entries and not args.allow_large_dense:
        raise SystemExit(
            "Refusing to allocate the dense Chu distribution locally: "
            f"shape=({n}, {n}, 2, {n}, {n}) has {dense_entries:,} entries. "
            "Use a coarser --space for smoke tests or pass --allow-large-dense on the server."
        )
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64

    if dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)

    q_axis = jnp.arange(n, dtype=dtype) * jnp.asarray(args.space, dtype=dtype) + jnp.asarray(r_min, dtype=dtype)
    qc = q_axis[:, None]
    qd = q_axis[None, :]
    xc = jax.nn.sigmoid(jnp.asarray(args.tau, dtype=dtype) * (qc - qd))
    payoffs = jnp.array(
        [
            [[1.0, 0.0], [args.r, args.r]],
            [[1.0, -args.r], [args.b, 0.0]],
        ],
        dtype=dtype,
    )

    key = jax.random.PRNGKey(args.seed)
    p = build_initial_distribution(
        key,
        n,
        r_min,
        r_max,
        args.space,
        args.samples_multiplier,
        dtype,
    )
    indices = build_index_arrays(n)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ts_results_jax.csv"

    print(f"JAX devices: {jax.devices()}")
    print(f"Grid: {n} x {n}; dense distribution shape={p.shape}; entries={p.size:,}")
    print(f"Writing {output_path}")

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["T", "ave_qc", "ave_qd", "ave_xc", "ave_xd", "ave_fp"])
        for t in range(args.time_steps + 1):
            p, summary = one_step(
                p,
                q_axis,
                xc,
                args.eta,
                payoffs,
                indices,
                r_min,
                args.space,
                n,
            )
            summary_np = np.asarray(jax.device_get(summary), dtype=float)
            writer.writerow([t, *summary_np.tolist()])
            if t % args.print_every == 0:
                print(
                    f"Step {t}: Q(C)={summary_np[0]:.6f} "
                    f"Q(D)={summary_np[1]:.6f} X(C)={summary_np[2]:.6f} fp={summary_np[4]:.6f}"
                )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX/GPU port of Chu et al. 2023 case2_1.py.")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.4)
    parser.add_argument("--b", type=float, default=1.2)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--space", type=float, default=0.01)
    parser.add_argument("--time-steps", type=int, default=200)
    parser.add_argument("--samples-multiplier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-dense-entries", type=int, default=100_000_000)
    parser.add_argument("--allow-large-dense", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", default="chu2023/outputs/case2_1_jax")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

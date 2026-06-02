'''
Particle solver of Equation (19) in the main text.
'''
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import os

os.makedirs("data/fig1", exist_ok=True)
os.makedirs("data/fig2", exist_ok=True)
os.makedirs("data/fig3", exist_ok=True)

sns.set_style('whitegrid')
from tqdm import tqdm, trange
from typing import Tuple, List, Dict, Any, Optional, Union, Callable, NamedTuple


@jit
def softmax(Q_mesh, beta):
    return jax.nn.softmax(beta * Q_mesh, axis=-1)


@jit
def stationary_dist_func(
    Xi: jnp.ndarray,
    Xj: jnp.ndarray,
    T_tensor: jnp.ndarray,
) -> jnp.ndarray:
    Tss_tensor = jnp.einsum('sa,sb,sabz->sz', Xi, Xj, T_tensor, optimize='optimal')
    T_AB, T_BA = Tss_tensor[0, 1], Tss_tensor[1, 0]
    pi_A = T_BA / (T_AB + T_BA)
    return jnp.array([pi_A, 1 - pi_A])

batched_pair_stationary_dist_func = jit(vmap(
    stationary_dist_func, in_axes=(0, 0, None)
))


@jit
def compute_TD_target_Q_learning(
    Qi: jnp.ndarray,
    Xj: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    T_tensor: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    Rsa = jnp.einsum('sb,sab->sa', Xj, payoff_matrices, optimize='optimal')
    Qmax = jnp.max(Qi, axis=-1)
    Tsas = jnp.einsum('sb,sabz->saz', Xj, T_tensor, optimize='optimal')

    gamma_term = jnp.einsum('saz,z->sa', Tsas, Qmax, optimize='optimal')
    return Rsa + gamma * gamma_term


batched_j_compute_TD_target_Q_learning = vmap(
    compute_TD_target_Q_learning, in_axes=(None, 0, None, None, None)
)
batched_i_compute_TD_target_Q_learning = jit(vmap(
    batched_j_compute_TD_target_Q_learning,
    in_axes=(0, None, None, None, None),
))


@jit
def compute_mu_values(
    Qi: jnp.ndarray,
    Xi: jnp.ndarray,
    X_values: jnp.ndarray,
    p_values_for_i: jnp.ndarray,
    T_tensor: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    alpha: float,
    gamma: float,
) -> jnp.ndarray:
    TD_target_for_i = batched_j_compute_TD_target_Q_learning(
        Qi,
        X_values,
        payoff_matrices,
        T_tensor,
        gamma,
    )
    return alpha * Xi * (
        jnp.einsum('js,jsa->sa', p_values_for_i, TD_target_for_i, optimize='optimal')
        - Qi
    )


batched_compute_mu_values = jit(vmap(
    compute_mu_values, in_axes=(0, 0, 0, 0, None, None, None, None)
))


@jit
def compute_conditional_values(
    query_Q_values: jnp.ndarray,
    Q_values: jnp.ndarray,
    pair_stationary_dist_values: jnp.ndarray,
    q_threshold: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    squared_distances = jnp.sum(
        (query_Q_values[:, None] - Q_values[None, :]) ** 2,
        axis=(-2, -1),
    )
    masks = squared_distances <= q_threshold ** 2
    masks = masks.astype(query_Q_values.dtype)

    raw_weights = masks[:, :, None] * pair_stationary_dist_values[None, :, :]
    weights = raw_weights / jnp.sum(raw_weights, axis=1, keepdims=True)
    p_state_values = jnp.sum(raw_weights, axis=1) / jnp.sum(
        masks,
        axis=1,
        keepdims=True,
    )

    return weights, p_state_values


@jit
def compute_mu_values_pair_particles(
    query_Q_values: jnp.ndarray,
    query_X_values: jnp.ndarray,
    Q_values: jnp.ndarray,
    X_tilde_values: jnp.ndarray,
    pair_stationary_dist_values: jnp.ndarray,
    q_threshold: float,
    k: int,
    T_tensor: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    alpha: float,
    gamma: float,
) -> jnp.ndarray:
    weights, p_state_values = compute_conditional_values(
        query_Q_values,
        Q_values,
        pair_stationary_dist_values,
        q_threshold,
    )
    TD_targets = batched_i_compute_TD_target_Q_learning(
        query_Q_values,
        X_tilde_values,
        payoff_matrices,
        T_tensor,
        gamma,
    )
    TD_targets = jnp.einsum('ils,ilsa->isa', weights, TD_targets, optimize='optimal')
    state_occurrence_values = 1 - (1 - p_state_values)[..., jnp.newaxis] ** k

    return alpha * state_occurrence_values * query_X_values * (TD_targets - query_Q_values)


def build_pair_particles(
    Q_values: jnp.ndarray,
    A: jnp.ndarray,
    max_pairs: Optional[int] = 500,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A_np = np.asarray(jax.device_get(A))
    src_idx, dst_idx = np.where(np.triu(A_np > 0, k=1))

    if max_pairs is not None:
        max_edges = max_pairs // 2

        if len(src_idx) > max_edges:
            rng = np.random.default_rng(seed)
            selected_idx = rng.choice(
                len(src_idx),
                size=max_edges,
                replace=False,
            )
            src_idx = src_idx[selected_idx]
            dst_idx = dst_idx[selected_idx]

    focal_idx = jnp.array(np.concatenate([src_idx, dst_idx]))
    neighbor_idx = jnp.array(np.concatenate([dst_idx, src_idx]))

    return Q_values[focal_idx], Q_values[neighbor_idx]


def simulation_theory_graph(
    Q_values: jnp.ndarray,
    time_steps: int,
    alpha: float,
    beta: float,
    A: jnp.ndarray,
    T_tensor: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    gamma: float,
    k: int,
    q_threshold: float = 0.01,
    max_pairs: Optional[int] = 500,
    seed: int = 0,
):
    X_history = []
    s_history = []

    Q_values, Q_tilde_values = build_pair_particles(
        Q_values,
        A,
        max_pairs=max_pairs,
        seed=seed,
    )
    all_Q_values = jnp.concatenate([Q_values, Q_tilde_values], axis=0)
    X_history.append(softmax(all_Q_values, beta).mean(axis=0))

    for _ in trange(time_steps):
        X_values = softmax(Q_values, beta)
        X_tilde_values = softmax(Q_tilde_values, beta)

        stationary_dist_values = batched_pair_stationary_dist_func(
            X_values,
            X_tilde_values,
            T_tensor,
        )

        shared_mu_args = (
            Q_values,
            X_tilde_values,
            stationary_dist_values,
            q_threshold,
            k,
            T_tensor,
            payoff_matrices,
            alpha,
            gamma,
        )
        mu_values = compute_mu_values_pair_particles(
            Q_values,
            X_values,
            *shared_mu_args,
        )
        mu_tilde_values = compute_mu_values_pair_particles(
            Q_tilde_values,
            X_tilde_values,
            *shared_mu_args,
        )

        Q_values = Q_values + mu_values
        Q_tilde_values = Q_tilde_values + mu_tilde_values

        all_Q_values = jnp.concatenate([Q_values, Q_tilde_values], axis=0)

        s_history.append(stationary_dist_values.mean(axis=0))
        X_history.append(softmax(all_Q_values, beta).mean(axis=0))

    X_history = jnp.array(X_history)
    s_history = jnp.array(s_history)
    return X_history, s_history


def compute_average_s(s_history: jnp.ndarray, A: jnp.ndarray, K: int) -> jnp.ndarray:
    B, T, N, _ = s_history.shape

    connected_mask = A == 1
    connected_states = s_history * connected_mask[None, None, :, :]

    connected_states_flat = connected_states.reshape(B, T, -1)
    mask_flat = connected_mask.flatten()
    connected_states_valid = connected_states_flat[:, :, mask_flat]

    state_counts = jnp.zeros((B, T, K))
    for k in range(K):
        counts = jnp.sum(connected_states_valid == k, axis=-1)
        state_counts = state_counts.at[:, :, k].set(counts)

    total_connections = connected_states_valid.shape[-1]
    state_proportions = state_counts / total_connections

    return state_proportions


if __name__ == "__main__":
    pass

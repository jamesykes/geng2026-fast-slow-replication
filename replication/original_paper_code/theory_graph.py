import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
import numpy as np
from functools import partial


from tqdm import tqdm, trange
from typing import Tuple, List, Dict, Any, Optional, Union, Callable, NamedTuple


@jit
def softmax(Q_mesh, beta):
    return jax.nn.softmax(beta * Q_mesh, axis=-1)

@jit
def stationary_dist_func(
    Xi: jnp.ndarray, Xj: jnp.ndarray,
    T_tensor: jnp.ndarray
) -> jnp.ndarray:
    Tss_tensor = jnp.einsum('sa,sb,sabz->sz', Xi, Xj, T_tensor, optimize='optimal')
    T_AB, T_BA = Tss_tensor[0, 1], Tss_tensor[1, 0]
    pi_A = T_BA / (T_AB + T_BA)
    return jnp.array([pi_A, 1 - pi_A])

batched_j_stationary_dist_func = jit(vmap(
    stationary_dist_func, in_axes=(None, 0, None)
))
batched_ij_stationary_dist_func = jit(vmap(
    batched_j_stationary_dist_func, in_axes=(0, 0, None)
))

@jit
def compute_TD_target_Q_learning(
    Qi: jnp.ndarray,
    Xj: jnp.ndarray,
    payoff_matrices: jnp.ndarray,
    T_tensor: jnp.ndarray,
    gamma: float
) -> jnp.ndarray:
    """
    Compute the TD target for the focal agent
    """
    Rsa = jnp.einsum('sb,sab->sa', Xj, payoff_matrices, optimize='optimal')
    Qmax = jnp.max(Qi, axis=-1)
    Tsas = jnp.einsum('sb,sabz->saz', Xj, T_tensor, optimize='optimal')
    
    gamma_term = jnp.einsum('saz,z->sa', Tsas, Qmax, optimize='optimal')
    return (Rsa + gamma * gamma_term)

batched_j_compute_TD_target_Q_learning = vmap(
    compute_TD_target_Q_learning, in_axes=(None, 0, None, None, None)
)

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
    """
    Compute the velocity vector for the focal agent
    """
    TD_target_for_i = batched_j_compute_TD_target_Q_learning(Qi, X_values, payoff_matrices, T_tensor, gamma) # (n_sep, K, M)
    return alpha * Xi * (jnp.einsum('js,jsa->sa', p_values_for_i, TD_target_for_i, optimize='optimal') - Qi)

batched_compute_mu_values = jit(vmap(
    compute_mu_values, in_axes=(0, 0, 0, 0, None, None, None, None)
))

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
):
    Q_values_history = []
    X_values_history = []
    stationary_dist_history = []
    Q_values_history.append(Q_values)
    X_values_history.append(softmax(Q_values, beta))

    for _ in trange(time_steps):
        X_values = softmax(Q_values, beta)
        neighbor_idx = jnp.argsort(-A, axis=1)[:, :k]
        X_neighbors = X_values[neighbor_idx]
        stationary_dist_values = batched_ij_stationary_dist_func(X_values, X_neighbors, T_tensor)
        p_values = stationary_dist_values / jnp.sum(stationary_dist_values, axis=1, keepdims=True)
        nots_values = jnp.mean(1 - stationary_dist_values, axis=1)
        mu_values = batched_compute_mu_values(Q_values, X_values, X_neighbors, p_values, T_tensor, payoff_matrices, alpha, gamma)
        Q_values = Q_values + (1 - nots_values[..., jnp.newaxis]**k) * mu_values

        stationary_dist_history.append(stationary_dist_values)
        Q_values_history.append(Q_values)
        X_values_history.append(X_values)

    return Q_values_history, X_values_history, stationary_dist_history

def compute_average_s(s_history: jnp.ndarray, A: jnp.ndarray, K: int) -> jnp.ndarray:
    """
    Compute the proportion of each state over time for connected agent pairs.
    
    Parameters
    ----------
    s_history : jnp.ndarray
        State history of shape (B, T, N, N) where B is batch size, T is time steps, N is number of agents.
    A : jnp.ndarray
        Adjacency matrix of shape (N, N) indicating connections between agents.
    K : int
        Total number of states.
        
    Returns
    -------
    jnp.ndarray
        Array of shape (B, T, K) containing the proportion of each state at each time step for each batch.
    """
    B, T, N, _ = s_history.shape
    
    connected_mask = A == 1  # [N, N]

    connected_states = s_history * connected_mask[None, None, :, :]  # [B, T, N, N]
    
    connected_states_flat = connected_states.reshape(B, T, -1)  # [B, T, N*N]
    mask_flat = connected_mask.flatten()  # [N*N]
    
    connected_states_valid = connected_states_flat[:, :, mask_flat]  # [B, T, num_connections]
    
    state_counts = jnp.zeros((B, T, K))
    for k in range(K):
        counts = jnp.sum(connected_states_valid == k, axis=-1)  # [B, T]
        state_counts = state_counts.at[:, :, k].set(counts)
    
    total_connections = connected_states_valid.shape[-1]
    state_proportions = state_counts / total_connections
    
    return state_proportions
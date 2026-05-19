from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, Sequence, Optional, Callable

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np

import networkx as nx
from tqdm import tqdm, trange

from utils.graph_utils import (
    create_random_regular_graph,
    create_lattice_graph,
    adjacency_to_neighbors,
)

@jit
def softmax(Q: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Boltzmann policy over the last axis."""
    return jax.nn.softmax(beta * Q, axis=-1)

@jit
def choose_actions(key: jax.Array, Q: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Sample an action for every (agent, state) pair."""
    logits = beta * Q  # [N, K, M]
    return random.categorical(key, logits, axis=-1)  # [N, K]

@jit
def sample_next_states(key: jax.Array,
                       s: jnp.ndarray,  # [N, N]
                       a: jnp.ndarray,  # [N, K]
                       T: jnp.ndarray  # [K, M, M, K]
                       ) -> jnp.ndarray:  # [N, N]
    """Draw the next environment state for every ordered agent pair (i, j)."""
    N = s.shape[0]
    i_idx = jnp.arange(N)[:, None]  # [N, 1]
    j_idx = jnp.arange(N)[None, :]  # [1, N]

    a_i = a[i_idx, s]  # [N, N]
    a_j = a[j_idx, s]  # [N, N]

    log_probs = jnp.log(T[s, a_i, a_j] + 1e-20)
    return random.categorical(key, log_probs, axis=-1)

@jit
def get_td_errors(payoff_matrices: jnp.ndarray,  # [K, M, M]
                  s_values:      jnp.ndarray,    # [N, N]
                  a_values:      jnp.ndarray,    # [N, K]
                  Q_values:      jnp.ndarray,    # [N, K, M]
                  s_next:        jnp.ndarray,    # [N, N]
                  gamma:         float
                 ) -> jnp.ndarray:               # returns [N, N]
    N = s_values.shape[0]

    i_idx = jnp.arange(N)[:, None]   # [N,1]
    j_idx = jnp.arange(N)[None, :]   # [1,N]

    a_i = a_values[i_idx, s_values]  # [N,N]
    a_j = a_values[j_idx, s_values]  # [N,N]

    r_ij = payoff_matrices[s_values, a_i, a_j]  # [N,N]
    
    Q_cur = Q_values[i_idx, s_values, a_i]      # [N,N]
    
    Q_next_max = jnp.max(Q_values[i_idx, s_next, :], axis=-1)  # [N,N]
    
    td_errors = r_ij + gamma * Q_next_max - Q_cur             # [N,N]
    return td_errors



@partial(jit, static_argnums=(3,))
def compute_average_td(td_matrix: jnp.ndarray,  # [N, N]
                       s_values:  jnp.ndarray,  # [N, N]
                       A:         jnp.ndarray,  # [N, N]
                       K:         int
                      ) -> jnp.ndarray:          # returns [N, K]
    s_eq = (s_values[None, :, :] == jnp.arange(K)[:, None, None]) & A[None, :, :]

    num = jnp.sum(s_eq * td_matrix[None, :, :], axis=2)   # [K, N]
    den = jnp.sum(s_eq,                 axis=2)          # [K, N]
    den_bool = den > 0
    avg = num / (den + 1e-20)                             # [K, N]
    return avg.T, den_bool.T                                # [N, K], [N, K]


def update_Q_values(Q: jnp.ndarray,  # [N, K, M]
                    average_td: jnp.ndarray,  # [N, K]
                    den_bool: jnp.ndarray,  # [N, K]
                    a_values: jnp.ndarray,  # [N, K]
                    M: int,
                    alpha: float
                    ) -> jnp.ndarray:  # [N, K, M]
    onehot_actions = jax.nn.one_hot(a_values, M)         # [N, K, M]
    td_values = average_td[..., None] * onehot_actions   # [N, K, M]
    den_bool_values = den_bool[..., None] * onehot_actions   # [N, K, M]
    Q_next = jnp.where(den_bool_values, Q + alpha * td_values, Q)
    return Q_next


def init_Q_s(key_Q, key_s, N, K, M, s_init_prob=(0.5, 0.5)):
    Q = jax.random.normal(key_Q, shape=[N, K, M])
    Q = (Q - jnp.mean(Q, axis=0)) / jnp.std(Q, axis=0) * 0.1

    Q = Q.at[:, 0, 0].add(0.5)
    Q = Q.at[:, 0, 1].add(0.0)
    Q = Q.at[:, 1, 0].add(0.0)
    Q = Q.at[:, 1, 1].add(0.5)

    if s_init_prob is None:
        s_upper = random.choice(key_s, jnp.array([0, 1]), shape=(N, N))
    else:
        s_upper = random.choice(key_s, jnp.array([0, 1]), shape=(N, N), p=s_init_prob)
    s = jnp.triu(s_upper) + jnp.triu(s_upper, k=1).T  # Make symmetric
    return Q, s

def sim_graph(
    N, K, M, time_steps, alpha, beta, gamma, payoff_matrices, T, A, num_reps=10, init_key=42, s_init_prob=jnp.array([0.5, 0.5])
):
    Q_history_all = []
    X_history_all = []
    s_history_all = []
    for rep in trange(num_reps):
        key = random.PRNGKey(init_key + rep)
        key, key_Q, key_s = random.split(key, 3)
        Q, s = init_Q_s(key_Q, key_s, N, K, M, s_init_prob)
        Q_history = [Q]
        X_history = [softmax(Q, beta)]
        s_history = [s]
        for _ in range(time_steps):
            key, key_a, key_s = random.split(key, 3)
            a = choose_actions(key_a, Q, beta)
            s_next = sample_next_states(key_s, s, a, T)

            td = get_td_errors(payoff_matrices, s, a, Q, s_next, gamma)
            avg_td, den_bool = compute_average_td(td, s, A, K)
            Q = update_Q_values(Q, avg_td, den_bool, a, M, alpha)
            s = s_next
            Q_history.append(Q)
            X_history.append(softmax(Q, beta))
            s_history.append(s)

        Q_history = jnp.array(Q_history)
        X_history = jnp.array(X_history)
        s_history = jnp.array(s_history)
        
        Q_history_all.append(Q_history)
        X_history_all.append(X_history)
        s_history_all.append(s_history)

    Q_history_all = jnp.array(Q_history_all) # (10, time_steps+1, N, K, M)
    X_history_all = jnp.array(X_history_all) # (10, time_steps+1, N, K, M)
    s_history_all = jnp.array(s_history_all) # (10, time_steps+1, N, N)
    return Q_history_all, X_history_all, s_history_all

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
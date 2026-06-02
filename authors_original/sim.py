'''
Agent-based simulation of multi-agent Q-learning in networked pairwise stochastic games.
'''
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
import time
from tqdm import tqdm, trange

import os
os.makedirs("data/fig1", exist_ok=True)
os.makedirs("data/fig2", exist_ok=True)
os.makedirs("data/fig3", exist_ok=True)
os.makedirs("data/fig_SI", exist_ok=True)

from utils import (
    create_random_regular_graph,
    create_lattice_graph,
    adjacency_to_neighbors,
)

@jit
def softmax(Q: jnp.ndarray, beta: float) -> jnp.ndarray:
    return jax.nn.softmax(beta * Q, axis=-1)

@jit
def choose_actions(key: jax.Array, Q: jnp.ndarray, beta: float) -> jnp.ndarray:
    logits = beta * Q
    return random.categorical(key, logits, axis=-1)

@jit
def sample_next_states(key: jax.Array,
                       s: jnp.ndarray,
                       a: jnp.ndarray,
                       T: jnp.ndarray
                       ) -> jnp.ndarray:
    N = s.shape[0]
    i_idx = jnp.arange(N)[:, None]
    j_idx = jnp.arange(N)[None, :]

    a_i = a[i_idx, s]
    a_j = a[j_idx, s]

    log_probs = jnp.log(T[s, a_i, a_j] + 1e-20)
    next_s = random.categorical(key, log_probs, axis=-1)
    s_next_symmetric = jnp.triu(next_s) + jnp.triu(next_s, k=1).T
    return s_next_symmetric

@jit
def get_td_errors(payoff_matrices: jnp.ndarray,
                  s_values:      jnp.ndarray,
                  a_values:      jnp.ndarray,
                  Q_values:      jnp.ndarray,
                  s_next:        jnp.ndarray,
                  gamma:         float
                 ) -> jnp.ndarray:
    N = s_values.shape[0]

    i_idx = jnp.arange(N)[:, None]
    j_idx = jnp.arange(N)[None, :]

    a_i = a_values[i_idx, s_values]
    a_j = a_values[j_idx, s_values]

    r_ij = payoff_matrices[s_values, a_i, a_j]
    
    Q_cur = Q_values[i_idx, s_values, a_i]
    
    Q_next_max = jnp.max(Q_values[i_idx, s_next, :], axis=-1)
    
    td_errors = r_ij + gamma * Q_next_max - Q_cur
    return td_errors



@partial(jit, static_argnums=(3,))
def compute_average_td(td_matrix: jnp.ndarray,
                       s_values:  jnp.ndarray,
                       A:         jnp.ndarray,
                       K:         int
                      ) -> jnp.ndarray:
    s_eq = (
        s_values[None, :, :] == jnp.arange(K)[:, None, None]
    ) & A[None, :, :]

    num = jnp.sum(s_eq * td_matrix[None, :, :], axis=2)
    den = jnp.sum(s_eq, axis=2)
    den_bool = den > 0
    avg = num / (den + 1e-20)
    return avg.T, den_bool.T


@partial(jit, static_argnums=(4,))
def update_Q_values(Q: jnp.ndarray,
                    average_td: jnp.ndarray,
                    den_bool: jnp.ndarray,
                    a_values: jnp.ndarray,
                    M: int,
                    alpha: float
                    ) -> jnp.ndarray:
    onehot_actions = jax.nn.one_hot(a_values, M)
    td_values = average_td[..., None] * onehot_actions
    den_bool_values = den_bool[..., None] * onehot_actions
    Q_next = jnp.where(den_bool_values, Q + alpha * td_values, Q)
    return Q_next


@partial(jit, static_argnums=(2,))
def compute_state_prop_step(s_values: jnp.ndarray,
                            A: jnp.ndarray,
                            K: int
                            ) -> jnp.ndarray:
    onehot = jax.nn.one_hot(s_values, K)
    mask = (A == 1).astype(onehot.dtype)[:, :, None]
    counts = jnp.sum(onehot * mask, axis=(0, 1))
    return counts / jnp.sum(A == 1)


def init_Q_s(key_Q: jax.Array,
             key_s: jax.Array,
             N: int,
             K: int,
             M: int,
             s_init_prob: Optional[Sequence[float]] = (0.5, 0.5),
             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    s = jnp.triu(s_upper) + jnp.triu(s_upper, k=1).T
    return Q, s

def sim_graph(
    N: int,
    K: int,
    M: int,
    time_steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    payoff_matrices: jnp.ndarray,
    T: jnp.ndarray,
    A: jnp.ndarray,
    num_reps: int = 10,
    init_key: int = 42,
    s_init_prob: jnp.ndarray = jnp.array([0.5, 0.5]),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Q_avg_all = []
    X_avg_all = []
    s_prop_all = []
    for rep in trange(num_reps):
        key = random.PRNGKey(init_key + rep)
        Q, s = init_Q_s(
            jax.random.PRNGKey(3154), jax.random.PRNGKey(3155),
            N, K, M, s_init_prob,
        )
        Q_avg_history = [Q.mean(axis=0)]
        X_avg_history = [softmax(Q, beta).mean(axis=0)]
        s_prop_history = [compute_state_prop_step(s, A, K)]
        for _ in range(time_steps):
            key, key_a, key_s = random.split(key, 3)
            a = choose_actions(key_a, Q, beta)
            s_next = sample_next_states(key_s, s, a, T)

            td = get_td_errors(payoff_matrices, s, a, Q, s_next, gamma)
            avg_td, den_bool = compute_average_td(td, s, A, K)
            Q = update_Q_values(Q, avg_td, den_bool, a, M, alpha)
            s = s_next

            Q_avg_history.append(Q.mean(axis=0))
            X_avg_history.append(softmax(Q, beta).mean(axis=0))
            s_prop_history.append(compute_state_prop_step(s, A, K))

        Q_avg_all.append(jnp.array(Q_avg_history))
        X_avg_all.append(jnp.array(X_avg_history))
        s_prop_all.append(jnp.array(s_prop_history))

    Q_avg_all = jnp.array(Q_avg_all)
    X_avg_all = jnp.array(X_avg_all)
    s_prop_all = jnp.array(s_prop_all)
    return Q_avg_all, X_avg_all, s_prop_all

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

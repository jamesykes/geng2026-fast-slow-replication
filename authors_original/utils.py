from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import networkx as nx


def create_random_regular_graph(N: int, k: int, seed: int = 42) -> Tuple[nx.Graph, jnp.ndarray]:
    G = nx.random_regular_graph(k, N, seed=seed)
    
    adj_matrix = nx.adjacency_matrix(G).toarray()
    adj_matrix_jax = jnp.array(adj_matrix)
    
    return G, adj_matrix_jax

def create_lattice_graph(N: int, dim: int = 2, periodic: bool = True) -> Tuple[nx.Graph, jnp.ndarray]:
    if dim == 1:
        if periodic:
            G = nx.cycle_graph(N)
        else:
            G = nx.path_graph(N)
    elif dim == 2:
        side_length = int(np.sqrt(N))
        if side_length * side_length != N:
            raise ValueError(f"For 2D lattice, N must be a perfect square. Got N={N}")
        
        if periodic:
            G = nx.grid_2d_graph(side_length, side_length, periodic=True)
        else:
            G = nx.grid_2d_graph(side_length, side_length, periodic=False)
        
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    else:
        raise ValueError(f"Dimension {dim} not supported. Use dim=1 or dim=2.")
    
    adj_matrix = nx.adjacency_matrix(G).toarray()
    adj_matrix_jax = jnp.array(adj_matrix)
    
    return G, adj_matrix_jax

def adjacency_to_neighbors(adj_matrix: jnp.ndarray) -> jnp.ndarray:
    N = adj_matrix.shape[0]
    
    neighbors_list = []
    degrees = []
    
    for i in range(N):
        neighbor_indices = jnp.where(adj_matrix[i] == 1)[0]
        neighbors_list.append(neighbor_indices)
        degrees.append(len(neighbor_indices))
    
    k = degrees[0]
    if not all(deg == k for deg in degrees):
        raise ValueError(f"Graph is not k-regular. Degrees: {set(degrees)}")
    
    neighbor_matrix = jnp.zeros((N, k), dtype=jnp.int32)
    
    for i in range(N):
        neighbor_matrix = neighbor_matrix.at[i].set(neighbors_list[i])
    
    return neighbor_matrix
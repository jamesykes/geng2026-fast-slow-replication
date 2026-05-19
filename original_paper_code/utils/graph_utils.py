from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import networkx as nx


def create_random_regular_graph(N: int, k: int, seed: int = 42) -> Tuple[nx.Graph, jnp.ndarray]:
    """
    Create a k-regular graph with N nodes and return the graph and adjacency matrix.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    k : int
        Degree of each node (must be even for undirected regular graphs).
    seed : int, optional
        Random seed for reproducible graph generation, by default 42.
        
    Returns
    -------
    Tuple[nx.Graph, jnp.ndarray]
        The NetworkX graph object and its adjacency matrix as a JAX array.
        
    Notes
    -----
    For a k-regular graph to exist, N*k must be even (since each edge contributes
    to the degree of exactly two nodes).
    """
    # Create a k-regular graph
    G = nx.random_regular_graph(k, N, seed=seed)
    
    # Get adjacency matrix as numpy array, then convert to JAX array
    adj_matrix = nx.adjacency_matrix(G).toarray()
    adj_matrix_jax = jnp.array(adj_matrix)
    
    return G, adj_matrix_jax

def create_lattice_graph(N: int, dim: int = 2, periodic: bool = True) -> Tuple[nx.Graph, jnp.ndarray]:
    """
    Create a lattice graph with N nodes and return the graph and adjacency matrix.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph. For 2D lattice, should be a perfect square.
    dim : int, optional
        Dimension of the lattice (1D or 2D), by default 2.
    periodic : bool, optional
        Whether to use periodic boundary conditions (torus topology), by default True.
    seed : int, optional
        Random seed for reproducible graph generation, by default 42.
        
    Returns
    -------
    Tuple[nx.Graph, jnp.ndarray]
        The NetworkX graph object and its adjacency matrix as a JAX array.
        
    Notes
    -----
    For 2D lattice, N should be a perfect square. The lattice will be arranged
    as a sqrt(N) x sqrt(N) grid. With periodic=True, each interior node has
    4 neighbors; with periodic=False, boundary nodes have fewer neighbors.
    """
    
    if dim == 1:
        # 1D lattice (path or cycle)
        if periodic:
            G = nx.cycle_graph(N)
        else:
            G = nx.path_graph(N)
    elif dim == 2:
        # 2D lattice (grid)
        side_length = int(np.sqrt(N))
        if side_length * side_length != N:
            raise ValueError(f"For 2D lattice, N must be a perfect square. Got N={N}")
        
        if periodic:
            G = nx.grid_2d_graph(side_length, side_length, periodic=True)
        else:
            G = nx.grid_2d_graph(side_length, side_length, periodic=False)
        
        # Convert node labels from (i,j) tuples to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    else:
        raise ValueError(f"Dimension {dim} not supported. Use dim=1 or dim=2.")
    
    # Get adjacency matrix as numpy array, then convert to JAX array
    adj_matrix = nx.adjacency_matrix(G).toarray()
    adj_matrix_jax = jnp.array(adj_matrix)
    
    return G, adj_matrix_jax

def adjacency_to_neighbors(adj_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Convert adjacency matrix to neighbor index matrix for k-regular graphs.
    
    Parameters
    ----------
    adj_matrix : jnp.ndarray
        N x N adjacency matrix where adj_matrix[i,j] = 1 if nodes i and j are connected.
        
    Returns
    -------
    jnp.ndarray
        N x k matrix where row i contains the indices of node i's neighbors.
        For k-regular graphs, each node has exactly k neighbors.
        
    Notes
    -----
    This function assumes the graph is k-regular (all nodes have the same degree k).
    If nodes have different degrees, the function will pad with -1 or raise an error.
    """
    N = adj_matrix.shape[0]
    
    # Find neighbors for each node
    neighbors_list = []
    degrees = []
    
    for i in range(N):
        # Get indices of neighbors for node i
        neighbor_indices = jnp.where(adj_matrix[i] == 1)[0]
        neighbors_list.append(neighbor_indices)
        degrees.append(len(neighbor_indices))
    
    # Check if graph is k-regular
    k = degrees[0]
    if not all(deg == k for deg in degrees):
        raise ValueError(f"Graph is not k-regular. Degrees: {set(degrees)}")
    
    # Create N x k matrix
    neighbor_matrix = jnp.zeros((N, k), dtype=jnp.int32)
    
    for i in range(N):
        neighbor_matrix = neighbor_matrix.at[i].set(neighbors_list[i])
    
    return neighbor_matrix
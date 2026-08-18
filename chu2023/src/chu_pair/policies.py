"""Action-selection policies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _two_action_boltzmann_probabilities(q_array, tau, array_namespace):
    """Backend-neutral stable policy kernel for NumPy and JAX arrays."""

    logit = tau * (q_array[..., 0] - q_array[..., 1])
    cooperate = array_namespace.exp(-array_namespace.logaddexp(0.0, -logit))
    defect = 1.0 - cooperate
    return array_namespace.stack((cooperate, defect), axis=-1)


def boltzmann_probabilities(q: ArrayLike, tau: float) -> NDArray[np.float64]:
    """Stable two-action Boltzmann probabilities in action order (C, D)."""

    q_array = np.asarray(q, dtype=np.float64)
    if q_array.shape == () or q_array.shape[-1] != 2:
        raise ValueError(f"q must have final dimension 2, got {q_array.shape}")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q must contain only finite values")

    return _two_action_boltzmann_probabilities(q_array, tau, np)

"""Pair-mass observables and diagnostics without silent renormalisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import LearningConfig
from .grids import QGrid
from .pair_density.numpy_reference import (
    DEFAULT_MAX_REFERENCE_ELEMENTS,
    OneEdgeMoments,
    ReferenceSizeError,
    endpoint_exchange_symmetry_error,
    one_edge_moments,
)
from .policies import boltzmann_probabilities


@dataclass(frozen=True, slots=True)
class PairDiagnostics:
    total_mass: float
    state_masses: NDArray[np.float64]
    mean_q: NDArray[np.float64]
    mean_action_probability: NDArray[np.float64]
    symmetry_error: float
    minimum_mass: float
    nonnegative: bool
    conditional_weight_error: float
    minimum_conditional_variance: float
    conditional_moments_valid: bool


def pair_diagnostics(
    pair_mass: NDArray[np.float64],
    grid: QGrid,
    learning: LearningConfig,
    *,
    moments: OneEdgeMoments | None = None,
    tolerance: float = 1e-12,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> PairDiagnostics:
    raw_mass = np.asarray(pair_mass)
    expected_shape = (grid.size, grid.size, 2, grid.size, grid.size)
    if raw_mass.shape != expected_shape:
        raise ValueError(f"pair mass must have shape {expected_shape}, got {raw_mass.shape}")
    if max_elements is not None and raw_mass.size > max_elements:
        raise ReferenceSizeError(
            f"reference pair has {raw_mass.size:,} elements, above limit {max_elements:,}"
        )
    mass = np.asarray(raw_mass, dtype=np.float64)
    if not np.all(np.isfinite(mass)):
        raise ValueError("pair mass must contain only finite values")

    total = float(mass.sum())
    state_masses = mass.sum(axis=(0, 1, 3, 4))
    focal = mass.sum(axis=(2, 3, 4))
    minimum = float(np.min(mass))
    nonnegative = minimum >= -tolerance

    if total > 0.0:
        mean_q = (focal[..., None] * grid.q_points).sum(axis=(0, 1)) / total
        policies = boltzmann_probabilities(grid.q_points, learning.tau)
        mean_policy = (focal[..., None] * policies).sum(axis=(0, 1)) / total
    else:
        mean_q = np.full(2, np.nan, dtype=np.float64)
        mean_policy = np.full(2, np.nan, dtype=np.float64)

    weight_error = np.inf
    minimum_variance = np.nan
    moments_valid = False
    if nonnegative:
        if moments is None:
            moments = one_edge_moments(
                mass,
                grid,
                learning,
                max_elements=max_elements,
            )
        if np.any(moments.occupied):
            weight_sums = moments.state_opponent_action_probability.sum(axis=(2, 3))
            weight_error = float(np.max(np.abs(weight_sums[moments.occupied] - 1.0)))
            minimum_variance = float(np.min(moments.variance[moments.occupied]))
            empty_zero = bool(
                np.all(moments.mean[~moments.occupied] == 0.0)
                and np.all(moments.second[~moments.occupied] == 0.0)
            )
            moments_valid = bool(
                weight_error <= tolerance
                and minimum_variance >= -tolerance
                and np.all(np.isfinite(moments.mean[moments.occupied]))
                and np.all(np.isfinite(moments.second[moments.occupied]))
                and empty_zero
            )
        else:
            weight_error = 0.0
            minimum_variance = 0.0
            moments_valid = True

    return PairDiagnostics(
        total_mass=total,
        state_masses=state_masses,
        mean_q=mean_q,
        mean_action_probability=mean_policy,
        symmetry_error=endpoint_exchange_symmetry_error(mass, max_elements=max_elements),
        minimum_mass=minimum,
        nonnegative=nonnegative,
        conditional_weight_error=weight_error,
        minimum_conditional_variance=minimum_variance,
        conditional_moments_valid=moments_valid,
    )

"""Readable NumPy oracle for conditional moments and pair-mass transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import LearningConfig
from ..grids import QGrid
from ..model import ACTION_ORDER, STATE_ORDER, PAYOFF_TENSOR, TRANSITION_TENSOR
from ..policies import boltzmann_probabilities


DEFAULT_MAX_REFERENCE_ELEMENTS = 5_000_000


class ReferenceSizeError(MemoryError):
    """Raised when the deliberately small reference is given a large pair array."""


class PairSymmetryError(ValueError):
    """Raised when the endpoint-symmetric reference assumption is violated."""


@dataclass(frozen=True, slots=True)
class OneEdgeMoments:
    focal_mass: NDArray[np.float64]
    state_opponent_action_probability: NDArray[np.float64]
    mean: NDArray[np.float64]
    second: NDArray[np.float64]
    variance: NDArray[np.float64]
    occupied: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ConditionalDynamics:
    focal_mass: NDArray[np.float64]
    expected_payoff: NDArray[np.float64]
    velocity: NDArray[np.float64]
    occupied: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PairStepResult:
    mass: NDArray[np.float64]
    dynamics: ConditionalDynamics
    destination_indices: NDArray[np.int64]


def expected_pair_shape(grid: QGrid) -> tuple[int, int, int, int, int]:
    return (grid.size, grid.size, 2, grid.size, grid.size)


def validate_pair_mass(
    pair_mass: NDArray[np.float64],
    grid: QGrid,
    *,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> NDArray[np.float64]:
    raw_mass = np.asarray(pair_mass)
    if raw_mass.shape != expected_pair_shape(grid):
        raise ValueError(
            f"pair mass must have shape {expected_pair_shape(grid)}, got {raw_mass.shape}"
        )
    if max_elements is not None and raw_mass.size > max_elements:
        raise ReferenceSizeError(
            f"reference pair has {raw_mass.size:,} elements, above limit {max_elements:,}"
        )
    mass = np.asarray(raw_mass, dtype=np.float64)
    if not np.all(np.isfinite(mass)):
        raise ValueError("pair mass must contain only finite values")
    if np.any(mass < 0.0):
        raise ValueError("pair mass must be non-negative")
    return mass


def focal_marginal(
    pair_mass: NDArray[np.float64],
    grid: QGrid,
    *,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> NDArray[np.float64]:
    mass = validate_pair_mass(pair_mass, grid, max_elements=max_elements)
    return mass.sum(axis=(2, 3, 4))


def endpoint_exchange_symmetry_error(
    pair_mass: NDArray[np.float64],
    *,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> float:
    raw_mass = np.asarray(pair_mass)
    if raw_mass.ndim != 5:
        raise ValueError("pair mass must have five axes")
    if raw_mass.shape[:2] != raw_mass.shape[3:5]:
        raise ValueError("endpoint Q axes must have matching sizes")
    if max_elements is not None and raw_mass.size > max_elements:
        raise ReferenceSizeError(
            f"reference pair has {raw_mass.size:,} elements, above limit {max_elements:,}"
        )
    mass = np.asarray(raw_mass, dtype=np.float64)
    exchanged = mass.transpose(3, 4, 2, 0, 1)
    return float(np.max(np.abs(mass - exchanged), initial=0.0))


def one_edge_moments(
    pair_mass: NDArray[np.float64],
    grid: QGrid,
    learning: LearningConfig,
    *,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> OneEdgeMoments:
    """Calculate w(s,b|q), mean, second moment, and variance for both actions."""

    mass = validate_pair_mass(pair_mass, grid, max_elements=max_elements)
    focal = mass.sum(axis=(2, 3, 4))
    occupied = focal > 0.0
    opponent_policy = boltzmann_probabilities(grid.q_points, learning.tau)

    weights = np.zeros((grid.size, grid.size, 2, 2), dtype=np.float64)
    mean = np.zeros((grid.size, grid.size, 2), dtype=np.float64)
    second = np.zeros_like(mean)

    for i in range(grid.size):
        for j in range(grid.size):
            if not occupied[i, j]:
                continue

            denominator = focal[i, j]
            for state in STATE_ORDER:
                state_index = int(state)
                for k in range(grid.size):
                    for m in range(grid.size):
                        conditional_edge_mass = mass[i, j, state_index, k, m] / denominator
                        if conditional_edge_mass == 0.0:
                            continue
                        for opponent_action in ACTION_ORDER:
                            action_index = int(opponent_action)
                            weights[i, j, state_index, action_index] += (
                                conditional_edge_mass * opponent_policy[k, m, action_index]
                            )

            for own_action in ACTION_ORDER:
                own_index = int(own_action)
                for state in STATE_ORDER:
                    state_index = int(state)
                    for opponent_action in ACTION_ORDER:
                        opponent_index = int(opponent_action)
                        probability = weights[i, j, state_index, opponent_index]
                        reward = PAYOFF_TENSOR[state_index, own_index, opponent_index]
                        mean[i, j, own_index] += probability * reward
                        second[i, j, own_index] += probability * reward * reward

    variance = second - mean * mean
    return OneEdgeMoments(
        focal_mass=focal,
        state_opponent_action_probability=weights,
        mean=mean,
        second=second,
        variance=variance,
        occupied=occupied,
    )


def conditional_dynamics(
    pair_mass: NDArray[np.float64],
    grid: QGrid,
    learning: LearningConfig,
    *,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> ConditionalDynamics:
    moments = one_edge_moments(
        pair_mass,
        grid,
        learning,
        max_elements=max_elements,
    )
    velocity = np.zeros_like(moments.mean)
    q_points = grid.q_points

    for i in range(grid.size):
        for j in range(grid.size):
            if not moments.occupied[i, j]:
                continue
            for action in ACTION_ORDER:
                action_index = int(action)
                velocity[i, j, action_index] = learning.alpha * (
                    moments.mean[i, j, action_index] - q_points[i, j, action_index]
                )

    return ConditionalDynamics(
        focal_mass=moments.focal_mass,
        expected_payoff=moments.mean,
        velocity=velocity,
        occupied=moments.occupied,
    )


def legacy_destination_indices(
    dynamics: ConditionalDynamics,
    grid: QGrid,
) -> NDArray[np.int64]:
    """Map every occupied Q-cell/action to the legacy projected destination."""

    destinations = np.empty((grid.size, grid.size, 2, 2), dtype=np.int64)
    for i in range(grid.size):
        for j in range(grid.size):
            destinations[i, j, :, :] = (i, j)
            if not dynamics.occupied[i, j]:
                continue

            new_q_c = grid.values[i] + dynamics.velocity[i, j, 0]
            destinations[i, j, 0] = (grid.legacy_project_index(new_q_c), j)

            new_q_d = grid.values[j] + dynamics.velocity[i, j, 1]
            destinations[i, j, 1] = (i, grid.legacy_project_index(new_q_d))
    return destinations


def pair_mass_step(
    pair_mass: NDArray[np.float64],
    grid: QGrid,
    learning: LearningConfig,
    *,
    symmetry_tolerance: float = 1e-12,
    max_elements: int | None = DEFAULT_MAX_REFERENCE_ELEMENTS,
) -> PairStepResult:
    """Apply one synchronous four-joint-action legacy pair-mass pushforward."""

    mass = validate_pair_mass(pair_mass, grid, max_elements=max_elements)
    symmetry_error = endpoint_exchange_symmetry_error(mass, max_elements=max_elements)
    if symmetry_error > symmetry_tolerance:
        raise PairSymmetryError(
            "the reference reuses first-endpoint velocities for the second endpoint; "
            f"exchange symmetry error {symmetry_error} exceeds {symmetry_tolerance}"
        )

    dynamics = conditional_dynamics(mass, grid, learning, max_elements=max_elements)
    destinations = legacy_destination_indices(dynamics, grid)
    source_policy = boltzmann_probabilities(grid.q_points, learning.tau)
    next_mass = np.zeros_like(mass)

    for i, j, old_state, k, m in np.argwhere(mass > 0.0):
        source_mass = mass[i, j, old_state, k, m]
        if not dynamics.occupied[k, m]:
            raise PairSymmetryError("positive source mass has an unoccupied second-endpoint type")

        for own_action in ACTION_ORDER:
            own_index = int(own_action)
            new_i, new_j = destinations[i, j, own_index]
            for opponent_action in ACTION_ORDER:
                opponent_index = int(opponent_action)
                new_k, new_m = destinations[k, m, opponent_index]
                new_state = int(TRANSITION_TENSOR[old_state, own_index, opponent_index])
                branch_mass = (
                    source_mass
                    * source_policy[i, j, own_index]
                    * source_policy[k, m, opponent_index]
                )
                next_mass[new_i, new_j, new_state, new_k, new_m] += branch_mass

    return PairStepResult(
        mass=next_mass,
        dynamics=dynamics,
        destination_indices=destinations,
    )

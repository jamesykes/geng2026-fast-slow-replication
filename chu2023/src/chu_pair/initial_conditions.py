"""Reproducible one-agent histograms and ordered initial pair masses."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from numpy.typing import NDArray

from .grids import QGrid


DEFAULT_MAX_REFERENCE_PAIR_ELEMENTS = 5_000_000


class PairAllocationError(MemoryError):
    """Raised before an unexpectedly large reference pair allocation."""


@dataclass(frozen=True, slots=True)
class DiscreteQHistogram:
    """A reproducible one-agent Q mass that can also seed a later ABM."""

    grid: QGrid
    mass: NDArray[np.float64]
    seed: int | None = None
    sample_count: int | None = None
    mode: str = "provided"

    def __post_init__(self) -> None:
        mass = np.array(self.mass, dtype=np.float64, copy=True)
        if mass.shape != (self.grid.size, self.grid.size):
            raise ValueError(
                f"histogram must have shape {(self.grid.size, self.grid.size)}, got {mass.shape}"
            )
        if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
            raise ValueError("histogram mass must be finite and non-negative")
        if not np.isclose(float(mass.sum()), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("histogram mass must sum to one")
        mass.setflags(write=False)
        object.__setattr__(self, "mass", mass)

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        """Sample initial Q-vectors from the stored discrete mass."""

        if size < 0:
            raise ValueError("size must be non-negative")
        flat = rng.choice(self.grid.agent_point_count, size=size, p=self.mass.ravel())
        indices = np.column_stack(np.unravel_index(flat, self.mass.shape))
        values = self.grid.values
        return np.column_stack((values[indices[:, 0]], values[indices[:, 1]]))


def seeded_legacy_histogram(
    grid: QGrid,
    *,
    seed: int,
    sample_count: int | None = None,
    samples_per_grid_cell: int = 10,
    beta_c: tuple[float, float] = (20.0, 80.0),
    beta_d: tuple[float, float] = (80.0, 20.0),
) -> DiscreteQHistogram:
    """Reproduce the original scaled-Beta draw order with a local RNG."""

    if samples_per_grid_cell <= 0:
        raise ValueError("samples_per_grid_cell must be positive")
    if any(parameter <= 0.0 for parameter in (*beta_c, *beta_d)):
        raise ValueError("Beta shape parameters must be positive")
    if sample_count is None:
        sample_count = grid.agent_point_count * samples_per_grid_cell
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    rng = random.Random(seed)
    counts = np.zeros((grid.size, grid.size), dtype=np.int64)
    span = grid.q_max - grid.q_min

    for _ in range(sample_count):
        q_c = grid.q_min + span * rng.betavariate(*beta_c)
        q_d = grid.q_min + span * rng.betavariate(*beta_d)
        i = grid.legacy_project_index(q_c)
        j = grid.legacy_project_index(q_d)
        counts[i, j] += 1

    mass = counts.astype(np.float64) / sample_count
    return DiscreteQHistogram(
        grid=grid,
        mass=mass,
        seed=seed,
        sample_count=sample_count,
        mode="seeded_legacy_histogram",
    )


def tiny_histogram(grid: QGrid) -> DiscreteQHistogram:
    """Return a deterministic two-cell histogram for small unit tests."""

    mass = np.zeros((grid.size, grid.size), dtype=np.float64)
    mass[0, 0] = 0.5
    mass[-1, -1] = 0.5
    return DiscreteQHistogram(grid=grid, mass=mass, mode="tiny_two_cell")


def ordered_pair_mass(
    histogram: DiscreteQHistogram,
    *,
    state_probabilities: tuple[float, float] = (0.5, 0.5),
    max_elements: int | None = DEFAULT_MAX_REFERENCE_PAIR_ELEMENTS,
) -> NDArray[np.float64]:
    """Construct P(q1) P(q2) P(state) as ordered pair probability mass."""

    state_mass = np.asarray(state_probabilities, dtype=np.float64)
    if state_mass.shape != (2,) or np.any(state_mass < 0.0):
        raise ValueError("state_probabilities must be two non-negative values")
    if not np.isclose(float(state_mass.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("state probabilities must sum to one")

    pair_elements = 2 * histogram.grid.size**4
    if max_elements is not None and pair_elements > max_elements:
        raise PairAllocationError(
            f"reference pair needs {pair_elements:,} elements, above limit {max_elements:,}"
        )

    return np.einsum(
        "ij,g,km->ijgkm",
        histogram.mass,
        state_mass,
        histogram.mass,
        optimize=True,
    )


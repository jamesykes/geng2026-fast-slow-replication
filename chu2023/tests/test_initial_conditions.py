from __future__ import annotations

import random

import numpy as np
import pytest

from chu_pair.grids import QGrid
from chu_pair.initial_conditions import (
    PairAllocationError,
    ordered_pair_mass,
    seeded_legacy_histogram,
    tiny_histogram,
)


def test_seeded_legacy_histogram_is_exactly_reproducible(coarse_grid: QGrid) -> None:
    first = seeded_legacy_histogram(coarse_grid, seed=17, sample_count=12)
    second = seeded_legacy_histogram(coarse_grid, seed=17, sample_count=12)

    expected_counts = np.zeros((5, 5), dtype=np.float64)
    expected_counts[0, 3] = 1
    expected_counts[1, 3] = 10
    expected_counts[1, 4] = 1
    np.testing.assert_array_equal(first.mass, expected_counts / 12)
    np.testing.assert_array_equal(first.mass, second.mass)
    assert first.seed == 17
    assert first.sample_count == 12


def test_seeded_histogram_does_not_touch_global_random_state(coarse_grid: QGrid) -> None:
    random.seed(12345)
    expected_next = random.random()
    random.seed(12345)
    seeded_legacy_histogram(coarse_grid, seed=99, sample_count=8)
    assert random.random() == expected_next


def test_initial_ordered_pair_mass_and_uniform_states(coarse_grid: QGrid) -> None:
    histogram = tiny_histogram(coarse_grid)
    pair = ordered_pair_mass(histogram)

    expected = np.zeros_like(pair)
    occupied = ((0, 0), (coarse_grid.size - 1, coarse_grid.size - 1))
    for state in range(2):
        for first_i, first_j in occupied:
            for second_i, second_j in occupied:
                expected[first_i, first_j, state, second_i, second_j] = 0.125

    assert pair.shape == (5, 5, 2, 5, 5)
    np.testing.assert_array_equal(pair, expected)
    assert pair.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pair.sum(axis=(0, 1, 3, 4)), [0.5, 0.5])
    np.testing.assert_array_equal(pair.sum(axis=(2, 3, 4)), histogram.mass)
    assert np.max(np.abs(pair - pair.transpose(3, 4, 2, 0, 1))) == 0.0


def test_histogram_can_sample_grid_matched_initial_q(coarse_grid: QGrid) -> None:
    histogram = tiny_histogram(coarse_grid)
    samples = histogram.sample(np.random.default_rng(7), size=20)
    assert samples.shape == (20, 2)
    assert set(map(tuple, samples)).issubset({(-1.0, -1.0), (1.0, 1.0)})


def test_full_legacy_pair_allocation_is_rejected_by_default() -> None:
    histogram = tiny_histogram(QGrid())
    with pytest.raises(PairAllocationError, match="588,999,842"):
        ordered_pair_mass(histogram)

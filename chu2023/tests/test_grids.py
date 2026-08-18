from __future__ import annotations

import numpy as np
import pytest

from chu_pair.grids import GridBoundsError, GridError, QGrid


def test_case_grid_construction_and_index_round_trip() -> None:
    grid = QGrid()
    assert grid.size == 131
    assert grid.agent_point_count == 17_161
    assert grid.values[0] == -0.1
    assert grid.values[-1] == 1.2
    assert grid.q_points.shape == (131, 131, 2)

    for indices in ((0, 0), (17, 29), (130, 130)):
        flat = grid.flatten_index(*indices)
        assert grid.unflatten_index(flat) == indices


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (-0.096, -0.10),
        (0.123, 0.12),
        (0.126, 0.13),
        (1.196, 1.20),
    ],
)
def test_exact_active_legacy_projection(number: float, expected: float) -> None:
    assert QGrid().legacy_project_value(number) == pytest.approx(expected)


def test_legacy_projection_for_coarser_aligned_grid() -> None:
    grid = QGrid(q_min=-1.0, q_max=1.0, spacing=0.5)
    projected = [grid.legacy_project_value(value) for value in (-0.76, -0.24, 0.24, 0.26, 0.74)]
    np.testing.assert_allclose(projected, [-1.0, 0.0, 0.0, 0.5, 0.5])


@pytest.mark.parametrize("number", [-0.111, 1.211])
def test_out_of_range_projection_is_explicit(number: float) -> None:
    with pytest.raises(GridBoundsError, match="outside"):
        QGrid().legacy_project_index(number)


def test_misaligned_legacy_grid_is_rejected() -> None:
    with pytest.raises(GridError, match="align"):
        QGrid(q_min=-0.1, q_max=1.1, spacing=0.3)


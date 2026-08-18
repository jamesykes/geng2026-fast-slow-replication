from __future__ import annotations

import numpy as np
import pytest

import chu_pair.pair_density.numpy_reference as numpy_reference
from chu_pair.config import LearningConfig
from chu_pair.grids import QGrid
from chu_pair.model import Action, State
from chu_pair.observables import pair_diagnostics
from chu_pair.pair_density.numpy_reference import conditional_dynamics, one_edge_moments


def test_hand_calculated_one_edge_moments(coarse_grid: QGrid) -> None:
    pair = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    centre = 2
    pair[centre, centre, State.SH, centre, centre] = 0.25
    pair[centre, centre, State.PD, centre, centre] = 0.75

    moments = one_edge_moments(pair, coarse_grid, LearningConfig(alpha=0.4, tau=0.0))

    np.testing.assert_allclose(
        moments.state_opponent_action_probability[centre, centre],
        [[0.125, 0.125], [0.375, 0.375]],
    )
    expected_mean = np.array([0.4625, 0.475])
    expected_second = np.array([0.50375, 0.5425])
    np.testing.assert_allclose(moments.mean[centre, centre], expected_mean)
    np.testing.assert_allclose(moments.second[centre, centre], expected_second)
    np.testing.assert_allclose(
        moments.variance[centre, centre],
        expected_second - expected_mean**2,
    )


def test_deterministic_payoff_has_zero_single_edge_variance(coarse_grid: QGrid) -> None:
    pair = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    centre = 2
    pair[centre, centre, State.SH, centre, centre] = 1.0

    moments = one_edge_moments(pair, coarse_grid, LearningConfig(alpha=0.4, tau=2.0))

    # In SH, own D pays 0.1 against either opponent action.
    assert moments.mean[centre, centre, Action.D] == pytest.approx(0.1)
    assert moments.second[centre, centre, Action.D] == pytest.approx(0.01)
    assert moments.variance[centre, centre, Action.D] == pytest.approx(0.0, abs=1e-17)


def test_unoccupied_focal_cell_is_handled_safely(coarse_grid: QGrid) -> None:
    pair = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    pair[2, 2, State.SH, 2, 2] = 1.0
    learning = LearningConfig(alpha=0.4, tau=2.0)

    moments = one_edge_moments(pair, coarse_grid, learning)
    dynamics = conditional_dynamics(pair, coarse_grid, learning)

    assert not moments.occupied[0, 0]
    np.testing.assert_array_equal(moments.state_opponent_action_probability[0, 0], 0.0)
    np.testing.assert_array_equal(moments.mean[0, 0], 0.0)
    np.testing.assert_array_equal(moments.second[0, 0], 0.0)
    np.testing.assert_array_equal(moments.variance[0, 0], 0.0)
    np.testing.assert_array_equal(dynamics.velocity[0, 0], 0.0)


def test_conditional_velocity_is_alpha_times_mean_minus_q(coarse_grid: QGrid) -> None:
    pair = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    centre = 2
    pair[centre, centre, State.PD, centre, centre] = 1.0

    dynamics = conditional_dynamics(pair, coarse_grid, LearningConfig(alpha=0.4, tau=0.0))

    np.testing.assert_allclose(dynamics.expected_payoff[centre, centre], [0.45, 0.6])
    np.testing.assert_allclose(dynamics.velocity[centre, centre], [0.18, 0.24])


@pytest.mark.parametrize("entry_point", ["validate", "symmetry", "diagnostics"])
def test_over_limit_float32_is_rejected_before_float64_conversion(
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    grid = QGrid(q_min=0.0, q_max=0.5, spacing=0.5)
    pair = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    original_asarray = np.asarray

    def reject_float64_conversion(array, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if dtype is not None and np.dtype(dtype) == np.dtype(np.float64):
            raise AssertionError("float64 conversion occurred before the size guard")
        return original_asarray(array, *args, **kwargs)

    monkeypatch.setattr(numpy_reference.np, "asarray", reject_float64_conversion)
    limit = pair.size - 1

    with pytest.raises(numpy_reference.ReferenceSizeError):
        if entry_point == "validate":
            numpy_reference.validate_pair_mass(pair, grid, max_elements=limit)
        elif entry_point == "symmetry":
            numpy_reference.endpoint_exchange_symmetry_error(pair, max_elements=limit)
        else:
            pair_diagnostics(pair, grid, LearningConfig(), max_elements=limit)

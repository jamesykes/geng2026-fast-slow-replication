from __future__ import annotations

import numpy as np
import pytest

from chu_pair.config import LearningConfig
from chu_pair.model import Action, State
from chu_pair.observables import pair_diagnostics
from chu_pair.pair_density.numpy_reference import PairSymmetryError, pair_mass_step
from chu_pair.policies import boltzmann_probabilities


def single_pair_mass(grid, state: int, q_index: tuple[int, int]) -> np.ndarray:
    mass = np.zeros((grid.size, grid.size, 2, grid.size, grid.size), dtype=np.float64)
    i, j = q_index
    mass[i, j, state, i, j] = 1.0
    return mass


def test_hand_calculated_pd_transport_moves_only_selected_coordinates(coarse_grid) -> None:
    centre = 2
    source = single_pair_mass(coarse_grid, State.PD, (centre, centre))
    result = pair_mass_step(source, coarse_grid, LearningConfig(alpha=1.0, tau=0.0))

    # In PD against a 50/50 opponent, E[r_C]=0.45 and E[r_D]=0.6.
    # Both project from 0 to 0.5 on this grid. Each joint action has mass 1/4.
    expected = np.zeros_like(source)
    expected[3, 2, State.SH, 3, 2] = 0.25  # CC
    expected[3, 2, State.PD, 2, 3] = 0.25  # CD
    expected[2, 3, State.PD, 3, 2] = 0.25  # DC
    expected[2, 3, State.PD, 2, 3] = 0.25  # DD

    np.testing.assert_array_equal(result.mass, expected)
    np.testing.assert_allclose(result.dynamics.expected_payoff[centre, centre], [0.45, 0.6])
    np.testing.assert_array_equal(result.destination_indices[centre, centre, 0], [3, 2])
    np.testing.assert_array_equal(result.destination_indices[centre, centre, 1], [2, 3])


def test_old_sh_state_supplies_payoff_and_transition_input(coarse_grid) -> None:
    centre = 2
    source = single_pair_mass(coarse_grid, State.SH, (centre, centre))
    result = pair_mass_step(source, coarse_grid, LearningConfig(alpha=1.0, tau=0.0))

    # Old SH gives E[r_C]=0.5 and E[r_D]=0.1. D therefore projects back to 0,
    # while the same Q/opponent mix in PD would have moved D to 0.5.
    np.testing.assert_allclose(result.dynamics.expected_payoff[centre, centre], [0.5, 0.1])
    assert result.mass[centre, centre, State.PD, centre, centre] == pytest.approx(0.25)
    assert result.mass[:, :, State.PD].sum() == pytest.approx(0.25)
    assert result.mass[:, :, State.SH].sum() == pytest.approx(0.75)


def test_heterogeneous_nonzero_tau_transport_is_exact(coarse_grid) -> None:
    # Grid values are (-1, -0.5, 0, 0.5, 1). The two source Q-types are
    # endpoint-exchanged, so the ordered pair distribution is symmetric.
    type_a = (1, 3)  # Q_A = (-0.5, 0.5)
    type_b = (3, 1)  # Q_B = (0.5, -0.5)
    source = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    source[type_a[0], type_a[1], State.SH, type_b[0], type_b[1]] = 0.5
    source[type_b[0], type_b[1], State.SH, type_a[0], type_a[1]] = 0.5

    # Since tau=log(3), sigmoid(tau * -1)=1/4 and sigmoid(tau * 1)=3/4.
    tau = float(np.log(3.0))
    expected_policy_a = np.array([0.25, 0.75])
    expected_policy_b = np.array([0.75, 0.25])
    np.testing.assert_allclose(
        boltzmann_probabilities(np.array([[-0.5, 0.5], [0.5, -0.5]]), tau),
        np.array([expected_policy_a, expected_policy_b]),
        rtol=0.0,
        atol=1e-15,
    )

    # Old-state SH payoffs are M_SH = [[1, 0], [0.1, 0.1]]. Thus A faces
    # B's (3/4, 1/4) policy and B faces A's (1/4, 3/4) policy.
    expected_payoff_a = np.array([0.75, 0.1])
    expected_payoff_b = np.array([0.25, 0.1])
    expected_velocity_a = expected_payoff_a - np.array([-0.5, 0.5])
    expected_velocity_b = expected_payoff_b - np.array([0.5, -0.5])

    result = pair_mass_step(source, coarse_grid, LearningConfig(alpha=1.0, tau=tau))

    np.testing.assert_allclose(
        result.dynamics.velocity[type_a[0], type_a[1]],
        expected_velocity_a,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.dynamics.velocity[type_b[0], type_b[1]],
        expected_velocity_b,
        rtol=0.0,
        atol=1e-15,
    )

    # Nearest-grid selected-coordinate destinations, calculated directly:
    # A_C=(0.5,0.5), A_D=(-0.5,0), B_C=(0,-0.5), B_D=(0.5,0).
    a_c = (3, 3)
    a_d = (1, 2)
    b_c = (2, 1)
    b_d = (3, 2)
    np.testing.assert_array_equal(
        result.destination_indices[type_a[0], type_a[1], Action.C], a_c
    )
    np.testing.assert_array_equal(
        result.destination_indices[type_a[0], type_a[1], Action.D], a_d
    )
    np.testing.assert_array_equal(
        result.destination_indices[type_b[0], type_b[1], Action.C], b_c
    )
    np.testing.assert_array_equal(
        result.destination_indices[type_b[0], type_b[1], Action.D], b_d
    )

    # Each source orientation has mass 1/2. Its CC/CD/DC/DD weights are
    # respectively (3,1,9,3)/32 for A->B and (3,9,1,3)/32 for B->A.
    # In old SH, only DD transitions to PD.
    expected = np.zeros_like(source)
    expected[a_c[0], a_c[1], State.SH, b_c[0], b_c[1]] = 3.0 / 32.0
    expected[a_c[0], a_c[1], State.SH, b_d[0], b_d[1]] = 1.0 / 32.0
    expected[a_d[0], a_d[1], State.SH, b_c[0], b_c[1]] = 9.0 / 32.0
    expected[a_d[0], a_d[1], State.PD, b_d[0], b_d[1]] = 3.0 / 32.0
    expected[b_c[0], b_c[1], State.SH, a_c[0], a_c[1]] = 3.0 / 32.0
    expected[b_c[0], b_c[1], State.SH, a_d[0], a_d[1]] = 9.0 / 32.0
    expected[b_d[0], b_d[1], State.SH, a_c[0], a_c[1]] = 1.0 / 32.0
    expected[b_d[0], b_d[1], State.PD, a_d[0], a_d[1]] = 3.0 / 32.0

    assert result.mass[a_c[0], a_c[1], State.SH, b_c[0], b_c[1]] == pytest.approx(
        3.0 / 32.0
    )
    assert result.mass[a_d[0], a_d[1], State.PD, b_d[0], b_d[1]] == pytest.approx(
        3.0 / 32.0
    )
    np.testing.assert_allclose(result.mass, expected, rtol=0.0, atol=1e-15)
    assert result.mass.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(
        result.mass,
        result.mass.transpose(3, 4, 2, 0, 1),
        rtol=0.0,
        atol=1e-15,
    )


def test_mass_nonnegativity_and_symmetry_over_several_steps(coarse_grid) -> None:
    centre = 2
    mass = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    mass[centre, centre, State.SH, centre, centre] = 0.5
    mass[centre, centre, State.PD, centre, centre] = 0.5
    learning = LearningConfig(alpha=0.4, tau=2.0)

    for _ in range(6):
        result = pair_mass_step(mass, coarse_grid, learning)
        mass = result.mass
        diagnostics = pair_diagnostics(mass, coarse_grid, learning)
        assert diagnostics.total_mass == pytest.approx(1.0, abs=2e-14)
        assert diagnostics.nonnegative
        assert diagnostics.minimum_mass >= 0.0
        assert diagnostics.symmetry_error <= 2e-15
        assert diagnostics.conditional_moments_valid


def test_asymmetric_pair_mass_is_rejected(coarse_grid) -> None:
    source = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    source[1, 2, State.SH, 3, 2] = 1.0

    with pytest.raises(PairSymmetryError, match="exchange symmetry"):
        pair_mass_step(source, coarse_grid, LearningConfig())

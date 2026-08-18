from __future__ import annotations

import numpy as np
import pytest

from chu_pair.config import LearningConfig
from chu_pair.model import State
from chu_pair.observables import pair_diagnostics


def test_observables_and_conditional_validity(coarse_grid) -> None:
    centre = 2
    mass = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    mass[centre, centre, State.SH, centre, centre] = 0.25
    mass[centre, centre, State.PD, centre, centre] = 0.75

    diagnostics = pair_diagnostics(mass, coarse_grid, LearningConfig(alpha=0.4, tau=0.0))

    assert diagnostics.total_mass == pytest.approx(1.0)
    np.testing.assert_allclose(diagnostics.state_masses, [0.25, 0.75])
    np.testing.assert_allclose(diagnostics.mean_q, [0.0, 0.0])
    np.testing.assert_allclose(diagnostics.mean_action_probability, [0.5, 0.5])
    assert diagnostics.symmetry_error == 0.0
    assert diagnostics.minimum_mass == 0.0
    assert diagnostics.nonnegative
    assert diagnostics.conditional_weight_error <= 1e-15
    assert diagnostics.minimum_conditional_variance >= -1e-15
    assert diagnostics.conditional_moments_valid


def test_diagnostics_report_mass_error_without_renormalising(coarse_grid) -> None:
    centre = 2
    mass = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    mass[centre, centre, State.SH, centre, centre] = 0.2
    mass[centre, centre, State.PD, centre, centre] = 0.6

    diagnostics = pair_diagnostics(mass, coarse_grid, LearningConfig(tau=0.0))

    assert diagnostics.total_mass == pytest.approx(0.8)
    np.testing.assert_allclose(diagnostics.state_masses, [0.2, 0.6])
    np.testing.assert_allclose(diagnostics.mean_q, [0.0, 0.0])
    np.testing.assert_allclose(diagnostics.mean_action_probability, [0.5, 0.5])


def test_diagnostics_flag_negative_mass(coarse_grid) -> None:
    mass = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
    mass[2, 2, State.SH, 2, 2] = 1.0
    mass[0, 0, State.SH, 0, 0] = -0.1

    diagnostics = pair_diagnostics(mass, coarse_grid, LearningConfig())

    assert diagnostics.minimum_mass == pytest.approx(-0.1)
    assert not diagnostics.nonnegative
    assert not diagnostics.conditional_moments_valid


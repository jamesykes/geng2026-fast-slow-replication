from __future__ import annotations

import numpy as np

from chu_pair.policies import boltzmann_probabilities


def test_boltzmann_matches_original_direct_formula_on_case_range() -> None:
    values = np.linspace(-0.1, 1.2, 17)
    q_c, q_d = np.meshgrid(values, values, indexing="ij")
    q = np.stack((q_c, q_d), axis=-1)

    stable = boltzmann_probabilities(q, tau=2.0)
    exp_c = np.exp(2.0 * q_c)
    exp_d = np.exp(2.0 * q_d)
    direct = np.stack((exp_c / (exp_c + exp_d), exp_d / (exp_c + exp_d)), axis=-1)

    np.testing.assert_allclose(stable, direct, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(stable.sum(axis=-1), 1.0, rtol=0.0, atol=1e-15)


def test_boltzmann_remains_finite_for_extreme_differences() -> None:
    probabilities = boltzmann_probabilities([[1.0e6, -1.0e6], [-1.0e6, 1.0e6]], tau=2.0)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_array_equal(probabilities, np.array([[1.0, 0.0], [0.0, 1.0]]))


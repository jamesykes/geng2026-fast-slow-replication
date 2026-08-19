from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.config import LearningConfig
from chu_pair.grids import GridBoundsError, QGrid
from chu_pair.initial_conditions import DiscreteQHistogram, ordered_pair_mass
from chu_pair.model import Action, State
from chu_pair.observables import pair_diagnostics
from chu_pair.pair_density.jax_solver import (
    JAXConditionalDynamics,
    _legacy_project_indices,
    build_jax_pair_grid,
    canonical_to_flat_layout,
    checked_pair_mass_step,
    checked_simulate_pair_density,
    conditional_dynamics_jax,
    flat_to_canonical_layout,
    legacy_destination_indices_jax,
    one_edge_moments_jax,
    ordered_pair_mass_jax,
    pair_diagnostics_jax,
    pair_mass_step_jax,
    pair_mass_step_jit,
    simulate_pair_density_jax,
    simulate_pair_density_jit,
    validate_jax_pair_mass,
)
from chu_pair.pair_density.numpy_reference import (
    PairSymmetryError,
    one_edge_moments,
    pair_mass_step,
)


def _single_pair(grid: QGrid, state: int, q_index: tuple[int, int], dtype):
    mass = np.zeros((grid.size, grid.size, 2, grid.size, grid.size), dtype=dtype)
    i, j = q_index
    mass[i, j, state, i, j] = 1
    return mass


def _heterogeneous_source(dtype):
    source = np.zeros((5, 5, 2, 5, 5), dtype=dtype)
    source[1, 3, State.SH, 3, 1] = 0.5
    source[3, 1, State.SH, 1, 3] = 0.5
    return source


def _small_symmetric_source(grid: QGrid) -> np.ndarray:
    histogram_mass = np.zeros((grid.size, grid.size), dtype=np.float64)
    histogram_mass[1, 2] = 0.2
    histogram_mass[2, 1] = 0.3
    histogram_mass[3, 3] = 0.5
    histogram = DiscreteQHistogram(grid, histogram_mass)
    return ordered_pair_mass(histogram, state_probabilities=(0.35, 0.65))


def test_layout_and_initial_pair_preserve_independent_ordered_endpoints() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    histogram_mass = np.zeros((5, 5), dtype=np.float64)
    histogram_mass[1, 3] = 0.5
    histogram_mass[3, 1] = 0.5
    histogram = DiscreteQHistogram(grid, histogram_mass)

    flat = ordered_pair_mass_jax(
        histogram, state_probabilities=(0.5, 0.5), dtype=jnp.float32
    )
    canonical = np.asarray(flat_to_canonical_layout(flat, jax_grid))
    expected = ordered_pair_mass(histogram)

    np.testing.assert_array_equal(canonical, expected.astype(np.float32))
    np.testing.assert_array_equal(
        np.asarray(canonical_to_flat_layout(canonical)), np.asarray(flat)
    )
    assert np.count_nonzero(canonical) == 8
    for first in ((1, 3), (3, 1)):
        for second in ((1, 3), (3, 1)):
            for state in (State.SH, State.PD):
                assert canonical[
                    first[0], first[1], state, second[0], second[1]
                ] == pytest.approx(0.125)
    np.testing.assert_array_equal(canonical.sum(axis=(2, 3, 4)), histogram_mass)
    np.testing.assert_array_equal(canonical.sum(axis=(0, 1, 2)), histogram_mass)


def test_hand_calculated_heterogeneous_tau_transport_is_independent() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = _heterogeneous_source(np.float32)
    tau = float(np.log(3.0))

    result = pair_mass_step_jit(
        canonical_to_flat_layout(source),
        jax_grid,
        1.0,
        tau,
        chunk_size=7,
    )
    canonical = np.asarray(flat_to_canonical_layout(result.mass, jax_grid))

    # Independent source-Q policies: A=(1/4,3/4), B=(3/4,1/4).
    # Old SH row payoffs then give E_A=(3/4,1/10), E_B=(1/4,1/10).
    np.testing.assert_allclose(
        np.asarray(result.dynamics.expected_payoff)[1 * 5 + 3],
        [0.75, 0.1],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.dynamics.expected_payoff)[3 * 5 + 1],
        [0.25, 0.1],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.dynamics.velocity)[1 * 5 + 3],
        [1.25, -0.4],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.dynamics.velocity)[3 * 5 + 1],
        [-0.25, 0.6],
        rtol=0,
        atol=2e-7,
    )

    # A_C, A_D, B_C, B_D are independently projected selected-coordinate maps.
    destinations = np.asarray(result.destination_indices)
    np.testing.assert_array_equal(destinations[1 * 5 + 3], [3 * 5 + 3, 1 * 5 + 2])
    np.testing.assert_array_equal(destinations[3 * 5 + 1], [2 * 5 + 1, 3 * 5 + 2])

    expected = np.zeros_like(source)
    expected[3, 3, State.SH, 2, 1] = 3 / 32
    expected[3, 3, State.SH, 3, 2] = 1 / 32
    expected[1, 2, State.SH, 2, 1] = 9 / 32
    expected[1, 2, State.PD, 3, 2] = 3 / 32
    expected[2, 1, State.SH, 3, 3] = 3 / 32
    expected[2, 1, State.SH, 1, 2] = 9 / 32
    expected[3, 2, State.SH, 3, 3] = 1 / 32
    expected[3, 2, State.PD, 1, 2] = 3 / 32
    np.testing.assert_allclose(canonical, expected, rtol=0, atol=3e-8)
    assert float(canonical.sum()) == pytest.approx(1.0, abs=2e-7)
    np.testing.assert_allclose(
        canonical, canonical.transpose(3, 4, 2, 0, 1), rtol=0, atol=3e-8
    )


def test_pd_endpoint_orientation_and_old_state_branches_are_hand_calculated() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = np.zeros((5, 5, 2, 5, 5), dtype=np.float32)
    source[1, 3, State.PD, 3, 1] = 0.5
    source[3, 1, State.PD, 1, 3] = 0.5
    result = pair_mass_step_jit(
        canonical_to_flat_layout(source),
        jax_grid,
        1.0,
        float(np.log(3.0)),
        chunk_size=11,
    )

    # A faces B=(3/4 C,1/4 D): (.725,.9). B faces A=(1/4 C,3/4 D):
    # (.175,.3). Reversing the row-player payoff orientation changes these.
    np.testing.assert_allclose(
        np.asarray(result.dynamics.expected_payoff)[1 * 5 + 3],
        [0.725, 0.9],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.dynamics.expected_payoff)[3 * 5 + 1],
        [0.175, 0.3],
        rtol=0,
        atol=2e-7,
    )
    # In old PD only CC moves to SH. The two source orientations each assign
    # 3/32 to CC, so SH mass is 3/16 and all other branches remain PD.
    state_mass = np.asarray(result.mass).sum(axis=(1, 2))
    np.testing.assert_allclose(state_mass, [3 / 16, 13 / 16], rtol=0, atol=2e-7)
    destinations = np.asarray(result.destination_indices)
    np.testing.assert_array_equal(destinations[1 * 5 + 3], [3 * 5 + 3, 1 * 5 + 4])
    np.testing.assert_array_equal(destinations[3 * 5 + 1], [2 * 5 + 1, 3 * 5 + 3])


@pytest.mark.parametrize(
    ("old_state", "expected_state_mass"),
    [(State.SH, (0.75, 0.25)), (State.PD, (0.25, 0.75))],
)
def test_all_four_action_branches_use_old_state_before_transition(
    old_state, expected_state_mass
) -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = _single_pair(grid, old_state, (2, 2), np.float32)
    result = pair_mass_step_jax(
        canonical_to_flat_layout(source), jax_grid, 0.0, 0.0, chunk_size=13
    )
    state_mass = np.asarray(result.mass).sum(axis=(1, 2))

    np.testing.assert_allclose(state_mass, expected_state_mass, rtol=0, atol=1e-7)
    assert np.count_nonzero(np.asarray(result.mass)) == 2
    if old_state == State.SH:
        np.testing.assert_allclose(
            np.asarray(result.dynamics.expected_payoff)[12], [0.5, 0.1], atol=1e-7
        )
    else:
        np.testing.assert_allclose(
            np.asarray(result.dynamics.expected_payoff)[12], [0.45, 0.6], atol=1e-7
        )


def test_numpy_and_jax_one_step_and_moments_match() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    learning = LearningConfig(alpha=0.4, tau=1.3)
    source = _small_symmetric_source(grid)
    expected = pair_mass_step(source, grid, learning)
    expected_moments = one_edge_moments(source, grid, learning)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    actual = pair_mass_step_jit(
        canonical_to_flat_layout(source.astype(np.float32)),
        jax_grid,
        learning.alpha,
        learning.tau,
        chunk_size=29,
    )
    actual_moments = one_edge_moments_jax(
        canonical_to_flat_layout(source.astype(np.float32)), jax_grid, learning.tau
    )

    np.testing.assert_allclose(
        np.asarray(flat_to_canonical_layout(actual.mass, jax_grid)),
        expected.mass,
        rtol=2e-6,
        atol=2e-7,
    )
    expected_coordinates = expected.destination_indices.reshape(-1, 2, 2)
    expected_flat_destinations = (
        expected_coordinates[..., 0] * grid.size + expected_coordinates[..., 1]
    )
    np.testing.assert_array_equal(
        np.asarray(actual.destination_indices), expected_flat_destinations
    )
    np.testing.assert_allclose(
        np.asarray(actual_moments.mean).reshape(5, 5, 2),
        expected_moments.mean,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(actual_moments.second).reshape(5, 5, 2),
        expected_moments.second,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(actual_moments.variance).reshape(5, 5, 2),
        expected_moments.variance,
        rtol=3e-6,
        atol=3e-7,
    )


def test_multistep_scan_matches_numpy_and_retains_only_lean_diagnostics() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    learning = LearningConfig(alpha=0.4, tau=1.3)
    source = _small_symmetric_source(grid)
    expected = source
    for _ in range(4):
        expected = pair_mass_step(expected, grid, learning).mass

    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    result = simulate_pair_density_jit(
        canonical_to_flat_layout(source.astype(np.float32)),
        jax_grid,
        learning.alpha,
        learning.tau,
        steps=4,
        chunk_size=31,
        diagnostic_tolerance=2e-6,
    )
    actual = np.asarray(flat_to_canonical_layout(result.final_mass, jax_grid))

    np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=8e-7)
    assert bool(np.asarray(result.destinations_valid))
    assert result.diagnostics.total_mass.shape == (4,)
    assert result.diagnostics.state_masses.shape == (4, 2)
    assert result.diagnostics.mean_q.shape == (4, 2)
    for leaf in jax.tree_util.tree_leaves(result.diagnostics):
        assert leaf.shape[:1] == (4,)
        assert leaf.shape != (4, 2, 25, 25)
    np.testing.assert_allclose(
        np.asarray(result.diagnostics.total_mass), 1.0, rtol=0, atol=8e-7
    )
    assert np.all(np.asarray(result.diagnostics.nonnegative))
    assert np.all(np.asarray(result.diagnostics.conditional_moments_valid))

    host = pair_diagnostics(expected, grid, learning)
    np.testing.assert_allclose(np.asarray(result.diagnostics.state_masses[-1]), host.state_masses, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.diagnostics.mean_q[-1]), host.mean_q, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(result.diagnostics.mean_action_probability[-1]),
        host.mean_action_probability,
        atol=2e-6,
    )


def test_chunk_sizes_and_jit_eager_paths_agree() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = canonical_to_flat_layout(_small_symmetric_source(grid).astype(np.float32))
    results = [
        pair_mass_step_jit(source, jax_grid, 0.4, 1.3, chunk_size=size)
        for size in (1, 17, source.size)
    ]
    eager = pair_mass_step_jax(source, jax_grid, 0.4, 1.3, chunk_size=17)

    for result in (*results[1:], eager):
        np.testing.assert_allclose(
            np.asarray(result.mass), np.asarray(results[0].mass), rtol=0, atol=2e-7
        )
        np.testing.assert_array_equal(
            np.asarray(result.destination_indices),
            np.asarray(results[0].destination_indices),
        )


def test_jaxpr_contains_no_host_callback_and_diagnostics_match_moments() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = canonical_to_flat_layout(_small_symmetric_source(grid).astype(np.float32))
    jaxpr = str(
        jax.make_jaxpr(
            lambda mass: pair_mass_step_jax(
                mass, jax_grid, 0.4, 1.3, chunk_size=19
            ).mass
        )(source)
    ).lower()

    assert "host_callback" not in jaxpr
    assert "pure_callback" not in jaxpr
    dynamics = conditional_dynamics_jax(source, jax_grid, 0.4, 1.3)
    diagnostics = pair_diagnostics_jax(source, jax_grid, 1.3, tolerance=2e-6)
    assert dynamics.velocity.shape == (25, 2)
    assert bool(np.asarray(diagnostics.conditional_moments_valid))


def test_legacy_projection_uses_left_ties_and_round_before_tick_search() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    points = grid.agent_point_count
    velocity = jnp.zeros((points, 2), dtype=jnp.float32)
    occupied = jnp.zeros((points,), dtype=bool)

    # From (0,0), +0.25 ties between 0 and 0.5 and -0.25 ties between
    # -0.5 and 0; the active appro() rule selects the left point in each case.
    velocity = velocity.at[12].set(jnp.asarray([0.25, -0.25], dtype=jnp.float32))
    occupied = occupied.at[12].set(True)
    # The decimal pre-round maps 0.26 upward and 0.24 downward before the
    # multiple-of-spacing search; distances are still compared to the raw value.
    velocity = velocity.at[6].set(jnp.asarray([0.76, 0.74], dtype=jnp.float32))
    occupied = occupied.at[6].set(True)
    dynamics = JAXConditionalDynamics(
        focal_mass=occupied.astype(jnp.float32),
        expected_payoff=jnp.zeros_like(velocity),
        velocity=velocity,
        occupied=occupied,
    )

    destinations, valid = legacy_destination_indices_jax(dynamics, jax_grid)

    assert bool(np.asarray(valid))
    np.testing.assert_array_equal(np.asarray(destinations[12]), [12, 11])
    np.testing.assert_array_equal(np.asarray(destinations[6]), [16, 7])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_active_spacing_projection_handles_adversarial_decimal_half_points(dtype) -> None:
    if dtype is np.float64 and not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 projection requires a fresh CPU+x64 process")
    grid = QGrid(-0.1, 1.2, 0.01)
    jax_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
    jax_grid = build_jax_pair_grid(grid, jax_dtype)
    half = dtype(0.005)
    values = np.asarray(
        [
            -0.095,
            -0.085,
            -0.005,
            0.005,
            0.015,
            0.025,
            0.995,
            1.195,
            np.nextafter(half, dtype(-np.inf), dtype=dtype),
            half,
            np.nextafter(half, dtype(np.inf), dtype=dtype),
        ],
        dtype=dtype,
    )

    indices, valid = _legacy_project_indices(jnp.asarray(values), jax_grid)

    np.testing.assert_array_equal(
        np.asarray(indices), [0, 2, 10, 10, 12, 12, 110, 130, 10, 10, 11]
    )
    np.testing.assert_array_equal(np.asarray(valid), True)


def test_zero_step_scan_returns_initial_mass_and_empty_lean_diagnostics() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = canonical_to_flat_layout(_small_symmetric_source(grid).astype(np.float32))

    result = checked_simulate_pair_density(
        source,
        jax_grid,
        0.4,
        1.3,
        steps=0,
        chunk_size=17,
        symmetry_tolerance=2e-6,
        diagnostic_tolerance=2e-6,
    )

    np.testing.assert_array_equal(np.asarray(result.final_mass), np.asarray(source))
    assert bool(np.asarray(result.destinations_valid))
    for leaf in jax.tree_util.tree_leaves(result.diagnostics):
        assert leaf.shape[0] == 0


def test_float64_grid_request_fails_instead_of_silently_truncating() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    if jax.config.read("jax_enable_x64"):
        assert build_jax_pair_grid(grid, jnp.float64).q_points.dtype == jnp.float64
    else:
        with pytest.raises(ValueError, match="JAX_ENABLE_X64=1"):
            build_jax_pair_grid(grid, jnp.float64)
        canonical = np.zeros((5, 5, 2, 5, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="JAX_ENABLE_X64=1"):
            canonical_to_flat_layout(canonical)


def test_validation_rejects_invalid_mass_and_out_of_grid_destinations() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = canonical_to_flat_layout(_single_pair(grid, State.PD, (2, 2), np.float32))

    with pytest.raises(ValueError, match="shape"):
        validate_jax_pair_mass(source[:, :-1], jax_grid)
    with pytest.raises(ValueError, match="finite"):
        validate_jax_pair_mass(source.at[0, 0, 0].set(jnp.nan), jax_grid)
    with pytest.raises(ValueError, match="non-negative"):
        validate_jax_pair_mass(source.at[0, 0, 0].set(-0.1), jax_grid)
    with pytest.raises(PairSymmetryError, match="symmetry"):
        validate_jax_pair_mass(source.at[0, 1, 2].set(0.1), jax_grid)
    with pytest.raises(GridBoundsError, match="outside"):
        checked_pair_mass_step(
            source,
            jax_grid,
            20.0,
            0.0,
            chunk_size=31,
            symmetry_tolerance=1e-6,
        )
    with pytest.raises(GridBoundsError, match="outside"):
        checked_pair_mass_step(
            source,
            jax_grid,
            1e30,
            0.0,
            chunk_size=31,
            symmetry_tolerance=1e-6,
        )


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="requires a fresh CPU+x64 process",
)
def test_cpu_x64_transport_matches_numpy_tightly() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    learning = LearningConfig(alpha=0.4, tau=1.3)
    source = _small_symmetric_source(grid)
    expected = source
    for _ in range(4):
        expected = pair_mass_step(expected, grid, learning).mass
    jax_grid = build_jax_pair_grid(grid, jnp.float64)
    actual = simulate_pair_density_jit(
        canonical_to_flat_layout(source),
        jax_grid,
        learning.alpha,
        learning.tau,
        steps=4,
        chunk_size=23,
        diagnostic_tolerance=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(flat_to_canonical_layout(actual.final_mass, jax_grid)),
        expected,
        rtol=2e-14,
        atol=2e-15,
    )

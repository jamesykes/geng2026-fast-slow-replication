from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chu_pair.config import LearningConfig
from chu_pair.grids import QGrid
from chu_pair.initial_conditions import DiscreteQHistogram, ordered_pair_mass
from chu_pair.model import State
from chu_pair.pair_density import (
    build_jax_pair_grid,
    canonical_to_flat_layout,
    flat_to_canonical_layout,
    ordered_pair_mass_from_histogram_jit,
    pair_mass_step_jit,
    pair_mass_step_separable_jax,
    pair_mass_step_separable_jit,
    simulate_pair_source_summaries_from_histogram_jit,
    simulate_pair_source_summaries_from_histogram_full_jit,
    simulate_pair_source_summaries_jit,
)
from chu_pair.pair_density.numpy_reference import pair_mass_step

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _heterogeneous_source(dtype) -> np.ndarray:
    source = np.zeros((5, 5, 2, 5, 5), dtype=dtype)
    source[1, 3, State.SH, 3, 1] = 0.5
    source[3, 1, State.SH, 1, 3] = 0.5
    return source


def _dense_source(grid: QGrid, dtype) -> tuple[np.ndarray, np.ndarray]:
    # Deliberately adversarial nonuniform state law; production uses half/half.
    raw = np.arange(1, grid.size * grid.size + 1, dtype=np.float64)
    histogram_mass = (raw / raw.sum()).reshape(grid.size, grid.size)
    histogram = DiscreteQHistogram(grid, histogram_mass)
    pair = ordered_pair_mass(histogram, state_probabilities=(0.4, 0.6))
    return histogram_mass.astype(dtype), pair.astype(dtype)


def _assert_trees_close(left, right, *, rtol: float, atol: float) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for actual, expected in zip(left_leaves, right_leaves, strict=True):
        if jnp.issubdtype(actual.dtype, jnp.inexact):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
            )
        else:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_hand_calculated_separable_transport_has_endpoint_specific_maps() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = _heterogeneous_source(np.float32)
    tau = float(np.log(3.0))

    result = pair_mass_step_separable_jit(
        canonical_to_flat_layout(source),
        jax_grid,
        1.0,
        tau,
        row_block_size=7,
        column_block_size=6,
    )

    # These are calculated from exp(tau*q) directly, not through a model helper.
    policy_a = 1.0 / (1.0 + np.exp(tau * (0.5 - -0.5)))
    policy_b = 1.0 / (1.0 + np.exp(tau * (-0.5 - 0.5)))
    assert policy_a == pytest.approx(0.25)
    assert policy_b == pytest.approx(0.75)
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
        np.asarray(result.dynamics.velocity)[[1 * 5 + 3, 3 * 5 + 1]],
        [[1.25, -0.4], [-0.25, 0.6]],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(result.destination_indices)[[1 * 5 + 3, 3 * 5 + 1]],
        [[3 * 5 + 3, 1 * 5 + 2], [2 * 5 + 1, 3 * 5 + 2]],
    )

    # Eight cells enumerate the independently calculated branch weights,
    # endpoint-specific destinations, and SH/DD -> PD transitions.
    expected = np.zeros_like(source)
    expected[3, 3, State.SH, 2, 1] = 3 / 32
    expected[3, 3, State.SH, 3, 2] = 1 / 32
    expected[1, 2, State.SH, 2, 1] = 9 / 32
    expected[1, 2, State.PD, 3, 2] = 3 / 32
    expected[2, 1, State.SH, 3, 3] = 3 / 32
    expected[2, 1, State.SH, 1, 2] = 9 / 32
    expected[3, 2, State.SH, 3, 3] = 1 / 32
    expected[3, 2, State.PD, 1, 2] = 3 / 32
    actual = np.asarray(flat_to_canonical_layout(result.mass, jax_grid))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-8)
    np.testing.assert_allclose(actual, actual.transpose(3, 4, 2, 0, 1), atol=3e-8)
    assert float(actual.sum()) == pytest.approx(1.0, abs=2e-7)


_COLLISION_EXPECTED = {
    (0, 3, 15): 0.060990427318801055,
    (0, 3, 16): 0.014706714755449608,
    (0, 3, 17): 0.015067309471616712,
    (0, 8, 15): 0.022224318643354571,
    (0, 8, 16): 0.0053589838486224539,
    (0, 8, 17): 0.0054903810567665783,
    (0, 15, 3): 0.060990427318801055,
    (0, 15, 8): 0.022224318643354571,
    (0, 15, 15): 0.092166790032091939,
    (0, 15, 16): 0.022224318643354571,
    (0, 15, 17): 0.022769237886466842,
    (0, 16, 3): 0.014706714755449607,
    (0, 16, 8): 0.0053589838486224539,
    (0, 16, 15): 0.022224318643354571,
    (0, 16, 16): 0.0053589838486224539,
    (0, 16, 17): 0.0054903810567665783,
    (0, 17, 3): 0.015067309471616712,
    (0, 17, 8): 0.0054903810567665783,
    (0, 17, 15): 0.022769237886466842,
    (0, 17, 16): 0.0054903810567665783,
    (0, 17, 17): 0.0056249999999999998,
    (1, 3, 3): 0.16143915712957913,
    (1, 3, 8): 0.058826859021798433,
    (1, 3, 15): 0.060990427318801055,
    (1, 3, 16): 0.014706714755449608,
    (1, 3, 17): 0.015067309471616712,
    (1, 8, 3): 0.058826859021798426,
    (1, 8, 8): 0.021435935394489816,
    (1, 8, 15): 0.022224318643354571,
    (1, 8, 16): 0.0053589838486224539,
    (1, 8, 17): 0.0054903810567665783,
    (1, 15, 3): 0.060990427318801055,
    (1, 15, 8): 0.022224318643354571,
    (1, 16, 3): 0.014706714755449607,
    (1, 16, 8): 0.0053589838486224539,
    (1, 17, 3): 0.015067309471616712,
    (1, 17, 8): 0.0054903810567665783,
}


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fixed_hand_collision_oracle_covers_both_axes_and_state_change(dtype) -> None:
    if dtype == jnp.float64 and not jax.config.read("jax_enable_x64"):
        pytest.skip("requires a fresh CPU+x64 process")
    grid = QGrid(-1.0, 1.0, 0.5)
    histogram = np.zeros(25, dtype=np.float64)
    histogram[[0, 1, 2, 5]] = [0.1, 0.2, 0.3, 0.4]
    source = np.stack([0.5 * np.outer(histogram, histogram)] * 2).astype(
        np.dtype(dtype)
    )
    expected = np.zeros((2, 25, 25), dtype=np.float64)
    for destination, mass in _COLLISION_EXPECTED.items():
        expected[destination] = mass

    result = pair_mass_step_separable_jit(
        jnp.asarray(source),
        build_jax_pair_grid(grid, dtype),
        1.0,
        float(np.log(3.0)),
        row_block_size=7,
        column_block_size=6,
    )
    # These fixed values were obtained from the explicit Boltzmann formula,
    # authoritative payoff rows, left-tie projection, and eight hand branches.
    np.testing.assert_allclose(
        1.0
        / (1.0 + np.exp(np.log(3.0) * np.array([0.0, 0.5, 1.0, -0.5]))),
        [0.5, 0.36602540378443865, 0.25, 0.6339745962155614],
        rtol=0,
        atol=2e-16,
    )
    np.testing.assert_array_equal(
        np.asarray(result.destination_indices)[[0, 1, 2, 5]],
        [[15, 3], [16, 3], [17, 3], [15, 8]],
    )
    actual = np.asarray(result.mass)
    tolerance = 8e-9 if dtype == jnp.float32 else 1e-16
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    assert np.count_nonzero(actual) == len(_COLLISION_EXPECTED)
    np.testing.assert_allclose(
        actual.sum(axis=(1, 2)),
        [0.4517949192431123, 0.5482050807568878],
        rtol=0,
        atol=tolerance * 2,
    )
    assert float(actual.sum()) == pytest.approx(1.0, abs=2e-7 if dtype == jnp.float32 else 1e-15)
    np.testing.assert_allclose(actual, actual.transpose(0, 2, 1), rtol=0, atol=tolerance)


@pytest.mark.parametrize("blocks", [(1, 1), (7, 6), (25, 31)])
def test_dense_collisions_numpy_flat_separable_and_jit_eager_agree(blocks) -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    _, canonical = _dense_source(grid, np.float32)
    source = canonical_to_flat_layout(canonical)
    learning = LearningConfig(alpha=0.4, tau=1.3)
    expected = pair_mass_step(canonical, grid, learning).mass
    flat = pair_mass_step_jit(
        source, jax_grid, learning.alpha, learning.tau, chunk_size=source.size
    )
    separable = pair_mass_step_separable_jit(
        source,
        jax_grid,
        learning.alpha,
        learning.tau,
        row_block_size=blocks[0],
        column_block_size=blocks[1],
    )
    eager = pair_mass_step_separable_jax(
        source,
        jax_grid,
        learning.alpha,
        learning.tau,
        row_block_size=blocks[0],
        column_block_size=blocks[1],
    )

    destinations = np.asarray(separable.destination_indices)
    assert np.unique(destinations[:, 0]).size < grid.size**2
    assert np.unique(destinations[:, 1]).size < grid.size**2
    actual = np.asarray(flat_to_canonical_layout(separable.mass, jax_grid))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(np.asarray(separable.mass), np.asarray(flat.mass), rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(np.asarray(eager.mass), np.asarray(separable.mass), rtol=0, atol=0)
    assert float(np.asarray(separable.mass).sum()) == pytest.approx(1.0, abs=3e-7)
    assert np.min(np.asarray(separable.mass)) >= 0
    np.testing.assert_allclose(
        np.asarray(separable.mass),
        np.asarray(separable.mass).transpose(0, 2, 1),
        rtol=0,
        atol=3e-8,
    )


@pytest.mark.parametrize(
    ("old_state", "expected_state_mass"),
    [(State.SH, (0.75, 0.25)), (State.PD, (0.25, 0.75))],
)
def test_separable_covers_all_action_branches_and_both_old_states(
    old_state, expected_state_mass
) -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    source = np.zeros((5, 5, 2, 5, 5), dtype=np.float32)
    source[2, 2, old_state, 2, 2] = 1
    result = pair_mass_step_separable_jit(
        canonical_to_flat_layout(source),
        jax_grid,
        0.0,
        0.0,
        row_block_size=4,
        column_block_size=7,
    )
    np.testing.assert_allclose(
        np.asarray(result.mass).sum(axis=(1, 2)), expected_state_mass, atol=1e-7
    )
    assert np.count_nonzero(np.asarray(result.mass)) == 2


def test_zero_rows_columns_and_left_projection_tie_are_preserved() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    # Only q=(0,0) is occupied. In SH at tau=0, E[Y_C]=.5, hence alpha=.5
    # gives the exact +.25 tie; the legacy rule keeps the left destination 0.
    source = np.zeros((5, 5, 2, 5, 5), dtype=np.float32)
    source[2, 2, State.SH, 2, 2] = 1
    result = pair_mass_step_separable_jit(
        canonical_to_flat_layout(source),
        jax_grid,
        0.5,
        0.0,
        row_block_size=6,
        column_block_size=7,
    )
    np.testing.assert_array_equal(
        np.asarray(result.destination_indices)[12], [12, 12]
    )
    assert np.count_nonzero(np.asarray(result.dynamics.focal_mass)) == 1
    assert bool(np.asarray(result.destinations_valid))


def test_multistep_bounded_scan_matches_flat_and_returns_no_density() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    histogram, canonical = _dense_source(grid, np.float32)
    histogram_device = jnp.asarray(histogram.reshape(-1))
    state_probabilities = jnp.asarray([0.4, 0.6], dtype=jnp.float32)
    source = ordered_pair_mass_from_histogram_jit(
        histogram_device, state_probabilities
    )
    np.testing.assert_allclose(
        np.asarray(source),
        np.asarray(canonical_to_flat_layout(canonical)),
        rtol=2e-7,
        atol=2e-10,
    )
    slots = jnp.asarray([0, -1, 1, 2], dtype=jnp.int32)
    common = dict(
        steps=3,
        summary_count=3,
        chunk_size=source.size,
        diagnostic_tolerance=3e-6,
    )
    flat = simulate_pair_source_summaries_jit(
        source, jax_grid, 0.4, 1.3, slots, kernel="flat", **common
    )
    separable = simulate_pair_source_summaries_jit(
        source,
        jax_grid,
        0.4,
        1.3,
        slots,
        kernel="separable",
        row_block_size=7,
        column_block_size=6,
        **common,
    )
    bounded = simulate_pair_source_summaries_from_histogram_jit(
        histogram_device,
        state_probabilities,
        jax_grid,
        0.4,
        1.3,
        slots,
        kernel="separable",
        row_block_size=7,
        column_block_size=6,
        **common,
    )

    np.testing.assert_allclose(np.asarray(separable.final_mass), np.asarray(flat.final_mass), rtol=8e-6, atol=8e-8)
    _assert_trees_close(separable.source_summaries, flat.source_summaries, rtol=8e-6, atol=8e-7)
    _assert_trees_close(bounded.source_summaries, separable.source_summaries, rtol=0, atol=0)
    _assert_trees_close(bounded.diagnostics, separable.diagnostics, rtol=0, atol=0)
    assert not hasattr(bounded, "final_mass")
    for leaf in jax.tree_util.tree_leaves(bounded):
        assert leaf.shape != (2, grid.size**2, grid.size**2)

    jaxpr = str(jax.make_jaxpr(lambda h: ordered_pair_mass_from_histogram_jit(h, jnp.asarray([0.4, 0.6], dtype=jnp.float32)))(jnp.asarray(histogram.reshape(-1)))).lower()
    assert "host_callback" not in jaxpr
    assert "pure_callback" not in jaxpr


def test_controlled_combined_initializer_is_uniform_independent_and_bounded() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float32)
    histogram = np.zeros(25, dtype=np.float32)
    histogram[[1, 7]] = [0.25, 0.75]
    states = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    slots = jnp.asarray([0], dtype=jnp.int32)

    # The validation-only combined object exposes P_0 on this tiny grid so the
    # production initializer law can be checked independently of summaries.
    full = simulate_pair_source_summaries_from_histogram_full_jit(
        jnp.asarray(histogram),
        states,
        jax_grid,
        0.4,
        1.3,
        slots,
        steps=0,
        summary_count=1,
        chunk_size=1,
        diagnostic_tolerance=1e-6,
        kernel="separable",
        row_block_size=7,
        column_block_size=6,
    )
    mass = np.asarray(full.final_mass)
    expected = np.stack([0.5 * np.outer(histogram, histogram)] * 2)
    np.testing.assert_allclose(mass, expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(mass.sum(axis=(1, 2)), [0.5, 0.5], atol=2e-8)
    np.testing.assert_allclose(mass.sum(axis=(0, 2)), histogram, atol=2e-8)
    np.testing.assert_allclose(mass.sum(axis=(0, 1)), histogram, atol=2e-8)
    np.testing.assert_allclose(mass, mass.transpose(0, 2, 1), atol=0)
    assert float(mass.sum()) == pytest.approx(1.0, abs=2e-7)

    bounded = simulate_pair_source_summaries_from_histogram_jit(
        jnp.asarray(histogram),
        states,
        jax_grid,
        0.4,
        1.3,
        slots,
        steps=0,
        summary_count=1,
        chunk_size=1,
        diagnostic_tolerance=1e-6,
        kernel="separable",
        row_block_size=7,
        column_block_size=6,
    )
    assert not hasattr(bounded, "final_mass")
    np.testing.assert_allclose(
        np.asarray(bounded.diagnostics.state_masses)[0], [0.5, 0.5], atol=2e-8
    )
    np.testing.assert_allclose(
        np.asarray(bounded.source_summaries.focal_mass)[0], histogram, atol=2e-8
    )
    assert all(leaf.shape != (2, 25, 25) for leaf in jax.tree_util.tree_leaves(bounded))


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="requires a fresh CPU+x64 process",
)
def test_separable_cpu_x64_matches_numpy_and_flat_tightly() -> None:
    grid = QGrid(-1.0, 1.0, 0.5)
    jax_grid = build_jax_pair_grid(grid, jnp.float64)
    _, canonical = _dense_source(grid, np.float64)
    learning = LearningConfig(alpha=0.4, tau=1.3)
    expected = pair_mass_step(canonical, grid, learning).mass
    source = canonical_to_flat_layout(canonical)
    flat = pair_mass_step_jit(
        source, jax_grid, learning.alpha, learning.tau, chunk_size=source.size
    )
    separable = pair_mass_step_separable_jit(
        source,
        jax_grid,
        learning.alpha,
        learning.tau,
        row_block_size=7,
        column_block_size=6,
    )
    actual = np.asarray(flat_to_canonical_layout(separable.mass, jax_grid))
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-16)
    np.testing.assert_allclose(np.asarray(separable.mass), np.asarray(flat.mass), rtol=3e-14, atol=3e-16)


def _precision_probe_source() -> str:
    """Fresh-process probe comparing default and explicit contraction precision."""

    return (
        "import json, os\n"
        "from chu_pair.gpu_pilot.allocator import apply_allocator_policy\n"
        "apply_allocator_policy('fraction', memory_fraction=0.85)\n"
        "import jax, jax.numpy as jnp, numpy as np\n"
        "if jax.default_backend() != 'gpu':\n"
        "    print(json.dumps({'skipped': 'no gpu backend'})); raise SystemExit(0)\n"
        "from chu_pair.grids import QGrid\n"
        "from chu_pair.initial_conditions import seeded_legacy_histogram\n"
        "from chu_pair.pair_density import (\n"
        "    build_jax_pair_grid, simulate_pair_source_summaries_from_histogram_jit as fn,\n"
        "    pair_contraction_precision)\n"
        "grid = QGrid(-0.4, 1.2, 0.4)\n"
        "hist = seeded_legacy_histogram(grid, seed=20230818,\n"
        "    samples_per_grid_cell=10).mass.reshape(-1)\n"
        "out = {'policy': pair_contraction_precision()}\n"
        "def run():\n"
        "    r = fn(jnp.asarray(hist, jnp.float32), jnp.asarray([0.5, 0.5], jnp.float32),\n"
        "           build_jax_pair_grid(grid, jnp.float32), 0.4, 1.3,\n"
        "           jnp.asarray([0, 1, 2], jnp.int32), steps=2, summary_count=3,\n"
        "           chunk_size=64, diagnostic_tolerance=1e-4, kernel='separable',\n"
        "           row_block_size=32, column_block_size=32)\n"
        "    return float(np.asarray(r.diagnostics.conditional_weight_error).max())\n"
        "out['explicit'] = run()\n"
        # Operation-local precision deliberately overrides any global matmul
        # context, so the pre-repair behaviour is reproduced by unpinning the
        # module policy and retracing.
        "from chu_pair.pair_density import jax_solver\n"
        "jax_solver._CONTRACTION_PRECISION = jax.lax.Precision.DEFAULT\n"
        "jax.clear_caches()\n"
        "out['default'] = run()\n"
        "print(json.dumps(out))\n"
    )


@pytest.mark.skipif(
    os.environ.get("CHU_PAIR_GPU_PRECISION_CHECK") != "1",
    reason="opt-in real-GPU precision check; set CHU_PAIR_GPU_PRECISION_CHECK=1",
)
def test_real_gpu_default_precision_violates_conditional_weight_tolerance() -> None:
    """Opt-in: demonstrate the defect and its repair on real NVIDIA hardware.

    On an H100 the platform-default TF32 lowering produces a conditional-weight
    residual around 4e-4, above the reviewed 1e-4 diagnostic tolerance, while
    the explicit full-float32 policy restores the float32 rounding scale near
    1.2e-7. Runs in a fresh subprocess because the allocator policy must be
    applied before JAX is imported, and never runs on CPU-only hosts.
    """

    completed = subprocess.run(
        [sys.executable, "-c", _precision_probe_source()], cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if "skipped" in result:
        pytest.skip(result["skipped"])
    assert result["policy"] == "highest"
    # The platform default is capable of violating the reviewed tolerance...
    assert result["default"] > 1e-4
    # ...and the explicit policy restores float32 rounding accuracy.
    assert result["explicit"] < 1e-5
    assert result["explicit"] < result["default"] / 100.0


def test_contraction_precision_is_explicit_in_both_kernels() -> None:
    """Both kernels share the one contraction path and its explicit precision."""

    from chu_pair.pair_density import jax_solver

    assert jax_solver.PAIR_CONTRACTION_PRECISION == "highest"
    assert jax_solver._CONTRACTION_PRECISION is jax.lax.Precision.HIGHEST

    source = Path(jax_solver.__file__).read_text()
    # Every contraction in the pair-density calculation must pass the policy.
    assert source.count("jnp.einsum(") == 3
    assert source.count("precision=_CONTRACTION_PRECISION") == 3


def test_conditional_weights_stay_at_float32_rounding_scale() -> None:
    """CPU float32 parity and invariants are unchanged by the precision policy."""

    grid = QGrid(-0.4, 1.2, 0.4)
    mass = np.full((grid.size, grid.size), 1.0 / grid.size**2)
    histogram = DiscreteQHistogram(grid, mass)
    for kernel in ("flat", "separable"):
        result = simulate_pair_source_summaries_from_histogram_jit(
            jnp.asarray(histogram.mass.reshape(-1), jnp.float32),
            jnp.asarray([0.5, 0.5], jnp.float32),
            build_jax_pair_grid(grid, jnp.float32), 0.4, 1.3,
            jnp.asarray([0, 1], jnp.int32), steps=1, summary_count=2,
            chunk_size=64, diagnostic_tolerance=1e-4, kernel=kernel,
            row_block_size=32, column_block_size=32,
        )
        diagnostics = result.diagnostics
        assert float(np.asarray(diagnostics.conditional_weight_error).max()) < 1e-5
        assert float(np.abs(np.asarray(diagnostics.total_mass) - 1.0).max()) < 1e-5
        assert bool(np.all(np.asarray(diagnostics.finite, dtype=bool)))
        assert bool(np.all(np.asarray(diagnostics.nonnegative, dtype=bool)))
        assert bool(np.all(np.asarray(result.destinations_valid, dtype=bool)))

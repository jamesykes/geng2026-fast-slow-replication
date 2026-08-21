"""JAX implementation of the exact legacy nearest-grid pair-mass transport."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..grids import GridBoundsError, QGrid
from ..initial_conditions import DiscreteQHistogram
from ..config import PAIR_CONTRACTION_PRECISION
from ..model import PAYOFF_TENSOR, TRANSITION_TENSOR
from ..policies import _two_action_boltzmann_probabilities
from .numpy_reference import PairSymmetryError


DEFAULT_MAX_JAX_PAIR_ELEMENTS = 5_000_000
PAIR_KERNEL_FLAT = "flat"
PAIR_KERNEL_SEPARABLE = "separable"
PAIR_KERNELS = frozenset({PAIR_KERNEL_FLAT, PAIR_KERNEL_SEPARABLE})


def _validate_requested_dtype(dtype, object_name: str) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{object_name} dtype must be float32 or float64")
    if dtype == np.dtype(np.float64) and not jax.config.read("jax_enable_x64"):
        raise ValueError(
            f"float64 {object_name} requires JAX_ENABLE_X64=1 before importing JAX"
        )
    return dtype


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class JAXPairGrid:
    """Dynamic grid arrays plus static metadata used by compiled transport."""

    size: int
    decimal_factor: int
    spacing_ticks: int
    q_min_ticks: int
    values: jax.Array
    q_points: jax.Array
    q_c_indices: jax.Array
    q_d_indices: jax.Array

    @property
    def agent_point_count(self) -> int:
        return self.size * self.size

    def tree_flatten(self):
        children = (
            self.values,
            self.q_points,
            self.q_c_indices,
            self.q_d_indices,
        )
        metadata = (
            self.size,
            self.decimal_factor,
            self.spacing_ticks,
            self.q_min_ticks,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        size, decimal_factor, spacing_ticks, q_min_ticks = metadata
        values, q_points, q_c_indices, q_d_indices = children
        return cls(
            size=size,
            decimal_factor=decimal_factor,
            spacing_ticks=spacing_ticks,
            q_min_ticks=q_min_ticks,
            values=values,
            q_points=q_points,
            q_c_indices=q_c_indices,
            q_d_indices=q_d_indices,
        )


class JAXOneEdgeMoments(NamedTuple):
    """JAX pytree for one-edge conditional moments on flattened Q points."""

    focal_mass: jax.Array
    state_opponent_action_probability: jax.Array
    mean: jax.Array
    second: jax.Array
    variance: jax.Array
    occupied: jax.Array


class JAXPairPointSufficient(NamedTuple):
    """Exact-grid selected-action raw sums for bounded Phase 5 comparison."""

    focal_mass: jax.Array
    selected_mass: jax.Array
    sum_y: jax.Array
    sum_y2: jax.Array
    sum_distinct_y: jax.Array
    sum_q: jax.Array
    sum_q2: jax.Array
    sum_y_q: jax.Array


class JAXConditionalDynamics(NamedTuple):
    focal_mass: jax.Array
    expected_payoff: jax.Array
    velocity: jax.Array
    occupied: jax.Array


class JAXPairStepResult(NamedTuple):
    mass: jax.Array
    dynamics: JAXConditionalDynamics
    destination_indices: jax.Array
    destinations_valid: jax.Array


class JAXPairDiagnostics(NamedTuple):
    total_mass: jax.Array
    state_masses: jax.Array
    mean_q: jax.Array
    mean_action_probability: jax.Array
    symmetry_error: jax.Array
    minimum_mass: jax.Array
    finite: jax.Array
    nonnegative: jax.Array
    conditional_weight_error: jax.Array
    minimum_conditional_variance: jax.Array
    conditional_moments_valid: jax.Array


class JAXPairSimulationResult(NamedTuple):
    final_mass: jax.Array
    diagnostics: JAXPairDiagnostics
    destinations_valid: jax.Array


class JAXPairSourceSimulationResult(NamedTuple):
    """Bounded source summaries plus lean diagnostics from one compiled scan."""

    final_mass: jax.Array
    source_summaries: JAXPairPointSufficient
    diagnostics: JAXPairDiagnostics
    destinations_valid: jax.Array


class JAXPairBoundedSourceResult(NamedTuple):
    """Production-oriented outputs without a complete final-density transfer."""

    source_summaries: JAXPairPointSufficient
    diagnostics: JAXPairDiagnostics
    destinations_valid: jax.Array


def build_jax_pair_grid(grid: QGrid, dtype=jnp.float32) -> JAXPairGrid:
    """Build small reusable device arrays after host-side resource preflight."""

    dtype = _validate_requested_dtype(dtype, "pair grid")
    factor = 10**grid.decimal_places
    spacing_ticks = int(np.around(grid.spacing * factor))
    q_min_ticks = int(np.around(grid.q_min * factor))
    q_max_ticks = int(np.around(grid.q_max * factor))
    if spacing_ticks <= 0:
        raise ValueError("grid spacing is too small for its decimal representation")
    tick_limit = np.iinfo(np.int32)
    extreme_ticks = max(abs(q_min_ticks), abs(q_max_ticks))
    span_ticks = q_max_ticks - q_min_ticks
    if (
        extreme_ticks > tick_limit.max
        or factor > tick_limit.max
        or span_ticks > tick_limit.max
        or grid.size > tick_limit.max
    ):
        raise ValueError("legacy projection ticks must fit in int32")
    values = np.asarray(grid.values, dtype=dtype)
    q_c, q_d = np.meshgrid(
        np.arange(grid.size, dtype=np.int32),
        np.arange(grid.size, dtype=np.int32),
        indexing="ij",
    )
    q_points = np.stack((values[q_c], values[q_d]), axis=-1).reshape(-1, 2)
    return JAXPairGrid(
        size=grid.size,
        decimal_factor=factor,
        spacing_ticks=spacing_ticks,
        q_min_ticks=q_min_ticks,
        values=jnp.asarray(values),
        q_points=jnp.asarray(q_points),
        q_c_indices=jnp.asarray(q_c.reshape(-1)),
        q_d_indices=jnp.asarray(q_d.reshape(-1)),
    )


def canonical_to_flat_layout(pair_mass) -> jax.Array:
    """Convert canonical ``(G,G,2,G,G)`` mass to internal ``(2,M,M)``."""

    raw_dtype = getattr(pair_mass, "dtype", None)
    if raw_dtype is not None:
        _validate_requested_dtype(raw_dtype, "canonical pair mass")
    mass = jnp.asarray(pair_mass)
    if mass.ndim != 5 or mass.shape[2] != 2:
        raise ValueError("canonical pair mass must have shape (G,G,2,G,G)")
    if mass.shape[0] != mass.shape[1] or mass.shape[0] != mass.shape[3] or mass.shape[0] != mass.shape[4]:
        raise ValueError("all canonical pair Q axes must have the same length")
    grid_size = mass.shape[0]
    points = grid_size * grid_size
    return jnp.transpose(mass, (2, 0, 1, 3, 4)).reshape(2, points, points)


def flat_to_canonical_layout(pair_mass: jax.Array, grid: JAXPairGrid) -> jax.Array:
    """Convert internal ``(2,M,M)`` mass to canonical ``(G,G,2,G,G)``."""

    mass = jnp.asarray(pair_mass)
    points = grid.agent_point_count
    if mass.shape != (2, points, points):
        raise ValueError(f"flat pair mass must have shape {(2, points, points)}")
    reshaped = mass.reshape(2, grid.size, grid.size, grid.size, grid.size)
    return jnp.transpose(reshaped, (1, 2, 0, 3, 4))


def ordered_pair_mass_jax(
    histogram: DiscreteQHistogram,
    *,
    state_probabilities: tuple[float, float] = (0.5, 0.5),
    dtype=jnp.float32,
) -> jax.Array:
    """Construct the independent ordered endpoint/state mass in flat layout."""

    dtype = _validate_requested_dtype(dtype, "pair mass")
    state_mass = np.asarray(state_probabilities, dtype=np.float64)
    if state_mass.shape != (2,) or not np.all(np.isfinite(state_mass)):
        raise ValueError("state_probabilities must contain two finite values")
    if np.any(state_mass < 0.0) or not np.isclose(
        float(state_mass.sum()), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("state probabilities must be non-negative and sum to one")
    return ordered_pair_mass_from_histogram_jax(
        jnp.asarray(histogram.mass.reshape(-1), dtype=dtype),
        jnp.asarray(state_mass, dtype=dtype),
    )


def ordered_pair_mass_from_histogram_jax(
    histogram_mass: jax.Array,
    state_probabilities: jax.Array,
) -> jax.Array:
    """Construct independent ordered endpoints entirely from device vectors."""

    flat = jnp.asarray(histogram_mass)
    states = jnp.asarray(state_probabilities, dtype=flat.dtype)
    if flat.ndim != 1:
        raise ValueError("histogram_mass must be a flat one-agent mass vector")
    if states.shape != (2,):
        raise ValueError("state_probabilities must have shape (2,)")
    return states[:, None, None] * flat[None, :, None] * flat[None, None, :]


ordered_pair_mass_from_histogram_jit = jax.jit(
    ordered_pair_mass_from_histogram_jax
)


def validate_jax_pair_mass(
    pair_mass,
    grid: JAXPairGrid,
    *,
    symmetry_tolerance: float = 1e-6,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> jax.Array:
    """Host-side validation for a device pair state; never called from a JIT."""

    if not math.isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance must be finite and non-negative")
    if max_elements is not None and (
        isinstance(max_elements, bool)
        or not isinstance(max_elements, int)
        or max_elements < 0
    ):
        raise ValueError("max_elements must be a non-negative integer or None")
    shape = getattr(pair_mass, "shape", None)
    points = grid.agent_point_count
    expected = (2, points, points)
    if shape != expected:
        raise ValueError(f"flat pair mass must have shape {expected}, got {shape}")
    size = math.prod(shape)
    if max_elements is not None and size > max_elements:
        raise MemoryError(
            f"JAX pair has {size:,} elements, above validation limit {max_elements:,}"
        )
    _validate_requested_dtype(pair_mass.dtype, "pair mass")
    host = np.asarray(jax.device_get(pair_mass))
    if not np.all(np.isfinite(host)):
        raise ValueError("pair mass must contain only finite values")
    if np.any(host < 0.0):
        raise ValueError("pair mass must be non-negative")
    symmetry_error = float(np.max(np.abs(host - host.transpose(0, 2, 1)), initial=0.0))
    if symmetry_error > symmetry_tolerance:
        raise PairSymmetryError(
            f"endpoint exchange symmetry error {symmetry_error} exceeds {symmetry_tolerance}"
        )
    return jnp.asarray(pair_mass)


def _validate_learning_scalars(alpha, tau) -> None:
    for name, value in (("alpha", alpha), ("tau", tau)):
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite non-negative scalar") from error
        if not math.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name} must be a finite non-negative scalar")


def _policy(grid: JAXPairGrid, tau, dtype) -> jax.Array:
    tau_value = jnp.asarray(tau, dtype=dtype)
    return _two_action_boltzmann_probabilities(grid.q_points.astype(dtype), tau_value, jnp)


# XLA lowers float32 ``dot_general`` to TF32 tensor cores by default on
# Ampere and newer NVIDIA hardware.  That is roughly 1e-3 relative accuracy,
# which on an H100 inflated the conditional-weight residual of ``w(s,b|q)``
# from the float32 rounding scale (~1.2e-7) to ~4e-4 and so violated the
# reviewed ``diagnostic_tolerance`` of 1e-4.  These three contractions are the
# only ones in the pair-density calculation, and they form the conditional
# one-edge law and its payoff moments, so they are pinned to full float32.
# Storage, arithmetic elsewhere, dtypes and every scientific formula are
# unchanged; this fixes only how the dot products are evaluated.
_PRECISION_BY_NAME = {"highest": jax.lax.Precision.HIGHEST}
if PAIR_CONTRACTION_PRECISION not in _PRECISION_BY_NAME:      # pragma: no cover
    raise RuntimeError(
        f"unsupported pair contraction precision {PAIR_CONTRACTION_PRECISION!r}"
    )
_CONTRACTION_PRECISION = _PRECISION_BY_NAME[PAIR_CONTRACTION_PRECISION]


def pair_contraction_precision() -> str:
    """Name of the explicit precision policy used by pair contractions."""

    return PAIR_CONTRACTION_PRECISION


def one_edge_moments_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    tau,
) -> JAXOneEdgeMoments:
    """Vectorized ``w(s,b|q)``, mean, second moment, and variance."""

    mass = jnp.asarray(pair_mass)
    points = grid.agent_point_count
    if mass.shape != (2, points, points):
        raise ValueError(f"flat pair mass must have shape {(2, points, points)}")
    focal = jnp.sum(mass, axis=(0, 2))
    occupied = focal > 0
    opponent_policy = _policy(grid, tau, mass.dtype)
    numerators = jnp.einsum(
        "smv,vb->msb", mass, opponent_policy, precision=_CONTRACTION_PRECISION,
    )
    weights = jnp.where(
        occupied[:, None, None],
        numerators / jnp.where(occupied, focal, 1)[:, None, None],
        jnp.zeros((), dtype=mass.dtype),
    )
    payoff = jnp.asarray(PAYOFF_TENSOR, dtype=mass.dtype)
    mean = jnp.einsum(
        "msb,sab->ma", weights, payoff, precision=_CONTRACTION_PRECISION,
    )
    second = jnp.einsum(
        "msb,sab->ma", weights, payoff * payoff, precision=_CONTRACTION_PRECISION,
    )
    variance = second - mean * mean
    return JAXOneEdgeMoments(focal, weights, mean, second, variance, occupied)


def pair_point_sufficient_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    tau,
) -> JAXPairPointSufficient:
    """Return exact-focal-Q raw selected-action sums without binning.

    Conditional independence is applied only at an exact focal point, where the
    distinct-opponent product is ``mu(q,j)**2``.  Finite-bin nonlinear moments
    are deliberately deferred until these weighted raw sums have been pooled.
    """

    mass = jnp.asarray(pair_mass)
    moments = one_edge_moments_jax(mass, grid, tau)
    policy = _policy(grid, tau, mass.dtype)
    selected_mass = moments.focal_mass[:, None] * policy
    q_selected = grid.q_points.astype(mass.dtype)
    return JAXPairPointSufficient(
        focal_mass=moments.focal_mass,
        selected_mass=selected_mass,
        sum_y=selected_mass * moments.mean,
        sum_y2=selected_mass * moments.second,
        sum_distinct_y=selected_mass * moments.mean * moments.mean,
        sum_q=selected_mass * q_selected,
        sum_q2=selected_mass * q_selected * q_selected,
        sum_y_q=selected_mass * moments.mean * q_selected,
    )


def conditional_dynamics_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
) -> JAXConditionalDynamics:
    moments = one_edge_moments_jax(pair_mass, grid, tau)
    alpha_value = jnp.asarray(alpha, dtype=pair_mass.dtype)
    velocity = jnp.where(
        moments.occupied[:, None],
        alpha_value * (moments.mean - grid.q_points.astype(pair_mass.dtype)),
        jnp.zeros((), dtype=pair_mass.dtype),
    )
    return JAXConditionalDynamics(
        moments.focal_mass,
        moments.mean,
        velocity,
        moments.occupied,
    )


def _legacy_project_indices(values: jax.Array, grid: JAXPairGrid) -> tuple[jax.Array, jax.Array]:
    dtype = values.dtype
    factor = jnp.asarray(grid.decimal_factor, dtype=dtype)
    rounded_ticks_float = jnp.rint(values * factor)
    int32 = np.iinfo(np.int32)
    representable = (
        jnp.isfinite(rounded_ticks_float)
        & (rounded_ticks_float >= int32.min)
        & (rounded_ticks_float <= int32.max)
    )
    rounded_ticks = jnp.where(representable, rounded_ticks_float, 0).astype(jnp.int32)
    spacing_ticks = jnp.asarray(grid.spacing_ticks, dtype=jnp.int32)
    left_ticks = jnp.floor_divide(rounded_ticks, spacing_ticks) * spacing_ticks
    right_ticks = jnp.where(
        left_ticks == rounded_ticks,
        left_ticks,
        left_ticks + spacing_ticks,
    )
    left_values = left_ticks.astype(dtype) / factor
    right_values = right_ticks.astype(dtype) / factor
    projected_ticks = jnp.where(
        jnp.abs(left_values - values) <= jnp.abs(right_values - values),
        left_ticks,
        right_ticks,
    )
    q_min_ticks = jnp.asarray(grid.q_min_ticks, dtype=jnp.int32)
    q_max_ticks = q_min_ticks + (grid.size - 1) * spacing_ticks
    within_grid = (projected_ticks >= q_min_ticks) & (projected_ticks <= q_max_ticks)
    safe_projected_ticks = jnp.where(
        representable & within_grid, projected_ticks, q_min_ticks
    )
    indices = jnp.floor_divide(
        safe_projected_ticks - q_min_ticks,
        spacing_ticks,
    )
    valid = representable & within_grid & (indices >= 0) & (indices < grid.size)
    return indices, valid


def legacy_destination_indices_jax(
    dynamics: JAXConditionalDynamics,
    grid: JAXPairGrid,
) -> tuple[jax.Array, jax.Array]:
    """Return flattened selected-coordinate destinations and one validity flag."""

    q_points = grid.q_points.astype(dynamics.velocity.dtype)
    projected_c, valid_c = _legacy_project_indices(
        q_points[:, 0] + dynamics.velocity[:, 0], grid
    )
    projected_d, valid_d = _legacy_project_indices(
        q_points[:, 1] + dynamics.velocity[:, 1], grid
    )
    source = jnp.arange(grid.agent_point_count, dtype=jnp.int32)
    destination_c = projected_c * grid.size + grid.q_d_indices
    destination_d = grid.q_c_indices * grid.size + projected_d
    destinations = jnp.stack((destination_c, destination_d), axis=-1)
    destinations = jnp.where(dynamics.occupied[:, None], destinations, source[:, None])
    valid = jnp.all((~dynamics.occupied) | (valid_c & valid_d))
    return destinations, valid


def pair_mass_step_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    *,
    chunk_size: int,
) -> JAXPairStepResult:
    """Apply one synchronous four-branch pushforward with bounded flat scatter."""

    mass = jnp.asarray(pair_mass)
    points = grid.agent_point_count
    expected = (2, points, points)
    if mass.shape != expected:
        raise ValueError(f"flat pair mass must have shape {expected}")
    if not jnp.issubdtype(mass.dtype, jnp.floating):
        raise TypeError("pair mass must use a floating dtype")
    source_cells = 2 * points * points
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    effective_chunk = min(chunk_size, source_cells)
    chunk_count = (source_cells + effective_chunk - 1) // effective_chunk

    dynamics = conditional_dynamics_jax(mass, grid, alpha, tau)
    destinations, destinations_valid = legacy_destination_indices_jax(dynamics, grid)
    safe_destinations = jnp.clip(destinations, 0, points - 1)
    source_policy = _policy(grid, tau, mass.dtype)
    branch_a = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    branch_b = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    transitions = jnp.asarray(TRANSITION_TENSOR, dtype=jnp.int32)
    flat_mass = mass.reshape(-1)
    output = jnp.zeros_like(flat_mass)
    offsets = jnp.arange(effective_chunk, dtype=jnp.int32)

    def add_chunk(chunk_index, accumulator):
        indices = chunk_index * effective_chunk + offsets
        valid_source = indices < source_cells
        safe_indices = jnp.minimum(indices, source_cells - 1)
        old_state = safe_indices // (points * points)
        remainder = safe_indices % (points * points)
        endpoint_u = remainder // points
        endpoint_v = remainder % points
        source_mass = flat_mass[safe_indices] * valid_source.astype(mass.dtype)

        destination_u = safe_destinations[endpoint_u[:, None], branch_a[None, :]]
        destination_v = safe_destinations[endpoint_v[:, None], branch_b[None, :]]
        new_state = transitions[
            old_state[:, None], branch_a[None, :], branch_b[None, :]
        ]
        probability = (
            source_policy[endpoint_u[:, None], branch_a[None, :]]
            * source_policy[endpoint_v[:, None], branch_b[None, :]]
        )
        branch_mass = source_mass[:, None] * probability
        linear_destination = (
            (new_state * points + destination_u) * points + destination_v
        )
        return accumulator.at[linear_destination.reshape(-1)].add(
            branch_mass.reshape(-1)
        )

    output = jax.lax.fori_loop(0, chunk_count, add_chunk, output)
    return JAXPairStepResult(
        output.reshape(expected),
        dynamics,
        destinations,
        destinations_valid,
    )


pair_mass_step_jit = partial(jax.jit, static_argnames=("chunk_size",))(
    pair_mass_step_jax
)


def pair_mass_step_separable_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    *,
    row_block_size: int,
    column_block_size: int,
) -> JAXPairStepResult:
    """Apply the same pushforward through sequential endpoint transports.

    Each old-state/action branch gathers a bounded source tile, applies the two
    one-dimensional policy factors, and scatters it through the independently
    gathered row and column maps.  This is the blocked algebraic equivalent of
    row transport followed by column transport, without retaining an ``M x M``
    intermediate. JAX/XLA scatter-add ordering is backend dependent when
    several sources collide.
    """

    mass = jnp.asarray(pair_mass)
    points = grid.agent_point_count
    expected = (2, points, points)
    if mass.shape != expected:
        raise ValueError(f"flat pair mass must have shape {expected}")
    if not jnp.issubdtype(mass.dtype, jnp.floating):
        raise TypeError("pair mass must use a floating dtype")
    for name, value in (
        ("row_block_size", row_block_size),
        ("column_block_size", column_block_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    row_block = min(row_block_size, points)
    column_block = min(column_block_size, points)
    row_block_count = (points + row_block - 1) // row_block
    column_block_count = (points + column_block - 1) // column_block

    dynamics = conditional_dynamics_jax(mass, grid, alpha, tau)
    destinations, destinations_valid = legacy_destination_indices_jax(dynamics, grid)
    safe_destinations = jnp.clip(destinations, 0, points - 1)
    source_policy = _policy(grid, tau, mass.dtype)
    transitions = jnp.asarray(TRANSITION_TENSOR, dtype=jnp.int32)
    row_offsets = jnp.arange(row_block, dtype=jnp.int32)
    column_offsets = jnp.arange(column_block, dtype=jnp.int32)
    output = jnp.zeros_like(mass)

    def add_branch(branch_index, branch_output):
        old_state = branch_index // 4
        action_pair = branch_index % 4
        action_u = action_pair // 2
        action_v = action_pair % 2
        next_state = transitions[old_state, action_u, action_v]

        def add_row_block(block_index, output_after_rows):
            source_rows = block_index * row_block + row_offsets
            valid_rows = source_rows < points
            safe_rows = jnp.minimum(source_rows, points - 1)
            destination_rows = safe_destinations[safe_rows, action_u]

            def add_column_block(column_index, output_after_columns):
                source_columns = column_index * column_block + column_offsets
                valid_columns = source_columns < points
                safe_columns = jnp.minimum(source_columns, points - 1)
                source = mass[
                    old_state,
                    safe_rows[:, None],
                    safe_columns[None, :],
                ]
                valid_tile = valid_rows[:, None] & valid_columns[None, :]
                weighted = (
                    source
                    * source_policy[safe_rows, action_u][:, None]
                    * source_policy[safe_columns, action_v][None, :]
                    * valid_tile.astype(mass.dtype)
                )
                destination_columns = safe_destinations[
                    safe_columns, action_v
                ]
                return output_after_columns.at[
                    next_state,
                    destination_rows[:, None],
                    destination_columns[None, :],
                ].add(weighted)

            return jax.lax.fori_loop(
                0,
                column_block_count,
                add_column_block,
                output_after_rows,
            )

        return jax.lax.fori_loop(
            0, row_block_count, add_row_block, branch_output
        )

    output = jax.lax.fori_loop(0, 8, add_branch, output)
    return JAXPairStepResult(output, dynamics, destinations, destinations_valid)


pair_mass_step_separable_jit = partial(
    jax.jit,
    static_argnames=("row_block_size", "column_block_size"),
)(pair_mass_step_separable_jax)


def pair_mass_step_selected_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    *,
    kernel: str = PAIR_KERNEL_FLAT,
    chunk_size: int = 1,
    row_block_size: int = 1,
    column_block_size: int = 1,
) -> JAXPairStepResult:
    """Dispatch one exact nearest-grid step through an explicit static kernel."""

    if kernel == PAIR_KERNEL_FLAT:
        return pair_mass_step_jax(
            pair_mass, grid, alpha, tau, chunk_size=chunk_size
        )
    if kernel == PAIR_KERNEL_SEPARABLE:
        return pair_mass_step_separable_jax(
            pair_mass,
            grid,
            alpha,
            tau,
            row_block_size=row_block_size,
            column_block_size=column_block_size,
        )
    raise ValueError(f"kernel must be one of {sorted(PAIR_KERNELS)}")


pair_mass_step_selected_jit = partial(
    jax.jit,
    static_argnames=(
        "kernel",
        "chunk_size",
        "row_block_size",
        "column_block_size",
    ),
)(pair_mass_step_selected_jax)


def pair_diagnostics_jax(
    pair_mass: jax.Array,
    grid: JAXPairGrid,
    tau,
    *,
    tolerance=0.0,
) -> JAXPairDiagnostics:
    """Lean device-side diagnostics; never renormalize the stored mass."""

    mass = jnp.asarray(pair_mass)
    moments = one_edge_moments_jax(mass, grid, tau)
    total = jnp.sum(mass)
    state_masses = jnp.sum(mass, axis=(1, 2))
    policy = _policy(grid, tau, mass.dtype)
    denominator = jnp.where(total > 0, total, jnp.ones((), dtype=mass.dtype))
    mean_q = jnp.sum(moments.focal_mass[:, None] * grid.q_points, axis=0) / denominator
    mean_policy = jnp.sum(moments.focal_mass[:, None] * policy, axis=0) / denominator
    nan_pair = jnp.full((2,), jnp.nan, dtype=mass.dtype)
    mean_q = jnp.where(total > 0, mean_q, nan_pair)
    mean_policy = jnp.where(total > 0, mean_policy, nan_pair)
    symmetry_error = jnp.max(jnp.abs(mass - jnp.transpose(mass, (0, 2, 1))))
    minimum_mass = jnp.min(mass)
    finite = jnp.all(jnp.isfinite(mass))
    tolerance_value = jnp.asarray(tolerance, dtype=mass.dtype)
    nonnegative = minimum_mass >= -tolerance_value
    weight_sums = jnp.sum(moments.state_opponent_action_probability, axis=(1, 2))
    weight_error = jnp.max(
        jnp.where(moments.occupied, jnp.abs(weight_sums - 1), 0)
    )
    occupied_variance = jnp.where(
        moments.occupied[:, None], moments.variance, jnp.inf
    )
    any_occupied = jnp.any(moments.occupied)
    minimum_variance = jnp.where(any_occupied, jnp.min(occupied_variance), 0)
    moments_finite = jnp.all(
        jnp.where(
            moments.occupied[:, None],
            jnp.isfinite(moments.mean) & jnp.isfinite(moments.second),
            True,
        )
    )
    moments_valid = (
        finite
        & nonnegative
        & (weight_error <= tolerance_value)
        & (minimum_variance >= -tolerance_value)
        & moments_finite
    )
    return JAXPairDiagnostics(
        total,
        state_masses,
        mean_q,
        mean_policy,
        symmetry_error,
        minimum_mass,
        finite,
        nonnegative,
        weight_error,
        minimum_variance,
        moments_valid,
    )


def simulate_pair_density_jax(
    initial_mass: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    *,
    steps: int,
    chunk_size: int,
    diagnostic_tolerance,
    kernel: str = PAIR_KERNEL_FLAT,
    row_block_size: int = 1,
    column_block_size: int = 1,
) -> JAXPairSimulationResult:
    """Run ``lax.scan`` while retaining only lean post-step diagnostics."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")

    def body(carry, _):
        mass, valid_so_far = carry
        step = pair_mass_step_selected_jax(
            mass,
            grid,
            alpha,
            tau,
            kernel=kernel,
            chunk_size=chunk_size,
            row_block_size=row_block_size,
            column_block_size=column_block_size,
        )
        diagnostics = pair_diagnostics_jax(
            step.mass,
            grid,
            tau,
            tolerance=diagnostic_tolerance,
        )
        return (step.mass, valid_so_far & step.destinations_valid), diagnostics

    (final_mass, destinations_valid), diagnostics = jax.lax.scan(
        body,
        (initial_mass, jnp.asarray(True)),
        xs=None,
        length=steps,
    )
    return JAXPairSimulationResult(final_mass, diagnostics, destinations_valid)


simulate_pair_density_jit = partial(
    jax.jit,
    static_argnames=(
        "steps",
        "chunk_size",
        "kernel",
        "row_block_size",
        "column_block_size",
    ),
)(simulate_pair_density_jax)


def simulate_pair_source_summaries_jax(
    initial_mass: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    source_slot_by_time: jax.Array,
    *,
    steps: int,
    summary_count: int,
    chunk_size: int,
    diagnostic_tolerance,
    kernel: str = PAIR_KERNEL_FLAT,
    row_block_size: int = 1,
    column_block_size: int = 1,
) -> JAXPairSourceSimulationResult:
    """Scan from ``P_0`` and retain only requested source summaries.

    ``source_slot_by_time[t]`` is ``-1`` when ``P_t`` is not requested and is
    otherwise the output slot for its point-sufficient summary. Diagnostics are
    retained for every ``P_t``, including the initial and final masses; no full
    density trajectory is retained.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if (
        isinstance(summary_count, bool)
        or not isinstance(summary_count, int)
        or summary_count < 1
    ):
        raise ValueError("summary_count must be a positive integer")
    slots = jnp.asarray(source_slot_by_time)
    if slots.shape != (steps + 1,) or not jnp.issubdtype(slots.dtype, jnp.integer):
        raise ValueError("source_slot_by_time must be an integer array of length steps + 1")

    points = grid.agent_point_count
    dtype = initial_mass.dtype
    summaries = JAXPairPointSufficient(
        focal_mass=jnp.zeros((summary_count, points), dtype=dtype),
        selected_mass=jnp.zeros((summary_count, points, 2), dtype=dtype),
        sum_y=jnp.zeros((summary_count, points, 2), dtype=dtype),
        sum_y2=jnp.zeros((summary_count, points, 2), dtype=dtype),
        sum_distinct_y=jnp.zeros((summary_count, points, 2), dtype=dtype),
        sum_q=jnp.zeros((summary_count, points, 2), dtype=dtype),
        sum_q2=jnp.zeros((summary_count, points, 2), dtype=dtype),
        sum_y_q=jnp.zeros((summary_count, points, 2), dtype=dtype),
    )

    def record_source(mass, stored, slot):
        def record(_):
            point = pair_point_sufficient_jax(mass, grid, tau)
            return jax.tree_util.tree_map(
                lambda destination, value: destination.at[slot].set(value),
                stored,
                point,
            )

        return jax.lax.cond(slot >= 0, record, lambda _: stored, operand=None)

    def body(carry, time_index):
        mass, stored = carry
        stored = record_source(mass, stored, slots[time_index])
        diagnostics = pair_diagnostics_jax(
            mass, grid, tau, tolerance=diagnostic_tolerance
        )
        step = pair_mass_step_selected_jax(
            mass,
            grid,
            alpha,
            tau,
            kernel=kernel,
            chunk_size=chunk_size,
            row_block_size=row_block_size,
            column_block_size=column_block_size,
        )
        return (step.mass, stored), (diagnostics, step.destinations_valid)

    (final_mass, summaries), (diagnostic_history, destinations_valid) = jax.lax.scan(
        body,
        (initial_mass, summaries),
        jnp.arange(steps, dtype=jnp.int32),
    )
    summaries = record_source(final_mass, summaries, slots[steps])
    final_diagnostics = pair_diagnostics_jax(
        final_mass, grid, tau, tolerance=diagnostic_tolerance
    )
    diagnostics = jax.tree_util.tree_map(
        lambda history, final: jnp.concatenate((history, final[None]), axis=0),
        diagnostic_history,
        final_diagnostics,
    )
    return JAXPairSourceSimulationResult(
        final_mass, summaries, diagnostics, destinations_valid
    )


simulate_pair_source_summaries_jit = partial(
    jax.jit,
    static_argnames=(
        "steps",
        "summary_count",
        "chunk_size",
        "kernel",
        "row_block_size",
        "column_block_size",
    ),
)(simulate_pair_source_summaries_jax)


def simulate_pair_source_summaries_from_histogram_jax(
    histogram_mass: jax.Array,
    state_probabilities: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    source_slot_by_time: jax.Array,
    *,
    steps: int,
    summary_count: int,
    chunk_size: int,
    diagnostic_tolerance,
    kernel: str = PAIR_KERNEL_SEPARABLE,
    row_block_size: int = 1,
    column_block_size: int = 1,
) -> JAXPairBoundedSourceResult:
    """Initialize on device, scan, and return no complete final density."""

    initial_mass = ordered_pair_mass_from_histogram_jax(
        histogram_mass, state_probabilities
    )
    result = simulate_pair_source_summaries_jax(
        initial_mass,
        grid,
        alpha,
        tau,
        source_slot_by_time,
        steps=steps,
        summary_count=summary_count,
        chunk_size=chunk_size,
        diagnostic_tolerance=diagnostic_tolerance,
        kernel=kernel,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
    )
    return JAXPairBoundedSourceResult(
        result.source_summaries,
        result.diagnostics,
        result.destinations_valid,
    )


def simulate_pair_source_summaries_from_histogram_full_jax(
    histogram_mass: jax.Array,
    state_probabilities: jax.Array,
    grid: JAXPairGrid,
    alpha,
    tau,
    source_slot_by_time: jax.Array,
    *,
    steps: int,
    summary_count: int,
    chunk_size: int,
    diagnostic_tolerance,
    kernel: str = PAIR_KERNEL_SEPARABLE,
    row_block_size: int = 1,
    column_block_size: int = 1,
) -> JAXPairSourceSimulationResult:
    """Validation-only combined initializer/scan that returns the final mass."""

    initial_mass = ordered_pair_mass_from_histogram_jax(
        histogram_mass, state_probabilities
    )
    return simulate_pair_source_summaries_jax(
        initial_mass,
        grid,
        alpha,
        tau,
        source_slot_by_time,
        steps=steps,
        summary_count=summary_count,
        chunk_size=chunk_size,
        diagnostic_tolerance=diagnostic_tolerance,
        kernel=kernel,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
    )


simulate_pair_source_summaries_from_histogram_jit = partial(
    jax.jit,
    static_argnames=(
        "steps",
        "summary_count",
        "chunk_size",
        "kernel",
        "row_block_size",
        "column_block_size",
    ),
)(simulate_pair_source_summaries_from_histogram_jax)


simulate_pair_source_summaries_from_histogram_full_jit = partial(
    jax.jit,
    static_argnames=(
        "steps",
        "summary_count",
        "chunk_size",
        "kernel",
        "row_block_size",
        "column_block_size",
    ),
)(simulate_pair_source_summaries_from_histogram_full_jax)


def validate_pair_source_diagnostics(
    diagnostics: JAXPairDiagnostics,
    destinations_valid,
    *,
    diagnostic_tolerance: float,
    symmetry_tolerance: float,
) -> None:
    """Apply the configured Phase 4 conditions to every retained ``P_t``."""

    if not math.isfinite(diagnostic_tolerance) or diagnostic_tolerance < 0:
        raise ValueError("diagnostic_tolerance must be finite and non-negative")
    if not math.isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance must be finite and non-negative")
    host = jax.device_get(diagnostics)
    if not bool(np.all(np.asarray(destinations_valid, dtype=bool))):
        raise GridBoundsError("legacy projected destination lies outside the JAX pair grid")
    if not bool(np.all(np.asarray(host.finite, dtype=bool))):
        raise ValueError("pair trajectory contains non-finite mass")
    if not bool(np.all(np.asarray(host.nonnegative, dtype=bool))):
        raise ValueError("pair trajectory contains negative mass beyond diagnostic_tolerance")
    if bool(np.any(np.abs(np.asarray(host.total_mass) - 1.0) > diagnostic_tolerance)):
        raise ValueError("pair trajectory mass error exceeds diagnostic_tolerance")
    if bool(np.any(np.asarray(host.symmetry_error) > symmetry_tolerance)):
        raise PairSymmetryError("pair trajectory lost endpoint exchange symmetry")
    if bool(
        np.any(np.asarray(host.conditional_weight_error) > diagnostic_tolerance)
    ):
        raise ValueError("conditional-weight error exceeds diagnostic_tolerance")
    if bool(
        np.any(np.asarray(host.minimum_conditional_variance) < -diagnostic_tolerance)
    ):
        raise ValueError("conditional variance is below -diagnostic_tolerance")
    if not bool(np.all(np.asarray(host.conditional_moments_valid, dtype=bool))):
        raise ValueError("pair trajectory contains invalid conditional moments")


def checked_simulate_pair_source_summaries(
    initial_mass,
    grid: JAXPairGrid,
    alpha,
    tau,
    source_slot_by_time,
    *,
    steps: int,
    summary_count: int,
    chunk_size: int,
    symmetry_tolerance: float,
    diagnostic_tolerance: float,
    kernel: str = PAIR_KERNEL_FLAT,
    row_block_size: int = 1,
    column_block_size: int = 1,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> JAXPairSourceSimulationResult:
    """Run and validate the jitted Phase 5 function outside guarded runners."""

    validated, slots = prepare_pair_source_inputs(
        initial_mass,
        grid,
        alpha,
        tau,
        source_slot_by_time,
        steps=steps,
        summary_count=summary_count,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    result = simulate_pair_source_summaries_jit(
        validated,
        grid,
        alpha,
        tau,
        slots,
        steps=steps,
        summary_count=summary_count,
        chunk_size=chunk_size,
        diagnostic_tolerance=diagnostic_tolerance,
        kernel=kernel,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
    )
    return validate_pair_source_result(
        result,
        grid,
        steps=steps,
        summary_count=summary_count,
        diagnostic_tolerance=diagnostic_tolerance,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )


def prepare_pair_source_inputs(
    initial_mass,
    grid: JAXPairGrid,
    alpha,
    tau,
    source_slot_by_time,
    *,
    steps: int,
    summary_count: int,
    symmetry_tolerance: float,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> tuple[jax.Array, jax.Array]:
    """Validate runtime inputs without selecting or compiling an executable."""

    _validate_learning_scalars(alpha, tau)
    slots = np.asarray(jax.device_get(source_slot_by_time))
    if slots.shape != (steps + 1,) or not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("source_slot_by_time must be an integer array of length steps + 1")
    if np.any(slots < -1) or np.any(slots >= summary_count):
        raise ValueError("source slots must be -1 or valid summary indices")
    if not np.array_equal(
        np.sort(slots[slots >= 0]), np.arange(summary_count, dtype=slots.dtype)
    ):
        raise ValueError("every summary slot must occur exactly once")
    validated = validate_jax_pair_mass(
        initial_mass,
        grid,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    return validated, jnp.asarray(slots)


def validate_pair_source_result(
    result: JAXPairSourceSimulationResult,
    grid: JAXPairGrid,
    *,
    steps: int,
    summary_count: int,
    diagnostic_tolerance: float,
    symmetry_tolerance: float,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> JAXPairSourceSimulationResult:
    """Validate outputs from an already executed pair-source executable."""

    points = grid.agent_point_count
    if result.final_mass.shape != (2, points, points):
        raise ValueError("compiled final pair mass has an unexpected shape")
    if result.source_summaries.focal_mass.shape != (summary_count, points):
        raise ValueError("compiled focal source summary has an unexpected shape")
    for name in (
        "selected_mass",
        "sum_y",
        "sum_y2",
        "sum_distinct_y",
        "sum_q",
        "sum_q2",
        "sum_y_q",
    ):
        if getattr(result.source_summaries, name).shape != (summary_count, points, 2):
            raise ValueError(f"compiled source summary {name} has an unexpected shape")
    if result.diagnostics.total_mass.shape != (steps + 1,):
        raise ValueError("compiled diagnostic trajectory has an unexpected length")
    if result.destinations_valid.shape != (steps,):
        raise ValueError("compiled destination-validity trajectory has an unexpected length")
    validate_pair_source_diagnostics(
        result.diagnostics,
        result.destinations_valid,
        diagnostic_tolerance=diagnostic_tolerance,
        symmetry_tolerance=symmetry_tolerance,
    )
    if max_elements is not None and result.final_mass.size > max_elements:
        raise MemoryError("compiled final pair mass exceeds its validation limit")
    host_mass = np.asarray(jax.device_get(result.final_mass))
    if not np.all(np.isfinite(host_mass)):
        raise ValueError("compiled final pair mass contains non-finite values")
    if float(np.min(host_mass, initial=0.0)) < -diagnostic_tolerance:
        raise ValueError("compiled final pair mass is negative beyond diagnostic_tolerance")
    if abs(float(host_mass.sum()) - 1.0) > diagnostic_tolerance:
        raise ValueError("compiled final pair mass error exceeds diagnostic_tolerance")
    symmetry_error = float(
        np.max(np.abs(host_mass - host_mass.transpose(0, 2, 1)), initial=0.0)
    )
    if symmetry_error > symmetry_tolerance:
        raise PairSymmetryError("compiled final pair mass lost endpoint exchange symmetry")
    return result


def execute_compiled_pair_source_summaries(
    compiled_callable,
    initial_mass,
    grid: JAXPairGrid,
    alpha,
    tau,
    source_slot_by_time,
    *,
    steps: int,
    summary_count: int,
    symmetry_tolerance: float,
    diagnostic_tolerance: float,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> JAXPairSourceSimulationResult:
    """Invoke one injected compiled callable exactly once, then validate outputs."""

    validated, slots = prepare_pair_source_inputs(
        initial_mass,
        grid,
        alpha,
        tau,
        source_slot_by_time,
        steps=steps,
        summary_count=summary_count,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    result = compiled_callable(
        validated,
        grid,
        alpha,
        tau,
        slots,
        diagnostic_tolerance=diagnostic_tolerance,
    )
    return validate_pair_source_result(
        result,
        grid,
        steps=steps,
        summary_count=summary_count,
        diagnostic_tolerance=diagnostic_tolerance,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )


def checked_pair_mass_step(
    pair_mass,
    grid: JAXPairGrid,
    alpha,
    tau,
    *,
    chunk_size: int,
    symmetry_tolerance: float,
    kernel: str = PAIR_KERNEL_FLAT,
    row_block_size: int = 1,
    column_block_size: int = 1,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> JAXPairStepResult:
    """Validated host wrapper around the compiled-compatible one-step kernel."""

    _validate_learning_scalars(alpha, tau)
    validated = validate_jax_pair_mass(
        pair_mass,
        grid,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    result = pair_mass_step_selected_jit(
        validated,
        grid,
        alpha,
        tau,
        kernel=kernel,
        chunk_size=chunk_size,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
    )
    if not bool(np.asarray(result.destinations_valid)):
        raise GridBoundsError("legacy projected destination lies outside the JAX pair grid")
    validate_jax_pair_mass(
        result.mass,
        grid,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    return result


def checked_simulate_pair_density(
    initial_mass,
    grid: JAXPairGrid,
    alpha,
    tau,
    *,
    steps: int,
    chunk_size: int,
    symmetry_tolerance: float,
    diagnostic_tolerance: float,
    kernel: str = PAIR_KERNEL_FLAT,
    row_block_size: int = 1,
    column_block_size: int = 1,
    max_elements: int | None = DEFAULT_MAX_JAX_PAIR_ELEMENTS,
) -> JAXPairSimulationResult:
    """Validate before/after a lean compiled scan and reject invalid destinations."""

    _validate_learning_scalars(alpha, tau)
    if not math.isfinite(diagnostic_tolerance) or diagnostic_tolerance < 0:
        raise ValueError("diagnostic_tolerance must be finite and non-negative")
    validated = validate_jax_pair_mass(
        initial_mass,
        grid,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    result = simulate_pair_density_jit(
        validated,
        grid,
        alpha,
        tau,
        steps=steps,
        chunk_size=chunk_size,
        diagnostic_tolerance=diagnostic_tolerance,
        kernel=kernel,
        row_block_size=row_block_size,
        column_block_size=column_block_size,
    )
    if not bool(np.asarray(result.destinations_valid)):
        raise GridBoundsError("legacy projected destination lies outside the JAX pair grid")
    if steps:
        diagnostics = jax.device_get(result.diagnostics)
        if not bool(np.all(np.asarray(diagnostics.finite))):
            raise ValueError("pair trajectory contains non-finite mass")
        if not bool(np.all(np.asarray(diagnostics.nonnegative))):
            raise ValueError("pair trajectory contains negative mass")
        if not bool(np.all(np.asarray(diagnostics.conditional_moments_valid))):
            raise ValueError("pair trajectory contains invalid conditional moments")
        if bool(
            np.any(np.asarray(diagnostics.symmetry_error) > symmetry_tolerance)
        ):
            raise PairSymmetryError("pair trajectory lost endpoint exchange symmetry")
    validate_jax_pair_mass(
        result.final_mass,
        grid,
        symmetry_tolerance=symmetry_tolerance,
        max_elements=max_elements,
    )
    return result

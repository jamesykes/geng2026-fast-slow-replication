"""Small NumPy pair-mass reference implementation."""

from .numpy_reference import (
    ConditionalDynamics,
    OneEdgeMoments,
    PairStepResult,
    ReferenceSizeError,
    conditional_dynamics,
    endpoint_exchange_symmetry_error,
    focal_marginal,
    one_edge_moments,
    pair_mass_step,
)

__all__ = [
    "ConditionalDynamics",
    "OneEdgeMoments",
    "PairStepResult",
    "ReferenceSizeError",
    "conditional_dynamics",
    "endpoint_exchange_symmetry_error",
    "focal_marginal",
    "one_edge_moments",
    "pair_mass_step",
]


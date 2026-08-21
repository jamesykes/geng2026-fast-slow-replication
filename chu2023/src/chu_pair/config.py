"""Immutable model configuration independent of grid policy."""

from __future__ import annotations

from dataclasses import dataclass
import math


MAX_ABM_SEED = 2**32 - 1

# Explicit dot-product precision for the pair-density contractions.  XLA lowers
# float32 ``dot_general`` to TF32 tensor cores by default on Ampere and newer
# NVIDIA hardware (~1e-3 relative accuracy), which inflated the H100
# conditional-weight residual from the float32 rounding scale (~1.2e-7) to
# ~4e-4 and violated the reviewed 1e-4 diagnostic tolerance.  Declared here so
# configuration and provenance can record it without importing JAX.  It is a
# fixed policy: configuration cannot restore the platform default.
PAIR_CONTRACTION_PRECISION = "highest"


def validate_abm_seed(seed: int) -> int:
    """Validate a seed representable without aliasing by ``PRNGKey``."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("abm_seed must be an integer")
    if not 0 <= seed <= MAX_ABM_SEED:
        raise ValueError(f"abm_seed must lie in [0, {MAX_ABM_SEED}]")
    return seed


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Parameters of the continuous two-action Q-learning model."""

    alpha: float = 0.4
    tau: float = 2.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not math.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("tau must be finite and non-negative")


DEFAULT_LEARNING = LearningConfig()


@dataclass(frozen=True, slots=True)
class ABMConfig:
    """Shape and seed parameters for a finite-population ABM run."""

    num_agents: int = 16
    steps: int = 12
    num_runs: int = 4
    abm_seed: int = 20230819
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("num_agents", self.num_agents, 2),
            ("steps", self.steps, 0),
            ("num_runs", self.num_runs, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
        validate_abm_seed(self.abm_seed)
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")

    @property
    def edge_count(self) -> int:
        return self.num_agents * (self.num_agents - 1) // 2

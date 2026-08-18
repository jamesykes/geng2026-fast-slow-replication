"""Immutable model configuration independent of grid policy."""

from __future__ import annotations

from dataclasses import dataclass
import math


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


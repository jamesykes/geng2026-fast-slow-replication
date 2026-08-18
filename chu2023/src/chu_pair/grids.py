"""Q-grid construction and exact legacy nearest-grid projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


class GridError(ValueError):
    """Base class for invalid grids or grid coordinates."""


class GridBoundsError(GridError):
    """Raised when a projected destination lies outside the configured grid."""


@dataclass(frozen=True, slots=True)
class QGrid:
    """Common one-dimensional grid used for both Q coordinates."""

    q_min: float = -0.1
    q_max: float = 1.2
    spacing: float = 0.01

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.q_min, self.q_max, self.spacing)):
            raise GridError("grid bounds and spacing must be finite")
        if self.spacing <= 0.0:
            raise GridError("spacing must be positive")
        if self.q_max <= self.q_min:
            raise GridError("q_max must exceed q_min")

        intervals = (self.q_max - self.q_min) / self.spacing
        if not np.isclose(intervals, round(intervals), rtol=0.0, atol=1e-10):
            raise GridError("the inclusive range must contain an integer number of intervals")

        # Legacy appro() searches multiples of spacing relative to zero.
        for name, value in (("q_min", self.q_min), ("q_max", self.q_max)):
            ratio = value / self.spacing
            if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10):
                raise GridError(f"{name} must align with legacy spacing multiples")

    @property
    def size(self) -> int:
        return int(round((self.q_max - self.q_min) / self.spacing)) + 1

    @property
    def agent_point_count(self) -> int:
        return self.size * self.size

    @property
    def decimal_places(self) -> int:
        text = str(self.spacing)
        return len(text.split(".", maxsplit=1)[1]) if "." in text else 0

    @property
    def values(self) -> NDArray[np.float64]:
        values = self.q_min + self.spacing * np.arange(self.size, dtype=np.float64)
        return np.around(values, self.decimal_places)

    @property
    def q_points(self) -> NDArray[np.float64]:
        q_c, q_d = np.meshgrid(self.values, self.values, indexing="ij")
        return np.stack((q_c, q_d), axis=-1)

    @property
    def flat_q_points(self) -> NDArray[np.float64]:
        return self.q_points.reshape(self.agent_point_count, 2)

    def flatten_index(self, q_c_index: int, q_d_index: int) -> int:
        self._validate_axis_index(q_c_index)
        self._validate_axis_index(q_d_index)
        return int(q_c_index) * self.size + int(q_d_index)

    def unflatten_index(self, flat_index: int) -> tuple[int, int]:
        if not 0 <= int(flat_index) < self.agent_point_count:
            raise GridBoundsError(f"flat index {flat_index} is outside the agent grid")
        return divmod(int(flat_index), self.size)

    def legacy_project_value(self, number: float) -> float:
        """Reproduce active case2_1.py appro() behaviour without clipping."""

        if not math.isfinite(number):
            raise GridBoundsError("cannot project a non-finite Q value")

        point_len = self.decimal_places
        factor = 10**point_len
        increment = 0.1**point_len
        spacing_ticks = int(np.around(self.spacing * factor))
        if spacing_ticks <= 0:
            raise GridError("spacing is too small for its decimal representation")

        left = float(np.around(number, point_len))
        right = left

        while True:
            right_ticks = int(np.around(right * factor))
            if right_ticks % spacing_ticks == 0:
                break
            right += increment

        while True:
            left_ticks = int(np.around(left * factor))
            if left_ticks % spacing_ticks == 0:
                break
            left -= increment

        left_value = left_ticks / factor
        right_value = right_ticks / factor
        if abs(left_value - number) <= abs(right_value - number):
            return float(left_value)
        return float(right_value)

    def legacy_project_index(self, number: float) -> int:
        """Project and return an index, explicitly rejecting out-of-range values."""

        projected = self.legacy_project_value(number)
        tolerance = 1e-12 * max(1.0, abs(self.q_min), abs(self.q_max))
        if projected < self.q_min - tolerance or projected > self.q_max + tolerance:
            raise GridBoundsError(
                f"projected Q value {projected} is outside [{self.q_min}, {self.q_max}]"
            )

        index = int(round((projected - self.q_min) / self.spacing))
        self._validate_axis_index(index)
        if not np.isclose(self.values[index], projected, rtol=0.0, atol=tolerance):
            raise GridError(f"projected value {projected} does not lie on the configured grid")
        return index

    def q_at(self, q_c_index: int, q_d_index: int) -> NDArray[np.float64]:
        self._validate_axis_index(q_c_index)
        self._validate_axis_index(q_d_index)
        return np.array([self.values[q_c_index], self.values[q_d_index]], dtype=np.float64)

    def _validate_axis_index(self, index: int) -> None:
        if not 0 <= int(index) < self.size:
            raise GridBoundsError(f"grid index {index} is outside [0, {self.size})")


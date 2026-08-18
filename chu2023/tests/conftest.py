from __future__ import annotations

import pytest

from chu_pair.grids import QGrid


@pytest.fixture
def coarse_grid() -> QGrid:
    return QGrid(q_min=-1.0, q_max=1.0, spacing=0.5)

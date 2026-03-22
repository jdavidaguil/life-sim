"""Phase: regenerate grid resources, slowed by grazing pressure."""

from __future__ import annotations

import numpy as np

from src.core.state import SimState


def regenerate(state: SimState) -> None:
    """Build a per-cell agent-pressure map and regenerate the grid.

    Reads from scratch:
        ``cell_groups``: dict mapping (x, y) -> list[agent index],
            recording how many agents occupied each cell this step.
    """
    cell_groups = state.scratch["cell_groups"]
    pressure = np.zeros(
        (state.grid.height, state.grid.width), dtype=np.float32
    )
    for (x, y), indices in cell_groups.items():
        pressure[y, x] = len(indices)
    state.grid.regenerate(pressure)

"""Phase: group agents by cell, split resources, compute crowding penalties."""

from __future__ import annotations

import collections
from typing import List

from src.core.state import SimState

# Overcrowding: agents beyond this count on one cell each pay an extra decay.
OVERCROWD_THRESHOLD: int = 3
# Extra energy drained per agent above the threshold per step.
OVERCROWD_PENALTY: float = 0.15


def consume(state: SimState) -> None:
    """Distribute cell resources proportionally and flag crowded cells.

    Reads from scratch:
        ``desired``: list[float] — per-agent desired consumption amounts.

    Writes to scratch:
        ``gains``: list[float] — actual energy gained by each agent.
        ``crowding_penalties``: list[float] — extra decay penalty per agent.
        ``cell_groups``: dict mapping (x, y) -> list[agent index].
    """
    desired: List[float] = state.scratch["desired"]

    cell_groups: dict[tuple[int, int], List[int]] = collections.defaultdict(list)
    for i, agent in enumerate(state.agents):
        cell_groups[(agent.x, agent.y)].append(i)

    gains: List[float] = [0.0] * len(state.agents)
    crowding_penalties: List[float] = [0.0] * len(state.agents)

    for (x, y), indices in cell_groups.items():
        total_desired = sum(desired[i] for i in indices)
        actual = state.grid.consume_resource(x, y, total_desired)
        # Distribute actual gain proportionally to each agent's desire.
        if total_desired > 0:
            for i in indices:
                gains[i] = actual * (desired[i] / total_desired)
        # Overcrowding penalty: every agent above the threshold pays extra.
        excess = max(0, len(indices) - OVERCROWD_THRESHOLD)
        if excess > 0:
            for i in indices:
                crowding_penalties[i] = excess * OVERCROWD_PENALTY

    state.scratch["gains"] = gains
    state.scratch["crowding_penalties"] = crowding_penalties
    state.scratch["cell_groups"] = cell_groups

"""Phase: move agents and sample desired consumption amounts."""

from __future__ import annotations

import collections
from typing import List

from src.core.state import SimState


def move(state: SimState) -> None:
    """Move every agent according to its policy and record desired consumption.

    Writes to scratch:
        ``desired``: list[float] — per-agent desired consumption amount,
            indexed in the same order as ``state.agents``.
    """
    occupancy: dict[tuple[int, int], int] = collections.defaultdict(int)
    for agent in state.agents:
        occupancy[(agent.x, agent.y)] += 1

    pop_mean_energy: float = (
        sum(a.energy for a in state.agents) / len(state.agents)
        if state.agents else 0.0
    )

    desired: List[float] = []
    for agent in state.agents:
        neighbors = state.grid.get_neighbors(agent.x, agent.y)
        if neighbors:
            dx, dy = agent.policy.decide(
                agent, state.grid, occupancy, state.rng,
                pop_mean_energy=pop_mean_energy,
            )
            agent.move(dx, dy)
            agent.last_dx, agent.last_dy = dx, dy
        desired.append(float(state.rng.uniform(0.5, 1.5)))

    state.scratch["desired"] = desired

"""Phase: apply energy gains and stochastic energy decay."""

from __future__ import annotations

from typing import List

from src.core.state import SimState


def decay(state: SimState) -> None:
    """Apply per-agent energy gain then subtract randomised decay.

    Reads from scratch:
        ``gains``: list[float] — energy to add to each agent.
        ``crowding_penalties``: list[float] — extra decay from overcrowding.
    """
    gains: List[float] = state.scratch["gains"]
    crowding_penalties: List[float] = state.scratch["crowding_penalties"]

    for i, agent in enumerate(state.agents):
        agent.energy += gains[i]
        energy_decay = (
            0.5 + float(state.rng.uniform(-0.1, 0.1)) + crowding_penalties[i]
        )
        agent.energy -= energy_decay

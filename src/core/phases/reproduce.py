"""Phase: asexual reproduction — energy split with policy inheritance."""

from __future__ import annotations

from typing import List

from src.core.agent import Agent
from src.core.state import SimState

# Energy threshold above which an agent reproduces.
REPRODUCTION_THRESHOLD: float = 18.0


def reproduce(state: SimState) -> None:
    """Reproduce agents whose energy exceeds the threshold.

    Each reproducing agent splits its energy 50/50 with a child.  The child
    is placed on the same cell and inherits a mutated copy of the parent
    policy.  Agent IDs are drawn from ``state.metrics["_next_id"]``.

    Writes to scratch:
        ``reproductions_this_step``: int — number of reproduction events.
    """
    newborns: List[Agent] = []
    count: int = 0

    for agent in state.agents:
        if agent.energy > REPRODUCTION_THRESHOLD:
            child_energy = agent.energy / 2.0
            agent.energy = child_energy
            next_id: int = state.metrics["_next_id"]
            child = Agent(
                id=next_id,
                x=agent.x,
                y=agent.y,
                policy=agent.policy.mutate(state.rng),
            )
            child.energy = child_energy
            state.metrics["_next_id"] = next_id + 1
            newborns.append(child)
            count += 1

    state.agents.extend(newborns)
    state.scratch["reproductions_this_step"] = count

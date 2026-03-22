"""Phase: sexual reproduction for NeuralPolicy agents — genome crossover."""

from __future__ import annotations

import collections
from typing import List

from src.core.agent import Agent
from src.core.state import SimState
from src.genome.crossover_neural import crossover_neural

# Minimum energy required to be eligible for mating.
MATING_ENERGY_THRESHOLD: float = 15.0
# Energy cost paid by each parent per mating event.
MATING_ENERGY_COST: float = 3.0
# Starting energy for every child.
CHILD_INITIAL_ENERGY: float = 10.0


def reproduce_sexual_neural(
    state: SimState,
    *,
    mating_cost: float = MATING_ENERGY_COST,
    mating_threshold: float = MATING_ENERGY_THRESHOLD,
) -> None:
    """Pair NeuralPolicy agents on the same cell; produce a child via genome crossover.

    For each cell:
    - Collect agents with ``energy > mating_threshold``.
    - If fewer than two eligible agents, skip.
    - Select the two highest-energy eligible agents as the mating pair.
    - Each parent pays ``mating_cost`` energy.
    - Child genome: whole-layer crossover via :func:`crossover_neural`
      (with Gaussian mutation, sigma=0.05).
    - Child starts at the same cell with ``energy = CHILD_INITIAL_ENERGY``.
    - One mating event per eligible cell per step.

    Parameters:
        mating_cost:      energy deducted from each parent (default 3.0).
        mating_threshold: minimum energy to be eligible (default 15.0).

    Reads from scratch:
        ``next_id`` (via ``setdefault``): seeded from current max agent id.

    Writes to scratch:
        ``mating_events``: int — number of mating events this step.
    """
    # Group agents by cell.
    cell_groups: dict[tuple[int, int], List[int]] = collections.defaultdict(list)
    for i, agent in enumerate(state.agents):
        cell_groups[(agent.x, agent.y)].append(i)

    # Seed next_id from current roster if not yet set this step.
    if state.agents:
        state.scratch.setdefault(
            "next_id",
            max(a.id for a in state.agents) + 1,
        )
    else:
        state.scratch.setdefault("next_id", 0)

    newborns: List[Agent] = []
    mating_events: int = 0

    for indices in cell_groups.values():
        eligible = [
            i for i in indices
            if state.agents[i].energy > mating_threshold
        ]
        if len(eligible) < 2:
            continue

        # Pick the two highest-energy eligible agents.
        eligible.sort(key=lambda i: state.agents[i].energy, reverse=True)
        idx_a, idx_b = eligible[0], eligible[1]
        parent_a = state.agents[idx_a]
        parent_b = state.agents[idx_b]

        # Deduct mating cost.
        parent_a.energy -= mating_cost
        parent_b.energy -= mating_cost

        # Produce child genome via crossover.
        child_policy = crossover_neural(parent_a.policy, parent_b.policy, state.rng)

        child_id: int = state.scratch["next_id"]
        state.scratch["next_id"] = child_id + 1

        child = Agent(
            id=child_id,
            x=parent_a.x,
            y=parent_a.y,
            policy=child_policy,
        )
        child.energy = CHILD_INITIAL_ENERGY

        newborns.append(child)
        mating_events += 1

    state.agents.extend(newborns)
    state.scratch["mating_events"] = mating_events

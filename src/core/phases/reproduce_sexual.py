"""Phase: sexual reproduction — pairwise crossover with mutation."""

from __future__ import annotations

import collections
from typing import List

import numpy as np

from src.core.agent import Agent
from src.core.policy import TraitPolicy
from src.core.state import SimState

# Minimum energy required to be eligible for mating.
MATING_ENERGY_THRESHOLD: float = 12.0
# Energy cost paid by each parent per mating event.
MATING_ENERGY_COST: float = 6.0
# Starting energy for every child.
CHILD_INITIAL_ENERGY: float = 10.0
# Per-trait Gaussian mutation standard deviation.
MUTATION_SIGMA: float = 0.05
# Trait value bounds after mutation.
TRAIT_MIN: float = 0.0
TRAIT_MAX: float = 2.0


def reproduce_sexual(
    state: SimState,
    *,
    mating_cost: float = MATING_ENERGY_COST,
    mating_threshold: float = MATING_ENERGY_THRESHOLD,
) -> None:
    """Pair agents on the same cell and produce one child via crossover.

    For each cell:
    - Collect agents with ``energy > mating_threshold``.
    - If fewer than two eligible agents, skip the cell.
    - Select the two with the highest energy as the mating pair.
    - Each parent pays ``mating_cost`` energy.
    - Child genome: uniform crossover — each of the four traits is drawn
      independently from parent A or parent B with 50/50 probability, then
      Gaussian noise (sigma=0.05) is added and the result is clamped to
      ``[TRAIT_MIN, TRAIT_MAX]``.
    - Child inherits the policy ``mode`` of parent A.
    - Child starts at the same cell with ``energy = CHILD_INITIAL_ENERGY``.
    - One mating event per eligible cell per step.

    Parameters:
        mating_cost:      energy deducted from each parent (default 6.0).
        mating_threshold: minimum energy to be eligible (default 12.0).

    Reads from scratch:
        ``next_id`` (via ``setdefault``): next available agent ID, seeded
        from the current maximum id if not already present.

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

        # Create child with a default policy first; traits will be assigned below.
        child_id: int = state.scratch["next_id"]
        state.scratch["next_id"] = child_id + 1

        child = Agent(
            id=child_id,
            x=parent_a.x,
            y=parent_a.y,
            policy=TraitPolicy(
                rng=state.rng,
                mode=getattr(parent_a.policy, "mode", "richer"),
            ),
        )
        child.energy = CHILD_INITIAL_ENERGY

        # Crossover: only when both parents expose a .traits attribute.
        if hasattr(parent_a.policy, "traits") and hasattr(parent_b.policy, "traits"):
            traits_a = parent_a.policy.traits
            traits_b = parent_b.policy.traits
            # Uniform crossover: each of the 4 trait positions drawn independently.
            mask = state.rng.integers(0, 2, size=4).astype(bool)
            child_traits = np.where(mask, traits_a, traits_b).copy()
            # Gaussian mutation, then clamp to valid range.
            child_traits += state.rng.normal(0.0, MUTATION_SIGMA, size=4)
            child_traits = np.clip(child_traits, TRAIT_MIN, TRAIT_MAX).astype(np.float32)
            child.policy.traits = child_traits
        # else: child keeps its default policy unchanged.
        newborns.append(child)
        mating_events += 1

    state.agents.extend(newborns)
    state.scratch["mating_events"] = mating_events

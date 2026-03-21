"""Policy module: defines movement decision strategies for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.agent import Agent
    from src.core.grid import Grid


class Policy:
    """Abstract base class for agent movement policies.

    A policy encapsulates the "brain" of an agent, separating the decision
    logic from the agent's physical state.  Subclasses must implement
    :meth:`decide`.
    """

    def decide(
        self,
        agent: "Agent",
        grid: "Grid",
        occupancy: dict[tuple[int, int], int],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Choose a movement delta for *agent* given the current environment.

        Args:
            agent: The agent making the decision.
            grid: The spatial grid (used to query resources and neighbours).
            occupancy: Mapping of ``(x, y)`` → agent count for the current
                step; lets policies account for competition at target cells.
            rng: Shared random generator for all stochastic decisions.

        Returns:
            ``(dx, dy)`` — the column and row delta to apply.  ``(0, 0)``
            means the agent stays in place.
        """
        raise NotImplementedError


class GreedyPolicy(Policy):
    """Score-maximising policy with inertia and a small exploratory burst.

    For 85% of moves the agent chooses the Moore neighbour with the highest
    score.  The score combines resource availability, a crowding discount for
    over-occupied cells, and an inertia bonus for continuing in the same
    direction as the last step.

    For the remaining 15% a random neighbour is chosen, preventing the
    population from converging to a single deterministic path.

    Class attributes:
        EXPLORE_PROB: Probability of taking a random step instead of greedy.
        OVERCROWD_THRESHOLD: Cell occupancy count above which crowding penalises
            the score.
        OVERCROWD_PENALTY: Score reduction per agent above the threshold.
        INERTIA_BONUS: Score bonus awarded to the cell that continues the
            agent's previous heading.
    """

    EXPLORE_PROB: float = 0.15
    OVERCROWD_THRESHOLD: int = 3
    OVERCROWD_PENALTY: float = 0.4
    INERTIA_BONUS: float = 0.1

    def decide(
        self,
        agent: "Agent",
        grid: "Grid",
        occupancy: dict[tuple[int, int], int],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        neighbors = grid.get_neighbors(agent.x, agent.y)
        if not neighbors:
            return (0, 0)

        if rng.random() < self.EXPLORE_PROB:
            nx, ny = neighbors[int(rng.integers(0, len(neighbors)))]
        else:
            adx, ady = agent.last_dx, agent.last_dy

            def _score(p: tuple[int, int]) -> float:
                res = grid.get_resource(p[0], p[1])
                crowd = occupancy.get(p, 0)
                excess = max(0, crowd - self.OVERCROWD_THRESHOLD)
                base = res - excess * self.OVERCROWD_PENALTY
                if (adx != 0 or ady != 0) and (
                    p[0] - agent.x == adx and p[1] - agent.y == ady
                ):
                    base += self.INERTIA_BONUS
                return base

            nx, ny = max(neighbors, key=_score)

        return (nx - agent.x, ny - agent.y)


class ExplorerPolicy(Policy):
    """Random-walk policy biased toward low-occupancy, resource-rich cells.

    For ``EXPLORE_PROB`` of moves the agent picks a uniformly random
    neighbour regardless of conditions.  Otherwise it moves to the
    least-occupied neighbour, breaking ties in favour of higher resource
    levels.  This encourages agents with this policy to spread into empty
    territory rather than piling onto established hotspots.

    Class attributes:
        EXPLORE_PROB: Probability of a fully random step (default 0.6 —
            explorers are predominantly random).
    """

    EXPLORE_PROB: float = 0.4

    def decide(
        self,
        agent: "Agent",
        grid: "Grid",
        occupancy: dict[tuple[int, int], int],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        neighbors = grid.get_neighbors(agent.x, agent.y)
        if not neighbors:
            return (0, 0)

        if rng.random() < self.EXPLORE_PROB:
            nx, ny = neighbors[int(rng.integers(0, len(neighbors)))]
        else:
            # Prefer least-occupied cell; break ties by highest resource.
            def _score(p: tuple[int, int]) -> tuple[int, float]:
                return (-occupancy.get(p, 0), grid.get_resource(p[0], p[1]))

            nx, ny = max(neighbors, key=_score)

        return (nx - agent.x, ny - agent.y)


POLICIES: list[type] = [GreedyPolicy, ExplorerPolicy]


def mutate(policy, rng: np.random.Generator) -> object:
    """Return a new policy instance, possibly of a different type.
    
    With probability MUTATION_RATE a random different policy type is chosen;
    otherwise the same type is re-instantiated.
    
    Args:
        policy: The parent agent's current policy instance.
        rng: Seeded NumPy random generator for reproducibility.
    
    Returns:
        A new policy object.
    """
    from src.core.simulation import MUTATION_RATE
    if rng.random() < MUTATION_RATE:
        other_types = [p for p in POLICIES if p is not type(policy)]
        if other_types:
            return other_types[int(rng.integers(0, len(other_types)))]()
    return type(policy)()

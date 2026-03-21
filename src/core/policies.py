"""policies.py: lightweight, stateless movement policies for grid agents.

Design
------
All policies expose a single method::

    choose_move(obs) -> (dx, dy)

where *obs* is the dict produced by :func:`get_observation`.  Policies carry
no per-agent state — they are safe to share across agent instances or
re-instantiate freely.

Public API
----------
* :func:`get_observation`  — build a local observation dict for an agent
* :class:`GreedyPolicy`    — always move toward the richest neighbour
* :class:`ExplorerPolicy`  — balances resource, crowding, and random noise
* :data:`POLICIES`         — list of available policy classes
* :data:`MUTATION_RATE`    — probability of a policy type flip on reproduction
* :func:`mutate`           — return a (possibly mutated) new policy instance
* :func:`make_agents`      — convenience factory for a fresh agent roster
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # forward refs only when type-checking


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def get_observation(agent, grid, agents: list) -> dict:
    """Build a local observation dict for *agent*.

    Parameters
    ----------
    agent:
        The agent being observed.
    grid:
        The :class:`~src.core.grid.Grid` the agent inhabits.  Must expose
        ``grid.resources`` (2-D array), ``grid.width``, and ``grid.height``.
    agents:
        Full agent roster (including *agent* itself).

    Returns
    -------
    dict with keys:

    ``neighbors``
        List of ``(dx, dy, resource)`` tuples for all 8 surrounding cells.
        Positions wrap toroidally.
    ``current_resource``
        Resource value at the agent's current cell.
    ``nearby_agents``
        Number of *other* agents within a 1-cell (Chebyshev) radius.
    """
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = (agent.x + dx) % grid.width
            ny = (agent.y + dy) % grid.height
            resource = float(grid.resources[ny, nx])
            neighbors.append((dx, dy, resource))

    current_resource = float(grid.resources[agent.y, agent.x])

    nearby_agents = sum(
        1 for a in agents
        if a is not agent
        and abs(a.x - agent.x) <= 1
        and abs(a.y - agent.y) <= 1
    )

    return {
        "neighbors": neighbors,
        "current_resource": current_resource,
        "nearby_agents": nearby_agents,
    }


# ---------------------------------------------------------------------------
# Policy classes
# ---------------------------------------------------------------------------

class GreedyPolicy:
    """Move to the neighbour with the highest resource level.

    Stays put when the current cell is at least as good as every neighbour.
    """

    name = "greedy"

    def choose_move(self, obs: dict) -> tuple[int, int]:
        """Return ``(dx, dy)`` of the best available move.

        Parameters
        ----------
        obs:
            Observation dict as returned by :func:`get_observation`.

        Returns
        -------
        (dx, dy) — ``(0, 0)`` means stay put.
        """
        best_dx, best_dy, best_res = 0, 0, obs["current_resource"]
        for dx, dy, resource in obs["neighbors"]:
            if resource > best_res:
                best_res = resource
                best_dx, best_dy = dx, dy
        return best_dx, best_dy

    def __repr__(self) -> str:  # pragma: no cover
        return "GreedyPolicy()"


class ExplorerPolicy:
    """Score every candidate cell and move to the highest-scoring one.

    Scoring function::

        score = resource + uniform(0, 0.2) - crowding_penalty

    where ``crowding_penalty = nearby_agents * 0.3`` is applied only to the
    stay-put option ``(0, 0)``.  The random noise breaks ties and occasionally
    sends the agent away from a locally optimal cell.
    """

    name = "explorer"

    _CROWDING_WEIGHT: float = 0.3
    _NOISE_MAX: float = 0.2

    def choose_move(self, obs: dict) -> tuple[int, int]:
        """Return ``(dx, dy)`` of the highest-scoring candidate.

        Parameters
        ----------
        obs:
            Observation dict as returned by :func:`get_observation`.

        Returns
        -------
        (dx, dy) — ``(0, 0)`` means stay put.
        """
        crowding_penalty = obs["nearby_agents"] * self._CROWDING_WEIGHT

        # Candidates: all 8 neighbours + stay-put.
        candidates: list[tuple[int, int, float]] = []

        # Stay-put gets crowding penalty applied.
        stay_score = (
            obs["current_resource"]
            + random.uniform(0, self._NOISE_MAX)
            - crowding_penalty
        )
        candidates.append((0, 0, stay_score))

        for dx, dy, resource in obs["neighbors"]:
            score = resource + random.uniform(0, self._NOISE_MAX)
            candidates.append((dx, dy, score))

        best_dx, best_dy, _ = max(candidates, key=lambda c: c[2])
        return best_dx, best_dy

    def __repr__(self) -> str:  # pragma: no cover
        return "ExplorerPolicy()"


# ---------------------------------------------------------------------------
# Policy registry and mutation
# ---------------------------------------------------------------------------

POLICIES: list[type] = [GreedyPolicy, ExplorerPolicy]
"""All available policy classes.  Extend this list to register new policies."""

MUTATION_RATE: float = 0.05
"""Probability that a reproduced child switches to a different policy type."""


def mutate(policy) -> object:
    """Return a new policy instance, possibly of a different type.

    With probability :data:`MUTATION_RATE` a random *different* policy type
    is chosen; otherwise the same type is re-instantiated.  Re-instantiation
    ensures the child always starts from a clean (stateless) policy object.

    Parameters
    ----------
    policy:
        The parent agent's current policy instance.

    Returns
    -------
    A new policy object.
    """
    if random.random() < MUTATION_RATE:
        other_types = [p for p in POLICIES if p is not type(policy)]
        if other_types:
            return random.choice(other_types)()
    return type(policy)()


# ---------------------------------------------------------------------------
# Agent factory helper
# ---------------------------------------------------------------------------

def make_agents(n: int, grid_w: int, grid_h: int) -> list:
    """Spawn *n* agents at random grid positions with random policies.

    Each agent receives:

    * a random ``(x, y)`` position within ``[0, grid_w) × [0, grid_h)``
    * a randomly chosen policy from :data:`POLICIES`
    * starting energy of ``10.0``

    Parameters
    ----------
    n:
        Number of agents to create.
    grid_w:
        Grid width (columns).
    grid_h:
        Grid height (rows).

    Returns
    -------
    List of :class:`~src.core.agent.Agent` instances.
    """
    # Import here to avoid a circular import at module load time.
    from src.core.agent import Agent

    agents = []
    for i in range(n):
        x = random.randrange(grid_w)
        y = random.randrange(grid_h)
        policy = random.choice(POLICIES)()
        agent = Agent(id=i, x=x, y=y, policy=policy)
        agent.energy = 10.0
        agents.append(agent)
    return agents

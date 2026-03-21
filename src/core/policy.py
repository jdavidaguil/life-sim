"""Policy module: defines the TraitPolicy genome for agents."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.agent import Agent
    from src.core.grid import Grid

MUTATION_SIGMA: float = 0.05
BASELINE_MODE = "baseline"
RICHER_MODE = "richer"


class TraitPolicy:
    """Holds four continuous traits that govern agent movement decisions.

    Traits:
        resource_weight:   base attraction toward resource-rich cells.
        crowd_sensitivity: penalty weight for crowded cells.
        noise:             random exploration coefficient.
        energy_awareness:  scales extra resource attraction when energy is low.

    Supports two scoring modes:

    `baseline`:
        effective_rw = resource_weight + energy_awareness * (1 - energy / MAX_ENERGY)
        score = effective_rw * resource - crowd_sensitivity * crowding + noise * rng.random()

    `richer`:

        effective_rw = resource_weight + energy_awareness * (1 - energy / MAX_ENERGY)
        gradient     = resource(nx,ny) - resource(agent.x, agent.y)
        local_density = mean occupancy of the 8 cells around the agent
        relative_energy = (agent.energy - pop_mean_energy) / MAX_ENERGY

        score = effective_rw * resource
              - crowd_sensitivity * crowding          # immediate cell crowding
              + noise * rng.random()                  # exploration
              + resource_weight * gradient            # directional resource signal
              - crowd_sensitivity * local_density     # neighbourhood density penalty
              + energy_awareness * relative_energy    # relative-energy drive

    Parameters for decide():
        agent:           the calling Agent instance.
        grid:            Grid providing resource and neighbour queries.
        occupancy:       dict mapping (x, y) -> agent count.
        rng:             numpy random Generator for the noise term.
        pop_mean_energy: population mean energy this step (default 0.0);
                         used to compute the relative-energy term.
    """

    MAX_ENERGY: float = 20.0  # mirrors Agent.INITIAL_ENERGY

    def __init__(
        self,
        traits: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        mode: str = RICHER_MODE,
    ) -> None:
        if mode not in {BASELINE_MODE, RICHER_MODE}:
            raise ValueError(f"unknown policy mode: {mode}")
        self.mode = mode
        if traits is None:
            if rng is None:
                raise ValueError("rng required when traits is None")
            self.traits = np.array(
                [
                    rng.uniform(0.5, 1.5),
                    rng.uniform(0.0, 1.0),
                    rng.uniform(0.0, 0.5),
                    rng.uniform(0.0, 1.0),
                ],
                dtype=np.float32,
            )
        else:
            self.traits = traits

    @property
    def resource_weight(self) -> float:
        return float(self.traits[0])

    @property
    def crowd_sensitivity(self) -> float:
        return float(self.traits[1])

    @property
    def noise(self) -> float:
        return float(self.traits[2])

    @property
    def energy_awareness(self) -> float:
        return float(self.traits[3])

    def decide(
        self,
        agent: "Agent",
        grid: "Grid",
        occupancy: dict,
        rng: np.random.Generator,
        pop_mean_energy: float = 0.0,
    ) -> tuple[int, int]:
        neighbors = grid.get_neighbors(agent.x, agent.y)
        if not neighbors:
            return (0, 0)
        best_score = float("-inf")
        best_nx, best_ny = neighbors[0]
        for nx, ny in neighbors:
            resource = grid.get_resource(nx, ny)
            crowding = occupancy.get((nx, ny), 0)

            # Existing terms
            effective_rw = self.resource_weight + self.energy_awareness * (
                1.0 - agent.energy / self.MAX_ENERGY
            )
            effective_rw = max(0.0, effective_rw)

            if self.mode == BASELINE_MODE:
                score = (
                    effective_rw * resource
                    - self.crowd_sensitivity * crowding
                    + self.noise * float(rng.random())
                )
            else:
                gradient = resource - grid.get_resource(agent.x, agent.y)
                local_density = sum(
                    occupancy.get((agent.x + dx, agent.y + dy), 0)
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if not (dx == 0 and dy == 0)
                ) / 8.0
                relative_energy = (
                    agent.energy - pop_mean_energy
                ) / self.MAX_ENERGY

                score = (
                    effective_rw * resource
                    - self.crowd_sensitivity * crowding
                    + self.noise * float(rng.random())
                    + self.resource_weight * gradient
                    - self.crowd_sensitivity * local_density
                    + self.energy_awareness * relative_energy
                )

            if score > best_score:
                best_score = score
                best_nx, best_ny = nx, ny
        return (best_nx - agent.x, best_ny - agent.y)

    def mutate(self, rng: np.random.Generator) -> TraitPolicy:
        new_traits = self.traits + rng.normal(0, MUTATION_SIGMA, size=4).astype(np.float32)
        new_traits = np.maximum(0.0, new_traits)
        return TraitPolicy(traits=new_traits, mode=self.mode)

    def __repr__(self) -> str:
        return (
            f"TraitPolicy(rw={self.resource_weight:.2f}, "
            f"cs={self.crowd_sensitivity:.2f}, "
            f"noise={self.noise:.2f}, "
            f"ea={self.energy_awareness:.2f}, "
            f"mode={self.mode})"
        )

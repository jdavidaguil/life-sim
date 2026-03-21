"""Simulation module: orchestrates the grid and all agents."""

from __future__ import annotations

import collections
from typing import List, Optional

import numpy as np

from src.core.grid import Grid
from src.core.agent import Agent
from src.core.policy import GreedyPolicy, ExplorerPolicy, mutate

# Energy threshold above which an agent reproduces.
REPRODUCTION_THRESHOLD: float = 18.0

# Overcrowding: agents beyond this count on one cell each pay an extra decay.
OVERCROWD_THRESHOLD: int = 3
# Extra energy drained per agent above the threshold per step.
OVERCROWD_PENALTY: float = 0.15
# Probability of policy mutation on reproduction.
MUTATION_RATE: float = 0.05


class Simulation:
    """Top-level controller that owns the grid and drives each time step.

    Attributes:
        width: Grid column count.
        height: Grid row count.
        grid: The spatial structure agents inhabit.
        agents: Active agent roster.
        current_step: Number of steps completed so far.
        rng: Seeded NumPy random number generator.
        reproductions_total: Cumulative count of all reproduction events.
    """

    def __init__(
        self,
        width: int,
        height: int,
        initial_agents: int,
        seed: Optional[int] = None,
    ) -> None:
        """Set up the simulation and place agents at random grid positions.

        Args:
            width: Number of grid columns.
            height: Number of grid rows.
            initial_agents: Number of agents to spawn at random positions.
            seed: Optional integer seed for reproducible runs.
        """
        self.width = width
        self.height = height
        self.current_step: int = 0
        self.reproductions_total: int = 0
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.grid = Grid(width, height, rng=self.rng)
        self._next_id: int = initial_agents

        self.agents: List[Agent] = []
        for i in range(initial_agents):
            agent = Agent(
                id=i,
                x=int(self.rng.integers(0, width)),
                y=int(self.rng.integers(0, height)),
            )
            agent.policy = GreedyPolicy() if self.rng.random() < 0.7 else ExplorerPolicy()
            self.agents.append(agent)

        # Per-step population history: lists grow by one entry per step.
        self.history: dict[str, List] = {
            "step": [],
            "total": [],
            "greedy": [],
            "explorer": [],
            "greedy_births": [],
            "explorer_births": [],
            "greedy_avg_energy": [],
            "explorer_avg_energy": [],
        }

    def step(self) -> None:
        """Advance the simulation by one time step.

        The step runs in three phases to handle resource competition:

        1. **Move** – every agent scores its Moore neighbours by
           ``resource - crowding_penalty`` and moves to the best cell (85%),
           or to a random neighbour (15%) to avoid lock-in.
        2. **Compete** – agents sharing a cell pool their desired consumption
           amounts.  The cell supplies as much as it can; each agent receives
           a share proportional to what it wanted.
        3. **Update** – energy decay (``0.5 ± 0.1`` plus an overcrowding
           penalty for cells above ``OVERCROWD_THRESHOLD``) is applied, dead
           agents are dropped, and agents above ``REPRODUCTION_THRESHOLD``
           reproduce.
        """
        # ── Phase 1: move all agents, record desired consumption ──────────────
        # Pre-compute current occupancy so movement can avoid crowded cells.
        occupancy: dict[tuple[int, int], int] = collections.defaultdict(int)
        for agent in self.agents:
            occupancy[(agent.x, agent.y)] += 1

        desired: List[float] = []
        for agent in self.agents:
            neighbors = self.grid.get_neighbors(agent.x, agent.y)
            if neighbors:
                dx, dy = agent.policy.decide(agent, self.grid, occupancy, self.rng)
                agent.move(dx, dy)
                agent.last_dx, agent.last_dy = dx, dy
            desired.append(float(self.rng.uniform(0.5, 1.5)))

        # ── Phase 2: split resources per cell ────────────────────────────────
        # Group agent indices by their current position.
        cell_groups: dict[tuple[int, int], List[int]] = collections.defaultdict(list)
        for i, agent in enumerate(self.agents):
            cell_groups[(agent.x, agent.y)].append(i)

        gains: List[float] = [0.0] * len(self.agents)
        crowding_penalties: List[float] = [0.0] * len(self.agents)
        for (x, y), indices in cell_groups.items():
            total_desired = sum(desired[i] for i in indices)
            actual = self.grid.consume_resource(x, y, total_desired)
            # Distribute actual gain proportionally to each agent's desire.
            if total_desired > 0:
                for i in indices:
                    gains[i] = actual * (desired[i] / total_desired)
            # Overcrowding penalty: every agent above the threshold pays extra.
            excess = max(0, len(indices) - OVERCROWD_THRESHOLD)
            if excess > 0:
                for i in indices:
                    crowding_penalties[i] = excess * OVERCROWD_PENALTY

        # ── Phase 3: apply energy, reproduction, death ───────────────────────
        survivors: List[Agent] = []
        newborns: List[Agent] = []
        reproductions_this_step: int = 0
        greedy_births_step: int = 0
        explorer_births_step: int = 0

        for i, agent in enumerate(self.agents):
            agent.energy += gains[i]
            energy_decay = 0.5 + float(self.rng.uniform(-0.1, 0.1)) + crowding_penalties[i]
            agent.energy -= energy_decay

            if agent.energy <= 0:
                continue  # agent dies

            # Reproduction: split energy when above threshold.
            if agent.energy > REPRODUCTION_THRESHOLD:
                child_energy = agent.energy / 2.0
                agent.energy = child_energy
                child = Agent(id=self._next_id, x=agent.x, y=agent.y)
                child.energy = child_energy
                # Mutate policy type on reproduction.
                child.policy = mutate(agent.policy, self.rng)
                # Track births per policy of parent.
                if isinstance(agent.policy, GreedyPolicy):
                    greedy_births_step += 1
                else:
                    explorer_births_step += 1
                self._next_id += 1
                newborns.append(child)
                reproductions_this_step += 1

            survivors.append(agent)

        self.agents = survivors + newborns
        self.reproductions_total += reproductions_this_step

        # Build a pressure map from post-move occupancy so heavily grazed cells
        # regenerate more slowly (rate / (1 + num_agents)).
        pressure = np.zeros((self.height, self.width), dtype=np.float32)
        for (x, y), indices in cell_groups.items():
            pressure[y, x] = len(indices)
        self.grid.regenerate(pressure)
        # Random resource shocks: sudden local windfalls or depletions.
        self.grid.apply_noise()
        # Advance hotspot random walk every step for continuous slow drift.
        self.grid.update_hotspots()
        self.current_step += 1

        # ── Record per-policy counts ──────────────────────────────────────────
        greedy_agents  = [a for a in self.agents if isinstance(a.policy, GreedyPolicy)]
        explorer_agents = [a for a in self.agents if isinstance(a.policy, ExplorerPolicy)]
        greedy_n   = len(greedy_agents)
        explorer_n = len(explorer_agents)
        greedy_avg   = sum(a.energy for a in greedy_agents)   / greedy_n   if greedy_n   else 0.0
        explorer_avg = sum(a.energy for a in explorer_agents) / explorer_n if explorer_n else 0.0
        self.history["step"].append(self.current_step)
        self.history["total"].append(self.agent_count())
        self.history["greedy"].append(greedy_n)
        self.history["explorer"].append(explorer_n)
        self.history["greedy_births"].append(greedy_births_step)
        self.history["explorer_births"].append(explorer_births_step)
        self.history["greedy_avg_energy"].append(greedy_avg)
        self.history["explorer_avg_energy"].append(explorer_avg)

        if self.current_step % 10 == 0:
            print(
                f"[step {self.current_step:4d}] "
                f"population={self.agent_count():4d}  "
                f"births_this_step={reproductions_this_step:3d}  "
                f"total_births={self.reproductions_total}"
            )

        if self.current_step % 20 == 0:
            print(
                f"  policies: greedy={greedy_n:4d} ({100*greedy_n//max(1,self.agent_count()):2d}%)  "
                f"explorer={explorer_n:4d} ({100*explorer_n//max(1,self.agent_count()):2d}%)"
            )
            print(
                f"  avg energy: greedy={greedy_avg:5.2f}  explorer={explorer_avg:5.2f}"
            )
            print(
                f"  births:     greedy={greedy_births_step:3d}  "
                f"explorer={explorer_births_step:3d}"
            )

    def plot_history(self, block: bool = True) -> None:  # pragma: no cover
        """Plot population counts, avg energy, and births per policy over time.

        Opens a standalone Matplotlib figure with three sub-panels:

        1. **Population** — total, greedy, and explorer agent counts.
        2. **Avg energy** — mean energy per policy, showing which strategy
           is better fed at each point in time.
        3. **Births per step** — reproduction events from each policy,
           revealing which strategy is expanding faster.

        Args:
            block: Whether ``plt.show()`` blocks until the window is closed.
        """
        import matplotlib.pyplot as plt

        steps = self.history["step"]
        if not steps:
            print("No history recorded yet.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.patch.set_facecolor("#111111")

        def _style(ax: plt.Axes, ylabel: str, title: str) -> None:
            ax.set_facecolor("#111111")
            ax.set_ylabel(ylabel, color="#aaaaaa")
            ax.set_title(title, color="#dddddd", fontsize=10)
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.legend(facecolor="#222222", edgecolor="#555555", labelcolor="#dddddd",
                      fontsize=8)

        # Panel 1: population counts
        ax0 = axes[0]
        ax0.plot(steps, self.history["total"],    color="#ffffff", lw=1.5, label="Total")
        ax0.plot(steps, self.history["greedy"],   color="#ff4444", lw=1.2, label="Greedy")
        ax0.plot(steps, self.history["explorer"], color="#4488ff", lw=1.2, label="Explorer")
        _style(ax0, "Agents", "Population per policy")

        # Panel 2: average energy per policy
        ax1 = axes[1]
        ax1.plot(steps, self.history["greedy_avg_energy"],
                 color="#ff4444", lw=1.2, label="Greedy avg E")
        ax1.plot(steps, self.history["explorer_avg_energy"],
                 color="#4488ff", lw=1.2, label="Explorer avg E")
        _style(ax1, "Avg energy", "Average energy per policy")

        # Panel 3: births per step per policy
        ax2 = axes[2]
        ax2.plot(steps, self.history["greedy_births"],
                 color="#ff4444", lw=1.0, alpha=0.8, label="Greedy births")
        ax2.plot(steps, self.history["explorer_births"],
                 color="#4488ff", lw=1.0, alpha=0.8, label="Explorer births")
        _style(ax2, "Births / step", "Births per step per policy")
        ax2.set_xlabel("Step", color="#aaaaaa")

        fig.tight_layout()
        plt.show(block=block)

    def add_agent(self, agent: Agent) -> None:
        """Register a new agent mid-simulation.

        Args:
            agent: The agent to add.
        """
        self.agents.append(agent)

    def agent_count(self) -> int:
        """Return the number of agents in the simulation."""
        return len(self.agents)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Simulation(width={self.width}, height={self.height}, "
            f"step={self.current_step}, agents={self.agent_count()})"
        )

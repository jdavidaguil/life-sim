"""Simulation module: orchestrates the grid and all agents."""

from __future__ import annotations

import collections
from typing import List, Optional

import numpy as np

from src.core.grid import Grid
from src.core.agent import Agent
from src.core.policy import TraitPolicy, NeuralPolicy, StatefulNeuralPolicy

# Energy threshold above which an agent reproduces.
REPRODUCTION_THRESHOLD: float = 18.0

# Overcrowding: agents beyond this count on one cell each pay an extra decay.
OVERCROWD_THRESHOLD: int = 3
# Extra energy drained per agent above the threshold per step.
OVERCROWD_PENALTY: float = 0.15


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
        env_config: dict | None = None,
        policy_mode: str = "baseline",
    ) -> None:
        """Set up the simulation and place agents at random grid positions.

        Args:
            width: Number of grid columns.
            height: Number of grid rows.
            initial_agents: Number of agents to spawn at random positions.
            seed: Optional integer seed for reproducible runs.
            env_config: Optional dict of environment overrides (drift_step,
                noise_rate, noise_magnitude).
            policy_mode: Movement scorer to use for all agents (`baseline`, `richer`, or `neural`).
        """
        self.width = width
        self.height = height
        self.current_step: int = 0
        self.reproductions_total: int = 0
        self.rng: np.random.Generator = np.random.default_rng(seed)
        env_config = env_config or {}
        self.grid = Grid(
            width, height, rng=self.rng,
            drift_step=env_config.get("drift_step"),
            noise_rate=env_config.get("noise_rate"),
            noise_magnitude=env_config.get("noise_magnitude"),
        )
        self._next_id: int = initial_agents

        self.agents: List[Agent] = []
        for i in range(initial_agents):
            agent = Agent(
                id=i,
                x=int(self.rng.integers(0, width)),
                y=int(self.rng.integers(0, height)),
                policy=(
                    NeuralPolicy(rng=self.rng)
                    if policy_mode == "neural"
                    else StatefulNeuralPolicy(rng=self.rng)
                    if policy_mode == "stateful"
                    else TraitPolicy(rng=self.rng, mode=policy_mode)
                ),
            )
            self.agents.append(agent)

        # Per-step population history: lists grow by one entry per step.
        self.history: dict[str, List] = {
            "step": [],
            "total": [],
            "mean_resource_weight": [],
            "mean_crowd_sensitivity": [],
            "mean_noise": [],
            "mean_energy_awareness": [],
            "std_resource_weight": [],
            "std_crowd_sensitivity": [],
            "std_noise": [],
            "std_energy_awareness": [],
            "mean_genome_norm": [],
            "std_genome_norm": [],
            "mean_state_norm": [],
            "std_state_norm": [],
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

        # Compute population mean energy once for richer perception.
        pop_mean_energy: float = (
            sum(a.energy for a in self.agents) / len(self.agents)
            if self.agents else 0.0
        )

        desired: List[float] = []
        for agent in self.agents:
            neighbors = self.grid.get_neighbors(agent.x, agent.y)
            if neighbors:
                dx, dy = agent.policy.decide(
                    agent, self.grid, occupancy, self.rng,
                    pop_mean_energy=pop_mean_energy,
                )
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
                child = Agent(id=self._next_id, x=agent.x, y=agent.y,
                             policy=agent.policy.mutate(self.rng))
                child.energy = child_energy
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

        self.history["step"].append(self.current_step)
        self.history["total"].append(self.agent_count())
        trait_agents = [a for a in self.agents
                        if isinstance(a.policy, TraitPolicy)]
        neural_agents = [a for a in self.agents
                         if isinstance(a.policy,
                            (NeuralPolicy, StatefulNeuralPolicy))]
        stateful_agents = [a for a in self.agents
                           if isinstance(a.policy, StatefulNeuralPolicy)]
        if trait_agents:
            traits = np.array(
                [a.policy.traits for a in trait_agents],
                dtype=np.float32,
            )
            self.history["mean_resource_weight"].append(float(traits[:, 0].mean()))
            self.history["mean_crowd_sensitivity"].append(float(traits[:, 1].mean()))
            self.history["mean_noise"].append(float(traits[:, 2].mean()))
            self.history["mean_energy_awareness"].append(float(traits[:, 3].mean()))
            self.history["std_resource_weight"].append(float(traits[:, 0].std()))
            self.history["std_crowd_sensitivity"].append(float(traits[:, 1].std()))
            self.history["std_noise"].append(float(traits[:, 2].std()))
            self.history["std_energy_awareness"].append(float(traits[:, 3].std()))
        else:
            for key in ["mean_resource_weight", "mean_crowd_sensitivity",
                        "mean_noise", "mean_energy_awareness",
                        "std_resource_weight", "std_crowd_sensitivity",
                        "std_noise", "std_energy_awareness"]:
                self.history[key].append(0.0)

        if neural_agents:
            genome_norms = np.array(
                [float(np.linalg.norm(a.policy.genome)) for a in neural_agents],
                dtype=np.float32,
            )
            self.history["mean_genome_norm"].append(float(genome_norms.mean()))
            self.history["std_genome_norm"].append(float(genome_norms.std()))
        else:
            self.history["mean_genome_norm"].append(0.0)
            self.history["std_genome_norm"].append(0.0)

        if stateful_agents:
            state_norms = np.array(
                [float(np.linalg.norm(a.policy.state)) for a in stateful_agents],
                dtype=np.float32,
            )
            self.history["mean_state_norm"].append(float(state_norms.mean()))
            self.history["std_state_norm"].append(float(state_norms.std()))
        else:
            self.history["mean_state_norm"].append(0.0)
            self.history["std_state_norm"].append(0.0)

        if self.current_step % 10 == 0:
            print(
                f"[step {self.current_step:4d}] "
                f"population={self.agent_count():4d}  "
                f"births_this_step={reproductions_this_step:3d}  "
                f"total_births={self.reproductions_total}"
            )

        if self.current_step % 20 == 0:
            if trait_agents:
                traits = np.array([a.policy.traits for a in trait_agents])
                print(
                    f"  traits (mean): "
                    f"rw={traits[:,0].mean():.2f}  "
                    f"cs={traits[:,1].mean():.2f}  "
                    f"noise={traits[:,2].mean():.2f}  "
                    f"ea={traits[:,3].mean():.2f}"
                )
            else:
                neural_agents = [a for a in self.agents
                                 if isinstance(a.policy,
                                    (NeuralPolicy, StatefulNeuralPolicy))]
                if neural_agents:
                    norms = [float(np.linalg.norm(a.policy.genome))
                             for a in neural_agents]
                    print(
                        f"  neural genome: "
                        f"mean_norm={float(np.mean(norms)):.2f}  "
                        f"std_norm={float(np.std(norms)):.2f}"
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

        def _style(ax, ylabel: str, title: str) -> None:
            ax.set_facecolor("#111111")
            ax.set_ylabel(ylabel, color="#aaaaaa")
            ax.set_title(title, color="#dddddd", fontsize=10)
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.legend(facecolor="#222222", edgecolor="#555555", labelcolor="#dddddd",
                      fontsize=8)

        # Panel 1: total population
        ax0 = axes[0]
        ax0.plot(steps, self.history["total"], color="#ffffff", lw=1.5, label="Total")
        _style(ax0, "Agents", "Population")

        # Panel 2: mean of each trait over time
        ax1 = axes[1]
        if self.history["mean_genome_norm"][-1] > 0:
            ax1.plot(
                steps,
                self.history["mean_genome_norm"],
                color="#c77dca",
                lw=1.5,
                label="genome norm",
            )
            _style(ax1, "Mean norm", "Neural genome norm over time")
        else:
            ax1.plot(steps, self.history["mean_resource_weight"],   color="#ff4444", lw=1.2, label="rw")
            ax1.plot(steps, self.history["mean_crowd_sensitivity"], color="#4488ff", lw=1.2, label="cs")
            ax1.plot(steps, self.history["mean_noise"],             color="#44cc44", lw=1.2, label="noise")
            ax1.plot(steps, self.history["mean_energy_awareness"],  color="#ffaa00", lw=1.2, label="ea")
            _style(ax1, "Mean trait", "Mean trait values over time")

        # Panel 3: std of each trait over time
        ax2 = axes[2]
        if self.history["mean_state_norm"][-1] > 0:
            ax2.plot(
                steps,
                self.history["mean_state_norm"],
                color="#4ecdc4",
                lw=1.5,
                label="state norm",
            )
            ax2.plot(
                steps,
                self.history["std_state_norm"],
                color="#ffe66d",
                lw=1.0,
                alpha=0.9,
                label="state std",
            )
            _style(ax2, "State", "Internal state dynamics over time")
        elif self.history["mean_genome_norm"][-1] > 0:
            ax2.plot(
                steps,
                self.history["std_genome_norm"],
                color="#c77dca",
                lw=1.2,
                alpha=0.9,
                label="genome std",
            )
            _style(ax2, "Std norm", "Genome diversity over time")
        else:
            ax2.plot(steps, self.history["std_resource_weight"],   color="#ff4444", lw=1.0, alpha=0.8, label="rw")
            ax2.plot(steps, self.history["std_crowd_sensitivity"], color="#4488ff", lw=1.0, alpha=0.8, label="cs")
            ax2.plot(steps, self.history["std_noise"],             color="#44cc44", lw=1.0, alpha=0.8, label="noise")
            ax2.plot(steps, self.history["std_energy_awareness"],  color="#ffaa00", lw=1.0, alpha=0.8, label="ea")
            _style(ax2, "Std trait", "Trait diversity (std) over time")
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

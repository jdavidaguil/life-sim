"""Simulation module: orchestrates the grid and all agents."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.core.grid import Grid
from src.core.agent import Agent
from src.core.policy import TraitPolicy, NeuralPolicy, StatefulNeuralPolicy
from src.core.state import SimState
from src.core.loop import SimulationLoop
from src.core.phases import DEFAULT_PHASES


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
        phases: list | None = None,
    ) -> None:
        """Set up the simulation and place agents at random grid positions.

        Args:
            width: Number of grid columns.
            height: Number of grid rows.
            initial_agents: Number of agents to spawn at random positions.
            seed: Optional integer seed for reproducible runs.
            env_config: Optional dict of environment overrides (drift_step,
                noise_rate, noise_magnitude).
            policy_mode: Movement scorer to use for all agents (`baseline`, `richer`, `neural`, or `stateful`).
            phases: Optional custom phase list.  Defaults to ``DEFAULT_PHASES``.
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
        self._state: SimState
        self._loop: SimulationLoop
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

        self._state = SimState(
            grid=self.grid,
            agents=self.agents,
            rng=self.rng,
            step=0,
            metrics={"_next_id": self._next_id},
        )
        self._loop = SimulationLoop(phases if phases is not None else DEFAULT_PHASES)

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
        """Advance the simulation by one time step via the phase-based loop.

        Delegates to :class:`~src.core.loop.SimulationLoop`, which executes
        the configured phase list in order.  After the loop returns, this
        method syncs ``current_step``, ``agents``, and ``reproductions_total``
        from the shared :class:`~src.core.state.SimState`, then records history.
        """
        self._loop.run(self._state, steps=1)
        self.current_step = self._state.step
        self.agents = self._state.agents
        reproductions_this_step: int = self._state.scratch.get(
            "reproductions_this_step",
            self._state.scratch.get("mating_events", 0),
        )
        self.reproductions_total += reproductions_this_step

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
                if neural_agents:
                    norms = [float(np.linalg.norm(a.policy.genome))
                             for a in neural_agents]
                    print(
                        f"  neural genome: "
                        f"mean_norm={float(np.mean(norms)):.2f}  "
                        f"std_norm={float(np.std(norms)):.2f}"
                    )

    def plot_history(self, block: bool = True) -> None:  # pragma: no cover
        """Plot population and genome/trait history over time.

        Opens a dark-themed Matplotlib figure with three sub-panels:

        1. **Population** — total agent count per step.
        2. **Mean** — mean genome norm (neural) or mean trait values (trait).
        3. **Diversity** — std of genome norm or traits; state norm if stateful.

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

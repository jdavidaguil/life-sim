"""Experiment configuration layer for the phase-based simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from src.core.grid import Grid
from src.core.agent import Agent
from src.core.policy import TraitPolicy
from src.core.state import SimState
from src.core.loop import SimulationLoop
from src.core.phases import DEFAULT_PHASES


@dataclass
class Experiment:
    """Configuration for a reproducible simulation experiment.

    Attributes:
        phases: Base ordered phase list.  Defaults to ``DEFAULT_PHASES``.
        overrides: Maps a phase ``__name__`` to a replacement callable.
            e.g. ``{"reproduce": sexual_reproduction}``.
        additions: Maps ``"before_<phase>"`` or ``"after_<phase>"`` to a list
            of callables that are injected immediately before or after the
            named phase.
        env_config: Environment parameters forwarded to ``Grid.__init__``
            (``drift_step``, ``noise_rate``, ``noise_magnitude``).
        steps: Number of simulation steps to run per seed.
        seeds: List of integer seeds; one ``SimState`` is produced per seed.
    """

    phases: List[Callable] = field(default_factory=lambda: list(DEFAULT_PHASES))
    overrides: dict[str, Callable] = field(default_factory=dict)
    additions: dict[str, List[Callable]] = field(default_factory=dict)
    env_config: dict = field(default_factory=dict)
    steps: int = 1000
    seeds: List[int] = field(default_factory=lambda: [42])

    def build_phase_list(self) -> List[Callable]:
        """Return the final ordered phase list after applying overrides and additions.

        Processing order:
        1. Start with ``self.phases``.
        2. Replace any phase whose ``__name__`` matches a key in ``self.overrides``.
        3. Inject ``additions["before_<name>"]`` callables immediately before,
           and ``additions["after_<name>"]`` callables immediately after, each
           named phase.

        Returns:
            New list of callables in execution order.
        """
        # Step 1 + 2: apply overrides
        result: List[Callable] = [
            self.overrides.get(p.__name__, p) for p in self.phases
        ]

        # Step 3: apply before/after injections (iterate over a snapshot)
        final: List[Callable] = []
        for phase in result:
            name = phase.__name__
            final.extend(self.additions.get(f"before_{name}", []))
            final.append(phase)
            final.extend(self.additions.get(f"after_{name}", []))

        return final

    def run(self) -> List[SimState]:
        """Run the experiment for every seed and return one SimState per seed.

        For each seed:
        1. Constructs an ``np.random.Generator`` from the seed.
        2. Builds a ``Grid`` using ``self.env_config``.
        3. Spawns 100 agents at random positions with a default ``TraitPolicy``.
        4. Constructs a ``SimState`` with the grid, agents, and generator.
        5. Runs ``SimulationLoop(build_phase_list())`` for ``self.steps`` steps.

        Returns:
            A list of ``SimState`` instances, one per entry in ``self.seeds``.
        """
        results: List[SimState] = []
        phase_list = self.build_phase_list()

        for seed in self.seeds:
            rng = np.random.default_rng(seed)

            grid = Grid(
                width=50,
                height=50,
                rng=rng,
                drift_step=self.env_config.get("drift_step"),
                noise_rate=self.env_config.get("noise_rate"),
                noise_magnitude=self.env_config.get("noise_magnitude"),
            )

            initial_agents = 100
            agents: List[Agent] = [
                Agent(
                    id=i,
                    x=int(rng.integers(0, 50)),
                    y=int(rng.integers(0, 50)),
                    policy=TraitPolicy(rng=rng),
                )
                for i in range(initial_agents)
            ]

            state = SimState(
                grid=grid,
                agents=agents,
                rng=rng,
                step=0,
                metrics={"_next_id": initial_agents},
            )

            loop = SimulationLoop(phase_list)
            loop.run(state, self.steps)

            results.append(state)

        return results

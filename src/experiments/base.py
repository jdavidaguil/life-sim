"""Experiment configuration layer for the phase-based simulation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List

import numpy as np

from src.core.grid import Grid
from src.core.agent import Agent
from src.core.state import SimState
from src.core.loop import SimulationLoop
from src.core.phases import DEFAULT_PHASES


def _safe_serialize(obj):
    """Recursively make *obj* JSON-serializable; non-serializable leaves become None."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str, type(None))):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    return None


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
        on_step: Optional callback invoked with the current ``SimState`` after
            each simulation step.
        result_id: Identifier prefix for saved result files.  Leave empty to
            disable saving.
        save_results: When ``True`` (default) and ``result_id`` is non-empty,
            persist a JSON snapshot to ``results/`` after each seed completes.
    """

    name: str = ""
    description: str = ""
    phases: List[Callable] = field(default_factory=lambda: list(DEFAULT_PHASES))
    overrides: dict[str, Callable] = field(default_factory=dict)
    additions: dict[str, List[Callable]] = field(default_factory=dict)
    env_config: dict = field(default_factory=dict)
    steps: int = 1000
    seeds: List[int] = field(default_factory=lambda: [42])
    on_step: Callable[[SimState], None] | None = None
    result_id: str = ""
    save_results: bool = True
    policy_mode: str = "trait"
    grid_config: dict = field(default_factory=dict)

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

    def _make_policy(self, rng):
        """Instantiate the agent policy determined by ``self.policy_mode``."""
        if self.policy_mode == "trait":
            from src.core.policy import TraitPolicy
            return TraitPolicy(rng=rng)
        elif self.policy_mode == "richer":
            from src.core.policy import TraitPolicy
            return TraitPolicy(rng=rng, mode="richer")
        elif self.policy_mode == "neural":
            from src.core.policy import NeuralPolicy
            return NeuralPolicy(rng=rng)
        elif self.policy_mode == "neural_warm":
            from src.core.policy import NeuralPolicy
            return NeuralPolicy(rng=rng, warm_start=True)
        elif self.policy_mode == "neural_noisy_warm":
            from src.genome.crossover_neural import warm_start_noisy
            return warm_start_noisy(rng, sigma=0.1)
        elif self.policy_mode == "stateful_warm":
            from src.core.policy import StatefulNeuralPolicy
            return StatefulNeuralPolicy(rng=rng, warm_start=True)
        else:
            raise ValueError(f"Unknown policy_mode: {self.policy_mode}")

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

            # Apply grid_config class-attribute overrides temporarily
            originals = {}
            for key, val in self.grid_config.items():
                originals[key] = getattr(Grid, key)
                setattr(Grid, key, val)

            try:
                grid = Grid(
                    width=50,
                    height=50,
                    rng=rng,
                    drift_step=self.env_config.get("drift_step"),
                    noise_rate=self.env_config.get("noise_rate"),
                    noise_magnitude=self.env_config.get("noise_magnitude"),
                    hotspot_sigma=self.env_config.get("hotspot_sigma"),
                    num_hotspots=self.env_config.get("num_hotspots"),
                )
            finally:
                for key, val in originals.items():
                    setattr(Grid, key, val)

            initial_agents = 100
            agents: List[Agent] = [
                Agent(
                    id=i,
                    x=int(rng.integers(0, 50)),
                    y=int(rng.integers(0, 50)),
                    policy=self._make_policy(rng),
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
            for _ in range(self.steps):
                loop.run(state, 1)
                if self.on_step is not None:
                    self.on_step(state)

            results.append(state)

            if self.save_results and self.result_id:
                os.makedirs("results", exist_ok=True)
                now = datetime.now(timezone.utc)
                fname = (
                    f"results/{self.result_id}_{seed}_"
                    f"{now.strftime('%Y%m%dT%H%M%SZ')}.json"
                )
                payload = {
                    "result_id": self.result_id,
                    "seed": seed,
                    "steps": self.steps,
                    "env_config": self.env_config,
                    "timestamp": now.isoformat(),
                    "final_population": len(state.agents),
                    "metrics": _safe_serialize(state.metrics),
                }
                with open(fname, "w") as fh:
                    json.dump(payload, fh)

        return results

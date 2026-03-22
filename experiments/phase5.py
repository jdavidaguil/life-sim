"""Phase 5 experiment: internal state and memory.

Adds a persistent state vector to the neural genome.
Uses warm-start initialization from Phase 4 solution.
Compares against Phase 4 warm-start baseline.

Usage:
    python -m experiments.phase5
    python -m experiments.phase5 --steps 1000 --seed 42
"""

from __future__ import annotations

import argparse
import numpy as np

from src.core.simulation import Simulation
from src.core.policy import StatefulNeuralPolicy
from src.core.grid import Grid
from experiments.probe_phase4 import (
    PROBE_SITUATIONS, probe_population,
    plot_probe_results, DIR_LABELS,
)
from experiments.phase2 import CONDITIONS


PHASE5_HOTSPOT_SIGMA: float = 3.0
PHASE5_NUM_HOTSPOTS: int = 6


def build_phase5_env_config(cond) -> dict:
    """Return the environment overrides used by the Phase 5 experiment."""
    return {
        "drift_step": cond.drift_step,
        "noise_rate": 0.0,
        "noise_magnitude": 0.0,
    }


def warm_start_phase5_population(
    sim: Simulation,
    seed: int | None,
) -> None:
    """Replace the default policies with Phase 5 warm-started stateful ones."""
    rng_ws = np.random.default_rng(seed) if seed is not None else sim.rng
    for agent in sim.agents:
        agent.policy = StatefulNeuralPolicy(
            rng=rng_ws,
            warm_start=True,
        )


def create_phase5_simulation(
    seed: int | None,
    condition,
    width: int = 50,
    height: int = 50,
    initial_agents: int = 100,
) -> Simulation:
    """Create a simulation configured exactly like the Phase 5 experiment."""
    original_sigma = Grid.HOTSPOT_SIGMA
    original_n = Grid.NUM_HOTSPOTS
    Grid.HOTSPOT_SIGMA = PHASE5_HOTSPOT_SIGMA
    Grid.NUM_HOTSPOTS = PHASE5_NUM_HOTSPOTS

    try:
        sim = Simulation(
            width=width,
            height=height,
            initial_agents=initial_agents,
            seed=seed,
            env_config=build_phase5_env_config(condition),
            policy_mode="stateful",
        )
    finally:
        Grid.HOTSPOT_SIGMA = original_sigma
        Grid.NUM_HOTSPOTS = original_n

    sim.grid.HOTSPOT_SIGMA = PHASE5_HOTSPOT_SIGMA
    sim.grid.NUM_HOTSPOTS = PHASE5_NUM_HOTSPOTS
    warm_start_phase5_population(sim, seed=seed)
    return sim


def run_phase5(
    seed: int,
    steps: int,
    condition_key: str = "A",
) -> tuple:
    """Run Phase 5 and return (sim, condition)."""
    cond_map = {c.name[0]: c for c in CONDITIONS}
    cond = cond_map[condition_key]

    sim = create_phase5_simulation(seed=seed, condition=cond)
    for _ in range(steps):
        sim.step()

    return sim, cond


def probe_and_report(sim, cond_name: str) -> None:
    """Probe network and print results."""
    print(f"\nFinal population: {sim.agent_count()}")

    # Genome norm
    norms = [float(np.linalg.norm(a.policy.genome))
             for a in sim.agents
             if isinstance(a.policy, StatefulNeuralPolicy)]
    print(f"Genome norm: mean={np.mean(norms):.3f} "
          f"std={np.std(norms):.3f}")

    # State vector diversity
    states = np.array([
        a.policy.state for a in sim.agents
        if isinstance(a.policy, StatefulNeuralPolicy)
    ])
    print(f"State vector mean: {np.round(states.mean(axis=0), 3)}")
    print(f"State vector std:  {np.round(states.std(axis=0), 3)}")

    # Probe
    rng = np.random.default_rng(42)
    print(f"\nProbing {min(50, sim.agent_count())} agents...")
    all_results = {cond_name: []}
    for situation in PROBE_SITUATIONS:
        probs = probe_population(
            sim.agents, situation,
            n_sample=50, rng=rng,
        )
        all_results[cond_name].append(probs)
        top_dir = DIR_LABELS[np.argmax(probs)]
        print(
            f"  [{situation['name'].replace(chr(10),' ')}] "
            f"top: {top_dir} ({probs.max():.3f})  "
            f"spread: {probs.max()-probs.min():.3f}"
        )

    plot_probe_results(all_results, [cond_name])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: internal state experiment."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps to run (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--condition", type=str, default="A",
        choices=["A", "B", "C", "D", "E"],
        help="Environment condition (default: A)"
    )
    args = parser.parse_args()

    print(f"Phase 5 — Internal State")
    print(f"Seed: {args.seed}  Steps: {args.steps}  "
          f"Condition: {args.condition}")
    print(
        f"HOTSPOT_SIGMA={PHASE5_HOTSPOT_SIGMA}  "
        f"NUM_HOTSPOTS={PHASE5_NUM_HOTSPOTS}  noise=0"
    )
    print(f"Warm-start: True\n")

    sim, cond = run_phase5(
        seed=args.seed,
        steps=args.steps,
        condition_key=args.condition,
    )

    probe_and_report(sim, cond.name)


if __name__ == "__main__":
    main()
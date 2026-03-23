"""Phase 4 steep landscape experiment.

Runs Phase 4 neural genome in Condition E (steep resource 
landscape) to test whether selection pressure for directional 
sensitivity emerges when resource gradients are strong.

Usage:
    python -m experiments.phase4_neural_steep
    python -m experiments.phase4_neural_steep --steps 1000 --seed 42
"""

from __future__ import annotations

import argparse
import numpy as np

from src.core.simulation import Simulation
from src.core.policy import NeuralPolicy
from src.core.state import SimState
from src.experiments.base import Experiment
from src.experiments.phase2_trait_volatility import CONDITIONS
from src.experiments.probe_phase4 import (
    PROBE_SITUATIONS,
    probe_population,
    plot_probe_results,
)


def _init_neural_steep(state: SimState) -> None:
    """Replace agents with NeuralPolicy at step 0."""
    if state.step != 0:
        return
    for agent in state.agents:
        agent.policy = NeuralPolicy(rng=state.rng)


EXPERIMENT = Experiment(
    additions={"before_move": [_init_neural_steep]},
    env_config={
        "drift_step": 1,
        "noise_rate": 0.0,
        "noise_magnitude": 0.0,
        "hotspot_sigma": 3.0,
        "num_hotspots": 6,
    },
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
    result_id="phase4_steep",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 steep landscape probe."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps to run (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Find condition E
    cond_map = {c.name[0]: c for c in CONDITIONS}
    cond = cond_map["E"]

    print(f"Running {cond.name}")
    print(f"  drift_step={cond.drift_step}  "
          f"noise_rate={cond.noise_rate}  "
          f"noise_magnitude={cond.noise_magnitude}")
    print(f"  HOTSPOT_SIGMA=3.0  NUM_HOTSPOTS=6  noise=0")
    print(f"  Steps: {args.steps}  Seed: {args.seed}\n")

    from src.core.grid import Grid
    original_sigma = Grid.HOTSPOT_SIGMA
    Grid.HOTSPOT_SIGMA = 3.0
    Grid.NUM_HOTSPOTS = 6

    try:
        sim = Simulation(
            width=50, height=50,
            initial_agents=100,
            seed=args.seed,
            env_config={
                "drift_step":      1,
                "noise_rate":      0.0,
                "noise_magnitude": 0.0,
            },
            policy_mode="neural",
        )
        for _ in range(args.steps):
            sim.step()

        print(f"\nFinal population: {sim.agent_count()}")

        # Quick weight check
        agent = sim.agents[0]
        x = np.zeros(18, dtype=np.float32)
        x[1] = 1.0  # max resource to north
        probs = agent.policy._forward(x)
        print(f"North probability (max resource north): {probs[1]:.4f}")
        print(f"Max weight in genome: "
              f"{np.abs(agent.policy.genome).max():.4f}")
        print(f"Genome norm: "
              f"{np.linalg.norm(agent.policy.genome):.3f}")

        # Full probe
        rng = np.random.default_rng(args.seed)
        all_results = {}
        print(f"\nProbing {min(50, sim.agent_count())} agents...")
        condition_results = []
        for situation in PROBE_SITUATIONS:
            probs = probe_population(
                sim.agents, situation,
                n_sample=50, rng=rng,
            )
            condition_results.append(probs)
            from src.experiments.probe_phase4 import DIR_LABELS
            top_dir = DIR_LABELS[np.argmax(probs)]
            print(f"  [{situation['name'].replace(chr(10),' ')}] "
                  f"top: {top_dir} ({probs.max():.3f})  "
                  f"min: {probs.min():.3f}  "
                  f"spread: {probs.max()-probs.min():.3f}")

        all_results[cond.name] = condition_results
        plot_probe_results(all_results, [cond.name])

    finally:
        Grid.HOTSPOT_SIGMA = original_sigma
        Grid.NUM_HOTSPOTS = 4


if __name__ == "__main__":
    main()


EXPERIMENT = Experiment(
    name="Neural Cold Start — Steep Landscape",
    description="Condition E: tight hotspots, no noise. Tests if strong gradients rescue cold-start evolution.",
    result_id="phase4_neural_steep",
    policy_mode="neural",
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
    save_results=True,
    env_config={"drift_step": 1, "noise_rate": 0.0, "noise_magnitude": 0.0},
    grid_config={"HOTSPOT_SIGMA": 3.0, "NUM_HOTSPOTS": 6},
)

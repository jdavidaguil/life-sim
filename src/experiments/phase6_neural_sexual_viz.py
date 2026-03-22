"""Run the Phase 6 neural-sexual reproduction simulation with the desktop renderer.

All agents start with warm-start + noisy NeuralPolicy genomes.
Reproduction is via whole-layer genome crossover (reproduce_sexual_neural).

Usage
-----
    python -m src.experiments.phase6_neural_sexual_viz
    python -m src.experiments.phase6_neural_sexual_viz --condition D --steps 2000
    python -m src.experiments.phase6_neural_sexual_viz --condition B --seed 7

Conditions
----------
    A — Baseline          (drift_step=1, noise_rate=3.0, noise_magnitude=2.0)
    B — Fast drift        (drift_step=3, noise_rate=3.0, noise_magnitude=2.0)
    C — Boom/bust         (drift_step=1, noise_rate=8.0, noise_magnitude=4.0)
    D — Combined          (drift_step=3, noise_rate=8.0, noise_magnitude=4.0)

Keyboard shortcuts (same as base renderer):
    p — pause / resume
    q — quit
    e — toggle agent panel between genome-norm colours and energy heatmap
"""

from __future__ import annotations

import argparse

from experiments.phase2 import CONDITIONS
from src.experiments.phase6_neural_sexual import (
    _build_phase_list,
    _replace_policies_warm_start,
    MATING_COST,
    MATING_THRESHOLD,
)
from src.core.simulation import Simulation
from src.viz.renderer import Renderer

# Map single-letter label → CONDITIONS index
_CONDITION_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live visualisation of the Phase 6 neural-sexual experiment."
    )
    parser.add_argument(
        "--condition", choices=list(_CONDITION_MAP), default="A",
        help="Environment condition A/B/C/D (default: A)",
    )
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps to run (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds between rendered frames (default: 0.05)")
    args = parser.parse_args()

    cond = CONDITIONS[_CONDITION_MAP[args.condition]]
    env_cfg = {
        "drift_step": cond.drift_step,
        "noise_rate": cond.noise_rate,
        "noise_magnitude": cond.noise_magnitude,
    }
    label = (
        f"Neural-sexual  |  cond={args.condition} ({cond.name})  "
        f"cost={MATING_COST}  threshold={MATING_THRESHOLD}  seed={args.seed}"
    )
    print(label)
    print(f"Phase list: {[p.__name__ for p in _build_phase_list()]}")
    print()

    sim = Simulation(
        width=50,
        height=50,
        initial_agents=100,
        seed=args.seed,
        env_config=env_cfg,
        policy_mode="neural",
        phases=_build_phase_list(),
    )
    _replace_policies_warm_start(sim)

    renderer = Renderer(delay=args.delay, condition_label=label)
    step = 0

    while step < args.steps and renderer.running:
        sim.step()
        step += 1
        renderer.render(sim, step)

    renderer.close()
    sim.plot_history(block=True)


if __name__ == "__main__":
    main()

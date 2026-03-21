"""Run a simulation session with the desktop renderer.

Usage:
    python -m tests.run
    python -m tests.run --steps 500
    python -m tests.run --steps 500 --seed 42
"""

import argparse

from src.core.simulation import Simulation
from src.viz.renderer import Renderer
from experiments.phase2 import CONDITIONS

POLICY_MODES = ["baseline", "richer"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulation with desktop visualisation."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: random)"
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        choices=["A", "B", "C", "D"],
        help="Environment condition: A=baseline B=fast_drift "
             "C=boom_bust D=combined (default: baseline)"
    )
    args = parser.parse_args()

    # Resolve environment config from condition label
    cond_map = {c.name[0]: c for c in CONDITIONS}
    cond = cond_map.get(args.condition or "A")
    env_config = {
        "drift_step":      cond.drift_step,
        "noise_rate":      cond.noise_rate,
        "noise_magnitude": cond.noise_magnitude,
    }

    current_mode = ["baseline"]

    def make_sim():
        return Simulation(
            width=50, height=50,
            initial_agents=100,
            seed=args.seed,
            env_config=env_config,
            policy_mode=current_mode[0],
        )

    sim = make_sim()
    step = 0
    renderer = Renderer(
        delay=0.05,
        condition_label=cond.name,
    )

    while step < args.steps and renderer.running:
        sim.step()
        step += 1
        renderer.render(sim, step)
        # Check if mode toggle was requested
        if renderer.mode_toggle_requested:
            renderer.mode_toggle_requested = False
            idx = POLICY_MODES.index(current_mode[0])
            current_mode[0] = POLICY_MODES[(idx + 1) % len(POLICY_MODES)]
            sim = make_sim()
            step = 0
            renderer._history.clear()

    renderer.close()
    sim.plot_history(block=True)


if __name__ == "__main__":
    main()

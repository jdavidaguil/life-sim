"""Phase 2 experiment: effect of environmental volatility on trait evolution.

Runs four conditions with identical seeds and compares trait attractors.

Usage:
    python -m experiments.phase2
    python -m experiments.phase2 --steps 1000 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.core.simulation import Simulation


@dataclass
class Condition:
    name: str
    drift_step: int
    noise_rate: float
    noise_magnitude: float


CONDITIONS: list[Condition] = [
    Condition("A — Baseline",         drift_step=1, noise_rate=3.0, noise_magnitude=2.0),
    Condition("B — Fast drift",       drift_step=3, noise_rate=3.0, noise_magnitude=2.0),
    Condition("C — Boom/bust",        drift_step=1, noise_rate=8.0, noise_magnitude=4.0),
    Condition("D — Combined",         drift_step=3, noise_rate=8.0, noise_magnitude=4.0),
]


def run_condition(cond: Condition, seed: int, steps: int) -> dict:
    """Run one condition and return final trait means and population."""
    sim = Simulation(
        width=50, height=50,
        initial_agents=100,
        seed=seed,
        env_config={
            "drift_step":       cond.drift_step,
            "noise_rate":       cond.noise_rate,
            "noise_magnitude":  cond.noise_magnitude,
        },
    )
    for _ in range(steps):
        sim.step()

    import numpy as np
    traits = np.array([a.policy.traits for a in sim.agents], dtype=np.float32)
    return {
        "population":  sim.agent_count(),
        "rw":          float(traits[:, 0].mean()),
        "cs":          float(traits[:, 1].mean()),
        "noise":       float(traits[:, 2].mean()),
        "ea":          float(traits[:, 3].mean()),
        "std_rw":      float(traits[:, 0].std()),
        "std_noise":   float(traits[:, 2].std()),
    }


def run_condition_multi_seed(
    cond: Condition,
    seeds: list[int],
    steps: int,
) -> dict:
    """Run one condition across multiple seeds and return mean ± std."""
    import numpy as np

    all_results = []
    for seed in seeds:
        r = run_condition(cond, seed=seed, steps=steps)
        all_results.append(r)

    keys = ["population", "rw", "cs", "noise", "ea"]
    out = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals))
    return out


def print_table(results: list[tuple[Condition, dict]]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Condition':<28} {'Pop':>5} "
        f"{'rw':>6} {'cs':>6} {'noise':>6} {'ea':>6} "
        f"{'std_rw':>7} {'std_noise':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for cond, r in results:
        print(
            f"{cond.name:<28} {r['population']:>5} "
            f"{r['rw']:>6.3f} {r['cs']:>6.3f} "
            f"{r['noise']:>6.3f} {r['ea']:>6.3f} "
            f"{r['std_rw']:>7.3f} {r['std_noise']:>9.3f}"
        )
    print("=" * len(header))


def print_multi_table(
    results: list[tuple[Condition, dict]],
    seeds: list[int],
) -> None:
    """Print mean ± std comparison table across seeds."""
    print(f"\nSeeds used: {seeds}")
    header = (
        f"{'Condition':<28} "
        f"{'pop':>9} "
        f"{'rw':>12} "
        f"{'cs':>12} "
        f"{'noise':>12} "
        f"{'ea':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for cond, r in results:
        print(
            f"{cond.name:<28} "
            f"{r['population_mean']:>5.0f}\u00b1{r['population_std']:>3.0f} "
            f"{r['rw_mean']:>6.3f}\u00b1{r['rw_std']:>5.3f} "
            f"{r['cs_mean']:>6.3f}\u00b1{r['cs_std']:>5.3f} "
            f"{r['noise_mean']:>6.3f}\u00b1{r['noise_std']:>5.3f} "
            f"{r['ea_mean']:>6.3f}\u00b1{r['ea_std']:>5.3f}"
        )
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: environmental volatility experiment."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps per condition (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for all conditions (default: 42)"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds to run per condition (default: 5)"
    )
    args = parser.parse_args()

    import numpy as np
    seeds = [args.seed + i for i in range(args.seeds)] \
            if args.seed is not None \
            else list(range(args.seeds))

    print(f"Phase 2 — Environmental Volatility")
    print(f"Seeds: {seeds}  |  Steps: {args.steps}")
    print(f"Running {len(CONDITIONS)} conditions...\n")

    multi_results = []
    for cond in CONDITIONS:
        print(f"  Running {cond.name} across {len(seeds)} seeds...")
        r = run_condition_multi_seed(cond, seeds=seeds, steps=args.steps)
        multi_results.append((cond, r))
        print(
            f"    done — "
            f"pop={r['population_mean']:.0f}\u00b1{r['population_std']:.0f}  "
            f"rw={r['rw_mean']:.3f}\u00b1{r['rw_std']:.3f}  "
            f"noise={r['noise_mean']:.3f}\u00b1{r['noise_std']:.3f}  "
            f"cs={r['cs_mean']:.3f}\u00b1{r['cs_std']:.3f}"
        )

    print_multi_table(multi_results, seeds)


if __name__ == "__main__":
    main()

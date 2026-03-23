"""Phase 6 baseline: re-run Phases 4 and 5 on standard environment.

Uses warm-start initialization for both neural phases.
Standard environment matches Phases 1-3 exactly:
    HOTSPOT_SIGMA=4.0, NUM_HOTSPOTS=4
    NOISE_RATE=3.0, NOISE_MAGNITUDE=2.0

Usage:
    python -m experiments.phase6_baseline
    python -m experiments.phase6_baseline --steps 1000 --seed 42 --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.experiments.phase2 import CONDITIONS, Condition
from src.experiments.probe_phase4 import (
    DIR_LABELS,
    PROBE_SITUATIONS,
    probe_population,
)
from src.core.grid import Grid
from src.core.policy import NeuralPolicy, StatefulNeuralPolicy
from src.core.state import SimState
from src.core.simulation import Simulation
from src.experiments.base import Experiment


RESULTS_DIR = Path(__file__).parent / "results"


def _init_neural_warmstart(state: SimState) -> None:
    """Replace agents with warm-start NeuralPolicy at step 0."""
    if state.step != 0:
        return
    for agent in state.agents:
        agent.policy = NeuralPolicy(rng=state.rng, warm_start=True)


EXPERIMENT = Experiment(
    additions={"before_move": [_init_neural_warmstart]},
    env_config={"drift_step": 1, "noise_rate": 3.0, "noise_magnitude": 2.0},
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
    result_id="phase6_baseline",
)


def build_phase6_env_config(cond: Condition) -> dict:
    """Return the standard-environment config used by Phase 6."""
    return {
        "drift_step": cond.drift_step,
        "noise_rate": cond.noise_rate,
        "noise_magnitude": cond.noise_magnitude,
    }


def warm_start_phase6_population(
    sim: Simulation,
    seed: int | None,
    policy_mode: str,
) -> None:
    """Replace the default policies with Phase 6 warm-started ones."""
    rng_ws = np.random.default_rng(seed) if seed is not None else sim.rng
    policy_class = (
        NeuralPolicy
        if policy_mode == "neural"
        else StatefulNeuralPolicy
    )
    for agent in sim.agents:
        agent.policy = policy_class(rng=rng_ws, warm_start=True)


def create_phase6_simulation(
    seed: int | None,
    condition: Condition,
    policy_mode: str = "neural",
    width: int = 50,
    height: int = 50,
    initial_agents: int = 100,
) -> Simulation:
    """Create a simulation configured like the Phase 6 baseline."""
    if policy_mode not in {"neural", "stateful"}:
        raise ValueError(f"unsupported Phase 6 policy mode: {policy_mode}")

    sim = Simulation(
        width=width,
        height=height,
        initial_agents=initial_agents,
        seed=seed,
        env_config=build_phase6_env_config(condition),
        policy_mode=policy_mode,
    )
    warm_start_phase6_population(sim, seed=seed, policy_mode=policy_mode)
    return sim


def run_neural_condition(
    cond: Condition,
    seed: int,
    steps: int,
    policy_mode: str = "neural",
) -> dict:
    """Run one condition with warm-start neural policy.
    Standard environment -- no Grid parameter overrides.
    """
    sim = create_phase6_simulation(
        seed=seed,
        condition=cond,
        policy_mode=policy_mode,
    )

    for _ in range(steps):
        sim.step()

    rng = np.random.default_rng(seed)
    probe_results = []
    for situation in PROBE_SITUATIONS:
        probs = probe_population(
            sim.agents,
            situation,
            n_sample=min(50, sim.agent_count()),
            rng=rng,
        )
        probe_results.append({
            "situation": situation["name"].replace("\n", " "),
            "top_dir": DIR_LABELS[np.argmax(probs)],
            "spread": float(probs.max() - probs.min()),
            "max_prob": float(probs.max()),
        })

    state_std = None
    if policy_mode == "stateful":
        states = np.array([
            agent.policy.state
            for agent in sim.agents
            if isinstance(agent.policy, StatefulNeuralPolicy)
        ])
        if len(states):
            state_std = float(states.std())

    norms = [
        float(np.linalg.norm(agent.policy.genome))
        for agent in sim.agents
    ]

    return {
        "population": sim.agent_count(),
        "genome_norm_mean": float(np.mean(norms)),
        "genome_norm_std": float(np.std(norms)),
        "total_births": sim.reproductions_total,
        "probe": probe_results,
        "state_std": state_std,
        "history_total": sim.history["total"],
        "history_steps": sim.history["step"],
    }


def run_multi_seed(
    cond: Condition,
    seeds: list[int],
    steps: int,
    policy_mode: str,
) -> dict:
    all_results = []
    for seed in seeds:
        result = run_neural_condition(
            cond,
            seed=seed,
            steps=steps,
            policy_mode=policy_mode,
        )
        all_results.append(result)

    populations = [result["population"] for result in all_results]
    spreads_by_situation = []
    for situation_index in range(len(PROBE_SITUATIONS)):
        spreads = [
            result["probe"][situation_index]["spread"]
            for result in all_results
        ]
        spreads_by_situation.append({
            "situation": all_results[0]["probe"][situation_index]["situation"],
            "mean_spread": float(np.mean(spreads)),
            "std_spread": float(np.std(spreads)),
        })

    totals = np.array([result["history_total"] for result in all_results])

    return {
        "population_mean": float(np.mean(populations)),
        "population_std": float(np.std(populations)),
        "probe_spreads": spreads_by_situation,
        "history_total_mean": totals.mean(axis=0).tolist(),
        "history_steps": all_results[0]["history_steps"],
    }


def save_results(
    phase4_results: list[tuple[Condition, dict]],
    phase5_results: list[tuple[Condition, dict]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "environment": "standard",
        "phase4_warmstart": {},
        "phase5_warmstart": {},
    }
    for cond, result in phase4_results:
        output["phase4_warmstart"][cond.name] = {
            "population_mean": result["population_mean"],
            "population_std": result["population_std"],
            "probe_spreads": result["probe_spreads"],
        }
    for cond, result in phase5_results:
        output["phase5_warmstart"][cond.name] = {
            "population_mean": result["population_mean"],
            "population_std": result["population_std"],
            "probe_spreads": result["probe_spreads"],
        }
    path = RESULTS_DIR / "phase6_baseline.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {path}")


def print_summary(
    phase4: list[tuple[Condition, dict]],
    phase5: list[tuple[Condition, dict]],
) -> None:
    print("\n" + "=" * 70)
    print(f"{'Phase 4 vs Phase 5 -- Standard Environment':^70}")
    print("=" * 70)
    header = (
        f"{'Condition':<28} "
        f"{'Phase 4 pop':>12} "
        f"{'Phase 5 pop':>12} "
        f"{'P4 spread[0]':>12} "
        f"{'P5 spread[0]':>12}"
    )
    print(header)
    print("-" * 70)
    for (cond4, result4), (_, result5) in zip(phase4, phase5):
        phase4_spread = result4["probe_spreads"][0]["mean_spread"]
        phase5_spread = result5["probe_spreads"][0]["mean_spread"]
        print(
            f"{cond4.name:<28} "
            f"{result4['population_mean']:>8.0f}"
            f"±{result4['population_std']:>3.0f} "
            f"{result5['population_mean']:>8.0f}"
            f"±{result5['population_std']:>3.0f} "
            f"{phase4_spread:>12.3f} "
            f"{phase5_spread:>12.3f}"
        )
    print("=" * 70)
    print("spread[0] = 'Rich neighbor no crowd' probe situation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 baseline: neural phases on standard env."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Steps per condition (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds (default: 5)",
    )
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    print("Phase 6 Baseline -- Standard Environment")
    print(f"Seeds: {seeds}  Steps: {args.steps}")
    print(
        f"Grid: HOTSPOT_SIGMA={Grid.HOTSPOT_SIGMA} "
        f"NUM_HOTSPOTS={Grid.NUM_HOTSPOTS}"
    )
    print("Warm-start: True\n")

    print("Running Phase 4 (neural, warm-start)...")
    phase4_results = []
    for cond in CONDITIONS:
        print(f"  {cond.name}...")
        result = run_multi_seed(
            cond,
            seeds=seeds,
            steps=args.steps,
            policy_mode="neural",
        )
        phase4_results.append((cond, result))
        print(
            f"    pop={result['population_mean']:.0f}"
            f"±{result['population_std']:.0f}  "
            f"spread={result['probe_spreads'][0]['mean_spread']:.3f}"
        )

    print("\nRunning Phase 5 (stateful, warm-start)...")
    phase5_results = []
    for cond in CONDITIONS:
        print(f"  {cond.name}...")
        result = run_multi_seed(
            cond,
            seeds=seeds,
            steps=args.steps,
            policy_mode="stateful",
        )
        phase5_results.append((cond, result))
        print(
            f"    pop={result['population_mean']:.0f}"
            f"±{result['population_std']:.0f}  "
            f"spread={result['probe_spreads'][0]['mean_spread']:.3f}"
        )

    print_summary(phase4_results, phase5_results)
    save_results(phase4_results, phase5_results)


if __name__ == "__main__":
    main()
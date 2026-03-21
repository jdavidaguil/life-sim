"""Phase 4 experiment: neural genome policy.

Replaces the linear scoring function with a small MLP.
Genome = 112 weights. Selection is the optimizer.
Compares against Phase 3 (richer perception) baseline.

Usage:
    python -m experiments.phase4
    python -m experiments.phase4 --steps 1000 --seed 42 --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.core.simulation import Simulation
from experiments.phase2 import CONDITIONS, Condition, load_results


RESULTS_DIR = Path(__file__).parent / "results"


def run_condition_neural(
    cond: Condition,
    seed: int,
    steps: int,
) -> dict:
    """Run one condition with neural policy and return behavioral metrics."""
    sim = Simulation(
        width=50, height=50,
        initial_agents=100,
        seed=seed,
        env_config={
            "drift_step":      cond.drift_step,
            "noise_rate":      cond.noise_rate,
            "noise_magnitude": cond.noise_magnitude,
        },
        policy_mode="neural",
    )
    for _ in range(steps):
        sim.step()

    # Behavioral proxies — no trait vector available
    agents = sim.agents
    genome_norms = [float(np.linalg.norm(a.policy.genome))
                    for a in agents]

    return {
        "population":       sim.agent_count(),
        "mean_genome_norm": float(np.mean(genome_norms)),
        "std_genome_norm":  float(np.std(genome_norms)),
        "total_births":     sim.reproductions_total,
        "history_total":    sim.history["total"],
        "history_steps":    sim.history["step"],
    }


def run_condition_neural_multi_seed(
    cond: Condition,
    seeds: list[int],
    steps: int,
) -> dict:
    """Run one condition across multiple seeds."""
    all_results = []
    for seed in seeds:
        r = run_condition_neural(cond, seed=seed, steps=steps)
        all_results.append(r)

    pops   = [r["population"]       for r in all_results]
    norms  = [r["mean_genome_norm"] for r in all_results]
    births = [r["total_births"]     for r in all_results]

    totals = np.array([r["history_total"] for r in all_results])

    return {
        "population_mean":       float(np.mean(pops)),
        "population_std":        float(np.std(pops)),
        "mean_genome_norm_mean": float(np.mean(norms)),
        "mean_genome_norm_std":  float(np.std(norms)),
        "total_births_mean":     float(np.mean(births)),
        "total_births_std":      float(np.std(births)),
        "history_total_mean":    totals.mean(axis=0).tolist(),
        "history_total_upper":   (totals.mean(axis=0) +
                                  totals.std(axis=0)).tolist(),
        "history_total_lower":   (totals.mean(axis=0) -
                                  totals.std(axis=0)).tolist(),
        "history_steps":         all_results[0]["history_steps"],
    }


def print_table(results: list[tuple[Condition, dict]]) -> None:
    header = (
        f"{'Condition':<28} {'pop':>9} "
        f"{'genome_norm':>13} {'total_births':>14}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for cond, r in results:
        print(
            f"{cond.name:<28} "
            f"{r['population_mean']:>5.0f}"
            f"±{r['population_std']:>3.0f} "
            f"{r['mean_genome_norm_mean']:>8.2f}"
            f"±{r['mean_genome_norm_std']:>4.2f} "
            f"{r['total_births_mean']:>10.0f}"
            f"±{r['total_births_std']:>3.0f}"
        )
    print("=" * len(header))


def print_population_comparison(
    phase3: dict | None,
    phase4: list[tuple[Condition, dict]],
) -> None:
    """Compare population at step 1000 across phases."""
    if not phase3:
        return
    print("\n" + "=" * 70)
    print(f"{'Population comparison — Phase 3 vs Phase 4':^70}")
    print("=" * 70)
    header = (
        f"{'Condition':<28} "
        f"{'Phase 3 pop':>12} "
        f"{'Phase 4 pop':>12} "
        f"{'diff':>8}"
    )
    print(header)
    print("-" * 70)
    for cond, r4 in phase4:
        r3 = phase3.get(cond.name, {})
        p3 = r3.get("population_mean", 0)
        p4 = r4["population_mean"]
        diff = p4 - p3
        sign = "+" if diff >= 0 else ""
        print(
            f"{cond.name:<28} "
            f"{p3:>8.0f}±{r3.get('population_std',0):>3.0f} "
            f"{p4:>8.0f}±{r4['population_std']:>3.0f} "
            f"{sign}{diff:>6.0f}"
        )
    print("=" * 70)


def plot_population(results: list[tuple[Condition, dict]]) -> None:
    """Plot population trajectories for all four conditions."""
    import matplotlib.pyplot as plt

    dark_bg = "#111111"
    panel_bg = "#1a1a1a"
    COND_COLORS = ["#ff4444", "#4488ff", "#44cc44", "#ffaa00"]

    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4), sharey=True,
    )
    fig.patch.set_facecolor(dark_bg)
    fig.suptitle(
        "Phase 4 — Neural genome: population over time (mean ± 1σ)",
        color="#eeeeee", fontsize=10,
    )

    for ci, (cond, r) in enumerate(results):
        ax = axes[ci]
        ax.set_facecolor(panel_bg)
        steps = r["history_steps"]
        mean  = r["history_total_mean"]
        upper = r["history_total_upper"]
        lower = r["history_total_lower"]
        color = COND_COLORS[ci]
        ax.plot(steps, mean, color=color, lw=1.5)
        ax.fill_between(steps, lower, upper,
                        color=color, alpha=0.2)
        ax.set_title(cond.name, color="#dddddd", fontsize=8)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_xlabel("Step", color="#aaaaaa", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Population", color="#aaaaaa", fontsize=8)

    fig.tight_layout()
    plt.show()


def save_results(results: list[tuple[Condition, dict]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for cond, r in results:
        output[cond.name] = {
            "population_mean": r["population_mean"],
            "population_std":  r["population_std"],
        }
    path = RESULTS_DIR / "phase4.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: neural genome experiment."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps per condition (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds per condition (default: 5)"
    )
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]

    print(f"Phase 4 — Neural Genome")
    print(f"Seeds: {seeds}  |  Steps: {args.steps}")
    print(f"Running {len(CONDITIONS)} conditions...\n")

    results = []
    for cond in CONDITIONS:
        print(f"  Running {cond.name} across {len(seeds)} seeds...")
        r = run_condition_neural_multi_seed(
            cond, seeds=seeds, steps=args.steps
        )
        results.append((cond, r))
        print(
            f"    done — "
            f"pop={r['population_mean']:.0f}±{r['population_std']:.0f}  "
            f"genome_norm={r['mean_genome_norm_mean']:.2f}  "
            f"births={r['total_births_mean']:.0f}"
        )

    print_table(results)

    phase3 = load_results("phase3")
    print_population_comparison(phase3, results)

    save_results(results)
    plot_population(results)


if __name__ == "__main__":
    main()

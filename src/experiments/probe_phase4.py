"""Phase 4 behavioral probe.

Runs a simulation with neural policy, samples agents at end state,
and probes their networks with standardized inputs to characterize
the emergent strategy.

Usage:
    python -m experiments.probe_phase4
    python -m experiments.probe_phase4 --condition A --seed 42
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from src.experiments.phase2 import CONDITIONS
from src.core.policy import NeuralPolicy, StatefulNeuralPolicy
from src.core.simulation import Simulation


PROBE_SITUATIONS = [
    {
        "name": "Rich neighbor\nno crowd",
        "inputs": np.array([
            0.2, 0.8, 0.2,
            0.1, 0.1,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0,
            0.5,
        ], dtype=np.float32),
        "description": "High resource ahead, uncrowded",
    },
    {
        "name": "Rich neighbor\ncrowded",
        "inputs": np.array([
            0.2, 0.8, 0.2,
            0.1, 0.1,
            0.0, 0.0, 0.0,
            0.2, 0.8, 0.2,
            0.1, 0.1,
            0.0, 0.0, 0.0,
            0.0,
            0.5,
        ], dtype=np.float32),
        "description": "High resource ahead but crowded",
    },
    {
        "name": "Low energy\ngradient up",
        "inputs": np.array([
            0.6, 0.8, 0.6,
            0.4, 0.4,
            0.1, 0.2, 0.1,
            0.2, 0.2, 0.2,
            0.2, 0.2,
            0.1, 0.1, 0.1,
            -0.3,
            0.4,
        ], dtype=np.float32),
        "description": "Below average energy, moving toward food",
    },
    {
        "name": "High energy\ncrowded",
        "inputs": np.array([
            0.4, 0.4, 0.4,
            0.4, 0.4,
            0.4, 0.4, 0.4,
            0.9, 0.9, 0.9,
            0.9, 0.9,
            0.9, 0.9, 0.9,
            0.3,
            0.4,
        ], dtype=np.float32),
        "description": "Above average energy, surrounded by crowds",
    },
    {
        "name": "Uniform\nneutral",
        "inputs": np.array([
            0.5, 0.5, 0.5,
            0.5, 0.5,
            0.5, 0.5, 0.5,
            0.3, 0.3, 0.3,
            0.3, 0.3,
            0.3, 0.3, 0.3,
            0.0,
            0.5,
        ], dtype=np.float32),
        "description": "All inputs near midpoint, no gradient",
    },
    {
        "name": "Depleted\nno crowd",
        "inputs": np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            -0.4,
            0.1,
        ], dtype=np.float32),
        "description": "Low resources, negative gradient, low energy",
    },
]


DIR_GRID_POS = [
    (0, 0), (1, 0), (2, 0),
    (0, 1),         (2, 1),
    (0, 2), (1, 2), (2, 2),
]


DIR_LABELS = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]


def probe_agent(policy: NeuralPolicy, situation: dict) -> np.ndarray:
    if isinstance(policy, StatefulNeuralPolicy):
        move_probs, _ = policy._forward(situation["inputs"])
        return move_probs
    return policy._forward(situation["inputs"])


def probe_population(
    agents: list,
    situation: dict,
    n_sample: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    neural_agents = [
        a for a in agents
        if isinstance(a.policy, (NeuralPolicy, StatefulNeuralPolicy))
    ]
    if not neural_agents:
        raise ValueError("No neural agents found")
    n = min(n_sample, len(neural_agents))
    idxs = rng.choice(len(neural_agents), size=n, replace=False)
    sampled = [neural_agents[i] for i in idxs]
    probs = np.array([
        probe_agent(a.policy, situation)
        for a in sampled
    ])
    return probs.mean(axis=0)


def probs_to_grid(probs: np.ndarray) -> np.ndarray:
    grid = np.zeros((3, 3), dtype=np.float32)
    for i, (col, row) in enumerate(DIR_GRID_POS):
        grid[row, col] = probs[i]
    return grid


def plot_probe_results(
    all_results: dict[str, list[np.ndarray]],
    condition_names: list[str],
) -> None:
    n_situations = len(PROBE_SITUATIONS)
    n_conditions = len(condition_names)

    dark_bg = "#111111"
    panel_bg = "#1a1a1a"

    fig = plt.figure(
        figsize=(3 * n_conditions + 1, 3.5 * n_situations),
        facecolor=dark_bg,
    )
    fig.suptitle(
        "Phase 4 — Neural strategy probe\n"
        "Each cell: probability of moving in that direction",
        color="#eeeeee",
        fontsize=11,
    )

    gs = gridspec.GridSpec(
        n_situations, n_conditions,
        figure=fig,
        hspace=0.6,
        wspace=0.3,
    )

    for si, situation in enumerate(PROBE_SITUATIONS):
        for ci, cond_name in enumerate(condition_names):
            ax = fig.add_subplot(gs[si, ci])
            ax.set_facecolor(panel_bg)

            probs = all_results[cond_name][si]
            grid = probs_to_grid(probs)

            ax.imshow(
                grid,
                cmap="YlOrRd",
                vmin=0.0,
                vmax=grid.max() + 0.01,
                interpolation="nearest",
            )

            for row in range(3):
                for col in range(3):
                    if row == 1 and col == 1:
                        ax.text(
                            col, row, "●",
                            ha="center", va="center",
                            color="#888888", fontsize=10,
                        )
                        continue
                    val = grid[row, col]
                    color = "#111111" if val > grid.max() * 0.6 else "#eeeeee"
                    ax.text(
                        col, row, f"{val:.2f}",
                        ha="center", va="center",
                        color=color, fontsize=8,
                    )

            if si == 0:
                ax.set_title(cond_name, color="#dddddd", fontsize=8)
            if ci == 0:
                ax.set_ylabel(
                    situation["name"],
                    color="#aaaaaa",
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    labelpad=60,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Phase 4 neural strategies."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps to run before probing (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        choices=["A", "B", "C", "D"],
        help="Single condition to probe (default: all four)"
    )
    parser.add_argument(
        "--n_sample", type=int, default=50,
        help="Agents to sample per probe (default: 50)"
    )
    args = parser.parse_args()

    cond_map = {c.name[0]: c for c in CONDITIONS}
    if args.condition:
        conditions_to_run = [cond_map[args.condition]]
    else:
        conditions_to_run = list(CONDITIONS)

    rng = np.random.default_rng(args.seed)
    all_results = {}

    for cond in conditions_to_run:
        print(f"Running {cond.name} for {args.steps} steps...")
        sim = Simulation(
            width=50, height=50,
            initial_agents=100,
            seed=args.seed,
            env_config={
                "drift_step": cond.drift_step,
                "noise_rate": cond.noise_rate,
                "noise_magnitude": cond.noise_magnitude,
            },
            policy_mode="neural",
        )
        for _ in range(args.steps):
            sim.step()

        print(
            f"  Pop: {sim.agent_count()}  Probing "
            f"{min(args.n_sample, sim.agent_count())} agents..."
        )

        condition_results = []
        for situation in PROBE_SITUATIONS:
            probs = probe_population(
                sim.agents, situation,
                n_sample=args.n_sample, rng=rng,
            )
            condition_results.append(probs)

            top_dir = DIR_LABELS[np.argmax(probs)]
            print(
                f"    [{situation['name'].replace(chr(10), ' ')}] "
                f"top direction: {top_dir} ({probs.max():.2f})"
            )

        all_results[cond.name] = condition_results

    plot_probe_results(
        all_results,
        condition_names=[c.name for c in conditions_to_run],
    )


if __name__ == "__main__":
    main()
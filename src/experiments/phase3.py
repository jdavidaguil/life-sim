"""Phase 3 experiment: effect of richer perception on trait evolution.

Runs the same four environmental conditions as Phase 2 but with
Phase 3 richer perception (gradient, relative energy, local density).
Compares results directly against saved Phase 2 baseline.

Usage:
    python -m experiments.phase3
    python -m experiments.phase3 --steps 1000 --seed 42 --seeds 5
"""

from __future__ import annotations

from src.experiments.phase2 import (
    CONDITIONS,
    Condition,
    run_condition_multi_seed,
    print_multi_table,
    plot_results,
    save_results,
    load_results,
)
import argparse

from src.core.state import SimState
from src.experiments.base import Experiment


PHASE3_METRICS = [
    "population_mean",
    "population_std",
    "rw_mean",
    "rw_std",
    "cs_mean",
    "cs_std",
    "noise_mean",
    "noise_std",
    "ea_mean",
    "ea_std",
    "agg_history",
    "traits_final",
]

# Default EXPERIMENT: Condition A (baseline) env, richer TraitPolicy (default).
EXPERIMENT = Experiment(
    env_config={"drift_step": 1, "noise_rate": 3.0, "noise_magnitude": 2.0},
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
    result_id="phase3",
)


def build_phase3_experiment(
    cond: Condition,
    seeds: list[int],
    steps: int,
) -> Experiment:
    """Build the explicit Phase 3 experiment configuration."""
    return Experiment(
        phase_overrides={"policy_mode": "richer"},
        phase_additions={
            "resource_gradient": True,
            "local_density": True,
            "relative_energy": True,
        },
        environment_config={
            "drift_step": cond.drift_step,
            "noise_rate": cond.noise_rate,
            "noise_magnitude": cond.noise_magnitude,
        },
        metrics=list(PHASE3_METRICS),
        seeds=list(seeds),
        steps=steps,
    )


def run_phase3_experiment(
    cond: Condition,
    experiment: Experiment,
) -> dict:
    """Run a Phase 3 experiment via explicit config."""
    return run_condition_multi_seed(
        cond,
        seeds=experiment.seeds,
        steps=experiment.steps,
        policy_mode=str(experiment.phase_overrides.get("policy_mode", "richer")),
    )


def print_comparison_table(
    phase2: dict,
    phase3_results: list[tuple[Condition, dict]],
) -> None:
    """Print Phase 2 vs Phase 3 side-by-side for each condition."""
    print("\n" + "=" * 100)
    print(f"{'Phase 2 vs Phase 3 — Trait Attractor Comparison':^100}")
    print("=" * 100)
    header = (
        f"{'Condition':<28}  "
        f"{'── Phase 2 ──':^35}  "
        f"{'── Phase 3 ──':^35}"
    )
    sub = (
        f"{'':28}  "
        f"{'rw':>7} {'cs':>7} {'noise':>7} {'ea':>7}  "
        f"{'rw':>7} {'cs':>7} {'noise':>7} {'ea':>7}"
    )
    print(header)
    print(sub)
    print("-" * 100)
    for cond, r3 in phase3_results:
        r2 = phase2.get(cond.name, {})
        rw2    = r2.get("rw_mean",    0.0)
        cs2    = r2.get("cs_mean",    0.0)
        noise2 = r2.get("noise_mean", 0.0)
        ea2    = r2.get("ea_mean",    0.0)
        # Direction indicators
        def arrow(v3, v2):
            if v3 > v2 + 0.02: return "↑"
            if v3 < v2 - 0.02: return "↓"
            return "→"
        print(
            f"{cond.name:<28}  "
            f"{rw2:>6.3f} {cs2:>6.3f} {noise2:>6.3f} {ea2:>6.3f}   "
            f"{r3['rw_mean']:>5.3f}{arrow(r3['rw_mean'],rw2)} "
            f"{r3['cs_mean']:>5.3f}{arrow(r3['cs_mean'],cs2)} "
            f"{r3['noise_mean']:>5.3f}{arrow(r3['noise_mean'],noise2)} "
            f"{r3['ea_mean']:>5.3f}{arrow(r3['ea_mean'],ea2)}"
        )
    print("=" * 100)
    print("↑ = increased  ↓ = decreased  → = stable (within 0.02)")


def plot_comparison(
    phase2: dict,
    phase3_results: list[tuple[Condition, dict]],
) -> None:
    """Grouped bar chart: Phase 2 vs Phase 3 trait attractors per condition."""
    import matplotlib.pyplot as plt
    import numpy as np

    TRAIT_KEYS   = ["rw",   "cs",    "noise", "ea"]
    TRAIT_LABELS = ["resource_weight", "crowd_sensitivity", "noise", "energy_awareness"]
    TRAIT_COLORS = ["#e07b54", "#5b9bd5", "#70bf7a", "#c77dca"]
    dark_bg = "#1a1a2e"

    cond_names = [cond.name for cond, _ in phase3_results]
    n_cond = len(cond_names)
    x = np.arange(n_cond)
    width = 0.35

    fig, axes = plt.subplots(
        1, len(TRAIT_KEYS),
        figsize=(4.5 * len(TRAIT_KEYS), 5),
        sharey=False,
    )
    fig.patch.set_facecolor(dark_bg)
    fig.suptitle(
        "Phase 2 vs Phase 3 — Trait Attractors (mean ± 1σ)",
        color="#eeeeee", fontsize=12,
    )

    for ai, (tk, label, color) in enumerate(
        zip(TRAIT_KEYS, TRAIT_LABELS, TRAIT_COLORS)
    ):
        ax = axes[ai]
        ax.set_facecolor(dark_bg)

        p2_means = np.array([
            phase2.get(cond.name, {}).get(f"{tk}_mean", 0.0)
            for cond, _ in phase3_results
        ])
        p2_stds = np.array([
            phase2.get(cond.name, {}).get(f"{tk}_std", 0.0)
            for cond, _ in phase3_results
        ])
        p3_means = np.array([r[f"{tk}_mean"] for _, r in phase3_results])
        p3_stds  = np.array([r[f"{tk}_std"]  for _, r in phase3_results])

        bar_kw = dict(capsize=4, error_kw=dict(ecolor="#aaaaaa", lw=1.2))
        ax.bar(x - width / 2, p2_means, width,
               yerr=p2_stds, label="Phase 2",
               color=color, alpha=0.5, **bar_kw)
        ax.bar(x + width / 2, p3_means, width,
               yerr=p3_stds, label="Phase 3",
               color=color, alpha=0.9, **bar_kw)

        ax.set_title(label, color="#dddddd", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.split(" ")[0] for c in cond_names],
            color="#aaaaaa", fontsize=8,
        )
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_ylim(bottom=0)
        if ai == 0:
            ax.legend(
                facecolor="#222222", edgecolor="#555555",
                labelcolor="#dddddd", fontsize=8,
            )

    fig.tight_layout()


def plot_trajectory_comparison(
    p2_results: list[tuple[Condition, dict]],
    p3_results: list[tuple[Condition, dict]],
) -> None:
    """Phase 2 and Phase 3 in separate rows — no overlaying.

    Layout: 4 rows × N_COND columns.
      Row 0 — Phase 2 population
      Row 1 — Phase 3 population
      Row 2 — Phase 2 trait means
      Row 3 — Phase 3 trait means
    """
    import matplotlib.pyplot as plt
    import numpy as np

    TRAIT_KEYS   = ["rw",   "cs",    "noise", "ea"]
    TRAIT_COLORS = ["#e07b54", "#5b9bd5", "#70bf7a", "#c77dca"]
    dark_bg = "#1a1a2e"
    N_COND = len(p3_results)

    fig, axes = plt.subplots(
        4, N_COND,
        figsize=(4 * N_COND, 12),
        sharex=True,
    )
    fig.patch.set_facecolor(dark_bg)
    fig.suptitle(
        "Phase 2 vs Phase 3 — Population & Trait Trajectories (mean ± 1σ)",
        color="#eeeeee", fontsize=11,
    )

    ROW_LABELS = [
        "Phase 2 — Population",
        "Phase 3 — Population",
        "Phase 2 — Mean trait",
        "Phase 3 — Mean trait",
    ]

    def _style(ax, title="", ylabel=""):
        ax.set_facecolor(dark_bg)
        if ylabel:
            ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        if title:
            ax.set_title(title, color="#dddddd", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    for ci, ((cond, r2), (_, r3)) in enumerate(zip(p2_results, p3_results)):
        h2 = r2["agg_history"]
        h3 = r3["agg_history"]

        # ── Row 0: Phase 2 population ──────────────────────────────
        ax = axes[0, ci]
        ax.plot(h2["steps"], h2["total_mean"], color="#aaaaaa", lw=1.4)
        ax.fill_between(h2["steps"], h2["total_lower"], h2["total_upper"],
                        color="#aaaaaa", alpha=0.15)
        _style(ax,
               ylabel=ROW_LABELS[0] if ci == 0 else "",
               title=cond.name)

        # ── Row 1: Phase 3 population ──────────────────────────────
        ax = axes[1, ci]
        ax.plot(h3["steps"], h3["total_mean"], color="#ffffff", lw=1.4)
        ax.fill_between(h3["steps"], h3["total_lower"], h3["total_upper"],
                        color="#ffffff", alpha=0.15)
        _style(ax, ylabel=ROW_LABELS[1] if ci == 0 else "")

        # ── Row 2: Phase 2 traits ──────────────────────────────────
        ax = axes[2, ci]
        for tk, color in zip(TRAIT_KEYS, TRAIT_COLORS):
            ax.plot(h2["steps"], h2[f"mean_{tk}"], color=color, lw=1.3, label=tk)
            ax.fill_between(h2["steps"],
                            h2[f"mean_{tk}_lower"], h2[f"mean_{tk}_upper"],
                            color=color, alpha=0.12)
        ax.set_ylim(0, 1.5)
        _style(ax, ylabel=ROW_LABELS[2] if ci == 0 else "")
        if ci == N_COND - 1:
            ax.legend(facecolor="#222222", edgecolor="#555555",
                      labelcolor="#dddddd", fontsize=7, loc="upper right")

        # ── Row 3: Phase 3 traits ──────────────────────────────────
        ax = axes[3, ci]
        for tk, color in zip(TRAIT_KEYS, TRAIT_COLORS):
            ax.plot(h3["steps"], h3[f"mean_{tk}"], color=color, lw=1.3, label=tk)
            ax.fill_between(h3["steps"],
                            h3[f"mean_{tk}_lower"], h3[f"mean_{tk}_upper"],
                            color=color, alpha=0.12)
        ax.set_ylim(0, 1.5)
        _style(ax, ylabel=ROW_LABELS[3] if ci == 0 else "")
        ax.set_xlabel("Step", color="#aaaaaa", fontsize=8)
        if ci == N_COND - 1:
            ax.legend(facecolor="#222222", edgecolor="#555555",
                      labelcolor="#dddddd", fontsize=7, loc="upper right")

    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: richer perception experiment."
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

    import numpy as np
    seeds = [args.seed + i for i in range(args.seeds)]

    print(f"Phase 3 — Richer Perception")
    print(f"Seeds: {seeds}  |  Steps: {args.steps}")
    print(f"Running {len(CONDITIONS)} conditions...\n")

    multi_results = []
    for cond in CONDITIONS:
        experiment = build_phase3_experiment(cond, seeds=seeds, steps=args.steps)
        print(f"  Running {cond.name} across {len(seeds)} seeds...")
        r = run_phase3_experiment(cond, experiment)
        multi_results.append((cond, r))
        print(
            f"    done — "
            f"pop={r['population_mean']:.0f}±{r['population_std']:.0f}  "
            f"rw={r['rw_mean']:.3f}±{r['rw_std']:.3f}  "
            f"noise={r['noise_mean']:.3f}±{r['noise_std']:.3f}  "
            f"cs={r['cs_mean']:.3f}±{r['cs_std']:.3f}  "
            f"ea={r['ea_mean']:.3f}±{r['ea_std']:.3f}"
        )

    print("\nPhase 3 results:")
    print_multi_table(multi_results, seeds)

    # Load Phase 2 for comparison
    phase2 = load_results("phase2")
    if phase2:
        print_comparison_table(phase2, multi_results)
    else:
        print(
            "\nNo Phase 2 results found. "
            "Run `python -m experiments.phase2` first to generate baseline."
        )

    save_results(multi_results, phase="phase3")

    print("\nGenerating plots...")
    if phase2:
        # Build p2_traj_results from saved agg_history — no re-run needed.
        p2_traj_results = []
        for cond in CONDITIONS:
            saved = phase2.get(cond.name, {})
            agg = saved.get("agg_history")
            if agg is not None:
                p2_traj_results.append((cond, {"agg_history": agg}))
        if p2_traj_results:
            print("  Opening comparison trajectories (4 rows × 4 conditions)...")
            plot_trajectory_comparison(p2_traj_results, multi_results)
        print("  Opening comparison summary bars (Phase 2 vs Phase 3)...")
        plot_comparison(phase2, multi_results)
    else:
        print(
            "Skipping trajectory comparison — "
            "run `python -m experiments.phase2` first."
        )

    print("  Opening Phase 3-only plots...")
    plot_results(multi_results)


if __name__ == "__main__":
    main()

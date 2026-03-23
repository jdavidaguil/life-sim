"""Phase 2 experiment: effect of environmental volatility on trait evolution.

Runs four conditions with identical seeds and compares trait attractors.

Usage:
    python -m experiments.phase2
    python -m experiments.phase2 --steps 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.simulation import Simulation
from src.core.policy import TraitPolicy
from src.core.state import SimState
from src.experiments.base import Experiment
from src.experiments.conditions import Condition, CONDITIONS  # noqa: F401  (re-exported)


RESULTS_DIR = Path(__file__).parent / "results"

PHASE1_METRICS = [
    "population",
    "rw",
    "cs",
    "noise",
    "ea",
    "std_rw",
    "std_noise",
]

PHASE2_METRICS = [
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


def _init_baseline_policies(state: SimState) -> None:
    """Replace agents with baseline-mode TraitPolicy at step 0."""
    if state.step != 0:
        return
    for agent in state.agents:
        agent.policy = TraitPolicy(rng=state.rng, mode="baseline")


EXPERIMENT = Experiment(
    additions={"before_move": [_init_baseline_policies]},
    env_config={"drift_step": 1, "noise_rate": 3.0, "noise_magnitude": 2.0},
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
    result_id="phase2",
)
def build_phase1_experiment(
    seed: int,
    steps: int,
    condition: Condition | None = None,
) -> Experiment:
    """Build the explicit Phase 1 baseline configuration."""
    cond = CONDITIONS[0] if condition is None else condition
    return Experiment(
        phase_overrides={"policy_mode": "baseline"},
        phase_additions={},
        environment_config={
            "drift_step": cond.drift_step,
            "noise_rate": cond.noise_rate,
            "noise_magnitude": cond.noise_magnitude,
        },
        metrics=list(PHASE1_METRICS),
        seeds=[seed],
        steps=steps,
    )


def build_phase2_experiment(
    cond: Condition,
    seeds: list[int],
    steps: int,
    policy_mode: str = "baseline",
) -> Experiment:
    """Build the explicit Phase 2 experiment configuration."""
    return Experiment(
        phase_overrides={"policy_mode": policy_mode},
        phase_additions={"environment_volatility": True},
        environment_config={
            "drift_step": cond.drift_step,
            "noise_rate": cond.noise_rate,
            "noise_magnitude": cond.noise_magnitude,
        },
        metrics=list(PHASE2_METRICS),
        seeds=list(seeds),
        steps=steps,
    )


def run_phase2_experiment(
    cond: Condition,
    experiment: Experiment,
) -> dict:
    """Run a Phase 2-style multi-seed experiment via explicit config."""
    return run_condition_multi_seed(
        cond,
        seeds=experiment.seeds,
        steps=experiment.steps,
        policy_mode=str(experiment.phase_overrides.get("policy_mode", "baseline")),
    )


def run_condition(
    cond: Condition,
    seed: int,
    steps: int,
    policy_mode: str = "baseline",
) -> dict:
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
        policy_mode=policy_mode,
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
        "history": {
            "steps":      sim.history["step"],
            "total":      sim.history["total"],
            "mean_rw":    sim.history["mean_resource_weight"],
            "mean_cs":    sim.history["mean_crowd_sensitivity"],
            "mean_noise": sim.history["mean_noise"],
            "mean_ea":    sim.history["mean_energy_awareness"],
            "std_rw":     sim.history["std_resource_weight"],
            "std_cs":     sim.history["std_crowd_sensitivity"],
            "std_noise":  sim.history["std_noise"],
            "std_ea":     sim.history["std_energy_awareness"],
        },
        "traits_final": np.array(
            [a.policy.traits for a in sim.agents],
            dtype=np.float32
        ),
    }


def run_condition_multi_seed(
    cond: Condition,
    seeds: list[int],
    steps: int,
    policy_mode: str = "baseline",
) -> dict:
    """Run one condition across multiple seeds and return mean ± std."""
    import numpy as np

    all_results = []
    for seed in seeds:
        r = run_condition(cond, seed=seed, steps=steps, policy_mode=policy_mode)
        all_results.append(r)

    keys = ["population", "rw", "cs", "noise", "ea"]
    out = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals))

    # Stack histories — all seeds should have same length
    # Average per-step mean and std across seeds
    steps = all_results[0]["history"]["steps"]
    trait_keys = ["rw", "cs", "noise", "ea"]
    agg_history = {"steps": steps}
    for tk in trait_keys:
        means = np.array([r["history"][f"mean_{tk}"]
                          for r in all_results])
        stds  = np.array([r["history"][f"std_{tk}"]
                          for r in all_results])
        agg_history[f"mean_{tk}"]       = means.mean(axis=0)
        agg_history[f"mean_{tk}_upper"] = means.mean(axis=0) + means.std(axis=0)
        agg_history[f"mean_{tk}_lower"] = means.mean(axis=0) - means.std(axis=0)
        agg_history[f"std_{tk}"]        = stds.mean(axis=0)

    totals = np.array([r["history"]["total"] for r in all_results])
    agg_history["total_mean"]  = totals.mean(axis=0)
    agg_history["total_upper"] = totals.mean(axis=0) + totals.std(axis=0)
    agg_history["total_lower"] = totals.mean(axis=0) - totals.std(axis=0)

    # Stack end-state trait vectors across seeds
    traits_final = np.vstack([r["traits_final"]
                               for r in all_results])

    out["agg_history"]  = agg_history
    out["traits_final"] = traits_final
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


def save_results(
    results: list[tuple[Condition, dict]],
    phase: str,
) -> None:
    """Save multi-seed results to JSON for cross-phase comparison."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for cond, r in results:
        # Convert numpy arrays in agg_history to plain lists for JSON.
        agg = {k: (v.tolist() if hasattr(v, "tolist") else list(v))
               for k, v in r["agg_history"].items()}
        output[cond.name] = {
            "population_mean":  r["population_mean"],
            "population_std":   r["population_std"],
            "rw_mean":          r["rw_mean"],
            "rw_std":           r["rw_std"],
            "cs_mean":          r["cs_mean"],
            "cs_std":           r["cs_std"],
            "noise_mean":       r["noise_mean"],
            "noise_std":        r["noise_std"],
            "ea_mean":          r["ea_mean"],
            "ea_std":           r["ea_std"],
            "agg_history":      agg,
        }
    path = RESULTS_DIR / f"{phase}.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {path}")


def load_results(phase: str) -> dict | None:
    """Load saved results for a phase. Returns None if not found."""
    path = RESULTS_DIR / f"{phase}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def plot_results(
    results: list[tuple[Condition, dict]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    TRAIT_KEYS   = ["rw",     "cs",     "noise",  "ea"]
    TRAIT_LABELS = ["resource_weight", "crowd_sensitivity",
                    "noise",  "energy_awareness"]
    TRAIT_COLORS = ["#ff4444", "#4488ff", "#44cc44", "#ffaa00"]
    COND_NAMES   = [c.name for c, _ in results]  # noqa: F841
    N_COND       = len(results)

    dark_bg  = "#111111"
    panel_bg = "#1a1a1a"

    def _style(ax, ylabel="", title=""):
        ax.set_facecolor(panel_bg)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.set_title(title,   color="#dddddd", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── Figure 1: Trait trajectories ──────────────────────────
    fig1, axes1 = plt.subplots(
        4, N_COND,
        figsize=(4 * N_COND, 10),
        sharex=True, sharey='row',
    )
    fig1.patch.set_facecolor(dark_bg)
    fig1.suptitle("Trait trajectories (mean \u00b1 1\u03c3 across seeds)",
                  color="#eeeeee", fontsize=11)

    for ci, (cond, r) in enumerate(results):
        h = r["agg_history"]
        steps = h["steps"]
        for ti, (tk, label, color) in enumerate(
            zip(TRAIT_KEYS, TRAIT_LABELS, TRAIT_COLORS)
        ):
            ax = axes1[ti, ci]
            mean  = h[f"mean_{tk}"]
            upper = h[f"mean_{tk}_upper"]
            lower = h[f"mean_{tk}_lower"]
            ax.plot(steps, mean, color=color, lw=1.5)
            ax.fill_between(steps, lower, upper,
                            color=color, alpha=0.2)
            ax.set_ylim(0, 1.5)
            _style(ax,
                   ylabel=label if ci == 0 else "",
                   title=cond.name if ti == 0 else "")
        axes1[-1, ci].set_xlabel("Step", color="#aaaaaa",
                                  fontsize=8)

    fig1.tight_layout()

    # ── Figure 2: Trait distributions at end state ────────────
    fig2, axes2 = plt.subplots(
        4, N_COND,
        figsize=(4 * N_COND, 10),
        sharey='row',
    )
    fig2.patch.set_facecolor(dark_bg)
    fig2.suptitle("Trait distributions at step 1000 (all seeds pooled)",
                  color="#eeeeee", fontsize=11)

    for ci, (cond, r) in enumerate(results):
        traits = r["traits_final"]
        for ti, (tk_idx, label, color) in enumerate(
            zip(range(4), TRAIT_LABELS, TRAIT_COLORS)
        ):
            ax = axes2[ti, ci]
            ax.hist(traits[:, tk_idx], bins=30,
                    color=color, alpha=0.8,
                    range=(0, 1.5))
            _style(ax,
                   ylabel=label if ci == 0 else "",
                   title=cond.name if ti == 0 else "")
            ax.axvline(traits[:, tk_idx].mean(),
                       color="#ffffff", lw=1, linestyle="--",
                       alpha=0.7)

    fig2.tight_layout()

    # ── Figure 3: Population + all traits with std bands ──────
    fig3, axes3 = plt.subplots(
        2, N_COND,
        figsize=(4 * N_COND, 7),
        sharex=True,
    )
    fig3.patch.set_facecolor(dark_bg)
    fig3.suptitle("Population and trait evolution (mean \u00b1 1\u03c3)",
                  color="#eeeeee", fontsize=11)

    for ci, (cond, r) in enumerate(results):
        h = r["agg_history"]
        steps = h["steps"]

        # Top: population
        ax_pop = axes3[0, ci]
        ax_pop.plot(steps, h["total_mean"],
                    color="#ffffff", lw=1.5)
        ax_pop.fill_between(steps,
                            h["total_lower"],
                            h["total_upper"],
                            color="#ffffff", alpha=0.15)
        _style(ax_pop,
               ylabel="Agents" if ci == 0 else "",
               title=cond.name)

        # Bottom: all trait means
        ax_tr = axes3[1, ci]
        for tk, color in zip(TRAIT_KEYS, TRAIT_COLORS):
            mean  = h[f"mean_{tk}"]
            upper = h[f"mean_{tk}_upper"]
            lower = h[f"mean_{tk}_lower"]
            ax_tr.plot(steps, mean, color=color,
                       lw=1.2, label=tk)
            ax_tr.fill_between(steps, lower, upper,
                               color=color, alpha=0.15)
        ax_tr.set_ylim(0, 1.5)
        _style(ax_tr,
               ylabel="Mean trait" if ci == 0 else "")
        if ci == N_COND - 1:
            ax_tr.legend(
                facecolor="#222222", edgecolor="#555555",
                labelcolor="#dddddd", fontsize=7,
                loc="upper right",
            )
        axes3[1, ci].set_xlabel("Step", color="#aaaaaa",
                                 fontsize=8)

    fig3.tight_layout()
    plt.show()


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
        experiment = build_phase2_experiment(cond, seeds=seeds, steps=args.steps)
        print(f"  Running {cond.name} across {len(seeds)} seeds...")
        r = run_phase2_experiment(cond, experiment)
        multi_results.append((cond, r))
        print(
            f"    done — "
            f"pop={r['population_mean']:.0f}\u00b1{r['population_std']:.0f}  "
            f"rw={r['rw_mean']:.3f}\u00b1{r['rw_std']:.3f}  "
            f"noise={r['noise_mean']:.3f}\u00b1{r['noise_std']:.3f}  "
            f"cs={r['cs_mean']:.3f}\u00b1{r['cs_std']:.3f}"
        )

    print_multi_table(multi_results, seeds)

    print("\nGenerating plots...")
    plot_results(multi_results)
    save_results(multi_results, phase="phase2")


if __name__ == "__main__":
    main()


from src.core.phases.reproduce import reproduce  # noqa: F401, E402

_BASE = dict(
    policy_mode="trait",
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
    save_results=True,
)

EXPERIMENTS = {
    "phase2_trait_A": Experiment(
        name="Trait Genome — Baseline Environment",
        description="Trait vector evolution under stable hotspots. Crowd sensitivity dominates.",
        result_id="phase2_trait_A",
        env_config={"drift_step": 1, "noise_rate": 3.0, "noise_magnitude": 2.0},
        **_BASE,
    ),
    "phase2_trait_B": Experiment(
        name="Trait Genome — Fast Drift",
        description="Fast-drifting hotspots. Resource weight rises as tracking becomes worthwhile.",
        result_id="phase2_trait_B",
        env_config={"drift_step": 3, "noise_rate": 3.0, "noise_magnitude": 2.0},
        **_BASE,
    ),
    "phase2_trait_C": Experiment(
        name="Trait Genome — Boom/Bust",
        description="High resource noise. Energy awareness and noise trait both rise.",
        result_id="phase2_trait_C",
        env_config={"drift_step": 1, "noise_rate": 8.0, "noise_magnitude": 4.0},
        **_BASE,
    ),
    "phase2_trait_D": Experiment(
        name="Trait Genome — Combined Volatility",
        description="Maximum volatility. No single strategy dominates — highest cross-seed variance.",
        result_id="phase2_trait_D",
        env_config={"drift_step": 3, "noise_rate": 8.0, "noise_magnitude": 4.0},
        **_BASE,
    ),
}

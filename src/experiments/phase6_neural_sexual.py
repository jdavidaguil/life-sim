"""Phase 6 — Neural Sexual Reproduction experiment.

Agents initialized with warm-start + noise NeuralPolicy genomes.
Reproduction via whole-layer crossover (reproduce_sexual_neural).
Conditions A–D (baseline through combined volatility) tested.

Usage:
    python -m src.experiments.phase6_neural_sexual
    python -m src.experiments.phase6_neural_sexual --steps 1000 --seeds 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np

from src.experiments.base import Experiment
from src.experiments.conditions import CONDITIONS
from src.core.phases import move, consume, decay, die, regenerate, noise, drift
from src.core.phases.reproduce_sexual_neural import reproduce_sexual_neural
from src.core.simulation import Simulation
from src.core.state import SimState
from src.genome.crossover_neural import warm_start_noisy

# ── experiment knobs ──────────────────────────────────────────────────────────
SEEDS: list[int] = [42, 43, 44, 45, 46]
STEPS: int = 1000
# Best config from mating_cost × mating_threshold sweep
MATING_COST: float = 3.0
MATING_THRESHOLD: float = 15.0

# Conditions A–D (indices 0–3); E omitted (steep landscape, different scope)
SELECTED_CONDITIONS = CONDITIONS[:4]

# ── probe input vectors (18 values, normalized) ───────────────────────────────
# NeuralPolicy DIRECTIONS order: NW,N,NE,W,E,SW,S,SE → N is index 1
# Input layout: resource[0:8] | crowd[8:16] | rel_energy[16] | current_res[17]
_NORTH_IDX = 1

_PROBE_NORTH_RICH = np.zeros(18, dtype=np.float32)
_PROBE_NORTH_RICH[_NORTH_IDX] = 1.0           # resource=1.0 at North

_PROBE_NORTH_CROWDED = np.zeros(18, dtype=np.float32)
_PROBE_NORTH_CROWDED[_NORTH_IDX] = 1.0         # resource=1.0 at North
_PROBE_NORTH_CROWDED[8 + _NORTH_IDX] = 1.0     # crowd=1.0 at North


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_phase_list() -> list:
    """Return the standard phase list with reproduce replaced by neural-sexual."""
    return [move, consume, decay, reproduce_sexual_neural, die, regenerate, noise, drift]


def _replace_policies_warm_start(sim: Simulation) -> None:
    """Replace every agent's policy with a warm-start noisy NeuralPolicy."""
    for agent in sim.agents:
        agent.policy = warm_start_noisy(sim.rng, sigma=0.1)


def _probe_agents(agents: list, probe_input: np.ndarray, n: int = 20) -> float:
    """Return mean P(north) over up to *n* sampled agents for *probe_input*.

    Only agents whose policy exposes ``_forward`` (NeuralPolicy) are included.
    Returns 0.0 if no eligible agents are found.
    """
    eligible = [a for a in agents if hasattr(a.policy, "_forward")]
    if not eligible:
        return 0.0
    sample = eligible[:n] if len(eligible) <= n else [
        eligible[i] for i in np.random.default_rng(0).choice(
            len(eligible), size=n, replace=False
        )
    ]
    probs_north = [float(a.policy._forward(probe_input)[_NORTH_IDX]) for a in sample]
    return float(np.mean(probs_north))


# ── Experiment object ────────────────────────────────────────────────────────
# Mirrors the pattern from phase6_sexual.py, adapted for NeuralPolicy agents.

def _init_neural_policies(state: SimState) -> None:
    """Replace every agent's policy with a warm-start NeuralPolicy on step 0."""
    if state.step != 0:
        return
    for agent in state.agents:
        agent.policy = warm_start_noisy(state.rng, sigma=0.1)


def _record_mating_events(state: SimState) -> None:
    """After-die hook: append this step's mating event count to metrics."""
    count = state.scratch.get("mating_events", 0)
    state.metrics.setdefault("mating_events", []).append(count)


EXPERIMENT = Experiment(
    overrides={"reproduce": reproduce_sexual_neural},
    additions={
        "before_move": [_init_neural_policies],
        "after_die": [_record_mating_events],
    },
    steps=STEPS,
    seeds=SEEDS,
)


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    condition: str
    seed: int
    final_pop: int
    total_matings: int
    probe_north_rich: float
    probe_north_crowded: float


# ── single run ────────────────────────────────────────────────────────────────

def _run_one(condition, seed: int, steps: int) -> RunResult:
    env_cfg = {
        "drift_step": condition.drift_step,
        "noise_rate": condition.noise_rate,
        "noise_magnitude": condition.noise_magnitude,
    }
    sim = Simulation(
        width=50,
        height=50,
        initial_agents=100,
        seed=seed,
        env_config=env_cfg,
        policy_mode="neural",
        phases=_build_phase_list(),
    )
    _replace_policies_warm_start(sim)

    total_matings = 0

    for s in range(1, steps + 1):
        sim.step()
        mating_events = sim._state.scratch.get("mating_events", 0)
        total_matings += mating_events

        if s % 10 == 0:
            print(
                f"  [{condition.name}] seed={seed} "
                f"step={s:4d}  pop={sim.agent_count():4d}  "
                f"matings={mating_events:3d}"
            )

    pnr = _probe_agents(sim.agents, _PROBE_NORTH_RICH, n=20)
    pnc = _probe_agents(sim.agents, _PROBE_NORTH_CROWDED, n=20)

    return RunResult(
        condition=condition.name,
        seed=seed,
        final_pop=sim.agent_count(),
        total_matings=total_matings,
        probe_north_rich=pnr,
        probe_north_crowded=pnc,
    )


# ── summary table ─────────────────────────────────────────────────────────────

def _print_summary(results: List[RunResult]) -> None:
    header = (
        f"{'condition':<22}  {'seed':>4}  {'pop':>5}  "
        f"{'matings':>7}  {'probe_north_rich':>16}  {'probe_north_crowded':>19}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.condition:<22}  {r.seed:>4}  {r.final_pop:>5}  "
            f"{r.total_matings:>7}  {r.probe_north_rich:>16.4f}  "
            f"{r.probe_north_crowded:>19.4f}"
        )
    print(sep)

    # Cross-seed means per condition
    print()
    print(f"{'Cross-seed means':<22}  {'':>4}  {'pop':>5}  {'matings':>7}  "
          f"{'probe_north_rich':>16}  {'probe_north_crowded':>19}")
    print(sep)
    for cond in SELECTED_CONDITIONS:
        subset = [r for r in results if r.condition == cond.name]
        if not subset:
            continue
        pops = [r.final_pop for r in subset]
        mats = [r.total_matings for r in subset]
        pnrs = [r.probe_north_rich for r in subset]
        pncs = [r.probe_north_crowded for r in subset]
        print(
            f"{cond.name:<22}  {'μ':>4}  {np.mean(pops):>5.1f}  "
            f"{np.mean(mats):>7.1f}  {np.mean(pnrs):>16.4f}  {np.mean(pncs):>19.4f}"
        )
        print(
            f"{'':22}  {'σ':>4}  {np.std(pops):>5.1f}  "
            f"{np.std(mats):>7.1f}  {np.std(pnrs):>16.4f}  {np.std(pncs):>19.4f}"
        )
    print(sep)


# ── entry point ───────────────────────────────────────────────────────────────

def main(steps: int = STEPS, seeds: List[int] = SEEDS) -> List[RunResult]:
    results: List[RunResult] = []
    for cond in SELECTED_CONDITIONS:
        print(f"\n=== Condition: {cond.name} ===")
        for seed in seeds:
            print(f"\n--- seed {seed} ---")
            result = _run_one(cond, seed, steps)
            results.append(result)
    _print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS,
                        help=f"Steps per run (default {STEPS})")
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help=f"Number of seeds starting at 42 (default {len(SEEDS)})")
    args = parser.parse_args()

    seed_list = list(range(42, 42 + args.seeds))
    main(steps=args.steps, seeds=seed_list)

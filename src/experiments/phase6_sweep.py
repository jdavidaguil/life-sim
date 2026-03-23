"""Phase 6 — parameter sweep over mating cost × mating threshold.

Sweep grid
----------
mating_cost      : [3.0, 6.0, 9.0]
mating_threshold : [10.0, 12.0, 15.0]

For each 9 combinations run seeds [42, 43, 44, 45, 46] for 1000 steps each and
record final population, total mating events, and mean trait values.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.core.agent import Agent
from src.core.grid import Grid
from src.core.loop import SimulationLoop
from src.core.phases import DEFAULT_PHASES
from src.core.phases.reproduce_sexual import reproduce_sexual
from src.core.policy import TraitPolicy
from src.core.state import SimState
from src.experiments.phase6_sexual import _record_mating_events

# ── Sweep axes ────────────────────────────────────────────────────────────────
MATING_COSTS: List[float] = [3.0, 6.0, 9.0]
MATING_THRESHOLDS: List[float] = [10.0, 12.0, 15.0]
SEEDS: List[int] = [42, 43, 44, 45, 46]
STEPS: int = 1000
INITIAL_AGENTS: int = 100
GRID_SIZE: int = 50


# ── Per-run result ─────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    seed: int
    mating_cost: float
    mating_threshold: float
    final_pop: int
    total_matings: int
    mean_rw: float
    mean_cs: float
    mean_noise: float
    mean_ea: float


def _mean_traits(state: SimState) -> tuple[float, float, float, float]:
    """Return (rw, cs, noise, ea) means over all TraitPolicy agents."""
    tas = [a for a in state.agents if hasattr(a.policy, "traits")]
    if not tas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t = np.array([a.policy.traits for a in tas], dtype=np.float32)
    return float(t[:, 0].mean()), float(t[:, 1].mean()), float(t[:, 2].mean()), float(t[:, 3].mean())


def _run_seed(seed: int, phase_list: list, steps: int) -> SimState:
    """Run one seed and return the final SimState."""
    rng = np.random.default_rng(seed)
    grid = Grid(width=GRID_SIZE, height=GRID_SIZE, rng=rng)
    agents: List[Agent] = [
        Agent(id=i, x=int(rng.integers(0, GRID_SIZE)), y=int(rng.integers(0, GRID_SIZE)),
              policy=TraitPolicy(rng=rng))
        for i in range(INITIAL_AGENTS)
    ]
    state = SimState(grid=grid, agents=agents, rng=rng, step=0,
                     metrics={"_next_id": INITIAL_AGENTS})
    SimulationLoop(phase_list).run(state, steps)
    return state


def _build_phase_list(mating_cost: float, mating_threshold: float) -> list:
    """Return phase list with reproduce_sexual bound to the given parameters."""
    bound = functools.partial(
        reproduce_sexual,
        mating_cost=mating_cost,
        mating_threshold=mating_threshold,
    )
    # Give the partial a recognisable __name__ so overrides work by name.
    bound.__name__ = "reproduce_sexual"  # type: ignore[attr-defined]

    return [
        bound if p.__name__ == "reproduce" else p
        for p in DEFAULT_PHASES
    ] + [_record_mating_events]


def run_sweep(
    costs: List[float] | None = None,
    thresholds: List[float] | None = None,
    seeds: List[int] | None = None,
    steps: int | None = None,
) -> List[RunResult]:
    """Execute the full sweep and return one RunResult per (cost, threshold, seed).

    Optional parameters override the module-level defaults, which is useful for
    testing with smaller grids.
    """
    _costs = costs if costs is not None else MATING_COSTS
    _thresholds = thresholds if thresholds is not None else MATING_THRESHOLDS
    _seeds = seeds if seeds is not None else SEEDS
    _steps = steps if steps is not None else STEPS

    results: List[RunResult] = []
    combos = [(c, th) for c in _costs for th in _thresholds]
    total = len(combos) * len(_seeds)
    done = 0

    for cost, threshold in combos:
        phase_list = _build_phase_list(cost, threshold)
        for seed in _seeds:
            state = _run_seed(seed, phase_list, _steps)
            total_matings = sum(state.metrics.get("mating_events", []))
            rw, cs, noise, ea = _mean_traits(state)
            results.append(RunResult(
                seed=seed,
                mating_cost=cost,
                mating_threshold=threshold,
                final_pop=len(state.agents),
                total_matings=total_matings,
                mean_rw=rw,
                mean_cs=cs,
                mean_noise=noise,
                mean_ea=ea,
            ))
            done += 1
            print(f"  [{done:2d}/{total}] cost={cost}  threshold={threshold}  "
                  f"seed={seed}  pop={len(state.agents)}  matings={total_matings}")

    return results


# ── Aggregation helpers ────────────────────────────────────────────────────────

@dataclass
class ComboStats:
    mating_cost: float
    mating_threshold: float
    pop_mean: float
    pop_std: float
    matings_mean: float
    rw_mean: float
    cs_mean: float
    cs_std: float
    noise_mean: float
    ea_mean: float


def _aggregate(results: List[RunResult]) -> List[ComboStats]:
    """Aggregate per-seed results into one ComboStats per (cost, threshold)."""
    from itertools import groupby
    key = lambda r: (r.mating_cost, r.mating_threshold)
    sorted_results = sorted(results, key=key)
    stats: List[ComboStats] = []
    for (cost, threshold), group in groupby(sorted_results, key=key):
        rows = list(group)
        pops = np.array([r.final_pop for r in rows], dtype=float)
        mates = np.array([r.total_matings for r in rows], dtype=float)
        rws = np.array([r.mean_rw for r in rows], dtype=float)
        css = np.array([r.mean_cs for r in rows], dtype=float)
        noises = np.array([r.mean_noise for r in rows], dtype=float)
        eas = np.array([r.mean_ea for r in rows], dtype=float)
        stats.append(ComboStats(
            mating_cost=cost,
            mating_threshold=threshold,
            pop_mean=float(pops.mean()),
            pop_std=float(pops.std()),
            matings_mean=float(mates.mean()),
            rw_mean=float(rws.mean()),
            cs_mean=float(css.mean()),
            cs_std=float(css.std()),
            noise_mean=float(noises.mean()),
            ea_mean=float(eas.mean()),
        ))
    return stats


# ── Pretty-print tables ───────────────────────────────────────────────────────

def _print_main_table(stats: List[ComboStats]) -> None:
    ordered = sorted(stats, key=lambda s: s.pop_mean, reverse=True)
    W = 88
    print("=" * W)
    print("MAIN SUMMARY  (sorted by population mean ↓)")
    print("-" * W)
    print(f"{'cost':>6}  {'thresh':>7}  │  "
          f"{'pop_mean':>8}  {'pop_std':>7}  {'mat_mean':>8}  │  "
          f"{'rw':>6}  {'cs':>6}  {'noise':>7}  {'ea':>6}")
    print("-" * W)
    for s in ordered:
        print(
            f"{s.mating_cost:>6.1f}  {s.mating_threshold:>7.1f}  │  "
            f"{s.pop_mean:>8.1f}  {s.pop_std:>7.1f}  {s.matings_mean:>8.1f}  │  "
            f"{s.rw_mean:>6.3f}  {s.cs_mean:>6.3f}  {s.noise_mean:>7.3f}  {s.ea_mean:>6.3f}"
        )
    print("=" * W)


def _print_cs_table(stats: List[ComboStats]) -> None:
    ordered = sorted(stats, key=lambda s: s.pop_mean, reverse=True)
    W = 46
    print()
    print("=" * W)
    print("CROWD SENSITIVITY  (sorted by population mean ↓)")
    print("-" * W)
    print(f"{'cost':>6}  {'thresh':>7}  │  {'cs_mean':>8}  {'cs_std':>7}")
    print("-" * W)
    for s in ordered:
        print(f"{s.mating_cost:>6.1f}  {s.mating_threshold:>7.1f}  │  "
              f"{s.cs_mean:>8.3f}  {s.cs_std:>7.3f}")
    print("=" * W)


if __name__ == "__main__":
    print("=== Phase 6 — Mating Cost × Threshold Sweep ===")
    print(f"Grid: {MATING_COSTS} × {MATING_THRESHOLDS}")
    print(f"Seeds: {SEEDS}  Steps: {STEPS}")
    print()

    raw = run_sweep()
    stats = _aggregate(raw)
    print()
    _print_main_table(stats)
    _print_cs_table(stats)

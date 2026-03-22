"""Phase 6 — sexual reproduction experiment."""

from __future__ import annotations

from src.core.phases.reproduce_sexual import reproduce_sexual
from src.core.policy import TraitPolicy
from src.core.state import SimState
from src.experiments.base import Experiment

import numpy as np


def _record_mating_events(state: SimState) -> None:
    """After-die hook: append this step's mating event count to metrics."""
    count = state.scratch.get("mating_events", 0)
    state.metrics.setdefault("mating_events", []).append(count)


def _print_summary(state: SimState) -> None:
    """After-die hook: print a per-step summary every 10 steps."""
    if state.step % 10 != 0:
        return
    population = len(state.agents)
    mating = state.scratch.get("mating_events", 0)
    trait_agents = [a for a in state.agents if isinstance(a.policy, TraitPolicy)]
    if trait_agents:
        traits = np.array([a.policy.traits for a in trait_agents], dtype=np.float32)
        mean_rw   = float(traits[:, 0].mean())
        mean_cs   = float(traits[:, 1].mean())
        mean_noise = float(traits[:, 2].mean())
        mean_ea   = float(traits[:, 3].mean())
        print(
            f"[step {state.step:4d}] "
            f"pop={population:4d}  "
            f"matings={mating:3d}  "
            f"rw={mean_rw:.3f}  "
            f"cs={mean_cs:.3f}  "
            f"noise={mean_noise:.3f}  "
            f"ea={mean_ea:.3f}"
        )
    else:
        print(
            f"[step {state.step:4d}] "
            f"pop={population:4d}  "
            f"matings={mating:3d}  "
            f"(no trait agents)"
        )


EXPERIMENT = Experiment(
    overrides={"reproduce": reproduce_sexual},
    additions={"after_die": [_record_mating_events, _print_summary]},
    steps=1000,
    seeds=[42, 43, 44, 45, 46],
)


def _final_traits(state: SimState) -> np.ndarray | None:
    """Return (N, 4) trait array for all TraitPolicy agents, or None."""
    tas = [a for a in state.agents if hasattr(a.policy, "traits")]
    if not tas:
        return None
    return np.array([a.policy.traits for a in tas], dtype=np.float32)


if __name__ == "__main__":
    print("=== Phase 6 — Sexual Reproduction Experiment ===")
    print(f"Seeds: {EXPERIMENT.seeds}  Steps: {EXPERIMENT.steps}")
    print(f"Phase list: {[p.__name__ for p in EXPERIMENT.build_phase_list()]}")
    print()

    rows: list[dict] = []
    for seed, state in zip(EXPERIMENT.seeds, EXPERIMENT.run()):
        total_matings = sum(state.metrics.get("mating_events", []))
        final_pop = len(state.agents)
        traits = _final_traits(state)
        mean_rw = mean_cs = mean_noise = mean_ea = float("nan")
        if traits is not None:
            mean_rw    = float(traits[:, 0].mean())
            mean_cs    = float(traits[:, 1].mean())
            mean_noise = float(traits[:, 2].mean())
            mean_ea    = float(traits[:, 3].mean())
        rows.append(dict(
            seed=seed,
            pop=final_pop,
            matings=total_matings,
            rw=mean_rw,
            cs=mean_cs,
            noise=mean_noise,
            ea=mean_ea,
        ))
        print(f"  Seed {seed} done: pop={final_pop}  matings={total_matings}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 74)
    print(f"{'Seed':>6}  {'Pop':>6}  {'Matings':>8}  {'rw':>6}  {'cs':>6}  {'noise':>7}  {'ea':>6}")
    print("-" * 74)
    for r in rows:
        print(
            f"{r['seed']:>6}  {r['pop']:>6}  {r['matings']:>8}  "
            f"{r['rw']:>6.3f}  {r['cs']:>6.3f}  {r['noise']:>7.3f}  {r['ea']:>6.3f}"
        )
    print("-" * 74)

    pops    = np.array([r["pop"]    for r in rows], dtype=float)
    rws     = np.array([r["rw"]     for r in rows], dtype=float)
    css     = np.array([r["cs"]     for r in rows], dtype=float)
    noises  = np.array([r["noise"]  for r in rows], dtype=float)
    eas     = np.array([r["ea"]     for r in rows], dtype=float)

    print(
        f"{'mean':>6}  {pops.mean():>6.1f}  {'':>8}  "
        f"{rws.mean():>6.3f}  {css.mean():>6.3f}  {noises.mean():>7.3f}  {eas.mean():>6.3f}"
    )
    print(
        f"{'std':>6}  {pops.std():>6.1f}  {'':>8}  "
        f"{rws.std():>6.3f}  {css.std():>6.3f}  {noises.std():>7.3f}  {eas.std():>6.3f}"
    )
    print("=" * 74)

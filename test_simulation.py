#!/usr/bin/env python
"""Test: run simulation twice with fixed seed and verify consistency."""

import argparse

from src.core.simulation import Simulation


def run_simulation(seed: int, steps: int = 50) -> dict:
    """Run a short simulation and return final state metrics."""
    sim = Simulation(width=30, height=30, initial_agents=50, seed=seed)
    
    for step in range(steps):
        sim.step()
    
    return {
        "agents": len(sim.agents),
        "reproductions": sim.reproductions_total,
        "current_step": sim.current_step,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation determinism check.")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run (default: 200)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = parser.parse_args()

    seed_label = str(args.seed) if args.seed is not None else "random"
    print("=" * 60)
    print(f"Running simulation with seed={seed_label}, {args.steps} steps")
    print("=" * 60)

    result1 = run_simulation(seed=args.seed, steps=args.steps)
    print("\nRun 1:")
    print(f"  Final agents: {result1['agents']}")
    print(f"  Total reproductions: {result1['reproductions']}")
    print(f"  Steps completed: {result1['current_step']}")

    if args.seed is not None:
        result2 = run_simulation(seed=args.seed, steps=args.steps)
        print("\nRun 2:")
        print(f"  Final agents: {result2['agents']}")
        print(f"  Total reproductions: {result2['reproductions']}")
        print(f"  Steps completed: {result2['current_step']}")

        print("\nConsistency check:")
        if result1 == result2:
            print("✅ Results match! Simulation is deterministic.")
        else:
            print("❌ Results differ! There may be a randomization issue.")
            print(f"  Diff: {set((k, (result1[k], result2[k])) for k in result1 if result1[k] != result2[k])}")
        print(f"  Diff: {set((k, (result1[k], result2[k])) for k in result1 if result1[k] != result2[k])}")

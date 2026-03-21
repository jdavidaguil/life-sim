#!/usr/bin/env python
"""Test: run simulation twice with fixed seed and verify consistency."""

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
    print("=" * 60)
    print("Running simulation twice with fixed seed (seed=42)")
    print("=" * 60)
    
    result1 = run_simulation(seed=42, steps=50)
    print("\nRun 1:")
    print(f"  Final agents: {result1['agents']}")
    print(f"  Total reproductions: {result1['reproductions']}")
    print(f"  Steps completed: {result1['current_step']}")
    
    result2 = run_simulation(seed=42, steps=50)
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

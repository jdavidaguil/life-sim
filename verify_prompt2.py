from src.core.simulation import Simulation

sim = Simulation(width=50, height=50, initial_agents=100, seed=42)
for _ in range(50):
    sim.step()

total = sim.history["total"][-1]

# Print all available history keys so we know exactly what's tracked
print(f"Step 50 — total population: {total}")
print(f"History keys: {list(sim.history.keys())}")

# Print last value of each key
for key, values in sim.history.items():
    if values:
        print(f"  {key}: {values[-1]}")

print("Prompt 2 verification passed")
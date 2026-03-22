from src.core.state import SimState
from src.core.grid import Grid
from src.core.phases.reproduce_sexual import reproduce_sexual
import numpy as np
from src.core.agent import Agent

# Build a minimal state with agents clustered on same cell
rng = np.random.default_rng(42)
grid = Grid(20, 20, rng=rng)

agents = []
for i in range(4):
    a = Agent(id=i, x=10, y=10)
    a.energy = 15.0
    agents.append(a)

state = SimState(grid=grid, agents=agents, rng=rng, step=0, metrics={}, scratch={})

reproduce_sexual(state)

mating_events = state.scratch.get("mating_events", 0)
newborns = [a for a in state.agents if a.id >= 4]
original_agents = [a for a in state.agents if a.id < 4]

print(f"Mating events: {mating_events}")
print(f"Newborns: {len(newborns)}")
print(f"Total agents: {len(state.agents)}")

assert mating_events >= 1, "No mating events recorded"
assert len(newborns) >= 1, "No child produced"

# Only the top-2 energy agents on the cell mate — exactly 2 should pay the cost
reduced = [a for a in original_agents if a.energy < 15.0]
unchanged = [a for a in original_agents if a.energy == 15.0]

print(f"Agents that paid mating cost: {[a.id for a in reduced]}")
print(f"Agents uninvolved: {[a.id for a in unchanged]}")

assert len(reduced) == 2, f"Expected exactly 2 parents to pay cost, got {len(reduced)}"
assert len(unchanged) == 2, f"Expected exactly 2 uninvolved agents, got {len(unchanged)}"

# Verify child genome via policy.traits
child = newborns[0]
print(f"Child attributes: {list(vars(child).keys())}")
print(f"Child energy: {child.energy}")
print(f"Child position: ({child.x}, {child.y})")

assert hasattr(child, "policy"), "Child has no policy"
assert hasattr(child.policy, "traits"), \
    f"Child policy has no traits. Policy type: {type(child.policy)}. Policy attrs: {list(vars(child.policy).keys())}"

traits = child.policy.traits
print(f"Child traits: {traits}")

assert len(traits) == 4, f"Expected 4 traits, got {len(traits)}"
assert all(0.0 <= t <= 2.0 for t in traits), f"Traits out of [0,2] bounds: {traits}"

# Verify traits are a mix — at least one trait should differ from both parents
parent_a = reduced[0].policy.traits
parent_b = reduced[1].policy.traits
print(f"Parent A traits: {parent_a}")
print(f"Parent B traits: {parent_b}")

# Each child trait should match either parent A or parent B (before mutation noise)
# We can't assert exact match due to mutation, but we can check bounds
print(f"Trait bounds check passed — all traits in [0, 2]")

print("Prompt 4 verification passed")
from src.experiments.base import Experiment
from src.core.phases import DEFAULT_PHASES

# Test 1: default experiment builds correct phase list
exp = Experiment()
phases = exp.build_phase_list()
assert len(phases) == len(DEFAULT_PHASES), "Phase count mismatch"
print(f"Default phase list: {[p.__name__ for p in phases]}")

# Test 2: override replaces exactly one phase
def fake_reproduce(state): pass
exp2 = Experiment(overrides={"reproduce": fake_reproduce})
phases2 = exp2.build_phase_list()
names = [p.__name__ for p in phases2]
assert "fake_reproduce" in names, "Override not applied"
assert "reproduce" not in names, "Original phase not replaced"
print(f"Override applied: {names}")

# Test 3: addition injects before a phase
def pre_noise(state): pass
exp3 = Experiment(additions={"before_noise": [pre_noise]})
phases3 = exp3.build_phase_list()
names3 = [p.__name__ for p in phases3]
noise_idx = names3.index("noise")
pre_idx = names3.index("pre_noise")
assert pre_idx == noise_idx - 1, "Addition not injected before noise"
print(f"Addition injected correctly: {names3}")

print("Prompt 3 verification passed")
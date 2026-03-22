import numpy as np
from src.core.policy import NeuralPolicy
from src.genome.crossover_neural import crossover_neural, warm_start_noisy

rng = np.random.default_rng(42)

# Test warm_start_noisy produces unique functional genomes
p1 = warm_start_noisy(rng, sigma=0.1)
p2 = warm_start_noisy(rng, sigma=0.1)

assert isinstance(p1, NeuralPolicy), "warm_start_noisy must return NeuralPolicy"
assert len(p1.genome) == 224, f"Expected 224 weights, got {len(p1.genome)}"
assert not np.array_equal(p1.genome, p2.genome), "Two noisy warm starts must differ"
print(f"Warm start 1 norm: {np.linalg.norm(p1.genome):.3f}")
print(f"Warm start 2 norm: {np.linalg.norm(p2.genome):.3f}")

# Test crossover produces a child distinct from both parents
child = crossover_neural(p1, p2, rng)
assert isinstance(child, NeuralPolicy), "crossover must return NeuralPolicy"
assert len(child.genome) == 224, f"Expected 224 weights, got {len(child.genome)}"
assert not np.array_equal(child.genome, p1.genome), "Child must differ from parent A"
assert not np.array_equal(child.genome, p2.genome), "Child must differ from parent B"
print(f"Child genome norm: {np.linalg.norm(child.genome):.3f}")

# Test child via forward pass
W1 = child.genome[0:144].reshape(18, 8)
b1 = child.genome[144:152]
W2 = child.genome[152:216].reshape(8, 8)
b2 = child.genome[216:224]
x = np.zeros(18, dtype=np.float32)
x[0] = 1.0
h = np.maximum(0.0, x @ W1 + b1)
logits = h @ W2 + b2
probs = np.exp(logits - logits.max())
probs /= probs.sum()
print(f"Child forward pass probabilities: {np.round(probs, 3)}")
assert abs(probs.sum() - 1.0) < 1e-5, "Probabilities must sum to 1"

print("Prompt 5a verification passed")
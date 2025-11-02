#!/usr/bin/env python
"""
Debug JAX RNG to see if random keys are actually different across trials.
"""

import jax
import jax.numpy as jnp
from jax import vmap

print("Testing JAX RNG key splitting...")

# Test 1: Basic RNG splitting
rng = jax.random.PRNGKey(42)
rng_keys = jax.random.split(rng, 10)

print("\n1. Split 10 keys:")
for i in range(5):
    print(f"   Key {i}: {rng_keys[i]}")

# Test 2: Generate random numbers with each key
print("\n2. Generate random numbers:")
for i in range(5):
    val = jax.random.normal(rng_keys[i], (3,))
    print(f"   Key {i} -> {val}")

# Test 3: Simulate what we do in the code
def trial_with_noise(rng_key):
    """Simulates one trial initialization."""
    rng_key, noise_key = jax.random.split(rng_key)

    # Initial noise
    init_noise = jax.random.normal(noise_key, (5,)) * 0.02

    # Step noise
    rng_key, step_key = jax.random.split(rng_key)
    step_noise = jax.random.normal(step_key, (5,))

    return init_noise, step_noise

# Test without vmap
print("\n3. Manual loop (no vmap):")
for i in range(3):
    init, step = trial_with_noise(rng_keys[i])
    print(f"   Trial {i}: init={init[0]:.6f}, step={step[0]:.6f}")

# Test with vmap (what we actually use)
print("\n4. With vmap (what code uses):")
batched_trial = vmap(trial_with_noise)
init_batch, step_batch = batched_trial(rng_keys[:10])

for i in range(5):
    print(f"   Trial {i}: init={init_batch[i, 0]:.6f}, step={step_batch[i, 0]:.6f}")

print("\n5. Check if all trials got same noise:")
print(f"   All init noise same? {jnp.allclose(init_batch[0], init_batch[1])}")
print(f"   Init noise variance: {jnp.var(init_batch[:, 0]):.6f}")

# Test 4: What about inside scan?
def trial_with_scan(rng_key):
    """Simulates dynamics loop with scan."""
    init_state = jax.random.normal(rng_key, (5,)) * 0.02

    def step_fn(carry, _):
        state, rng = carry
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, state.shape)
        state = state + noise * 0.001
        return (state, rng), state

    (final_state, _), trajectory = jax.lax.scan(
        step_fn,
        (init_state, rng_key),
        None,
        length=10
    )
    return final_state

print("\n6. Test with scan (full simulation):")
batched_scan = vmap(trial_with_scan)
final_states = batched_scan(rng_keys[:10])

for i in range(5):
    print(f"   Trial {i}: final={final_states[i, 0]:.6f}")

print(f"\n   All final states same? {jnp.allclose(final_states[0], final_states[1])}")
print(f"   Final state variance: {jnp.var(final_states[:, 0]):.6f}")

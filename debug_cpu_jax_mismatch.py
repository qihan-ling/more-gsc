#!/usr/bin/env python
"""
Detailed debugging script to find the mismatch between CPU and JAX.
"""

import gsc
import numpy as np
import jax.numpy as jnp

print("="*70)
print("Detailed Debugging: CPU vs JAX")
print("="*70)

net = gsc.load_model('g1_model.pkl')

# Extract parameters
from gsc import _extract_net_params_for_jax
params = _extract_net_params_for_jax(net)

print("\n1. Check extracted parameters:")
print(f"   q_init: {params['q_init']}")
print(f"   q_max: {params['q_max']}")
print(f"   q_rate: {params['q_rate']}")
print(f"   dt_init: {params['dt_init']}")
print(f"   T_init: {params['T_init']}")
print(f"   init_noise_mag: {params['init_noise_mag']}")

print(f"\n2. Check S matrix:")
print(f"   S is identity? {np.allclose(net.S, np.eye(net.S.shape[0]))}")
print(f"   S diagonal: {np.diag(net.S)[:5]}")
print(f"   S off-diagonal sample: {net.S[0, 1:6]}")

print(f"\n3. Check C matrix:")
print(f"   C is identity? {np.allclose(net.C, np.eye(net.C.shape[0]))}")
print(f"   C diagonal: {np.diag(net.C)[:5]}")
print(f"   C off-diagonal sample: {net.C[0, 1:6]}")

print(f"\n4. Run one step manually (CPU style):")
# Simulate one step of CPU dynamics
actC = net.ep + np.random.normal(0, net.train_opts['init_noise_mag'], net.num_bindings)
q = net.opts['q_init'] * np.ones(net.num_roles)
T = net.opts['T_init']
dt = net.opts['dt_init']

print(f"   Initial actC range: [{actC.min():.3f}, {actC.max():.3f}]")
print(f"   Initial actC sample: {actC[:5]}")

# One gradient step (conceptual coordinates)
actCmat = actC.reshape((net.num_fillers, net.num_roles), order='F')
extC = np.zeros(net.num_bindings)

# Compute HGradC
hgrad_g = net.WC @ actC + net.bC + extC
hgrad_b = net.opts['bowl_strength'] * (net.opts['bowl_center'] - actC)
q_extended = np.repeat(q, net.num_fillers)
hgrad_q0 = -2 * q_extended * actC * (1 - actC) * (1 - 2 * actC)
ssq = np.sum(actCmat ** 2, axis=0)
ssq_extended = np.repeat(ssq - 1, net.num_fillers)
hgrad_q1 = -4 * net.opts['m'] * actC * ssq_extended

HGradC_val = hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1
gradC = net.scale_constants * net.S.dot(HGradC_val)

print(f"   HGradC range: [{HGradC_val.min():.3f}, {HGradC_val.max():.3f}]")
print(f"   gradC range: [{gradC.min():.3f}, {gradC.max():.3f}]")

# Update
actC_new = actC + dt * gradC

# Add noise
noise = np.sqrt(2 * T * dt) * np.random.randn(net.num_units)
noiseC = np.sqrt(net.scale_constants) * net.C.dot(noise)
actC_final = actC_new + noiseC

print(f"   After gradient step: [{actC_new.min():.3f}, {actC_new.max():.3f}]")
print(f"   Noise range: [{noiseC.min():.3f}, {noiseC.max():.3f}]")
print(f"   Final actC range: [{actC_final.min():.3f}, {actC_final.max():.3f}]")

print(f"\n5. Check duration calculation:")
duration = np.max(net.opts['q_max'] - q) / net.opts['q_rate']
num_steps = int(np.ceil(duration / dt))
print(f"   q_init: {net.opts['q_init']}")
print(f"   q_max: {net.opts['q_max']}")
print(f"   Duration: {duration:.3f}")
print(f"   dt: {dt}")
print(f"   Num steps: {num_steps}")
print(f"   Total time: {num_steps * dt:.3f}")

print(f"\n6. Run actual CPU trial:")
prefix = []
net.reset(mu=net.ep, sd=net.train_opts['init_noise_mag'])
net.run_wrapup(update_q_discrete=False, log_trace=False, clear_input=True)
gp = net.read_grid_point()
print(f"   Final actC range: [{net.actC.min():.3f}, {net.actC.max():.3f}]")
print(f"   Final actC sample: {net.actC[:10]}")
print(f"   Final q: {net.q}")
print(f"   Grid point (first 3): {gp[:3]}")

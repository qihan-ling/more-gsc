"""
Pinpoint where dynamics diverge between speedup and speedup_sap
by comparing step-by-step computation
"""

import numpy as np

print("="*70)
print("PINPOINTING DYNAMICS DIVERGENCE")
print("="*70)

# Load both models
import only_gscnet_speedup as gsc1
net1 = gsc1.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

import only_gscnet_speedup_sap as gsc2
net2 = gsc2.load_model('ds_jax_sap_test_on_g1_model.pkl')

# Set up identical sentence
test_words = ['N', 'Vi', 'P', 'N']

print("\nProcessing sentence: N Vi P N")
print("Both models use same seed for reset")

# Process words to set external inputs (identical for both)
for net in [net1, net2]:
    for wi, word in enumerate(test_words):
        net.run_word(word, wi + 1, log_trace=False)

print("\n" + "="*70)
print("BEFORE run_wrapup() - Check initial states")
print("="*70)

# Reset both with same seed
np.random.seed(1024)
net1.reset(mu=net1.ep, sd=0.01)

np.random.seed(1024)
net2.reset(mu=net2.ep, sd=0.01)

print(f"\nInitial actC (first 10 values):")
print(f"  Speedup: {net1.actC[:10]}")
print(f"  SAP:     {net2.actC[:10]}")
if np.allclose(net1.actC, net2.actC):
    print(f"  ✓ Initial activations MATCH")
else:
    print(f"  ❌ Initial activations DIFFER!")
    diff = np.abs(net1.actC - net2.actC)
    print(f"  Max diff: {np.max(diff):.10f}")

print(f"\nInitial extC (non-zero count):")
print(f"  Speedup: {np.count_nonzero(net1.extC)} non-zero")
print(f"  SAP:     {np.count_nonzero(net2.extC)} non-zero")

# Check scale_constants
print(f"\nScale constants:")
print(f"  Speedup: {net1.scale_constants[:5]}")
print(f"  SAP:     {net2.scale_constants[:5]}")
if np.allclose(net1.scale_constants, net2.scale_constants):
    print(f"  ✓ Scale constants MATCH")
else:
    print(f"  ❌ Scale constants DIFFER!")

# Check other parameters
print(f"\nDynamics parameters:")
print(f"  Speedup - T: {net1.T}, dt: {net1.dt}, bowl_strength: {net1.opts['bowl_strength']}")
print(f"  SAP -     T: {net2.T}, dt: {net2.dt}, bowl_strength: {net2.opts['bowl_strength']}")

print("\n" + "="*70)
print("FIRST DYNAMICS STEP - Compute harmony gradient")
print("="*70)

# Compute HGradC for both
hgrad1 = net1.HGradC(net1.actC, net1.q)
hgrad2 = net2.HGradC(net2.actC, net2.q)

print(f"\nHarmony gradient (first 10 values):")
print(f"  Speedup: {hgrad1[:10]}")
print(f"  SAP:     {hgrad2[:10]}")

if np.allclose(hgrad1, hgrad2, atol=1e-6):
    print(f"  ✓ Harmony gradients MATCH")
else:
    print(f"  ❌ Harmony gradients DIFFER!")
    diff = np.abs(hgrad1 - hgrad2)
    print(f"  Max diff: {np.max(diff):.10f}")
    print(f"  Mean diff: {np.mean(diff):.10f}")

    # Find where they differ most
    max_idx = np.argmax(diff)
    print(f"\n  Largest difference at binding {max_idx}:")
    print(f"    Binding name: {net1.binding_names[max_idx]}")
    print(f"    Speedup hgrad: {hgrad1[max_idx]:.10f}")
    print(f"    SAP hgrad:     {hgrad2[max_idx]:.10f}")

print("\n" + "="*70)
print("CHECK HGradC COMPONENTS")
print("="*70)

# Break down HGradC computation
print("\nComputing HGradC components separately...")

# Component 1: WC.dot(actC)
wc_dot1 = net1.WC.dot(net1.actC)
wc_dot2 = net2.WC.dot(net2.actC)

print(f"\nWC.dot(actC) (first 10):")
print(f"  Speedup: {wc_dot1[:10]}")
print(f"  SAP:     {wc_dot2[:10]}")
if np.allclose(wc_dot1, wc_dot2, atol=1e-6):
    print(f"  ✓ WC.dot(actC) MATCH")
else:
    print(f"  ❌ WC.dot(actC) DIFFER!")
    diff = np.abs(wc_dot1 - wc_dot2)
    print(f"  Max diff: {np.max(diff):.10f}")

# Component 2: bC
print(f"\nbC (first 10):")
print(f"  Speedup: {net1.bC[:10]}")
print(f"  SAP:     {net2.bC[:10]}")

# Component 3: extC
print(f"\nextC (first 10):")
print(f"  Speedup: {net1.extC[:10]}")
print(f"  SAP:     {net2.extC[:10]}")

# Component 4: Bowl term
bowl1 = net1.opts['bowl_strength'] * (net1.opts['bowl_center'] - net1.actC)
bowl2 = net2.opts['bowl_strength'] * (net2.opts['bowl_center'] - net2.actC)

print(f"\nBowl term (first 10):")
print(f"  Speedup: {bowl1[:10]}")
print(f"  SAP:     {bowl2[:10]}")
if np.allclose(bowl1, bowl2, atol=1e-6):
    print(f"  ✓ Bowl terms MATCH")
else:
    print(f"  ❌ Bowl terms DIFFER!")

# Component 5: q-dependent terms
q_term1_base = -2 * net1.extend_rvec(rvec=net1.q) * net1.actC * (1 - net1.actC) * (1 - 2 * net1.actC)
q_term2_base = -2 * net2.extend_rvec(rvec=net2.q) * net2.actC * (1 - net2.actC) * (1 - 2 * net2.actC)

print(f"\nQ-term (first 10):")
print(f"  Speedup: {q_term1_base[:10]}")
print(f"  SAP:     {q_term2_base[:10]}")
if np.allclose(q_term1_base, q_term2_base, atol=1e-6):
    print(f"  ✓ Q-terms MATCH")
else:
    print(f"  ❌ Q-terms DIFFER!")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("""
If HGradC differs, check which component differs:
  - WC.dot(actC): Problem with matrix-vector multiplication
  - Bowl term: Problem with bowl_strength or bowl_center
  - Q-term: Problem with extend_rvec or q computation
  - Something else: Check scale_constants, vec2mat, etc.
""")

print("="*70)

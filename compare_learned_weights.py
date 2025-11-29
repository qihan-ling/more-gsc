"""
Compare learned weights between speedup and speedup_sap models
to identify what's different in the learned representations
"""

import numpy as np

print("="*70)
print("COMPARING LEARNED WEIGHTS BETWEEN MODELS")
print("="*70)

# Load both models
print("\nLoading models...")

import only_gscnet_speedup as gsc1
net1 = gsc1.load_model('g1_ds_speedup_model_copy_trainjax.pkl')
print(f"✓ Loaded speedup model (correct learning)")

import only_gscnet_speedup_sap as gsc2
net2 = gsc2.load_model('ds_jax_sap_test_on_g1_model.pkl')
print(f"✓ Loaded SAP model (wrong learning)")

print("\n" + "="*70)
print("COMPARING BIAS VALUES (bC)")
print("="*70)

# Find the binding for *@:1/(3,2) - the wrongly activated one
binding_name = '*@:1/(3,2)'
idx1 = net1.binding_names.index(binding_name)
idx2 = net2.binding_names.index(binding_name)

print(f"\nBinding: {binding_name} (wrongly activated in SAP)")
print(f"  Speedup bC[{idx1}]: {net1.bC[idx1]:.6f}")
print(f"  SAP bC[{idx2}]:     {net2.bC[idx2]:.6f}")
print(f"  Difference: {net2.bC[idx2] - net1.bC[idx1]:.6f}")

# Find the binding for VP[1]:1/(3,2) - the correctly activated one
binding_name = 'VP[1]:1/(3,2)'
idx1 = net1.binding_names.index(binding_name)
idx2 = net2.binding_names.index(binding_name)

print(f"\nBinding: {binding_name} (should be activated)")
print(f"  Speedup bC[{idx1}]: {net1.bC[idx1]:.6f}")
print(f"  SAP bC[{idx2}]:     {net2.bC[idx2]:.6f}")
print(f"  Difference: {net2.bC[idx2] - net1.bC[idx1]:.6f}")

print("\n" + "="*70)
print("COMPARING WEIGHT STATISTICS (WC)")
print("="*70)

# Get WC matrices (convert sparse to dense if needed)
import scipy.sparse as sparse

if sparse.issparse(net1.WC):
    WC1 = net1.WC.toarray()
else:
    WC1 = net1.WC

if sparse.issparse(net2.WC):
    WC2 = net2.WC.toarray()
else:
    WC2 = net2.WC

print(f"\nWC matrix statistics:")
print(f"  Speedup WC mean: {np.mean(WC1):.6f}, std: {np.std(WC1):.6f}")
print(f"  SAP WC mean:     {np.mean(WC2):.6f}, std: {np.std(WC2):.6f}")

# Check specific weights
print(f"\n" + "="*70)
print("COMPARING SPECIFIC WEIGHTS")
print("="*70)

# Weight from Vi binding to VP[1] binding
vi_binding = 'Vi:1/(1,2)'
vp_binding = 'VP[1]:1/(3,2)'
vi_idx1 = net1.binding_names.index(vi_binding)
vp_idx1 = net1.binding_names.index(vp_binding)
vi_idx2 = net2.binding_names.index(vi_binding)
vp_idx2 = net2.binding_names.index(vp_binding)

print(f"\nWeight from {vi_binding} to {vp_binding}:")
print(f"  Speedup WC[{vp_idx1},{vi_idx1}]: {WC1[vp_idx1, vi_idx1]:.6f}")
print(f"  SAP WC[{vp_idx2},{vi_idx2}]:     {WC2[vp_idx2, vi_idx2]:.6f}")
print(f"  Difference: {WC2[vp_idx2, vi_idx2] - WC1[vp_idx1, vi_idx1]:.6f}")

# Weight from Vi to *@ (the wrong activation)
at_binding = '*@:1/(3,2)'
at_idx1 = net1.binding_names.index(at_binding)
at_idx2 = net2.binding_names.index(at_binding)

print(f"\nWeight from {vi_binding} to {at_binding}:")
print(f"  Speedup WC[{at_idx1},{vi_idx1}]: {WC1[at_idx1, vi_idx1]:.6f}")
print(f"  SAP WC[{at_idx2},{vi_idx2}]:     {WC2[at_idx2, vi_idx2]:.6f}")
print(f"  Difference: {WC2[at_idx2, vi_idx2] - WC1[at_idx1, vi_idx1]:.6f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Compare overall bias distribution
empty_markers = ['*@', '@', '#']
empty_bias_speedup = []
empty_bias_sap = []
real_bias_speedup = []
real_bias_sap = []

for i, bname in enumerate(net1.binding_names):
    filler = bname.split(':')[0]
    if any(marker in filler for marker in empty_markers):
        empty_bias_speedup.append(net1.bC[i])
    else:
        real_bias_speedup.append(net1.bC[i])

for i, bname in enumerate(net2.binding_names):
    filler = bname.split(':')[0]
    if any(marker in filler for marker in empty_markers):
        empty_bias_sap.append(net2.bC[i])
    else:
        real_bias_sap.append(net2.bC[i])

print(f"\nBias statistics:")
print(f"  Speedup - Empty marker biases: mean={np.mean(empty_bias_speedup):.4f}, std={np.std(empty_bias_speedup):.4f}")
print(f"  Speedup - Real word biases:    mean={np.mean(real_bias_speedup):.4f}, std={np.std(real_bias_speedup):.4f}")
print(f"\n  SAP - Empty marker biases:     mean={np.mean(empty_bias_sap):.4f}, std={np.std(empty_bias_sap):.4f}")
print(f"  SAP - Real word biases:        mean={np.mean(real_bias_sap):.4f}, std={np.std(real_bias_sap):.4f}")

if np.mean(empty_bias_sap) > np.mean(real_bias_sap):
    print(f"\n❌ PROBLEM: SAP learned HIGHER biases for empty markers than real words!")
    print(f"   This explains why *@ activates instead of VP[1]")
else:
    print(f"\n✓ Bias distribution looks reasonable")

print("="*70)

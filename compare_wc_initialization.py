#!/usr/bin/env python3
"""
Compare WC initialization between gsc.py and only_gscnet_speedup_sap.py
"""

import numpy as np
import sys

# Import both modules
import gsc
import only_gscnet_speedup_sap as sap

# G1 grammar
PCFG_G1 = '''
1.0 S[x] -> N Vi | N BE Vpp | N Vpp P N Vi
0.5 VP[x] -> Vi | BE Vpp
0.5 VP[x] -> Vpp P N Vi
0.5 VPpp[x] -> Vpp | Vpp P N
1.0 NP[x] -> N | N RC[x]
1.0 RC[x] -> Vpp P N
1.0 PP[x] -> P N
'''

print("="*70)
print("Creating gsc.py network...")
print("="*70)
np.random.seed(1024)
hg_gsc = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim_gsc = hg_gsc.get_simlist(dp=0.0)
net_gsc = gsc.GscNet(hg=hg_gsc, encodings={'similarity': sim_gsc},
                     opts={'use_sparse_wc': False}, seed=1024)

print("\n" + "="*70)
print("Creating SAP network...")
print("="*70)
np.random.seed(1024)
hg_sap = sap.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
sim_sap = hg_sap.get_simlist(dp=0.0)
net_sap = sap.GscNet(hg=hg_sap, encodings={'similarity': sim_sap},
                     opts={'use_sparse_wc': False, 'use_jax': False},
                     seed=1024)

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

# Compare WC
print("\n### WC Matrix Comparison ###")
print(f"gsc.py WC sum: {net_gsc.WC.sum():.6f}")
print(f"SAP WC sum: {net_sap.WC.sum():.6f}")
print(f"Difference: {(net_gsc.WC.sum() - net_sap.WC.sum()):.6f}")

print(f"\ngsc.py WC diagonal sum: {np.diag(net_gsc.WC).sum():.6f}")
print(f"SAP WC diagonal sum: {np.diag(net_sap.WC).sum():.6f}")
print(f"Difference: {(np.diag(net_gsc.WC).sum() - np.diag(net_sap.WC).sum()):.6f}")

# Find positions where diagonal differs
diag_gsc = np.diag(net_gsc.WC)
diag_sap = np.diag(net_sap.WC)
diff_indices = np.where(np.abs(diag_gsc - diag_sap) > 1e-6)[0]
print(f"\nNumber of diagonal positions that differ: {len(diff_indices)}")
if len(diff_indices) > 0 and len(diff_indices) <= 20:
    print("Positions where diagonal differs:")
    for idx in diff_indices[:20]:
        print(f"  Position {idx}: gsc={diag_gsc[idx]:.2f}, SAP={diag_sap[idx]:.2f}, diff={diag_gsc[idx]-diag_sap[idx]:.2f}")
        # Print binding name
        print(f"    Binding: {net_gsc.binding_names[idx]}")

# Compare full WC matrices
wc_diff = net_gsc.WC - net_sap.WC
diff_mask = np.abs(wc_diff) > 1e-6
num_diffs = np.count_nonzero(diff_mask)
print(f"\n### Full WC Matrix ###")
print(f"Number of positions that differ: {num_diffs}")
print(f"Max absolute difference: {np.abs(wc_diff).max():.6f}")

if num_diffs > 0 and num_diffs <= 50:
    print("\nAll positions where WC differs:")
    diff_positions = np.argwhere(diff_mask)
    for i, j in diff_positions[:50]:
        print(f"  WC[{i},{j}]: gsc={net_gsc.WC[i,j]:.2f}, SAP={net_sap.WC[i,j]:.2f}, diff={wc_diff[i,j]:.2f}")
        print(f"    {net_gsc.binding_names[i]} -> {net_gsc.binding_names[j]}")

# Test dynamics
print("\n### Dynamics Test ###")
test_actC = np.random.randn(net_gsc.num_bindings)
result_gsc = net_gsc.WC.dot(test_actC)
result_sap = net_sap.WC.dot(test_actC)
print(f"gsc.py WC.dot(test_actC) sum: {result_gsc.sum():.10f}")
print(f"SAP WC.dot(test_actC) sum: {result_sap.sum():.10f}")
print(f"Difference: {(result_gsc.sum() - result_sap.sum()):.10f}")
print(f"\ngsc.py WC.dot(test_actC) first 5: {result_gsc[:5]}")
print(f"SAP WC.dot(test_actC) first 5: {result_sap[:5]}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

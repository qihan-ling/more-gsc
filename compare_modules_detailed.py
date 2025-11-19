"""Detailed comparison of only_gscnet vs only_gscnet_speedup behavior"""

import sys

# Test with only_gscnet first
print("="*70)
print("Testing with only_gscnet (used by debug script)")
print("="*70)

import only_gscnet as gsc1
import numpy as np

net1 = gsc1.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

# Find 'N Vi P N'
s1_idx = None
for si, sent in enumerate(net1.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vi P N':
        s1_idx = si
        break

sent = net1.corpus['sentence'][s1_idx]
words = [bname.split('/')[0] for bname in sent]

# Run without random seed (like debug script originally)
net1.reset(mu=net1.ep, sd=0.01)
net1.initialize_traces(trace_list='all')

for wi, word in enumerate(words):
    net1.run_word(word, wi + 1, log_trace=True)
net1.run_wrapup(log_trace=True)

# Get rules
rules0_1 = net1.hg.g.get_rules()
rules1 = []
for rule in rules0_1:
    if rule not in rules1:
        rules1.append(rule)

# Compute activations
rname = '(3,2)'
actC_trace1 = net1.traces['actC']
dp_all1 = gsc1.compute_treelet_act_trace(net1, actC_trace1, rules1, rname)

temp1 = np.argsort(dp_all1.sum(axis=0))
focus_idx1 = temp1[::-1][:4]

print(f"\nonly_gscnet results:")
print(f"  Trace shape: {actC_trace1.shape}")
print(f"  net.t: {net1.t}")
print(f"  len(traces['t']): {len(net1.traces['t'])}")
print(f"  traces['t'][-1]: {net1.traces['t'][-1]}")
print(f"  Top 4 indices: {focus_idx1}")
print(f"  Top 4 rules:")
for idx in focus_idx1:
    print(f"    {idx}: {gsc1.rule2str(rules1[idx], suppress_pos=True)} (activation sum: {dp_all1[:, idx].sum():.2f})")

# Clean up
del net1
del sys.modules['only_gscnet']

print("\n" + "="*70)
print("Testing with only_gscnet_speedup (used by plotting script)")
print("="*70)

import only_gscnet_speedup as gsc2

net2 = gsc2.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

# Run without random seed (same as above)
net2.reset(mu=net2.ep, sd=0.01)
net2.initialize_traces(trace_list='all')

for wi, word in enumerate(words):
    net2.run_word(word, wi + 1, log_trace=True)
net2.run_wrapup(log_trace=True)

# Get rules
rules0_2 = net2.hg.g.get_rules()
rules2 = []
for rule in rules0_2:
    if rule not in rules2:
        rules2.append(rule)

# Compute activations
actC_trace2 = net2.traces['actC']
dp_all2 = gsc2.compute_treelet_act_trace(net2, actC_trace2, rules2, rname)

temp2 = np.argsort(dp_all2.sum(axis=0))
focus_idx2 = temp2[::-1][:4]

print(f"\nonly_gscnet_speedup results:")
print(f"  Trace shape: {actC_trace2.shape}")
print(f"  net.t: {net2.t}")
print(f"  len(traces['t']): {len(net2.traces['t'])}")
print(f"  traces['t'][-1]: {net2.traces['t'][-1]}")
print(f"  Top 4 indices: {focus_idx2}")
print(f"  Top 4 rules:")
for idx in focus_idx2:
    print(f"    {idx}: {gsc2.rule2str(rules2[idx], suppress_pos=True)} (activation sum: {dp_all2[:, idx].sum():.2f})")

print("\n" + "="*70)
print("Comparison")
print("="*70)
print(f"Same top 4 indices? {np.array_equal(focus_idx1, focus_idx2)}")
print(f"Same trace shapes? {actC_trace1.shape == actC_trace2.shape}")
print(f"Same final time? {net1.t == net2.t}")

if not np.array_equal(focus_idx1, focus_idx2):
    print(f"\nDifference: only_gscnet={focus_idx1}, only_gscnet_speedup={focus_idx2}")
    print("\nPossible causes:")
    print("  1. Random initialization differences (even without explicit seed)")
    print("  2. JAX vs NumPy numerical differences")
    print("  3. Different trace lengths or timestep sampling")

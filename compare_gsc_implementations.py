#!/usr/bin/env python3
"""
Compare behavior of original gsc.py vs only_gscnet.py on S4 parsing
"""
import gsc  # Original
import only_gscnet as gsc_new  # Split version
import numpy as np

print("="*70)
print("Comparing gsc.py vs only_gscnet.py for S4")
print("="*70)

# Same setup for both
PCFG_G1 = '''
0.35 S -> N Vi
0.60 S -> N VP
0.05 S -> NP Vi
1.0 NP -> N RC
1.0 RC -> Vpp PP
1.0 VPpp -> Vpp PP
1.0 PP -> P N
0.5 VP -> Vi PP
0.3 VP -> BE Vpp
0.2 VP -> BE VPpp
'''

# Load both models
print("\nLoading models...")
net_orig = gsc.load_model('g1_original_test.pkl')
net_new = gsc_new.load_model('g1_ds_speedup_model_copy.pkl')

# Find S4 in both
s4_orig = None
for si, sent in enumerate(net_orig.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vpp P N Vi':
        s4_orig = (si, sent, net_orig.corpus['target'][si])
        break

s4_new = None
for si, sent in enumerate(net_new.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vpp P N Vi':
        s4_new = (si, sent, net_new.corpus['target'][si])
        break

print(f"\nS4 found in original at index: {s4_orig[0]}")
print(f"S4 found in new at index: {s4_new[0]}")

print("\n" + "="*70)
print("1. Comparing learned weights/biases for S4 target bindings")
print("="*70)

# Get S4 target bindings
orig_target = s4_orig[2]
new_target = s4_new[2]

orig_active = [i for i in range(len(net_orig.binding_names)) if orig_target[i] > 0.5]
new_active = [i for i in range(len(net_new.binding_names)) if new_target[i] > 0.5]

print(f"\nNumber of active bindings in S4 target:")
print(f"  Original: {len(orig_active)}")
print(f"  New:      {len(new_active)}")

# Compare biases for S4's key structural bindings
key_bindings = ['NP[1]:0/(4,1)', 'RC[1]:1/(3,2)', 'Vpp:0/(1,2)', 'Vi:1/(1,5)']

print(f"\nBias comparison for key S4 bindings:")
for bname in key_bindings:
    if bname in net_orig.binding_names and bname in net_new.binding_names:
        idx_orig = net_orig.binding_names.index(bname)
        idx_new = net_new.binding_names.index(bname)
        bias_orig = net_orig.bC[idx_orig]
        bias_new = net_new.bC[idx_new]
        diff = bias_new - bias_orig
        marker = "⚠️ DIFF" if abs(diff) > 0.01 else "✓"
        print(f"  {bname:25s} orig={bias_orig:7.4f}, new={bias_new:7.4f}, diff={diff:7.4f} {marker}")

print("\n" + "="*70)
print("2. Comparing equilibrium points (ep)")
print("="*70)

print(f"\nEquilibrium point statistics:")
print(f"  Original: mean={net_orig.ep.mean():.4f}, std={net_orig.ep.std():.4f}")
print(f"  New:      mean={net_new.ep.mean():.4f}, std={net_new.ep.std():.4f}")

# Check S4 target bindings in ep
print(f"\nEquilibrium values for S4 target bindings:")
ep_diffs = []
for bname in key_bindings[:5]:
    if bname in net_orig.binding_names and bname in net_new.binding_names:
        idx_orig = net_orig.binding_names.index(bname)
        idx_new = net_new.binding_names.index(bname)
        ep_orig = net_orig.ep[idx_orig]
        ep_new = net_new.ep[idx_new]
        diff = ep_new - ep_orig
        ep_diffs.append(abs(diff))
        marker = "⚠️ DIFF" if abs(diff) > 0.01 else "✓"
        print(f"  {bname:25s} orig={ep_orig:7.4f}, new={ep_new:7.4f}, diff={diff:7.4f} {marker}")

max_ep_diff = max(ep_diffs) if ep_diffs else 0
print(f"\nMax ep difference for key bindings: {max_ep_diff:.4f}")

print("\n" + "="*70)
print("3. Testing single trial parsing comparison")
print("="*70)

# Remove empty fillers
f_empty_type_orig = net_orig.hg.g.get_types(net_orig.hg.opts['f_empty'])
f_empty_type_new = net_new.hg.g.get_types(net_new.hg.opts['f_empty'])

sent0_orig = [bname for bname in s4_orig[1] if bname.split('/')[0] not in f_empty_type_orig]
sent0_new = [bname for bname in s4_new[1] if bname.split('/')[0] not in f_empty_type_new]

print(f"\nS4 sentence (no empty): {sent0_orig}")

# Run one trial at t=5 with same seed
t = 5
max_sent_len = 5
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
qpolicy = np.insert(dq.cumsum(), 0, 0.)

print(f"\nCommitment level t={t}, qpolicy={qpolicy}")

# Original
print("\n--- Running with ORIGINAL gsc.py ---")
np.random.seed(2048 + t)
net_orig.qpolicy = qpolicy
net_orig.reset(mu=net_orig.ep, sd=0.02)

print(f"Initial state: mean={net_orig.actC.mean():.4f}, std={net_orig.actC.std():.4f}")

for ii, bname in enumerate(sent0_orig):
    net_orig.extC *= 0.5
    net_orig.set_input(bname, use_type=True, cumulative=True)
    word_rt = (qpolicy[ii + 1] - qpolicy[ii]) / net_orig.opts['q_rate']
    if net_orig.opts['use_runC']:
        net_orig.runC(word_rt)

net_orig.run_wrapup()
net_orig.set_discrete_state(net_orig.read_grid_point())

orig_result = np.allclose(net_orig.actC, orig_target)
print(f"Result: {'✓ CORRECT' if orig_result else '✗ WRONG'}")

if not orig_result:
    wrong_orig = [net_orig.binding_names[i] for i in range(len(net_orig.binding_names))
                  if net_orig.actC[i] > 0.5]
    print(f"  Wrong bindings: {wrong_orig[:5]}")

# New
print("\n--- Running with NEW only_gscnet.py ---")
np.random.seed(2048 + t)
net_new.qpolicy = qpolicy
net_new.reset(mu=net_new.ep, sd=0.02)

print(f"Initial state: mean={net_new.actC.mean():.4f}, std={net_new.actC.std():.4f}")

for ii, bname in enumerate(sent0_new):
    net_new.extC *= 0.5
    net_new.set_input(bname, use_type=True, cumulative=True)
    word_rt = (qpolicy[ii + 1] - qpolicy[ii]) / net_new.opts['q_rate']
    if net_new.opts['use_runC']:
        net_new.runC(word_rt)

net_new.run_wrapup()
net_new.set_discrete_state(net_new.read_grid_point())

new_result = np.allclose(net_new.actC, new_target)
print(f"Result: {'✓ CORRECT' if new_result else '✗ WRONG'}")

if not new_result:
    wrong_new = [net_new.binding_names[i] for i in range(len(net_new.binding_names))
                 if net_new.actC[i] > 0.5]
    print(f"  Wrong bindings: {wrong_new[:5]}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if orig_result and not new_result:
    print("\n✓ Original parses correctly, New fails")
    print("\nThis confirms the bug is in only_gscnet.py")

    if max_ep_diff > 0.01:
        print(f"\n⚠️  Equilibrium points differ (max diff={max_ep_diff:.4f})")
        print("    This suggests training produced different results")
    else:
        print("\n✓ Equilibrium points are similar")
        print("    Bug is likely in runtime behavior (set_input, run, read_grid_point, etc.)")

elif not orig_result and not new_result:
    print("\n✗ Both fail - may need more trials or different seed")
else:
    print("\n✓ Both work - S4 can succeed under certain conditions")

print("\n" + "="*70)
print("Next steps:")
print("="*70)
print("""
If ep values differ significantly:
  → Compare weight/bias updates during training
  → Check if cost_grad or parameter update logic changed

If ep values are similar:
  → Compare runtime behavior: set_input, runC, read_grid_point
  → Check if discretization changed
  → Compare C implementation calls (use_runC)
""")

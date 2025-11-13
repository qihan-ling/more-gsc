#!/usr/bin/env python3
"""
Compare grammar, corpus, and initial parameters between gsc.py and only_gscnet.py
"""
import gsc  # Original
import only_gscnet as gsc_new  # Split version
import numpy as np

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

print("="*70)
print("Comparing Grammar Setup")
print("="*70)

# Initialize both with same seed
hg_orig = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
hg_new = gsc_new.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)

print(f"\nNumber of fillers:")
print(f"  Original: {len(hg_orig.filler_names)}")
print(f"  New:      {len(hg_new.filler_names)}")
if len(hg_orig.filler_names) != len(hg_new.filler_names):
    print(f"  ⚠️  MISMATCH: {abs(len(hg_orig.filler_names) - len(hg_new.filler_names))} filler difference!")

print(f"\nNumber of roles:")
print(f"  Original: {len(hg_orig.role_names)}")
print(f"  New:      {len(hg_new.role_names)}")
if len(hg_orig.role_names) != len(hg_new.role_names):
    print(f"  ⚠️  MISMATCH: {abs(len(hg_orig.role_names) - len(hg_new.role_names))} role difference!")

print(f"\nNumber of bindings:")
print(f"  Original: {len(hg_orig.binding_names)}")
print(f"  New:      {len(hg_new.binding_names)}")
if len(hg_orig.binding_names) != len(hg_new.binding_names):
    print(f"  ⚠️  MISMATCH: {abs(len(hg_orig.binding_names) - len(hg_new.binding_names))} binding difference!")

# Show filler names if counts differ
if len(hg_orig.filler_names) != len(hg_new.filler_names):
    print(f"\nOriginal filler names ({len(hg_orig.filler_names)}):")
    print(f"  {hg_orig.filler_names}")
    print(f"\nNew filler names ({len(hg_new.filler_names)}):")
    print(f"  {hg_new.filler_names}")

    # Find which fillers are in new but not original
    orig_set = set(hg_orig.filler_names)
    new_set = set(hg_new.filler_names)
    extra_in_new = new_set - orig_set
    missing_in_new = orig_set - new_set

    if extra_in_new:
        print(f"\n  Extra fillers in NEW ({len(extra_in_new)}): {sorted(extra_in_new)}")
    if missing_in_new:
        print(f"\n  Missing in NEW ({len(missing_in_new)}): {sorted(missing_in_new)}")
else:
    # Check if filler names match
    filler_mismatch = []
    for i, (fo, fn) in enumerate(zip(hg_orig.filler_names, hg_new.filler_names)):
        if fo != fn:
            filler_mismatch.append((i, fo, fn))

    if filler_mismatch:
        print(f"\n✗ Filler name mismatches: {len(filler_mismatch)}")
        for i, fo, fn in filler_mismatch[:5]:
            print(f"    Index {i}: orig='{fo}', new='{fn}'")
    else:
        print(f"\n✓ All filler names match")

# Check rules
rules_orig = hg_orig.g.get_rules()
rules_new = hg_new.g.get_rules()

print(f"\nNumber of rules:")
print(f"  Original: {len(rules_orig)}")
print(f"  New:      {len(rules_new)}")

# Compare first 10 rules
print(f"\nFirst 10 rules comparison:")
for i in range(min(10, len(rules_orig), len(rules_new))):
    ro = rules_orig[i]
    rn = rules_new[i]
    ro_str = f"{ro['m']}({ro['d1']},{ro['d2']})"
    rn_str = f"{rn['m']}({rn['d1']},{rn['d2']})"
    match = "✓" if ro_str == rn_str else "✗"
    print(f"  {i:2d}: {match} orig={ro_str:30s} new={rn_str}")

print("\n" + "="*70)
print("Comparing Network Initialization")
print("="*70)

sim_orig = hg_orig.get_simlist(dp=0.0)
sim_new = hg_new.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

net_orig = gsc.GscNet(hg=hg_orig, encodings={'similarity': sim_orig}, opts=net_opts, seed=1024)
net_new = gsc_new.GscNet(hg=hg_new, encodings={'similarity': sim_new}, opts=net_opts, seed=1024)

print(f"\nInitial WC statistics:")
print(f"  Original: mean={net_orig.WC.mean():.6f}, std={net_orig.WC.std():.6f}")
print(f"  New:      mean={net_new.WC.mean():.6f}, std={net_new.WC.std():.6f}")

print(f"\nInitial bC statistics:")
print(f"  Original: mean={net_orig.bC.mean():.6f}, std={net_orig.bC.std():.6f}")
print(f"  New:      mean={net_new.bC.mean():.6f}, std={net_new.bC.std():.6f}")

# Check if WC and bC are identical
wc_match = np.allclose(net_orig.WC, net_new.WC, atol=1e-10)
bc_match = np.allclose(net_orig.bC, net_new.bC, atol=1e-10)

print(f"\n✓ WC matrices match: {wc_match}")
print(f"✓ bC vectors match: {bc_match}")

if not wc_match:
    wc_diff = np.abs(net_orig.WC - net_new.WC)
    print(f"  Max WC difference: {wc_diff.max():.10f}")
    max_idx = np.unravel_index(wc_diff.argmax(), wc_diff.shape)
    print(f"  At indices {max_idx}: orig={net_orig.WC[max_idx]:.6f}, new={net_new.WC[max_idx]:.6f}")

if not bc_match:
    bc_diff = np.abs(net_orig.bC - net_new.bC)
    print(f"  Max bC difference: {bc_diff.max():.10f}")

print("\n" + "="*70)
print("Comparing Corpus Generation")
print("="*70)

net_orig.generate_corpus(use_freq=True)
net_new.generate_corpus(use_freq=True)

print(f"\nCorpus sizes:")
print(f"  Original: {len(net_orig.corpus['sentence'])} sentences")
print(f"  New:      {len(net_new.corpus['sentence'])} sentences")

print(f"\nSentence comparison:")
for si in range(min(len(net_orig.corpus['sentence']), len(net_new.corpus['sentence']))):
    sent_orig = ' '.join([b.split('/')[0] for b in net_orig.corpus['sentence'][si]])
    sent_new = ' '.join([b.split('/')[0] for b in net_new.corpus['sentence'][si]])
    prob_orig = net_orig.corpus['prob_sent'][si]
    prob_new = net_new.corpus['prob_sent'][si]

    match_sent = "✓" if sent_orig == sent_new else "✗"
    match_prob = "✓" if abs(prob_orig - prob_new) < 1e-6 else "✗"

    print(f"  S{si}: {match_sent}{match_prob} orig='{sent_orig:20s}' p={prob_orig:.4f}")
    if sent_orig != sent_new or abs(prob_orig - prob_new) >= 1e-6:
        print(f"       new='{sent_new:20s}' p={prob_new:.4f}")

print("\n" + "="*70)
print("Summary")
print("="*70)

if wc_match and bc_match:
    print("\n✓ Initial parameters are IDENTICAL")
    print("\nSince initial params match but final ep differs, the bug is in:")
    print("  → Training loop (train2)")
    print("  → Cost/gradient computation (cost, cost_grad)")
    print("  → Parameter updates")
    print("\nNext: Compare training step-by-step to find divergence point")
else:
    print("\n✗ Initial parameters DIFFER")
    print("\nBug is in initialization phase:")
    print("  → Grammar setup")
    print("  → Weight/bias initialization")
    print("  → Network construction")

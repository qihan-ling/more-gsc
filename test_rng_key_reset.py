"""Test that net.reset() properly resets JAX rng_key to respect np.random.seed()"""

import only_gscnet_speedup as gsc
import numpy as np

print("="*70)
print("Testing JAX rng_key reset behavior")
print("="*70)

# Load model
net = gsc.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

# Find 'N Vi P N'
s1_idx = None
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vi P N':
        s1_idx = si
        break

sent = net.corpus['sentence'][s1_idx]
words = [bname.split('/')[0] for bname in sent]

# Test 1: Run with seed, contaminate rng_key, then run again with same seed
print("\nTest 1: Verify reset() properly resets rng_key")
print("-"*70)

# First run with fresh seed
np.random.seed(1024 + s1_idx)
net.reset(mu=net.ep, sd=0.01)
initial_actC_1 = net.actC.copy()
print(f"Run 1 - Initial actC[0:5]: {initial_actC_1[0:5]}")

# Contaminate rng_key by running some dynamics
net.initialize_traces(trace_list='all')
for wi, word in enumerate(words):
    net.run_word(word, wi + 1, log_trace=True)
net.run_wrapup(log_trace=True)
print(f"After contamination - rng_key has been advanced through dynamics")

# Second run with SAME seed - should get same initial state
np.random.seed(1024 + s1_idx)
net.reset(mu=net.ep, sd=0.01)
initial_actC_2 = net.actC.copy()
print(f"Run 2 - Initial actC[0:5]: {initial_actC_2[0:5]}")

# Check if they match
if np.allclose(initial_actC_1, initial_actC_2, rtol=1e-5):
    print("✓ SUCCESS: reset() properly resets rng_key - same seed gives same initial state")
else:
    print("✗ FAILURE: reset() does not properly reset rng_key")
    print(f"  Max difference: {np.max(np.abs(initial_actC_1 - initial_actC_2))}")

# Test 2: Run full pipeline twice with same seed
print("\nTest 2: Verify full pipeline reproducibility")
print("-"*70)

def run_and_compute_activations(seed):
    np.random.seed(seed)
    net.reset(mu=net.ep, sd=0.01)
    net.initialize_traces(trace_list='all')

    for wi, word in enumerate(words):
        net.run_word(word, wi + 1, log_trace=True)
    net.run_wrapup(log_trace=True)

    # Get rules
    rules0 = net.hg.g.get_rules()
    rules = []
    for rule in rules0:
        if rule not in rules:
            rules.append(rule)

    # Compute activations
    rname = '(3,2)'
    actC_trace = net.traces['actC']
    dp_all = gsc.compute_treelet_act_trace(net, actC_trace, rules, rname)

    temp = np.argsort(dp_all.sum(axis=0))
    focus_idx = temp[::-1][:4]

    return focus_idx, dp_all

# Run 1
focus_idx_1, dp_all_1 = run_and_compute_activations(1024 + s1_idx)
print(f"Run 1 - Top 4 indices: {focus_idx_1}")
print(f"Run 1 - Top 4 rules:")
rules0 = net.hg.g.get_rules()
rules = []
for rule in rules0:
    if rule not in rules:
        rules.append(rule)
for idx in focus_idx_1:
    print(f"  {idx}: {gsc.rule2str(rules[idx], suppress_pos=True)}")

# Contaminate by running parsing tests (simulating what cho_grammar1_fulljax.py does)
print("\nSimulating parsing tests contamination...")
for t in range(1, 5):  # Just a few iterations
    max_sent_len = net.hg.opts['max_sent_len']
    dq = np.ones(max_sent_len) * (float(t) / max_sent_len)
    np.random.seed(1024 + t)
    try:
        gsc.test_parse_inc(net, dq=dq, num_trials=2, estr=2, estr_null=2, disp=False)
    except:
        pass
print("Contamination complete - rng_key has been heavily modified")

# Run 2 with SAME seed as Run 1
focus_idx_2, dp_all_2 = run_and_compute_activations(1024 + s1_idx)
print(f"\nRun 2 - Top 4 indices: {focus_idx_2}")
print(f"Run 2 - Top 4 rules:")
for idx in focus_idx_2:
    print(f"  {idx}: {gsc.rule2str(rules[idx], suppress_pos=True)}")

# Check if they match
if np.array_equal(focus_idx_1, focus_idx_2):
    print("\n✓ SUCCESS: Full pipeline is reproducible - same seed gives same top-4 treelets")
    print("  This confirms the plotting function will work correctly after parsing tests")
else:
    print("\n✗ FAILURE: Full pipeline is not reproducible")
    print(f"  Run 1: {focus_idx_1}")
    print(f"  Run 2: {focus_idx_2}")

print("\n" + "="*70)
print("Test complete")
print("="*70)

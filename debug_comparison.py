"""Compare debug script vs plotting function to find the mismatch"""

import only_gscnet_speedup as gsc
import numpy as np

# Load the trained model
print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

# Find sentence 'N Vi P N'
s1_idx = None
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vi P N':
        s1_idx = si
        break

print(f"Found 'N Vi P N' at sentence index: {s1_idx}")
sent = net.corpus['sentence'][s1_idx]
words = [bname.split('/')[0] for bname in sent]

# ============================================================================
# Method 1: Debug script approach (NO random seed set)
# ============================================================================
print("\n" + "="*70)
print("Method 1: Debug script approach (no explicit random seed)")
print("="*70)

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

# Compute activations for (3,2)
rname = '(3,2)'
actC_trace_method1 = net.traces['actC']
dp_all_method1 = gsc.compute_treelet_act_trace(net, actC_trace_method1, rules, rname)

# Get top 4
temp = np.argsort(dp_all_method1.sum(axis=0))
focus_idx_method1 = temp[::-1][:4]

print(f"\nTop 4 indices: {focus_idx_method1}")
print(f"Top 4 total activations: {[dp_all_method1[:, idx].sum() for idx in focus_idx_method1]}")
print(f"Top 4 rules:")
for idx in focus_idx_method1:
    print(f"  {idx}: {gsc.rule2str(rules[idx], suppress_pos=True)}")

print(f"\nTrace shape: {actC_trace_method1.shape}")
print(f"Number of timesteps: {len(net.traces['t'])}")
print(f"Final time net.t: {net.t}")

# ============================================================================
# Method 2: Plotting function approach (with random seed)
# ============================================================================
print("\n" + "="*70)
print("Method 2: Plotting function approach (with seed 1024 + sent_idx)")
print("="*70)

np.random.seed(1024 + s1_idx)
net.reset(mu=net.ep, sd=0.01)
net.initialize_traces(trace_list='all')

for wi, word in enumerate(words):
    net.run_word(word, wi + 1, log_trace=True)
net.run_wrapup(log_trace=True)

# Simulate what plot_treelet_act_trace does
tmin = 0
tmax = net.t
idx = (net.traces['t'] >= tmin) * (net.traces['t'] <= tmax)
actC_trace_method2 = net.traces['actC'][idx, :]
dp_all_method2 = gsc.compute_treelet_act_trace(net, actC_trace_method2, rules, rname)

# Get top 4
temp = np.argsort(dp_all_method2.sum(axis=0))
focus_idx_method2 = temp[::-1][:4]

print(f"\nTop 4 indices: {focus_idx_method2}")
print(f"Top 4 total activations: {[dp_all_method2[:, idx].sum() for idx in focus_idx_method2]}")
print(f"Top 4 rules:")
for idx in focus_idx_method2:
    print(f"  {idx}: {gsc.rule2str(rules[idx], suppress_pos=True)}")

print(f"\nTrace shape after filtering: {actC_trace_method2.shape}")
print(f"Number of timesteps included: {idx.sum()}")
print(f"Total timesteps in trace: {len(net.traces['t'])}")
print(f"Final time net.t: {net.t}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*70)
print("Comparison")
print("="*70)
print(f"Method 1 top 4: {focus_idx_method1}")
print(f"Method 2 top 4: {focus_idx_method2}")
print(f"Are they the same? {np.array_equal(focus_idx_method1, focus_idx_method2)}")

if not np.array_equal(focus_idx_method1, focus_idx_method2):
    print("\nDifferences found!")
    print(f"Method 1 unique: {set(focus_idx_method1) - set(focus_idx_method2)}")
    print(f"Method 2 unique: {set(focus_idx_method2) - set(focus_idx_method1)}")

# Check if the time filtering is the issue
print(f"\nMethod 1 actC_trace shape: {actC_trace_method1.shape}")
print(f"Method 2 actC_trace shape: {actC_trace_method2.shape}")
print(f"Are shapes the same? {actC_trace_method1.shape == actC_trace_method2.shape}")

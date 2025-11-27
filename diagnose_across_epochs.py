"""
Diagnose why probabilities don't change even though weights update
Track weights and predictions across multiple epochs
"""
import numpy as np

print("\n" + "="*70)
print("TRACKING WEIGHTS AND PREDICTIONS ACROSS EPOCHS")
print("="*70)

# Store initial state
if hasattr(net, 'use_sparse') and net.use_sparse:
    WC_epoch0 = net.WC.copy()
else:
    WC_epoch0 = net.WC.copy()

print(f"\nEpoch 0 (initial):")
print(f"  Running estimate_prob_inc...")
stat_Q_epoch0 = net.estimate_prob_inc(prefix=[], num_trials=4)
if isinstance(stat_Q_epoch0, tuple):
    stat_Q_epoch0, _ = stat_Q_epoch0

# Get predictions
pred_probs_0 = []
for si, sent in enumerate(net.corpus['sentence']):
    sent_str_key = ' '.join([bname.split('/')[0] for bname in sent])
    model_prob = 0.0
    for tree_key, prob in stat_Q_epoch0['trees'].items():
        try:
            tree_sent = [net.binding_names[idx] for idx in tree_key]
            tree_str = ' '.join([bname.split('/')[0] for bname in tree_sent])
            if tree_str == sent_str_key:
                model_prob = prob
                break
        except:
            pass
    pred_probs_0.append(model_prob)

print(f"  Predicted probs: {[f'{p:.4f}' for p in pred_probs_0]}")
print(f"  Top 3 predicted trees:")
sorted_trees = sorted(stat_Q_epoch0['trees'].items(), key=lambda x: x[1], reverse=True)[:3]
for tree_key, prob in sorted_trees:
    try:
        tree_sent = [net.binding_names[idx] for idx in tree_key]
        tree_str = ' '.join([bname.split('/')[0] for bname in tree_sent])
        print(f"    p={prob:.4f}: {tree_str}")
    except:
        print(f"    p={prob:.4f}: (parsing error)")

# Sample some weight values to track
if hasattr(net, 'use_sparse') and net.use_sparse:
    # Get a few non-zero entries to track
    WC_coo = net.WC.tocoo()
    sample_indices = [(WC_coo.row[i], WC_coo.col[i]) for i in range(min(5, len(WC_coo.row)))]
else:
    # Get indices of max absolute values
    flat_idx = np.argsort(np.abs(net.WC.ravel()))[-5:]
    sample_indices = [np.unravel_index(idx, net.WC.shape) for idx in flat_idx]

print(f"\n  Sample weight values to track:")
for idx in sample_indices[:3]:
    val = net.WC[idx]
    print(f"    WC[{idx}] = {val:.6f}")

# Run 3 epochs and track changes
for epoch in [1, 2, 3]:
    print(f"\n{'='*70}")
    print(f"After epoch {epoch}:")

    # Train one epoch
    net.train2(train_opts={'num_epochs': 1}, savefilename=None)

    # Check weight changes
    if hasattr(net, 'use_sparse') and net.use_sparse:
        WC_diff = net.WC - WC_epoch0
        n_changed = WC_diff.nnz
        max_change = abs(WC_diff).max() if WC_diff.nnz > 0 else 0
    else:
        WC_diff = net.WC - WC_epoch0
        n_changed = np.count_nonzero(WC_diff)
        max_change = np.max(np.abs(WC_diff))

    print(f"  Weight changes from epoch 0:")
    print(f"    Entries changed: {n_changed:,}")
    print(f"    Max change: {max_change:.6e}")

    print(f"  Sample weight values:")
    for idx in sample_indices[:3]:
        val = net.WC[idx]
        val0 = WC_epoch0[idx]
        print(f"    WC[{idx}] = {val:.6f} (was {val0:.6f}, Δ={val-val0:.6f})")

    # Get predictions
    stat_Q = net.estimate_prob_inc(prefix=[], num_trials=4)
    if isinstance(stat_Q, tuple):
        stat_Q, _ = stat_Q

    pred_probs = []
    for si, sent in enumerate(net.corpus['sentence']):
        sent_str_key = ' '.join([bname.split('/')[0] for bname in sent])
        model_prob = 0.0
        for tree_key, prob in stat_Q['trees'].items():
            try:
                tree_sent = [net.binding_names[idx] for idx in tree_key]
                tree_str = ' '.join([bname.split('/')[0] for bname in tree_sent])
                if tree_str == sent_str_key:
                    model_prob = prob
                    break
            except:
                pass
        pred_probs.append(model_prob)

    print(f"  Predicted probs: {[f'{p:.4f}' for p in pred_probs]}")

    # Check if predictions changed
    probs_changed = any(abs(p - p0) > 1e-6 for p, p0 in zip(pred_probs, pred_probs_0))
    if probs_changed:
        print(f"  ✅ Predictions CHANGED from epoch 0")
    else:
        print(f"  ❌ Predictions UNCHANGED from epoch 0")

    print(f"  Top 3 predicted trees:")
    sorted_trees = sorted(stat_Q['trees'].items(), key=lambda x: x[1], reverse=True)[:3]
    for tree_key, prob in sorted_trees:
        try:
            tree_sent = [net.binding_names[idx] for idx in tree_key]
            tree_str = ' '.join([bname.split('/')[0] for bname in tree_sent])
            print(f"    p={prob:.4f}: {tree_str}")
        except:
            print(f"    p={prob:.4f}: (parsing error)")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if n_changed == 0:
    print("❌ PROBLEM: Weights are not accumulating changes across epochs!")
    print("   Each epoch might be resetting weights to initial values.")
elif not probs_changed:
    print("❌ PROBLEM: Weights are changing, but predictions stay the same!")
    print("   Possible causes:")
    print("   1. The model is stuck in a very deep attractor basin")
    print("   2. The weights being updated don't affect the parse trees")
    print("   3. The learning rate is too small to escape the local minimum")
    print("   4. The initial state from setdiag bug is too corrupted")
    print("\n   RECOMMENDED FIX: Try much larger learning rate (e.g., lrate=1.0)")
else:
    print("✅ Weights are changing AND predictions are changing")
    print("   Training should be working now!")

print("="*70)

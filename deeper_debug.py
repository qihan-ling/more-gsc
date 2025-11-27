"""
Deeper diagnostic - check what's happening during gradient computation
Add this code right before calling net.train2()
"""
import numpy as np

# After net.initialize(train_opts=train_opts) and before net.train2()

print("\n" + "="*70)
print("DEEPER DIAGNOSTIC")
print("="*70)

# Check 1: What's in the corpus?
print("\n1. Corpus Information:")
print(f"   Number of sentences: {len(net.corpus['sentence'])}")
print(f"   Target probabilities (stat_P):")
for si in range(len(net.corpus['sentence'])):
    sent_str = ' '.join([bname.split('/')[0] for bname in net.corpus['sentence'][si]])
    prob = net.corpus['prob_sent'][si]
    print(f"     Sentence {si}: p = {prob:.4f}  ({sent_str})")

# Check 2: Run one trial to see what the model produces
print("\n2. Model Initial Predictions (before training):")
print("   Running estimate_prob_inc with num_trials=4...")
prefix = []
stat_Q = net.estimate_prob_inc(prefix=prefix, num_trials=4)
if isinstance(stat_Q, tuple):
    stat_Q, actC_set = stat_Q

# Extract sentence probabilities from stat_Q
print("   Model probabilities (stat_Q['trees']):")
if 'trees' in stat_Q:
    total_prob = sum(stat_Q['trees'].values())
    print(f"     Total probability mass: {total_prob:.6f}")

    # Match stat_Q trees to corpus sentences
    for si, sent in enumerate(net.corpus['sentence']):
        # Try to find the sentence in stat_Q trees
        # The keys in stat_Q['trees'] are tuples of binding indices
        try:
            sent_tuple = tuple([net.binding_names.index(bname) for bname in sent])
            model_prob = stat_Q['trees'].get(sent_tuple, 0.0)
        except (ValueError, AttributeError):
            # If exact match fails, try to find by matching structure
            model_prob = 0.0
            sent_str_key = ' '.join([bname.split('/')[0] for bname in sent])
            # Search through stat_Q trees to find matching sentence
            for tree_key, prob in stat_Q['trees'].items():
                try:
                    tree_sent = [net.binding_names[idx] for idx in tree_key]
                    tree_str = ' '.join([bname.split('/')[0] for bname in tree_sent])
                    if tree_str == sent_str_key:
                        model_prob = prob
                        break
                except:
                    pass

        target_prob = net.corpus['prob_sent'][si]
        sent_str = ' '.join([bname.split('/')[0] for bname in sent])
        print(f"     Sent {si}: Q={model_prob:.4f} (target P={target_prob:.4f})  {sent_str}")

# Check 3: Compute error
print("\n3. Computing Errors:")
stat_P = net.get_corpus_stat(net.corpus)
kl_curr, xent_curr, err, err_log = net.cost(stat_P, stat_Q)
print(f"   KL divergence: {kl_curr['trees']:.6f}")
print(f"   Error dictionary keys: {list(err.keys())}")
for key in ['trees', 'treelets', 'binding_pairs', 'bindings']:
    if key in err and isinstance(err[key], dict):
        num_entries = len(err[key])
        if num_entries > 0:
            max_err = max(abs(v) for v in err[key].values())
            min_err = min(abs(v) for v in err[key].values())
            print(f"   err['{key}']: {num_entries} entries, |error| range: [{min_err:.6f}, {max_err:.6f}]")
        else:
            print(f"   err['{key}']: 0 entries")

# Check 4: Compute gradients
print("\n4. Computing Gradients:")
net.clear_input()
extC_token = net.extC.astype(bool).astype(int)
dWC_curr, destr_curr, dq_curr, dbC_curr = net.cost_grad(err, extC_token)

if hasattr(net, 'use_sparse') and net.use_sparse:
    print(f"   dWC: {dWC_curr.nnz:,} non-zero entries")
    if dWC_curr.nnz > 0:
        print(f"        max |gradient|: {abs(dWC_curr).max():.6e}")
    else:
        print("   ❌ WARNING: dWC has no non-zero entries!")
else:
    nnz_dWC = np.count_nonzero(dWC_curr)
    print(f"   dWC: {nnz_dWC:,} non-zero entries")
    if nnz_dWC > 0:
        print(f"        max |gradient|: {np.max(np.abs(dWC_curr)):.6e}")
        print(f"        mean |gradient|: {np.mean(np.abs(dWC_curr[dWC_curr != 0])):.6e}")
    else:
        print("   ❌ WARNING: dWC is all zeros!")

nnz_dbC = np.count_nonzero(dbC_curr)
print(f"   dbC: {nnz_dbC:,} non-zero entries")
if nnz_dbC > 0:
    print(f"        max |gradient|: {np.max(np.abs(dbC_curr)):.6e}")
    print(f"        mean |gradient|: {np.mean(np.abs(dbC_curr[dbC_curr != 0])):.6e}")
else:
    print("   ❌ WARNING: dbC is all zeros!")

# Check 5: Training options
print("\n5. Training Configuration:")
print(f"   Learning rate: {net.train_opts['lrate']}")
print(f"   Optimizer: {net.train_opts['optimizer']}")
print(f"   update_w: {net.train_opts['update_w']}")
print(f"   bias1_only: {net.train_opts['bias1_only']}")
print(f"   Coefficients: {net.train_opts['coef']}")

# Check what will actually be updated
print("\n6. What will be updated?")
if net.train_opts['coef']['trees'] > 0:
    print(f"   ✓ Trees coefficient: {net.train_opts['coef']['trees']}")
else:
    print(f"   ✗ Trees coefficient is 0 - trees won't contribute to gradients!")

if net.train_opts['update_w']:
    if not net.train_opts['bias1_only']:
        print(f"   ✓ Weight matrix WC will be updated")
    else:
        print(f"   ✗ Weight matrix WC will NOT be updated (bias1_only=True)")
else:
    print(f"   ✗ Weight matrix WC will NOT be updated (update_w=False)")

if net.train_opts['bias1_only']:
    print(f"   ✓ Bias vector bC will be updated (bias1_only=True)")
elif not net.opts['use_second_order_bias']:
    print(f"   ✓ Bias vector bC will be updated")
else:
    print(f"   ? Bias update depends on other conditions")

# Check 6: Expected parameter update magnitude
if net.train_opts['update_w'] and not net.train_opts['bias1_only']:
    if hasattr(net, 'use_sparse') and net.use_sparse:
        if dWC_curr.nnz > 0:
            expected_update = net.train_opts['lrate'] * abs(dWC_curr).max()
            print(f"\n7. Expected max WC update: {expected_update:.6e}")
    else:
        if nnz_dWC > 0:
            expected_update = net.train_opts['lrate'] * np.max(np.abs(dWC_curr))
            print(f"\n7. Expected max WC update: {expected_update:.6e}")

if nnz_dbC > 0:
    expected_update_bc = net.train_opts['lrate'] * np.max(np.abs(dbC_curr))
    print(f"   Expected max bC update: {expected_update_bc:.6e}")

print("="*70)
print("Now running training...\n")

"""
Verify that weights actually change during training
Add this AFTER running deeper_debug.py
"""
import numpy as np

print("\n" + "="*70)
print("VERIFYING WEIGHT UPDATES")
print("="*70)

# Store initial WC
if hasattr(net, 'use_sparse') and net.use_sparse:
    # For sparse matrices, we can't use copy() the same way
    WC_before = net.WC.copy()
    print(f"Initial WC: {WC_before.nnz:,} non-zero entries")
else:
    WC_before = net.WC.copy()
    print(f"Initial WC shape: {WC_before.shape}")

# Run ONE training epoch
print("\nRunning 1 training epoch...")
net.train2(train_opts={'num_epochs': 1}, savefilename=None)

# Check if WC changed
print("\nChecking if WC changed...")
if hasattr(net, 'use_sparse') and net.use_sparse:
    WC_diff = net.WC - WC_before
    n_changed = WC_diff.nnz
    if n_changed > 0:
        max_change = abs(WC_diff).max()
        print(f"✅ WC CHANGED! {n_changed:,} entries changed")
        print(f"   Max change: {max_change:.6e}")

        # Sample some changed values
        WC_diff_coo = WC_diff.tocoo()
        print(f"\n   Sample of changed entries:")
        for i in range(min(5, len(WC_diff_coo.data))):
            row, col, val = WC_diff_coo.row[i], WC_diff_coo.col[i], WC_diff_coo.data[i]
            before_val = WC_before[row, col]
            after_val = net.WC[row, col]
            print(f"     WC[{row},{col}]: {before_val:.6f} → {after_val:.6f} (Δ={val:.6f})")
    else:
        print("❌ WC DID NOT CHANGE!")
        print("   This is the problem - gradients are computed but not applied!")

        # Debug: check if the training loop was entered
        print("\n   Diagnostic:")
        print(f"   epoch_num: {net.epoch_num}")
        print(f"   update_w: {net.train_opts['update_w']}")
        print(f"   Learning rate: {net.train_opts['lrate']}")
else:
    WC_diff = net.WC - WC_before
    max_change = np.max(np.abs(WC_diff))
    n_changed = np.count_nonzero(WC_diff)

    if max_change > 0:
        print(f"✅ WC CHANGED! {n_changed:,} entries changed")
        print(f"   Max change: {max_change:.6e}")
        print(f"   Mean change: {np.mean(np.abs(WC_diff[WC_diff != 0])):.6e}")

        # Find indices of max change
        max_idx = np.unravel_index(np.argmax(np.abs(WC_diff)), WC_diff.shape)
        print(f"\n   Largest change at WC[{max_idx}]:")
        print(f"     Before: {WC_before[max_idx]:.6f}")
        print(f"     After:  {net.WC[max_idx]:.6f}")
        print(f"     Change: {WC_diff[max_idx]:.6f}")
    else:
        print("❌ WC DID NOT CHANGE!")
        print("   This is the problem - gradients are computed but not applied!")

        print("\n   Diagnostic:")
        print(f"   epoch_num: {net.epoch_num}")
        print(f"   update_w: {net.train_opts['update_w']}")
        print(f"   Learning rate: {net.train_opts['lrate']}")

print("="*70)

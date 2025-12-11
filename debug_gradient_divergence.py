"""
Deep dive: Compare gradient computation during first word to find exact operation that diverges.
"""

import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig
import numpy as np

print("="*70)
print("Gradient Computation Analysis: First Word of S3")
print("="*70)

# Load both models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Setup S3 at t=5
sent_idx = 3
sent = net_sparse.corpus['sentence'][sent_idx]
sent_words = [bname.split('/')[0] for bname in sent]
commitment = 5
max_sent_len = net_sparse.hg.opts['max_sent_len']
dq = np.ones(max_sent_len) * (float(commitment) / max_sent_len)

net_sparse.qpolicy = dq.cumsum()
net_sparse.qpolicy = np.insert(net_sparse.qpolicy, 0, 0.)
net_orig.qpolicy = dq.cumsum()
net_orig.qpolicy = np.insert(net_orig.qpolicy, 0, 0.)

# Reset with same seed
seed = 12345
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)

print(f"\n✓ Initial states synchronized")
print(f"Processing first word: '{sent_words[0]}'")

# Set input for both
word = sent_words[0]
wpos = 1
bname = word + net_sparse.hg.opts['bsep'] + '(1,%d)' % wpos

print(f"\n--- Setting Input: {bname} ---")

# SPARSE: Set input
net_sparse.set_input(bname)
extC_sparse = net_sparse.extC.copy()
print(f"Sparse extC sum: {extC_sparse.sum():.6f}, nnz: {np.count_nonzero(extC_sparse)}")

# ORIGINAL: Set input
net_orig.set_input(bname)
extC_orig = net_orig.extC.copy()
print(f"Original extC sum: {extC_orig.sum():.6f}, nnz: {np.count_nonzero(extC_orig)}")

# Compare inputs
diff_extC = np.abs(extC_sparse - extC_orig).sum()
print(f"extC difference: {diff_extC:.10e}")

if diff_extC > 1e-10:
    print("⚠ WARNING: Input differs!")
    top_diff = np.argsort(np.abs(extC_sparse - extC_orig))[-5:][::-1]
    for idx in top_diff:
        if abs(extC_sparse[idx] - extC_orig[idx]) > 1e-6:
            print(f"  {net_sparse.binding_names[idx]:30s}: sparse={extC_sparse[idx]:.6f}, orig={extC_orig[idx]:.6f}")
else:
    print("✓ Inputs match")

# Now compute ONE gradient step for comparison
print(f"\n--- Computing Gradient (HGradC) ---")

# SPARSE: Compute gradient components
actC_sparse = net_sparse.actC.copy()
q_sparse = net_sparse.q.copy()

print(f"\nSparse actC: sum={actC_sparse.sum():.6f}, max={actC_sparse.max():.6f}")
print(f"Sparse q: {q_sparse[:3]}")

# Compute hgrad_g component by component
WC_dot_actC_sparse = net_sparse.WC.dot(actC_sparse)
hgrad_g_sparse = WC_dot_actC_sparse + net_sparse.bC + net_sparse.extC

print(f"\nSparse gradient components:")
print(f"  WC.dot(actC): sum={WC_dot_actC_sparse.sum():.6f}, max={WC_dot_actC_sparse.max():.6f}")
print(f"  bC: sum={net_sparse.bC.sum():.6f}")
print(f"  extC: sum={net_sparse.extC.sum():.6f}")
print(f"  hgrad_g: sum={hgrad_g_sparse.sum():.6f}, max={hgrad_g_sparse.max():.6f}")

# ORIGINAL: Compute gradient components
actC_orig = net_orig.actC.copy()
q_orig = net_orig.q.copy()

print(f"\nOriginal actC: sum={actC_orig.sum():.6f}, max={actC_orig.max():.6f}")
print(f"Original q: {q_orig[:3]}")

WC_dot_actC_orig = net_orig.WC.dot(actC_orig)
hgrad_g_orig = WC_dot_actC_orig + net_orig.bC + net_orig.extC

print(f"\nOriginal gradient components:")
print(f"  WC.dot(actC): sum={WC_dot_actC_orig.sum():.6f}, max={WC_dot_actC_orig.max():.6f}")
print(f"  bC: sum={net_orig.bC.sum():.6f}")
print(f"  extC: sum={net_orig.extC.sum():.6f}")
print(f"  hgrad_g: sum={hgrad_g_orig.sum():.6f}, max={hgrad_g_orig.max():.6f}")

# COMPARE
print(f"\n--- Comparison ---")
diff_WC_dot = np.abs(WC_dot_actC_sparse - WC_dot_actC_orig).sum()
diff_hgrad_g = np.abs(hgrad_g_sparse - hgrad_g_orig).sum()

print(f"WC.dot(actC) difference: {diff_WC_dot:.6e}")
print(f"hgrad_g difference: {diff_hgrad_g:.6e}")

if diff_WC_dot > 1e-6:
    print(f"\n⚠ WC.dot(actC) DIFFERS SIGNIFICANTLY!")
    print("This is the root cause - sparse WC.dot() produces different results!")

    # Find which bindings have biggest differences
    abs_diff = np.abs(WC_dot_actC_sparse - WC_dot_actC_orig)
    top_diff_idx = np.argsort(abs_diff)[-10:][::-1]
    print(f"\nTop 10 diverging bindings in WC.dot(actC):")
    for idx in top_diff_idx:
        print(f"  {net_sparse.binding_names[idx]:30s}: sparse={WC_dot_actC_sparse[idx]:.6f}, orig={WC_dot_actC_orig[idx]:.6f}, diff={abs_diff[idx]:.6f}")

    # Check WC matrix properties
    print(f"\n--- WC Matrix Properties ---")
    if hasattr(net_sparse, 'use_sparse') and net_sparse.use_sparse:
        print(f"Sparse WC: nnz={net_sparse.WC.nnz:,}, sparsity={100*(1-net_sparse.WC.nnz/net_sparse.WC.shape[0]**2):.2f}%")
        print(f"Sparse WC dtype: {net_sparse.WC.dtype}")
        print(f"Sparse WC format: {net_sparse.WC.format}")
    print(f"Original WC: nnz={np.count_nonzero(net_orig.WC):,}")
    print(f"Original WC dtype: {net_orig.WC.dtype}")

    # Check if WC matrices themselves differ
    if hasattr(net_sparse, 'use_sparse') and net_sparse.use_sparse:
        WC_sparse_dense = net_sparse.WC.toarray()
    else:
        WC_sparse_dense = net_sparse.WC

    WC_diff = np.abs(WC_sparse_dense - net_orig.WC).sum()
    print(f"\nWC matrix difference: {WC_diff:.10e}")
    if WC_diff > 1e-10:
        print("⚠ WC MATRICES DIFFER! This shouldn't happen if training matched!")
    else:
        print("✓ WC matrices match - issue is in the .dot() operation itself")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

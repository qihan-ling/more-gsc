"""
Track dynamics iteration-by-iteration to see where divergence starts during runC loop.
"""

import only_gscnet_speedup_sap as gsc_sparse
import gsc as gsc_orig
import numpy as np

print("="*70)
print("Iteration-by-Iteration Dynamics Analysis")
print("="*70)

# Load models
net_sparse = gsc_sparse.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')
net_orig = gsc_orig.load_model('sap_g1_model_orig.pkl')

# Setup S3 at t=5
sent_idx = 3
sent = net_sparse.corpus['sentence'][sent_idx]
sent_words = [bname.split('/')[0] for bname in sent]
commitment = 5
max_sent_len = 5
dq = np.ones(max_sent_len) * (float(commitment) / max_sent_len)

net_sparse.qpolicy = dq.cumsum()
net_sparse.qpolicy = np.insert(net_sparse.qpolicy, 0, 0.)
net_orig.qpolicy = dq.cumsum()
net_orig.qpolicy = np.insert(net_orig.qpolicy, 0, 0.)

# Reset with SAME seed
seed = 12345
np.random.seed(seed)
net_sparse.reset(mu=net_sparse.ep, sd=0.02)
np.random.seed(seed)
net_orig.reset(mu=net_orig.ep, sd=0.02)

print(f"✓ States synchronized after reset")

# Set input for first word
word = sent_words[0]
wpos = 1
bname = word + net_sparse.hg.opts['bsep'] + '(1,%d)' % wpos

net_sparse.set_input(bname)
net_orig.set_input(bname)

print(f"\n✓ Input set: {bname}")

# Now manually run a few iterations of the dynamics loop to see where divergence occurs
print(f"\n--- Running dynamics iterations manually ---")

# Get initial parameters
qinc_sparse = net_sparse.qpolicy[wpos] - net_sparse.qpolicy[wpos - 1]
qinc_orig = net_orig.qpolicy[wpos] - net_orig.qpolicy[wpos - 1]
duration_sparse = np.max(qinc_sparse) / net_sparse.opts['q_rate']
duration_orig = np.max(qinc_orig) / net_orig.opts['q_rate']

print(f"Duration: sparse={duration_sparse:.4f}, orig={duration_orig:.4f}")
print(f"dt: sparse={net_sparse.dt:.6f}, orig={net_orig.dt:.6f}")

num_steps = 10  # Just test first 10 steps
print(f"\nTesting first {num_steps} dynamics steps:")

for step in range(num_steps):
    print(f"\n--- Step {step+1} ---")

    # SPARSE: One iteration
    actC_before_sparse = net_sparse.actC.copy()

    # Manually call update_stateC for sparse
    hgrad_sparse = net_sparse.HGradC()
    temp_sparse = net_sparse.C_T.dot(hgrad_sparse)
    gradC_sparse = net_sparse.C.dot(temp_sparse)
    gradC_sparse = net_sparse.scale_constants * gradC_sparse

    actC_after_grad_sparse = actC_before_sparse + net_sparse.dt * gradC_sparse

    # Save random state before noise
    rng_state_sparse = np.random.get_state()

    # Add noise
    noise_sparse = np.sqrt(2 * net_sparse.T * net_sparse.dt) * np.random.randn(net_sparse.num_bindings)
    noiseC_sparse = np.sqrt(net_sparse.scale_constants) * net_sparse.N2C(noise_sparse)

    actC_after_noise_sparse = actC_after_grad_sparse + noiseC_sparse
    net_sparse.actC = actC_after_noise_sparse
    net_sparse.actCmat = net_sparse.vec2mat()
    net_sparse.t += net_sparse.dt

    # ORIGINAL: One iteration
    actC_before_orig = net_orig.actC.copy()

    # Manually call update_stateC for original
    hgrad_orig = net_orig.HGradC()
    gradC_orig = net_orig.scale_constants * net_orig.S.dot(hgrad_orig)

    actC_after_grad_orig = actC_before_orig + net_orig.dt * gradC_orig

    # Save random state before noise
    rng_state_orig = np.random.get_state()

    # Add noise
    noise_orig = np.sqrt(2 * net_orig.T * net_orig.dt) * np.random.randn(net_orig.num_bindings)
    noiseC_orig = np.sqrt(net_orig.scale_constants) * net_orig.N2C(noise_orig)

    actC_after_noise_orig = actC_after_grad_orig + noiseC_orig
    net_orig.actC = actC_after_noise_orig
    net_orig.actCmat = net_orig.vec2mat()
    net_orig.t += net_orig.dt

    # COMPARE
    diff_before = np.abs(actC_before_sparse - actC_before_orig).sum()
    diff_hgrad = np.abs(hgrad_sparse - hgrad_orig).sum()
    diff_gradC = np.abs(gradC_sparse - gradC_orig).sum()
    diff_after_grad = np.abs(actC_after_grad_sparse - actC_after_grad_orig).sum()
    diff_noise = np.abs(noiseC_sparse - noiseC_orig).sum()
    diff_after_noise = np.abs(actC_after_noise_sparse - actC_after_noise_orig).sum()

    print(f"  Before:        {diff_before:.6e}")
    print(f"  hgrad diff:    {diff_hgrad:.6e}")
    print(f"  gradC diff:    {diff_gradC:.6e}")
    print(f"  After grad:    {diff_after_grad:.6e}")
    print(f"  Noise diff:    {diff_noise:.6e}")
    print(f"  After noise:   {diff_after_noise:.6e}")

    if diff_gradC > 1e-10:
        print(f"  ⚠ gradC DIVERGES at step {step+1}!")
        print(f"    This is BEFORE noise, so it's the gradient computation!")
        break

    if diff_after_noise > 1e-6:
        print(f"  ⚠ DIVERGENCE at step {step+1}!")
        if diff_after_grad < 1e-10 and diff_noise > 1e-6:
            print(f"    Divergence caused by NOISE (random state desync)")
        elif diff_gradC > 1e-10:
            print(f"    Divergence caused by GRADIENT (lazy S computation)")
        break

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

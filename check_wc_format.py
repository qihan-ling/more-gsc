import only_gscnet_speedup_sap as gsc
import numpy as np
import scipy.sparse as sparse

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

ROOT = 'S'
MAXLEN = 24

print("="*70)
print("CHECKING WC MATRIX FORMAT")
print("="*70)

hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}

encodings = {'similarity': sim}

print("\n1. Creating network...")
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)

print(f"\n2. Checking WC after __init__:")
print(f"   num_bindings: {net.num_bindings}")
print(f"   WC type: {type(net.WC)}")
print(f"   WC shape: {net.WC.shape}")
if sparse.issparse(net.WC):
    print(f"   WC format: {net.WC.format.upper()}")
    print(f"   WC nnz: {net.WC.nnz:,}")
    print(f"   ⚠️  WC is SPARSE!")
    if net.WC.format == 'dok':
        print(f"   ❌ WC IS IN DOK FORMAT - THIS IS THE PROBLEM!")
        print(f"   DOK is 10,000x slower for matrix-vector multiplication!")
    elif net.WC.format == 'csr':
        print(f"   ✓ WC is in CSR format (good for operations)")
else:
    print(f"   ✓ WC is dense numpy array (good)")

print(f"\n3. Generating corpus and initializing...")
net.generate_corpus(use_freq=True, nsamples=100)
net.initialize(train_opts={'lrate': 0.1, 'num_trials': 4})

print(f"\n4. Checking WC after initialize():")
print(f"   WC type: {type(net.WC)}")
if sparse.issparse(net.WC):
    print(f"   WC format: {net.WC.format.upper()}")
    if net.WC.format == 'dok':
        print(f"   ❌ PROBLEM CONFIRMED: WC is in DOK format!")
        print(f"   This is why run_wrapup() takes 574 seconds!")
    elif net.WC.format == 'csr':
        print(f"   ✓ WC is CSR (should be fast)")
else:
    print(f"   ✓ WC is dense (should be fast)")

print(f"\n5. Testing WC.dot() performance...")
import time
actC = np.random.rand(net.num_bindings)
t0 = time.time()
result = net.WC.dot(actC)
t_dot = time.time() - t0

print(f"   WC.dot(actC) time: {t_dot*1000:.2f}ms")
if t_dot > 0.1:
    print(f"   ❌ VERY SLOW! (should be < 1ms)")
    print(f"   This explains the 574s run_wrapup() time!")
elif t_dot > 0.001:
    print(f"   ⚠️  Slow (should be < 1ms)")
else:
    print(f"   ✓ Fast!")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
if sparse.issparse(net.WC) and net.WC.format == 'dok':
    print("ROOT CAUSE: WC is in DOK format")
    print("\nSOLUTION: Convert WC to CSR or dense after initialization")
    print("Add this line after net.initialize():")
    print("  net.WC = net.WC.tocsr()  # Convert DOK → CSR")
elif sparse.issparse(net.WC):
    print(f"WC is sparse {net.WC.format} - should be OK for small grammars")
    print("Consider using dense matrices for better performance")
else:
    print("WC is dense - should be fast. Issue must be elsewhere.")
print("="*70)

"""
Test sparse model parsing accuracy independently (no comparison to original)
"""

import numpy as np
import only_gscnet_speedup_sap as gsc

# Load sparse model
net = gsc.load_model('sap_g1_model_fixed_sparse_nocompress.pkl')

# Check current use_jax setting
print(f"Loaded model use_jax: {net.use_jax if hasattr(net, 'use_jax') else 'N/A'}")
print(f"Loaded model opts['use_jax']: {net.opts.get('use_jax', 'NOT SET')}")

# DON'T set use_jax - use whatever was used during training
# The model should work with its saved configuration

print("\n" + "="*70)
print("Testing parsing at commitment t=5")
print("="*70)

# Test at commitment level 5
t = 5
max_sent_len = net.hg.opts['max_sent_len']
dq = np.ones(max_sent_len) * (float(t) / max_sent_len)

# Run parsing test
parse_results = gsc.test_parse_inc(
    net,
    dq=dq,
    num_trials=10,
    estr=2,
    estr_null=2,
    disp=False
)

# Display results for first 5 sentences
print("\nParsing accuracy results:")
for si in range(min(5, len(net.corpus['sentence']))):
    sent = net.corpus['sentence'][si]
    sent_str = ' '.join([w.split('/')[0] for w in sent])
    acc = parse_results[si]['acc'] if si in parse_results else 0.0
    print(f"S{si}: {sent_str:20s} → {acc*100:5.1f}%")

print("\n" + "="*70)
print("Expected results:")
print("S0: N Vi                → 100%")
print("S1: N Vi P N            → 100%")
print("S2: N BE Vpp            → 100%")
print("S3: N BE Vpp P N        → 100%")
print("S4: N Vpp P N Vi        →  80%")
print("="*70)

print("\nIf results don't match expected, check:")
print("1. Is use_jax set correctly in loaded model?")
print("2. Are estr/dq parameters correct?")
print("3. Is random seed affecting results?")

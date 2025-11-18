#!/usr/bin/env python3
"""
FAST diagnostic script to measure corpus generation speed
and estimate time for large nsamples values.
"""
import sys
sys.path.insert(0, '/home/user/more-gsc')
import only_gscnet_speedup as gsc
import numpy as np
import time

# Load the grammar
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

print("=" * 80)
print("FAST CORPUS GENERATION DIAGNOSTIC")
print("=" * 80)
print(f"Grammar: collapsed_filtered_sm5.grammar")
print(f"Root: {ROOT}")
print(f"Max sentence length: {MAXLEN}")
print()

# Initialize the grammar
print("Initializing grammar...")
t0 = time.time()
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
print(f"  Grammar initialized in {time.time()-t0:.2f}s")
print(f"  Number of fillers: {len(hg.filler_names)}")
print(f"  Number of roles: {len(hg.role_names)}")
print()

# Set all filler similarities to 0
sim = hg.get_simlist(dp=0.0)

# Network options
net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

# Initialize network
print("Initializing network...")
t0 = time.time()
net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)
print(f"  Network initialized in {time.time()-t0:.2f}s")
print()

# Test single sentence generation
print("=" * 80)
print("TESTING SINGLE SENTENCE GENERATION")
print("=" * 80)
print("Generating 1 sentence...")
t0 = time.time()
try:
    sent, target, p = net.generate_sentence()
    t_single = time.time() - t0
    print(f"  SUCCESS! Generated in {t_single:.3f}s")
    print(f"  Sentence length: {len(sent)}")
    print(f"  Probability: {p:.6f}")
    print(f"  First 5 words: {sent[:5]}")
except Exception as e:
    print(f"  FAILED! Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test small batch generation with progress reporting
print("=" * 80)
print("TESTING SMALL BATCH GENERATION")
print("=" * 80)

test_sizes = [10, 50, 100, 500]

for nsamples in test_sizes:
    print(f"\nGenerating {nsamples} sentences...")
    t0 = time.time()

    # Generate with progress
    sentences = []
    for i in range(nsamples):
        if (i+1) % 10 == 0 or i == 0:
            print(f"  Progress: {i+1}/{nsamples} ({(i+1)/nsamples*100:.1f}%)", end='\r')
        sent, target, p = net.generate_sentence()
        sentences.append(sent)

    elapsed = time.time() - t0
    rate = nsamples / elapsed

    # Count unique sentences
    unique = len(set(tuple(s) for s in sentences))

    print(f"  Completed {nsamples} sentences in {elapsed:.2f}s")
    print(f"  Rate: {rate:.1f} sentences/second")
    print(f"  Unique sentences: {unique} ({unique/nsamples*100:.1f}%)")
    print(f"  Estimated time for 5,000: {5000/rate:.1f}s ({5000/rate/60:.1f}min)")
    print(f"  Estimated time for 100,000: {100000/rate:.1f}s ({100000/rate/60:.1f}min)")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"\nBased on generation rate of ~{rate:.1f} sentences/second:")

if rate < 1:
    print("\n⚠️  WARNING: Very slow generation rate!")
    print("   Recommend using nsamples <= 1,000")
    print("   Or reduce MAXLEN from 24 to something smaller (e.g., 10)")
elif rate < 10:
    print("\n⚠️  Slow generation rate")
    print("   Recommend using nsamples <= 10,000")
elif rate < 100:
    print("\n✓  Moderate generation rate")
    print("   Can use nsamples up to 50,000 reasonably")
else:
    print("\n✓  Fast generation rate")
    print("   Can use nsamples up to 200,000")

print("\n" + "=" * 80)

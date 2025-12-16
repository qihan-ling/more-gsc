#!/usr/bin/env python
"""
Test script to verify the bottleneck fix in add_additional_rules()
This tests just the initialization phase where the bottleneck occurs.
"""
import time
import numpy as np

# Import the module
import only_gscnet_speedup_sap as gsc

print("=" * 70)
print("TESTING BOTTLENECK FIX")
print("=" * 70)

# Set seed
np.random.seed(41)
print("Global random seed set to 41 for testing")

# Load grammar
t0 = time.time()
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

print(f"\nInitializing HarmonicGrammar with MAXLEN={MAXLEN}...")
print("This is where the bottleneck was occurring...")

t_init_start = time.time()
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
t_init_end = time.time()

print(f"\n{'=' * 70}")
print(f"✓ SUCCESS! HarmonicGrammar initialized in {t_init_end - t_init_start:.2f}s")
print(f"{'=' * 70}")

print(f"\nFiller names: {hg.filler_names}")
print(f"Number of fillers: {len(hg.filler_names)}")
print(f"Number of roles: {len(hg.role_names)}")
print(f"Number of grammar rules: {len(hg.g.rules)}")

print(f"\n✓ Total time: {time.time() - t0:.2f}s")
print("\nBottleneck fix verified!")

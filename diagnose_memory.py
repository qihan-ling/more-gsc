import gsc
import numpy as np

# Load the same configuration as the training script
with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

ROOT = 'S'
MAXLEN = 24

print("="*70)
print("DIAGNOSING MEMORY REQUIREMENTS")
print("="*70)

# Initialize harmonic grammar
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)

print(f"\nNumber of fillers: {len(hg.filler_names)}")
print(f"Number of roles: {len(hg.role_names)}")
print(f"Number of bindings: {hg.num_bindings}")

num_bindings = hg.num_bindings

# Calculate memory requirements
dense_wc_gb = num_bindings ** 2 * 8 / 1e9
print(f"\n{'='*70}")
print(f"MEMORY REQUIREMENTS:")
print(f"{'='*70}")
print(f"Dense WC matrix: {dense_wc_gb:.2f} GB (float64)")

# For sparse optimizer, we need 3 sparse matrices:
# 1. WC (the weights)
# 2. M_WC (momentum)
# 3. R_WC (RMSprop second moment)

# Initially empty sparse matrices still need to allocate some overhead
# The problem is during _build_model() when we populate them
# Let's estimate assuming 1% sparsity (common for these types of models)
sparsity = 0.01
nnz = int(num_bindings ** 2 * sparsity)

# lil_matrix uses rows (list of lists) - memory intensive!
# Each lil_matrix needs:
# - rows: list of lists (Python overhead)
# - data: list of lists (Python overhead)
# Rough estimate: 100 bytes per row minimum + actual data
lil_overhead_mb = (num_bindings * 200) / 1e6  # Rough overhead per matrix
sparse_data_mb = (nnz * 16) / 1e6  # value (8 bytes) + col index (8 bytes)

print(f"\nSparse matrices (assuming {sparsity*100:.1f}% sparsity, {nnz:,} non-zeros):")
print(f"  Per lil_matrix overhead: ~{lil_overhead_mb:.1f} MB")
print(f"  Per lil_matrix data: ~{sparse_data_mb:.1f} MB")
print(f"  Total per lil_matrix: ~{lil_overhead_mb + sparse_data_mb:.1f} MB")
print(f"\nTotal for 3 lil_matrices (WC, M_WC, R_WC): ~{3 * (lil_overhead_mb + sparse_data_mb) / 1000:.2f} GB")

# But the REAL problem is during initialization in lil format
# lil_matrix becomes very memory intensive when building
print(f"\n{'='*70}")
print(f"MEMORY ISSUE:")
print(f"{'='*70}")
print("During _build_model(), lil_matrix becomes memory-intensive because:")
print("1. Multiple lil_matrices are held in memory simultaneously")
print("2. Python list overhead for each row")
print("3. Each set_weight() call modifies the lil_matrix element by element")
print("\nPOTENTIAL SOLUTIONS:")
print("1. Use dok_matrix (Dictionary of Keys) instead of lil_matrix for building")
print("2. Build in CSR format directly (but slower)")
print("3. Lazy initialization: only create optimizer states when needed")
print("4. Use float32 instead of float64 (halves memory)")

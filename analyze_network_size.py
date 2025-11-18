"""
Analyze the network size for G1 grammar to understand memory requirements
"""

# We can do this analysis without numpy by examining the grammar structure

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
MAXLEN = 5

# Parse the grammar to extract filler types
rules = [line.strip() for line in PCFG_G1.strip().split('\n') if line.strip()]
print("="*70)
print("GRAMMAR ANALYSIS")
print("="*70)
print(f"Number of production rules: {len(rules)}")
print("\nRules:")
for rule in rules:
    print(f"  {rule}")

# Extract unique nonterminals and terminals
nonterminals = set()
terminals = set()

for rule in rules:
    parts = rule.split()
    if len(parts) >= 4:
        lhs = parts[1]  # Left-hand side
        rhs = parts[3:]  # Right-hand side

        nonterminals.add(lhs)
        for symbol in rhs:
            # Heuristic: terminals are typically lowercase or have specific patterns
            # In this grammar: N, Vi, Vpp, P, BE are terminals
            # S, NP, RC, VPpp, PP, VP are nonterminals
            if symbol not in ['->', '|']:
                if symbol in ['S', 'NP', 'RC', 'VPpp', 'PP', 'VP']:
                    nonterminals.add(symbol)
                else:
                    terminals.add(symbol)

print(f"\nNonterminals: {sorted(nonterminals)}")
print(f"Terminals: {sorted(terminals)}")
print(f"Total unique symbols: {len(nonterminals) + len(terminals)}")

# In GSCNet, fillers represent lexical items
# Each terminal can have multiple instances (e.g., N:0, N:1, N:2, ...)
# The number of fillers depends on how many instances are created

# From the grammar structure:
# - Terminals: N, Vi, Vpp, P, BE (5 types)
# - The system may create multiple instances of each
# - Plus special markers like start/end markers, null fillers, etc.

print("\n" + "="*70)
print("NETWORK SIZE ESTIMATION")
print("="*70)

# Role system: Positional roles up to max sentence length
max_sent_len = MAXLEN
print(f"Max sentence length: {max_sent_len}")

# Roles are organized in a tree structure
# For each position (1 to MAXLEN), we have roles at different levels
# Level 1: (1,1), (1,2), ..., (1,MAXLEN)
# Level 2: (2,1), (2,2), ..., (2,MAXLEN)
# etc.

# The number of roles grows with tree depth
# Typical role systems might have ~100-300 roles for MAXLEN=5

# Estimate (conservative):
estimated_num_roles = 100
print(f"Estimated number of roles: {estimated_num_roles}")

# Number of fillers:
# - Each terminal type (5) might have 2-5 instances
# - Plus special fillers (null, start, end markers with various prefixes)
# Looking at the code output: "Number of fillers: {len(hg.filler_names)}"
# From the grammar file comment: "should have 27 fillers × 15 roles = 405 units"

num_fillers = 27  # From the comment in cho_grammar1_new_copy.py
num_roles = 15    # From the comment in cho_grammar1_new_copy.py

print(f"\nFrom code comment (cho_grammar1_new_copy.py:34):")
print(f"  Number of fillers: {num_fillers}")
print(f"  Number of roles: {num_roles}")

num_bindings = num_fillers * num_roles
num_units = num_bindings  # For this encoding

print(f"  Number of bindings: {num_bindings}")
print(f"  Number of neural units: {num_units}")

print("\n" + "="*70)
print("MEMORY ANALYSIS")
print("="*70)

# Matrix sizes:
# WC: num_bindings × num_bindings
# S: num_bindings × num_bindings (if materialized)
# C: num_bindings × num_units
# N: num_units × num_bindings

print(f"\nMatrix dimensions:")
print(f"  WC: {num_bindings} × {num_bindings}")
print(f"  C: {num_bindings} × {num_units}")
print(f"  N: {num_units} × {num_bindings}")
print(f"  S (if materialized): {num_bindings} × {num_bindings}")

# Memory calculation (assuming float32 = 4 bytes)
bytes_per_float = 4

wc_memory = num_bindings * num_bindings * bytes_per_float
s_memory = num_bindings * num_bindings * bytes_per_float
c_memory = num_bindings * num_units * bytes_per_float

print(f"\nMemory requirements (float32):")
print(f"  WC: {wc_memory / 1024**2:.2f} MB")
print(f"  S (if materialized): {s_memory / 1024**2:.2f} MB")
print(f"  C: {c_memory / 1024**2:.2f} MB")
print(f"  Total (without S): {(wc_memory + c_memory) / 1024**2:.2f} MB")
print(f"  Total (with S): {(wc_memory + c_memory + s_memory) / 1024**2:.2f} MB")

print("\n" + "="*70)
print("SPEEDUP ANALYSIS")
print("="*70)

if s_memory > 100 * 1024**2:  # > 100 MB
    print(f"⚠ S matrix is {s_memory / 1024**2:.2f} MB - materializing it is expensive!")
    print(f"  Lazy evaluation (C @ C.T @ v) should save memory and improve cache locality")
else:
    print(f"✓ S matrix is only {s_memory / 1024**2:.2f} MB - materialization is cheap")
    print(f"  Lazy evaluation may not provide significant speedup for this size")

print("\n" + "="*70)
print("EXPECTED SPEEDUP LOCATIONS")
print("="*70)
print("""
1. JAX AVAILABILITY:
   - If JAX is installed and working, BOTH versions should use it
   - The speedup version has additional optimizations:
     * Lazy S matrix multiplication
     * JIT-compiled dynamics_step_jax
     * Batched trial execution via vmap

2. TRAINING LOOP (estimate_prob_inc_jax):
   - Runs multiple trials in parallel on GPU
   - Expected speedup: 5-20x depending on GPU
   - CRITICAL: Only if JAX is working!

3. EQUILIBRIUM FINDING:
   - Uses JIT-compiled loop via jax.lax.fori_loop
   - Avoids Python overhead
   - Expected speedup: 2-10x
   - CRITICAL: Only works when log_trace=False!

4. GRADIENT COMPUTATION:
   - Lazy S matrix saves memory (not much for this size)
   - GPU acceleration of matrix operations
   - Expected speedup: 1.5-3x

WHY BOTH MIGHT TAKE SAME TIME:

1. JAX NOT WORKING:
   - If JAX is not installed or GPU not available
   - Both fall back to CPU NumPy
   - No speedup expected

2. JIT COMPILATION OVERHEAD:
   - First run compiles functions
   - For short training runs, overhead dominates
   - Speedup only apparent after compilation

3. BOTTLENECK ELSEWHERE:
   - Gradient computation and bookkeeping
   - Corpus generation and statistics
   - These may dominate runtime

4. NETWORK TOO SMALL:
   - 405 units is relatively small
   - GPU overhead may exceed benefits
   - Speedup more apparent for larger grammars

RECOMMENDATION:
Run the diagnostic script to check JAX availability and configuration.
""")

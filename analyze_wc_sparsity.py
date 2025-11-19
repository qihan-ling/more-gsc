"""Analyze WC matrix sparsity without allocating full matrix."""
import numpy as np
import sys
sys.path.append('/home/user/more-gsc')
import only_gscnet_speedup as gsc

# Load grammar
GRAMMAR_FILE = 'collapsed_filtered_sm5.grammar'
ROOT = 'S'
MAXLEN = 24

with open(GRAMMAR_FILE, 'r') as f:
    PCFG_sap = f.read()

print("Loading grammar...")
hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)

num_bindings = hg.num_bindings
print(f"\nGrammar size:")
print(f"  Fillers: {hg.num_fillers}")
print(f"  Roles: {hg.num_roles}")
print(f"  Bindings: {num_bindings}")

# Count how many WC entries would be set
print("\nAnalyzing WC sparsity (counting non-zero entries)...")

# Count entries from binary and copy rules
binary_copy_count = 0
for rule in hg.subset_rules(['binary', 'copy']):
    for ri in range(len(hg.role_names)):
        role = hg.roles.role_names[ri]
        if hg.roles.role_is_bracketed[ri] == rule['br']:
            focus_mother_roles_indices = hg.roles.role_mothers_idx[rule['rel']][ri]
            for focus_mother_roles_ind in focus_mother_roles_indices:
                focus_mother_role = hg.role_names[focus_mother_roles_ind]
                if focus_mother_role in hg.roles.role_names:
                    binary_copy_count += 1

print(f"  Binary/copy rules: {binary_copy_count} non-zero entries")

# Competition rules - estimate
competition_count = 0
for rule in hg.subset_rules('competition'):
    r1, r2 = rule['rel'].split('/')
    if r1 == 'ub' and r2 == 'ub':
        for ri in range(len(hg.role_names)):
            if not hg.roles.role_is_bracketed[ri]:
                competition_count += 1
    elif r1 == 's' and r2 == 's':
        competition_count += len(hg.roles.role_names)
    # Other cases...

print(f"  Competition rules: ~{competition_count} non-zero entries")

# Null rules
null_count = 0
for rule in hg.subset_rules(['null']):
    for ri in range(len(hg.role_names)):
        role = hg.roles.role_names[ri]
        if hg.roles.role_is_bracketed[ri] == rule['br']:
            focus_mother_roles_indices = hg.roles.role_mothers_idx[rule['rel']][ri]
            for focus_mother_roles_ind in focus_mother_roles_indices:
                focus_mother_role = hg.role_names[focus_mother_roles_ind]
                if focus_mother_role in hg.roles.role_names:
                    null_count += 1

print(f"  Null rules: {null_count} non-zero entries")

total_nonzero = binary_copy_count + competition_count + null_count
total_entries = num_bindings ** 2
sparsity = 100 * (1 - total_nonzero / total_entries)

print(f"\nSparsity analysis:")
print(f"  Total possible entries: {total_entries:,}")
print(f"  Estimated non-zero entries: {total_nonzero:,}")
print(f"  Sparsity: {sparsity:.4f}%")
print(f"\nMemory comparison:")
print(f"  Dense WC: {total_entries * 8 / 1e12:.2f} TB")
print(f"  Sparse WC: {total_nonzero * (8 + 8) / 1e9:.2f} GB (16 bytes per entry: value + indices)")
print(f"  Memory reduction: {total_entries * 8 / (total_nonzero * 16):.0f}x")

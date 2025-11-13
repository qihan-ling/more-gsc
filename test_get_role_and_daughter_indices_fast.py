#!/usr/bin/env python3
"""
Test get_role_and_daughter_indices_fast() vs manual find_roles() calls
"""
import only_gscnet as gsc
import numpy as np

print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy.pkl')

print("\n" + "="*70)
print("Testing get_role_and_daughter_indices_fast()")
print("="*70)

print("\nComparing fast vs slow for non-terminal roles:")

errors_found = 0

for ri in range(len(net.hg.role_names)):
    role_name = net.hg.role_names[ri]
    is_terminal = net.hg.roles.role_is_terminal[ri]

    if not is_terminal:
        # Fast version
        indices_fast = net.get_role_and_daughter_indices_fast(ri)

        if indices_fast is None:
            print(f"\n✗ Role {ri} ({role_name}): get_role_and_daughter_indices_fast returned None!")
            errors_found += 1
            continue

        # Slow version (manual)
        daughters = net.hg.roles.get_daughters(role_name)
        if daughters:
            daughter_l = daughters['l'][0] if daughters['l'] else None
            daughter_r = daughters['r'][0] if daughters['r'] else None

            idx_slow = net.find_roles(role_name)
            idx_l_slow = net.find_roles(daughter_l) if daughter_l else None
            idx_r_slow = net.find_roles(daughter_r) if daughter_r else None

            # Compare
            idx_match = np.array_equal(indices_fast['self'], idx_slow)
            idx_l_match = np.array_equal(indices_fast['l'], idx_l_slow) if idx_l_slow is not None else (indices_fast['l'] is None or len(indices_fast['l']) == 0)
            idx_r_match = np.array_equal(indices_fast['r'], idx_r_slow) if idx_r_slow is not None else (indices_fast['r'] is None or len(indices_fast['r']) == 0)

            if not (idx_match and idx_l_match and idx_r_match):
                print(f"\n✗ Role {ri} ({role_name}):")
                if not idx_match:
                    print(f"    self mismatch:")
                    print(f"      Fast: {indices_fast['self'][:5]}")
                    print(f"      Slow: {idx_slow[:5]}")
                if not idx_l_match:
                    print(f"    left daughter mismatch ({daughter_l}):")
                    print(f"      Fast: {indices_fast['l'][:5] if indices_fast['l'] is not None else None}")
                    print(f"      Slow: {idx_l_slow[:5] if idx_l_slow is not None else None}")
                if not idx_r_match:
                    print(f"    right daughter mismatch ({daughter_r}):")
                    print(f"      Fast: {indices_fast['r'][:5] if indices_fast['r'] is not None else None}")
                    print(f"      Slow: {idx_r_slow[:5] if idx_r_slow is not None else None}")
                errors_found += 1
            else:
                print(f"✓ Role {ri:2d} ({role_name:15s}): daughters {daughter_l}, {daughter_r}")

print("\n" + "="*70)
if errors_found == 0:
    print("✓ All non-terminal roles match!")
else:
    print(f"✗ Found {errors_found} mismatches!")
    print("\nThis would cause INCORRECT WEIGHT AVERAGING during training,")
    print("leading to wrong equilibrium points for rare structures like S4!")
print("="*70)

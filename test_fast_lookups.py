#!/usr/bin/env python3
"""
Test that fast lookup functions match slow versions exactly
"""
import only_gscnet as gsc
import numpy as np

print("Loading model...")
net = gsc.load_model('g1_ds_speedup_model_copy.pkl')

print("\n" + "="*70)
print("Testing fast vs slow lookup functions")
print("="*70)

# Test find_fillers_fast vs find_fillers
print("\n1. Testing find_fillers_fast:")
test_fillers = ['N', 'Vi', 'Vpp', 'P', 'BE', 'RC', 'NP']

for filler in test_fillers:
    try:
        fast_result = net.find_fillers_fast([filler])
        slow_result = net.find_fillers([filler])

        match = np.array_equal(fast_result, slow_result)
        symbol = "✓" if match else "✗"

        print(f"\n  {filler:10s} {symbol}")
        print(f"    Fast: {len(fast_result)} results - {fast_result[:5]}")
        print(f"    Slow: {len(slow_result)} results - {slow_result[:5]}")

        if not match:
            print(f"    MISMATCH! Fast has {len(fast_result)}, Slow has {len(slow_result)}")
            fast_set = set(fast_result)
            slow_set = set(slow_result)
            if fast_set != slow_set:
                print(f"      In fast but not slow: {fast_set - slow_set}")
                print(f"      In slow but not fast: {slow_set - fast_set}")
    except Exception as e:
        print(f"  {filler:10s} ✗ ERROR: {e}")

# Test find_roles_fast vs find_roles
print("\n" + "="*70)
print("2. Testing find_roles_fast:")
print("="*70)

test_roles = ['(1,1)', '(2,1)', '(3,2)', '(4,1)', '(5,1)', 'NP', 'RC', 'VP']

for role in test_roles:
    try:
        fast_result = net.find_roles_fast([role])
        slow_result = net.find_roles([role])

        match = np.array_equal(fast_result, slow_result)
        symbol = "✓" if match else "✗"

        print(f"\n  {role:10s} {symbol}")
        print(f"    Fast: {len(fast_result)} results - {fast_result[:5]}")
        print(f"    Slow: {len(slow_result)} results - {slow_result[:5]}")

        if not match:
            print(f"    MISMATCH! Fast has {len(fast_result)}, Slow has {len(slow_result)}")
            fast_set = set(fast_result)
            slow_set = set(slow_result)
            if fast_set != slow_set:
                print(f"      In fast but not slow: {fast_set - slow_set}")
                print(f"      In slow but not fast: {slow_set - fast_set}")
    except Exception as e:
        print(f"  {role:10s} ✗ ERROR: {e}")

# Test find_bindings_fast vs find_bindings
print("\n" + "="*70)
print("3. Testing find_bindings_fast:")
print("="*70)

test_bindings = [
    'N:0/(1,1)',
    'Vpp:0/(1,2)',
    'Vi:1/(1,5)',
    'NP[1]:0/(4,1)',
    'RC[1]:1/(3,2)',
    'VP[1]:1/(3,2)',
]

for binding in test_bindings:
    try:
        fast_result = net.find_bindings_fast([binding])
        slow_result = net.find_bindings([binding])

        match = (fast_result == slow_result)
        symbol = "✓" if match else "✗"

        print(f"\n  {binding:20s} {symbol}")
        print(f"    Fast: {fast_result}")
        print(f"    Slow: {slow_result}")

        if not match:
            print(f"    MISMATCH!")
    except Exception as e:
        print(f"  {binding:20s} ✗ ERROR: {e}")

# Test if S4-specific structures are handled correctly
print("\n" + "="*70)
print("4. Testing S4-specific lookups:")
print("="*70)

# Check if root fillers (used in bias updates) work correctly
print("\nRoot fillers (used in training bias updates):")
roots = net.hg.g.get_roots() + [net.hg.g.opts['f_root']]
print(f"  Root filler types: {roots}")

try:
    fast_roots = net.find_fillers_fast(roots)
    slow_roots = net.find_fillers(roots)
    match = np.array_equal(fast_roots, slow_roots)
    symbol = "✓" if match else "✗"
    print(f"  {symbol} Root filler indices match: {match}")
    print(f"    Fast: {len(fast_roots)} indices")
    print(f"    Slow: {len(slow_roots)} indices")
    if not match:
        print(f"    CRITICAL: Root filler lookup mismatch affects training!")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# Check null fillers
print("\nNull fillers (used in training):")
null_type = net.hg.g.opts['null']
print(f"  Null filler type: {null_type}")

try:
    fast_null = net.find_fillers_fast([null_type])
    slow_null = net.find_fillers([null_type])
    match = np.array_equal(fast_null, slow_null)
    symbol = "✓" if match else "✗"
    print(f"  {symbol} Null filler indices match: {match}")
    if not match:
        print(f"    CRITICAL: Null filler lookup mismatch affects training!")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("""
If any mismatches are found above, the fast lookup functions are
returning different results than the slow versions, which would cause
incorrect weight/bias updates during training and lead to S4 failures.

Special attention to:
- Root filler lookups (used in bias updates for root bindings)
- Null filler lookups (used in free_update_null)
- Role/filler lookups for rare structures like NP, RC, Vpp
""")

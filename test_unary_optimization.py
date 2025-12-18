#!/usr/bin/env python
"""
Test to demonstrate the optimization in _add_unary_rules()
Shows the performance difference between repeated method calls vs. cached lookups.
"""
import time

# Simulate the old bottleneck approach
def old_approach_simulation(fillers, roots, terminals):
    """Simulates the old approach with repeated O(n) lookups"""

    def get_roots():
        """Returns a list (simulating self.g.get_roots())"""
        return roots

    def get_terminals():
        """Returns a list (simulating self.g.get_terminals())"""
        return terminals

    def is_bracketed(filler):
        """Simulates is_bracketed() check"""
        return filler.startswith('[')

    count = 0
    for filler in fillers:
        # Each filler triggers multiple method calls
        br1 = is_bracketed(filler)
        br2 = is_bracketed(filler)

        if filler in get_roots():  # O(n) list search
            br3 = is_bracketed(filler)
            count += 1

        if filler in get_terminals():  # O(n) list search
            br4 = is_bracketed(filler)
            count += 1

        # Multiple more is_bracketed calls
        br5 = is_bracketed(filler)
        br6 = is_bracketed(filler)

    return count


# Simulate the new optimized approach
def new_approach_simulation(fillers, roots, terminals):
    """Simulates the new approach with cached O(1) lookups"""

    # OPTIMIZATION: Cache once before the loop
    roots_set = set(roots)
    terminals_set = set(terminals)

    bracketed_cache = {
        filler: filler.startswith('[')
        for filler in fillers
    }

    count = 0
    for filler in fillers:
        # All lookups are now O(1)
        br1 = bracketed_cache[filler]
        br2 = bracketed_cache[filler]

        if filler in roots_set:  # O(1) set lookup
            br3 = bracketed_cache[filler]
            count += 1

        if filler in terminals_set:  # O(1) set lookup
            br4 = bracketed_cache[filler]
            count += 1

        br5 = bracketed_cache[filler]
        br6 = bracketed_cache[filler]

    return count


# Create test data simulating the grammar
print("=" * 70)
print("_add_unary_rules() OPTIMIZATION TEST")
print("=" * 70)

# Simulate 27 fillers (typical for the grammar)
fillers = [f'Filler_{i}' for i in range(27)]
fillers += [f'[Bracketed_{i}]' for i in range(10)]  # Add bracketed fillers
fillers += [f'*Copy_{i}' for i in range(8)]  # Add copy fillers

# Some fillers are roots
roots = ['Filler_0', 'Filler_5', 'Filler_10', '[Bracketed_0]', '[Bracketed_3]']

# Some fillers are terminals
terminals = [f'Filler_{i}' for i in range(15)]

print(f"\nTest setup:")
print(f"  Number of fillers: {len(fillers)}")
print(f"  Number of roots: {len(roots)}")
print(f"  Number of terminals: {len(terminals)}")

# Test old approach (slow)
print(f"\n{'=' * 70}")
print("Testing OLD approach (repeated O(n) list searches)...")
print(f"{'=' * 70}")
t0 = time.time()
result_old = old_approach_simulation(fillers, roots, terminals)
t1 = time.time()
time_old = t1 - t0
print(f"✓ Completed in {time_old*1000:.3f}ms")
print(f"  Rules processed: {result_old}")

# Test new approach (fast)
print(f"\n{'=' * 70}")
print("Testing NEW approach (cached O(1) set lookups)...")
print(f"{'=' * 70}")
t0 = time.time()
result_new = new_approach_simulation(fillers, roots, terminals)
t1 = time.time()
time_new = t1 - t0
print(f"✓ Completed in {time_new*1000:.3f}ms")
print(f"  Rules processed: {result_new}")

# Compare results
print(f"\n{'=' * 70}")
print("PERFORMANCE COMPARISON")
print(f"{'=' * 70}")
speedup = time_old / time_new if time_new > 0 else float('inf')
print(f"Old approach: {time_old*1000:.3f}ms")
print(f"New approach: {time_new*1000:.3f}ms")
print(f"Speedup: {speedup:.1f}x faster")
print(f"\n✓ Results match: {result_old == result_new}")

# Demonstrate the savings
print(f"\n{'=' * 70}")
print("OPERATIONS SAVED")
print(f"{'=' * 70}")
print(f"Without optimization:")
print(f"  - get_roots() called: {len(fillers)} times (once per filler)")
print(f"  - get_terminals() called: {len(fillers)} times (once per filler)")
print(f"  - is_bracketed() called: ~{len(fillers) * 6} times (6× per filler)")
print(f"\nWith optimization:")
print(f"  - get_roots() called: 1 time (cached before loop)")
print(f"  - get_terminals() called: 1 time (cached before loop)")
print(f"  - is_bracketed() computed: {len(fillers)} times (once per filler, cached)")
print(f"\n✓ Optimization converts O(n) operations to O(1) lookups!")

#!/usr/bin/env python
"""
Test to demonstrate the loop optimization in _add_additional_rules()
Simulates the bottleneck scenario and shows the performance improvement.
"""
import time

# Simulate the old bottleneck approach
def old_approach_simulation(rules, fillers):
    """Simulates the old O(n²) approach"""

    def get_nonterminals_slow():
        """Simulates get_nonterminals() - iterates over all rules"""
        mothers = [rule['m'] for rule in rules]
        return list(set(mothers))

    def get_terminals_slow():
        """Simulates get_terminals() - calls get_nonterminals() for each check"""
        nonterminals = get_nonterminals_slow()
        return [f for f in fillers if f not in nonterminals]

    def is_terminal_slow(fname):
        """Simulates the slow is_terminal() that recomputes everything"""
        return fname in get_terminals_slow()

    count = 0
    for rule in rules:
        d1 = rule['d1']
        d2 = rule['d2']
        # Each rule calls is_terminal() multiple times
        if d1 and d2:
            if not is_terminal_slow(d1) and not is_terminal_slow(d2):
                count += 1
            elif is_terminal_slow(d1) and not is_terminal_slow(d2):
                count += 1
            elif not is_terminal_slow(d1) and is_terminal_slow(d2):
                count += 1
    return count


# Simulate the new optimized approach
def new_approach_simulation(rules, fillers):
    """Simulates the new O(n) approach with caching"""

    def get_nonterminals():
        mothers = [rule['m'] for rule in rules]
        return list(set(mothers))

    def get_terminals():
        nonterminals = get_nonterminals()
        return [f for f in fillers if f not in nonterminals]

    # OPTIMIZATION: Cache once before the loop
    terminals_set = set(get_terminals())

    def is_terminal_cached(fname):
        """Cached O(1) lookup"""
        return fname in terminals_set

    count = 0
    for rule in rules:
        d1 = rule['d1']
        d2 = rule['d2']
        # Each rule calls is_terminal_cached() - now O(1)!
        if d1 and d2:
            if not is_terminal_cached(d1) and not is_terminal_cached(d2):
                count += 1
            elif is_terminal_cached(d1) and not is_terminal_cached(d2):
                count += 1
            elif not is_terminal_cached(d1) and is_terminal_cached(d2):
                count += 1
    return count


# Create test data simulating the grammar
print("=" * 70)
print("LOOP OPTIMIZATION TEST")
print("=" * 70)

# Simulate 1072 rules (similar to collapsed_filtered_sm5.grammar)
num_rules = 1072
fillers = [f'Filler_{i}' for i in range(27)]  # 27 fillers
nonterminals = [f'NT_{i}' for i in range(20)]  # 20 non-terminals
terminals = [f for f in fillers if not f.startswith('NT_')]

rules = []
for i in range(num_rules):
    # Mix of rules with terminal and non-terminal daughters
    if i % 3 == 0:
        d1, d2 = nonterminals[i % len(nonterminals)], nonterminals[(i+1) % len(nonterminals)]
    elif i % 3 == 1:
        d1, d2 = fillers[i % len(fillers)], nonterminals[i % len(nonterminals)]
    else:
        d1, d2 = nonterminals[i % len(nonterminals)], fillers[i % len(fillers)]

    rules.append({
        'm': nonterminals[i % len(nonterminals)],
        'd1': d1,
        'd2': d2,
        'p': 0.1
    })

print(f"\nTest setup:")
print(f"  Number of rules: {len(rules)}")
print(f"  Number of fillers: {len(fillers)}")
print(f"  Number of non-terminals: {len(nonterminals)}")

# Test old approach (slow)
print(f"\n{'=' * 70}")
print("Testing OLD approach (O(n²) - recomputes terminals for each check)...")
print(f"{'=' * 70}")
t0 = time.time()
result_old = old_approach_simulation(rules, fillers)
t1 = time.time()
time_old = t1 - t0
print(f"✓ Completed in {time_old*1000:.2f}ms")
print(f"  Result: {result_old} rules processed")

# Test new approach (fast)
print(f"\n{'=' * 70}")
print("Testing NEW approach (O(n) - cached terminal set)...")
print(f"{'=' * 70}")
t0 = time.time()
result_new = new_approach_simulation(rules, fillers)
t1 = time.time()
time_new = t1 - t0
print(f"✓ Completed in {time_new*1000:.2f}ms")
print(f"  Result: {result_new} rules processed")

# Compare results
print(f"\n{'=' * 70}")
print("PERFORMANCE COMPARISON")
print(f"{'=' * 70}")
speedup = time_old / time_new if time_new > 0 else float('inf')
print(f"Old approach: {time_old*1000:.2f}ms")
print(f"New approach: {time_new*1000:.2f}ms")
print(f"Speedup: {speedup:.1f}x faster")
print(f"\n✓ Results match: {result_old == result_new}")
print(f"\n✓ Optimization reduces complexity from O(n²) to O(n)!")

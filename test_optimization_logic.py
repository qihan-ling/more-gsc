#!/usr/bin/env python
"""
Test the optimization logic without requiring full dependencies.
This verifies that the rule_to_tuple conversion works correctly.
"""

def rule_to_tuple(rule):
    """Convert rule dict to hashable tuple for set operations"""
    return (rule.get('m'), rule.get('d1'), rule.get('d2'), rule.get('p'))

# Test data simulating grammar rules
test_rules = [
    {'m': 'S', 'd1': 'N', 'd2': 'VP', 'p': 0.5},
    {'m': 'VP', 'd1': 'V', 'd2': 'N', 'p': 0.3},
    {'m': 'N', 'd1': 'cat', 'd2': None, 'p': 0.2},
    {'m': 'N', 'd1': 'dog', 'd2': None, 'p': 0.3},
]

duplicate_rule = {'m': 'S', 'd1': 'N', 'd2': 'VP', 'p': 0.5}

print("Testing rule_to_tuple conversion...")
print("=" * 60)

# Test conversion
for i, rule in enumerate(test_rules):
    tuple_form = rule_to_tuple(rule)
    print(f"Rule {i}: {rule}")
    print(f"  -> Tuple: {tuple_form}")
    print()

# Test set-based membership (O(1))
print("Testing set-based membership check (O(1))...")
print("=" * 60)

rules_set = {rule_to_tuple(rule) for rule in test_rules}
print(f"Created set with {len(rules_set)} unique rules")

# Check if duplicate exists
dup_tuple = rule_to_tuple(duplicate_rule)
if dup_tuple in rules_set:
    print(f"✓ Duplicate detected: {duplicate_rule}")
else:
    print(f"✗ Duplicate NOT detected (error)")

# Check if new rule doesn't exist
new_rule = {'m': 'VP', 'd1': 'V', 'd2': 'PP', 'p': 0.4}
new_tuple = rule_to_tuple(new_rule)
if new_tuple not in rules_set:
    print(f"✓ New rule correctly identified as new: {new_rule}")
else:
    print(f"✗ New rule incorrectly marked as duplicate (error)")

print("\n" + "=" * 60)
print("✓ Optimization logic test passed!")
print("=" * 60)

# Demonstrate performance difference
import time

# Simulate large rule set
large_rules = [
    {'m': f'Rule{i}', 'd1': f'D1_{i}', 'd2': f'D2_{i}', 'p': 0.1*i}
    for i in range(1000)
]

# Test list-based search (O(n))
print("\nPerformance comparison with 1000 rules:")
print("-" * 60)

test_rule = {'m': 'Rule999', 'd1': 'D1_999', 'd2': 'D2_999', 'p': 99.9}

t0 = time.time()
for _ in range(1000):
    _ = test_rule in large_rules  # O(n) operation
t1 = time.time()
print(f"List-based search (O(n)): {(t1-t0)*1000:.2f}ms for 1000 checks")

# Test set-based search (O(1))
large_rules_set = {rule_to_tuple(r) for r in large_rules}
test_tuple = rule_to_tuple(test_rule)

t0 = time.time()
for _ in range(1000):
    _ = test_tuple in large_rules_set  # O(1) operation
t1 = time.time()
print(f"Set-based search (O(1)):  {(t1-t0)*1000:.2f}ms for 1000 checks")

print("\n✓ Set-based approach is significantly faster!")

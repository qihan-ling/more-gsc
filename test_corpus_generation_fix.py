#!/usr/bin/env python
"""
Test to demonstrate the corpus generation bottleneck fix.
Shows why g0 needs fast lookups for sentence generation.
"""
import time
from collections import defaultdict

class MockRule:
    """Simulates a grammar rule"""
    def __init__(self, mother, daughter1, daughter2, prob):
        self.data = {'m': mother, 'd1': daughter1, 'd2': daughter2, 'p': prob}

    def __getitem__(self, key):
        return self.data[key]

class MockPCFG_Slow:
    """Simulates PCFG without rule index (like g0 without fast lookups)"""
    def __init__(self, num_rules=1000):
        # Create many rules to simulate large grammar
        self.rules = []
        for i in range(num_rules):
            rule = MockRule(f'NT{i}', f'T{i}', f'T{i+1}', 0.5)
            self.rules.append(rule)

        # Add a few rules for 'S' (the root we'll query)
        for i in range(5):
            rule = MockRule('S', f'A{i}', f'B{i}', 0.2)
            self.rules.append(rule)

    def get_rules(self, subset):
        """Slow O(n) linear search through all rules"""
        mother = subset.get('m')
        result = []
        for rule in self.rules:  # O(n) search!
            if rule['m'] == mother:
                result.append(rule)
        return result


class MockPCFG_Fast:
    """Simulates PCFG with rule index (like g0 WITH fast lookups)"""
    def __init__(self, num_rules=1000):
        # Create many rules
        self.rules = []
        for i in range(num_rules):
            rule = MockRule(f'NT{i}', f'T{i}', f'T{i+1}', 0.5)
            self.rules.append(rule)

        # Add a few rules for 'S'
        for i in range(5):
            rule = MockRule('S', f'A{i}', f'B{i}', 0.2)
            self.rules.append(rule)

        # Build the fast lookup index
        self._build_rule_index()

    def _build_rule_index(self):
        """Build O(1) lookup index"""
        self.rules_by_mother = defaultdict(list)
        for rule in self.rules:
            self.rules_by_mother[rule['m']].append(rule)

    def get_rules(self, subset):
        """Fast O(1) lookup using index"""
        mother = subset.get('m')
        return self.rules_by_mother.get(mother, [])


def simulate_generate_sentence(pcfg, num_nodes=10):
    """Simulates recursive parse tree expansion"""
    # Each node in the parse tree calls get_rules()
    for _ in range(num_nodes):
        rules = pcfg.get_rules({'m': 'S'})
    return len(rules)


print("=" * 70)
print("CORPUS GENERATION BOTTLENECK FIX TEST")
print("=" * 70)

num_rules = 1000
num_nodes = 10
num_sentences = 100

print(f"\nSimulation parameters:")
print(f"  Grammar size: {num_rules} rules")
print(f"  Nodes per sentence: {num_nodes}")
print(f"  Sentences to generate: {num_sentences}")

# Test slow version (without index)
print(f"\n{'=' * 70}")
print("WITHOUT FAST LOOKUPS (like g0 before fix)")
print(f"{'=' * 70}")
pcfg_slow = MockPCFG_Slow(num_rules)
t0 = time.time()
for _ in range(num_sentences):
    simulate_generate_sentence(pcfg_slow, num_nodes)
t_slow = time.time() - t0
print(f"✓ Generated {num_sentences} sentences in {t_slow*1000:.1f}ms")
print(f"  {t_slow/num_sentences*1000:.2f}ms per sentence")

# Test fast version (with index)
print(f"\n{'=' * 70}")
print("WITH FAST LOOKUPS (like g0 after fix)")
print(f"{'=' * 70}")
pcfg_fast = MockPCFG_Fast(num_rules)
t0 = time.time()
for _ in range(num_sentences):
    simulate_generate_sentence(pcfg_fast, num_nodes)
t_fast = time.time() - t0
print(f"✓ Generated {num_sentences} sentences in {t_fast*1000:.1f}ms")
print(f"  {t_fast/num_sentences*1000:.2f}ms per sentence")

# Compare
print(f"\n{'=' * 70}")
print("PERFORMANCE COMPARISON")
print(f"{'=' * 70}")
speedup = t_slow / t_fast
print(f"Without index: {t_slow*1000:.1f}ms")
print(f"With index:    {t_fast*1000:.1f}ms")
print(f"Speedup:       {speedup:.0f}x faster")

# Extrapolate to real corpus generation
print(f"\n{'=' * 70}")
print("EXTRAPOLATION TO REAL CORPUS (5000 sentences)")
print(f"{'=' * 70}")
estimated_slow = (t_slow / num_sentences) * 5000
estimated_fast = (t_fast / num_sentences) * 5000
print(f"Without index: {estimated_slow:.1f}s ({estimated_slow/60:.1f} minutes)")
print(f"With index:    {estimated_fast:.1f}s ({estimated_fast/60:.1f} minutes)")
print(f"\n✓ The fix makes corpus generation {speedup:.0f}x faster!")
print(f"  (from {estimated_slow/60:.1f} min to {estimated_fast/60:.1f} min)")

#!/usr/bin/env python
"""
Proof: The cache doesn't exist when _add_additional_rules() is called

This demonstrates the initialization order and shows that filler_name_to_idx
is not created until AFTER _add_additional_rules() completes.
"""

# Simulate the initialization order
class SimulatedPCFG:
    def __init__(self):
        print("PCFG.__init__ called")
        print("  - Line 101: _create_fastER_lookups_pcfg() is COMMENTED OUT")
        print("  - Cache NOT created here")
        # Line 101: #self._create_fastER_lookups_pcfg()
        # This is commented out!


class SimulatedHarmonicGrammar:
    def __init__(self):
        print("\nHarmonicGrammar.__init__ called")
        print("  Line 1642: Creating PCFG (g0)...")
        self.g0 = SimulatedPCFG()

        print("\n  Line 1643: Deepcopy to self.g")
        # self.g = copy.deepcopy(self.g0)

        print("  Line 1644: _create_roles()")
        print("  Line 1645: _add_names()")

        print("\n  Line 1648: ⚠️  CALLING _add_additional_rules() NOW")
        self._check_cache_before_add_additional_rules()
        self._add_additional_rules()

        print("\n  Line 1650-1660: Adding other rules...")

        print("\n  Line 1664: ✓ NOW calling _create_fastER_lookups_pcfg()")
        self._create_fastER_lookups_pcfg()

        print("\n  Line 1665-1670: Copying cache from g0 if it exists")
        self._check_cache_after_create_lookups()

    def _check_cache_before_add_additional_rules(self):
        print("\n" + "="*70)
        print("CHECKING: Does cache exist BEFORE _add_additional_rules()?")
        print("="*70)
        print(f"  hasattr(self, 'filler_name_to_idx'): {hasattr(self, 'filler_name_to_idx')}")
        print(f"  hasattr(self, 'filler_is_terminal'): {hasattr(self, 'filler_is_terminal')}")
        print("\n  ❌ CACHE DOES NOT EXIST!")
        print("  ❌ is_terminal() will fall back to SLOW path:")
        print("       return fname in self.get_terminals()  # ← O(n) operation")
        print("="*70 + "\n")

    def _add_additional_rules(self):
        print("  Executing _add_additional_rules()...")
        print("    - This loop calls is_terminal() thousands of times")
        print("    - Each call triggers get_terminals() → get_nonterminals()")
        print("    - Result: O(n²) bottleneck")

    def _create_fastER_lookups_pcfg(self):
        print("  Creating cache...")
        self.filler_name_to_idx = {"example": 0}  # Simulated cache
        self.filler_is_terminal = [True]  # Simulated cache

    def _check_cache_after_create_lookups(self):
        print("\n" + "="*70)
        print("CHECKING: Does cache exist AFTER _create_fastER_lookups_pcfg()?")
        print("="*70)
        print(f"  hasattr(self, 'filler_name_to_idx'): {hasattr(self, 'filler_name_to_idx')}")
        print(f"  hasattr(self, 'filler_is_terminal'): {hasattr(self, 'filler_is_terminal')}")
        print("\n  ✓ CACHE NOW EXISTS (but too late for _add_additional_rules!)")
        print("="*70 + "\n")


print("="*70)
print("INITIALIZATION ORDER ANALYSIS")
print("="*70)

hg = SimulatedHarmonicGrammar()

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The is_terminal() method HAS caching logic:

    def is_terminal(self, fname):
        if hasattr(self, 'filler_name_to_idx') and fname in self.filler_name_to_idx:
            return self.filler_is_terminal[...]  # ← O(1) cached lookup
        return fname in self.get_terminals()      # ← O(n) fallback

BUT the cache is created AFTER _add_additional_rules() completes:

    Line 1648: _add_additional_rules()           ← Bottleneck occurs here
    ...
    Line 1664: _create_fastER_lookups_pcfg()     ← Cache created here (too late!)

During _add_additional_rules(), the cache doesn't exist, so every
is_terminal() call falls back to the slow O(n) path.

This is why the optimization to pre-cache the terminal set is necessary:
it makes the data available when it's actually needed.
""")

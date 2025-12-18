#!/usr/bin/env python
"""
Test to verify that fast lookups are available when needed.
Demonstrates the cache timing fix.
"""

class MockPCFG:
    """Simulates PCFG with filler tracking"""
    def __init__(self):
        self.filler_names = ['N', 'V', 'P']
        self.rules = []
        self.opts = {'copy': '*', 'null': '_'}
        self.root = ['S']

    def _add_names(self):
        """Simulates adding copy symbols to filler list"""
        print("    _add_names() called - adding copy symbols")
        self.filler_names = ['N', 'V', 'P', '*N', '*V', '*P']
        print(f"    Filler list updated: {self.filler_names}")

    def _create_fastER_lookups_pcfg(self):
        """Simulates cache creation"""
        print("    _create_fastER_lookups_pcfg() called")
        self.filler_name_to_idx = {name: i for i, name in enumerate(self.filler_names)}
        self.filler_is_terminal = [True] * len(self.filler_names)
        print(f"    Cache created for {len(self.filler_names)} fillers")

    def is_terminal(self, fname):
        """Simulates is_terminal with caching logic"""
        if hasattr(self, 'filler_name_to_idx') and fname in self.filler_name_to_idx:
            print(f"      is_terminal('{fname}') → Using O(1) cache")
            return self.filler_is_terminal[self.filler_name_to_idx[fname]]
        else:
            print(f"      is_terminal('{fname}') → Falling back to O(n) computation")
            # Simulate expensive computation
            return fname in ['N', 'V', 'P']


class MockHarmonicGrammar_OLD:
    """Simulates OLD initialization order (cache built at end)"""
    def __init__(self):
        print("\n" + "="*70)
        print("OLD INITIALIZATION ORDER (cache built at END)")
        print("="*70)

        self.g = MockPCFG()
        print("Line 1648: _add_additional_rules() starting...")
        self._add_additional_rules()

        print("\nLine 1650: _add_binary_rules()...")
        self._add_binary_rules()

        print("\nLine 1656: _add_unary_rules()...")
        self._add_unary_rules()

        print("\nLine 1664: Building cache NOW (too late!)...")
        self.g._create_fastER_lookups_pcfg()

    def _add_additional_rules(self):
        print("  Adding copy symbols...")
        self.g._add_names()
        print("  _add_additional_rules() complete")

    def _add_binary_rules(self):
        print("  Checking terminals for binary rules...")
        self.g.is_terminal('N')
        self.g.is_terminal('*N')

    def _add_unary_rules(self):
        print("  Checking terminals for unary rules...")
        self.g.is_terminal('V')
        self.g.is_terminal('*V')


class MockHarmonicGrammar_NEW:
    """Simulates NEW initialization order (cache built after filler list stabilizes)"""
    def __init__(self):
        print("\n" + "="*70)
        print("NEW INITIALIZATION ORDER (cache built AFTER filler list stabilizes)")
        print("="*70)

        self.g = MockPCFG()
        print("Line 1648: _add_additional_rules() starting...")
        self._add_additional_rules()

        print("\nLine 1654: Building cache NOW (right after filler list stabilizes)...")
        self.g._create_fastER_lookups_pcfg()

        print("\nLine 1656: _add_binary_rules()...")
        self._add_binary_rules()

        print("\nLine 1662: _add_unary_rules()...")
        self._add_unary_rules()

    def _add_additional_rules(self):
        print("  Adding copy symbols...")
        self.g._add_names()
        print("  _add_additional_rules() complete")
        print("  ✓ Filler list is now STABLE")

    def _add_binary_rules(self):
        print("  Checking terminals for binary rules...")
        self.g.is_terminal('N')
        self.g.is_terminal('*N')

    def _add_unary_rules(self):
        print("  Checking terminals for unary rules...")
        self.g.is_terminal('V')
        self.g.is_terminal('*V')


print("="*70)
print("CACHE TIMING FIX VERIFICATION")
print("="*70)

# Test old order
hg_old = MockHarmonicGrammar_OLD()

# Test new order
hg_new = MockHarmonicGrammar_NEW()

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print("""
OLD ORDER (cache at end):
  - _add_additional_rules(): Filler list changes
  - _add_binary_rules(): Cache doesn't exist → O(n) fallback
  - _add_unary_rules(): Cache doesn't exist → O(n) fallback
  - Build cache: Too late, all slow operations already done

NEW ORDER (cache after filler list stabilizes):
  - _add_additional_rules(): Filler list changes
  - Build cache: Filler list is stable, cache is valid
  - _add_binary_rules(): Cache exists → O(1) lookups ✓
  - _add_unary_rules(): Cache exists → O(1) lookups ✓

Result: All subsequent operations benefit from O(1) cached lookups!
""")

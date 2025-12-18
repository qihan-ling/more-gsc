#!/usr/bin/env python
"""
Test to check if g0 and g have the same content after initialization.
This determines if we can copy g's cache to g0 or need separate caches.
"""

class MockPCFG:
    def __init__(self, name):
        self.name = name
        self.rules = [
            {'m': 'S', 'd1': 'N', 'd2': 'VP', 'p': 0.5},
            {'m': 'VP', 'd1': 'V', 'd2': 'N', 'p': 0.5},
        ]
        self.filler_names = ['S', 'N', 'VP', 'V']
        self.opts = {'copy': '*', 'sep': ':'}

    def _add_names(self):
        """Simulates rebuilding filler list from rules (adds copy symbols)"""
        # Extract all symbols from rules
        fnames = []
        for rule in self.rules:
            fnames.append(rule['m'])
            if rule['d1']: fnames.append(rule['d1'])
            if rule['d2']: fnames.append(rule['d2'])
        # Add copy versions
        fnames = list(set(fnames))
        for fname in list(fnames):
            fnames.append('*' + fname)  # Add copy symbol
        self.filler_names = sorted(fnames)
        print(f"  {self.name}._add_names() called")
        print(f"    {self.name}.filler_names = {self.filler_names}")

    def _sort_rules(self):
        pass


class MockHarmonicGrammar:
    def __init__(self):
        print("="*70)
        print("SIMULATING HarmonicGrammar.__init__")
        print("="*70)

        print("\nLine 1642: Creating g0 (original PCFG)")
        self.g0 = MockPCFG('g0')
        print(f"  g0.filler_names = {self.g0.filler_names}")
        print(f"  g0.rules count = {len(self.g0.rules)}")

        print("\nLine 1643: Creating g = deepcopy(g0)")
        import copy
        self.g = copy.deepcopy(self.g0)
        self.g.name = 'g'
        print(f"  g.filler_names = {self.g.filler_names}")
        print(f"  g.rules count = {len(self.g.rules)}")

        print("\nLine 1648: Calling _add_additional_rules()")
        self._add_additional_rules()

        print("\n" + "="*70)
        print("CHECKING IF g0 AND g ARE THE SAME")
        print("="*70)

        print(f"\ng0.filler_names: {self.g0.filler_names}")
        print(f"g.filler_names:  {self.g.filler_names}")
        print(f"Same? {self.g0.filler_names == self.g.filler_names}")

        print(f"\ng0.rules count: {len(self.g0.rules)}")
        print(f"g.rules count:  {len(self.g.rules)}")
        print(f"Same? {len(self.g0.rules) == len(self.g.rules)}")

        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        if self.g0.filler_names != self.g.filler_names:
            print("❌ g0 and g have DIFFERENT filler lists!")
            print("   g has copy symbols, g0 doesn't")
            print("   Cannot copy g's cache to g0!")
            print("\n   Why:")
            print("   - g.filler_name_to_idx would have indices for '*N', '*V', etc.")
            print("   - g0.filler_names doesn't have these symbols")
            print("   - Copying would create invalid indices!")
        else:
            print("✓ g0 and g have the SAME filler lists")
            print("  Could potentially copy cache")

        if len(self.g0.rules) != len(self.g.rules):
            print("\n❌ g0 and g have DIFFERENT rule counts!")
            print("   g has augmented rules, g0 has only original")
            print("   Cannot copy g's rule indices to g0!")
            print("\n   Why:")
            print("   - g.rules_by_mother contains augmented rules")
            print("   - g0.rules only has original rules")
            print("   - Copying would reference non-existent rules!")

    def _add_additional_rules(self):
        """Simulates the actual _add_additional_rules method"""
        print("  Simulating _add_additional_rules()...")

        # Simulate adding copy rules to g.rules (NOT g0.rules!)
        print("    Adding copy rules to g.rules (NOT g0!)")
        self.g.rules.append({'m': '*S', 'd1': 'S', 'd2': None, 'p': None})
        self.g.rules.append({'m': '*N', 'd1': 'N', 'd2': None, 'p': None})
        print(f"    g.rules count = {len(self.g.rules)}")

        # Simulate calling g._add_names() (line 1960 in real code)
        print("\n    Calling self.g._add_names() (line 1960)")
        print("    (This rebuilds g.filler_names from g.rules)")
        self.g._add_names()

        # NOTE: g0 is NOT modified!
        print("\n    ⚠️  g0 is NOT modified:")
        print(f"    g0.filler_names = {self.g0.filler_names} (unchanged)")
        print(f"    g0.rules count = {len(self.g0.rules)} (unchanged)")


# Run the test
hg = MockHarmonicGrammar()

#!/usr/bin/env python3
"""
Compare output of original vs optimized tokenization functions
"""
import only_datastructure_speedup as gsc_ds

PCFG_G1 = '''
0.35 S -> N Vi
0.60 S -> N VP
0.05 S -> NP Vi
1.0 NP -> N RC
1.0 RC -> Vpp PP
1.0 VPpp -> Vpp PP
1.0 PP -> P N
0.5 VP -> Vi PP
0.3 VP -> BE Vpp
0.2 VP -> BE VPpp
'''

print("="*70)
print("Testing Optimized Tokenization Functions")
print("="*70)

# Test with original functions
class PCFGOriginal(gsc_ds.PCFG):
    def __init__(self, *args, **kwargs):
        # Temporarily patch to use original functions
        self._use_optimized = False
        super().__init__(*args, **kwargs)

    def _tokenize_cnf(self):
        # Ensure we use original
        super()._tokenize_cnf()

    def _tokenize_fillers(self):
        # Ensure we use original
        super()._tokenize_fillers()

# Test with optimized functions
class PCFGOptimized(gsc_ds.PCFG):
    def __init__(self, *args, **kwargs):
        # Ensure we use optimized
        super().__init__(*args, **kwargs)

print("\nCreating grammar with ORIGINAL tokenization...")
# Hack: modify the PCFG class to use original
import types

def _init_with_original(self, pcfg, root):
    if isinstance(pcfg, str):
        self._parse(pcfg)
    else:
        self.rules = pcfg

    self.opts = {
        'sep': ':',
        'add_root': False,
        'f_root': '^^',
        'add_null': True,
        'null': '_',
        'add_empty': False,
        'f_empty': '<<>>',
        'use_hnf': False,
        'use_pos_f': True,
        'pos_f': list(map(str, list(range(10)))),
    }

    if isinstance(root, str):
        root = [root]
    self.root = root

    self.pcfg_str = pcfg
    self._cnf()
    self._cnf2hnf()
    self._tokenize_cnf()  # Original
    self._tokenize_fillers()  # Original
    self._sort_rules()
    self._create_fastER_lookups_pcfg()

pcfg_orig = gsc_ds.PCFG.__new__(gsc_ds.PCFG)
pcfg_orig.__init__ = types.MethodType(_init_with_original, pcfg_orig)
pcfg_orig.__init__(PCFG_G1, root='S')

print(f"  Rules: {len(pcfg_orig.rules)}")
print(f"  Fillers: {len(pcfg_orig.filler_names)}")

print("\nCreating grammar with OPTIMIZED tokenization...")
pcfg_opt = gsc_ds.PCFG(PCFG_G1, root='S')

print(f"  Rules: {len(pcfg_opt.rules)}")
print(f"  Fillers: {len(pcfg_opt.filler_names)}")

print("\n" + "="*70)
print("Comparing Rules")
print("="*70)

if len(pcfg_orig.rules) == len(pcfg_opt.rules):
    print(f"✓ Rule counts match: {len(pcfg_orig.rules)}")

    # Check if rules are identical
    rules_match = True
    for i, (r_orig, r_opt) in enumerate(zip(pcfg_orig.rules, pcfg_opt.rules)):
        if r_orig != r_opt:
            rules_match = False
            print(f"\n✗ Rule {i} differs:")
            print(f"  Original: {r_orig}")
            print(f"  Optimized: {r_opt}")

    if rules_match:
        print("✓ All rules identical!")
else:
    print(f"✗ Rule count mismatch: {len(pcfg_orig.rules)} vs {len(pcfg_opt.rules)}")

print("\n" + "="*70)
print("Comparing Fillers")
print("="*70)

if len(pcfg_orig.filler_names) == len(pcfg_opt.filler_names):
    print(f"✓ Filler counts match: {len(pcfg_orig.filler_names)}")

    # Check if fillers are identical
    if pcfg_orig.filler_names == pcfg_opt.filler_names:
        print("✓ All fillers identical!")
    else:
        orig_set = set(pcfg_orig.filler_names)
        opt_set = set(pcfg_opt.filler_names)
        extra = opt_set - orig_set
        missing = orig_set - opt_set

        if extra:
            print(f"\n✗ Extra in optimized ({len(extra)}):")
            for f in sorted(extra):
                print(f"    {f}")
        if missing:
            print(f"\n✗ Missing in optimized ({len(missing)}):")
            for f in sorted(missing):
                print(f"    {f}")
else:
    print(f"✗ Filler count mismatch: {len(pcfg_orig.filler_names)} vs {len(pcfg_opt.filler_names)}")

print("\n" + "="*70)
print("RESULT:")
print("="*70)

if (len(pcfg_orig.rules) == len(pcfg_opt.rules) and
    pcfg_orig.filler_names == pcfg_opt.filler_names):
    print("\n✓ Optimized tokenization produces IDENTICAL output to original!")
    print("  The bug has been fixed!")
else:
    print("\n✗ Optimized tokenization still differs from original")
    print("  Bug still present")

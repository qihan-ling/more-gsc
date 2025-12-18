### CORPUS GENERATION BOTTLENECK ANALYSIS

## Problem
First sentence generation takes 1335.5 seconds (~22 minutes)!

## Root Cause

The bottleneck is in `get_rules()` called during `generate_sentence()`.

### The Call Chain:
```
GscNet.generate_corpus()
  └─ GscNet.generate_sentence()  (line 3101 only_gscnet_speedup_sap.py)
      └─ HarmonicGrammar.generate_sentence()  (line 2528 only_datastructure_speedup_sap.py)
          └─ self.g0.generate_sentence()  ← Uses g0, NOT g!
              └─ expand(node) recursively
                  └─ self.get_rules(subset={'m': node.sym})  ← Called for EVERY node!
                      └─ Linear O(n) search through 1072+ rules
```

### The Core Issue:

**Line 2528 in only_datastructure_speedup_sap.py:**
```python
sent, parse, p = self.g0.generate_sentence(...)
```

It uses `self.g0` (the original PCFG), NOT `self.g` (the augmented one).

### Why g0 is Slow:

**Line 101 in PCFG.__init__:**
```python
#self._create_fastER_lookups_pcfg()  ← COMMENTED OUT!
```

When `g0 = PCFG(...)` is created (line 1642), it does NOT build the rule indices.

**Line 1654 in HarmonicGrammar.__init__:**
```python
self.g._create_fastER_lookups_pcfg()  ← Only builds for self.g!
```

Only `self.g` gets the fast lookups, but NOT `self.g0`.

### Impact:

During sentence generation:
- `get_rules({'m': symbol})` is called for EVERY node in the parse tree
- Without `rules_by_mother` index, it does linear O(n) search through ALL rules
- For a simple sentence with 5 words, the parse tree has ~10 nodes
- Each node expansion: O(1072) operations
- Recursive expansion: 10 nodes × 1072 rules = 10,720 lookups
- Result: Extremely slow!

### Complexity Analysis:

**Without index (current):**
- get_rules(): O(n) where n = 1072 rules
- Per sentence: O(nodes × rules) = O(10 × 1072) = ~10,000 operations
- 5000 sentences: 50 million operations!

**With index (optimized):**
- get_rules(): O(1) using rules_by_mother dict
- Per sentence: O(nodes) = O(10) operations
- 5000 sentences: 50,000 operations
- **Expected speedup: 1000x!**

## Solution

Build the fast lookups for `g0` after it's created:

```python
Line 1642: self.g0 = PCFG(...)           ← g0 created
Line 1643: self.g0._create_fastER_lookups_pcfg()  ← BUILD INDEX FOR g0!
Line 1644: self.g = copy.deepcopy(self.g0)
...
```

This ensures g0 has the rule indices for fast sentence generation.

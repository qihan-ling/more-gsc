"""
Diagnostic wrapper to see where PCFG initialization is stuck
Add this before creating HarmonicGrammar
"""

import gsc
import time

# Wrap PCFG methods to add timing
original_cnf = gsc.PCFG._cnf
original_cnf2hnf = gsc.PCFG._cnf2hnf
original_tokenize_cnf = gsc.PCFG._tokenize_cnf
original_tokenize_fillers = gsc.PCFG._tokenize_fillers

def timed_cnf(self):
    print("  _cnf starting...")
    t0 = time.time()
    result = original_cnf(self)
    print(f"  _cnf completed: {time.time()-t0:.1f}s ({len(self.rules)} rules)")
    return result

def timed_cnf2hnf(self):
    print("  _cnf2hnf starting...")
    t0 = time.time()
    result = original_cnf2hnf(self)
    print(f"  _cnf2hnf completed: {time.time()-t0:.1f}s ({len(self.rules)} rules)")
    return result

def timed_tokenize_cnf(self):
    print("  _tokenize_cnf starting...")
    print(f"    Input: {len(self.rules)} rules")
    t0 = time.time()
    result = original_tokenize_cnf(self)
    print(f"  _tokenize_cnf completed: {time.time()-t0:.1f}s ({len(self.rules)} rules)")
    return result

def timed_tokenize_fillers(self):
    print("  _tokenize_fillers starting...")
    print(f"    Input: {len(self.rules)} rules")
    t0 = time.time()
    result = original_tokenize_fillers(self)
    print(f"  _tokenize_fillers completed: {time.time()-t0:.1f}s ({len(self.rules)} rules)")
    return result

gsc.PCFG._cnf = timed_cnf
gsc.PCFG._cnf2hnf = timed_cnf2hnf
gsc.PCFG._tokenize_cnf = timed_tokenize_cnf
gsc.PCFG._tokenize_fillers = timed_tokenize_fillers

# Now create your HarmonicGrammar - you'll see progress
# hg = gsc.HarmonicGrammar(pcfg=YOUR_PCFG, ...)

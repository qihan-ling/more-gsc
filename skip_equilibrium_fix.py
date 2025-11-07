"""
Quick fix to skip expensive equilibrium point calculation

Add this to your script BEFORE creating GscNet:
"""

import gsc
import numpy as np

# Monkey-patch the get_ep method to skip Newton's method
original_get_ep = gsc.GscNet.get_ep

def fast_get_ep(self, dur=10, plot=True, q=None, actC=None, method='newton'):
    """Skip expensive Newton's method for large networks"""

    print("Skipping equilibrium point calculation (too expensive for 1k rules)")

    # Just use bowl_center as equilibrium point
    if actC is None:
        actC = self.bowl_center.copy()

    self.ep = actC

    print("Using bowl_center as equilibrium point")

# Replace the method
gsc.GscNet.get_ep = fast_get_ep

# Now create your network normally
# hg = gsc.HarmonicGrammar(...)
# net = gsc.GscNet(hg=hg, ...)  # Will use fast version

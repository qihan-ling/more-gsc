"""
Quick fix: Force single GPU mode

Add this at the TOP of your script (before importing gsc):
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0

# Then continue with your normal code
import gsc

# Rest of your script...

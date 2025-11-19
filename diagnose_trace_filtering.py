"""Diagnostic: Check if trace filtering is causing the mismatch"""

import only_gscnet_speedup as gsc
import numpy as np

# Load model
net = gsc.load_model('g1_ds_speedup_model_copy_trainjax.pkl')

# Find sentence 'N Vi P N'
s1_idx = None
for si, sent in enumerate(net.corpus['sentence']):
    word_seq = ' '.join([bname.split('/')[0] for bname in sent])
    if word_seq == 'N Vi P N':
        s1_idx = si
        break

sent = net.corpus['sentence'][s1_idx]
words = [bname.split('/')[0] for bname in sent]

# Run sentence
np.random.seed(1024 + s1_idx)
net.reset(mu=net.ep, sd=0.01)
net.initialize_traces(trace_list='all')

for wi, word in enumerate(words):
    net.run_word(word, wi + 1, log_trace=True)
net.run_wrapup(log_trace=True)

# Check trace properties
print("="*70)
print("Trace Diagnostics")
print("="*70)
print(f"net.t = {net.t}")
print(f"len(net.traces['t']) = {len(net.traces['t'])}")
print(f"net.traces['t'][0] = {net.traces['t'][0]}")
print(f"net.traces['t'][-1] = {net.traces['t'][-1]}")
print(f"min(net.traces['t']) = {np.min(net.traces['t'])}")
print(f"max(net.traces['t']) = {np.max(net.traces['t'])}")

# Check filtering
tmin = 0
tmax = net.t
idx = (net.traces['t'] >= tmin) * (net.traces['t'] <= tmax)
print(f"\nWith tmin={tmin}, tmax={tmax}:")
print(f"Number of timesteps in full trace: {len(net.traces['t'])}")
print(f"Number of timesteps passing filter: {idx.sum()}")
print(f"Percentage kept: {100 * idx.sum() / len(net.traces['t']):.1f}%")

if idx.sum() < len(net.traces['t']):
    print(f"\nWARNING: Filter is excluding {len(net.traces['t']) - idx.sum()} timesteps!")
    print(f"First few excluded timestep values: {net.traces['t'][~idx][:5]}")

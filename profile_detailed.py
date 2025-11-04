"""
Detailed profiling of train2() to find the mysterious 49% overhead
"""
import gsc
import numpy as np
import time
from collections import defaultdict

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

ROOT = 'S'
MAXLEN = 5

print("Initializing network...")
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

net = gsc.GscNet(hg=hg, encodings={'similarity': sim}, opts=net_opts, seed=1024)
net.generate_corpus(use_freq=True)

train_opts = {
    'lrate': 0.1,
    'num_trials': 500,  # Use 500 to see where time goes
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

net.initialize(train_opts=train_opts)

# Manually time each section of train2() for ONE epoch
print("\n" + "="*70)
print("DETAILED TIMING FOR ONE EPOCH (500 trials)")
print("="*70)

epoch_start = time.time()

# Initialize variables (from train2)
mask = np.ones(net.WC.shape)
dWC = np.zeros(net.WC.shape)
dbC = np.zeros(net.bC.shape)
dqpolicy = np.zeros(net.qpolicy.shape)
destr = np.zeros(net.estr.shape)
xent = {'trees': 0., 'treelets': 0., 'binding_pairs': 0., 'bindings': 0.}
kl = {'trees': 0., 'treelets': 0., 'binding_pairs': 0., 'bindings': 0.}

t_setup = time.time() - epoch_start
print(f"Setup (arrays, dicts):           {t_setup*1000:.2f}ms")

# Main computation loop
prefix = []
prefix_bnames = []

t0 = time.time()
stat_P = net.get_corpus_stat(net.subset_corpus(prefix_bnames))
t_corpus = time.time() - t0
print(f"get_corpus_stat + subset:        {t_corpus*1000:.2f}ms")

t0 = time.time()
stat_Q, actC_set = net.estimate_prob_inc_jax(prefix=prefix, num_trials=500)
t_estimate = time.time() - t0
print(f"estimate_prob_inc_jax:           {t_estimate*1000:.2f}ms")

stat_Q_new = stat_Q

t0 = time.time()
net.clear_input()
extC_token = net.extC.astype(bool).astype(int)
t_input = time.time() - t0
print(f"clear_input + extC processing:   {t_input*1000:.2f}ms")

t0 = time.time()
kl_curr, xent_curr, err, err_log = net.cost(stat_P, stat_Q_new)
t_cost = time.time() - t0
print(f"cost (KL divergence):            {t_cost*1000:.2f}ms")

t0 = time.time()
dWC_curr, destr_curr, dq_curr, dbC_curr = net.cost_grad(err, extC_token)
t_grad = time.time() - t0
print(f"cost_grad (gradients):           {t_grad*1000:.2f}ms")

t0 = time.time()
dWC += dWC_curr
dbC += dbC_curr
for key in xent:
    xent[key] += xent_curr[key]
for key in kl:
    kl[key] += kl_curr[key]
t_accum = time.time() - t0
print(f"Gradient accumulation:           {t_accum*1000:.2f}ms")

# Parameter updates (simplified from train2)
t0 = time.time()
maskWC_update = np.ones((net.num_bindings, net.num_bindings))
maskbC_update = np.ones(net.num_bindings)

if net.train_opts['update_w']:
    weight_decay = np.zeros(net.WC.shape)
    net.WC = net.WC + train_opts['lrate'] * (dWC * mask * maskWC_update + weight_decay)

if net.train_opts['update_b']:
    net.bC = net.bC + train_opts['lrate'] * (dbC * maskbC_update)

t_update = time.time() - t0
print(f"Parameter updates (WC, bC):      {t_update*1000:.2f}ms")

# Trace updates
t0 = time.time()
for varname in net.train_opts['trace_varnames']:
    if varname == 'kl_trees':
        net.traces_train[varname].append(kl['trees'])
    elif varname == 'kl_treelets':
        net.traces_train[varname].append(kl['treelets'])
    elif varname == 'acc':
        net.traces_train[varname].append(0.0)  # Placeholder
t_trace = time.time() - t0
print(f"Trace updates:                   {t_trace*1000:.2f}ms")

total_time = time.time() - epoch_start
print(f"\n{'='*70}")
print(f"TOTAL TIME FOR ONE EPOCH:        {total_time*1000:.2f}ms ({total_time:.3f}s)")
print(f"{'='*70}")

# Calculate percentages
measured_time = t_setup + t_corpus + t_estimate + t_input + t_cost + t_grad + t_accum + t_update + t_trace
unmeasured_time = total_time - measured_time

print(f"\nBreakdown by percentage:")
print(f"  estimate_prob_inc_jax:         {t_estimate/total_time*100:5.1f}%")
print(f"  Parameter updates:             {t_update/total_time*100:5.1f}%")
print(f"  cost (KL):                     {t_cost/total_time*100:5.1f}%")
print(f"  cost_grad:                     {t_grad/total_time*100:5.1f}%")
print(f"  Corpus operations:             {t_corpus/total_time*100:5.1f}%")
print(f"  Trace updates:                 {t_trace/total_time*100:5.1f}%")
print(f"  Other measured:                {(t_setup+t_input+t_accum)/total_time*100:5.1f}%")
print(f"  UNMEASURED:                    {unmeasured_time/total_time*100:5.1f}%")
print(f"{'='*70}")

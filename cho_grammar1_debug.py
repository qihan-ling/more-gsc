"""
Debug version with timing and output flushing
"""
import sys
import time
import gsc

print("="*70, flush=True)
print("DEBUG: Starting initialization...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

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

start = time.time()
print(f"[{time.time()-start:.1f}s] Creating HarmonicGrammar...", flush=True)
hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root=ROOT, max_sent_len=MAXLEN)
print(f"[{time.time()-start:.1f}s]   Fillers: {len(hg.filler_names)}", flush=True)

print(f"[{time.time()-start:.1f}s] Getting similarity list...", flush=True)
sim = hg.get_simlist(dp=0.0)
print(f"[{time.time()-start:.1f}s]   Done", flush=True)

net_opts = {
    'T_init': 0.01,
    'q_max': 15.0,
    'q_init': 0.0,
    'dt_init': 0.005,
    'm': 30,
    'use_runC': True,
}

print(f"[{time.time()-start:.1f}s] Creating GscNet...", flush=True)
net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                 opts=net_opts, seed=1024)
print(f"[{time.time()-start:.1f}s]   num_bindings: {net.num_bindings}", flush=True)

print(f"[{time.time()-start:.1f}s] Generating corpus...", flush=True)
corpus_start = time.time()
net.generate_corpus(use_freq=True)
corpus_time = time.time() - corpus_start
print(f"[{time.time()-start:.1f}s]   Corpus generated in {corpus_time:.1f}s", flush=True)
print(f"[{time.time()-start:.1f}s]   Sentences: {len(net.corpus['sentence'])}", flush=True)

train_opts = {
    'lrate': 0.1,
    'num_trials': 4,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 10,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

print(f"[{time.time()-start:.1f}s] Initializing training...", flush=True)
net.initialize(train_opts=train_opts)
print(f"[{time.time()-start:.1f}s]   Done", flush=True)

print(f"[{time.time()-start:.1f}s] Starting training (10 epochs only for test)...", flush=True)
train_start = time.time()

# ONLY 10 EPOCHS FOR DEBUG
net.train2(
    train_opts={'num_epochs': 10},
    savefilename='g1_model_debug.pkl'
)

train_time = time.time() - train_start
print(f"[{time.time()-start:.1f}s] Training 10 epochs took {train_time:.1f}s ({train_time/10:.1f}s per epoch)", flush=True)

total_time = time.time() - start
print(f"\n[{total_time:.1f}s] TOTAL TIME: {total_time:.1f}s", flush=True)
print(f"Estimated time for 1000 epochs: {train_time/10*1000/60:.1f} minutes", flush=True)

if gsc.GPU_AVAILABLE:
    print("\nGPU was available and should have been used", flush=True)
else:
    print("\nGPU was NOT available (running on CPU)", flush=True)

print("="*70, flush=True)

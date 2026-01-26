# 1. Load and plot training curves
import matplotlib.pyplot as plt
import only_gscnet_speedup_sap as gsc
import pickle
from save_load_model_efficiently import load_model_efficient

ROOT = 'S'
MAXLEN = 24
USE_SPARSE = True


with open('NAME_TBD.pkl', 'rb') as f:
    state = pickle.load(f)

plt.plot(state['traces_train']['acc'])
plt.xlabel('Training step')
plt.ylabel('Accuracy')
plt.show()

with open('collapsed_filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()

hg = gsc.HarmonicGrammar(pcfg=PCFG_sap, root=ROOT, max_sent_len=MAXLEN)
sim = hg.get_simlist(dp=0.0)

net_opts = {
    'use_jax': False,
    'T_init': 0.01,
    'q_max': 12.0,
    'q_init': 0.0,
    'dt_init': 0.04,
    'm': 30,
    'use_runC': True,
    'ep_method': 'integration',
}
if USE_SPARSE:
    net_opts['use_sparse_wc'] = True

encodings = {
    'similarity': sim,
    'dim_f': 150,
    'dim_r': 60,
}
train_opts = {
    'lrate': 0.1,
    'num_trials': 30,
    'ema_stat_weight': 0.0,
    'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
    'report_cycle': 5,
    'init_noise_mag': 0.02,
    'average_weight': False,
    'average_filler_bias': False,
}

# 2. Load model for inference (F, R reconstructed identically)
net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=state['seed'])
net.generate_corpus(use_freq=True, nsamples=5000)
net.initialize(train_opts=train_opts)
load_model_efficient('sap_checkpoint_epoch_0010.pkl', net)

# Now run inference
test_acc = net.run_test()  # or whatever method

# 3. Track tree activations for sentences
activations = net.process_sentence(["the", "dog", "barked"])  # example

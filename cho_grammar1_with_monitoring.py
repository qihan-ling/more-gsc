"""
Training script with integrated memory monitoring

This shows how to integrate monitor_memory.py into your training pipeline
to track memory usage at each stage.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU

import gsc
import numpy as np
import monitor_memory
import optimized_tokenize_cnf

# ================================================
# STEP 1: Start Memory Monitoring
# ================================================
print("Starting memory monitoring...")
mem = monitor_memory.start_monitoring()

# ================================================
# STEP 2: Apply Optimizations
# ================================================
print("\nApplying PCFG tokenization optimization...")
optimized_tokenize_cnf.apply_optimization()
mem.checkpoint("After applying optimizations")

# ================================================
# STEP 3: Load Grammar
# ================================================
print("\nLoading grammar from file...")
# Replace with your actual grammar file path
grammar_file = "path/to/your/grammar.txt"

# For testing, create a large grammar
from gsc import PCFG
pcfg = PCFG()

# Example: Load from file (uncomment and adjust)
# pcfg.read_grammar(grammar_file)

# Or generate example grammar for testing
print("Generating example grammar with 1,000 rules...")
pcfg.start_symbol = 'S'
for i in range(1000):
    pcfg.add_rule(f'X{i}', [f'Y{i}', f'Z{i}'], prob=0.8)
    pcfg.add_rule(f'Y{i}', [f'a{i}'], prob=1.0)
    pcfg.add_rule(f'Z{i}', [f'b{i}'], prob=1.0)

mem.checkpoint("After loading grammar")

# ================================================
# STEP 4: Create HarmonicGrammar
# ================================================
print("\nInitializing HarmonicGrammar...")
print("This will take 30-60 seconds with optimized tokenization...")

hg = gsc.HarmonicGrammar(
    pcfg=pcfg,
    max_sent_len=18,  # Use 18 for 256GB, 20 for 512GB
    num_fillers=10,
    num_roles=10,
    init_a=0.1
)

mem.checkpoint("After HarmonicGrammar creation")

# ================================================
# STEP 5: Create GscNet
# ================================================
print("\nInitializing GscNet...")
print("Using integration method for equilibrium point...")

net_opts = {
    'ep_method': 'integration',  # Avoid Newton's method bottleneck
    'integration_dur': 50,
    'dt': 0.01
}

net = gsc.GscNet(
    hg=hg,
    bowl_r=10.0,
    **net_opts
)

mem.checkpoint("After GscNet creation")

# ================================================
# STEP 6: Generate Corpus
# ================================================
print("\nGenerating training corpus...")

corpus_size = 25000  # For 1,756 rules
net.generate_corpus(nsamples=corpus_size, max_sent_len=18)

mem.checkpoint("After corpus generation")

# ================================================
# STEP 7: Training
# ================================================
print("\nStarting training...")

train_opts = {
    'num_trials': 500,     # High trials = fewer epochs needed
    'num_epochs': 50,      # Reduced from 750 (500 trials ≈ 15-20x speedup)
    'lr': 0.01,
    'use_jax': True,       # Use JAX-accelerated estimate_prob_inc
    'checkpoint_every': 5,
    'checkpoint_dir': 'checkpoints'
}

# Create checkpoint directory
os.makedirs(train_opts['checkpoint_dir'], exist_ok=True)

# Train with checkpoints
net.train2(**train_opts)

mem.checkpoint("After training completion")

# ================================================
# STEP 8: Summary
# ================================================
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
mem.summary()

# ================================================
# STEP 9: Save Final Model
# ================================================
final_model_path = 'checkpoints/final_model.pkl'
net.save_checkpoint(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

print("\n" + "="*70)
print("All done! Check memory summary above.")
print("="*70)

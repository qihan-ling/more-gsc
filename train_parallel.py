"""
Parallel Training Script for GscNet

This script trains a GscNet model using parallel gradient aggregation.
It uses the ParallelTrainer from only_gscnet_speedup_sap_parallel.py
to distribute gradient computation across multiple workers.

Usage:
    # Single node with multiprocessing:
    python train_parallel.py --workers 4 --backend multiprocessing

    # Single node with Ray:
    python train_parallel.py --workers 8 --backend ray

    # On SLURM cluster with Ray:
    # First: ray start --head --port=6379
    # Then: python train_parallel.py --workers 16 --backend ray --ray-address auto

Author: Generated for parallel training optimization
"""

import argparse
import numpy as np
import time
import os

# Import GscNet and parallel trainer
# Use dtype version for float32 memory optimization (~50% memory savings)
import only_gscnet_speedup_sap_dtype as gsc
from only_gscnet_speedup_sap_parallel import ParallelTrainer, train_parallel
from save_load_model_efficiently import save_model_efficient, load_model_efficient
from scipy import sparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Parallel GscNet Training')
    
    # Parallelization options
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--backend', type=str, default='ray',
                        choices=['ray', 'multiprocessing'],
                        help='Parallel backend: ray or multiprocessing (default: ray)')
    parser.add_argument('--ray-address', type=str, default=None,
                        help='Ray cluster address (use "auto" for SLURM, default: None starts local)')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Base number of trials per epoch (default: 50)')
    parser.add_argument('--scale-trials', action='store_true', default=False,
                        help='Scale trials by num_workers (trials × workers) for more gradient signal')
    parser.add_argument('--lrate', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Model options
    parser.add_argument('--grammar', type=str, default='collapsed_filtered_sm5.grammar',
                        help='Grammar file path')
    parser.add_argument('--sparse', action='store_true', default=True,
                        help='Use sparse WC matrix (default: True)')
    parser.add_argument('--compressed', action='store_true', default=True,
                        help='Use compressed encodings (default: True)')
    parser.add_argument('--dim-f', type=int, default=150,
                        help='Filler encoding dimension (default: 150)')
    parser.add_argument('--dim-r', type=int, default=60,
                        help='Role encoding dimension (default: 60)')
    
    # Checkpointing
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1 = every epoch)')
    parser.add_argument('--output-prefix', type=str, default='parallel_model',
                        help='Output filename prefix (default: parallel_model)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("PARALLEL GSCNET TRAINING")
    print("=" * 70)
    print(f"Workers: {args.workers}")
    print(f"Backend: {args.backend}")
    print(f"Epochs: {args.epochs}")
    print(f"Base trials: {args.trials}" + (f" (× {args.workers} workers = {args.trials * args.workers})" if args.scale_trials else ""))
    print(f"Scale trials: {args.scale_trials}")
    print(f"Learning rate: {args.lrate}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(args.seed)
    
    t0 = time.time()
    
    # =========================================================================
    # Initialize Ray if using ray backend
    # =========================================================================
    if args.backend == 'ray':
        try:
            import ray
            if args.ray_address:
                print(f"Connecting to Ray cluster at: {args.ray_address}")
                ray.init(address=args.ray_address)
            else:
                print("Starting local Ray instance...")
                ray.init(ignore_reinit_error=True)
            print(f"Ray initialized: {ray.cluster_resources()}")
        except ImportError:
            print("WARNING: Ray not installed. Falling back to multiprocessing.")
            args.backend = 'multiprocessing'
    
    # =========================================================================
    # Load or create network
    # =========================================================================
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        
        # Step 1: Load checkpoint state dict
        with open(args.resume, 'rb') as f:
            state = pickle.load(f)
        print(f"  Checkpoint contains: {list(state.keys())}")
        
        # Step 2: Reconstruct HarmonicGrammar from saved config
        print("  Reconstructing HarmonicGrammar...")
        hg = gsc.HarmonicGrammar(
            pcfg=state['hg_pcfg'],
            root=state['hg_root'],
            max_sent_len=state['hg_maxlen']
        )
        
        # Step 3: Get encodings config
        enc_config = state.get('encodings_config', {})
        encodings = {}
        if enc_config.get('similarity') is not None:
            encodings['similarity'] = enc_config['similarity']
        else:
            encodings['similarity'] = hg.get_simlist(dp=0.0)
        
        if enc_config.get('dim_f') is not None:
            encodings['dim_f'] = enc_config['dim_f']
            encodings['dim_r'] = enc_config['dim_r']
        
        # Step 4: Network options
        net_opts = state.get('opts', {})
        net_opts['use_jax'] = False
        if 'use_sparse_wc' not in net_opts and sparse.issparse(state['WC']):
            net_opts['use_sparse_wc'] = True
        
        # Step 5: Create network
        print("  Creating GscNet...")
        seed = state.get('seed', 1024)
        net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=seed)
        
        # Step 6: Load weights and training state
        net = load_model_efficient(args.resume, net)
        print(f"  Loaded model with {net.num_bindings} bindings")
        
        # Step 7: Regenerate corpus (not saved in checkpoint)
        print("  Regenerating corpus...")
        net.generate_corpus(use_freq=True, nsamples=5000)
        print(f"  Corpus sentences: {len(net.corpus['sentence'])}")
        
    else:
        # Load grammar
        print(f"\nLoading grammar from: {args.grammar}")
        with open(args.grammar, 'r') as f:
            PCFG_str = f.read()
        
        ROOT = 'S'
        MAXLEN = 24
        
        # Create HarmonicGrammar
        print("Creating HarmonicGrammar...")
        hg = gsc.HarmonicGrammar(pcfg=PCFG_str, root=ROOT, max_sent_len=MAXLEN)
        print(f"  Fillers: {len(hg.filler_names)}")
        print(f"  Roles: {len(hg.role_names)}")
        
        # Set similarities
        sim = hg.get_simlist(dp=0.0)
        
        # Network options
        net_opts = {
            'use_jax': False,  # Workers use CPU
            'T_init': 0.01,
            'q_max': 15.0,
            'q_init': 0.0,
            'dt_init': 0.02,
            'm': 30,
            'use_runC': True,
            'ep_method': 'integration',
        }
        
        if args.sparse:
            net_opts['use_sparse_wc'] = True
        
        # Encodings
        encodings = {'similarity': sim}
        if args.compressed:
            encodings['dim_f'] = args.dim_f
            encodings['dim_r'] = args.dim_r
        
        # Create network
        print("\nCreating GscNet...")
        net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=1024)
        
        print(f"  num_bindings: {net.num_bindings}")
        print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
        if hasattr(net.WC, 'nnz'):
            print(f"  WC nnz: {net.WC.nnz}")
        
        # Generate corpus
        print("\nGenerating corpus...")
        net.generate_corpus(use_freq=True, nsamples=5000)
        print(f"  Corpus sentences: {len(net.corpus['sentence'])}")
    
    # =========================================================================
    # Initialize training
    # =========================================================================
    
    # Scale trials by number of workers if requested
    effective_trials = args.trials
    if args.scale_trials:
        effective_trials = args.trials * args.workers
        print(f"\n*** SCALED TRIALS: {args.trials} base × {args.workers} workers = {effective_trials} prefixes/epoch ***")
        print(f"    (Same epoch time as serial with {args.trials} trials, but {args.workers}× more gradient signal)")
    
    train_opts = {
        'lrate': args.lrate,
        'num_trials': effective_trials,
        'ema_stat_weight': 0.0,
        'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
        'report_cycle': 10,
        'init_noise_mag': 0.02,
        'average_weight': False,
        'optimizer': 'adam',
        'update_w': True,
        'update_b': False,
        'update_gram_only': False,
        'num_epochs': args.checkpoint_every,  # Will run in blocks
    }
    
    if not hasattr(net, 'train_opts') or net.train_opts is None:
        print("\nInitializing training state...")
        net.initialize(train_opts=train_opts)
    else:
        print("\nUsing existing training state...")
        net.update_train_opts(train_opts)
    
    # =========================================================================
    # Create parallel trainer
    # =========================================================================
    print(f"\nCreating ParallelTrainer with {args.workers} workers...")
    trainer = ParallelTrainer(
        net,
        num_workers=args.workers,
        backend=args.backend,
        verbose=True
    )
    
    # =========================================================================
    # Training loop with checkpointing
    # =========================================================================
    total_epochs = 0
    n_blocks = args.epochs // args.checkpoint_every
    remaining = args.epochs % args.checkpoint_every
    
    if args.checkpoint_every == 1:
        print(f"\nTraining {args.epochs} epochs, saving checkpoint after EVERY epoch")
    else:
        print(f"\nTraining {args.epochs} epochs in {n_blocks} blocks of {args.checkpoint_every}")
        if remaining > 0:
            print(f"  Plus {remaining} remaining epochs")
    
    for block in range(n_blocks):
        block_start = time.time()
        
        # Only print block header if checkpointing less frequently than every epoch
        if args.checkpoint_every > 1:
            print(f"\n{'='*70}")
            print(f"TRAINING BLOCK {block + 1}/{n_blocks}")
            print(f"Epochs {total_epochs + 1} to {total_epochs + args.checkpoint_every}")
            print(f"{'='*70}")
        
        # Train for checkpoint_every epochs
        trainer.train(
            num_epochs=args.checkpoint_every,
            report_every=1  # Always report every epoch
        )
        
        total_epochs += args.checkpoint_every
        block_time = time.time() - block_start
        
        # Save checkpoint
        checkpoint_name = f'{args.output_prefix}_epoch_{total_epochs:04d}.pkl'
        print(f"  Saving: {checkpoint_name} ({block_time:.1f}s, total: {time.time() - t0:.1f}s)")
        save_model_efficient(net, checkpoint_name)
    
    # Train remaining epochs (only happens if epochs % checkpoint_every != 0)
    if remaining > 0:
        print(f"\nTraining final {remaining} epochs...")
        
        for ep in range(remaining):
            trainer.train(num_epochs=1, report_every=1)
            total_epochs += 1
            
            # Save checkpoint for each remaining epoch too
            checkpoint_name = f'{args.output_prefix}_epoch_{total_epochs:04d}.pkl'
            print(f"  Saving: {checkpoint_name}")
            save_model_efficient(net, checkpoint_name)
    
    # =========================================================================
    # Save final model
    # =========================================================================
    final_name = f'{args.output_prefix}_final.pkl'
    print(f"\nSaving final model: {final_name}")
    save_model_efficient(net, final_name)
    
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total epochs: {total_epochs}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Time per epoch: {total_time/total_epochs:.2f}s")
    print(f"Final model saved: {final_name}")
    print(f"{'='*70}")
    
    # Cleanup
    trainer.shutdown()


if __name__ == "__main__":
    main()


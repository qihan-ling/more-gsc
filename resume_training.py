"""
Resume Training from Checkpoint

Usage:
    python resume_training.py --checkpoint sap_checkpoint_epoch_0002.pkl --epochs 500
"""

import argparse
import numpy as np
import time
import pickle
from scipy import sparse
import only_gscnet_speedup_sap as gsc  # Use same version as original training
from save_load_model_efficiently import save_model_efficient, load_model_efficient

def parse_args():
    parser = argparse.ArgumentParser(description='Resume GscNet Training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file to resume from')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Additional epochs to train (default: 500)')
    parser.add_argument('--output-prefix', type=str, default='resumed_model',
                        help='Output filename prefix')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--corpus-samples', type=int, default=5000,
                        help='Number of corpus samples to generate (default: 5000)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Additional epochs: {args.epochs}")
    print(f"Output prefix: {args.output_prefix}")
    print("=" * 70)
    
    t0 = time.time()
    
    # =========================================================================
    # Step 1: Load checkpoint state dict
    # =========================================================================
    print(f"\nLoading checkpoint state: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        state = pickle.load(f)
    
    print(f"  Checkpoint contains: {list(state.keys())}")
    
    # =========================================================================
    # Step 2: Reconstruct HarmonicGrammar and GscNet from saved config
    # =========================================================================
    print("\nReconstructing network from checkpoint config...")
    
    # Reconstruct HarmonicGrammar
    hg = gsc.HarmonicGrammar(
        pcfg=state['hg_pcfg'],
        root=state['hg_root'],
        max_sent_len=state['hg_maxlen']
    )
    print(f"  HarmonicGrammar: {len(hg.filler_names)} fillers, {len(hg.role_names)} roles")
    
    # Get encodings config
    enc_config = state.get('encodings_config', {})
    encodings = {}
    if enc_config.get('similarity') is not None:
        encodings['similarity'] = enc_config['similarity']
    else:
        # Default: no similarity
        encodings['similarity'] = hg.get_simlist(dp=0.0)
    
    if enc_config.get('dim_f') is not None:
        encodings['dim_f'] = enc_config['dim_f']
        encodings['dim_r'] = enc_config['dim_r']
        print(f"  Encodings: dim_f={encodings['dim_f']}, dim_r={encodings['dim_r']}")
    
    # Network options
    net_opts = state.get('opts', {})
    net_opts['use_jax'] = False  # CPU training
    if 'use_sparse_wc' not in net_opts:
        # Check if WC was sparse
        if sparse.issparse(state['WC']):
            net_opts['use_sparse_wc'] = True
    
    # Create network
    seed = state.get('seed', 1024)
    net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=seed)
    print(f"  GscNet created: {net.num_bindings} bindings")
    
    # =========================================================================
    # Step 3: Load weights and training state into network
    # =========================================================================
    print("\nLoading weights and training state...")
    
    # Use the already-loaded state dict instead of loading again
    net.WC = state['WC']
    net.bC = state['bC']
    net.epoch_num = state.get('epoch_num', 0)
    if 'traces_train' in state:
        net.traces_train = state['traces_train']
    if 'train_opts' in state and state['train_opts'] is not None:
        net.train_opts = state['train_opts']
    if 'optim' in state and state['optim'] is not None:
        net.optim = state['optim']
    
    print(f"  ✓ Loaded weights from checkpoint")
    print(f"    Resuming from epoch {net.epoch_num}")
    if hasattr(net.WC, 'nnz'):
        print(f"    WC nnz: {net.WC.nnz:,}")
    
    # Free the state dict to release memory
    del state
    import gc
    gc.collect()
    print(f"  ✓ Freed checkpoint memory")
    
    print(f"  num_bindings: {net.num_bindings}")
    print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
    if hasattr(net.WC, 'nnz'):
        print(f"  WC nnz: {net.WC.nnz:,}")
    
    # Check training state
    if hasattr(net, 'traces_train') and net.traces_train:
        prev_epochs = len(net.traces_train.get('kl_trees', []))
        print(f"  Previous epochs completed: {prev_epochs}")
    else:
        prev_epochs = 0
        print("  No previous training history found")
    
    # =========================================================================
    # Step 4: Regenerate corpus (not saved in checkpoint to save space)
    # =========================================================================
    print(f"\nRegenerating corpus with {args.corpus_samples} samples...")
    print(f"  (Use --corpus-samples to adjust for memory constraints)")
    net.generate_corpus(use_freq=True, nsamples=args.corpus_samples)
    print(f"  Corpus sentences: {len(net.corpus['sentence'])}")
    
    # =========================================================================
    # Step 5: Initialize training state (mask0, optimizer, etc.)
    # =========================================================================
    print("\nInitializing training state...")
    
    # Save values that initialize() might reset
    saved_epoch_num = net.epoch_num if hasattr(net, 'epoch_num') else 0
    saved_traces = net.traces_train if hasattr(net, 'traces_train') else None
    saved_WC = net.WC  # Save the loaded weights
    saved_bC = net.bC
    saved_optim = net.optim if hasattr(net, 'optim') else None
    
    # Training options (keep same as checkpoint or set defaults)
    train_opts = {
        'lrate': 0.1,
        'num_trials': 50,
        'ema_stat_weight': 0.0,
        'trace_varnames': ['kl_trees', 'kl_treelets', 'prob_sent', 'acc'],
        'report_cycle': 10,
        'init_noise_mag': 0.02,
        'average_weight': False,
        'average_filler_bias': False,
    }
    
    # Merge with saved train_opts if available
    if hasattr(net, 'train_opts') and net.train_opts:
        saved_opts = net.train_opts.copy()
        saved_opts.update({'num_epochs': 1})  # Override num_epochs for our loop
        train_opts.update(saved_opts)
    
    # Initialize (this rebuilds mask0, optimizer states, etc.)
    net.initialize(train_opts=train_opts)
    
    # Restore saved state (weights, epoch, traces)
    net.WC = saved_WC
    net.bC = saved_bC
    net.epoch_num = saved_epoch_num
    if saved_traces is not None:
        net.traces_train = saved_traces
    if saved_optim is not None:
        net.optim = saved_optim
    
    print(f"  Training state initialized (resuming from epoch {saved_epoch_num})")
    
    print(f"\nStarting training for {args.epochs} additional epochs...")
    print(f"(Total will be {prev_epochs + args.epochs} epochs)")
    print("=" * 70)
    
    # Training loop with checkpointing
    for epoch in range(args.epochs):
        epoch_num = prev_epochs + epoch
        
        net.train2(
            train_opts={'num_epochs': 1},
            savefilename=None,
        )
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            checkpoint_name = f'{args.output_prefix}_epoch_{epoch_num + 1:04d}.pkl'
            save_model_efficient(net, checkpoint_name)
            
            # Print progress
            if hasattr(net, 'traces_train') and 'kl_trees' in net.traces_train:
                recent_kl = np.mean(net.traces_train['kl_trees'][-10:])
                recent_acc = np.mean(net.traces_train['acc'][-10:])
                elapsed = time.time() - t0
                print(f"Epoch {epoch_num + 1}: KL={recent_kl:.4f}, Acc={recent_acc:.4f}, "
                      f"Time={elapsed:.1f}s, Saved: {checkpoint_name}")
    
    # Save final model
    final_name = f'{args.output_prefix}_final.pkl'
    save_model_efficient(net, final_name)
    
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total epochs: {prev_epochs + args.epochs}")
    print(f"Time for {args.epochs} new epochs: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Final model: {final_name}")
    
    # Final statistics
    if hasattr(net, 'traces_train') and net.traces_train:
        final_kl = np.mean(net.traces_train['kl_trees'][-100:])
        final_acc = np.mean(net.traces_train['acc'][-100:])
        print(f"Final KL: {final_kl:.4f}")
        print(f"Final Acc: {final_acc:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()


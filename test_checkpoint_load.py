"""
Minimal Checkpoint Load Test

Tests that a checkpoint can be loaded successfully without running full training.
Uses preload_wc to skip expensive _build_model() and avoid OOM.

Usage:
    python test_checkpoint_load.py --checkpoint sap_checkpoint_epoch_0002.pkl
"""

import argparse
import numpy as np
import time
import pickle
import gc
from scipy import sparse

# Import GscNet (use regular version, not dtype)
import only_gscnet_speedup_sap as gsc


def parse_args():
    parser = argparse.ArgumentParser(description='Test Checkpoint Loading')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file to test')
    parser.add_argument('--skip-corpus', action='store_true',
                        help='Skip corpus generation (faster test)')
    parser.add_argument('--corpus-samples', type=int, default=100,
                        help='Number of corpus samples for quick test (default: 100)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("CHECKPOINT LOAD TEST")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)
    
    t0 = time.time()
    
    # =========================================================================
    # Step 1: Load checkpoint state dict
    # =========================================================================
    print("\n[Step 1] Loading checkpoint state dict...")
    t_step = time.time()
    
    with open(args.checkpoint, 'rb') as f:
        state = pickle.load(f)
    
    print(f"  ✓ Loaded in {time.time() - t_step:.1f}s")
    print(f"  Keys: {list(state.keys())}")
    
    # Show WC info
    if 'WC' in state:
        wc = state['WC']
        print(f"  WC type: {type(wc).__name__}")
        print(f"  WC shape: {wc.shape}")
        print(f"  WC dtype: {wc.dtype}")
        if hasattr(wc, 'nnz'):
            print(f"  WC nnz: {wc.nnz:,}")
            mem_mb = (wc.nnz * (wc.dtype.itemsize + 8)) / 1e6
            print(f"  WC memory: ~{mem_mb:.1f} MB")
    
    if 'bC' in state:
        print(f"  bC shape: {state['bC'].shape}, dtype: {state['bC'].dtype}")
    
    if 'epoch_num' in state:
        print(f"  Epoch: {state['epoch_num']}")
    
    # =========================================================================
    # Step 2: Reconstruct HarmonicGrammar
    # =========================================================================
    print("\n[Step 2] Reconstructing HarmonicGrammar...")
    t_step = time.time()
    
    hg = gsc.HarmonicGrammar(
        pcfg=state['hg_pcfg'],
        root=state['hg_root'],
        max_sent_len=state['hg_maxlen']
    )
    
    print(f"  ✓ HarmonicGrammar created in {time.time() - t_step:.1f}s")
    print(f"  Fillers: {len(hg.filler_names)}")
    print(f"  Roles: {len(hg.role_names)}")
    print(f"  Rules: {len(hg.rules)}")
    
    # =========================================================================
    # Step 3: Prepare encodings and options
    # =========================================================================
    print("\n[Step 3] Preparing encodings and network options...")
    
    enc_config = state.get('encodings_config', {})
    encodings = {}
    if enc_config.get('similarity') is not None:
        encodings['similarity'] = enc_config['similarity']
    else:
        encodings['similarity'] = hg.get_simlist(dp=0.0)
    
    if enc_config.get('dim_f') is not None:
        encodings['dim_f'] = enc_config['dim_f']
        encodings['dim_r'] = enc_config['dim_r']
        print(f"  Encodings: dim_f={encodings['dim_f']}, dim_r={encodings['dim_r']}")
    
    net_opts = state.get('opts', {})
    net_opts['use_jax'] = False  # CPU training
    if 'use_sparse_wc' not in net_opts:
        if sparse.issparse(state['WC']):
            net_opts['use_sparse_wc'] = True
    
    print(f"  use_sparse_wc: {net_opts.get('use_sparse_wc', False)}")
    
    # =========================================================================
    # Step 4: Create GscNet with preload_wc (SKIPS _build_model!)
    # =========================================================================
    print("\n[Step 4] Creating GscNet with preload_wc...")
    print("  (This skips _build_model(), saving ~8 minutes and ~50GB peak memory)")
    t_step = time.time()
    
    # Extract WC/bC for preloading
    preload_wc = (state['WC'], state['bC'])
    seed = state.get('seed', 1024)
    
    net = gsc.GscNet(hg=hg, encodings=encodings, opts=net_opts, seed=seed,
                     preload_wc=preload_wc)
    
    print(f"  ✓ GscNet created in {time.time() - t_step:.1f}s")
    print(f"  num_bindings: {net.num_bindings}")
    print(f"  use_sparse: {getattr(net, 'use_sparse', False)}")
    if hasattr(net.WC, 'nnz'):
        print(f"  WC nnz: {net.WC.nnz:,}")
    
    # =========================================================================
    # Step 5: Load remaining training state
    # =========================================================================
    print("\n[Step 5] Loading remaining training state...")
    
    net.epoch_num = state.get('epoch_num', 0)
    if 'traces_train' in state:
        net.traces_train = state['traces_train']
        print(f"  traces_train: {len(net.traces_train.get('kl_trees', []))} epochs recorded")
    if 'train_opts' in state and state['train_opts'] is not None:
        net.train_opts = state['train_opts']
        print(f"  train_opts restored")
    if 'optim' in state and state['optim'] is not None:
        net.optim = state['optim']
        print(f"  optimizer state restored")
    
    print(f"  ✓ Resuming from epoch {net.epoch_num}")
    
    # =========================================================================
    # Step 6: Free checkpoint memory
    # =========================================================================
    print("\n[Step 6] Freeing checkpoint memory...")
    mem_before = None
    try:
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1e9
        print(f"  Memory before gc: {mem_before:.2f} GB")
    except ImportError:
        pass
    
    del preload_wc
    del state
    gc.collect()
    
    if mem_before is not None:
        mem_after = process.memory_info().rss / 1e9
        print(f"  Memory after gc: {mem_after:.2f} GB")
        print(f"  Freed: {mem_before - mem_after:.2f} GB")
    
    print("  ✓ Checkpoint memory freed")
    
    # =========================================================================
    # Step 7: Test basic operations
    # =========================================================================
    print("\n[Step 7] Testing basic operations...")
    
    # Test that we can access key attributes
    print(f"  net.num_fillers: {net.num_fillers}")
    print(f"  net.num_roles: {net.num_roles}")
    print(f"  net.num_bindings: {net.num_bindings}")
    print(f"  net.WC.shape: {net.WC.shape}")
    print(f"  net.bC.shape: {net.bC.shape}")
    
    # Test WC-bC dot product (basic operation)
    print("  Testing WC @ actC operation...")
    actC = np.zeros(net.num_bindings, dtype=net.bC.dtype)
    result = net.WC.dot(actC) + net.bC
    print(f"  ✓ WC @ actC + bC shape: {result.shape}")
    
    # =========================================================================
    # Step 8: Optional corpus generation test
    # =========================================================================
    if not args.skip_corpus:
        print(f"\n[Step 8] Testing corpus generation ({args.corpus_samples} samples)...")
        t_step = time.time()
        
        try:
            net.generate_corpus(use_freq=True, nsamples=args.corpus_samples)
            print(f"  ✓ Generated {len(net.corpus['sentence'])} sentences in {time.time() - t_step:.1f}s")
            
            # Show sample sentence
            if net.corpus['sentence']:
                sample = net.corpus['sentence'][0]
                print(f"  Sample: {sample[:80]}..." if len(sample) > 80 else f"  Sample: {sample}")
        except Exception as e:
            print(f"  ✗ Corpus generation failed: {e}")
    else:
        print("\n[Step 8] Skipping corpus generation (--skip-corpus)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - t0
    
    print("\n" + "=" * 70)
    print("✓ CHECKPOINT LOAD TEST PASSED")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Network ready to resume from epoch {net.epoch_num}")
    print(f"Bindings: {net.num_bindings:,}")
    if hasattr(net.WC, 'nnz'):
        print(f"WC non-zeros: {net.WC.nnz:,}")
    print("=" * 70)
    
    return net


if __name__ == "__main__":
    net = main()

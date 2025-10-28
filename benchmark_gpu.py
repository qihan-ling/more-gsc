"""
Benchmark script to compare CPU vs GPU performance

This script runs a shorter version of the training to measure speedup.

Usage:
    python benchmark_gpu.py --mode [cpu|gpu|both] --epochs [N]
"""

import argparse
import time
import sys
import numpy as np_cpu

def benchmark_cpu(n_epochs=100):
    """Benchmark CPU version"""
    print("\n" + "="*70)
    print("BENCHMARKING CPU VERSION")
    print("="*70)

    import gsc

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

    start_time = time.time()

    # Initialize
    hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
    sim = hg.get_simlist(dp=0.0)

    net_opts = {
        'T_init': 0.01,
        'q_max': 15.0,
        'q_init': 0.0,
        'dt_init': 0.005,
        'm': 30,
        'use_runC': True,
    }

    net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                     opts=net_opts, seed=1024)
    net.generate_corpus(use_freq=True)

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

    net.initialize(train_opts=train_opts)

    init_time = time.time() - start_time
    print(f"Initialization: {init_time:.2f}s")

    # Training
    train_start = time.time()
    epoch_times = []

    for epoch_block in range(n_epochs // 10):
        block_start = time.time()
        net.train2(train_opts={'num_epochs': 10})
        block_time = time.time() - block_start
        epoch_times.append(block_time)

        epochs_done = (epoch_block + 1) * 10
        print(f"  Epochs {epochs_done}/{n_epochs}: {block_time:.2f}s")

    train_time = time.time() - train_start
    total_time = time.time() - start_time

    # Results
    final_kl = np_cpu.mean(net.traces_train['kl_trees'][-10:])
    final_acc = np_cpu.mean(net.traces_train['acc'][-10:])

    return {
        'mode': 'CPU',
        'init_time': init_time,
        'train_time': train_time,
        'total_time': total_time,
        'avg_time_per_10_epochs': np_cpu.mean(epoch_times),
        'final_kl': final_kl,
        'final_acc': final_acc,
        'epochs': n_epochs
    }

def benchmark_gpu(n_epochs=100):
    """Benchmark GPU version"""
    print("\n" + "="*70)
    print("BENCHMARKING GPU VERSION")
    print("="*70)

    import gsc_gpu as gsc

    if not gsc.GPU_AVAILABLE:
        print("ERROR: GPU not available!")
        return None

    gsc.print_gpu_memory()

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

    start_time = time.time()

    # Initialize
    hg = gsc.HarmonicGrammar(pcfg=PCFG_G1, root='S', max_sent_len=5)
    sim = hg.get_simlist(dp=0.0)

    net_opts = {
        'T_init': 0.01,
        'q_max': 15.0,
        'q_init': 0.0,
        'dt_init': 0.005,
        'm': 30,
        'use_runC': True,
    }

    net = gsc.GscNet(hg=hg, encodings={'similarity': sim},
                     opts=net_opts, seed=1024)
    net.generate_corpus(use_freq=True)

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

    net.initialize(train_opts=train_opts)

    init_time = time.time() - start_time
    print(f"Initialization: {init_time:.2f}s")
    gsc.print_gpu_memory()

    # Training
    train_start = time.time()
    epoch_times = []

    for epoch_block in range(n_epochs // 10):
        block_start = time.time()
        net.train2(train_opts={'num_epochs': 10})
        block_time = time.time() - block_start
        epoch_times.append(block_time)

        epochs_done = (epoch_block + 1) * 10
        print(f"  Epochs {epochs_done}/{n_epochs}: {block_time:.2f}s")

    # Synchronize GPU before final timing
    if gsc.GPU_AVAILABLE:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

    train_time = time.time() - train_start
    total_time = time.time() - start_time

    # Results (convert from GPU to CPU for printing)
    final_kl = np_cpu.mean(gsc.to_cpu(net.traces_train['kl_trees'][-10:]))
    final_acc = np_cpu.mean(gsc.to_cpu(net.traces_train['acc'][-10:]))

    gsc.print_gpu_memory()

    return {
        'mode': 'GPU',
        'init_time': init_time,
        'train_time': train_time,
        'total_time': total_time,
        'avg_time_per_10_epochs': np_cpu.mean(epoch_times),
        'final_kl': final_kl,
        'final_acc': final_acc,
        'epochs': n_epochs
    }

def print_comparison(cpu_result, gpu_result):
    """Print comparison table"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    if cpu_result:
        print(f"\nCPU Results ({cpu_result['epochs']} epochs):")
        print(f"  Init time:     {cpu_result['init_time']:.2f}s")
        print(f"  Train time:    {cpu_result['train_time']:.2f}s ({cpu_result['train_time']/60:.2f}m)")
        print(f"  Total time:    {cpu_result['total_time']:.2f}s ({cpu_result['total_time']/60:.2f}m)")
        print(f"  Avg/10 epochs: {cpu_result['avg_time_per_10_epochs']:.2f}s")
        print(f"  Final KL:      {cpu_result['final_kl']:.4f}")
        print(f"  Final Acc:     {cpu_result['final_acc']:.4f}")

    if gpu_result:
        print(f"\nGPU Results ({gpu_result['epochs']} epochs):")
        print(f"  Init time:     {gpu_result['init_time']:.2f}s")
        print(f"  Train time:    {gpu_result['train_time']:.2f}s ({gpu_result['train_time']/60:.2f}m)")
        print(f"  Total time:    {gpu_result['total_time']:.2f}s ({gpu_result['total_time']/60:.2f}m)")
        print(f"  Avg/10 epochs: {gpu_result['avg_time_per_10_epochs']:.2f}s")
        print(f"  Final KL:      {gpu_result['final_kl']:.4f}")
        print(f"  Final Acc:     {gpu_result['final_acc']:.4f}")

    if cpu_result and gpu_result:
        speedup_init = cpu_result['init_time'] / gpu_result['init_time']
        speedup_train = cpu_result['train_time'] / gpu_result['train_time']
        speedup_total = cpu_result['total_time'] / gpu_result['total_time']

        print("\n" + "-"*70)
        print("SPEEDUP (CPU time / GPU time):")
        print(f"  Initialization: {speedup_init:.2f}x")
        print(f"  Training:       {speedup_train:.2f}x {'⚡' if speedup_train > 1 else '⚠️'}")
        print(f"  Total:          {speedup_total:.2f}x {'⚡' if speedup_total > 1 else '⚠️'}")

        # Estimate full run time
        if cpu_result['epochs'] < 1000:
            scale_factor = 1000 / cpu_result['epochs']
            est_cpu_full = cpu_result['train_time'] * scale_factor / 60
            est_gpu_full = gpu_result['train_time'] * scale_factor / 60
            print("\n" + "-"*70)
            print("ESTIMATED TIME FOR 1000 EPOCHS:")
            print(f"  CPU: {est_cpu_full:.1f} minutes")
            print(f"  GPU: {est_gpu_full:.1f} minutes")
            print(f"  Time saved: {est_cpu_full - est_gpu_full:.1f} minutes")

    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU performance')
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'both'], default='both',
                        help='Which version to benchmark (default: both)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')

    args = parser.parse_args()

    cpu_result = None
    gpu_result = None

    if args.mode in ['cpu', 'both']:
        try:
            cpu_result = benchmark_cpu(args.epochs)
        except Exception as e:
            print(f"CPU benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    if args.mode in ['gpu', 'both']:
        try:
            gpu_result = benchmark_gpu(args.epochs)
        except Exception as e:
            print(f"GPU benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    if cpu_result or gpu_result:
        print_comparison(cpu_result, gpu_result)
    else:
        print("No benchmarks completed successfully!")
        sys.exit(1)

if __name__ == '__main__':
    main()

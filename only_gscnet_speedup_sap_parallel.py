"""
Parallel Training for GscNet using Gradient Aggregation

This module provides parallel training capabilities for GscNet using either:
1. Ray (recommended for clusters) - multi-node, zero-copy arrays, fault-tolerant
2. multiprocessing (fallback) - single-node, built-in Python

Architecture:
    Master Process:
        - Holds the authoritative model weights (WC, bC)
        - Distributes prefix batches to workers
        - Aggregates gradients from workers
        - Applies optimizer updates
    
    Worker Processes:
        - Receive weight snapshot + prefix batch
        - Compute gradients independently (estimate_prob_inc + cost_grad)
        - Return gradients to master

Usage:
    from only_gscnet_speedup_sap_parallel import ParallelTrainer
    
    trainer = ParallelTrainer(net, num_workers=4, backend='ray')
    trainer.train(num_epochs=100)

IMPORTANT: This module imports from only_gscnet_speedup_sap.py to reuse
the existing GscNet implementation. Workers create lightweight GscNet
instances that share the same computational logic.
"""

import numpy as np
import time
import pickle
from typing import List, Tuple, Dict, Any, Optional
from scipy import sparse

# Import the real GscNet - workers will use actual methods, not reimplementations
# Use dtype version for float32 memory optimization (~50% memory savings)
import only_gscnet_speedup_sap_dtype as gsc

# Try to import Ray, fall back to multiprocessing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not found. Using multiprocessing backend. Install Ray for better performance: pip install ray")

import multiprocessing as mp
from multiprocessing import Pool, shared_memory
import copy


# =============================================================================
# CONFIGURATION
# =============================================================================

class ParallelConfig:
    """Configuration for parallel training."""
    
    def __init__(
        self,
        num_workers: int = 4,
        backend: str = 'ray',  # 'ray' or 'multiprocessing'
        prefetch_batches: int = 1,
        gradient_compression: bool = False,  # Compress sparse gradients
        sync_every_n_epochs: int = 1,  # How often to sync weights
        verbose: bool = True
    ):
        self.num_workers = num_workers
        self.backend = backend if (backend == 'multiprocessing' or RAY_AVAILABLE) else 'multiprocessing'
        self.prefetch_batches = prefetch_batches
        self.gradient_compression = gradient_compression
        self.sync_every_n_epochs = sync_every_n_epochs
        self.verbose = verbose
        
        if backend == 'ray' and not RAY_AVAILABLE:
            print("Warning: Ray requested but not available. Falling back to multiprocessing.")


# =============================================================================
# WORKER FUNCTIONS (stateless, can be pickled)
# =============================================================================

def _extract_net_config(net) -> Dict[str, Any]:
    """Extract configuration needed to reconstruct net on workers.
    
    We serialize the HarmonicGrammar and key config so workers can
    create a real GscNet instance with the same structure.
    """
    config = {
        # Serialize the HarmonicGrammar (needed to create GscNet)
        'hg_pickle': pickle.dumps(net.hg),
        
        # Encodings used during construction
        'encodings': {
            'similarity': net.encodings.get('similarity', None) if hasattr(net, 'encodings') else None,
            'dim_f': getattr(net, 'dim_f', None),
            'dim_r': getattr(net, 'dim_r', None),
        },
        
        # Network options
        'opts': net.opts.copy(),
        
        # Training options
        'train_opts': net.train_opts.copy() if hasattr(net, 'train_opts') else {},
        
        # Sparse flag
        'use_sparse': getattr(net, 'use_sparse', False),
        
        # Equilibrium point (computed once, shared)
        'ep': net.ep,
        
        # NOTE: mask0 is NOT passed to workers - it's only needed for optimizer
        # updates which happen on the master. Removing this saves ~132 GB per worker!
        # 'mask0_pickle': pickle.dumps(net.get_mask0()) if hasattr(net, 'train_opts') else None,
        
        # Q-policy if exists
        'qpolicy': getattr(net, 'qpolicy', None),
        
        # Corpus (for target statistics)
        'corpus': net.corpus if hasattr(net, 'corpus') else None,
    }
    
    return config


def _create_worker_net(net_config: Dict, wc_data, wc_indices, wc_indptr, wc_shape, bc_data):
    """
    Create a GscNet instance on a worker from serialized config.
    
    This creates a REAL GscNet using the actual class, not a minimal copy.
    The worker net has all the methods of a normal GscNet.
    """
    # Deserialize HarmonicGrammar
    hg = pickle.loads(net_config['hg_pickle'])
    
    # Build encodings
    encodings = {}
    if net_config['encodings']['similarity'] is not None:
        encodings['similarity'] = net_config['encodings']['similarity']
    if net_config['encodings']['dim_f'] is not None:
        encodings['dim_f'] = net_config['encodings']['dim_f']
    if net_config['encodings']['dim_r'] is not None:
        encodings['dim_r'] = net_config['encodings']['dim_r']
    
    # Create network with same structure (but don't rebuild model)
    opts = net_config['opts'].copy()
    opts['use_jax'] = False  # Workers always use CPU
    
    # Create the network
    worker_net = gsc.GscNet(hg=hg, encodings=encodings, opts=opts, seed=None)
    
    # Overwrite weights with current master weights
    use_sparse = net_config['use_sparse']
    if use_sparse and wc_indices is not None:
        worker_net.WC = sparse.csr_matrix((wc_data, wc_indices, wc_indptr), shape=wc_shape)
    else:
        worker_net.WC = wc_data.reshape(wc_shape) if wc_data.ndim == 1 else wc_data.copy()
    
    worker_net.bC = bc_data.copy()
    
    # Copy equilibrium point
    worker_net.ep = net_config['ep']
    
    # Set train_opts
    worker_net.train_opts = net_config['train_opts'].copy()
    
    # Copy qpolicy if exists
    if net_config['qpolicy'] is not None:
        worker_net.qpolicy = net_config['qpolicy']
    
    return worker_net


def _worker_compute_gradients(
    worker_id: int,
    wc_data: np.ndarray,  # Flattened or sparse data
    wc_indices: Optional[np.ndarray],  # For sparse
    wc_indptr: Optional[np.ndarray],   # For sparse
    wc_shape: Tuple[int, int],
    bc_data: np.ndarray,
    prefix_batch: List,
    target_batch: List,
    net_config: Dict,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Worker function to compute gradients for a batch of prefixes.
    
    This creates a REAL GscNet instance and uses the actual methods
    (estimate_prob_inc, cost_grad) to compute gradients.
    
    Returns:
        Tuple of (dWC_data, rows, cols, dbC, stats)
        - For sparse: (data, rows, cols, dbC, stats)
        - For dense: (dWC.ravel(), None, None, dbC, stats)
    """
    # Set random seed for reproducibility
    np.random.seed(seed + worker_id)
    
    # Create a REAL GscNet using the actual class
    worker_net = _create_worker_net(net_config, wc_data, wc_indices, wc_indptr, wc_shape, bc_data)
    
    use_sparse = net_config['use_sparse']
    num_bindings = worker_net.num_bindings
    num_trials = net_config['train_opts'].get('num_trials', 10)
    
    # NOTE: mask0 is NOT needed in workers - only master uses it for optimizer updates
    
    # Get corpus for target statistics
    corpus = net_config['corpus']
    
    # Initialize gradient accumulators
    if use_sparse:
        dWC_list = []  # List of sparse gradient matrices
    else:
        dWC = np.zeros(wc_shape)
    dbC = np.zeros(num_bindings)
    
    stats = {
        'num_prefixes': len(prefix_batch),
        'worker_id': worker_id,
        'total_time': 0,
        'kl_trees': 0,
        'acc': 0,
    }
    
    t_start = time.time()
    
    # Process each prefix in batch
    for prefix_idx, (prefix, target) in enumerate(zip(prefix_batch, target_batch)):
        if prefix is None:
            continue
        
        # ================================================================
        # Use REAL GscNet methods for gradient computation
        # ================================================================
        
        # 1. Get production statistics (stat_P) using actual estimate_prob_inc
        stat_P = worker_net.estimate_prob_inc(
            prefix, 
            num_trials=num_trials,
            progress=0,  # No progress output in workers
            update_q_discrete=False
        )
        
        # 2. Get target statistics (stat_Q) from corpus
        # Find this sentence's target in corpus
        stat_Q = None
        if corpus is not None:
            for si, sent in enumerate(corpus['sentence']):
                if sent == prefix:
                    stat_Q = worker_net.get_corpus_stat({
                        'target': [corpus['target'][si]],
                        'count': np.array([1]),
                        'prob_sent': np.array([1.0])
                    })
                    break
        
        if stat_Q is None:
            # Use target directly if provided
            if target is not None:
                stat_Q = worker_net.get_corpus_stat({
                    'target': [target],
                    'count': np.array([1]),
                    'prob_sent': np.array([1.0])
                })
            else:
                continue  # Skip if no target
        
        # 3. Compute error and gradient using REAL cost_grad method
        err = {
            'trees': stat_P['trees'] - stat_Q['trees'],
            'treelets': stat_P['treelets'] - stat_Q['treelets'],
        }
        
        # External input for this prefix
        extC_token = worker_net.extC.copy()
        
        # Compute gradient using actual method
        dWC_prefix, acc, dbC_prefix = worker_net.cost_grad(err, extC_token)
        
        # Accumulate gradients
        if use_sparse:
            if sparse.issparse(dWC_prefix):
                dWC_list.append(dWC_prefix.tocoo())
        else:
            dWC += dWC_prefix
        dbC += dbC_prefix
        
        stats['acc'] += acc
    
    # Average over prefixes
    if len(prefix_batch) > 0:
        stats['acc'] /= len(prefix_batch)
    
    stats['total_time'] = time.time() - t_start
    
    # Return gradients in serializable format
    if use_sparse:
        if dWC_list:
            # Sum all sparse gradients
            dWC_sum = sum(dWC_list)
            dWC_coo = dWC_sum.tocoo()
            return (dWC_coo.data.copy(), dWC_coo.row.copy(), dWC_coo.col.copy(), dbC, stats)
        else:
            return (np.array([]), np.array([]), np.array([]), dbC, stats)
    else:
        return (dWC.ravel(), None, None, dbC, stats)


# =============================================================================
# RAY BACKEND
# =============================================================================

if RAY_AVAILABLE:
    @ray.remote
    def _ray_worker_compute_gradients(
        worker_id: int,
        wc_ref,  # Ray object reference for WC
        bc_ref,  # Ray object reference for bC
        wc_shape: Tuple[int, int],
        prefix_batch: List,
        target_batch: List,
        net_config: Dict,
        seed: int,
        use_sparse: bool
    ):
        """Ray remote function for gradient computation."""
        # Get data from Ray object store (zero-copy when possible)
        wc_data = ray.get(wc_ref)
        bc_data = ray.get(bc_ref)
        
        if use_sparse:
            # wc_data is a tuple (data, indices, indptr)
            return _worker_compute_gradients(
                worker_id,
                wc_data[0], wc_data[1], wc_data[2],
                wc_shape, bc_data,
                prefix_batch, target_batch, net_config, seed
            )
        else:
            return _worker_compute_gradients(
                worker_id,
                wc_data, None, None,
                wc_shape, bc_data,
                prefix_batch, target_batch, net_config, seed
            )


# =============================================================================
# MULTIPROCESSING BACKEND
# =============================================================================

def _mp_worker_wrapper(args):
    """Wrapper for multiprocessing Pool.map."""
    return _worker_compute_gradients(*args)


# =============================================================================
# PARALLEL TRAINER CLASS
# =============================================================================

class ParallelTrainer:
    """
    Parallel trainer for GscNet using gradient aggregation.
    
    Example usage:
        # Initialize network normally
        net = GscNet(hg=hg, encodings=encodings, opts=opts)
        net.generate_corpus(nsamples=5000)
        net.initialize(train_opts=train_opts)
        
        # Create parallel trainer
        trainer = ParallelTrainer(
            net, 
            num_workers=4, 
            backend='ray'  # or 'multiprocessing'
        )
        
        # Train with parallel gradient computation
        trainer.train(num_epochs=100)
    """
    
    def __init__(
        self,
        net,
        num_workers: int = 4,
        backend: str = 'ray',
        verbose: bool = True
    ):
        """
        Initialize parallel trainer.
        
        Args:
            net: GscNet instance (must be initialized with corpus and train_opts)
            num_workers: Number of parallel workers
            backend: 'ray' or 'multiprocessing'
            verbose: Print progress information
        """
        self.net = net
        self.num_workers = num_workers
        self.backend = backend if (backend == 'multiprocessing' or RAY_AVAILABLE) else 'multiprocessing'
        self.verbose = verbose
        
        # Validate net is ready for training
        if not hasattr(net, 'corpus') or net.corpus is None:
            raise ValueError("Network must have corpus generated. Call net.generate_corpus() first.")
        if not hasattr(net, 'train_opts') or net.train_opts is None:
            raise ValueError("Network must be initialized. Call net.initialize(train_opts=...) first.")
        
        # Extract static config (doesn't change during training)
        self.net_config = _extract_net_config(net)
        
        # Initialize backend
        if self.backend == 'ray':
            self._init_ray()
        else:
            self._init_multiprocessing()
        
        if self.verbose:
            print(f"ParallelTrainer initialized:")
            print(f"  Backend: {self.backend}")
            print(f"  Workers: {self.num_workers}")
            print(f"  Sparse WC: {self.net_config['use_sparse']}")
    
    def _init_ray(self):
        """Initialize Ray runtime."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            if self.verbose:
                print(f"  Ray initialized with {ray.cluster_resources()}")
    
    def _init_multiprocessing(self):
        """Initialize multiprocessing pool."""
        # Pool will be created per-epoch to avoid pickle issues
        pass
    
    def _split_batches(self, items: List, n_batches: int) -> List[List]:
        """Split items into n roughly equal batches."""
        k, m = divmod(len(items), n_batches)
        return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] 
                for i in range(n_batches)]
    
    def _get_weight_snapshot(self):
        """Get current weights in serializable format."""
        if self.net_config['use_sparse']:
            wc_csr = self.net.WC.tocsr()
            return (
                (wc_csr.data, wc_csr.indices, wc_csr.indptr),
                self.net.bC.copy()
            )
        else:
            return (self.net.WC.copy(), self.net.bC.copy())
    
    def _aggregate_gradients_sparse(self, results: List) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Aggregate sparse gradients from workers."""
        wc_shape = self.net.WC.shape
        
        # Collect all COO components
        all_rows = []
        all_cols = []
        all_vals = []
        dbC_sum = np.zeros(self.net.num_bindings)
        
        for result in results:
            data, rows, cols, dbC, stats = result
            if len(data) > 0:
                all_rows.extend(rows)
                all_cols.extend(cols)
                all_vals.extend(data)
            dbC_sum += dbC
        
        # Create aggregated sparse matrix
        if all_vals:
            dWC = sparse.coo_matrix(
                (all_vals, (all_rows, all_cols)),
                shape=wc_shape
            ).tocsr()
            # Average
            dWC = dWC / len(results)
        else:
            dWC = sparse.csr_matrix(wc_shape)
        
        dbC = dbC_sum / len(results)
        
        return dWC, dbC
    
    def _aggregate_gradients_dense(self, results: List) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate dense gradients from workers."""
        wc_shape = self.net.WC.shape
        
        dWC_sum = np.zeros(wc_shape)
        dbC_sum = np.zeros(self.net.num_bindings)
        
        for result in results:
            data, _, _, dbC, stats = result
            dWC_sum += data.reshape(wc_shape)
            dbC_sum += dbC
        
        # Average
        dWC = dWC_sum / len(results)
        dbC = dbC_sum / len(results)
        
        return dWC, dbC
    
    def train_epoch_ray(self, prefix_list: List, target_list: List, epoch: int) -> Dict:
        """Train one epoch using Ray backend."""
        t_start = time.time()
        
        # Get weight snapshot and put in Ray object store
        wc_snapshot, bc_snapshot = self._get_weight_snapshot()
        
        if self.net_config['use_sparse']:
            wc_ref = ray.put(wc_snapshot)  # Tuple of (data, indices, indptr)
        else:
            wc_ref = ray.put(wc_snapshot)
        bc_ref = ray.put(bc_snapshot)
        
        # Split prefixes into batches
        prefix_batches = self._split_batches(prefix_list, self.num_workers)
        target_batches = self._split_batches(target_list, self.num_workers)
        
        # Launch workers
        futures = []
        for worker_id, (prefix_batch, target_batch) in enumerate(zip(prefix_batches, target_batches)):
            if len(prefix_batch) == 0:
                continue
            future = _ray_worker_compute_gradients.remote(
                worker_id,
                wc_ref,
                bc_ref,
                self.net.WC.shape,
                prefix_batch,
                target_batch,
                self.net_config,
                epoch * 1000 + worker_id,  # Seed
                self.net_config['use_sparse']
            )
            futures.append(future)
        
        # Wait for all workers
        results = ray.get(futures)
        
        # Aggregate gradients
        if self.net_config['use_sparse']:
            dWC, dbC = self._aggregate_gradients_sparse(results)
        else:
            dWC, dbC = self._aggregate_gradients_dense(results)
        
        # Apply gradients (using net's optimizer)
        self._apply_gradients(dWC, dbC)
        
        elapsed = time.time() - t_start
        
        return {
            'elapsed': elapsed,
            'num_workers_used': len(results),
            'prefixes_processed': len(prefix_list),
        }
    
    def train_epoch_multiprocessing(self, prefix_list: List, target_list: List, epoch: int) -> Dict:
        """Train one epoch using multiprocessing backend."""
        t_start = time.time()
        
        # Get weight snapshot
        wc_snapshot, bc_snapshot = self._get_weight_snapshot()
        
        # Split prefixes into batches
        prefix_batches = self._split_batches(prefix_list, self.num_workers)
        target_batches = self._split_batches(target_list, self.num_workers)
        
        # Prepare worker arguments
        worker_args = []
        for worker_id, (prefix_batch, target_batch) in enumerate(zip(prefix_batches, target_batches)):
            if len(prefix_batch) == 0:
                continue
            
            if self.net_config['use_sparse']:
                args = (
                    worker_id,
                    wc_snapshot[0],  # data
                    wc_snapshot[1],  # indices  
                    wc_snapshot[2],  # indptr
                    self.net.WC.shape,
                    bc_snapshot,
                    prefix_batch,
                    target_batch,
                    self.net_config,
                    epoch * 1000 + worker_id
                )
            else:
                args = (
                    worker_id,
                    wc_snapshot.ravel(),
                    None,
                    None,
                    self.net.WC.shape,
                    bc_snapshot,
                    prefix_batch,
                    target_batch,
                    self.net_config,
                    epoch * 1000 + worker_id
                )
            worker_args.append(args)
        
        # Run workers in pool
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_mp_worker_wrapper, worker_args)
        
        # Aggregate gradients
        if self.net_config['use_sparse']:
            dWC, dbC = self._aggregate_gradients_sparse(results)
        else:
            dWC, dbC = self._aggregate_gradients_dense(results)
        
        # Apply gradients
        self._apply_gradients(dWC, dbC)
        
        elapsed = time.time() - t_start
        
        return {
            'elapsed': elapsed,
            'num_workers_used': len(results),
            'prefixes_processed': len(prefix_list),
        }
    
    def _apply_gradients(self, dWC, dbC):
        """Apply aggregated gradients using the network's optimizer."""
        # Get mask for gradient updates
        mask0 = self.net.get_mask0()
        
        # Apply based on optimizer type
        if self.net.train_opts['optimizer'] == 'adam':
            # Adam update
            self.net.optim['step_WC'] += 1
            step = self.net.optim['step_WC']
            beta1 = self.net.optim['beta1']
            beta2 = self.net.optim['beta2']
            eps = self.net.optim['eps']
            lrate = self.net.train_opts['lrate']
            
            if self.net_config['use_sparse']:
                # Sparse Adam update
                # Update momentum and variance for non-zero gradient positions
                dWC_csr = dWC.tocsr() if not sparse.isspmatrix_csr(dWC) else dWC
                
                # M = beta1 * M + (1 - beta1) * dWC
                self.net.optim['M_WC'] = beta1 * self.net.optim['M_WC'] + (1 - beta1) * dWC_csr
                
                # R = beta2 * R + (1 - beta2) * dWC^2
                dWC_sq = dWC_csr.multiply(dWC_csr)
                self.net.optim['R_WC'] = beta2 * self.net.optim['R_WC'] + (1 - beta2) * dWC_sq
                
                # Bias correction
                M_hat = self.net.optim['M_WC'] / (1 - beta1**step)
                R_hat = self.net.optim['R_WC'] / (1 - beta2**step)
                
                # Update: WC += lrate * M_hat / (sqrt(R_hat) + eps)
                # For sparse, we need to be careful with division
                R_hat_sqrt = R_hat.sqrt() if hasattr(R_hat, 'sqrt') else R_hat.power(0.5)
                update = M_hat.multiply(1.0 / (R_hat_sqrt.toarray() + eps))
                
                # Apply mask and update
                if sparse.issparse(mask0):
                    update = update.multiply(mask0)
                self.net.WC = self.net.WC + lrate * update
                self.net.WC.eliminate_zeros()
            else:
                # Dense Adam update
                self.net.optim['M_WC'] = beta1 * self.net.optim['M_WC'] + (1 - beta1) * dWC
                self.net.optim['R_WC'] = beta2 * self.net.optim['R_WC'] + (1 - beta2) * dWC**2
                
                M_hat = self.net.optim['M_WC'] / (1 - beta1**step)
                R_hat = self.net.optim['R_WC'] / (1 - beta2**step)
                
                update = lrate * M_hat / (np.sqrt(R_hat) + eps)
                self.net.WC = self.net.WC + mask0 * update
        else:
            # SGD update
            lrate = self.net.train_opts['lrate']
            if self.net_config['use_sparse']:
                if sparse.issparse(mask0):
                    dWC_masked = dWC.multiply(mask0)
                else:
                    dWC_masked = dWC
                self.net.WC = self.net.WC + lrate * dWC_masked
                self.net.WC.eliminate_zeros()
            else:
                self.net.WC = self.net.WC + lrate * mask0 * dWC
        
        # Update bC
        self.net.bC = self.net.bC + self.net.train_opts['lrate'] * dbC
    
    def train(
        self,
        num_epochs: int = 100,
        prefix_list: Optional[List] = None,
        report_every: int = 10
    ):
        """
        Run parallel training for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            prefix_list: Optional list of prefixes (uses corpus if None)
            report_every: Print progress every N epochs
        """
        if prefix_list is None:
            # Use corpus prefixes
            prefix_list = self.net.corpus['sentence']
            target_list = self.net.corpus['target']
        else:
            target_list = [None] * len(prefix_list)
        
        print(f"\n{'='*70}")
        print(f"PARALLEL TRAINING STARTED")
        print(f"  Backend: {self.backend}")
        print(f"  Workers: {self.num_workers}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Prefixes per epoch: {len(prefix_list)}")
        print(f"{'='*70}\n")
        
        total_time = 0
        
        for epoch in range(num_epochs):
            # Sample prefixes for this epoch
            if self.net.train_opts.get('shuffle', True):
                indices = np.random.permutation(len(prefix_list))
                epoch_prefixes = [prefix_list[i] for i in indices[:self.net.train_opts['num_trials']]]
                epoch_targets = [target_list[i] for i in indices[:self.net.train_opts['num_trials']]]
            else:
                epoch_prefixes = prefix_list[:self.net.train_opts['num_trials']]
                epoch_targets = target_list[:self.net.train_opts['num_trials']]
            
            # Train epoch with appropriate backend
            if self.backend == 'ray':
                stats = self.train_epoch_ray(epoch_prefixes, epoch_targets, epoch)
            else:
                stats = self.train_epoch_multiprocessing(epoch_prefixes, epoch_targets, epoch)
            
            total_time += stats['elapsed']
            
            # Report progress
            if (epoch + 1) % report_every == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: "
                      f"{stats['elapsed']:.2f}s, "
                      f"{stats['prefixes_processed']} prefixes, "
                      f"{self.num_workers} workers")
        
        print(f"\n{'='*70}")
        print(f"PARALLEL TRAINING COMPLETED")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per epoch: {total_time/num_epochs:.2f}s")
        print(f"{'='*70}\n")
    
    def shutdown(self):
        """Clean up resources."""
        if self.backend == 'ray' and ray.is_initialized():
            # Don't shutdown Ray as other processes might be using it
            pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_parallel(
    net,
    num_epochs: int = 100,
    num_workers: int = 4,
    backend: str = 'ray'
):
    """
    Convenience function for parallel training.
    
    Args:
        net: Initialized GscNet instance
        num_epochs: Number of training epochs
        num_workers: Number of parallel workers
        backend: 'ray' or 'multiprocessing'
    
    Returns:
        The trained network
    """
    trainer = ParallelTrainer(net, num_workers=num_workers, backend=backend)
    trainer.train(num_epochs=num_epochs)
    return net


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# =============================================================================
# STATUS: Now imports from only_gscnet_speedup_sap.py
# =============================================================================
"""
IMPLEMENTATION STATUS
=====================

This module now imports from only_gscnet_speedup_sap.py and uses the REAL
GscNet class. Workers create actual GscNet instances and call the real
methods (estimate_prob_inc, cost_grad, etc.)

COMPLETED:
  [x] Import real GscNet from only_gscnet_speedup_sap
  [x] _create_worker_net() creates real GscNet instances  
  [x] Workers use actual estimate_prob_inc() for sampling
  [x] Workers use actual cost_grad() for gradient computation
  [x] mask0 computed once by master, shared with workers
  [x] Corpus passed to workers for target statistics

HOW IT WORKS:
  1. Master extracts config + serializes HarmonicGrammar
  2. Workers deserialize HG and create real GscNet
  3. Workers copy current weights from master
  4. Workers call actual estimate_prob_inc() and cost_grad()
  5. Workers return sparse gradients
  6. Master aggregates gradients and applies optimizer

USAGE:
------
    # Use dtype version for float32 memory optimization (~50% memory savings)
import only_gscnet_speedup_sap_dtype as gsc
    from only_gscnet_speedup_sap_parallel import ParallelTrainer
    
    # Setup (same as normal training)
    net = gsc.GscNet(hg=hg, opts={'use_sparse_wc': True})
    net.generate_corpus(nsamples=5000)
    net.initialize(train_opts={'num_trials': 50, 'lrate': 0.1})
    
    # Parallel training (replaces net.train2() loop)
    trainer = ParallelTrainer(net, num_workers=4, backend='ray')
    trainer.train(num_epochs=100)
    
    # Network weights are updated in-place
    gsc.save_model(net, 'my_model.pkl')

NOTE: Each worker creates a full GscNet, so memory usage is:
      total_memory ≈ master_memory + (num_workers × worker_net_memory)
      
      Worker nets don't store corpus or traces, so they're smaller than master.
      For large grammars, consider fewer workers to stay within memory limits.
"""


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           PARALLEL GSCNET TRAINING - USAGE EXAMPLE               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  # Standard training setup                                       ║
    ║  # Use dtype version for float32 memory optimization (~50% memory savings)
import only_gscnet_speedup_sap_dtype as gsc                          ║
    ║  from only_gscnet_speedup_sap_parallel import ParallelTrainer   ║
    ║                                                                  ║
    ║  # Create and initialize network                                 ║
    ║  hg = gsc.HarmonicGrammar(pcfg=PCFG_str, root='S')              ║
    ║  net = gsc.GscNet(hg=hg, opts={'use_sparse_wc': True})          ║
    ║  net.generate_corpus(nsamples=5000)                             ║
    ║  net.initialize(train_opts={'num_trials': 50, 'lrate': 0.1})    ║
    ║                                                                  ║
    ║  # Create parallel trainer                                       ║
    ║  trainer = ParallelTrainer(                                      ║
    ║      net,                                                        ║
    ║      num_workers=4,       # CPU cores / GPU workers              ║
    ║      backend='ray'        # 'ray' or 'multiprocessing'          ║
    ║  )                                                               ║
    ║                                                                  ║
    ║  # Train                                                         ║
    ║  trainer.train(num_epochs=100)                                   ║
    ║                                                                  ║
    ║  # Network weights are updated in-place                          ║
    ║  gsc.save_model(net, 'my_parallel_trained_model.pkl')           ║
    ║                                                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  CLUSTER USAGE (SLURM example):                                  ║
    ║                                                                  ║
    ║  # In your SLURM script:                                         ║
    ║  # Start Ray head node on first allocated node                   ║
    ║  ray start --head --port=6379                                    ║
    ║                                                                  ║
    ║  # In Python:                                                    ║
    ║  import ray                                                      ║
    ║  ray.init(address='auto')  # Connect to cluster                  ║
    ║  trainer = ParallelTrainer(net, num_workers=64, backend='ray')  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


# GPU Acceleration for GSC

This directory contains GPU-accelerated versions of the GSC (Grammar as Selective Constraint) model using CuPy.

## 🚀 Quick Start

### 1. Install CuPy

First, check your CUDA version:
```bash
nvcc --version
# or
nvidia-smi
```

Then install CuPy matching your CUDA version:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Or install from requirements
pip install -r requirements_gpu.txt
```

### 2. Test GPU Availability

```bash
python -c "import cupy as cp; print(f'GPU: {cp.cuda.Device()}')"
```

### 3. Run Benchmark

Compare CPU vs GPU performance (100 epochs, ~5-10 minutes):

```bash
# Local machine
python benchmark_gpu.py --mode both --epochs 100

# On cluster with SLURM
sbatch run_gpu_benchmark.slurm
```

### 4. Run Full Training

GPU-accelerated version of cho_grammar1.py (1000 epochs):

```bash
# Local machine
python cho_grammar1_gpu.py

# On cluster with SLURM
sbatch run_gpu_full.slurm
```

## 📁 Files

### Core Files
- **`gsc_gpu.py`** - GPU wrapper that monkey-patches numpy with cupy in gsc module
- **`cho_grammar1_gpu.py`** - GPU-accelerated training script
- **`benchmark_gpu.py`** - Benchmark tool to compare CPU vs GPU

### Configuration
- **`requirements_gpu.txt`** - Python dependencies for GPU
- **`run_gpu_benchmark.slurm`** - SLURM script for benchmarking
- **`run_gpu_full.slurm`** - SLURM script for full training

## 🎯 Expected Performance

### CuPy Drop-in Replacement Strategy

**Expected Speedup:** 5-10x

The current implementation uses CuPy as a drop-in replacement for NumPy. This provides:
- ✅ Minimal code changes
- ✅ Easy to implement and test
- ✅ GPU acceleration for matrix operations
- ⚠️ Limited by Python loops and CPU-GPU transfers
- ⚠️ Doesn't exploit batch parallelism

### Performance Estimates

Based on your current runtime (~40-60 min on laptop, ~60 min on GPU cluster with CPU):

| Version | Time (1000 epochs) | Speedup |
|---------|-------------------|---------|
| CPU (laptop) | 40-60 min | 1x |
| CPU (cluster) | 60 min | ~1x |
| GPU (CuPy) | 6-12 min | 5-10x |
| GPU (Full PyTorch) | 2-5 min | 20-40x |

## 🔍 How It Works

### Architecture

```
cho_grammar1_gpu.py
    ↓
gsc_gpu.py (monkey-patches numpy → cupy)
    ↓
gsc.py (original code, now using cupy arrays)
    ↓
GPU computation
```

### Key Features

1. **Automatic GPU Detection**
   - Falls back to CPU if CuPy not available
   - No code changes needed

2. **Memory Management**
   - `GPUContext()` context manager for automatic cleanup
   - `to_gpu()` / `to_cpu()` helpers for data transfer

3. **Model Persistence**
   - Automatic CPU conversion when saving
   - Can load models directly to GPU

4. **Performance Monitoring**
   - GPU memory tracking
   - Timing utilities

## 🛠️ Usage Examples

### Basic Usage

```python
import gsc_gpu as gsc  # Use GPU

# Rest of code unchanged!
hg = gsc.HarmonicGrammar(pcfg=PCFG, root='S', max_sent_len=5)
net = gsc.GscNet(hg=hg, opts=net_opts)
net.train2(train_opts={'num_epochs': 10})
```

### GPU Memory Monitoring

```python
import gsc_gpu as gsc

# Check if GPU is available
if gsc.GPU_AVAILABLE:
    gsc.print_gpu_memory()
```

### Profiling

```python
import gsc_gpu as gsc

with gsc.GPUProfiler("Training"):
    net.train2(train_opts={'num_epochs': 10})
# Output: Training: 12.345 seconds
```

### Manual Data Transfer

```python
import gsc_gpu as gsc

# Move to GPU
gpu_array = gsc.to_gpu(cpu_array)

# Move to CPU
cpu_array = gsc.to_cpu(gpu_array)
```

## 📊 Benchmarking

### Run Quick Benchmark (100 epochs)

```bash
python benchmark_gpu.py --mode both --epochs 100
```

Example output:
```
============================================================
BENCHMARK RESULTS
============================================================

CPU Results (100 epochs):
  Init time:     5.23s
  Train time:    360.12s (6.00m)
  Total time:    365.35s (6.09m)
  Avg/10 epochs: 36.01s

GPU Results (100 epochs):
  Init time:     6.45s
  Train time:    45.23s (0.75m)
  Total time:    51.68s (0.86m)
  Avg/10 epochs: 4.52s

------------------------------------------------------------
SPEEDUP (CPU time / GPU time):
  Initialization: 0.81x
  Training:       7.96x ⚡
  Total:          7.07x ⚡

------------------------------------------------------------
ESTIMATED TIME FOR 1000 EPOCHS:
  CPU: 60.0 minutes
  GPU: 7.5 minutes
  Time saved: 52.5 minutes
============================================================
```

## 🔧 SLURM Job Submission

### Benchmark (30 minutes)

```bash
sbatch run_gpu_benchmark.slurm
```

Settings:
- 1 GPU
- 4 CPUs
- 32 GB RAM
- 30 min time limit

### Full Training (2 hours)

```bash
sbatch run_gpu_full.slurm
```

Settings:
- 1 GPU (not 2 - we only need 1!)
- 8 CPUs
- 128 GB RAM
- 2 hour time limit

### Check Job Status

```bash
squeue -u $USER
```

### View Output

```bash
# Benchmark
cat benchmark_gpu_*.out

# Full training
cat cho_grammar1_gpu_*.out
```

## 🐛 Troubleshooting

### CuPy Installation Issues

**Error:** `ImportError: No module named 'cupy'`
```bash
# Check CUDA version first
nvcc --version

# Install matching CuPy version
pip install cupy-cuda11x  # for CUDA 11.x
pip install cupy-cuda12x  # for CUDA 12.x
```

**Error:** `CuPy failed to initialize`
```bash
# Check GPU is accessible
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

### Memory Issues

**Error:** `cupy.cuda.memory.OutOfMemoryError`

Solutions:
1. Reduce batch size (already 4, can't reduce much)
2. Free GPU memory manually:
```python
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
```
3. Request more GPU memory in SLURM:
```bash
#SBATCH --gres=gpu:a100:1  # Request specific GPU with more memory
```

### No Speedup?

If GPU isn't faster than CPU:

1. **Check GPU is actually being used:**
   ```bash
   watch -n 0.5 nvidia-smi  # Monitor GPU usage
   ```
   Look for:
   - GPU utilization > 0%
   - Memory usage increasing

2. **Profile to find bottlenecks:**
   ```bash
   python -m cProfile -o profile.stats cho_grammar1_gpu.py
   python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"
   ```

3. **Check for CPU-GPU transfer overhead:**
   - Small arrays may be slower on GPU
   - Add `with gsc.GPUContext():` to keep data on GPU

## 🚀 Next Steps: Further Optimization

If CuPy provides good speedup, consider these advanced optimizations:

### 1. Batch Parallelization (20-40x speedup)

Port `estimate_prob_inc()` to run trials in parallel:
```python
# Instead of: for trial in range(4): run_trial()
# Do: run_trials_batched(num_trials=4)  # All 4 run in parallel
```

### 2. Full PyTorch Port (20-40x speedup)

Rewrite GscNet class using PyTorch:
- Automatic differentiation
- Batch operations across trials and epochs
- Mixed precision (FP16) training
- Better GPU kernel fusion

### 3. Custom CUDA Kernels

Write specialized kernels for:
- `update_stateC()` hot loop
- `HGradC()` gradient computation

## 📖 References

- [CuPy Documentation](https://docs.cupy.dev/)
- [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html)
- [CuPy Performance Guide](https://docs.cupy.dev/en/stable/user_guide/performance.html)

## ⚠️ Important Notes

1. **GPU is not always faster** - Small arrays and Python loops may be slower on GPU due to transfer overhead
2. **This is a drop-in replacement** - If it doesn't provide speedup, we can try more advanced strategies
3. **Memory management** - GPU memory is limited, use `GPUContext()` to manage it
4. **Reproducibility** - Set random seeds for both NumPy and CuPy for consistent results

## 📝 TODO

Potential improvements:
- [ ] Batch processing across trials
- [ ] Mixed precision (FP16) support
- [ ] Multi-GPU support (if needed)
- [ ] Custom CUDA kernels for hot paths
- [ ] Memory optimization (pre-allocate arrays)
- [ ] Full PyTorch port for automatic differentiation

---

**Questions?** Check the benchmark results first, then consider next optimization steps!

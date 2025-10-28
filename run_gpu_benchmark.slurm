#!/bin/bash
#SBATCH -J gsc_gpu_benchmark       # Job name
#SBATCH -N 1                       # Total number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # CPUs per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --time=00:30:00            # Time limit (30 minutes)
#SBATCH --get-user-env             # Get user environment
#SBATCH -o benchmark_gpu_%j.out    # Standard output log
#SBATCH -e benchmark_gpu_%j.err    # Standard error log

echo "======================================================================"
echo "GPU Benchmark Job Started"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Print GPU info
echo "----------------------------------------------------------------------"
echo "GPU Information:"
echo "----------------------------------------------------------------------"
nvidia-smi
echo ""

# Print CUDA version
echo "----------------------------------------------------------------------"
echo "CUDA Version:"
echo "----------------------------------------------------------------------"
nvcc --version || echo "nvcc not found"
echo ""

# Load modules if needed (uncomment and modify as needed for your cluster)
# module load cuda/11.8
# module load python/3.9

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0

# Print Python and package versions
echo "----------------------------------------------------------------------"
echo "Python Environment:"
echo "----------------------------------------------------------------------"
python --version
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cupy; print(f'CuPy: {cupy.__version__}')" || echo "CuPy not installed!"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
echo ""

# Run benchmark
echo "======================================================================"
echo "Running Benchmark (100 epochs)"
echo "======================================================================"
python benchmark_gpu.py --mode both --epochs 100

echo ""
echo "======================================================================"
echo "Job Completed"
echo "End Time: $(date)"
echo "======================================================================"

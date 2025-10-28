#!/bin/bash
#SBATCH -J gsc_gpu_full            # Job name
#SBATCH -N 1                       # Total number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # CPUs per task
#SBATCH --mem=128G                 # Memory per node
#SBATCH --gres=gpu:1               # Request 1 GPU (change to 2 if you want both)
#SBATCH --time=02:00:00            # Time limit (2 hours)
#SBATCH --get-user-env             # Get user environment
#SBATCH -o cho_grammar1_gpu_%j.out # Standard output log
#SBATCH -e cho_grammar1_gpu_%j.err # Standard error log

echo "======================================================================"
echo "GSC GPU Training Job Started"
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
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only

# Optional: Set CuPy memory pool settings for better performance
export CUPY_CACHE_DIR=/tmp/cupy_cache_${SLURM_JOB_ID}
mkdir -p $CUPY_CACHE_DIR

# Print Python and package versions
echo "----------------------------------------------------------------------"
echo "Python Environment:"
echo "----------------------------------------------------------------------"
python --version
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cupy; print(f'CuPy: {cupy.__version__}')" || echo "CuPy not installed! Install with: pip install cupy-cuda11x"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
echo ""

# Check if CuPy is installed
if ! python -c "import cupy" 2>/dev/null; then
    echo "ERROR: CuPy is not installed!"
    echo "Please install CuPy with: pip install cupy-cuda11x"
    echo "Or see: https://docs.cupy.dev/en/stable/install.html"
    exit 1
fi

# Run full training
echo "======================================================================"
echo "Running Full Training (1000 epochs)"
echo "======================================================================"
python cho_grammar1_gpu.py

# Save GPU utilization log
echo ""
echo "----------------------------------------------------------------------"
echo "Final GPU Status:"
echo "----------------------------------------------------------------------"
nvidia-smi

# Cleanup
rm -rf $CUPY_CACHE_DIR

echo ""
echo "======================================================================"
echo "Job Completed"
echo "End Time: $(date)"
echo "======================================================================"

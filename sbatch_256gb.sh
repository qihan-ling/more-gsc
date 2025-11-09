#!/bin/bash
#SBATCH --job-name=gsc_1k_rules
#SBATCH --output=gsc_training_%j.log
#SBATCH --error=gsc_training_%j.err
#SBATCH --time=7-00:00:00           # 7 days for full training
#SBATCH --mem=256G                  # 256GB RAM (requires max_sent_len=18)
#SBATCH --cpus-per-task=16          # 16 CPU cores
#SBATCH --gres=gpu:1                # 1 GPU (avoid multi-GPU hang)
#SBATCH --partition=gpu             # Adjust to your cluster's GPU partition name

# Load modules (adjust to your cluster)
module load python/3.9
module load cuda/11.8

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Environment variables
export CUDA_VISIBLE_DEVICES=0       # Force single GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Prevent JAX from grabbing all GPU memory

# Print system info
echo "=================================================="
echo "Job started at: $(date)"
echo "Hostname: $(hostname)"
echo "RAM allocated: 256GB"
echo "⚠️  WARNING: Must use max_sent_len=18 (not 20) to fit in 256GB"
echo "=================================================="
nvidia-smi
free -h
echo "=================================================="

# Run training with max_sent_len=18 (IMPORTANT!)
python cho_grammar1_256gb.py

echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="

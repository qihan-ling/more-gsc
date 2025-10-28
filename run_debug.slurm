#!/bin/bash
#SBATCH -J gsc_debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o debug_%j.out
#SBATCH -e debug_%j.err

echo "======================================================================"
echo "DEBUG RUN"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Disable output buffering
export PYTHONUNBUFFERED=1

# Run debug version (only 10 epochs)
python -u cho_grammar1_debug.py

echo ""
echo "======================================================================"
echo "Job Completed"
echo "End Time: $(date)"
echo "======================================================================"

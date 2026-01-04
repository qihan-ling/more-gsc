#!/usr/bin/env python3
"""
Diagnostic script to estimate training time and test computational speed
"""
import numpy as np
import time
from scipy import sparse

print("="*70)
print("SAP Grammar Training Speed Diagnostic")
print("="*70)

# Load the sparse matrix info from your configuration
matrix_size = 844800
density = 0.0004  # 0.04% from your output
nnz = int(matrix_size * matrix_size * density)

print(f"\nMatrix configuration:")
print(f"  Size: {matrix_size} x {matrix_size}")
print(f"  Density: {density*100:.4f}%")
print(f"  Non-zeros: {nnz:,}")

# Create a test sparse matrix
print(f"\nCreating test sparse matrix...")
data = np.random.randn(nnz)
row = np.random.randint(0, matrix_size, nnz)
col = np.random.randint(0, matrix_size, nnz)
WC_test = sparse.csr_matrix((data, (row, col)), shape=(matrix_size, matrix_size))

# Test sparse matrix-vector multiplication speed
print(f"Testing sparse matrix-vector multiplication speed...")
x = np.random.randn(matrix_size)

n_tests = 10
t0 = time.time()
for _ in range(n_tests):
    y = WC_test.dot(x)
elapsed = time.time() - t0
time_per_matvec = elapsed / n_tests

print(f"  Time per matvec: {time_per_matvec:.4f} seconds")

# Calculate expected training time
print(f"\n" + "="*70)
print("Training Time Estimates")
print("="*70)

num_trials = 200
q_max = 15.0
q_rate = 1.0
dt = 0.005
num_epochs = 500

steps_per_trial = (q_max / q_rate) / dt
steps_per_epoch = num_trials * steps_per_trial
total_steps = num_epochs * steps_per_epoch

time_per_epoch = steps_per_epoch * time_per_matvec
time_total = total_steps * time_per_matvec

print(f"\nCurrent configuration:")
print(f"  num_trials: {num_trials}")
print(f"  Integration steps per trial: {steps_per_trial:.0f}")
print(f"  Steps per epoch: {steps_per_epoch:.0f}")
print(f"  Total steps (500 epochs): {total_steps:.0f}")
print(f"")
print(f"Expected time:")
print(f"  Per epoch: {time_per_epoch/3600:.1f} hours ({time_per_epoch/3600/24:.1f} days)")
print(f"  Total (500 epochs): {time_total/3600:.1f} hours ({time_total/3600/24:.1f} days)")

# Recommend faster configurations
print(f"\n" + "="*70)
print("RECOMMENDED OPTIMIZATIONS")
print("="*70)

configs = [
    ("Original", 200, 0.005, 500),
    ("Reduce trials", 50, 0.005, 500),
    ("Increase dt", 200, 0.02, 500),
    ("Both optimizations", 50, 0.02, 500),
    ("Aggressive", 25, 0.05, 500),
]

print(f"\n{'Configuration':<25} {'Trials':>8} {'dt':>8} {'Epochs':>8} {'Time (days)':>12}")
print("-"*70)

for name, trials, dt_val, epochs in configs:
    steps = epochs * trials * (q_max / q_rate) / dt_val
    days = (steps * time_per_matvec) / 3600 / 24
    print(f"{name:<25} {trials:>8} {dt_val:>8.3f} {epochs:>8} {days:>12.1f}")

print("\n" + "="*70)
print("RECOMMENDATION: Use 'Both optimizations' (50 trials, dt=0.02)")
print("This should complete in a reasonable time while maintaining accuracy")
print("="*70)

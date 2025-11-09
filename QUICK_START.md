# Quick Start Guide - Choose Your Path

## PATH 1: 512GB RAM + max_sent_len=20 (Full Goal)

### Step 1: Modify your SBATCH script

Update your existing sbatch script or use the provided one:

```bash
#SBATCH --mem=512G
```

### Step 2: Ensure optimizations are applied

Add to the **top** of your training script (before `import gsc`):

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU

import gsc
import optimized_tokenize_cnf

# Apply optimization
optimized_tokenize_cnf.apply_optimization()
```

### Step 3: Use integration method for equilibrium point

When creating GscNet:

```python
net_opts = {
    'ep_method': 'integration',
    'integration_dur': 50,
    'dt': 0.01
}

net = gsc.GscNet(hg=hg, bowl_r=10.0, **net_opts)
```

### Step 4: Submit job

```bash
sbatch sbatch_512gb.sh
```

**Expected timeline:**
- Grammar initialization: 30-60 seconds
- GscNet initialization: 60 seconds
- Training: ~37.5 hours (50 epochs)
- Total: ~38 hours

---

## PATH 2: 256GB RAM + max_sent_len=18 (Compromise)

### Step 1: Modify your SBATCH script

```bash
#SBATCH --mem=256G
```

### Step 2: **CRITICAL** - Change max_sent_len to 18

In your training script:

```python
# OLD
hg = gsc.HarmonicGrammar(
    pcfg=pcfg,
    max_sent_len=20,  # ← CHANGE THIS
    ...
)

# NEW
hg = gsc.HarmonicGrammar(
    pcfg=pcfg,
    max_sent_len=18,  # ← Changed to 18
    ...
)

# Also update corpus generation
net.generate_corpus(nsamples=25000, max_sent_len=18)  # ← Changed to 18
```

### Step 3: Apply same optimizations as PATH 1

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import optimized_tokenize_cnf
optimized_tokenize_cnf.apply_optimization()
```

### Step 4: Submit job

```bash
sbatch sbatch_256gb.sh
```

**Expected timeline:**
- Grammar initialization: 30-60 seconds
- GscNet initialization: 60 seconds
- Training: ~29 hours (50 epochs)
- Total: ~30 hours

---

## Checklist Before Running

### ✅ Pre-flight Checks

```bash
# 1. Check available GPUs
python -c "import jax; print('GPUs:', jax.devices())"
# Should show: [CudaDevice(id=0)]

# 2. Check JAX is installed
python -c "import jax; print('JAX version:', jax.__version__)"

# 3. Check available RAM
free -h
# Should show your allocated memory

# 4. Verify optimization files exist
ls -lh optimized_tokenize_cnf.py monitor_memory.py
```

### ✅ Required Files in Your Directory

- ✅ `optimized_tokenize_cnf.py` - PCFG tokenization optimization
- ✅ `monitor_memory.py` - Memory monitoring utility
- ✅ `sbatch_512gb.sh` OR `sbatch_256gb.sh` - SBATCH script
- ✅ Your grammar file
- ✅ Your training script (cho_grammar1.py or similar)

---

## Monitoring Your Job

### Check job status
```bash
squeue -u $USER
```

### Monitor output in real-time
```bash
tail -f gsc_training_<JOBID>.log
```

### Check memory usage (while job is running)
```bash
srun --jobid=<JOBID> --pty free -h
```

### Check GPU usage
```bash
srun --jobid=<JOBID> --pty nvidia-smi
```

---

## Expected Output Timeline

### First 5 minutes:
```
Starting memory monitoring...
MEMORY CHECKPOINT: Start
Process memory:   0.50 GB
...
Applying PCFG tokenization optimization...
Loading grammar from file...
MEMORY CHECKPOINT: After loading grammar
Process memory:   2.30 GB
...
Initializing HarmonicGrammar...
Tokenizing CNF grammar (optimized)...
Processing 1756 rules...
Progress: 500/1756 rules (28.5%) - 15s elapsed
Progress: 1000/1756 rules (56.9%) - 30s elapsed
Progress: 1500/1756 rules (85.4%) - 45s elapsed
Tokenization complete: 1756 rules in 52.3s
Fillers: ['filler_0', 'filler_1', ...]  ← This prints!
MEMORY CHECKPOINT: After HarmonicGrammar creation
Process memory:  45.20 GB
...
```

### Minutes 5-7:
```
Initializing GscNet...
Using integration method for equilibrium point...
Finding equilibrium point...
Integration complete in 58.2s
MEMORY CHECKPOINT: After GscNet creation
Process memory:  125.30 GB  ← Peak initialization memory
...
```

### Minutes 7-10:
```
Generating training corpus...
Generated 25000 sentences
MEMORY CHECKPOINT: After corpus generation
Process memory:  142.15 GB
...
```

### Hours 0-38 (or 0-30):
```
Starting training...
Epoch 1/50:
  Running 500 trials on 1 device...
  Epoch 1 complete: loss=0.234, acc=0.876 (45.2 min)
  Checkpoint saved: checkpoints/epoch_001.pkl
Epoch 2/50:
  Running 500 trials on 1 device...
  ...
```

---

## What If Something Goes Wrong?

### Issue: "Out of Memory" error

**If using 512GB + max_len=20:**
- This shouldn't happen. Check: `free -h` - did you actually get 512GB?
- Emergency fix: Reduce max_sent_len to 18

**If using 256GB + max_len=18:**
- Reduce to max_sent_len=17
- Or reduce num_trials from 500 to 300

### Issue: Still hanging at "Multi-GPU enabled: 3 devices"

**Check:**
```bash
echo $CUDA_VISIBLE_DEVICES
# Should show: 0
```

**Fix:** Add to top of script:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Issue: Grammar initialization still taking > 5 minutes

**Check:** Did you apply the optimization?

```python
# This MUST be before creating HarmonicGrammar
import optimized_tokenize_cnf
optimized_tokenize_cnf.apply_optimization()
```

### Issue: Job killed/cancelled unexpectedly

**Common reasons:**
1. Time limit exceeded - increase `#SBATCH --time`
2. Memory exceeded - reduce max_sent_len or request more RAM
3. Node failure - resubmit job

**Check logs:**
```bash
cat gsc_training_<JOBID>.err
```

---

## Post-Training: What to Check

### 1. Training completed successfully?
```bash
grep "Training complete" gsc_training_<JOBID>.log
```

### 2. Memory summary
```bash
grep "MEMORY USAGE SUMMARY" -A 20 gsc_training_<JOBID>.log
```

### 3. Final model saved?
```bash
ls -lh checkpoints/final_model.pkl
ls -lh checkpoints/epoch_*.pkl
```

### 4. Load and test model
```python
import gsc

# Load checkpoint
net = gsc.GscNet.load_checkpoint('checkpoints/final_model.pkl')

# Test parsing
sentence = "the cat sat on the mat"
parse_tree = net.parse(sentence)
print(parse_tree)
```

---

## My Recommendation (Summary)

Based on your stated goals:

### 🎯 **GO WITH: 512GB + max_sent_len=20**

**Why:**
1. Your goal: "full 1k rules or even more in the future"
2. You specifically asked about max_sent_len=20
3. 512GB gives headroom for scaling to 2k+ rules later
4. The extra 8-9 hours of training is worth full linguistic coverage
5. Avoids having to re-run experiments if max_len=18 proves insufficient

**How:**
```bash
sbatch sbatch_512gb.sh
```

---

## Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `sbatch_512gb.sh` | Submit 512GB job | PATH 1 |
| `sbatch_256gb.sh` | Submit 256GB job | PATH 2 |
| `optimized_tokenize_cnf.py` | PCFG optimization | Always (both paths) |
| `monitor_memory.py` | Track memory usage | Optional (debugging) |
| `cho_grammar1_with_monitoring.py` | Example script | Reference/testing |
| `RAM_ALLOCATION_GUIDE.md` | Detailed comparison | Decision-making |
| `QUICK_START.md` | This file | Getting started |

---

## You're Ready!

Everything is set up. Just decide:

- **512GB + max_len=20** → Run `sbatch sbatch_512gb.sh`
- **256GB + max_len=18** → Edit script to use max_len=18, run `sbatch sbatch_256gb.sh`

Good luck with your training! 🚀

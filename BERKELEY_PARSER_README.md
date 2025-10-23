# Berkeley Parser Training & Grammar Extraction for GSC

This directory contains scripts to train a Berkeley (Petrov) parser on Penn Treebank data and extract the learned grammar for use with GSC parsers.

## Overview

**Pipeline:**
```
Penn Treebank 3 (.mrg files)
    ↓
[1] preprocess_ptb_for_berkeley.py
    ↓
Preprocessed trees (train/dev/test.trees)
    ↓
[2] train_berkeley_parser.py
    ↓
Trained grammar (berkeley_sm5.gr)
    ↓
[3] extract_berkeley_grammar.py
    ↓
GSC-compatible grammar + POS→words mapping
    ↓
Use with GSC parser + embeddings
```

---

## Prerequisites

### 1. Penn Treebank 3

You need access to Penn Treebank 3 (LDC99T42):
- Available from: https://catalog.ldc.upenn.edu/LDC99T42
- Requires LDC license
- Directory structure: `treebank_3/parsed/mrg/wsj/`

### 2. Berkeley Parser

Download from: https://github.com/slavpetrov/berkeleyparser/releases

```bash
# Download latest release
wget https://github.com/slavpetrov/berkeleyparser/releases/download/v1.7/BerkeleyParser-1.7.jar

# Or place in ./berkeley/ directory
mkdir berkeley
mv BerkeleyParser-1.7.jar berkeley/BerkeleyParser.jar
```

### 3. Java

Requires Java 8 or higher:
```bash
java -version
# Should show version 1.8 or higher
```

### 4. Python

Requires Python 3.7+:
```bash
python3 --version
```

---

## Quick Start

### End-to-End Example

```bash
# Step 1: Preprocess Penn Treebank
python preprocess_ptb_for_berkeley.py \
    --ptb-root /path/to/treebank_3/parsed/mrg/wsj \
    --output-dir ./berkeley_data \
    --statistics

# Step 2: Train Berkeley parser with SM-5
python train_berkeley_parser.py \
    --train-file berkeley_data/train.trees \
    --dev-file berkeley_data/dev.trees \
    --output-grammar berkeley_sm5.gr \
    --num-splits 5 \
    --memory 8g

# Step 3: Extract grammar rules
python extract_berkeley_grammar.py \
    --grammar berkeley_sm5.gr \
    --output-gsc berkeley_rules.txt \
    --output-pos-words pos_to_words.json \
    --collapse-splits
```

**Time estimate:**
- Step 1: ~5 minutes
- Step 2: ~4-6 hours (SM-5 on full PTB)
- Step 3: ~2 minutes

---

## Step 1: Preprocess PTB Data

### Purpose

Convert Penn Treebank `.mrg` files to format expected by Berkeley parser:
- One tree per line
- Standard train/dev/test splits
- Clean formatting

### Usage

```bash
python preprocess_ptb_for_berkeley.py \
    --ptb-root /path/to/treebank_3/parsed/mrg/wsj \
    --output-dir ./berkeley_data \
    --statistics
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ptb-root` | Required | Path to PTB `parsed/mrg/wsj` directory |
| `--output-dir` | `./berkeley_data` | Output directory |
| `--train-sections` | `2-21` | Training sections |
| `--dev-sections` | `22` | Development sections |
| `--test-sections` | `23` | Test sections |
| `--statistics` | Off | Print dataset statistics |

### Output

Creates three files:
- `train.trees` - ~39,832 trees (sections 02-21)
- `dev.trees` - ~1,700 trees (section 22)
- `test.trees` - ~2,416 trees (section 23)

### Penn Treebank Structure

Expected directory structure:
```
treebank_3/
  parsed/
    mrg/
      wsj/
        00/
          wsj_0001.mrg
          wsj_0002.mrg
          ...
        01/
          wsj_0100.mrg
          ...
        02/  # Training starts here
        ...
        21/  # Training ends here
        22/  # Dev set
        23/  # Test set
        24/
```

### Example Output

```
[TRAIN] Processing sections [2, 3, ..., 21]
  Processed 1,989 files
  Extracted 39,832 trees
  Wrote to: berkeley_data/train.trees

[DEV] Processing sections [22]
  Processed 85 files
  Extracted 1,700 trees
  Wrote to: berkeley_data/dev.trees

[TEST] Processing sections [23]
  Processed 107 files
  Extracted 2,416 trees
  Wrote to: berkeley_data/test.trees
```

---

## Step 2: Train Berkeley Parser

### Purpose

Train a latent variable PCFG using the split-merge algorithm:
- **SM-1**: Quick baseline (~30 min)
- **SM-2**: Faster training (~1 hour)
- **SM-5**: Best accuracy (~4-6 hours) **← Recommended**
- **SM-6**: Marginal improvement (~8-10 hours)

### Usage

```bash
python train_berkeley_parser.py \
    --train-file berkeley_data/train.trees \
    --dev-file berkeley_data/dev.trees \
    --output-grammar berkeley_sm5.gr \
    --num-splits 5 \
    --memory 8g
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--train-file` | Required | Training trees file |
| `--dev-file` | None | Development trees (optional) |
| `--output-grammar` | `berkeley_sm5.gr` | Output grammar file |
| `--num-splits` | 5 | Split-merge cycles (1-6) |
| `--num-iterations` | 50 | EM iterations per cycle |
| `--memory` | `8g` | JVM memory allocation |
| `--jar-path` | Auto-detect | Path to BerkeleyParser.jar |
| `--max-sentence-length` | 40 | Max sentence length to train on |
| `--rare-word-threshold` | 20 | Rare word threshold |
| `--num-threads` | Auto | Number of threads |
| `--test` | Off | Run quick test after training |

### Memory Requirements

| Dataset | Recommended Memory |
|---------|-------------------|
| Small subset | 4GB |
| Full PTB training | 8GB |
| Large corpora | 16GB |

### Training Time (Approximate)

On modern CPU (e.g., Intel i7, 8 cores):

| SM Cycles | Time | F1 Score |
|-----------|------|----------|
| SM-1 | ~30 min | ~85% |
| SM-2 | ~1 hour | ~88% |
| SM-3 | ~2 hours | ~89% |
| SM-5 | ~4-6 hours | **~90%** |
| SM-6 | ~8-10 hours | ~90.5% |

**Recommendation:** Use SM-5 for best balance of accuracy and training time.

### What Happens During Training

```
Split-Merge Iteration 1:
  ├─ Split all categories (NP → NP-0, NP-1)
  ├─ Run EM to learn new probabilities
  ├─ Merge least useful splits
  └─ Result: Some categories split, others not

Split-Merge Iteration 2:
  ├─ Further split categories
  ├─ EM re-estimation
  ├─ Merge again
  └─ Result: More refined grammar

...

Split-Merge Iteration 5:
  └─ Final grammar with 5 levels of splits
```

### Output

Creates:
- `berkeley_sm5.gr` - Trained grammar (~50-100 MB)
- Console output showing:
  - EM iteration progress
  - Log-likelihood improvements
  - Dev set F1 scores (if dev file provided)

### Example Training Output

```
=====================================
Berkeley Parser Training with Split-Merge
=====================================

Configuration:
  Train file: berkeley_data/train.trees
  Dev file: berkeley_data/dev.trees
  Output grammar: berkeley_sm5.gr
  Split-merge cycles: 5
  EM iterations per cycle: 50
  Memory: 8g

Training started...
=====================================

SM Iteration 1:
  EM Iteration 10: log-likelihood = -245678.32
  EM Iteration 20: log-likelihood = -243521.45
  ...
  Merging 23% of splits
  Dev F1: 87.3%

SM Iteration 2:
  ...
```

---

## Step 3: Extract Grammar Rules

### Purpose

Extract PCFG rules from trained Berkeley grammar and format for GSC:
- Binary rules (A → B C)
- Unary rules (A → B)
- Lexical rules (POS → word)
- Option to collapse split categories

### Usage

```bash
python extract_berkeley_grammar.py \
    --grammar berkeley_sm5.gr \
    --output-gsc berkeley_rules.txt \
    --output-pos-words pos_to_words.json \
    --collapse-splits
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--grammar` | Required | Berkeley .gr file |
| `--output-gsc` | `berkeley_grammar.txt` | GSC format output |
| `--output-pos-words` | None | POS→words JSON for embeddings |
| `--collapse-splits` | Off | Merge NP-0, NP-1, ... → NP |
| `--min-prob` | 0.0001 | Min probability to include |
| `--top-n-words` | 50 | Top N words per POS |
| `--include-lexical` | Off | Include lexical rules in GSC output |
| `--example` | Off | Run with example grammar (demo) |

### Split Categories: Keep or Collapse?

Berkeley learns latent subcategories:

```
Original PTB:        Berkeley SM-5:
  NP                   NP-0, NP-1, NP-2, ..., NP-7
  VP                   VP-0, VP-1, VP-2, ..., VP-7
  S                    S-0, S-1, S-2, ..., S-7
```

**Option A: Keep Splits (More Detailed)**

```python
# Extracted rules with splits
0.421 NP-0 -> DT-3 NN-5
0.312 NP-1 -> DT-1 NN-2
0.289 NP-2 -> NNP-0

# GSC network has more categories:
# Fillers: NP-0, NP-1, NP-2, ..., DT-0, DT-1, ...
# More bindings but potentially more accurate
```

**Option B: Collapse Splits (Simpler)** ← Recommended for GSC

```python
# Collapsed rules (probabilities summed)
0.756 NP -> DT NN    # Sum of all NP-i -> DT-j NN-k
0.289 NP -> NNP      # Sum of all NP-i -> NNP-j

# GSC network has standard categories:
# Fillers: NP, VP, DT, NN, ...
# Smaller, more interpretable
```

**Recommendation:** Use `--collapse-splits` for GSC unless you specifically want to study latent subcategories.

### Output Format (GSC)

```
# Berkeley Parser Grammar for GSC
# Extracted from: berkeley_sm5.gr

# Binary rules (A -> B C)
0.950000 S -> NP VP
0.350000 NP -> DT NN
0.200000 NP -> DT JJ
0.400000 VP -> VBD NP
0.300000 VP -> VBZ NP
0.980000 PP -> IN NP

# Unary rules (A -> B)
0.150000 NP -> NNP
0.150000 VP -> VBD
0.020000 S -> VP
```

### POS → Words Mapping

If `--output-pos-words` specified, creates JSON:

```json
{
  "DT": ["the", "a", "an", "this", "that", ...],
  "NN": ["time", "year", "people", "way", "day", ...],
  "VBD": ["said", "was", "had", "made", "did", ...],
  "NNP": ["Mr.", "Bush", "New", "York", "October", ...],
  ...
}
```

**Use for embeddings:**
```python
# Load POS → words mapping
import json
with open('pos_to_words.json', 'r') as f:
    pos_to_words = json.load(f)

# Get Llama embeddings
from transformers import AutoModel
llama = AutoModel.from_pretrained("nvidia/llama-embed-nemotron-8b")

# Create POS embeddings by averaging
pos_embeddings = {}
for pos, words in pos_to_words.items():
    embeddings = [llama.encode(word) for word in words]
    pos_embeddings[pos] = np.mean(embeddings, axis=0)
```

### Example Mode (No Grammar Needed)

Test the extraction workflow without a trained grammar:

```bash
python extract_berkeley_grammar.py \
    --example \
    --output-gsc example_grammar.txt \
    --output-pos-words example_pos_words.json
```

---

## Complete Workflow Example

### Scenario: Train GSC with Berkeley-extracted grammar

```bash
# 1. Preprocess PTB
python preprocess_ptb_for_berkeley.py \
    --ptb-root /data/ptb3/parsed/mrg/wsj \
    --output-dir ./berkeley_data

# 2. Train Berkeley parser (SM-5)
python train_berkeley_parser.py \
    --train-file berkeley_data/train.trees \
    --dev-file berkeley_data/dev.trees \
    --output-grammar berkeley_sm5.gr \
    --num-splits 5 \
    --memory 8g

# 3. Extract grammar + POS→words
python extract_berkeley_grammar.py \
    --grammar berkeley_sm5.gr \
    --output-gsc berkeley_pcfg.txt \
    --output-pos-words pos_to_words.json \
    --collapse-splits

# 4. Create embeddings from POS→words (separate script)
python create_pos_embeddings.py \
    --pos-words pos_to_words.json \
    --embedding-model llama-embed-nemotron-8b \
    --output pos_embeddings.pkl

# 5. Train GSC with Berkeley grammar + embeddings
python train_gsc_from_berkeley.py \
    --pcfg berkeley_pcfg.txt \
    --embeddings pos_embeddings.pkl \
    --output gsc_berkeley_model.pkl
```

---

## Troubleshooting

### Issue: "Berkeley Parser jar not found"

**Solution:**
```bash
# Download Berkeley parser
wget https://github.com/slavpetrov/berkeleyparser/releases/download/v1.7/BerkeleyParser-1.7.jar

# Specify path explicitly
python train_berkeley_parser.py \
    --jar-path ./BerkeleyParser-1.7.jar \
    ...
```

### Issue: "Out of memory during training"

**Solution:**
```bash
# Increase JVM memory
python train_berkeley_parser.py \
    --memory 16g \
    ...

# Or reduce training data
python train_berkeley_parser.py \
    --max-sentence-length 30 \  # Skip very long sentences
    ...
```

### Issue: "Training is very slow"

**Solutions:**
```bash
# 1. Use fewer SM iterations
--num-splits 2  # Instead of 5

# 2. Reduce EM iterations
--num-iterations 30  # Instead of 50

# 3. Use multiple threads
--num-threads 8

# 4. Train on subset first (testing)
python preprocess_ptb_for_berkeley.py \
    --train-sections 2 3 4 5  # Only 4 sections instead of 20
```

### Issue: "PTB directory not found"

**Solution:**
```bash
# Verify PTB structure
ls /path/to/treebank_3/parsed/mrg/wsj/
# Should show: 00/ 01/ 02/ ... 24/

# If structure is different, adjust --ptb-root
```

---

## File Summary

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `preprocess_ptb_for_berkeley.py` | Prepare PTB data | PTB .mrg files | train/dev/test.trees |
| `train_berkeley_parser.py` | Train Berkeley parser | .trees files | .gr grammar file |
| `extract_berkeley_grammar.py` | Extract PCFG rules | .gr file | GSC format + POS→words |

---

## Next Steps

After extracting Berkeley grammar:

1. **Create embeddings** from POS→words mapping
2. **Train GSC parser** with Berkeley grammar
3. **Compare** GSC vs. Berkeley performance
4. **Analyze** what GSC learns from Berkeley grammar

---

## References

### Berkeley Parser
- Petrov, S., Barrett, L., Thibaux, R., & Klein, D. (2006). Learning accurate, compact, and interpretable tree annotation. *ACL 2006*.
- GitHub: https://github.com/slavpetrov/berkeleyparser

### Penn Treebank
- Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank. *Computational Linguistics*, 19(2), 313-330.
- LDC: https://catalog.ldc.upenn.edu/LDC99T42

### Split-Merge Algorithm
- Best results typically with SM-5 or SM-6
- F1 score on PTB section 23: ~90%
- Learns latent syntactic categories automatically

---

## License

Scripts provided for research purposes. Penn Treebank and Berkeley Parser have their own licenses.

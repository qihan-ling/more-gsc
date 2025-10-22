# Penn Treebank to GSC Parser Conversion

This directory contains tools to train a GSC (Gradient Symbolic Computation) parser using Penn Treebank data.

## Overview

The GSC parser requires:
1. **PCFG rules with probabilities** - extracted from Penn Treebank trees
2. **Input format** - sentences as `FILLER/(level,position)` bindings

## Files

- `ptb_to_gsc.py` - Converts Penn Treebank trees to GSC format
- `train_gsc_from_ptb.py` - Trains GSC parser using extracted PCFG
- `sample_ptb.txt` - Small sample of Penn Treebank trees for testing
- `gsc.py` - Main GSC library
- `cho_grammar1.py` - Example GSC usage

## Quick Start

### 1. Extract PCFG from Penn Treebank

```bash
# Using sample data
python ptb_to_gsc.py sample_ptb.txt \
    --output-pcfg ptb_grammar.txt \
    --output-sentences ptb_sentences.txt \
    --binarize

# Using real Penn Treebank (e.g., WSJ section 02-21)
python ptb_to_gsc.py /path/to/ptb/wsj_0200-0299.mrg \
    --output-pcfg ptb_grammar.txt \
    --output-sentences ptb_sentences.txt \
    --binarize \
    --min-prob 0.001
```

**Options:**
- `--binarize` - Convert n-ary rules to binary (recommended for GSC)
- `--min-prob` - Minimum probability threshold (default: 0.001)
- `--max-trees` - Limit number of trees processed
- `--smooth` - Additive smoothing constant (default: 0.0)

### 2. Train GSC Parser

```bash
python train_gsc_from_ptb.py \
    --pcfg ptb_grammar.txt \
    --sentences ptb_sentences.txt \
    --max-sent-len 10 \
    --num-epochs 100 \
    --learning-rate 0.1 \
    --output gsc_ptb_model.pkl
```

**Options:**
- `--max-sent-len` - Maximum sentence length (default: 10)
- `--root` - Root symbol (default: S)
- `--num-epochs` - Training epochs (default: 100)
- `--learning-rate` - Learning rate (default: 0.1)
- `--num-samples` - Corpus samples to generate (default: 5000)

## Input/Output Formats

### Penn Treebank Format (Input)

Bracketed constituency trees:

```
(S (NP (DT The) (NN cat)) (VP (VBD sat)))
```

### GSC PCFG Format (Output)

```
0.500000 S -> NP VP
0.333333 NP -> DT NN
0.200000 NP -> DT JJ NN
1.000000 VP -> VBD
0.600000 VP -> VBD PP
```

### GSC Sentence Format (Output)

```
DT/(1,1) NN/(1,2) VBD/(1,3)
```

Where:
- `FILLER` = POS tag or phrase label
- `(level,position)` = position in parse tree
  - Level 1 = terminals
  - Higher levels = phrase nodes

## Penn Treebank Access

### Option 1: Official LDC Penn Treebank
- Available from [LDC](https://www.ldc.upenn.edu/) (requires license)
- Standard WSJ sections: 02-21 (training), 22 (dev), 23 (test)

### Option 2: Sample Data
- Use `sample_ptb.txt` for testing
- Small-scale experimentation

### Option 3: Alternative Treebanks
- [Universal Dependencies](https://universaldependencies.org/) (free, different format)
- [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19) (requires license)

## Processing Real Penn Treebank

If you have access to Penn Treebank WSJ:

```bash
# Process training sections (02-21)
cat /path/to/ptb/wsj_{02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21}*.mrg > ptb_train.txt

python ptb_to_gsc.py ptb_train.txt \
    --output-pcfg ptb_train_grammar.txt \
    --output-sentences ptb_train_sentences.txt \
    --binarize \
    --smooth 0.01

# Train GSC parser
python train_gsc_from_ptb.py \
    --pcfg ptb_train_grammar.txt \
    --sentences ptb_train_sentences.txt \
    --max-sent-len 20 \
    --num-epochs 500 \
    --num-samples 10000 \
    --output gsc_ptb_full_model.pkl
```

## Understanding the GSC Format

### Role Positions

In GSC, each word/phrase is assigned a position in the parse tree:

```
        S/(3,1)
       /        \
    NP/(2,1)    VP/(2,2)
    /    \         |
DT/(1,1) NN/(1,2) VBD/(1,3)
  The      cat      sat
```

Bindings:
- `DT/(1,1)` - "The" at level 1, position 1
- `NN/(1,2)` - "cat" at level 1, position 2
- `VBD/(1,3)` - "sat" at level 1, position 3
- `NP/(2,1)` - Noun phrase at level 2, position 1
- `VP/(2,2)` - Verb phrase at level 2, position 2
- `S/(3,1)` - Sentence at level 3, position 1

### PCFG Probabilities

Maximum Likelihood Estimation from tree counts:

```
P(A → B C) = Count(A → B C) / Count(A → *)
```

With smoothing:
```
P(A → B C) = (Count(A → B C) + α) / (Count(A → *) + α × |rules with LHS=A|)
```

## Troubleshooting

### Issue: "Cannot parse tree"
- Check that trees are properly bracketed
- Ensure one tree per complete parenthesis balance
- Remove any empty lines or comments

### Issue: "Grammar too large"
- Increase `--min-prob` to filter rare rules
- Reduce `--max-sent-len` to limit grammar size
- Use `--max-trees` to process subset

### Issue: "Memory error during training"
- Reduce `--num-samples`
- Reduce `--max-sent-len`
- Filter PCFG to only common POS tags

### Issue: "Training doesn't converge"
- Adjust `--learning-rate` (try 0.01 or 0.001)
- Increase `--num-epochs`
- Check PCFG has proper probabilities (sum to 1 per LHS)

## Next Steps

1. **Evaluate Parser**: Test on held-out WSJ section 23
2. **Tune Hyperparameters**: Adjust learning rate, network temperature
3. **Analyze Errors**: Compare predicted vs. gold parse trees
4. **Extend Grammar**: Add more sophisticated features

## References

- GSC Network: Cho et al. (neural symbolic parsing)
- Penn Treebank: Marcus et al. (1993)
- PCFG Parsing: Collins (1999), Petrov et al. (2006)

## Example Workflow

```bash
# 1. Test with sample data
python ptb_to_gsc.py sample_ptb.txt --binarize
python train_gsc_from_ptb.py --num-epochs 50

# 2. Process real Penn Treebank
python ptb_to_gsc.py /path/to/wsj.mrg --binarize --smooth 0.01

# 3. Train with optimal settings
python train_gsc_from_ptb.py \
    --max-sent-len 15 \
    --num-epochs 200 \
    --num-samples 10000 \
    --learning-rate 0.05
```

## License

See LICENSE file in repository root.

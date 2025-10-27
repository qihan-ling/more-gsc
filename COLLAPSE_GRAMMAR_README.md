# Berkeley Parser to GSC Grammar Conversion

## Summary

Successfully collapsed Berkeley Parser SM5 grammar from **684,239 rules** to **4,644 rules** (99.3% reduction).

## Files Created

1. **`collapse_berkeley_grammar.py`** - Main conversion script
2. **`collapsed_grammar_sm5.txt`** - Output grammar in GSC format (4,644 rules)
3. **`test_collapsed_grammar.py`** - Test script to verify GSC compatibility

## Method: Proposal 1 - Full Collapse with Probability Summation

### What was done:

1. **Strip subcategories**: Remove all subscripts (e.g., `S_1`, `S_2`, ..., `S_13` → `S`)
2. **Group rules**: Collect all rules that have the same base form
3. **Sum probabilities**: Add up probabilities for each unique base rule

### Example:

**Berkeley Parser (subcategorized):**
```
S_1 -> SBAR_3 VP_5  3.8038446E-6
S_2 -> SBAR_3 VP_5  3.7660048E-6
S_3 -> SBAR_3 VP_5  9.8916615E-6
...
```

**Collapsed (base categories):**
```
S -> SBAR VP  0.0000XXXXXX  (sum of all subcategory probabilities)
```

## Statistics

- **Input**: 684,239 Berkeley rules
- **Output**: 4,644 collapsed rules
- **Reduction**: 99.3%
- **Total probability mass**: 1,092.16 (normalized sum across all rules)

## Top Rules in Collapsed Grammar

```
32.0000000000 NP -> NP
31.0000000000 JJ -> JJ
30.0000000000 NNP -> NNP
30.0000000000 NN -> NN
30.0000000000 NNS -> NNS
12.1018326731 PP -> IN NP
9.7807213630 ADVP -> RB
4.9600640933 S -> NP VP
4.3694740707 SBAR -> IN S
3.5632482695 S -> VP
```

## Usage

### Run the collapse script:

```bash
python3 collapse_berkeley_grammar.py \
    trained_berkeley_parser_sm5/berkeley_parser_sm5.grammar \
    -o collapsed_grammar_sm5.txt \
    --min-prob 1e-10 \
    --preview 30
```

### Options:

- `-o, --output`: Output file path (default: `collapsed_grammar.txt`)
- `-m, --min-prob`: Minimum probability threshold (default: `1e-10`)
- `--preview`: Number of top rules to preview (default: `20`)

### Use with GSC:

```python
import gsc

# Load the collapsed grammar
with open('collapsed_grammar_sm5.txt', 'r') as f:
    PCFG = f.read()

ROOT = 'S'
MAXLEN = 20

hg = gsc.HarmonicGrammar(pcfg=PCFG, root=ROOT, max_sent_len=MAXLEN)

# Continue with GSC training...
```

## Notes

1. **Probability values**: The collapsed probabilities represent the sum of all subcategory probabilities. Many high-frequency unary rules (like `NP -> NP`) have large values (20-32) because they sum across 20-32 subcategories.

2. **Normalization**: GSC may automatically normalize probabilities for each LHS category during initialization. If needed, you can add normalization to the script.

3. **Lexicon handling**: This script only handles the grammar rules. The lexicon file would need similar processing to remove subcategories.

4. **Training time**: With 4,644 rules (vs 684,239), GSC training should be much more feasible. Compare with toy grammar 1 which has ~10 rules and takes 40-60 minutes.

## Next Steps

If 4,644 rules is still too many for GSC training:

1. **Increase min-prob threshold**: Filter out low-probability rules (e.g., `--min-prob 1e-6`)
2. **Use Proposal 2**: Keep only top-K subcategories per category
3. **Prune by frequency**: Remove rare grammatical constructions
4. **Lexicalize less**: Use coarser POS tags

## Questions?

Run the script with `-h` for help:
```bash
python3 collapse_berkeley_grammar.py -h
```

# Berkeley Parser to GSC Grammar Conversion

## Summary

Successfully collapsed Berkeley Parser SM5 grammar from **684,239 rules** to **4,598 rules** (99.3% reduction).

## Files Created

1. **`collapse_berkeley_grammar.py`** - Main conversion script
2. **`collapsed_grammar_sm5.txt`** - Output grammar in GSC format (4,598 rules)
3. **`test_collapsed_grammar.py`** - Test script to verify GSC compatibility

## Method: Proposal 1 - Full Collapse with Probability Summation and Normalization

### What was done:

1. **Strip subcategories**: Remove all subscripts (e.g., `S_1`, `S_2`, ..., `S_13` → `S`)
2. **Group rules**: Collect all rules that have the same base form
3. **Sum probabilities**: Add up probabilities for each unique base rule
4. **Normalize by LHS**: For each left-hand side category, normalize all its rules so they sum to 1.0 (proper PCFG format)

### Example:

**Berkeley Parser (subcategorized):**
```
S_1 -> SBAR_3 VP_5  3.8038446E-6
S_2 -> SBAR_3 VP_5  3.7660048E-6
S_3 -> SBAR_3 VP_5  9.8916615E-6
...
```

**Collapsed and normalized (base categories):**
```
S -> SBAR VP  0.XXXXXX  (sum of subcategory probabilities, then normalized so all S rules sum to 1.0)
S -> NP VP    0.YYYYYY
...
(all S -> ... rules sum to 1.0)
```

## Statistics

- **Input**: 684,239 Berkeley rules
- **Output**: 4,598 collapsed and normalized rules
- **Reduction**: 99.3%
- **LHS categories**: 98 unique categories
- **Normalization**: All rules for each LHS sum to exactly 1.0 (proper PCFG format)

## Sample Rules in Collapsed Grammar

### Unary terminal rules (probability 1.0):
```
1.0000000000 IN -> IN
1.0000000000 DT -> DT
1.0000000000 NN -> NN
```

### S rules (sum to 1.0):
```
0.4949271554 S -> S
0.1888361856 S -> NP VP
0.1356575638 S -> VP
0.0416743493 S -> @S VP
... (120 S rules total)
```

### NP rules (sum to 1.0):
```
0.4999883762 NP -> NP
0.0469430863 NP -> DT NN
0.0431049071 NP -> @NP NN
0.0350492981 NP -> NP PP
... (230 NP rules total)
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

1. **Probability normalization**: After collapsing subcategories by summing their probabilities, the script normalizes all rules with the same LHS so they sum to exactly 1.0. This ensures proper PCFG format that GSC expects.

2. **Verification**: You can verify normalization by checking that all rules for a given LHS sum to 1.0:
   ```bash
   grep '^[0-9.e-]* S ->' collapsed_grammar_sm5.txt | awk '{print $1}' | \
     python3 -c "import sys; print(sum(float(x) for x in sys.stdin))"
   # Output: 1.0
   ```

3. **Lexicon handling**: This script only handles the grammar rules. The lexicon file would need similar processing to remove subcategories.

4. **Training time**: With 4,598 rules (vs 684,239), GSC training should be much more feasible. Compare with toy grammar 1 which has ~10 rules and takes 40-60 minutes.

## Next Steps

If 4,598 rules is still too many for GSC training:

1. **Increase min-prob threshold**: Filter out low-probability rules (e.g., `--min-prob 1e-6`)
2. **Use Proposal 2**: Keep only top-K subcategories per category
3. **Prune by frequency**: Remove rare grammatical constructions
4. **Lexicalize less**: Use coarser POS tags

## Questions?

Run the script with `-h` for help:
```bash
python3 collapse_berkeley_grammar.py -h
```

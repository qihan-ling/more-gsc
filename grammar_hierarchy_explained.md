# Understanding Grammar Hierarchies in GSC

## The Question
**How do you make N, V into non-terminals instead of terminals?**

## Original cho_grammar1.py Approach

In `cho_grammar1.py`, symbols like `N`, `Vi`, `P` are **terminal symbols**:

```
PCFG_G1 = '''
0.35 S -> N Vi
0.30 S -> N Vi PP
1.0 PP -> P N
'''
```

**Parse tree:**
```
      S
     / \
    N  Vi
```

**GSC Input:** `N/(1,1) Vi/(1,2)`

The symbols `N` and `Vi` are the **leaf nodes** - they don't expand further.

---

## Penn Treebank Approach: Multi-Level Hierarchy

Penn Treebank has **multiple levels**:

1. **Phrasal level** (S, NP, VP, PP) - high-level structure
2. **POS level** (DT, NN, VBD) - part-of-speech tags
3. **Word level** (the, cat, sat) - actual words

### Option 1: Direct POS Grammar (Recommended)

Skip intermediate categories; use POS tags as terminals:

```python
PCFG_PTB_STYLE = '''
0.40 S -> NP VP
0.40 NP -> DT NN
0.35 VP -> VBD
1.0 PP -> IN NP
'''
```

**Parse tree:**
```
        S
       / \
      NP  VP
     / \   |
    DT NN VBD
   the cat sat
```

**GSC Input:** `DT/(1,1) NN/(1,2) VBD/(1,3)`

**This is what `ptb_to_gsc.py` produces!**

### Option 2: Abstracted Grammar (N/V as intermediate)

Add abstract categories `N` and `V` that expand to POS tags:

```python
PCFG_WITH_ABSTRACTIONS = '''
0.40 S -> NP VP
0.50 NP -> DT N         # N is now non-terminal
0.70 N -> NN            # N expands to POS tags
0.20 N -> NNP
0.10 N -> NNS

0.35 VP -> V            # V is non-terminal
0.40 V -> VBD           # V expands to POS tags
0.30 V -> VBZ
0.20 V -> VBP
'''
```

**Parse tree with abstractions:**
```
        S
       / \
      NP  VP
     / \   |
    DT  N  V     <- N and V are intermediate nodes
        |  |
       NN VBD    <- POS tags
       |   |
      cat sat    <- words
```

**GSC Input:** Still `NN/(1,1) VBD/(1,2)`

**Key insight:** The abstractions affect the **parse tree structure** but not the **GSC input format**!

---

## Practical Comparison

### cho_grammar1.py (Abstract Terminals)
```python
PCFG = '''
0.35 S -> N Vi
'''
```
- **Terminals:** N, Vi (abstract word classes)
- **Input:** `N/(1,1) Vi/(1,2)`
- **Use case:** Cognitive modeling, simplified grammars

### PTB Option 1 (Direct POS)
```python
PCFG = '''
0.40 S -> NP VP
0.40 NP -> DT NN
0.35 VP -> VBD
'''
```
- **Terminals:** DT, NN, VBD (POS tags)
- **Input:** `DT/(1,1) NN/(1,2) VBD/(1,3)`
- **Use case:** Penn Treebank training, practical parsing

### PTB Option 2 (With N/V Abstraction)
```python
PCFG = '''
0.40 S -> NP VP
0.50 NP -> DT N
0.70 N -> NN
0.35 VP -> V
0.40 V -> VBD
'''
```
- **Terminals:** Still NN, VBD (POS tags)
- **Input:** Still `NN/(1,1) VBD/(1,2)`
- **Use case:** Studying category abstraction

---

## Converting cho_grammar1.py to PTB Style

### Original (Abstract Terminals):
```python
PCFG_G1 = '''
0.35 S -> N Vi
0.30 S -> N Vi PP
1.0 PP -> P N
'''

# Generates sentences like:
# N Vi
# N Vi P N
```

### Converted to PTB Style:

**Option A - Map to POS directly:**
```python
PCFG_PTB = '''
0.35 S -> NP VP
0.30 S -> NP VP PP

0.50 NP -> NN           # "N" becomes NN
0.30 NP -> NNP
0.20 NP -> DT NN

0.40 VP -> VBD          # "Vi" becomes VBD
0.30 VP -> VBZ
0.30 VP -> VBP

1.0 PP -> IN NP         # "P" becomes IN
'''

# Generates sentences like:
# NN VBD
# DT NN VBD IN NN
```

**Option B - Keep N/V as intermediate:**
```python
PCFG_PTB_ABSTRACT = '''
0.35 S -> N V           # Keep abstract categories
0.30 S -> N V PP

# N expands to various noun types
0.40 N -> NN
0.30 N -> NNP
0.30 N -> DT NN

# V expands to various verb types
0.40 V -> VBD
0.30 V -> VBZ
0.30 V -> VBP

1.0 PP -> P N

# P expands to preposition
1.0 P -> IN
'''
```

---

## Recommendation

### For Penn Treebank Training: Use Option 1 (Direct POS)

**Why?**
- ✅ Simpler grammar (fewer non-terminals)
- ✅ Matches PTB annotation directly
- ✅ Faster training (fewer rules)
- ✅ POS tags already provide abstraction
- ✅ **Already implemented in `ptb_to_gsc.py`!**

### When to use Option 2 (N/V abstraction)?

Only if you need to:
- Model abstract grammatical categories explicitly
- Study category learning and generalization
- Match specific theoretical linguistics requirements
- Compare with psycholinguistic models

---

## Code Example: Updating cho_grammar1.py

To make `N` and `V` non-terminals in cho_grammar1.py:

```python
# BEFORE (N as terminal)
PCFG_G1 = '''
0.35 S -> N Vi
'''

# AFTER (N as non-terminal expanding to POS tags)
PCFG_G1_PTB = '''
0.35 S -> NP VP

# NP is a phrase containing N
1.0 NP -> N

# N is now an abstract category that expands to POS
0.50 N -> NN
0.30 N -> NNP
0.20 N -> PRP

# VP contains V
1.0 VP -> V

# V is abstract, expands to verb POS tags
0.40 V -> VBD
0.30 V -> VBZ
0.30 V -> VBP
'''
```

**Key change:** Add intermediate rules that make N/V expand to POS tags rather than being terminal themselves.

---

## Summary

| Approach | N/V Status | Terminals | GSC Input | Use Case |
|----------|------------|-----------|-----------|----------|
| cho_grammar1.py | Terminals | N, Vi, P | `N/(1,1) Vi/(1,2)` | Simple abstract grammars |
| PTB Direct POS | Not used | DT, NN, VBD | `DT/(1,1) NN/(1,2) VBD/(1,3)` | **Recommended for PTB** |
| PTB with N/V | Non-terminals | Still POS tags | `NN/(1,1) VBD/(1,2)` | Category learning research |

**Bottom line:** The `ptb_to_gsc.py` tool already produces the correct format. You don't need to explicitly model N/V as separate categories unless you have a specific research reason to do so!

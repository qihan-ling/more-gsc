# gsc.py
# Pyeong Whan Cho (pyeongwhan.cho@gmail.com)

import sys
import numpy as np
import numbers
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import itertools
import copy
import time
from matplotlib.patches import Rectangle
from collections import defaultdict
# JAX imports for GPU acceleration
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit
    from functools import partial
    JAX_AVAILABLE = True
    print("JAX detected - GPU acceleration enabled")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not found - running in CPU mode. Install with: pip install jax jaxlib")


def unique(fillers):
    '''Returns (list) of unique names of fillers (list of str)'''

    fillers = list(set(fillers))
    fillers.sort()
    return fillers


class Node():
    # Class PCFG uses this Node class to represent a binary tree structure

    def __init__(self, sym):
        self.sym = sym
        self.children = []
        self.mother = None

    def __str__(self):
        str1 = self.extract().replace('()', '').replace(', )', ')')[:-2]
        str1 = str1.replace('(', '( ').replace(')', ' )')
        return str1

    def add_child(self, node):
        self.children.append(node)
        self.children[-1].mother = self

    def extract(self, level=0):
        ret = self.sym + "("
        for child in self.children:
            ret += child.extract(level + 1)
        ret += "), "
        return ret

    def get_descendants(self):
        ret = []
        for child in self.children:
            ret.append(child)
            ret += child.get_descendants()
        return ret

    def get_terminals(self):
        terminals = []
        for node in self.get_descendants():
            if len(node.children) == 0:
                terminals.append(node)
        return terminals


class PCFG():

    def __init__(self, pcfg, root, opts=None):

        self._set_opts()
        self._update_opts(opts)

        if not isinstance(root, list):
            # Multiple root symbols are allowed
            root = [root]
        self.root = root

        self.pcfg_str = pcfg
        self._cnf()
        self._cnf2hnf()
        self._tokenize_cnf()  # Use original, not optimized
        # self._tokenize_cnf_optimized()  # BUG: Creates extra non-terminal mothers → extra fillers → 405 bindings instead of 195
        print("DEBUG1")
        self._tokenize_fillers()  # Use original, not optimized
        # self._tokenize_fillers_optimized()
        print("DEBUG2")
        self._sort_rules()
        # lookup will still be very sloe for 10K grammar
        print("DEBUG3")
        # self._create_fast_lookups_pcfg()
        self._create_fastER_lookups_pcfg()

    def _set_opts(self):
        # the default setting

        self.opts = {}

        self.opts['add_null'] = True
        self.opts['f_empty'] = '@'
        self.opts['f_root'] = '#'

        self.opts['use_hnf'] = False
        self.opts['use_pos_f'] = True
        self.opts['add_copy_rules'] = False
        # self.opts['use_minimal_copy_rules'] = True  # not matter much

        self.opts['pos_m'] = ['l', 'r', 'm', 'l0', 'r0']
        self.opts['pos_d'] = ['l', 'r', 'm', 'l0', 'r0']
        self.opts['pos_s'] = ['l', 'r']
        self.opts['pos_f'] = ['0', '1', '9']

        self.opts['pos_copy'] = 'l'  # (l)eft or (r)ight
        self.opts['copy'] = '*'
        self.opts['null'] = '_'
        self.opts['sep'] = ':'

    def _update_opts(self, opts):
        # Update opts

        if opts is not None:
            for key, val in opts.items():
                if key in self.opts.keys():
                    self.opts[key] = val
        if not self.opts['use_hnf']:
            self.opts['pos_m'] = self.opts['pos_m'][0:2]
            self.opts['pos_d'] = self.opts['pos_d'][0:2]
        if self.opts['use_pos_f']:
            if not self.opts['use_hnf']:
                self.opts['pos_f'] = self.opts['pos_f'][0:2]
        else:
            self.opts['pos_f'] = None

    def _cnf(self):
        # pcfg_str: "prob X -> Y Z"
        # {'m': fname_m, 'd1': fname_d1, 'd2': fname_d2, 'p': probability}

        rules = [rule_str.strip() for rule_str in self.pcfg_str.split('\n')]

        # remove comments
        rules = [rule.split('#')[0].strip() for rule in rules]
        rules = [rule for rule in rules if rule != '']

        rules_new = []
        for rule in rules:
            rule_dict = {}
            LHS, RHS = rule.split('->')
            prob, LHS = [term for term in LHS.strip().split(' ') if term != '']
            RHS = RHS.strip().split(' ')
            rule_dict['m'] = LHS
            for ti, term in enumerate(RHS):
                rule_dict['d' + str(ti+1)] = term
            rule_dict['p'] = float(prob)
            rules_new.append(rule_dict)

        self.rules = rules_new
        self._add_names()

        # Normalize
        nonterminals = self.get_nonterminals()
        for LHS in nonterminals:
            rule_subset = self.get_rules(subset={'m': LHS})
            prob = np.array([rule['p'] for rule in rule_subset])
            prob = prob / prob.sum()
            for rulei, rule in enumerate(rule_subset):
                rule['p'] = prob[rulei]

    def _cnf2hnf(self):
        # Convert (non-tokenized) CNF to HNF (harmonic normal form)
        # and assign it (list of dict) to self.hnf.
        # {'m': fname_m, 'd1': fname_d1, 'd2': fname_d2 or None }

        mothers = [rule['m'] for rule in self.rules]
        mothers_unique = list(set(mothers))

        rules_hnf = []
        for mother in mothers_unique:
            rules = [rule for rule in self.rules if rule['m'] == mother]
            for rulei, rule in enumerate(rules):
                bracketed_sym = rule['m'] + '[{:d}]'.format(rulei + 1)

                rule_1 = {'m': rule['m'], 'd1': bracketed_sym}
                for key, val in rule.items():
                    if key not in ['m', 'd1', 'p']:
                        rule_1[key] = None
                rule_1['p'] = rule['p']

                rule_2 = {'m': bracketed_sym}
                for key, val in rule.items():
                    if key not in ['m', 'p']:
                        rule_2[key] = val
                rule_2['p'] = 1.

                rules_hnf.append(rule_1)
                rules_hnf.append(rule_2)

        self.rules = rules_hnf
        self._add_names()

    def _tokenize_cnf_optimized(self):
        if not self.opts['use_hnf']:
            t0 = time.time()
            # Separate rules by type once
            unary_rules = [r for r in self.rules if r['d2'] is None]
            binary_rules = [r for r in self.rules if r['d2'] is not None]

            mothers = set(rule['m'] for rule in self.rules)
            sym_prob = {rule['d1']: rule['p'] for rule in unary_rules}

            # Build lookup from unary rules only
            mother_to_children = {}
            for rule in unary_rules:
                mother_to_children.setdefault(rule['m'], []).append(rule['d1'])

            rules_seen = set()
            rules_new = []

            for rule in binary_rules:
                d1_syms = mother_to_children.get(rule['d1'], [rule['d1']])
                d2_syms = mother_to_children.get(rule['d2'], [rule['d2']])

                for d1_sym in d1_syms:
                    for d2_sym in d2_syms:
                        p = (sym_prob.get(rule['m'], 1.0) *
                             sym_prob.get(d1_sym, 1.0) *
                             sym_prob.get(d2_sym, 1.0))

                        rule_key = (rule['m'], d1_sym, d2_sym, p)
                        if rule_key not in rules_seen:
                            rules_seen.add(rule_key)
                            rules_new.append({'m': rule['m'], 'd1': d1_sym,
                                              'd2': d2_sym, 'p': p})

            self.rules = rules_new
            self._add_names()
            print(f"    Lookup tables built in {time.time()-t0:.2f}s")

    def _tokenize_cnf(self):
        # Tokenize CNF by replacing each type symbol (e.g., X)
        # with a list of tokens (e.g., X[1], X[2], etc).
        # (In this implementation, a tokenized CNF will be created
        # by removing unary branching rules in the HNF of grammar.)

        if not self.opts['use_hnf']:

            rules_new = []
            mothers = [rule['m'] for rule in self.rules]
            mothers = list(set(mothers))

            sym_prob = {}
            for rule in self.rules:
                if rule['d2'] is None:
                    sym_prob[rule['d1']] = rule['p']

            for rule in self.rules:
                if rule['d2'] is not None:

                    if rule['d1'] in mothers:
                        # d1 is a non-terminal symbol. Find tokens of d1
                        # d1_syms = [rr['d1'] for rr in self.hnf
                        d1_syms = [rr['d1'] for rr in self.rules
                                   if rr['m'] == rule['d1']]
                    else:
                        d1_syms = [rule['d1']]

                    if rule['d2'] in mothers:
                        # d2 is a non-terminal symbol. Find tokens of d2
                        # d2_syms = [rr['d1'] for rr in self.hnf
                        d2_syms = [rr['d1'] for rr in self.rules
                                   if rr['m'] == rule['d2']]
                    else:
                        d2_syms = [rule['d2']]

                    for d1_sym in d1_syms:
                        for d2_sym in d2_syms:
                            p = 1.0
                            for sym in [rule['m'], d1_sym, d2_sym]:
                                if sym in sym_prob:
                                    p *= sym_prob[sym]
                            rule_new = {'m': rule['m'],
                                        'd1': d1_sym,
                                        'd2': d2_sym,
                                        'p': p}
                            if rule_new not in rules_new:
                                rules_new.append(rule_new)

            self.rules = rules_new
            self._add_names()

    def _tokenize_fillers_optimized(self):
        """Replace filler symbols with position-specific filler symbols.

        OPTIMIZED VERSION - O(n) instead of O(n²)
        """
        import time

        if self.opts['use_pos_f']:
            t0 = time.time()

            sep = self.opts['sep']
            role_names = self.opts['pos_f']

            rules = self.rules.copy()
            mothers = unique([rule['m'] for rule in rules])

            # ==================== OPTIMIZATION 1: Build lookup structures ====================
            # Pre-group rules by where each symbol appears (O(n) preprocessing)
            rules_with_d1 = {}  # symbol -> [rules where d1 == symbol]
            rules_with_d2 = {}  # symbol -> [rules where d2 == symbol]
            unary_rule_indices = set()  # Track which rules are unary

            for i, rule in enumerate(rules):
                # Cache unary status
                if rule.get('d2') is None:
                    unary_rule_indices.add(i)

                # Build d1 lookup
                if rule['d1'] is not None:
                    if rule['d1'] not in rules_with_d1:
                        rules_with_d1[rule['d1']] = []
                    rules_with_d1[rule['d1']].append((i, rule))

                # Build d2 lookup
                if rule.get('d2') is not None:
                    if rule['d2'] not in rules_with_d2:
                        rules_with_d2[rule['d2']] = []
                    rules_with_d2[rule['d2']].append((i, rule))

            print(f"    Built lookup structures in {time.time()-t0:.3f}s")
            # ================================================================================

            # ==================== OPTIMIZATION 2: Build type->token mapping =================
            type_token_dict = {}  # mother -> list of tokens

            for mother in mothers:
                tokens = []

                # Use lookup dicts instead of scanning all rules
                for i, rule in rules_with_d1.get(mother, []):
                    if i in unary_rule_indices:  # O(1) check
                        tokens.append(rule['d1'] + sep + role_names[2])
                    else:
                        tokens.append(rule['d1'] + sep + role_names[0])

                for i, rule in rules_with_d2.get(mother, []):
                    if i not in unary_rule_indices:  # O(1) check
                        tokens.append(rule['d2'] + sep + role_names[1])

                if self.get_types(mother)[0] in self.root:
                    tokens.append(mother + sep + role_names[0])

                # Remove duplicates and store
                type_token_dict[mother] = list(set(tokens))

            print(f"    Built type->token mapping in {time.time()-t0:.3f}s")
            # ================================================================================

            # ==================== OPTIMIZATION 3: Generate new rules with set dedup =========
            rules_new = []
            rules_seen = set()  # Use set for O(1) deduplication

            for i, rule in enumerate(rules):
                # O(1) dict lookup instead of O(n) list comprehension
                tokens = type_token_dict.get(rule['m'], [])
                if len(tokens) == 0:
                    tokens = [rule['m']]

                for token in tokens:
                    if i in unary_rule_indices:  # O(1) check
                        d1_new = rule['d1'] + sep + role_names[2]

                        # Create hashable key for deduplication
                        rule_key = (token, d1_new, None, rule['p'])

                        if rule_key not in rules_seen:  # O(1) set lookup
                            rules_seen.add(rule_key)
                            rules_new.append({
                                'm': token,
                                'd1': d1_new,
                                'd2': None,
                                'p': rule['p']
                            })

                    else:  # Binary rule
                        d1_new = rule['d1'] + sep + \
                            role_names[0] if rule['d1'] is not None else None
                        d2_new = rule['d2'] + sep + \
                            role_names[1] if rule['d2'] is not None else None

                        rule_key = (token, d1_new, d2_new, rule['p'])

                        if rule_key not in rules_seen:  # O(1) set lookup
                            rules_seen.add(rule_key)
                            rules_new.append({
                                'm': token,
                                'd1': d1_new,
                                'd2': d2_new,
                                'p': rule['p']
                            })

            print(
                f"    Generated {len(rules_new)} rules in {time.time()-t0:.3f}s")
            # ================================================================================

            # Add separator to root symbols if needed
            for rule in rules_new:
                if self.opts['sep'] not in rule['m']:
                    rule['m'] += sep + role_names[0]

            self.rules = rules_new
            self._add_names()

            print(f"    _tokenize_fillers completed in {time.time()-t0:.3f}s")

    def _tokenize_fillers(self):
        # Replace filler symbols with position-specific filler symbols

        if self.opts['use_pos_f']:

            sep = self.opts['sep']
            role_names = self.opts['pos_f']

            rules = self.rules.copy()
            mothers = unique([rule['m'] for rule in rules])

            rules_new = []
            type_token_pairs = []

            for mother in mothers:
                tokens = []

                for rule in rules:
                    if self.is_hnf_unary_rule(rule):
                        if rule['d1'] == mother:
                            tokens.append(rule['d1'] + sep + role_names[2])
                    else:
                        if rule['d1'] == mother:
                            tokens.append(rule['d1'] + sep + role_names[0])
                        if rule['d2'] == mother:
                            tokens.append(rule['d2'] + sep + role_names[1])

                if self.get_types(mother)[0] in self.root:
                    tokens.append(mother + sep + role_names[0])

                type_token_pairs.append({'type': mother, 'token': tokens})

            for rule in rules:

                tokens = [pair['token'] for pair in type_token_pairs
                          if pair['type'] == rule['m']][0]
                if len(tokens) == 0:
                    tokens = [rule['m']]

                for token in tokens:
                    if self.is_hnf_unary_rule(rule):
                        rule_new = {'m': token,
                                    'd1': rule['d1'] + sep + role_names[2],
                                    'd2': None,
                                    'p': rule['p']}
                        if rule_new not in rules_new:
                            rules_new.append(rule_new)

                    else:
                        if rule['d1'] is not None:
                            d1 = rule['d1'] + sep + role_names[0]
                        else:
                            d1 = None
                        if rule['d2'] is not None:
                            d2 = rule['d2'] + sep + role_names[1]
                        else:
                            d2 = None

                        rule_new = {'m': token, 'd1': d1,
                                    'd2': d2, 'p': rule['p']}
                        if rule_new not in rules_new:
                            rules_new.append(rule_new)

            for rule in rules_new:
                # Root symbols may occur on the RHS of a recursive rewrite rule.
                # In those cases, pos_f symbols will be added to those root symbols.
                # For consistency, add pos_f symbols to root symbols
                # when a grammar does not have a recursive rule with root symbols.
                if self.opts['sep'] not in rule['m']:
                    rule['m'] += sep + role_names[0]

            self.rules = rules_new
            self._add_names()

    def _add_names(self):

        fnames = [val for rule in self.rules for key, val in rule.items()
                  if (key != 'p') and (val is not None)]
        fnames = list(set(fnames))
        fnames.sort()

        if self.opts['add_null']:
            fnames.append(self.opts['null'])

        self.filler_names = fnames
        # print(f"DEBUG: PCFG _add_names filler names is {self.filler_names}")

    def _sort_rules(self):

        expansion_rules = [rule for rule in self.rules
                           if (self.opts['f_empty'] in rule['m']) or
                              (self.opts['f_root'] in rule['m'])]
        copy_rules = self.subset_copy_rules()
        non_copy_rules = [rule for rule in self.rules
                          if (rule not in copy_rules) and
                             (rule not in expansion_rules)]

        copy_rules_sorted = sorted(copy_rules, key=lambda x: x['m'])
        non_copy_rules_sorted = sorted(non_copy_rules, key=lambda x: x['m'])
        expansion_rules_sorted = sorted(
            expansion_rules, key=lambda x: x['m'])

        self.rules = non_copy_rules_sorted + copy_rules_sorted + expansion_rules_sorted

    def _create_fastER_lookups_pcfg(self):
        '''Pre-compute lookup arrays for PCFG filler operations.

        Creates cached arrays for fast filler property checks during training.

        OPTIMIZED VERSION:
        - Single pass through fillers instead of 5 separate passes
        - Pre-computed sets for O(1) lookups instead of calling expensive methods
        - Complexity: O(N + R) instead of O(N² × R)
        '''
        import time
        t0 = time.time()

        num_fillers = len(self.filler_names)
        num_rules = len(self.rules)

        # ==================== PHASE 1: Pre-compute sets (O(N + R)) ====================

        # 1. Build nonterminal set (O(R))
        nonterminal_set = set(rule['m'] for rule in self.rules)

        # 2. Cache filler types if needed (O(N))
        #    Only compute if we need them for terminal check
        root_types = set(self.root)
        null_symbol = self.opts['null']

        # Build filler type cache
        filler_types = {}
        for fname in self.filler_names:
            filler_types[fname] = self.get_types(fname)[0]

        # 3. Build terminal set (O(N))
        #    Terminal = not nonterminal AND not null AND type not in root
        terminal_set = {
            fname for fname in self.filler_names
            if (fname not in nonterminal_set and
                fname != null_symbol and
                filler_types[fname] not in root_types)
        }

        # 4. Build bracketed set (O(R))
        bracketed_set = set()
        if self.opts['use_hnf']:
            copy_symbol = self.opts['copy']
            for rule in self.rules:
                if (rule.get('d2') is None and              # Unary rule
                    rule['m'] != rule['d1'] and             # Not identity
                        rule['m'] != rule['d1'] + copy_symbol):  # Not copy rule
                    bracketed_set.add(rule['d1'])

        # 5. Build root set (already have this, just ensure it's a set)
        roots_set = set(self.get_roots())

        # 6. Cache copy symbol
        copy_symbol = self.opts['copy']

        # ==================== PHASE 2: Single pass through fillers (O(N)) ====================

        # Pre-allocate arrays
        self.filler_is_terminal = np.empty(num_fillers, dtype=bool)
        self.filler_is_copy = np.empty(num_fillers, dtype=bool)
        self.filler_is_bracketed = np.empty(num_fillers, dtype=bool)
        self.filler_is_root = np.empty(num_fillers, dtype=bool)

        # Single pass with O(1) set lookups
        self.filler_name_to_idx = {}
        for i, fname in enumerate(self.filler_names):
            # Build index mapping
            self.filler_name_to_idx[fname] = i

            # All property checks are now O(1) set membership tests
            self.filler_is_terminal[i] = fname in terminal_set
            self.filler_is_copy[i] = copy_symbol in fname
            self.filler_is_bracketed[i] = fname in bracketed_set
            self.filler_is_root[i] = fname in roots_set

        elapsed = time.time() - t0
        print(
            f"    Fast lookups built in {elapsed:.3f}s ({num_fillers} fillers, {num_rules} rules)")

    def _create_fast_lookups_pcfg(self):
        '''Pre-compute lookup arrays for PCFG filler operations.

        Creates cached arrays for fast filler property checks during training.
        '''
        # Name → index mapping
        self.filler_name_to_idx = {
            name: i for i, name in enumerate(self.filler_names)
        }

        # Pre-compute filler properties
        self.filler_is_terminal = np.array([
            self.is_terminal(fname) for fname in self.filler_names
        ], dtype=bool)

        self.filler_is_copy = np.array([
            self.opts['copy'] in fname for fname in self.filler_names
        ], dtype=bool)

        self.filler_is_bracketed = np.array([
            self.is_bracketed(fname) for fname in self.filler_names
        ], dtype=bool)

        roots = self.get_roots()
        self.filler_is_root = np.array([
            fname in roots for fname in self.filler_names
        ], dtype=bool)

    def get_fillers(self, idx=None):
        '''Returns (list) of filler names for idx (int or list of int)'''

        if idx is None:
            return self.filler_names
        else:
            if not isinstance(idx, list):
                idx = [idx]
            return [self.filler_names[ii] for ii in idx]

    def find_fillers(self, fnames):
        '''Returns (list) of indices for fnames (str or list of str)'''

        if not isinstance(fnames, list):
            fnames = [fnames]
        return [fi for fi, fname in enumerate(self.filler_names)
                if fname in fnames]

    def get_copy(self, fname):
        '''Returns a copy version (str) of fname (str).

        Regardless of whether fname is used in a given grammar,
        it will create its copy version. An exception is when
        fname itself is a copy version of another symbol.
        In this case, this method returns None.
        '''

        if self.opts['copy'] not in fname:
            if self.opts['sep'] in fname:
                fname, role = fname.split(self.opts['sep'])
                if self.opts['pos_copy'] == 'l':
                    fname = self.opts['copy'] + fname
                elif self.opts['pos_copy'] == 'r':
                    fname = fname + self.opts['copy']
                fname = fname + self.opts['sep'] + role
            else:
                if self.opts['pos_copy'] == 'l':
                    fname = self.opts['copy'] + fname
                elif self.opts['pos_copy'] == 'r':
                    fname = fname + self.opts['copy']
            return fname

        else:
            return None

    def get_uncopy(self, fname):
        '''Returns an original version (str) of fname (str).

        It does not guarantee that the uncopied version of
        fname is used in a given grammar. If fname itself
        is not a copy version of another symbol, the method
        returns None.
        '''

        if self.opts['copy'] in fname:
            if self.opts['sep'] in fname:
                fname, rname = fname.split(self.opts['sep'])
                if self.opts['pos_copy'] == 'l':
                    fname = fname.split(self.opts['copy'])[1]
                elif self.opts['pos_copy'] == 'r':
                    fname = fname.split(self.opts['copy'])[0]
                fname = fname + self.opts['sep'] + rname

            else:
                if self.opts['pos_copy'] == 'l':
                    fname = fname.split(self.opts['copy'])[1]
                elif self.opts['pos_copy'] == 'r':
                    fname = fname.split(self.opts['copy'])[0]

            return fname
        else:
            return None

    def is_copy(self, fname1, fname2=''):
        '''Returns (bool) after checking fname1 (str) is a copy version
        of fname2 (str). If fname2 is not given, it will test if
        fname1 (str) is a copy version of any other symbol.'''

        if fname2 == '' and fname1 is not None:
            return self.opts['copy'] in fname1
        else:
            return ((fname1 is not None) and (fname2 is not None)) and\
                   (fname1 == self.get_copy(fname2))

    def is_copy_rule(self, rule):
        '''Returns (bool) after checking whether rule (dict) is a copy rule.'''

        return self.is_copy(rule['m'])

    def subset_copy_rules(self):
        '''Returns (list) of copy rules (dict)'''

        return [rule for rule in self.rules if self.is_copy_rule(rule)]

    def get_types(self, fnames_token, ignore_copy=True,
                  ignore_bracket=True, ignore_pos_f=True):
        '''Returns (list) of type names for fnames_token (str or list of str).

        If ignore_copy is True, it removes copy symbols (opts['copy']). If
        ignore_bracket is True, it removes bracket symbols. If ignore_pos_f
        is True, it removes context-free roles (opts['pos_f']).
        '''

        if not isinstance(fnames_token, list):
            fnames_token = [fnames_token]

        if ignore_pos_f and self.opts['use_pos_f']:
            fnames_new = []
            for f in fnames_token:
                if self.opts['sep'] in f:
                    fnames_new.append(f.split(self.opts['sep'])[0])
                else:
                    fnames_new.append(f)
            fnames_token = fnames_new
        if ignore_bracket:
            fnames_new = []
            for f in fnames_token:
                if '[' in f:
                    fnames_new.append(f.split('[')[0] + f.split(']')[1])
                else:
                    fnames_new.append(f)
            fnames_token = fnames_new
        if ignore_copy:
            fnames_new = []
            for f in fnames_token:
                if self.is_copy(f):
                    fnames_new.append(self.get_uncopy(f))
                else:
                    fnames_new.append(f)
        return fnames_new

    def find_fillers_type(self, fnames_type, ignore_copy=True,
                          ignore_bracket=True, ignore_pos_f=True):
        '''Returns (list) of the indices of token filler names for
        fnames_type (str or list of str).

        Be sure to provide correct type filler names and set the parameters
        ignore_copy, ignore_bracket, ignore_pos_f for your chosen abstraction
        level. See also get_types().
        '''

        if not isinstance(fnames_type, list):
            fnames_type = [fnames_type]

        fnames = self.get_types(self.filler_names,
                                ignore_pos_f=ignore_pos_f,
                                ignore_bracket=ignore_bracket,
                                ignore_copy=ignore_copy)
        return [fi for fi, fname in enumerate(fnames)
                if fname in fnames_type]

    def get_fillers_type(self, fnames_type, ignore_copy=True,
                         ignore_bracket=True, ignore_pos_f=True):
        '''Returns (list) of token filler names of fnames_type (str or
        list of str).

        Be sure to provide correct type filler names and set the parameters
        ignore_copy, ignore_bracket, ignore_pos_f for your chosen abstraction
        level. See also get_types().
        '''

        return self.get_fillers(self.find_fillers_type(
            fnames_type, ignore_pos_f=ignore_pos_f,
            ignore_bracket=ignore_bracket, ignore_copy=ignore_copy))

    def get_rules(self, subset=None, rules=None):
        '''Returns (list) of rules (dict) satisfying searching condition
        subset (dict).

        subset (dict) can have any combination of three keys 'm', 'd1', 'd2'
        representing mother, first/left, and second/right daughter respecitvely
        in treelets defined by rewrite rules. The values for the keys must be
        (list of str).

        Example:
            >> g.get_rules(subset={'m': g.get_roots()})
            >> g.get_rules(subset={'d1': ['A:0', '*A:0']})
        '''

        subset0 = {'m': [], 'd1': [], 'd2': []}

        if subset is not None:
            for key, val in subset.items():
                if not isinstance(val, list):
                    val = [val]
                subset0[key] = val

        if rules is None:
            rules = self.rules.copy()

        if len(subset0['m']) > 0:
            rules = [rule for rule in rules
                     if rule['m'] in subset0['m']]
        if len(subset0['d1']) > 0:
            rules = [rule for rule in rules
                     if rule['d1'] in subset0['d1']]
        if len(subset0['d2']) > 0:
            rules = [rule for rule in rules
                     if rule['d2'] in subset0['d2']]
        return rules

    def read_rules(self, subset=None, decimals=4):
        '''Print rewriate rules satisfying conditions specified
        in subset (dict). When subset is not given, print all
        rules.

        See also get_rules() to check how to define subset (dict).
        '''

        rules = self.get_rules(subset=subset)

        for rule in rules:
            if rule['p'] is not None:
                if rule['d2'] is None:
                    print(('({:.{decimals}f}) '.format(
                        rule['p'], decimals=decimals)) + rule['m'] + ' -> ' + rule['d1'])
                elif rule['d1'] is None:
                    print(('({:.{decimals}f}) '.format(
                        rule['p'], decimals=decimals)) + rule['m'] + ' -> ' + rule['d2'])
                else:
                    print(('({:.{decimals}f}) '.format(
                        rule['p'], decimals=decimals)) + rule['m'] +
                        ' -> ' + rule['d1'] + ' ' + rule['d2'])

        rules_added = False
        for rule in rules:
            if rule['p'] is None:
                rules_added = True
                break

        if rules_added:
            print('-' * 40)
            print('Additional rules for brick roles')
            print('-' * 40)
            for rule in rules:
                if rule['p'] is None:
                    if rule['d2'] is None:
                        print(rule['m'] + ' -> ' + rule['d1'])
                    elif rule['d1'] is None:
                        print(rule['m'] + ' -> ' + rule['d2'])
                    else:
                        print(rule['m'] + ' -> ' +
                              rule['d1'] + ' ' + rule['d2'])

    def has_rule(self, rule):
        '''Returns (bool) after testing whether rule (dict) is in
        the current rule set.'''

        return rule in self.get_rules()

    def get_nonterminals(self):
        '''Returns (list) of names (str) of non-terminal symbols.'''

        mothers = [rule['m'] for rule in self.get_rules()]
        mothers = list(set(mothers))
        mothers.sort()
        return mothers

    def find_nonterminals(self):
        '''Returns (list) of indices (int) of nonterminal symbols.'''

        return self.find_fillers(self.get_nonterminals())

    def get_terminals(self):
        '''Returns (list) of names (str) of terminal symbols, excluding
        a null symbol if any.'''

        return [f for f in self.filler_names
                if (f not in self.get_nonterminals()) and
                   (f != self.opts['null']) and
                   (self.get_types(f)[0] not in self.root)]  # and (f != self.opts['f_empty'])]

    def find_terminals(self):
        '''Returns (list) of indices (int) of terminal symbols, excluding
        a null symbol if any.'''

        return self.find_fillers(self.get_terminals())

    def is_terminal(self, fname):
        '''Returns (bool), whether whether fname (str) is a terminal symbol.'''

        return fname in self.get_terminals()

    def is_bracketed(self, fname):
        '''Returns (bool) whether fname (str) is a bracketed symbol.

        Note that the method does not test whether a symbol has brackets
        in its name. If opts['use_hnf'] is False, symbol names with brackets
        do not have any special status. By a "bracketed symbol", we mean
        a symbol that is a daughter in a non-copy, unary branching rule
        when opts['use_hnf'] is True.'''

        return self.opts['use_hnf'] and \
            (fname in [rule['d1'] for rule in self.rules
                       if (rule['d2'] is None) and
                          (rule['m'] != rule['d1']) and
                          (rule['m'] != rule['d1'] + self.opts['copy'])])

    def get_bracketed(self):
        '''Returns (list) of bracketed symbols (str).'''

        return [fname for fi, fname in enumerate(self.filler_names)
                if self.is_bracketed(fname)]

    def find_bracketed(self):
        '''Returns (list) of indices (int) of bracketed symbols.'''

        return self.find_fillers(self.get_bracketed())

    def get_mothers(self, fname):
        '''Returns (dict) of potential mothers (str) of fname (str).

        The returned dictionary may have different keys depending on
        which formalism is chosen. Check opts['pos_m'] to see the keys
        for different mother positions. For example,
        {'l': ['*C:1', 'C:1'], 'r': []} suggests that fname can have
        either *C:1 or C:1 as its mother on the left upward branching
        direction. In other words, fname can be a right daughter of
        either *C:1 or C:1.
        '''

        pos_m = self.opts['pos_m']
        res = {}
        for key in pos_m:
            res[key] = []

        if self.is_bracketed(fname):
            mothers_m = [rule['m'] for rule in self.rules
                         if (rule['d1'] == fname) and
                            (not self.is_copy_rule(rule))]
            res[pos_m[2]] += mothers_m
        else:
            mothers_r = [rule['m'] for rule in self.rules
                         if (rule['d1'] == fname) and
                            (not self.is_copy_rule(rule))]
            mothers_l = [rule['m'] for rule in self.rules
                         if (rule['d2'] == fname) and
                            (not self.is_copy_rule(rule))]
            res[pos_m[1]] += mothers_r
            res[pos_m[0]] += mothers_l

            if self.opts['add_copy_rules']:
                mothers_r0 = [rule['m'] for rule in self.rules
                              if (rule['d1'] == fname) and
                                 (self.is_copy_rule(rule))]
                mothers_l0 = [rule['m'] for rule in self.rules
                              if (rule['d2'] == fname) and
                                 (self.is_copy_rule(rule))]
                if self.opts['use_hnf']:
                    res[pos_m[4]] += mothers_r0
                    res[pos_m[3]] += mothers_l0
                else:
                    res[pos_m[1]] += mothers_r0
                    res[pos_m[0]] += mothers_l0

        for key, val in res.items():
            val = list(set(val))
            val.sort()
            res[key] = val

        return res

    def find_mothers(self, fname):
        '''Returns (dict) of indices (list of int) of potential mothers
        for different potential mother positions.

        See also get_mothers().'''

        res = {}
        for key, val in self.get_mothers(fname).items():
            res[key] = self.find_fillers(val)
        return res

    def is_mother(self, fname_m, fname_d=None):
        '''Returns (bool) whether fname_m (str) is a potential mother of
        fname_d (str). If fname_d is not given, it tests whether fname_m
        is a non-terminal symbol.
        '''

        if fname_d is None:
            return not self.is_terminal(fname_m)
        else:
            mothers = []
            for key, val in self.get_mothers(fname_d).items():
                mothers += val
            return fname_m in mothers

    def is_unary_rule(self, rule):
        '''Returns (bool) on whether rule (dict) is a unary rule or not.'''

        return (rule['d1'] is None) or (rule['d2'] is None)

    def is_hnf_unary_rule(self, rule):
        '''Returns (bool) on whether rule (dict) is a non-copy, unary branching
        rule in HNF.'''

        return (self.opts['use_hnf'] and
                self.is_unary_rule(rule) and
                not self.is_copy_rule(rule))

    def get_daughters(self, fname):
        '''Returns (dict) of potential duaghters of fname (str)
        in different daughter positions (cf., see opts['pos_d']).'''

        pos_d = self.opts['pos_d']
        res = {}
        for key in pos_d:
            res[key] = []

        hnf_unary_rules = [rule for rule in self.get_rules()
                           if self.is_hnf_unary_rule(rule)]
        binary_rules = [rule for rule in self.get_rules()
                        if not self.is_unary_rule(rule)]
        copy_rules = [rule for rule in self.get_rules()
                      if self.is_copy_rule(rule)]

        daughters_l = unique([rule['d1'] for rule in binary_rules
                              if rule['m'] == fname])
        daughters_r = unique([rule['d2'] for rule in binary_rules
                              if rule['m'] == fname])
        res[pos_d[0]] = daughters_l
        res[pos_d[1]] = daughters_r

        if self.opts['use_hnf']:
            daughters_m = unique([rule['d1'] for rule in hnf_unary_rules
                                  if rule['m'] == fname])
            res[pos_d[2]] = daughters_m

        if self.opts['add_copy_rules']:
            daughters_l0 = [rule['d1'] for rule in copy_rules
                            if rule['d1'] is not None and rule['m'] == fname]
            daughters_r0 = [rule['d2'] for rule in copy_rules
                            if rule['d2'] is not None and rule['m'] == fname]
            if self.opts['use_hnf']:
                res[pos_d[3]] = daughters_l0
                res[pos_d[4]] = daughters_r0
            else:
                res[pos_d[0]] += daughters_l0
                res[pos_d[1]] += daughters_r0

        return res

    def find_daughters(self, fname):
        '''Returns (dict) of indices (int) of potential duaghters of fname (str)
        in different daughter positions (cf., see opts['pos_d']).

        See also get_daughters().'''

        daughters = self.get_daughters(fname)
        for key, val in daughters.items():
            daughters[key] = self.find_fillers(val)
        return daughters

    def has_mother(self, fname):
        '''Returns (bool) on whether fname (str) can have a mother'''

        res = self.find_mothers(fname)
        mothers = []
        for key, val in res.items():
            mothers += val
        return len(mothers) > 0

    def get_roots(self):
        '''Returns (list of str) of root symbols.

        Note that when opts['use_hnf'] is False, a grammar may have
        multiple root symbols.'''

        # if self.opts['root'] is not None:
        #     roots = self.opts['root']
        # else:
        #     roots = [f for f in self.get_fillers()
        #              if (not self.has_mother(f)) and (f != self.opts['null'])]

        # roots = [f for f in self.get_fillers()
        #          if self.root in f]

        roots = self.get_fillers_type(self.root,
                                      ignore_pos_f=self.opts['use_pos_f'],
                                      ignore_bracket=True, ignore_copy=False)

        if self.opts['use_pos_f']:
            roots = [root for root in roots
                     if root.split(self.opts['sep'])[1] == self.opts['pos_f'][0]]

        return roots

    def find_roots(self):
        '''Return (list of int) of indices of root symbols.'''

        return self.find_fillers(self.get_roots())

    def get_sisters(self, fname):
        '''Returns (dict) of potential sisters of fname (str) with
        possible sister positions (see opts['pos_s']).'''

        # {'l0': list of indices, 'r0': list of indices }
        pos_s = self.opts['pos_s']
        res = {}
        for pos in pos_s:
            res[pos] = []

        for rule in self.rules:
            if (rule['d1'] is not None) and (rule['d2'] is not None):
                if rule['d1'] == fname:
                    res[pos_s[1]].append(rule['d2'])
                if rule['d2'] == fname:
                    res[pos_s[0]].append(rule['d1'])

        res[pos_s[0]] = unique(res[pos_s[0]])
        res[pos_s[1]] = unique(res[pos_s[1]])
        return res

    def find_sisters(self, fname):
        '''Returns (dict) of indices of potential sisters of fname (str)
        with possible sister positions.

        See also get_sisters()'''
        # {'l0': list of indices, 'r0': list of indices }

        res = {}
        for key, val in self.get_sisters(fname).items():
            res[key] = self.find_fillers(val)
        return res

    def has_nonterminal_sister(self, fname, pos_s):
        '''Returns (bool) on whether fname (str) can have a non-terminal
        sister in position pos_s (str) (see opts['pos_s']).'''

        res = False
        idx_list = self.find_sisters(fname)[pos_s]
        for fi in idx_list:
            if not self.is_terminal(self.filler_names[fi]):
                res = True
                break
        return res

    def aggregate_prob(self, fname):
        p = 0.
        for rule in self.rules:
            if rule['m'] == fname:
                p += rule['p']
        return p

    def generate_sentence(self, min_sent_len=1, max_sent_len=20, use_type=True):

        def sample0(fillers):
            prob = []
            for fname in fillers:
                prob.append(self.aggregate_prob(fname))
            prob = np.array(prob)
            prob /= prob.sum()
            fi = np.random.choice(len(fillers), size=1, p=prob)
            return fillers[fi[0]], prob[fi[0]]

        def expand(node):
            terminals = []
            rules = self.get_rules(subset={'m': node.sym})
            p = 1.
            if len(rules) > 0:

                prob = np.array([rule['p'] for rule in rules])
                prob /= prob.sum()
                rulei = np.random.choice(len(prob), size=1, p=prob)
                # print(rules, prob, rulei)
                # print(rules[rulei])
                rule = rules[rulei[0]]
                p *= prob[rulei[0]]

                d1sym = rule['d1']
                d2sym = rule['d2']

                d1 = Node(d1sym)
                node.add_child(d1)
                if self.is_terminal(d1sym):
                    terminals.append(d1.sym)
                else:
                    str1, d1, p1 = expand(d1)
                    terminals += str1
                    p *= p1

                d2 = Node(d2sym)
                node.add_child(d2)
                if self.is_terminal(d2sym):
                    terminals.append(d2.sym)
                else:
                    str2, d2, p2 = expand(d2)
                    terminals += str2
                    p *= p2

            return terminals, node, p

        short_or_long = True
        while short_or_long:
            symbol, p_root = sample0(self.get_roots())
            root = Node(symbol)
            terminals, parse, p = expand(root)
            if (len(terminals) >= min_sent_len) and (len(terminals) <= max_sent_len):
                short_or_long = False

        if use_type:
            terminals = self.get_types(
                terminals, ignore_pos_f=True,
                ignore_bracket=True, ignore_copy=True)

        return terminals, parse, p_root * p


class BrickRole(object):

    def __init__(self, max_sent_len, use_hnf=False):

        self._set_opts(max_sent_len=max_sent_len, use_hnf=use_hnf)
        self._create_role_names()
        # self._create_fast_lookups()
        self._create_fastER_lookups()

    def _set_opts(self, max_sent_len, use_hnf):

        self.opts = {}
        self.opts['max_sent_len'] = max_sent_len
        self.opts['use_hnf'] = use_hnf

        self.opts['pos_m'] = ['l', 'r', 'm', 'l0', 'r0']
        self.opts['pos_d'] = ['l', 'r', 'm', 'l0', 'r0']
        self.opts['pos_s'] = ['l', 'r']

        if not self.opts['use_hnf']:
            self.opts['pos_m'] = self.opts['pos_m'][0:2]
            self.opts['pos_d'] = self.opts['pos_d'][0:2]

    def _create_role_names(self):
        '''Create a list of role names (lv, pos) where
        lv is the level and pos is the position at the level.
        For a bracketed role, -1 is multiplied to lv.
        '''

        max_sent_len = self.opts['max_sent_len']
        use_hnf = self.opts['use_hnf']

        rnames_all = []
        row_idx = 0

        for lv in range(1, max_sent_len + 1):

            if use_hnf and lv > 1:
                for pos in range(1, max_sent_len - lv + 2):
                    rname = '({:d},{:d})'.format(-lv, pos)
                    rnames_all.append(rname)
                    row_idx += 1

            for pos in range(1, max_sent_len - lv + 2):
                rname = '({:d},{:d})'.format(lv, pos)
                rnames_all.append(rname)
                row_idx += 1

        self.role_names = rnames_all

    def _create_fastER_lookups(self):
        '''Pre-compute lookup arrays for O(1) access during training.

        OPTIMIZED VERSION:
        - Single pass for role_tuples + name_to_idx + property arrays
        - Efficient daughter/mother relationship building
        - Complexity: O(N) for properties + O(N×D) for relationships
        '''
        import time
        t0 = time.time()

        num_roles = len(self.role_names)

        # ============ PHASE 1: Single pass for all per-role properties ============

        # Pre-allocate arrays
        self.role_tuples = np.empty((num_roles, 2), dtype=np.int32)
        self.role_is_terminal = np.empty(num_roles, dtype=bool)
        self.role_is_bracketed = np.empty(num_roles, dtype=bool)
        self.role_name_to_idx = {}

        # Single loop: build everything that depends only on role name
        for ri, rname in enumerate(self.role_names):
            # Index mapping
            self.role_name_to_idx[rname] = ri

            # Parse tuple (do this ONCE per role)
            lv, pos = self.str2tuple(rname)
            self.role_tuples[ri, 0] = lv
            self.role_tuples[ri, 1] = pos

            # Derive properties directly from tuple (no additional method calls)
            self.role_is_terminal[ri] = (lv == 1)
            self.role_is_bracketed[ri] = (lv < 0)

        # ============ PHASE 2: Daughter and mother relationships ============

        # Initialize structures
        self.role_daughters_idx = {
            pos_type: [[] for _ in range(num_roles)]
            for pos_type in self.opts['pos_d']
        }

        self.role_mothers_idx = {
            pos_type: [[] for _ in range(num_roles)]
            for pos_type in self.opts['pos_m']
        }

        # Build relationships (can't avoid this loop if get_daughters/mothers is necessary)
        for ri, rname in enumerate(self.role_names):
            # Get daughters
            daughters = self.get_daughters(rname)
            for pos_type, daughter_names in daughters.items():
                # Batch convert names to indices using pre-built mapping
                self.role_daughters_idx[pos_type][ri] = [
                    self.role_name_to_idx[dname]
                    for dname in daughter_names
                    if dname in self.role_name_to_idx
                ]

            # Get mothers
            mothers = self.get_mothers(rname)
            for pos_type, mother_names in mothers.items():
                # Batch convert names to indices
                self.role_mothers_idx[pos_type][ri] = [
                    self.role_name_to_idx[mname]
                    for mname in mother_names
                    if mname in self.role_name_to_idx
                ]

        # ============ PHASE 3: Simplified arrays for common access patterns ============

        self.role_daughter_l_idx = np.full(num_roles, -1, dtype=np.int32)
        self.role_daughter_r_idx = np.full(num_roles, -1, dtype=np.int32)

        # Vectorize this extraction (faster than individual assignments)
        for ri in range(num_roles):
            l_daughters = self.role_daughters_idx['l'][ri]
            r_daughters = self.role_daughters_idx['r'][ri]

            if l_daughters:
                self.role_daughter_l_idx[ri] = l_daughters[0]
            if r_daughters:
                self.role_daughter_r_idx[ri] = r_daughters[0]

        elapsed = time.time() - t0
        print(
            f"    BrickRole fast lookups built in {elapsed:.3f}s ({num_roles} roles)")

    def _create_fast_lookups(self):
        '''Pre-compute lookup arrays for O(1) access during training.

        This method creates NumPy arrays that cache expensive string parsing
        and dictionary lookup operations. Called once during __init__, provides
        major speedup for training loops that iterate over role_names.
        '''
        # Pre-compute str2tuple for all roles (avoids string parsing)
        self.role_tuples = np.array([
            self.str2tuple(rname) for rname in self.role_names
        ], dtype=np.int32)  # Shape: [num_roles, 2]

        # Pre-compute terminal status (level == 1)
        self.role_is_terminal = (self.role_tuples[:, 0] == 1)

        # Pre-compute bracketed status (level < 0, used in HNF)
        self.role_is_bracketed = (self.role_tuples[:, 0] < 0)

        # Create name → index mapping for fast lookups
        self.role_name_to_idx = {
            name: i for i, name in enumerate(self.role_names)
        }

        # Pre-compute daughter relationships (indices not names)
        self.role_daughters_idx = {}
        for pos_type in self.opts['pos_d']:
            self.role_daughters_idx[pos_type] = [[]
                                                 for _ in range(len(self.role_names))]

        for ri, rname in enumerate(self.role_names):
            daughters = self.get_daughters(rname)
            for pos_type, daughter_names in daughters.items():
                for daughter_name in daughter_names:
                    if daughter_name in self.role_name_to_idx:
                        daughter_idx = self.role_name_to_idx[daughter_name]
                        self.role_daughters_idx[pos_type][ri].append(
                            daughter_idx)

        # Simplified arrays for common case (first 'l' and 'r' daughters)
        self.role_daughter_l_idx = np.full(
            len(self.role_names), -1, dtype=np.int32)
        self.role_daughter_r_idx = np.full(
            len(self.role_names), -1, dtype=np.int32)

        for ri in range(len(self.role_names)):
            if len(self.role_daughters_idx['l'][ri]) > 0:
                self.role_daughter_l_idx[ri] = self.role_daughters_idx['l'][ri][0]
            if len(self.role_daughters_idx['r'][ri]) > 0:
                self.role_daughter_r_idx[ri] = self.role_daughters_idx['r'][ri][0]

        # Pre-compute mother relationships
        self.role_mothers_idx = {}
        for pos_type in self.opts['pos_m']:
            self.role_mothers_idx[pos_type] = [[]
                                               for _ in range(len(self.role_names))]

        for ri, rname in enumerate(self.role_names):
            mothers = self.get_mothers(rname)
            for pos_type, mother_names in mothers.items():
                for mother_name in mother_names:
                    if mother_name in self.role_name_to_idx:
                        mother_idx = self.role_name_to_idx[mother_name]
                        self.role_mothers_idx[pos_type][ri].append(mother_idx)

    def str2tuple(self, rname_str):
        return tuple([int(r) for r in rname_str[1:-1].split(',')])

    def tuple2str(self, rname_tuple):
        return '({:d},{:d})'.format(rname_tuple[0], rname_tuple[1])

    def find_roles(self, rnames):

        if not isinstance(rnames, list):
            rnames = [rnames]
        return [idx for idx, rname in enumerate(self.role_names)
                if rname in rnames]

    def get_roles(self, idx=None):

        if idx is None:
            return self.role_names
        else:
            if not isinstance(idx, list):
                idx = [idx]
            return [self.role_names[ii] for ii in idx]

    def is_bracketed(self, rname_str):

        rname_tuple = self.str2tuple(rname_str=rname_str)
        return self.opts['use_hnf'] and (rname_tuple[0] < 0)

    def get_bracketed(self):
        return [rname for rname in self.role_names if self.is_bracketed(rname)]

    def find_bracketed(self):
        return self.find_roles(self.get_bracketed())

    def is_terminal(self, rname_str):
        rname_tuple = self.str2tuple(rname_str=rname_str)
        return rname_tuple[0] == 1

    def get_terminals(self):
        return [rname for rname in self.role_names if self.is_terminal(rname)]

    def find_terminals(self):
        return self.find_roles(self.get_terminals())

    def get_mothers(self, rname_str):
        # Each brick role has one role as mother for each mother type.

        use_hnf = self.opts['use_hnf']
        max_sent_len = self.opts['max_sent_len']
        pos_m = self.opts['pos_m']

        res = {}
        for key in pos_m:
            res[key] = []

        lv, pos = self.str2tuple(rname_str)
        if lv > 0 and lv < max_sent_len:
            # non-bracketed role
            # if pos > 1:
            if (lv + 1 + pos - 2 >= lv + 1) and (lv + 1 + pos - 2 <= max_sent_len):
                # have a mother on the left
                if use_hnf:
                    # left bracketed (l)
                    res[pos_m[0]].append(
                        self.tuple2str((-(lv + 1), pos - 1)))
                    # left copy (l0)
                    res[pos_m[3]].append(
                        self.tuple2str((lv + 1, pos - 1)))
                else:
                    res[pos_m[0]].append(
                        self.tuple2str((lv + 1, pos - 1)))

            # if pos < max_sent_len:
            if (lv + 1 + pos - 1 >= lv + 1) and (lv + 1 + pos - 1 <= max_sent_len):
                # have a mother on the right
                if use_hnf:
                    # right copy (r0)
                    res[pos_m[4]].append(
                        self.tuple2str((lv + 1, pos)))
                    # right bracketed (r)
                    res[pos_m[1]].append(
                        self.tuple2str((-(lv + 1), pos)))
                else:
                    res[pos_m[1]].append(
                        self.tuple2str((lv + 1, pos)))

        if use_hnf and lv < 0:
            # bracketed role
            res[pos_m[2]].append(self.tuple2str((-lv, pos)))

        return res

    def find_mothers(self, rname_str):

        res = {}
        for key, val in self.get_mothers(rname_str).items():
            res[key] = self.find_roles(val)

        return res

    def is_mother(self, rname_mother_str, rname_daughter_str):

        mothers = []
        for key, val in self.get_mothers(rname_daughter_str).items():
            mothers += val
        return rname_mother_str in mothers

    def get_daughters(self, rname_str):
        # Each brick role has one role as mother for each mother type.

        use_hnf = self.opts['use_hnf']
        max_sent_len = self.opts['max_sent_len']
        pos_d = self.opts['pos_d']

        res = {}
        for key in pos_d:
            res[key] = []

        lv, pos = self.str2tuple(rname_str)
        if pos >= 1 and pos <= max_sent_len - abs(lv) + 1:
            if use_hnf:
                if lv > 1:
                    res[pos_d[2]].append(self.tuple2str((-lv, pos)))
                    res[pos_d[3]].append(self.tuple2str((lv - 1, pos)))
                    res[pos_d[4]].append(self.tuple2str((lv - 1, pos + 1)))
                if lv < 0:
                    res[pos_d[0]].append(self.tuple2str((-lv - 1, pos)))
                    res[pos_d[1]].append(self.tuple2str((-lv - 1, pos + 1)))
            else:
                if lv > 1:
                    res[pos_d[0]].append(self.tuple2str((lv - 1, pos)))
                    res[pos_d[1]].append(self.tuple2str((lv - 1, pos + 1)))

        return res

    def find_daughters(self, rname_str):

        res = {}
        for key, val in self.get_daughters(rname_str).items():
            res[key] = self.find_roles(val)

        return res

    def is_daughter(self, rname_daughter_str, rname_mother_str):

        daughters = []
        for key, val in self.get_daughters(rname_mother_str).items():
            daughters += val
        return rname_daughter_str in daughters


class HarmonicGrammar():

    def __init__(self, pcfg, root, max_sent_len, opts=None):
        self._set_opts(root=root, max_sent_len=max_sent_len)
        self._update_opts(opts)
        self.pcfg_str = pcfg
        self.g0 = PCFG(pcfg=pcfg, root=root, opts=self.opts)  # original rule
        self.g = copy.deepcopy(self.g0)
        self._create_roles()
        self._add_names()
        self.rules = []
        self._add_additional_rules()
        self._add_binary_rules()
        self._add_copy_rules()
        self._add_unary_rules()
        self._add_expansion_rules()
        print("Optimizing HarmonicGrammar with fast lookups...")
        # Note: g0's fast lookups already created in PCFG.__init__
        # self.g is a deepcopy, so copy the lookups
        if hasattr(self.g0, 'filler_name_to_idx'):
            self.g.filler_name_to_idx = self.g0.filler_name_to_idx.copy()
            self.g.filler_is_terminal = self.g0.filler_is_terminal.copy()
            self.g.filler_is_copy = self.g0.filler_is_copy.copy()
            self.g.filler_is_bracketed = self.g0.filler_is_bracketed.copy()
            self.g.filler_is_root = self.g0.filler_is_root.copy()
        # Roles fast lookups already created in BrickRole.__init__
        print("Optimization complete!")

    def get_simlist(self, dp=0.5):

        sim = []
        fnames_types = []
        fnames_tokens = []
        for fname in self.g.filler_names:
            ftype = self.g.get_types(
                fname, ignore_copy=False, ignore_pos_f=False)[0]
            if ftype not in fnames_types:
                fnames_types.append(ftype)
                fnames_tokens.append([fname])
            else:
                idx = fnames_types.index(ftype)
                fnames_tokens[idx].append(fname)

        for f_tokens in fnames_tokens:
            n_tokens = len(f_tokens)
            if n_tokens > 1:
                for t1 in range(n_tokens - 1):
                    for t2 in range(t1 + 1, n_tokens):
                        sim.append([[f_tokens[t1], f_tokens[t2]], dp])

        return sim

    def replace_symbols(self, sym_old, sym_new):
        '''Replace sym_old with sym_new'''
        for rule in self.g0.rules:
            if rule['m'] == sym_old:
                rule['m'] = sym_new
            if rule['d1'] == sym_old:
                rule['d1'] = sym_new
            if rule['d2'] == sym_old:
                rule['d2'] = sym_new
        temp = []
        for sym in self.g0.filler_names:
            if sym != sym_old:
                temp.append(sym)
        self.g0.filler_names = temp

        # Update HG
        self.g = copy.deepcopy(self.g0)
        self._create_roles()
        self._add_names()

        self.rules = []
        self._add_additional_rules()
        self._add_binary_rules()
        self._add_copy_rules()
        # self._add_competition_rules()
        # self._add_null_rules()
        self._add_unary_rules()
        self._add_expansion_rules()

    def _set_opts(self, root, max_sent_len):

        self.opts = {}

        self.opts['use_same_len'] = True
        self.opts['f_root'] = '#'
        self.opts['f_empty'] = '@'
        self.opts['f_empty_copy'] = '@'

        self.opts['pos_m'] = ['l', 'r', 'm', 'l0', 'r0']
        self.opts['pos_d'] = ['l', 'r', 'm', 'l0', 'r0']
        self.opts['pos_s'] = ['l', 'r']
        self.opts['pos_f'] = ['0', '1', '9']

        # === opts for class Grammar ==========
        self.opts['use_hnf'] = False
        self.opts['use_pos_f'] = True
        self.opts['pos_f'] = ['0', '1', '9']
        self.opts['root'] = root
        self.opts['null'] = '_'
        self.opts['add_null'] = True
        self.opts['sep'] = ':'
        self.opts['copy'] = '*'
        self.opts['pos_copy'] = 'l'
        self.opts['add_copy_rules'] = True
        self.opts['use_minimal_copy_rules'] = True

        # === opts for HarmonicGrammar ========
        self.opts['max_sent_len'] = max_sent_len
        self.opts['bsep'] = '/'
        self.opts['role_system'] = 'brick_role'
        self.opts['add_null_rules'] = False
        self.opts['add_competition_rules'] = False
        self.opts['competition_rule_type'] = 'btw_mothers'
        # btw_tokens, or btw_mothers

        self.opts['unary_base'] = 'filler'
        # harmony values for unary rules
        self.opts['H_null'] = 0.
        # self.opts['H_root'] = -2.
        self.opts['H_root_illegitimate'] = -5.
        self.opts['H_terminal'] = -1.
        self.opts['H_terminal_illegitimate'] = -5.
        self.opts['H_nonterminal_illegitimate'] = -5.
        self.opts['H_copy_illegitimate'] = -5.
        # harmony values for binary rules
        self.opts['H_binary'] = 2.0
        self.opts['H_copy'] = 2.0
        self.opts['H_unary_2'] = -2.0
        self.opts['H_unary_3'] = -3.0
        self.opts['H_competition'] = -1.0
        self.opts['H_null_0'] = 0.5
        self.opts['H_null_1'] = 0.5
        self.opts['H_null_2'] = 0.

        # self.opts['add1_to_root'] = False
        # Consider the following combination
        self.opts['add1_to_root'] = True
        self.opts['H_root'] = -3.

    def _update_opts(self, opts):

        if opts is not None:
            for key, val in opts.items():
                if key in self.opts.keys():
                    self.opts[key] = val

        if self.opts['use_pos_f']:
            # CHECK
            self.opts['f_root'] += self.opts['sep'] + self.opts['pos_f'][0]
            self.opts['f_empty'] += self.opts['sep'] + self.opts['pos_f'][1]
            self.opts['f_empty_copy'] += self.opts['sep'] + \
                self.opts['pos_f'][1]

        if self.opts['add_copy_rules']:
            if self.opts['pos_copy'] == 'l':
                self.opts['f_empty_copy'] = self.opts['copy'] + \
                    self.opts['f_empty_copy']
            elif self.opts['pos_copy'] == 'r':
                self.opts['f_empty_copy'] = self.opts['f_empty_copy'] + \
                    self.opts['copy']

    def _create_roles(self):
        if self.opts['role_system'] == 'brick_role':
            self.roles = BrickRole(max_sent_len=self.opts['max_sent_len'],
                                   use_hnf=self.opts['use_hnf'])
        # elif self.opts['role_system'] == 'span_role':
        #     self.roles = SpanRole(max_sent_len=self.opts['max_sent_len'],
        #                           use_hnf=self.opts['use_hnf'])
        # elif self.opts['role_system'] == 'recursive_role':
        #     self.roles = RecursiveRole(max_sent_len=self.opts['max_sent_len'],
        #                                use_hnf=self.opts['use_hnf'])
        else:
            sys.exit('You chose role_system that is not supported.')

    def _add_names(self):
        self.filler_names = self.g.filler_names
        self.role_names = self.roles.role_names
        self.update_binding_names()
        self.num_fillers = len(self.filler_names)
        self.num_roles = len(self.role_names)
        self.num_bindings = len(self.binding_names)

    def has_rule(self, rule):
        # Check whether rule is in self.rules
        return rule in self.rules

    def _add_additional_rules(self):

        # {'m': fname_m, 'd1': fname_d1, 'd2': fname_d2 }
        if self.opts['add_copy_rules']:

            rules_new = []
            rules_copy = []

            for rule in self.g.rules:

                m = rule['m']
                d1 = rule['d1']
                d2 = rule['d2']
                p = rule['p']

                if not self.g.is_hnf_unary_rule(rule):
                    # Now rule is a binary branching rule. Note that at this
                    # point, all unary rules are hnf unary branching rules.

                    # If a daughter has a non-terminal sister,
                    # replace the daughter with its copy version.
                    if not self.g.is_terminal(d1) and not self.g.is_terminal(d2):
                        d1_copy = self.get_copy(d1)
                        d2_copy = self.get_copy(d2)

                        # NOT POSSIBLE ================================
                        # if not self.opts['use_minimal_copy_rules']:
                        #     rules_new.append(rule)
                        # rule1 = {'m': m, 'd1': d1_copy, 'd2': d2}
                        # rules_new.append(rule1)
                        # rule2 = {'m': m, 'd1': d1, 'd2': d2_copy}
                        # rules_new.append(rule2)
                        # =============================================
                        rule3 = {'m': m, 'd1': d1_copy, 'd2': d2_copy, 'p': p}
                        rules_new.append(rule3)

                        copy_rule1 = {'m': d1_copy,
                                      'd1': d1, 'd2': None, 'p': None}
                        copy_rule2 = {'m': d1_copy,
                                      'd1': d1_copy, 'd2': None, 'p': None}
                        copy_rule3 = {'m': d2_copy,
                                      'd1': None, 'd2': d2, 'p': None}
                        copy_rule4 = {'m': d2_copy, 'd1': None,
                                      'd2': d2_copy, 'p': None}
                        if copy_rule1 not in rules_copy:
                            rules_copy.append(copy_rule1)
                        if copy_rule2 not in rules_copy:
                            rules_copy.append(copy_rule2)
                        if copy_rule3 not in rules_copy:
                            rules_copy.append(copy_rule3)
                        if copy_rule4 not in rules_copy:
                            rules_copy.append(copy_rule4)

                    elif self.g.is_terminal(d1) and not self.g.is_terminal(d2):
                        d1_copy = self.get_copy(d1)
                        rule['d1'] = d1_copy
                        rules_new.append(rule)

                        copy_rule1 = {'m': d1_copy,
                                      'd1': d1, 'd2': None, 'p': None}
                        copy_rule2 = {'m': d1_copy,
                                      'd1': d1_copy, 'd2': None, 'p': None}
                        if copy_rule1 not in rules_copy:
                            rules_copy.append(copy_rule1)
                        if copy_rule2 not in rules_copy:
                            rules_copy.append(copy_rule2)

                    elif not self.g.is_terminal(d1) and self.g.is_terminal(d2):
                        d2_copy = self.get_copy(d2)
                        rule['d2'] = d2_copy
                        rules_new.append(rule)

                        copy_rule1 = {'m': d2_copy,
                                      'd1': None, 'd2': d2, 'p': None}
                        copy_rule2 = {'m': d2_copy, 'd1': None,
                                      'd2': d2_copy, 'p': None}
                        if copy_rule1 not in rules_copy:
                            rules_copy.append(copy_rule1)
                        if copy_rule2 not in rules_copy:
                            rules_copy.append(copy_rule2)
                    else:
                        rules_new.append(rule)

                else:
                    rules_new.append(rule)

            self.g.rules = rules_new + rules_copy
            self.g._sort_rules()
            self.g._add_names()
            self._add_names()
            # self.filler_names = self.g.filler_names
            # self.update_binding_names()

        if self.opts['use_same_len']:
            # ADD binary rules
            roots = self.g.get_roots()

            for root in roots:
                rule = {
                    'm': self.opts['f_root'],
                    'd1': root, 'd2': self.opts['f_empty_copy'], 'p': None}
                if rule not in self.g.rules:
                    self.g.rules.append(rule)

            rule = {'m': self.opts['f_empty_copy'],
                    'd1': None, 'd2': self.opts['f_empty'], 'p': None}
            if rule not in self.g.rules:
                self.g.rules.append(rule)

            rule = {'m': self.opts['f_empty_copy'],
                    'd1': None, 'd2': self.opts['f_empty_copy'], 'p': None}
            if rule not in self.g.rules:
                self.g.rules.append(rule)

            self.g._sort_rules()
            self.g._add_names()
            self._add_names()
            # self.filler_names = self.g.filler_names
            # self.update_binding_names()

    def _add_expansion_rules(self):

        if self.opts['use_same_len']:
            # ADD binary rules
            roots = self.g.get_roots()
            for root in roots:
                rule = {'f1': root, 'f2': self.opts['f_root'], 'rel': 'r',
                        'H': 2.0, 'rule': 'expansion_binary', 'br': False}
                if rule not in self.rules:
                    self.rules.append(rule)

            rule = {'f1': self.opts['f_empty'], 'f2': self.opts['f_empty_copy'],
                    'rel': 'l', 'H': 2.0, 'rule': 'expansion_binary', 'br': False}
            if rule not in self.rules:
                self.rules.append(rule)

            rule = {'f1': self.opts['f_empty_copy'], 'f2': self.opts['f_empty_copy'],
                    'rel': 'l', 'H': 2.0, 'rule': 'expansion_binary', 'br': False}
            if rule not in self.rules:
                self.rules.append(rule)

            rule = {'f1': self.opts['f_empty_copy'], 'f2': self.opts['f_root'],
                    'rel': 'l', 'H': 2.0, 'rule': 'expansion_binary', 'br': False}
            if rule not in self.rules:
                self.rules.append(rule)

            # rule = {'f1': self.opts['f_empty'], 'f2': self.opts['f_root'],
            #         'rel': 'l', 'H': 2.0, 'rule': 'expansion_binary', 'br': False}
            # if rule not in self.rules:
            #     self.rules.append(rule)

            # ADD unary rules
            rule = {'f1': self.opts['f_root'], 'f2': None, 'rel': '0',
                    'H': -2., 'rule': 'expansion_unary', 'br': False}
            if rule not in self.rules:
                self.rules.append(rule)

            rule = {'f1': self.opts['f_empty'], 'f2': None, 'rel': '0',
                    'H': -1., 'rule': 'expansion_unary', 'br': False}
            if rule not in self.rules:
                self.rules.append(rule)

            rule = {'f1': self.opts['f_empty_copy'], 'f2': None, 'rel': '0',
                    'H': -2., 'rule': 'expansion_unary', 'br': False}
            if rule not in self.rules:
                self.rules.append(rule)

            # for fname in [self.opts['f_root'], self.opts['f_empty'], self.opts['f_empty_copy']]:
            #     if fname not in self.filler_names:
            #         self.filler_names.append(fname)

    def _add_binary_rules(self):

        # {'f1': fname1, 'f2': fname2, 'rel': relation, 'H', harmony,
        # 'rule': rule_type, 'br': is_bracketed_filler(fname1)}
        # rel:
        #   'l': mother on the left
        #   'r': mother on the right
        #   'm': mother on the right above
        #   'l0': mother on the left (direct copy in HNF)
        #   'r0': mother on the right (direct copy in HNF)
        #   '0': no mother in unary HG rules
        # rule_type: 'binary', 'unary', 'copy', 'competition'
        # role_order = l, r, l0, r0 (easy to copy)

        # OLD
        # [ {'d': fname_d, 'm': fname_m, 'rel_type': rel_type, 'H': harmony} ]
        # rel_type: 'l', 'r', 'm', 'l0', 'r0'

        val = self.opts['H_binary']
        use_hnf = self.opts['use_hnf']
        for rule in self.g.rules:
            if not self.g.is_copy_rule(rule):
                if use_hnf and (rule['d2'] is None):
                    # unary branching in HNF
                    new_rule = {'f1': rule['d1'], 'f2': rule['m'],
                                'rel': 'm', 'H': val, 'rule': 'binary',
                                'br': self.g.is_bracketed(rule['d1'])}
                    if not self.has_rule(new_rule):
                        self.rules.append(new_rule)
                else:
                    new_rule1 = {'f1': rule['d1'], 'f2': rule['m'],
                                 'rel': 'r', 'H': val, 'rule': 'binary',
                                 'br': self.g.is_bracketed(rule['d1'])}
                    new_rule2 = {'f1': rule['d2'], 'f2': rule['m'],
                                 'rel': 'l', 'H': val, 'rule': 'binary',
                                 'br': self.g.is_bracketed(rule['d1'])}
                    if not self.has_rule(new_rule1):
                        self.rules.append(new_rule1)
                    if not self.has_rule(new_rule2):
                        self.rules.append(new_rule2)

    def _add_copy_rules(self):

        if self.g.opts['add_copy_rules']:

            val = self.opts['H_copy']
            if self.opts['use_hnf']:
                rel_names = self.opts['pos_m'][3:5]  # 'l0', 'r0'
            else:
                rel_names = self.opts['pos_m'][0:2]  # 'l', 'r'

            # copy_rules = self.subset_copy_rules()

            copy_rules = [
                rule for rule in self.g.rules
                if self.is_copy(rule['m'])]

            for copy_rule in copy_rules:
                if copy_rule['d1'] is None:
                    rule = {'f1': copy_rule['d2'], 'f2': copy_rule['m'],
                            'rel': rel_names[0], 'H': val,
                            'rule': 'copy',
                            'br': self.g.is_bracketed(copy_rule['d2'])}
                    if not self.has_rule(rule):
                        self.rules.append(rule)
                if copy_rule['d2'] is None:
                    rule = {'f1': copy_rule['d1'], 'f2': copy_rule['m'],
                            'rel': rel_names[1], 'H': val,
                            'rule': 'copy',
                            'br': self.g.is_bracketed(copy_rule['d1'])}
                    if not self.has_rule(rule):
                        self.rules.append(rule)

    def _add_unary_rules(self):

        root_bias = self.opts['H_root']
        null_bias = self.opts['H_null']
        terminal_bias = self.opts['H_terminal']
        unary_base = self.opts['unary_base']
        use_hnf = self.opts['use_hnf']

        h_unary_2 = self.opts['H_unary_2']
        h_unary_3 = self.opts['H_unary_3']

        val = self.opts['H_binary'] * 0.5

        if unary_base == 'filler':
            for filler in self.g.filler_names:

                # Assign -3 to non-terminal fillers
                rule = {'f1': filler, 'f2': None,
                        'rel': '0', 'H': h_unary_3 * val, 'rule': 'unary',
                        'br': self.g.is_bracketed(filler)}

                if use_hnf and (not self.g.is_bracketed(filler)):
                    # Update H if filler is unbracketed in HNF
                    rule = {'f1': filler, 'f2': None,
                            'rel': '0', 'H': h_unary_2 * val, 'rule': 'unary',
                            'br': self.g.is_bracketed(filler)}

                if filler in self.g.get_roots():
                    rule = {'f1': filler, 'f2': None,
                            'rel': '0', 'H': root_bias * val, 'rule': 'unary',
                            'br': self.g.is_bracketed(filler)}

                if filler == self.g.opts['null']:
                    rule = {'f1': filler, 'f2': None,
                            'rel': '0', 'H': null_bias, 'rule': 'unary',
                            'br': self.g.is_bracketed(filler)}

                # if ('c' not in filler) and self.g.is_terminal(filler):
                # PWC-20190731: I don't know why but self.opts['copy'] was
                #             : unintendedly replaced with 'c', which is wrong.
                if (self.opts['copy'] not in filler) and self.g.is_terminal(filler):
                    rule = {'f1': filler, 'f2': None,
                            'rel': '0', 'H': terminal_bias * val, 'rule': 'unary',
                            'br': self.g.is_bracketed(filler)}

                if self.opts['copy'] in filler:
                    # copy symbols
                    rule = {'f1': filler, 'f2': None,
                            'rel': '0', 'H': h_unary_2 * val, 'rule': 'unary',
                            'br': self.g.is_bracketed(filler)}

                if not self.has_rule(rule):
                    self.rules.append(rule)

    def get_roots(self):

        if self.opts['use_same_len']:
            return self.g.get_roots() + [self.opts['f_root']]
        else:
            return self.g.get_roots()

    def get_terminals(self):

        return self.g.get_terminals()

    def get_copy(self, fname):
        '''Returns a copy version (str) of fname (str).

        Regardless of whether fname is used in a given grammar,
        it will create its copy version. An exception is when
        fname itself is a copy version of another symbol.
        In this case, this method returns None.
        '''

        if self.opts['copy'] not in fname:
            if self.opts['sep'] in fname:
                fname, role = fname.split(self.opts['sep'])
                if self.opts['pos_copy'] == 'l':
                    fname = self.opts['copy'] + fname
                elif self.opts['pos_copy'] == 'r':
                    fname = fname + self.opts['copy']
                fname = fname + self.opts['sep'] + role
            else:
                if self.opts['pos_copy'] == 'l':
                    fname = self.opts['copy'] + fname
                elif self.opts['pos_copy'] == 'r':
                    fname = fname + self.opts['copy']
            return fname

        else:
            return None

    def get_uncopy(self, fname):
        '''Returns an original version (str) of fname (str).

        It does not guarantee that the uncopied version of
        fname is used in a given grammar. If fname itself
        is not a copy version of another symbol, the method
        returns None.
        '''

        if self.opts['copy'] in fname:
            if self.opts['sep'] in fname:
                fname, rname = fname.split(self.opts['sep'])
                if self.opts['pos_copy'] == 'l':
                    fname = fname.split(self.opts['copy'])[1]
                elif self.opts['pos_copy'] == 'r':
                    fname = fname.split(self.opts['copy'])[0]
                fname = fname + self.opts['sep'] + rname

            else:
                if self.opts['pos_copy'] == 'l':
                    fname = fname.split(self.opts['copy'])[1]
                elif self.opts['pos_copy'] == 'r':
                    fname = fname.split(self.opts['copy'])[0]

            return fname
        else:
            return None

    def is_copy(self, fname1, fname2=''):
        '''Returns (bool) after checking fname1 (str) is a copy version
        of fname2 (str). If fname2 is not given, it will test if
        fname1 (str) is a copy version of any other symbol.'''

        if fname2 == '' and fname1 is not None:
            return self.opts['copy'] in fname1
        else:
            return ((fname1 is not None) and (fname2 is not None)) and\
                   (fname1 == self.get_copy(fname2))

    # def is_copy_rule(self, rule):
    #     '''Returns (bool) after checking whether rule (dict) is a copy rule.'''

    #     return self.is_copy(rule['m'])

    # def subset_copy_rules(self):
    #     '''Returns (list) of copy rules (dict)'''

    #     return [rule for rule in self.rules if self.is_copy_rule(rule)]

    def read_rules(self, rule_types=None):

        # max_flen = max([len(f) for f in self.g.get_fillers()])

        sep = self.g.opts['sep']
        bsep = self.opts['bsep']
        use_pos_f = self.g.opts['use_pos_f']

        if rule_types is None:
            rule_types = ['binary', 'unary', 'copy',
                          'expansion_binary', 'expansion_unary', 'competition']
        elif not isinstance(rule_types, list):
            rule_types = [rule_types]

        def maxlen(sym_list):
            if len(sym_list) > 0:
                maxlen = max([len(sym) for sym in sym_list])
            else:
                maxlen = 0
            return maxlen

        def pretty_print(rules, is_competition=False):

            f1_list = []
            f2_list = []
            r1_list = []
            r2_list = []
            rel_list = []

            for rule in rules:

                f1 = rule['f1']
                f2 = rule['f2']
                if f2 is None:
                    f2 = []
                rel = rule['rel']

                if use_pos_f:
                    if sep in f1:
                        f1, r1 = rule['f1'].split(sep)
                        r1_list.append(r1)
                    if sep in f2:
                        f2, r2 = rule['f2'].split(sep)
                        r2_list.append(r2)

                f1_list.append(f1)
                f2_list.append(f2)
                if is_competition:
                    rel_list += rel.split('/')
                else:
                    rel_list.append(rel)

            maxlen_f1 = maxlen(f1_list)
            maxlen_f2 = maxlen(f2_list)
            maxlen_r1 = maxlen(r1_list)
            maxlen_r2 = maxlen(r2_list)
            maxlen_rel = maxlen(rel_list)

            for rule in rules:

                f1 = rule['f1']
                f2 = rule['f2']
                r1 = ''
                r2 = ''
                rel = rule['rel']
                val = rule['H']

                if use_pos_f:
                    if sep in rule['f1']:
                        f1, r1 = f1.split(sep)
                    if (f2 is not None) and (sep in rule['f2']):
                        f2, r2 = rule['f2'].split(sep)

                # binary rules
                str1 = '{:>{:d}}'.format(f1, maxlen_f1)
                if len(r1) > 0:
                    str1 += sep
                else:
                    str1 += ' '
                str1 += '{:<{:d}}'.format(r1, maxlen_r1)

                if f2 is not None:
                    str2 = '{:>{:d}}'.format(f2, maxlen_f2)
                    if len(r2) > 0:
                        str2 += sep
                    else:
                        str2 += ' '
                    str2 += '{:<{:d}}'.format(r2, maxlen_r2)

                    if not is_competition:
                        hg_rule = ('H({}, {}{}{:<{:d}}) = {: .3f}').format(
                            str1, str2, bsep, rel, maxlen_rel, val)
                        print(hg_rule)
                    else:
                        rel1, rel2 = rel.split('/')
                        str1 = '{}{}{:<{:d}}'.format(
                            str1, bsep, rel1, maxlen_rel)
                        str2 = '{}{}{:<{:d}}'.format(
                            str2, bsep, rel2, maxlen_rel)
                        hg_rule = ('H({}, {}) = {: .3f}').format(
                            str1, str2, val)
                        print(hg_rule)
                else:
                    hg_rule = ('H({}) = {: .3f}').format(str1, val)
                    print(hg_rule)

        for rule_type in rule_types:
            rules = self.subset_rules(rule_type)
            if len(rules) > 0:
                print('\n=== {} rules =====================\n'.format(rule_type))
                pretty_print(rules, rule_type == 'competition')

        print('\n')

    def subset_rules(self, rule_type):

        if not isinstance(rule_type, list):
            rule_type = [rule_type]

        return [rule for rule in self.rules if rule['rule'] in rule_type]

    def find_bindings(self, bnames):

        if not isinstance(bnames, list):
            bnames = [bnames]

        return [bi for bi, bname in enumerate(self.binding_names)
                if bname in bnames]

    def get_bindings(self, idx=None):

        if idx is None:
            return self.binding_names
        else:
            if not isinstance(idx, list):
                idx = [idx]
            return [self.binding_names[ii] for ii in idx]

    def get_mothers(self, bname):

        fname, rname = bname.split(self.opts['bsep'])
        mothers_f = self.g.get_mothers(fname)
        mothers_r = self.roles.get_mothers(rname)
        # print(mothers_f)
        # print(mothers_r)
        assert len(mothers_f) == len(mothers_r)

        res = {}
        for key in mothers_f.keys():
            res[key] = []
            for f in mothers_f[key]:
                for r in mothers_r[key]:
                    res[key].append(f + self.opts['bsep'] + r)

        return res

    def find_mothers(self, bname):

        res = {}
        for key, val in self.get_mothers(bname).items():
            res[key] = self.find_bindings(val)
        return res

    def has_mother(self, bname):

        res = self.find_mothers(bname)
        mothers = []
        for key, val in res.items():
            mothers += val
        return len(mothers) > 0

    def is_mother(self, bname_m, bname_d):

        mothers = []
        for key, val in self.find_mothers(bname_d).items():
            mothers += val

        return bname_m in self.get_bindings(mothers)

    def get_daughters(self, bname):

        fname, rname = bname.split(self.opts['bsep'])
        daughters_f = self.g.get_daughters(fname)
        daughters_r = self.roles.get_daughters(rname)
        assert len(daughters_f) == len(daughters_r)

        res = {}
        for key in daughters_f.keys():
            res[key] = []
            for f in daughters_f[key]:
                for r in daughters_r[key]:
                    res[key].append(f + self.opts['bsep'] + r)

        return res

    def find_daughters(self, bname):

        daughters = self.get_daughters(bname)
        for key, val in daughters.items():
            daughters[key] = self.find_bindings(val)
        return daughters

    def has_daughter(self, bname):

        res = self.find_daughters(bname)
        daughters = []
        for key, val in res.items():
            daughters += val
        return len(daughters) > 0

    def is_daughter(self, bname_d, bname_m):

        daughters = []
        for key, val in self.find_daughters(bname_m).items():
            daughters += val

        return bname_d in self.get_bindings(daughters)

    def generate_sentence(self, min_sent_len=None, max_sent_len=None, use_type=True):
        '''Returns a sentence (list of str) with its parse tree.'''

        if max_sent_len is None:
            max_sent_len = self.opts['max_sent_len']
        if min_sent_len is None:
            min_sent_len = 1

        # NOTE: the program uses the orginal grammar object g0, not the augmented version g.
        sent, parse, p = self.g0.generate_sentence(
            min_sent_len=min_sent_len, max_sent_len=max_sent_len, use_type=use_type)

        if self.opts['role_system'] == 'brick_role':
            parse = self.convert(parse)

        return sent, parse, p

    def convert(self, parse):

        terminals = parse.get_terminals()

        parse_new = []
        parse_new.append(terminals)
        for lv in range(1, len(terminals)):
            maxpos = len(terminals) - lv
            parse_new.append([Node('_')] * maxpos)

        for lv in range(len(parse_new) - 1):
            pos = 1
            while pos < len(parse_new[lv]):
                node1 = parse_new[lv][pos - 1]
                node2 = parse_new[lv][pos]

                if (node1.mother is not None) and (node2.mother is not None) and \
                    (node1.mother.sym == node2.mother.sym) and \
                    (node1.mother.children.index(node1) == 0) and \
                        (node2.mother.children.index(node2) == 1):
                    parse_new[lv + 1][pos - 1] = node1.mother

                else:
                    if node1.mother is not None:
                        idx1 = node1.mother.children.index(node1)
                        node1_copy = copy.deepcopy(node1)
                        if idx1 == 0:
                            if parse_new[lv + 1][pos - 1].sym == '_':
                                if self.opts['add_copy_rules'] and (self.opts['copy'] not in node1_copy.sym):
                                    if self.opts['pos_copy'] == 'l':
                                        node1_copy.sym = self.opts['copy'] + \
                                            node1_copy.sym
                                    else:
                                        node1_copy.sym = node1_copy.sym + \
                                            self.opts['copy']
                                parse_new[lv + 1][pos - 1] = node1_copy
                        elif idx1 == 1:
                            if parse_new[lv + 1][pos - 2].sym == '_':
                                if self.opts['add_copy_rules'] and (self.opts['copy'] not in node1_copy.sym):
                                    if self.opts['pos_copy'] == 'l':
                                        node1_copy.sym = self.opts['copy'] + \
                                            node1_copy.sym
                                    else:
                                        node1_copy.sym = node1_copy.sym + \
                                            self.opts['copy']
                                parse_new[lv + 1][pos -
                                                  2] = node1_copy   # bug_fix

                    if node2.mother is not None:
                        idx2 = node2.mother.children.index(node2)
                        node2_copy = copy.deepcopy(node2)
                        if idx2 == 0:
                            if parse_new[lv + 1][pos].sym == '_':
                                if self.opts['add_copy_rules'] and (self.opts['copy'] not in node2_copy.sym):
                                    if self.opts['pos_copy'] == 'l':
                                        node2_copy.sym = self.opts['copy'] + \
                                            node2_copy.sym
                                    else:
                                        node2_copy.sym = node2_copy.sym + \
                                            self.opts['copy']
                                parse_new[lv + 1][pos] = node2_copy   # bug_fix
                        elif idx2 == 1:
                            if parse_new[lv + 1][pos - 1].sym == '_':
                                if self.opts['add_copy_rules'] and (self.opts['copy'] not in node2_copy.sym):
                                    if self.opts['pos_copy'] == 'l':
                                        node2_copy.sym = self.opts['copy'] + \
                                            node2_copy.sym
                                    else:
                                        node2_copy.sym = node2_copy.sym + \
                                            self.opts['copy']
                                parse_new[lv + 1][pos - 1] = node2_copy
                pos += 1

        for lv in range(len(parse_new)):
            for pos in range(len(parse_new) - lv):
                parse_new[lv][pos] = parse_new[lv][pos].sym

        parse = parse_new
        if self.opts['use_same_len']:
            parse_new = []
            for lv in range(self.opts['max_sent_len']):
                targ_len = self.opts['max_sent_len'] - lv
                if lv < len(parse):
                    bnames = parse[lv]
                    if lv == 0:
                        bnames += [self.opts['f_empty']] * \
                            (targ_len - len(bnames))
                    else:
                        bnames += [self.opts['f_empty_copy']] * \
                            (targ_len - len(bnames))
                    parse_new.append(bnames)
                elif lv + 1 == self.opts['max_sent_len']:
                    parse_new.append([self.opts['f_root']])
                else:
                    parse_new.append(
                        [self.opts['f_root']] + [self.opts['f_empty_copy']] * (targ_len - 1))

            parse = parse_new

        return parse

    def update_binding_names(self):
        self.binding_names = [f + self.opts['bsep'] + r
                              for r in self.role_names
                              for f in self.filler_names]

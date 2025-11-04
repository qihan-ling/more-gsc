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
        self._tokenize_cnf()
        self._tokenize_fillers()
        self._sort_rules()

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
        # self._add_competition_rules()
        # self._add_null_rules()
        self._add_unary_rules()
        self._add_expansion_rules()

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


# =============================================================================
# JAX-ACCELERATED TRIAL EXECUTION
# =============================================================================
# These functions enable GPU-accelerated parallel trial execution

if JAX_AVAILABLE:

    def _build_filler_type_map(net):
        """
        Precompute mapping from filler base types to all matching fillers.

        This is needed because JAX can't call Python methods during JIT compilation.
        The CPU version's set_input() expands types dynamically, but JAX needs
        a precomputed mapping.

        Args:
            net: GscNet instance

        Returns:
            dict: Mapping from base filler type (e.g., 'N:0') to list of matching
                  filler names (e.g., ['N:0', '*N:0', '#N:0'])
        """
        filler_type_map = {}
        g = net.hg.g

        # Build mapping for each unique filler type
        seen_types = set()
        for filler in net.filler_names:
            # Get the base type (remove brackets, copy symbols, pos markers)
            base_type = g.get_types([filler],
                                    ignore_copy=True,
                                    ignore_bracket=True,
                                    ignore_pos_f=g.opts['use_pos_f'])[0]

            if base_type in seen_types:
                continue
            seen_types.add(base_type)

            # Find all fillers matching this type
            fi_list = g.find_fillers_type(base_type,
                                          ignore_bracket=True,
                                          ignore_copy=True,
                                          ignore_pos_f=g.opts['use_pos_f'])

            matching_fillers = g.get_fillers(fi_list)

            # Filter out copy symbols if requested (matching CPU behavior)
            copy_symbol = g.opts.get('copy', '@')
            matching_fillers = [
                f for f in matching_fillers if copy_symbol not in f]

            filler_type_map[base_type] = matching_fillers

        return filler_type_map

    def _compute_scale_constants_jax(pos, net_params):
        """
        Compute scale_constants for role masking based on word position.

        Args:
            pos: Word position (0 for wrapup, 1+ for prefix words)
            net_params: Dictionary containing role information

        Returns:
            scale_constants: Array of shape (num_bindings,) with exponential weights
        """
        num_bindings = net_params['num_bindings']
        num_roles = net_params['num_roles']
        num_fillers = net_params['num_fillers']

        if pos == 0:
            # Wrapup: all roles active
            return jnp.ones(num_bindings)

        # Get parameters
        scale_type = net_params.get('scale_type', 'diagonal')
        scaling_factor = net_params.get('scaling_factor', 1.0)
        # List of (lv, pos) tuples
        role_names_tuples = net_params['role_names_tuples']

        # Convert to JAX arrays for vectorized operations
        # Level for each role
        lv_array = jnp.array([t[0] for t in role_names_tuples])
        # Position for each role
        pos_array = jnp.array([t[1] for t in role_names_tuples])

        # Compute weight for each role
        if scale_type == 'diagonal':
            # Symmetric diagonal weighting
            role_weights = jnp.exp(-jnp.abs(lv_array +
                                   pos_array - (pos + 1)) * scaling_factor)
        elif scale_type == 'pos':
            # Position-based weighting
            role_weights = jnp.exp(-jnp.abs(pos_array - pos) * scaling_factor)
        else:
            role_weights = jnp.ones(num_roles)

        # Expand role weights to binding weights
        # Each role's weight applies to all its fillers
        # weights[ri * num_fillers : (ri+1) * num_fillers] = role_weights[ri]
        weights = jnp.repeat(role_weights, num_fillers)

        return weights

    def _compute_external_input_jax(binding_name, net_params):
        """
        Compute external input extC for a given binding name with type expansion.

        Matches CPU behavior: expands filler types to all matching fillers.
        For example, 'N:0/(1,1)' expands to ['N:0/(1,1)', '*N:0/(1,1)', '#N:0/(1,1)', ...]

        Args:
            binding_name: String like "N:0/(1,1)"
            net_params: Dictionary containing binding_names list, estr, filler_type_map

        Returns:
            extC: Array of shape (num_bindings,) with input at specified binding
        """
        num_bindings = net_params['num_bindings']
        binding_names = net_params['binding_names']
        estr = net_params['estr']

        filler_type_map = net_params['filler_type_map']
        bsep = net_params['bsep']

        extC = jnp.zeros(num_bindings)

        # Split binding name into filler and role
        try:
            filler, role = binding_name.split(bsep)
        except ValueError:
            # Binding not found, return zero input
            return extC

        # Get all fillers matching this type
        matching_fillers = filler_type_map.get(filler, [filler])

        # Set external input for all matching bindings
        for matching_filler in matching_fillers:
            expanded_binding = matching_filler + bsep + role
            try:
                idx = binding_names.index(expanded_binding)
                extC = extC.at[idx].set(estr)
            except ValueError:
                # Binding not found, skip
                pass

        return extC

    def _run_single_trial_jax(rng_key, net_params, prefix, update_q_discrete):
        """
        Pure functional version of a single trial for JAX.

        Args:
            rng_key: JAX random key for this trial
            net_params: Dictionary containing network parameters and state
            prefix: List of filler names for the prefix  (e.g., ['N:0', 'Vi:0'])
            update_q_discrete: Boolean for q update mode

        Returns:
            actC: Final activation state (num_bindings,)
            grid_point_indices: Grid point as filler indices per role (num_roles,)
        """
        # Initialize state with noise
        rng_key, noise_key = jax.random.split(rng_key)
        noise = jax.random.normal(
            noise_key, (net_params['num_bindings'],)) * net_params['init_noise_mag']
        actC = net_params['ep'] + noise

        # Initialize other state variables
        q = jnp.ones(net_params['num_roles']) * net_params['q_init']
        T = net_params['T_init']
        dt = net_params['dt_init']

        def dynamics_step(carry, _):
            actC, q, T, rng, extC_val, scale_const, q_max_val = carry
            # Split RNG for this step
            rng, step_rng = jax.random.split(rng)

            # Extract parameters
            WC = net_params['WC']
            bC = net_params['bC']
            S = net_params['S']
            C = net_params['C']
            bowl_strength = float(net_params['bowl_strength'])
            bowl_center = float(net_params['bowl_center'])
            m = float(net_params['m'])

            # Reshape actC to matrix form (fillers × roles)
            actCmat = actC.reshape(
                (net_params['num_fillers'], net_params['num_roles']), order='F')

            # ===================================================================
            # Compute HGradC (Harmony gradient in conceptual coordinates)
            # ===================================================================

            # 1. Grammar component (weights + biases + external input)
            hgrad_g = WC @ actC + bC + extC_val

            # 2. Bowl constraints (attraction to bowl center)
            hgrad_b = bowl_strength * (bowl_center - actC)

            # 3. Commitment energy (q term) - pushes toward 0 or 1
            q_extended = jnp.repeat(q, net_params['num_fillers'])
            hgrad_q0 = -2 * q_extended * actC * (1 - actC) * (1 - 2 * actC)

            # 4. Role-filling constraint - one filler per role
            ssq = jnp.sum(actCmat ** 2, axis=0)
            ssq_extended = jnp.repeat(ssq - 1, net_params['num_fillers'])
            hgrad_q1 = -4 * m * actC * ssq_extended

            # Combine and transform gradient components
            HGradC_val = hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1

            # ===================================================================
            # CRITICAL: Apply S matrix and scale_constants
            # ===================================================================
            # This matches the original: gradC = scale_constants * S.dot(HGradC())
            gradC = scale_const * (S @ HGradC_val)

            # Euler integration
            actC = actC + dt * gradC

            # Add noise
            # Add noise in neural space, transform to conceptual
            noise_neural = jax.random.normal(
                step_rng, (net_params['num_units'],)) * jnp.sqrt(2 * T * dt)
            noiseC = jnp.sqrt(scale_const) * (C @ noise_neural)
            actC = actC + noiseC

            # Update q
            q = q + net_params['q_rate'] * dt
            q = jnp.clip(q, 0, q_max_val)

            return (actC, q, T, rng, extC_val, scale_const, q_max_val), None

        # Process prefix words if any
        if prefix is not None and len(prefix) > 0:
            qpolicy = net_params['qpolicy']
            bsep = net_params['bsep']
            for wpos, fname in enumerate(prefix, start=1):
                # Construct binding name
                binding_name = f"{fname}{bsep}(1,{wpos})"
                # Compute external input
                extC = _compute_external_input_jax(binding_name, net_params)

                # Compute scale_constants if enabled
                if net_params['update_scale_constants']:
                    scale_constants = _compute_scale_constants_jax(
                        wpos, net_params)
                else:
                    scale_constants = jnp.ones(net_params['num_bindings'])

                # Run dynamics loop
                # Set q_max for this word
                q_max_word = qpolicy[wpos]

                # Calculate duration for this word
                q_inc = qpolicy[wpos] - qpolicy[wpos - 1]
                duration = jnp.max(q_inc) / net_params['q_rate']
                num_steps = jnp.int32(jnp.ceil(duration / dt))

                # Run dynamics for this prefix word
                (actC, q, T, rng_key, _, _, _), _ = jax.lax.scan(
                    dynamics_step,
                    (actC, q, T, rng_key, extC, scale_constants, q_max_word),
                    None,
                    length=num_steps
                )

        # Wrapup phase: clear input, reset scale_constants, run to completion
        extC_wrapup = jnp.zeros(net_params['num_bindings'])

        if net_params['update_scale_constants']:
            scale_constants_wrapup = _compute_scale_constants_jax(
                0, net_params)
        else:
            scale_constants_wrapup = jnp.ones(net_params['num_bindings'])

        q_max_final = net_params['q_max']

        # Calculate wrapup duration
        duration_wrapup = jnp.max(q_max_final - q) / net_params['q_rate']
        num_steps_wrapup = jnp.int32(jnp.ceil(duration_wrapup / dt))

        # Run wrapup dynamics
        (actC, q, T, rng_key, _, _, _), _ = jax.lax.scan(
            dynamics_step,
            (actC, q, T, rng_key, extC_wrapup, scale_constants_wrapup, q_max_final),
            None,
            length=num_steps_wrapup
        )

        # Extract grid point (argmax per role)
        actCmat = actC.reshape(
            (net_params['num_fillers'], net_params['num_roles']), order='F')
        grid_point = jnp.argmax(actCmat, axis=0)

        return actC, grid_point

    def _extract_net_params_for_jax(net):

        # Extract role position tuples for scale_constants computation
        role_names_tuples = []
        if hasattr(net.hg, 'roles'):
            for rname in net.role_names:
                lv, pos = net.hg.roles.str2tuple(rname)
                role_names_tuples.append((lv, pos))
        # Build filler type map for external input type expansion
        filler_type_map = _build_filler_type_map(net)
        """Extract network parameters into a JAX-compatible dictionary."""
        params = {
            'num_bindings': net.num_bindings,
            'num_roles': net.num_roles,
            'num_fillers': net.num_fillers,
            'num_units': net.num_units,
            'WC': jnp.array(net.WC),
            'bC': jnp.array(net.bC),
            # External input strength
            'estr': float(net.estr[0]) if hasattr(net.estr, '__len__') else float(net.estr),
            'ep': jnp.array(net.ep),
            'init_noise_mag': net.train_opts['init_noise_mag'],
            'q_init': net.opts['q_init'],
            'q_max': net.opts['q_max'],
            # Default to 1.0 if not found
            'q_rate': net.opts.get('q_rate', 1.0),
            'dt_init': net.opts['dt_init'],
            'T_init': net.opts['T_init'],
            'qpolicy': jnp.array(net.qpolicy) if hasattr(net, 'qpolicy') else None,
            # Critical parameters for correct gradient computation
            'S': jnp.array(net.S),  # Inverse similarity matrix
            'C': jnp.array(net.C),
            'bowl_strength': net.opts.get('bowl_strength', 0.0),
            'bowl_center': net.opts.get('bowl_center', 0.5),
            'm': net.opts.get('m', 1.0),  # Role-filling constraint strength
            'scale_constants': jnp.array(net.scale_constants) if hasattr(net, 'scale_constants') else jnp.ones(net.num_bindings),
            # For prefix handling
            'binding_names': net.binding_names,  # List of binding name strings
            # Binding separator
            'bsep': net.hg.opts['bsep'] if hasattr(net.hg, 'opts') else '/',
            'role_names_tuples': role_names_tuples,  # List of (lv, pos) tuples
            'scale_type': net.opts.get('scale_type', 'diagonal'),
            'scaling_factor': net.opts.get('scaling_factor', 1.0),
            'update_scale_constants': net.train_opts.get('update_scale_constants', False),
            'filler_type_map': filler_type_map,  # Filler type expansion mapping
        }
        return params

    # Create batched version using vmap
    _run_trials_batched_jax = vmap(
        _run_single_trial_jax, in_axes=(0, None, None, None))


class GscNet():
    # NOTE: CHECK method backup_parametres()

    def __init__(self, hg=None, encodings=None, opts=None, qpolicy=None, seed=None):

        if seed is not None:
            np.random.seed(seed)

        t0 = time.time()
        self.hg = hg
        self._set_encodings()
        self._update_encodings(encodings=encodings)
        self._set_opts()
        self._update_opts(opts=opts)
        self._add_names()
        self._generate_encodings()
        self._add_change_of_basis_matrices()
        dur = time.time() - t0
        print('{} s for generating encodings'.format(dur))

        t0 = time.time()
        # Add parameters ==========================================
        self.WC = np.zeros((self.num_bindings, self.num_bindings))
        self.bC = np.zeros(self.num_bindings)
        self.estr = self.opts['init_estr'] * np.ones(self.num_bindings)
        if hg is not None:
            self._build_model()
            self._adjust_default_param_vals()
            if self.opts['use_second_order_bias']:
                self.bias2weight()
        dur = time.time() - t0
        print('{} s for initializing parameter values'.format(dur))

        self.extC = np.zeros(self.num_bindings)
        self.ext = self.C2N(actC=self.extC)
        self._set_bowl_parameters()

        self.q = self.opts['q_init'] * np.ones(self.num_roles)
        self.T = self.opts['T_init']
        self.dt = self.opts['dt_init']   # NOTE: Consider using a vector

        # Add state variables =====================================
        # Previously, implemented as a method _add_state_variables()
        self.t = 0.
        self.actC = np.zeros(self.num_bindings)
        self.actCmat = self.vec2mat(self.actC)
        self.act = self.C2N()
        # # activation state at previous step. used to compute speed
        # self.actC_prev = np.zeros(self.num_bindings)
        # self.act_prev = self.C2N(actC=self.actC_prev)
        # # external input at previous step. used to change external input
        # # gradually by linear interpolation
        # self.extC_prev = np.zeros(self.num_bindings)
        # self.ext_prev = self.C2N(actC=self.extC_prev)

        # self._set_quant_list()
        self.update_scale_constants(pos=0)

        t0 = time.time()
        self.get_ep(method=self.opts['ep_method'])
        dur = time.time() - t0
        print('{} s for finding a global equilibrium point'.format(dur))

        self.set_state(mu=self.ep)
        if qpolicy is None:
            self.qpolicy = np.arange(self.hg.opts['max_sent_len'] + 1)
        else:
            self.qpolicy = qpolicy
        self.backup_parameters()
        # NOTE: only a subset of parameters will be backed up.

    #####################################################################
    #
    # Make pretrained model consistent with the new version
    #
    #####################################################################

    def update_model(self):

        if not hasattr(self, 'scale_constants'):
            self.opts['scaling_factor'] = 0.0
            self.opts['scale_type'] = 'diagonal'
            self.update_scale_constants(pos=0)

    #####################################################################
    #
    # Vector encodings/embeddings
    #
    #####################################################################

    def _set_encodings(self):

        self.encodings = {}

        # filler_names: list of filler names
        # role_names: list of role names
        self.encodings['filler_names'] = None
        self.encodings['role_names'] = None

        # seed_f and seed_r: (int) seed numbers
        # used when generating distributed representations
        # of fillers and roles (see dot_products())
        self.encodings['seed_f'] = None
        self.encodings['seed_r'] = None

        # coord_f: (str) coordinates of filler representation space
        # coord_r: (str) coordinates of role representation space
        # 'N' or 'dist' for neural coordinates (distributed rep.)
        # 'C' or 'local' for conceptual coordinates (local rep.)
        self.encodings['coord_f'] = 'N'
        self.encodings['coord_r'] = 'N'

        # dim_f: dimension of filler representation space
        # dim_r: dimension of role representation space
        self.encodings['dim_f'] = None
        self.encodings['dim_r'] = None

        # F: encodings of fillers
        # R: encodings of roles
        # F and R are 2d NumPy arrays of floating numbers.
        # Each column vectors is an encoding of a symbol.
        self.encodings['F'] = None
        self.encodings['R'] = None

        # dp_f: pairwise filler similarity
        # dp_r: pairwise role similarity
        # dp_f and dp_r can be either a float between 0 and 1
        # or a 2d NumPy array (similarity matrix) in which
        # the (i,j)-th element represents pairwise similarity
        # (dot product) between the i-th symbol and the j-th
        # symbol. When a float number is given, it represents
        # similarity for every unique pair of symbols.
        self.encodings['dp_f'] = 0.
        self.encodings['dp_r'] = 0.

        # similarity: list of
        # [[symbol1, symbol2], pairwise similarity (dot product)]
        # Example:
        #   [[['A', 'B'], 0.3],
        #    [['A', 'C'], 0.1]]
        # When this option is given, it will overwite
        # dp_f and dp_r.
        self.encodings['similarity'] = None

    def _update_encodings(self, encodings):
        """Updates encodings."""
        if encodings is not None:
            for key in encodings:
                if key in self.encodings:
                    self.encodings[key] = encodings[key]
                else:
                    sys.exit('Cannot find `{}` in encodings.'.format(key))

    def _add_names(self):

        if self.hg is not None:
            self.encodings['filler_names'] = self.hg.filler_names
            self.encodings['role_names'] = self.hg.role_names
            bsep = self.hg.opts['bsep']
        else:
            bsep = '/'

        if self.encodings['filler_names'] is None:
            sys.exit("Please provide a list of filler names.")
        if self.encodings['role_names'] is None:
            sys.exit("Please provide a list of role names.")

        self.filler_names = self.encodings['filler_names']
        self.role_names = self.encodings['role_names']
        self.binding_names = [
            f + bsep + r for r in self.role_names
            for f in self.filler_names]

        self.num_fillers = len(self.filler_names)
        self.num_roles = len(self.role_names)
        self.num_bindings = len(self.binding_names)

    def _generate_encodings(self, overwrite=False):

        if self.encodings['seed_f'] is None:
            self.encodings['seed_f'] = np.random.randint(10000)
        if self.encodings['seed_r'] is None:
            self.encodings['seed_r'] = np.random.randint(10000)

        if self.encodings['similarity'] is not None:
            # Update dp_f and dp_r
            dp_f = np.diag(np.ones(self.num_fillers))
            dp_r = np.diag(np.ones(self.num_roles))

            for dp in self.encodings['similarity']:
                if all(sym in self.filler_names for sym in dp[0]):
                    dp_f[self.filler_names.index(dp[0][0]),
                         self.filler_names.index(dp[0][1])] = dp[1]
                    dp_f[self.filler_names.index(dp[0][1]),
                         self.filler_names.index(dp[0][0])] = dp[1]
                elif all(sym in self.role_names for sym in dp[0]):
                    dp_r[self.role_names.index(dp[0][0]),
                         self.role_names.index(dp[0][1])] = dp[1]
                    dp_r[self.role_names.index(dp[0][1]),
                         self.role_names.index(dp[0][0])] = dp[1]
                else:
                    sys.exit(('Cannot find some symbols (fillers or roles) '
                              'in your similarity list.'))

            self.encodings['dp_f'] = dp_f
            self.encodings['dp_r'] = dp_r

        if (self.encodings['F'] is None) or (overwrite):
            self.encodings['F'] = encode_symbols(
                self.num_fillers,
                coord=self.encodings['coord_f'],
                dp=self.encodings['dp_f'],
                dim=self.encodings['dim_f'],
                seed=self.encodings['seed_f'])

        if (self.encodings['R'] is None) or (overwrite):
            self.encodings['R'] = encode_symbols(
                self.num_roles,
                coord=self.encodings['coord_r'],
                dp=self.encodings['dp_r'],
                dim=self.encodings['dim_r'],
                seed=self.encodings['seed_r'])

        self.F = self.encodings['F']
        self.R = self.encodings['R']
        self.dim_f = self.F.shape[0]
        self.dim_r = self.R.shape[0]
        self.num_units = self.dim_f * self.dim_r
        self.encodings['dim_f'] = self.F.shape[0]
        self.encodings['dim_r'] = self.R.shape[0]

        ndigits = len(str(self.num_units))
        self.unit_names = [
            'U' + str(ii + 1).zfill(ndigits)
            for ii in list(range(self.num_units))]

    def _add_change_of_basis_matrices(self):

        # For justification of kronecker product, see:
        # http://en.wikipedia.org/wiki/Vectorization_(mathematics)
        N = np.kron(self.R, self.F)    # Pay attention to the argument order
        # Column vectors of N are the neural coordinates of the conceptual basis vectors.
        if N.shape[0] == N.shape[1]:
            C = np.linalg.inv(N)
        else:
            # N may be a non-square matrix. If so, compute pseudo-inverse.
            # CHECK if this is valid.
            C = np.linalg.pinv(N)
        self.N = N
        self.C = C

        self.Gc = self.C.T.dot(self.C)
        self.C_reshaped = self.C.reshape(
            (self.num_fillers, self.num_roles, self.num_units), order='F')

        self.S = self.C.dot(self.C.T)  # inverse of similarity matrix

    #####################################################################
    #
    # Hyperparameters (options)
    #
    #####################################################################

    def _set_opts(self):
        # Set default options

        self.opts = {}

        # Time step size
        # min_dt: (float) minimal value of dt
        # max_dt: (float) maximal value of dt
        # dt_init: (float) initial value of dt
        # adaptive_dt: (bool) update dt adaptively or not?
        self.opts['min_dt'] = 0.0005
        self.opts['max_dt'] = 0.01
        self.opts['dt_init'] = 0.005
        self.opts['adaptive_dt'] = False

        # Temperature parameters: for simulated annealing
        # T_init: (float) initial temperature
        # T_min: (float) minimal temperature
        # T_decay_rate: (float) exponenetial decay rate
        self.opts['T_init'] = 1e-3
        self.opts['T_min'] = 0.
        self.opts['T_decay_rate'] = 0.  # qe-3

        # Bowl parameters
        # bowl_center: (float)
        # bowl_strength: (float) will be updated
        # beta_min_offset: (float) value to be added to minimal bowl strength
        if self.hg is not None:
            self.opts['bowl_center'] = 1 / np.sqrt(self.hg.num_fillers)
        else:
            self.opts['bowl_center'] = 0.1
        self.opts['bowl_strength'] = None
        self.opts['beta_min_offset'] = 0.1

        # Quantization parameters: the first three parameters
        #   will be ignored when q_policy is given.
        # q_init: (float, >= 0) initial value of q
        # q_max: (float) maximum value of q
        # q_rate: (float) rate of change in q (i.e., dq/dt)
        # q_policy: (2d NumPy array) quantizatoin policy
        #   first column: time points
        #   second column: q values at the time points
        #   q values between the time points are linearly interpolated
        # c: (float, 0 <= c <= 1) relative strength of
        #   the first quantization constraint (see Hq0)
        self.opts['q_init'] = 0.
        self.opts['q_max'] = 20.      # perviously 200
        # self.opts['q_min'] = 0.
        self.opts['q_rate'] = 1.
        self.opts['q_policy'] = None
        # self.opts['c'] = 0.5

        # trace_varnames: (list) of names (str) of variables
        #   to log their changes in time
        # self.opts['trace_varnames'] = [
        #     't', 'actC', 'q', 'T', 'H', 'Hg', 'Hg', 'Hq0', 'Hq1']
        self.opts['trace_varnames'] = [
            't', 'actC', 'q']

        # Parameters used when computing distance and (ema_)speed
        self.opts['coord'] = 'N'
        self.opts['norm_ord'] = np.inf
        self.opts['ema_factor'] = 0.001
        self.opts['ema_tau'] = -1 / np.log(self.opts['ema_factor'])

        # quantization constraint type (with_null vs. without_null)
        # with_null: sum of act^2 = 1
        # without_null: sum of act^2 = 0 or 1
        self.opts['quant_type'] = 'with_null'

        # # Not much important
        # self.opts['H0_on'] = 1.
        # self.opts['H1_on'] = 1.
        # self.opts['Hq_on'] = 1.
        self.opts['m'] = 30.   # Hq1 strength
        self.opts['bias_factor'] = 30.

        self.opts['min_H_increase'] = 1e-3
        # self.opts['use_Hq1_maxvar'] = False  # CHECK the functions
        # self.opts['add_neg_weight_btw_roles'] = False
        # self.opts['neg_weight_btw_roles'] = None

        self.opts['use_second_order_bias'] = True
        self.opts['init_estr'] = 2.

        self.opts['scaling_factor'] = 0.
        self.opts['scale_type'] = 'diagonal'
        self.opts['ep_method'] = 'integration'
        self.opts['use_runC'] = False

        self.opts['penalize_root_posN'] = True

    def _update_opts(self, opts):

        if opts is not None:
            for key in opts:
                if key in self.opts:
                    self.opts[key] = opts[key]
                    if key == 'ema_factor':
                        self.opts['ema_tau'] = -1 / np.log(self.opts[key])
                    if key == 'ema_tau':
                        self.opts['ema_factor'] = np.exp(-1 / self.opts[key])
                else:
                    sys.exit('Cannot find `{}` in opts.'.format(key))

    #####################################################################
    #
    # Update model parameters (weight and bias values)
    #
    #####################################################################

    def _build_model(self):
        # Initialize the model by setting weight and bias parameters to
        # some default values specified in HG.
        # NOTE: Complex competition rules and null rules were removed temporarily.

        # max_sent_len = self.hg.opts['max_sent_len']
        # use_hnf = self.hg.g.opts['use_hnf']
        role_system = self.hg.opts['role_system']
        roles = self.hg.roles
        bsep = self.hg.opts['bsep']

        H_root_illegitimate = self.hg.opts['H_root_illegitimate']
        H_terminal_illegitimate = self.hg.opts['H_terminal_illegitimate']
        H_nonterminal_illegitimate = self.hg.opts['H_nonterminal_illegitimate']
        H_copy_illegitimate = self.hg.opts['H_copy_illegitimate']

        self.WC = np.zeros((self.num_bindings, self.num_bindings))

        # t1 = time.time()
        # Binary and copy rules =========================
        for rule in self.hg.subset_rules(['binary', 'copy']):
            for role in roles.role_names:
                if roles.is_bracketed(role) == rule['br']:
                    mother_roles = roles.get_mothers(role)
                    focus_mother_roles = mother_roles[rule['rel']]
                    for focus_mother_role in focus_mother_roles:
                        if focus_mother_role in roles.role_names:
                            b1name = rule['f1'] + bsep + role
                            b2name = rule['f2'] + bsep + focus_mother_role
                            self.set_weight(b1name, b2name, rule['H'],
                                            cumulative=True, c2n=False)
        # dur = time.time() - t1
        # print('{} ms for implementing binrary HG rules'.format(dur))

        # Competition rules =========================
        cumulative = False
        for rule in self.hg.subset_rules('competition'):
            r1, r2 = rule['rel'].split('/')
            if r1 == 'ub' and r2 == 'ub':
                for role in roles.role_names:
                    if not roles.is_bracketed(role):
                        bname1 = rule['f1'] + bsep + role
                        bname2 = rule['f2'] + bsep + role
                        self.set_weight(bname1, bname2, rule['H'],
                                        cumulative=cumulative, c2n=False)

            elif r1 == 's' and r2 == 's':
                for role in roles.role_names:
                    bname1 = rule['f1'] + bsep + role
                    bname2 = rule['f2'] + bsep + role
                    self.set_weight(bname1, bname2, rule['H'],
                                    cumulative=cumulative, c2n=False)

            elif r1 == 's' and r2 != 's':
                for role in roles.role_names:
                    bname1 = rule['f1'] + bsep + role
                    mother_roles = roles.get_mothers(role)
                    focus_mother_roles = mother_roles[r2]
                    for mr in focus_mother_roles:
                        if mr in roles.role_names:
                            bname2 = rule['f2'] + bsep + mr
                            self.set_weight(b1name, b2name, rule['H'],
                                            cumulative=cumulative, c2n=False)

            else:
                for role in roles.role_names:
                    if roles.is_bracketed(role) == rule['br']:
                        mother_roles = roles.get_mothers(role)
                        focus_mother_roles1 = mother_roles[r1]
                        focus_mother_roles2 = mother_roles[r2]
                        for mr1 in focus_mother_roles1:
                            for mr2 in focus_mother_roles2:
                                if (mr1 in roles.role_names) and \
                                   (mr2 in roles.role_names):
                                    b1name = rule['f1'] + bsep + mr1
                                    b2name = rule['f2'] + bsep + mr2
                                    self.set_weight(b1name, b2name, rule['H'],
                                                    cumulative=cumulative, c2n=False)

        # Null rules
        for rule in self.hg.subset_rules(['null']):
            for role in roles.role_names:
                if roles.is_bracketed(role) == rule['br']:
                    mother_roles = roles.get_mothers(role)
                    focus_mother_roles = mother_roles[rule['rel']]
                    for focus_mother_role in focus_mother_roles:
                        if focus_mother_role in roles.role_names:
                            b1name = rule['f1'] + bsep + role
                            b2name = rule['f2'] + bsep + focus_mother_role
                            self.set_weight(b1name, b2name, rule['H'],
                                            cumulative=True, c2n=False)

        # Unary rules
        self.bC = np.zeros(self.num_bindings)
        if self.hg.opts['unary_base'] == 'filler':
            for rule in self.hg.subset_rules('unary'):
                self.set_filler_bias(rule['f1'], rule['H'], c2n=False)
        else:
            sys.exit('CHECK "unary_base"!')

        # Additional constraints (penalty for ungrammatical bindings)
        if H_root_illegitimate < 0:
            for rname in roles.role_names:
                if role_system == 'brick_role':
                    lv, pos = roles.str2tuple(rname)
                    if (lv > 0 and pos > 1) or (lv < 0):
                        bnames = [f + bsep + rname
                                  for f in self.hg.get_roots()]
                        self.set_bias(bnames, H_root_illegitimate, c2n=False)
                elif role_system == 'span_role':
                    rname_tuple = roles.str2tuple(rname)
                    if rname_tuple[0] > 1:
                        bnames = [f + bsep + rname
                                  for f in self.hg.get_roots()]
                        self.set_bias(bnames, H_root_illegitimate, c2n=False)
                elif role_system == 'recursive_role':
                    role_root = roles.opts['root']
                    if rname != role_root:
                        bnames = [f + bsep + rname
                                  for f in self.hg.get_roots()]
                        self.set_bias(bnames, H_root_illegitimate, c2n=False)

        if H_terminal_illegitimate < 0:
            for rname in roles.role_names:
                if role_system == 'brick_role':
                    lv, pos = roles.str2tuple(rname)
                    if lv != 1:
                        bnames = [f + bsep + rname
                                  for f in self.hg.g.get_fillers()
                                  if self.hg.g.is_terminal(f)]
                        self.set_bias(
                            bnames, H_terminal_illegitimate, c2n=False)

        if H_nonterminal_illegitimate < 0:
            for rname in roles.role_names:
                if role_system == 'brick_role':
                    lv, pos = roles.str2tuple(rname)
                    if lv == 1:
                        bnames = [f + bsep + rname
                                  for f in self.hg.g.get_fillers()
                                  if (not self.hg.g.is_terminal(f) and
                                      f != self.hg.g.opts['null'])]
                        self.set_bias(
                            bnames, H_nonterminal_illegitimate, c2n=False)

        if H_copy_illegitimate < 0:
            for rname in roles.role_names:
                if role_system == 'brick_role':
                    lv, pos = roles.str2tuple(rname)
                    if lv == 1:
                        bnames = [f + bsep + rname
                                  for f in self.hg.g.get_fillers()
                                  if self.hg.g.is_copy(f)]
                        self.set_bias(bnames, H_copy_illegitimate, c2n=False)

        self._set_weights()
        self._set_biases()

    def _adjust_default_param_vals(self, method='Newton'):

        if self.hg.opts['use_same_len']:
            # Adjust bias values of root bindings before adding expansion rules
            # and bias values of newly added empty bindings.

            bC = self.bC.copy()

            if not self.hg.opts['add1_to_root']:
                # default

                if not np.isclose(self.hg.opts['H_root_illegitimate'], 0.):

                    for fname in self.hg.get_roots():
                        for rname in self.role_names:
                            bname = fname + self.hg.opts['bsep'] + rname
                            idx = self.find_bindings(bname)
                            lv, pos = self.hg.roles.str2tuple(rname)
                            if lv == 1:
                                self.set_bias(
                                    bname, self.hg.opts['H_root_illegitimate'])
                                bC[idx] = self.bC[idx]
                            elif lv == self.hg.opts['max_sent_len']:
                                self.set_bias(bname, self.hg.opts['H_unary_2'])
                                bC[idx] = self.bC[idx]
                            else:
                                if pos == 1:
                                    self.set_bias(
                                        bname, self.hg.opts['H_unary_3'])
                                    bC[idx] = self.bC[idx]
                                else:
                                    self.set_bias(
                                        bname, self.hg.opts['H_root_illegitimate'])
                                    bC[idx] = self.bC[idx]

            else:
                # NEW

                if not np.isclose(self.hg.opts['H_root_illegitimate'], 0.):

                    for fname in self.hg.get_roots():
                        for rname in self.role_names:
                            bname = fname + self.hg.opts['bsep'] + rname
                            idx = self.find_bindings(bname)
                            lv, pos = self.hg.roles.str2tuple(rname)
                            if lv == 1:
                                self.set_bias(
                                    bname, self.hg.opts['H_root_illegitimate'])
                                bC[idx] = self.bC[idx]
                            elif lv == self.hg.opts['max_sent_len']:
                                self.set_bias(bname, self.hg.opts['H_root'])
                                bC[idx] = self.bC[idx]
                            else:
                                if pos == 1:
                                    self.set_bias(
                                        bname, self.hg.opts['H_root'])
                                    bC[idx] = self.bC[idx]
                                else:
                                    if 'penalize_root_posN' in self.opts:
                                        if self.opts['penalize_root_posN']:
                                            self.set_bias(
                                                bname, self.hg.opts['H_root_illegitimate'])
                                            bC[idx] = self.bC[idx]
                                    else:
                                        self.set_bias(
                                            bname, self.hg.opts['H_root_illegitimate'])
                                        bC[idx] = self.bC[idx]

            self.bC = bC.copy()
            if self.hg.opts['add1_to_root']:

                roots = self.hg.g.get_roots() + [self.hg.g.opts['f_root']]
                rid = self.find_roles(self.role_names[-1])
                rid = [ii for ii in rid if ii in self.find_fillers(roots)]
                # rid = self.find_roles(self.role_names[-1])
                # # ignore null '_' (no connection)
                # rid = [ii for ii in rid
                #        if ii not in self.find_fillers(self.hg.g.opts['null'])]
                self.bC[rid] += 1.

            self._set_biases()

            # self.update_bowl_strength()
            # self.get_ep(method=method)
            # self.params_base_bias = self.bC.copy()

    def set_weight(self, bname1, bname2, weight,
                   symmetric=True, cumulative=False, c2n=True):
        '''Set the weight of a connection from binding1 (str or list of str) to
        binding2 (str or list of str). When symmetric is True (default), the
        connection weight from binding2 to binding1 is set to the same value.

        Args:
            bname1: (str or list of str) source binding names
            bname2: (str or list of str) target binding names
            weight: (float) weight value
            symmetric: (bool)

        Example:
            >>> net.set_weight('A/0', 'B/1', 2.)
            >>> net.set_weight('A/0', ['B/1', 'C/2'], 2.)
        '''

        idx1 = self.find_bindings(bname1)
        idx2 = self.find_bindings(bname2)
        if not cumulative:
            if symmetric:
                self.WC[idx1, idx2] = self.WC[idx2, idx1] = weight
            else:
                self.WC[idx2, idx1] = weight
        else:
            # WC = np.zeros(self.WC.shape)
            if symmetric:
                # WC[idx1, idx2] = WC[idx2, idx1] = weight
                # self.WC += WC
                self.WC[idx1, idx2] += weight
                self.WC[idx2, idx1] += weight
            else:
                # WC[idx2, idx1] = weight
                # self.WC += WC
                self.WC[idx2, idx1] += weight

        if c2n:
            self._set_weights()

    def set_bias(self, binding_name, bias, c2n=True):
        '''Set bias values of binding_name (str or list of str) to bias (float).

        Args:
            binding_name: (str or list of str) binding names
            bias: (float) bias value

        Precondition:
            binding_name must contain legitimate binding names.

        Example:
            >>> net.set_bias('A/0', -1.)
        '''

        idx = self.find_bindings(binding_name)
        self.bC[idx] = bias
        if c2n:
            self._set_biases()

    def set_filler_bias(self, filler_name, bias, c2n=True):
        '''Find f/r bindings with filler_name (str or list of str) and
        set their bias values to bias (float).

        Args:
            filler_name: (str or list of str) filler names
            bias: (float) bias value

        Precondition:
            filler_name must contain legitimate filler names.

        Example:
            >>> net.set_filler_name('A', -1.)
        '''

        filler_list = [bb.split('/')[0] for bb in self.binding_names]
        if not isinstance(filler_name, list):
            filler_name = [filler_name]
        for jj, filler in enumerate(filler_name):
            idx = [ii for ii, ff in enumerate(filler_list) if filler == ff]
            self.bC[idx] = bias

        if c2n:
            self._set_biases()

    def set_role_bias(self, role_name, bias, c2n=True):
        '''Find f/r bindings with role_name (str or list of str) and
        set their bias values to bias (float).

        Args:
            role_name: (str or list of str) filler names
            bias: (float) bias value

        Precondition:
            role_name must contain legitimate filler names.

        Example:
            >>> net.set_role_name('0', -1.)
        '''

        role_list = [bb.split('/')[1] for bb in self.binding_names]
        if not isinstance(role_name, list):
            role_name = [role_name]
        for jj, role in enumerate(role_name):
            idx = [ii for ii, rr in enumerate(role_list) if role == rr]
            self.bC[idx] = bias

        if c2n:
            self._set_biases()

    def _set_weights(self):
        '''Converts WC to W.

        WC: W_c, weight matrix for conceptual cooridantes
        W : W_n, weight matrix for neural coordinates
        '''

        self.W = self.C.T.dot(self.WC).dot(self.C)

    def _set_biases(self):
        '''Converts bC to b.

        bC: b_c, bias vector for conceptual coordinates
        b : b_n, bias vector for neural coordinates
        '''

        self.b = self.C.T.dot(self.bC)

    def bias2weight(self):
        '''Set recurrent weights given bias values in conceptual coordinates'''

        self.WC = self.WC + np.diag(2 * self.bC)
        self.bC = np.zeros(self.num_bindings)
        self._set_weights()
        self._set_biases()

    def _set_bowl_parameters(self):
        '''Sets bowl parameters to default values. Default values
        must be updated after setting the weight and bias values.'''

        if isinstance(self.opts['bowl_center'], numbers.Number):
            self.bowl_center = (self.opts['bowl_center'] *
                                np.ones(self.num_bindings))
        else:
            self.bowl_center = self.opts['bowl_center']

        if self.opts['bowl_strength'] is None:
            self.opts['bowl_strength'] = (
                self._compute_recommended_bowl_strength() +
                self.opts['beta_min_offset'])
        else:
            self.check_bowl_strength()

        self.bowl_strength = self.opts['bowl_strength']
        self.zeta = self.C2N(actC=self.bowl_center)

    def update_bowl_strength(self, bowl_strength=None):
        """Replaces the current bowl strength with
        the recommended bowl strength (+ offset)

        Usage:

            >>> net = gsc.GscNet(...)
            >>> net.set_weight('a/(0,1)', 'b/(1,2)', 2.0)
            >>> net.update_bowl_strength()

        : bowl_strength : float or None (=default)
        """

        if bowl_strength is None:
            self.opts['bowl_strength'] = (
                self._compute_recommended_bowl_strength() +
                self.opts['beta_min_offset'])
        else:
            self.opts['bowl_strength'] = bowl_strength
        self.bowl_strength = self.opts['bowl_strength']

    def backup_parameters(self):

        self.params_backup = {}
        self.params_backup['encodings'] = copy.deepcopy(self.encodings)
        self.params_backup['WC'] = self.WC.copy()
        self.params_backup['bC'] = self.bC.copy()
        self.params_backup['estr'] = self.estr.copy()
        self.params_backup['ep'] = self.ep.copy()
        if hasattr(self, 'qpolicy'):
            self.params_backup['qpolicy'] = self.qpolicy.copy()

    #####################################################################
    #
    # Update state variables
    #
    #####################################################################

    def run(self,
            duration,
            update_T=True,
            update_q=True,
            log_trace=True,
            plot=True,
            tol=None,
            trace_list='all'):

        t_max = self.t + duration
        self.converged = False
        self.lapse = 0
        self.maxH = -np.inf

        if log_trace:
            self.initialize_traces(trace_list)

        while self.t < t_max:
            self.update_state()

            if update_T and (self.opts['T_decay_rate'] > 0):
                self.update_T()
            if update_q:
                self.update_q()
            if log_trace:
                self.update_traces()

            if self.check_divergence():
                # if dt is too big, the model may diverge.
                break

            if tol is not None:
                self.check_convergence(tol=tol)
                if self.converged:
                    break

        if log_trace:
            self.finalize_traces()

        # if log_trace and plot:
        #     heatmap(self.traces['actC'].T,
        #             xticklabels='', yticklabels=self.binding_names)

    def check_divergence(self, tol=2.):
        return max(self.actC) > tol

    def run2(self, duration,
             update_T=True,
             log_trace=True,
             plot=True,
             tol=0.5,
             phase1_maxdur=5,
             phase2_maxdur=np.inf,
             phase3_maxdur=5,
             trace_list='all'):

        lapse = []
        hdiff = []

        dur_phase1 = 0
        dur_phase2 = 0
        dur_phase3 = 0
        t_init = self.t

        # Phase 1
        # q_rate_backup = self.opts['q_rate'].copy()
        # self.opts['q_rate'] = 0.
        if phase1_maxdur > 0:
            H_pre_phase1 = self.H()
            self.run(phase1_maxdur,
                     update_T=update_T,
                     update_q=False,
                     log_trace=log_trace,
                     trace_list=trace_list,
                     tol=tol,
                     plot=False)
            # H_after_phase1 = self.H()
            dur_phase1 = self.t - t_init
            # hdiff.append(self.H() - H_pre_phase1)
            hdiff.append(self.maxH - H_pre_phase1)
            lapse.append(dur_phase1)

        # Phase 2
        # self.opts['q_rate'] = q_rate_backup.copy()
        if phase2_maxdur > 0:
            self.run(duration=duration,
                     update_T=update_T,
                     update_q=True,
                     log_trace=log_trace,
                     trace_list=trace_list,
                     plot=False)
            dur_phase2 = self.t - t_init - dur_phase1
            lapse.append(dur_phase2)

        # Phase 3
        # q_rate_backup = self.opts['q_rate'].copy()
        # self.opts['q_rate'] = 0.
        if phase3_maxdur > 0:
            H_pre_phase3 = self.H()
            self.run(phase3_maxdur,
                     update_T=update_T,
                     update_q=False,
                     log_trace=log_trace,
                     trace_list=trace_list,
                     tol=tol,
                     plot=False)
            dur_phase3 = self.t - t_init - dur_phase1 - dur_phase2
            # hdiff.append(self.H() - H_pre_phase3)
            hdiff.append(self.maxH - H_pre_phase3)
            lapse.append(dur_phase3)

        return np.array(lapse), np.array(hdiff)

    def runC(self,
             duration,
             update_T=True,
             update_q=True,
             log_trace=True,
             plot=True,
             tol=None,
             trace_list='all'):

        t_max = self.t + duration
        self.converged = False
        self.lapse = 0
        self.maxH = -np.inf

        if log_trace:
            self.initialize_traces(trace_list)

        while self.t < t_max:
            self.update_stateC()

            if update_T and (self.opts['T_decay_rate'] > 0):
                self.update_T()
            if update_q:
                self.update_q()
            if log_trace:
                self.update_traces()

            if self.check_divergence():
                # if dt is too big, the model may diverge.
                break

            if tol is not None:
                self.check_convergence(tol=tol)
                if self.converged:
                    break

        self.act = self.C2N()

        if log_trace:
            self.finalize_traces()

        # if log_trace and plot:
        #     heatmap(self.traces['actC'].T,
        #             xticklabels='', yticklabels=self.binding_names)

    def reset(self, mu=None, sd=0.):
        '''Reset the model. q and T will be set to their initial values'''

        self.dt = self.opts['dt_init']
        self.q = self.opts['q_init'] * np.ones(self.num_roles)
        self.T = self.opts['T_init']
        self.t = 0.
        self.update_scale_constants(pos=0)

        if mu is None:
            self.set_random_state()
        else:
            self.set_state(mu=mu, sd=sd)

        self.clear_input()

        if hasattr(self, 'traces'):
            del self.traces

    def set_discrete_state(self, binding_names):

        idx = self.find_bindings(binding_names)
        self.actC = np.zeros(self.num_bindings)
        self.actC[idx] = 1.0
        self.actCmat = self.vec2mat()
        self.act = self.C2N()

    def get_discrete_state(self, binding_names):

        idx = self.find_bindings(binding_names)
        actC = np.zeros(self.num_bindings)
        actC[idx] = 1.0
        return actC

    def set_state(self, mu, sd=0.):

        noise_vec = np.random.normal(
            loc=0., scale=sd, size=self.num_bindings)
        self.actC = mu + noise_vec
        self.actCmat = self.vec2mat()
        self.act = self.C2N()

    def set_random_state(self, minact=0, maxact=1):

        self.actC = np.random.uniform(
            minact, maxact, size=self.num_bindings)
        self.actCmat = self.vec2mat()
        self.act = self.C2N(self.actC)

    def update_state(self):
        # ToDo:
        # (1) adaptive stepsize
        # (2) different time scales -> consider dt as a vector (scale_constants)
        #     (scale_constants for noise?)
        # (3) clamp

        grad = self.HGrad()
        grad = self.C2N(self.scale_constants * self.N2C(grad))
        # NOTE: compute scale_constants_N for neural coordinates

        # adaptive stepsize
        # update_dt()  # --> dt as a vector

        self.t += self.dt             # update time
        self.act += self.dt * grad    # Euler integration
        self.add_noise()

        # if self.clamped:
        #     self.act = self.act_clamped()

        # Update actC and actCmat which are needed to compute Hq0Grad and Hq1Grad
        self.actC = self.N2C()
        self.actCmat = self.vec2mat()

    def update_stateC(self):
        # ToDo:
        # (1) adaptive stepsize
        # (2) different time scales -> consider dt as a vector (scale_constants)
        #     (scale_constants for noise?)
        # (3) clamp

        gradC = self.scale_constants * self.S.dot(self.HGradC())

        # adaptive stepsize
        # update_dt()  # --> dt as a vector

        self.t += self.dt             # update time
        self.actC = self.actC + self.dt * gradC    # Euler integration
        self.add_noiseC()

        # if self.clamped:
        #     self.act = self.act_clamped()

        # Update actC and actCmat which are needed to compute Hq0Grad and Hq1Grad
        # self.actC = self.N2C()
        self.actCmat = self.vec2mat()

    def add_noise(self):

        noise = np.sqrt(2 * self.T * self.dt) * \
            np.random.randn(self.num_units)
        noise = self.C2N(np.sqrt(self.scale_constants) *
                         self.N2C(noise))  # rescaling noise
        self.act += noise

    def add_noiseC(self):

        noise = np.sqrt(2 * self.T * self.dt) * \
            np.random.randn(self.num_units)
        noiseC = np.sqrt(self.scale_constants) * \
            self.N2C(noise)  # rescaling noise
        self.actC += noiseC

    def update_T(self):
        # CHECK: dt could be a vector
        self.T = (np.exp(-self.opts['T_decay_rate'] * self.dt) *
                  (self.T - self.opts['T_min']) + self.opts['T_min'])

    def update_q(self):

        if hasattr(self, 'q_mask'):
            self.q += self.opts['q_rate'] * self.q_mask * self.dt
        else:
            self.q += self.opts['q_rate'] * self.dt
        self.q = np.maximum(
            np.minimum(self.q, self.opts['q_max']), 0)

    def update_q_mask(self, pos, lookahead=0, symmetric=False, decay=None):

        self.q_mask = np.ones(self.num_roles)

        if self.hg.opts['role_system'] == 'brick_role':
            if pos > 0:
                self.q_mask = np.zeros(self.num_bindings)
                for rname in self.role_names:
                    idx = self.find_roles(rname)
                    lv0, pos0 = self.hg.roles.str2tuple(rname)
                    lag = (lv0 + pos0) - (pos + 1)

                    if decay is None:
                        if lag == 0:
                            self.q_mask[idx] = 1.
                        if lookahead == lag:
                            self.q_mask[idx] = 1.

                    if decay is not None:
                        if lag == 0:
                            self.q_mask[idx] = 1.
                        if lag < 0:
                            if symmetric:
                                self.q_mask[idx] = np.exp(-abs(lag) * decay)
                            # else:
                            #     self.q_mask[idx] = 0.
                        if lag > 0:
                            self.q_mask[idx] = np.exp(-lag * decay)

                self.q_mask = self.vec2mat(self.q_mask).mean(axis=0)

    def clear_input(self):

        self.extC = np.zeros(self.num_bindings)
        self.ext = self.C2N(self.extC)

    def set_input(self, binding_names,  # ext_vals=1.,
                  cumulative=False,  # inhib_comp=False,
                  use_type=True,  # extend=False,
                  ignore_copy_symbols=True):
        '''Set external input.'''
        # use_type -- allow A in addition to A:0

        # NOTE: Now self.estr is multiplied to curr_extC

        if not cumulative:
            # print('cleared')
            self.clear_input()

        if not isinstance(binding_names, list):
            binding_names = [binding_names]

        g = self.hg.g
        bsep = self.hg.opts['bsep']

        # if g.opts['use_pos_f']:
        if use_type:
            binding_names_new = []
            for bname in binding_names:
                # CHange ext_vals as well.
                f, r = bname.split(self.hg.opts['bsep'])
                fi_list = g.find_fillers_type(
                    f, ignore_bracket=True, ignore_copy=True,
                    ignore_pos_f=g.opts['use_pos_f'])

                fillers_target = self.hg.g.get_fillers(fi_list)
                if ignore_copy_symbols:
                    fillers_target = [f for f in fillers_target
                                      if self.hg.g.opts['copy'] not in f]

                b_list = [f + bsep + r for f in fillers_target]
                binding_names_new += b_list

            binding_names = binding_names_new

        curr_extC = np.zeros(self.num_bindings)
        idx = self.find_bindings(binding_names)
        curr_extC[idx] = 1.

        self.extC += self.estr * curr_extC
        self.ext = self.C2N(self.extC)

    def update_scale_constants(
            self, pos=0, lv=0, scale_type=None, q_only=False,
            symmetric=True):

        if scale_type is None:
            scale_type = self.opts['scale_type']

        if self.hg.opts['role_system'] == 'brick_role':
            # c = self.hg.opts['max_sent_len']
            # c = 1.
            c = self.opts['scaling_factor']
            weights = np.ones(self.num_bindings)

            if pos > 0:
                for rname in self.role_names:
                    idx = self.find_roles(rname)
                    lv0, pos0 = self.hg.roles.str2tuple(rname)

                    if scale_type == 'lv':
                        if lv > 0:
                            if lv0 >= lv:
                                weights[idx] = np.exp(-(lv0 - lv) * c)

                    # elif scale_type == 'lv_r':
                    #     if lv > 0:
                    #         if lv0 <= lv:
                    #             weights[idx] = np.exp(-(lv - lv0) * c)
                    # time may be a better label

                    elif scale_type == 'diagonal':
                        if symmetric:
                            weights[idx] = np.exp(-abs(lv0 +
                                                       pos0 - (pos + 1)) * c)
                        else:
                            if lv0 + pos0 >= pos + 1:
                                weights[idx] = np.exp(-(lv0 +
                                                        pos0 - (pos + 1)) * c)
                    elif scale_type == 'pos':
                        if symmetric:
                            weights[idx] = np.exp(-abs(pos0 - pos) * c)
                        else:
                            if pos0 >= pos:
                                weights[idx] = np.exp(-(pos0 - pos) * c)

            if q_only:
                self.scale_constants = np.ones(self.num_bindings)
                self.scale_constants_q = weights
            else:
                self.scale_constants = weights
                self.scale_constants_q = np.ones(self.num_bindings)

        else:
            # Not yet implemneted
            self.scale_constants = np.ones(self.num_bindings)

    #####################################################################
    #
    # Harmony, Harmony Gradient, and Harmony Hessian
    #
    #####################################################################

    def H(self, act=None, q=None):
        return (self.Hg(act) +
                self.Hb(act) +
                self.Hq1(act) +
                self.Hq0(act, q=q))

    def Hg(self, act=None):
        if act is None:
            act = self.act
        return 0.5 * act.T.dot(self.W).dot(act) + self.b.dot(act) + self.ext.dot(act)

    def Hb(self, act=None):
        if act is None:
            act = self.act
        diff = act - self.zeta
        return self.opts['bowl_strength'] * (-0.5 * diff.T.dot(self.Gc).dot(diff))

    def Hq0(self, act=None, q=None):
        # if act is None:
        #     act = self.act
        # actC = self.N2C(act)
        if act is None:
            actC = self.actC
        else:
            actC = self.N2C(act)
        if q is None:
            q = self.q
        q = self.extend_rvec(rvec=q)
        return -1. * np.sum(q * actC**2 * (1 - actC)**2)

    def Hq0_role(self, act=None, q=None):
        # NOTE: Check this function
        if act is None:
            actC = self.actC
        else:
            actC = self.N2C(act)
        if q is None:
            q = self.q
        q = self.extend_rvec(rvec=q)
        temp = self.scale_constants * q * actC**2 * (1 - actC)**2
        temp = self.vec2mat(temp).sum(axis=0)
        return -1. * temp

    def Hq1(self, act=None):
        # if act is None:
        #     act = self.act
        # actC = self.N2C(act)
        if act is None:
            # actC = self.actC
            actCmat = self.vec2mat(self.actC)
        else:
            actC = self.N2C(act=act)
            actCmat = self.vec2mat(actC)

        return -1. * self.opts['m'] * np.sum((np.sum(actCmat**2, axis=0) - 1)**2)

    def HGrad(self, act=None, q=None):
        return (self.HgGrad(act) +
                self.HbGrad(act) +
                self.Hq1Grad(act) +
                self.Hq0Grad(act, q=q))

    def HGradC(self, actC=None, q=None):
        # conceptual coordinates (ignoring similarity structure)
        if actC is None:
            actC = self.actC
            actCmat = self.vec2mat(actC)
        else:
            # act = self.C2N(actC=actC)
            actCmat = self.vec2mat(actC)
        if q is None:
            q = self.q

        hgrad_g = self.WC.dot(actC) + self.bC + self.extC
        hgrad_b = self.opts['bowl_strength'] * \
            (self.opts['bowl_center'] - actC)
        hgrad_q0 = -2 * self.extend_rvec(rvec=q) * \
            actC * (1 - actC) * (1 - 2 * actC)
        ssq = np.sum(actCmat ** 2, axis=0)
        hgrad_q1 = -4 * self.opts['m'] * actC * self.extend_rvec(rvec=ssq - 1)
        return (hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1)

    def HgGrad(self, act=None):
        if act is None:
            act = self.act
        return self.W.dot(act) + self.b + self.ext

    def HbGrad(self, act=None):
        if act is None:
            act = self.act
        return self.opts['bowl_strength'] * self.Gc.dot(self.zeta - act)

    def Hq0Grad(self, act=None, q=None):
        # if act is None:
        #     act = self.act
        # actC = self.N2C(act)
        if act is None:
            actC = self.actC  # check
            # actC = self.N2C()
        else:
            actC = self.N2C(act)
        if q is None:
            q = self.q
        q = self.extend_rvec(rvec=q)
        g = 2 * q * actC * (1 - actC) * (1 - 2 * actC)
        # g_{fr} vectorized
        return -1. * np.einsum('ij,i', self.C, g)

    def Hq1Grad(self, act=None):
        # if act is None:
        #     act = self.act
        # actC = self.N2C(act)
        if act is None:
            # check
            actC = self.actC
            actCmat = self.vec2mat(actC)
            # actC = self.N2C()
            # actCmat = self.vec2mat(actC)
        else:
            actC = self.N2C(act=act)
            actCmat = self.vec2mat(actC)

        term1 = np.einsum('ij->j', actCmat**2) - 1
        term2 = np.einsum('ij,ijk->jk', actCmat, self.C_reshaped)
        # == in term2 ==
        # i: filler index (f)
        # j: role index (r)
        # k: unit index (phi-rho pair)
        # h: trial index
        return -4 * self.opts['m'] * np.einsum('j,jk->k', term1, term2)

    def HHess(self, act=None, q=None):

        return (self.HgHess() +
                self.HbHess() +
                self.Hq1Hess(act) +
                self.Hq0Hess(act, q=q))

    def HgHess(self):

        return self.W

    def HbHess(self):
        # CHECK

        temp = -1. * self.opts['bowl_strength'] * np.eye(self.num_units)
        return temp

    def Hq0Hess(self, act=None, q=None):

        if act is None:
            actC = self.actC
        else:
            actC = self.N2C(act)
        if q is None:
            q = self.q

        q = self.extend_rvec(rvec=q)
        g = q * 2 * (1 - 6 * actC + 6 * actC**2)
        # i: fr
        # j: \phi \rho
        # k: \phi' \rho'
        # h: trial
        return -1. * np.einsum('ij,ik,i->kj', self.C, self.C, g)

    def Hq1Hess(self, act=None):

        if act is None:
            actC = self.actC
            actCmat = self.vec2mat(actC=actC)
        else:
            actC = self.N2C(act)
            actCmat = self.vec2mat(actC=actC)

        # ==========================
        # i: filler index (f)
        # j: role index (r)
        # k: unit index1 (phi rho)    (col index of Hessian matrix)
        # l: unit index2 (phi' rho')  (row index of Hessian matrix)
        # h: trial

        term1 = np.einsum('ij,ijk->jk', actCmat, self.C_reshaped)
        term1 = 2. * np.einsum('jl,jk->lk', term1, term1)

        term2a = np.einsum('ij->j', actCmat**2) - 1
        term2b = np.einsum('ijk,ijl->jlk', self.C_reshaped, self.C_reshaped)
        term2 = np.einsum('j,jlk->lk', term2a, term2b)

        return -4. * self.opts['m'] * (term1 + term2)

    #####################################################################
    #
    # Traces
    #
    #####################################################################

    def initialize_traces(self, trace_list='all'):
        """Create storage for traces."""

        if trace_list == 'all':
            trace_list = self.opts['trace_varnames']
        else:
            if not isinstance(trace_list, list):
                msg = "trace_list must be a list object."
                sys.exit(msg)

            var_not_in_varnames = [var for var in trace_list
                                   if var not in self.opts['trace_varnames']]
            if len(var_not_in_varnames) > 0:
                msg = ('No variable in trace_list is found. '
                       'Currently, the following variables are available:\n')
                sys.exit(msg + self.opts['trace_varnames'])

        if hasattr(self, 'traces'):
            for key in trace_list:
                self.traces[key] = list(self.traces[key])
        else:
            self.traces = {}
            for key in trace_list:
                self.traces[key] = []

            self.update_traces()

    def update_traces(self):
        """Log traces"""

        if 'act' in self.traces:
            self.traces['act'].append(self.act)
        if 'actC' in self.traces:
            self.traces['actC'].append(self.actC)
        if 'extC' in self.traces:
            self.traces['extC'].append(self.extC)
        if 'H' in self.traces:
            self.traces['H'].append(self.H())
        if 'Hg' in self.traces:
            self.traces['Hg'].append(self.Hg())
        if 'Hq0' in self.traces:
            self.traces['Hq0'].append(self.Hq0(q=np.ones(self.q.shape)))
            # self.traces['Hq0'].append(self.Hq0())
        if 'Hq1' in self.traces:
            self.traces['Hq1'].append(self.Hq1())
        if 'q' in self.traces:
            self.traces['q'].append(self.q)
        if 't' in self.traces:
            self.traces['t'].append(self.t)
        if 'T' in self.traces:
            self.traces['T'].append(self.T)
        if 'maxeig' in self.traces:
            self.traces['maxeig'].append(maxeig(self.HHess()))
        if 'Hq0_role' in self.traces:
            self.traces['Hq0_role'].append(
                self.Hq0_role(q=np.ones(self.q.shape)))
        if 'scale_constants' in self.traces:
            self.traces['scale_constants'].append(self.scale_constants)

    def finalize_traces(self):
        """Convert list objects of traces to NumPy array objects."""

        for key in self.opts['trace_varnames']:
            self.traces[key] = np.array(self.traces[key])

    #####################################################################
    #
    # Utility functions
    #
    #####################################################################

    def _compute_recommended_bowl_strength(self):
        '''Compute the recommended value of bowl strength.
        Note that the value may change depending on external input.'''

        # Condition 1: beta > eig_max to be stable
        # WC must be a symmetric matrix. So eigh() was used instead of eig()
        eigvals, eigvecs = np.linalg.eigh(self.WC)
        eig_max = max(eigvals)

        if np.sum(abs(self.bowl_center)) > 0:
            # TODO(PWC) Check there is only one binding

            # Condition 2: beta > beta1
            beta1 = -min((self.bC + self.extC) / self.bowl_center)
            # Condition 3: beta > beta2  [CHECK]
            beta2 = max(
                (self.bC + self.extC + eig_max) / (1 - self.bowl_center))
            val = max(eig_max, beta1, beta2)
        else:
            val = eig_max

        return val

    def find_bindings(self, binding_names):
        '''Return (list) of binding indices for a given binding_names (str or list).

        Args:
            binding_names: (str) binding name or
                           (list of str) binding names

        Precondition:
            binding_names must contain legitimate binding names.

        Examples:
            >>> net.find_bindings('A/0')
            >>> net.find_bindings(['A/0', 'B/0'])
        '''

        if not isinstance(binding_names, list):
            binding_names = [binding_names]
        return [self.binding_names.index(bb) for bb in binding_names]

    def find_fillers(self, filler_name):
        '''Return (list) of binding indices for a given filler_names (str or list).

        Args:
            filler_names: (str) filler name or
                          (list of str) filler names

        Precondition:
            filler_names must contain legitimate filler names.

        Examples:
            >>> net.find_fillers('A')
            >>> net.find_fillers(['A', 'B'])
        '''

        if not isinstance(filler_name, list):
            filler_name = [filler_name]

        filler_list = [bb.split('/')[0] for bb in self.binding_names]
        filler_idx = []
        for jj, filler in enumerate(filler_name):
            idx = [ii for ii, ff in enumerate(filler_list) if filler == ff]
            filler_idx += idx

        return filler_idx

    def find_roles(self, role_name):
        '''Return (list) of binding indices for a given role_names (str or list).

        Args:
            filler_names: (str) role name or
                          (list of str) role names

        Precondition:
            role_names must contain legitimate role names.

        Examples:
            >>> net.find_role('0')
            >>> net.find_role(['0', '1'])
        '''

        if not isinstance(role_name, list):
            role_name = [role_name]

        role_list = [bb.split('/')[1] for bb in self.binding_names]
        role_idx = []
        for jj, role in enumerate(role_name):
            idx = [ii for ii, rr in enumerate(role_list) if role == rr]
            role_idx += idx

        return role_idx

    def vec2mat(self, actC=None):

        if actC is None:
            actC = self.actC

        actCmat = actC.reshape(
            (self.num_fillers, self.num_roles), order='F')

        return actCmat

    def C2N(self, actC=None):
        '''Change basis: from conceptual/pattern to neural space.'''

        if actC is None:
            actC = self.actC
        return self.N.dot(actC)

    def N2C(self, act=None):
        '''Change basis: from neural to conceptual/pattern space.'''

        if act is None:
            act = self.act
        return self.C.dot(act)

    def get_ep(self, dur=10, plot=True, q=None, actC=None, method='newton'):

        q_backup = self.q.copy()

        if q is not None:
            self.q = q

        if actC is None:
            actC = self.bowl_center.copy()

        if method == 'newton':

            act = self.C2N(actC=actC)
            ep = self.newton(act=act)
            self.ep = self.N2C(ep)

        elif method == 'integration':

            T_init_backup = self.opts['T_init']
            q_rate_backup = self.opts['q_rate']

            self.opts['T_init'] = 0.
            self.opts['q_rate'] = 0.

            self.reset()
            self.set_state(mu=actC, sd=0.)
            if self.opts['use_runC']:
                self.runC(dur)
            else:
                self.run(dur)
            # if plot:
            #     self.plot_trace('actC')
            self.ep = self.actC.copy()
            self.opts['T_init'] = T_init_backup
            self.opts['q_rate'] = q_rate_backup

        self.q = q_backup.copy()

    def newton(self, act, tol=1e-12, max_iter=500):
        # Find a nearby local optimum with the Newton's method
        # Divergence case may occur.

        grad1 = self.HGrad(act)
        count = 0
        run = True
        while (np.sqrt(sum(grad1**2)) > tol) and run:
            count += 1
            jacobian1 = self.HHess(act)
            s, _, _, _ = np.linalg.lstsq(jacobian1, grad1, rcond=None)
            act = act - s
            grad1 = self.HGrad(act)
            if count == max_iter:
                run = False
                act = None

        return act

    def extend_rvec(self, rvec):
        return np.tile(
            rvec, (self.num_fillers, 1)).flatten('F')

    def extend_fvec(self, fvec):
        return np.tile(
            fvec, (1, self.num_roles)).flatten('F')

    def generate_sentence(self, min_sent_len=None, max_sent_len=None, use_type=True, add_null_input=False):

        if max_sent_len is None:
            max_sent_len = self.hg.opts['max_sent_len']

        sent, parse_tree, p = self.hg.generate_sentence(
            min_sent_len=min_sent_len, max_sent_len=max_sent_len, use_type=use_type)
        sent_input = [bname + self.hg.opts['bsep'] + '(1,{})'.format(pos + 1)
                      for pos, bname in enumerate(sent)]

        if self.hg.opts['use_same_len']:
            if add_null_input:
                num_empty_terminal_roles = self.hg.opts['max_sent_len'] - len(
                    sent)
                f_empty_type = self.hg.opts['f_empty'].split(
                    self.hg.g.opts['sep'])[0]
                sent_input += [
                    f_empty_type + self.hg.opts['bsep'] +
                    '(1,{})'.format(len(sent) + pos + 1)
                    for pos in range(num_empty_terminal_roles)]

        return sent_input, self.get_target_state(parse_tree), p

    def get_target_state(self, parse_tree):

        if self.hg.opts['role_system'] == 'brick_role':

            max_sent_len = self.hg.opts['max_sent_len']

            parse_tree_padded = []
            for lv in range(1, max_sent_len + 1):
                num_roles = max_sent_len - lv + 1
                parse_tree_padded.append(
                    [self.hg.g.opts['null']] * num_roles)

            sent_len = len(parse_tree[0])
            for lv in range(1, sent_len + 1):
                lv_id = lv - 1
                num_words = sent_len - lv + 1
                parse_tree_padded[lv_id][0:num_words] = parse_tree[lv_id]

            bnames = []
            for lv in range(1, max_sent_len + 1):
                for pos in range(1, max_sent_len - lv + 2):
                    rname = '({},{})'.format(lv, pos)
                    fname = parse_tree_padded[lv - 1][pos - 1]
                    bname = fname + self.hg.opts['bsep'] + rname
                    bnames.append(bname)

        else:
            sys.exit('Currently, only brick roles are supported.')

        return self.get_discrete_state(bnames)

    def get_discrete_state(self, bnames):

        actC = np.zeros(self.num_bindings)
        actC[self.find_bindings(bnames)] = 1.
        return actC

    def read_state(self, actC=None):
        '''Print activation state (in conceptual or neural coordinates)
        in a readable format. Pandas should be installed.

        Args:
            coord: 'C' for conceptual and 'N' for neural coordinates
            act : (1d NumPy array) activation vector in neural coord.
            actC: (1d NumPy array) activation vector in conceptual coord.

            Note that act will be ignored when coord is set to 'C'.
            Likewise, actC will be ignored when coord is set to 'N'.
        '''

        if actC is None:
            actCmat = self.vec2mat(actC=self.actC)
        else:
            actCmat = self.vec2mat(actC=actC)

        maxlen_c = max([len(rname) for rname in self.role_names]) + 2
        maxlen_r = max([len(fname) for fname in self.filler_names]) + 2
        maxlen_c = max(maxlen_c, 9)

        print('{:s}'.format('').rjust(maxlen_r), end='')
        for ri, rname in enumerate(self.role_names):
            print('{:s}'.format(rname).rjust(maxlen_c), end='')
        print('')

        for fi, fname in enumerate(self.filler_names):
            print('{:s}'.format(fname).rjust(maxlen_r), end='')
            for ri, rname in enumerate(self.role_names):
                print('{:.4f}'.format(actCmat[fi, ri]).rjust(maxlen_c), end='')
            print('')

    def read_grid_point(self, actC=None, disp=False):

        if actC is None:
            actCmat = self.vec2mat(actC=self.actC)
        else:
            actCmat = self.vec2mat(actC=actC)

        winner_idx = np.argmax(actCmat, axis=0)
        winners = [self.filler_names[ii] for ii in winner_idx]
        winners = ["%s/%s" % bb for bb in zip(winners, self.role_names)]

        if disp:
            print(winners)
        return winners

    def check_convergence(self, tol, testvar='H_increase'):

        if testvar == 'H_increase':
            self.H_now = self.H()
            if self.H_now - self.maxH > self.opts['min_H_increase']:
                self.maxH = self.H_now.copy()
                self.lapse = 0
            else:
                self.lapse += self.dt

            if self.lapse > tol:
                self.converged = True

    # =======================================================================
    # Plotting functions
    # =======================================================================

    def plot_state(self, actC=None,
                   colorbar=True, grayscale=True, disp=True, figsize=None,
                   rotate_xticklabels=False, fontsize=16):
        """Plot the activation state (conceptual coordinate) in a heatmap."""

        if actC is None:
            actC = self.actC
            actCmat = self.vec2mat(actC)
        else:
            actCmat = self.vec2mat(actC)

        heatmap(
            actCmat, xticklabels=self.role_names,
            yticklabels=self.filler_names, grayscale=grayscale,
            colorbar=colorbar, disp=disp, val_range=[0, 1],
            figsize=figsize, rotate_xticklabels=rotate_xticklabels, fontsize=fontsize)

    def plot_trace(self, varname, ylab=None, xlim=None, ylim=None, fontsize=16):
        """Plot the trace of a given variable"""

        x = self.traces['t']
        y = self.traces[varname]  # trial, time, bindings

        if ylab is None:
            ylab = varname

        plt.plot(x, y)
        plt.xlabel('Time', fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)

        plt.grid(True)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    def plot_act_trace(self,
                       fnames=None, rnames=None, legend=None,
                       xlim=None, ylim=None):

        state_trace = self.traces['actC']

        if fnames is None:
            fnames = self.filler_names

        if rnames is None:
            rnames = self.role_names

        bnames = [f + self.hg.opts['bsep'] + r
                  for f in fnames for r in rnames]

        bnames = [b for b in bnames if b in self.binding_names]

        state_trace = state_trace[:, self.find_bindings(bnames)]
        plt.plot(self.traces['t'], state_trace)

        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        if legend is not None:
            labels = copy.copy(legend)
            if '_' in labels:
                idx = labels.index('_')
                labels[idx] = '.'
            plt.legend(labels)

    def plot_tree(self, actC=None,
                  panel_size=None, add_line=True, add_point=False,
                  skip_bracketed=False, figsize=None, add_text=True,
                  savefilename=None, gradC=None, add_grad=False, grad_scale=0.2,
                  linecolor='lightblue', linewidth=4, fontsize=10):

        def add_panel(pos, size, color='black'):
            x1 = pos[0]
            x2 = pos[0] + size[0]
            y1 = pos[1]
            y2 = pos[1] + size[1]
            xval = [x1, x2, x2, x1, x1]
            yval = [y1, y1, y2, y2, y1]
            plt.plot(xval, yval, linestyle='-', color=color)

        num_print_labels = self.num_fillers
        if isinstance(add_text, int):
            num_print_labels = copy.copy(add_text)
            add_text = True

        if self.hg.opts['use_hnf']:
            skip_bracketed = skip_bracketed
        else:
            skip_bracketed = True

        xoffset = 0.5
        max_sent_len = self.hg.roles.opts['max_sent_len']

        if actC is None:
            actC = self.actC
            act = self.act
            actCmat = self.vec2mat(actC=actC)
        else:
            act = self.C2N(actC=actC)
            actCmat = self.vec2mat(actC=actC)

        if gradC is None:
            gradC = self.N2C(self.HGrad(act=act))

        gradCmat = self.vec2mat(actC=gradC)

        if panel_size is None:
            panel_width = self.num_fillers
            panel_height = 1.

        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        for lv in range(1, max_sent_len + 1):

            if (not skip_bracketed) and (lv > 1):
                # plot bracketed roles
                for pos in range(1, max_sent_len - lv + 2):

                    x1 = (pos - 1) * panel_width + (lv - 1) * 0.5 * panel_width
                    y1 = (lv - 1) * (2 * panel_height) - panel_height
                    add_panel((x1, y1), (panel_width, panel_height),
                              color='darkgray')

                    rname = '({:d},{:d})'.format(-lv, pos)
                    rind = self.role_names.index(rname)
                    curr_actC = actCmat[:, rind]
                    curr_gradC = gradCmat[:, rind]

                    idx_temp = np.argsort(curr_actC)[::-1]
                    for ii in idx_temp[:num_print_labels]:
                        if add_text:
                            plt.text(x1 + ii + xoffset,
                                     y1 + curr_actC[ii] * 0.5,
                                     self.filler_names[ii],
                                     fontsize=fontsize * (curr_actC[ii] + 0.3),
                                     color='k', fontweight='bold',
                                     horizontalalignment='center',
                                     verticalalignment='center')

                    for ii in range(self.num_fillers):
                        # if add_text:
                        #     plt.text(x1 + ii + xoffset,
                        #              y1 + curr_actC[ii] * 0.5,
                        #              self.filler_names[ii],
                        #              fontsize=fontsize * (curr_actC[ii] + 0.3),
                        #              color='k', fontweight='bold',
                        #              horizontalalignment='center',
                        #              verticalalignment='center')

                        if add_point:
                            plt.plot(x1 + ii + xoffset,
                                     y1 + curr_actC[ii] * 0.9,
                                     'o', color='lightblue')

                        if add_line:
                            plt.plot([x1 + ii + xoffset] * 2,
                                     [y1, y1 + curr_actC[ii] * 0.9],
                                     linestyle='-', linewidth=linewidth,
                                     color=linecolor)

                        if add_grad:
                            plt.arrow(x1 + ii + xoffset,
                                      y1 + curr_actC[ii] * 0.9,
                                      0,
                                      curr_gradC[ii] * grad_scale,
                                      width=0.01, head_width=0.3, head_length=0.05,
                                      facecolor='lightblue', edgecolor='lightblue')

            for pos in range(1, max_sent_len - lv + 2):
                x1 = (pos - 1) * panel_width + (lv - 1) * 0.5 * panel_width

                if skip_bracketed:
                    y1 = (lv - 1) * panel_height
                else:
                    y1 = (lv - 1) * (2 * panel_height)

                add_panel((x1, y1), (panel_width, panel_height),
                          color='darkgray')

                rname = '({:d},{:d})'.format(lv, pos)
                rind = self.role_names.index(rname)
                curr_actC = actCmat[:, rind]
                curr_gradC = gradCmat[:, rind]

                idx_temp = np.argsort(curr_actC)[::-1]
                for ii in idx_temp[:num_print_labels]:
                    if add_text:
                        plt.text(x1 + ii + xoffset,
                                 y1 + curr_actC[ii] * 0.5,
                                 self.filler_names[ii],
                                 fontsize=fontsize * (curr_actC[ii] + 0.3),
                                 color='k', fontweight='bold',
                                 horizontalalignment='center',
                                 verticalalignment='center')

                for ii in range(self.num_fillers):
                    # if add_text:
                    #     plt.text(x1 + ii + xoffset,
                    #              y1 + curr_actC[ii] * 0.5,
                    #              self.filler_names[ii],
                    #              fontsize=fontsize * (curr_actC[ii] + 0.3),
                    #              color='k', fontweight='bold',
                    #              horizontalalignment='center',
                    #              verticalalignment='center')

                    if add_point:
                        plt.plot(x1 + ii + xoffset,
                                 y1 + curr_actC[ii] * 0.9,
                                 'o', color='lightblue')

                    if add_line:
                        plt.plot([x1 + ii + xoffset] * 2,
                                 [y1, y1 + curr_actC[ii] * 0.9],
                                 linestyle='-', linewidth=linewidth,
                                 color=linecolor)

                    if add_grad:
                        plt.arrow(x1 + ii + xoffset,
                                  y1 + curr_actC[ii] * 0.9,
                                  0,
                                  curr_gradC[ii] * grad_scale,
                                  width=0.01, head_width=0.3, head_length=0.05,
                                  facecolor='lightblue', edgecolor='lightblue')

        if skip_bracketed:
            ymax = 0.1 + max_sent_len * panel_height
        else:
            ymax = 0.1 + max_sent_len * panel_height * 2 - panel_height

        plt.xlim(-0.1, 0.1 + max_sent_len * panel_width)
        plt.ylim(-0.1, ymax)
        plt.axis('off')
        if savefilename is not None:
            plt.tight_layout()
            plt.savefig(savefilename)
        else:
            plt.show()

    def plot_tree_actC_trace(
            self, actC_trace=None, t_trace=None, downsampling=20,
            windows=None, add_q_trace=False, top_k=None,
            panel_size=None, skip_bracketed=False, figsize=None, print_winner=True,
            savefilename=None, linecolor='lightblue', linewidth=4, fontsize=10):

        def add_panel(pos, size, color='black'):
            x1 = pos[0]
            x2 = pos[0] + size[0]
            y1 = pos[1]
            y2 = pos[1] + size[1]
            xval = [x1, x2, x2, x1, x1]
            yval = [y1, y1, y2, y2, y1]
            plt.plot(xval, yval, linestyle='-', color=color)

        if self.hg.opts['use_hnf']:
            skip_bracketed = skip_bracketed
        else:
            skip_bracketed = True

        xoffset = 0.5
        max_sent_len = self.hg.roles.opts['max_sent_len']

        if actC_trace is None:
            actC_trace = self.traces['actC'][::downsampling, :]
            t_trace = self.traces['t'][::downsampling]
            q_trace = self.traces['q'][::downsampling, :]

        if panel_size is None:
            panel_width = self.num_fillers
            panel_height = 1.

        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        if windows is not None:
            windows0 = windows.copy()
            windows = []
            for window in windows0:
                win = np.array(window[0]) * panel_width / max(t_trace)
                windows.append((win, window[1]))

        for lv in range(1, max_sent_len + 1):

            if (not skip_bracketed) and (lv > 1):
                # plot bracketed roles
                for pos in range(1, max_sent_len - lv + 2):

                    x1 = (pos - 1) * panel_width + (lv - 1) * 0.5 * panel_width
                    y1 = (lv - 1) * (2 * panel_height) - panel_height
                    add_panel((x1, y1), (panel_width, panel_height),
                              color='darkgray')

                    rname = '({:d},{:d})'.format(-lv, pos)
                    rind = self.find_roles(rname)
                    curr_actC_trace = actC_trace[:, rind]

                    plt.plot(x1 + t_trace * panel_width / max(t_trace),
                             y1 + curr_actC_trace[:, ii],
                             linestyle='-', linewidth=linewidth)

            for pos in range(1, max_sent_len - lv + 2):
                x1 = (pos - 1) * panel_width + (lv - 1) * 0.5 * panel_width

                if skip_bracketed:
                    y1 = (lv - 1) * panel_height
                else:
                    y1 = (lv - 1) * (2 * panel_height)

                add_panel((x1, y1), (panel_width, panel_height),
                          color='darkgray')

                rname = '({:d},{:d})'.format(lv, pos)
                rind = self.find_roles(rname)
                curr_actC_trace = actC_trace[:, rind]
                fidx = np.argsort(curr_actC_trace.sum(axis=0))[::-1]

                if top_k is None:
                    plt.plot(x1 + t_trace * panel_width / max(t_trace),
                             y1 + curr_actC_trace,
                             linestyle='-', linewidth=linewidth)
                else:
                    plt.plot(x1 + t_trace * panel_width / max(t_trace),
                             y1 + curr_actC_trace[:, fidx[:top_k]],
                             linestyle='-', linewidth=linewidth)

                if add_q_trace:
                    rind2 = self.role_names.index(rname)
                    curr_q_trace = q_trace[:, rind2] / \
                        self.opts['q_max']   # normalize
                    plt.plot(x1 + t_trace * panel_width / max(t_trace),
                             y1 + curr_q_trace, color='grey',
                             linestyle='-.', linewidth=linewidth)

                labs = '{}\n----\n{}'.format(
                    self.filler_names[fidx[0]], self.filler_names[fidx[1]])

                if print_winner:
                    plt.text(x1 + panel_width * 0.8,
                             y1 + panel_height * 0.5,
                             labs,
                             fontsize=fontsize,
                             color='k',  # fontweight='bold',
                             horizontalalignment='center',
                             verticalalignment='center')

                if windows is not None:

                    roleset_id = pos + lv - 1

                    for window in windows:

                        plt.plot([x1 + window[0][1], x1 + window[0][1]],
                                 [y1, y1 + 1.0],
                                 color='darkgray', linestyle=':')
                        plt.text(x1 + window[0][0] + (window[0][1]-window[0][0])/2,
                                 y1 + 0.85,
                                 window[1],
                                 horizontalalignment='center',
                                 fontsize=fontsize)

                if windows is not None:

                    roleset_id = pos + lv - 1

                    if roleset_id <= len(windows):
                        window = windows[roleset_id - 1]
                        add_window = True

                    else:
                        # wrap-up
                        window = [[windows[-1][0][1], panel_width]]
                        add_window = False

                    if add_window:
                        plt.gca().add_patch(Rectangle((x1 + window[0][0], y1),
                                            window[0][1] - window[0][0],
                                            panel_height,
                                            facecolor='red',
                                            alpha=0.05))

        if skip_bracketed:
            ymax = 0.1 + max_sent_len * panel_height
        else:
            ymax = 0.1 + max_sent_len * panel_height * 2 - panel_height

        plt.xlim(-0.1, 0.1 + max_sent_len * panel_width)
        plt.ylim(-0.1, ymax)
        plt.axis('off')
        if savefilename is not None:
            plt.tight_layout()
            plt.savefig(savefilename)
        else:
            plt.show()

    def plot_treelet_actC_trace(
            self, actC_trace=None, downsampling=20,
            windows=None, num_treelets=3,
            panel_size=None, skip_bracketed=False, figsize=None, print_winner=True,
            savefilename=None, linecolor='lightblue', linewidth=4, fontsize=10):

        def add_panel(pos, size, color='black'):
            x1 = pos[0]
            x2 = pos[0] + size[0]
            y1 = pos[1]
            y2 = pos[1] + size[1]
            xval = [x1, x2, x2, x1, x1]
            yval = [y1, y1, y2, y2, y1]
            plt.plot(xval, yval, linestyle='-', color=color)

        if self.hg.opts['use_hnf']:
            skip_bracketed = skip_bracketed
        else:
            skip_bracketed = True

        xoffset = 0.5
        max_sent_len = self.hg.roles.opts['max_sent_len']

        if actC_trace is None:
            actC_trace = self.traces['actC'][::downsampling, :]
            t_trace = self.traces['t'][::downsampling]

        rules0 = self.hg.g.get_rules()
        rules = []
        for rule in rules0:
            if rule not in rules:
                rules.append(rule)

        labs = create_rule_labels(rules)

    #     idx = (net.traces['t'] >= tmin) * (net.traces['t'] <= tmax)
    #     actC_trace = net.traces['actC'][idx, :]

        if panel_size is None:
            panel_width = self.num_fillers
            panel_height = 1.

        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        if windows is not None:
            windows0 = windows.copy()
            windows = []
            for window in windows0:
                win = np.array(window[0]) * panel_width / max(t_trace)
                windows.append((win, window[1]))

        for lv in range(2, max_sent_len + 1):

            #         if (not skip_bracketed) and (lv > 1):
            #             # plot bracketed roles
            #             for pos in range(1, max_sent_len - lv + 2):

            #                 x1 = (pos - 1) * panel_width + (lv - 1) * 0.5 * panel_width
            #                 y1 = (lv - 1) * (2 * panel_height) - panel_height
            #                 add_panel((x1, y1), (panel_width, panel_height),
            #                           color='darkgray')

            #                 rname = '({:d},{:d})'.format(-lv, pos)
            #                 rind = self.find_roles(rname)
            #                 curr_actC_trace = actC_trace[:, rind]

            #                 plt.plot(x1 + t_trace * panel_width / max(t_trace),
            #                     y1 + curr_actC_trace[:, ii],
            #                     linestyle='-', linewidth=linewidth)

            for pos in range(1, max_sent_len - lv + 2):
                x1 = (pos - 1) * panel_width + (lv - 1) * 0.5 * panel_width

                if skip_bracketed:
                    y1 = (lv - 1) * panel_height
                else:
                    y1 = (lv - 1) * (2 * panel_height)

                add_panel((x1, y1), (panel_width, panel_height),
                          color='darkgray')

                rname = '({},{})'.format(lv, pos)
                dp_all = compute_treelet_act_trace(
                    self, actC_trace, rules, rname)

                temp = np.argsort(dp_all.sum(axis=0))
                focus_idx = temp[::-1][:num_treelets]
                labs_focus = [labs[ii] for ii in focus_idx]

    #             plt.plot(net.traces['t'][idx][::downsampling], dp_all[::downsampling, focus_idx])
    #             plt.legend(labs_focus)
    #             plt.ylim(0, 1)
    #             plt.xlabel('Time')
    #             plt.ylabel('Treelet activation at {}'.format(rname))
    #             plt.grid()
    #             plt.show()

    #             rname = '({:d},{:d})'.format(lv, pos)
    #             rind = self.find_roles(rname)
    #             curr_actC_trace = actC_trace[:, rind]

    #             fidx = np.argsort(curr_actC_trace.sum(axis=0))[::-1]

    #             dp_all[:, focus_idx]

                plt.plot(x1 + t_trace * panel_width / max(t_trace),
                         y1 + dp_all[:, focus_idx],
                         linestyle='-', linewidth=linewidth)

                yoff = 0.1
                for lbi, lb in enumerate(labs_focus):
                    plt.text(x1 + panel_width * 0.5,
                             y1 + panel_height * 0.5 - yoff * lbi,
                             lb,
                             fontsize=8,
                             color='k',
                             horizontalalignment='left',
                             verticalalignment='center')

    #             labs = '{}\n----\n{}'.format(
    #                 self.filler_names[fidx[0]], self.filler_names[fidx[1]])

    #             if print_winner:
    #                 plt.text(x1 + panel_width * 0.8,
    #                          y1 + panel_height * 0.5,
    #                          labs_focus, #labs,
    #                          fontsize=fontsize,
    #                          color='k', #fontweight='bold',
    #                          horizontalalignment='center',
    #                          verticalalignment='center')

                if windows is not None:

                    roleset_id = pos + lv - 1

                    if roleset_id <= len(windows):
                        window = windows[roleset_id - 1]
                        add_window = True
                    else:
                        window = [[windows[-1][0][1], panel_width]]
                        add_window = False

                    if add_window:
                        plt.gca().add_patch(Rectangle((x1 + window[0][0], y1),
                                            window[0][1] - window[0][0],
                                            panel_height,
                                            facecolor='red',
                                            alpha=0.05))

        if skip_bracketed:
            ymax = 0.1 + max_sent_len * panel_height
        else:
            ymax = 0.1 + max_sent_len * panel_height * 2 - panel_height

        plt.xlim(-0.1, 0.1 + max_sent_len * panel_width)
        plt.ylim(-0.1, ymax)
        plt.axis('off')
        if savefilename is not None:
            plt.tight_layout()
            plt.savefig(savefilename)
        else:
            plt.tight_layout()
            plt.show()

    def plot_tree2(self, actC=None,
                   figsize=None, scale=1, savefilename=None):

        from matplotlib.patches import Circle

        def add_node(x, y, strength, radius=0.3, scale=scale):
            return Circle((x, y), radius, facecolor='none',
                          edgecolor=cm(1 - strength),
                          linewidth=scale * 2 * strength)

        def add_edge(xy1, xy2, angle, strength, radius=0.3, scale=scale):
            x1, y1 = xy1
            x2, y2 = xy2
            x1 += radius * np.cos(angle)
            y1 += radius * np.sin(angle)
            x2 += 1.3 * radius * np.cos(np.pi + angle)
            y2 += 1.3 * radius * np.sin(np.pi + angle)
            plt.plot([x1, x2], [y1, y2],
                     color=cm(1 - strength),
                     linewidth=scale * 2 * strength)

        radius = 0.25
        cm = plt.cm.gray
        max_sent_len = self.hg.opts['max_sent_len']

        if figsize is None:
            figsize = (scale * max_sent_len, scale * 0.5 * max_sent_len)

        if self.hg.g.opts['use_hnf']:
            return None
        if self.hg.opts['role_system'] != 'brick_role':
            return None

        if actC is None:
            actC = self.actC

        if self.hg.g.opts['use_pos_f']:

            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            for role in self.role_names:
                lv, pos = self.hg.roles.str2tuple(role)
                xx = pos + (lv - 1) * 0.5
                # lv_diff = 0.5 * np.tan(np.pi / 4)
                lv_diff = 0.5
                yy = 1 + (lv - 1) * lv_diff
                state = actC[self.find_roles(role)]
                fname = self.filler_names[np.argmax(state)]
                maxact = np.max(state) / np.max(actC)
                if self.hg.g.opts['sep'] in fname:
                    f, r = fname.split(self.hg.g.opts['sep'])
                else:
                    f = fname
                    r = None

                if r is not None:
                    if r == '0':
                        add_edge((xx, yy), (xx + 0.5, yy + lv_diff),
                                 radius=radius, angle=np.pi / 4,
                                 strength=maxact, scale=scale)
                    elif r == '1':
                        add_edge((xx, yy), (xx - 0.5, yy + lv_diff),
                                 radius=radius, angle=3 * np.pi / 4,
                                 strength=maxact, scale=scale)

                ax.add_patch(
                    add_node(xx, yy, radius=radius,
                             strength=maxact, scale=scale))
                plt.text(xx, yy, f, fontsize=scale * 8, color='k',
                         horizontalalignment='center',
                         verticalalignment='center')

            plt.xlim(0, max_sent_len + 1)
            plt.ylim(0, 0.5 * max_sent_len + 1)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()

        else:
            # opts['pos_f'] = False

            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            for role in self.role_names:
                lv, pos = self.hg.roles.str2tuple(role)
                xx = pos + (lv - 1) * 0.5
                # lv_diff = 0.5 * np.tan(np.pi / 4)
                lv_diff = 0.5
                yy = 1 + (lv - 1) * lv_diff
                state = actC[self.find_roles(role)]
                fname = self.filler_names[np.argmax(state)]
                maxact = np.max(state) / np.max(actC)

                if lv < max_sent_len:
                    mothers = self.hg.roles.get_mothers(role)
                    # print(lv, pos, mothers)
                    fname_l = None
                    fname_r = None
                    if len(mothers['l']) > 0 and \
                       mothers['l'][0] in self.role_names:
                        # mrole_l = mothers['l'][0]
                        state_l = actC[self.find_roles(mothers['l'][0])]
                        fname_l = self.filler_names[np.argmax(state_l)]
                    if len(mothers['r']) > 0 and \
                       mothers['r'][0] in self.role_names:
                        # mrole_r = mothers['r'][0]
                        state_r = actC[self.find_roles(mothers['r'][0])]
                        fname_r = self.filler_names[np.argmax(state_r)]

                    for rule in self.hg.rules:
                        if fname_l is not None and \
                           rule['f1'] == fname and \
                           rule['f2'] == fname_l and \
                           rule['rel'] == 'l':
                            # r = '1'
                            add_edge((xx, yy), (xx - 0.5, yy + lv_diff),
                                     radius=radius, angle=3 * np.pi / 4,
                                     strength=maxact, scale=scale)

                        if fname_r is not None and \
                           rule['f1'] == fname and \
                           rule['f2'] == fname_r and \
                           rule['rel'] == 'r':
                            # r = '0'
                            add_edge((xx, yy), (xx + 0.5, yy + lv_diff),
                                     radius=radius, angle=np.pi / 4,
                                     strength=maxact, scale=scale)

                    # if self.hg.g.opts['sep'] in fname:
                    #     f, r = fname.split(self.hg.g.opts['sep'])
                    # else:
                    #     f = fname
                    #     r = None

                    # if r is not None:
                    #     if r == '0':
                    #         add_edge((xx, yy), (xx + 0.5, yy + lv_diff),
                    #                  radius=radius, angle=np.pi / 4,
                    #                  strength=maxact, scale=scale)
                    #     # elif r == '1':
                    #     if r == '1':
                    #         add_edge((xx, yy), (xx - 0.5, yy + lv_diff),
                    #                  radius=radius, angle=3 * np.pi / 4,
                    #                  strength=maxact, scale=scale)

                ax.add_patch(
                    add_node(xx, yy, radius=radius,
                             strength=maxact, scale=scale))
                plt.text(xx, yy, fname, fontsize=scale * 8, color='k',
                         horizontalalignment='center',
                         verticalalignment='center')

            plt.xlim(0, max_sent_len + 1)
            plt.ylim(0, 0.5 * max_sent_len + 1)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()

        if savefilename is not None:
            print('Saving figure')
            plt.savefig(savefilename)
        else:
            plt.show()

    # =======================================================================
    # Training
    # =======================================================================

    def generate_corpus(self, nsamples=5000,
                        min_sent_len=None, max_sent_len=None,
                        use_type=True, use_freq=True):

        if max_sent_len is None:
            max_sent_len = self.hg.opts['max_sent_len']

        sentences = []
        targets = []
        pvals = []
        counts = []
        for _ in range(nsamples):
            sentence, target, p = self.generate_sentence(
                min_sent_len=min_sent_len,
                max_sent_len=max_sent_len,
                use_type=use_type)
            if sentence in sentences:
                idx = sentences.index(sentence)
                counts[idx] += 1
            else:
                sentences.append(sentence)
                targets.append(list(target))
                pvals.append(p)
                counts.append(1)

        if use_freq:
            counts = np.array(counts)
            pvals = counts / counts.sum()

        idx = np.argsort(pvals)[::-1]
        sentences = [sentences[si] for si in idx]
        pvals = np.array([pvals[si] for si in idx])
        targets = np.array([targets[si] for si in idx])
        counts = np.array([counts[si] for si in idx])

        self.corpus = {'sentence': sentences,
                       'target': targets,  # targets_unique,
                       'count': counts,
                       'prob_sent': pvals}

        # self.add_corpus_stat()

    def subset_corpus(self, bnames):
        # NOTE: Currently, filler name types (e.g., A instead of A:0)
        # are assumed to be used in binding names in both bnames and self.corpus['sentence'].

        if not isinstance(bnames, list):
            sys.exit('`bnames` should be a list object.')

        nsent = len(self.corpus['sentence'])

        idx = []
        for si, sent in enumerate(self.corpus['sentence']):
            if set(bnames).issubset(set(sent)):
                idx.append(si)

        corpus = {}
        for key in self.corpus:
            corpus[key] = [self.corpus[key][si]
                           for si in range(nsent)
                           if si in idx]
            if key != 'sentence':
                corpus[key] = np.array(corpus[key])

        # Normalize probabilities
        corpus['prob_sent'] /= corpus['prob_sent'].sum()

        return corpus

    def train_parallel_parsing(self, add_null_input=True):

        # parallel_parser_nsamples = round(self.train_opts['num_trials'] * self.train_opts['parallel_parser_sample_ratio'])
        parallel_parser_nsamples = round(
            self.train_opts['num_trials'] * self.train_opts['parallel_parser_num_trials'])

        dWC = np.zeros((self.num_bindings, self.num_bindings))
        dbC = np.zeros(self.num_bindings)
        scale_dWC = self.train_opts['parallel_parser_dWC_scaler']

        n_sent = len(self.corpus['sentence'])
        f_empty_type = self.hg.g.get_types(self.hg.g.opts['f_empty'])[0]

        if self.train_opts['parallel_parser_sample_uniform']:
            p = np.ones(n_sent) / n_sent
        else:
            p = self.corpus['prob_sent']

        # print('Parsing {} sentences'.format(parallel_parser_nsamples))
        idx_sent = np.random.choice(
            n_sent, parallel_parser_nsamples, replace=True, p=p)
        acc = 0.

        for idx in idx_sent:

            # idx = np.random.choice(n_sent, 1, replace=False, p=p)[0]
            sent = self.corpus['sentence'][idx]
            corpus0 = {}
            corpus0['sentence'] = [sent]
            corpus0['target'] = self.corpus['target'][idx][None, :]
            corpus0['prob_sent'] = np.array([0.999])
            stat_P = self.get_corpus_stat(corpus0)

            if add_null_input and (len(sent) < self.hg.opts['max_sent_len']):
                sent += [f_empty_type + self.hg.opts['bsep'] + '(1,{})'.format(ii)
                         for ii in range(len(sent) + 1, self.hg.opts['max_sent_len'] + 1)]

            self.reset(mu=self.ep, sd=self.train_opts['init_noise_mag'])
            self.set_input(sent)
            self.run_wrapup(update_q_discrete=False, clear_input=False)
            self.set_discrete_state(self.read_grid_point())

            if np.allclose(self.corpus['target'][idx], self.actC):
                acc += 1.

            corpus1 = {}
            corpus1['sentence'] = sent
            corpus1['target'] = self.actC[None, :]
            corpus1['prob_sent'] = np.array([0.999])
            stat_Q = self.get_corpus_stat(corpus1)

            self.clear_input()
            extC_token = self.extC.astype(bool).astype(int)
            kl_curr, xent_curr, err, err_log = self.cost(stat_P, stat_Q)
            dWC_curr, destr_curr, dq_curr, dbC_curr = self.cost_grad(
                err, extC_token)
            dWC += dWC_curr * scale_dWC
            dbC += dbC_curr * scale_dWC

        return dWC, acc / len(idx_sent), dbC

    def get_corpus_stat(self, corpus):
        # No need for corpus['sentence']

        stat = {}
        stat['trees'] = {}
        stat['treelets'] = {}
        stat['binding_pairs'] = {}
        stat['bindings'] = {}
        # stat['bindings'] = np.zeros(self.num_bindings)
        # binding probability
        # prob_sent: change format
        for si, state in enumerate(corpus['target']):

            p = corpus['prob_sent'][si]
            # stat['bindings'] += p * state
            gp_key = tuple(np.where(state == 1)[0])
            stat['trees'][gp_key] = p

            for bid in list(gp_key):
                if bid not in stat['bindings']:
                    stat['bindings'][bid] = p
                else:
                    stat['bindings'][bid] += p

            for role in self.role_names:
                if not self.hg.roles.is_terminal(role):
                    daughters = self.hg.roles.get_daughters(role)
                    l = daughters['l']
                    r = daughters['r']
                    idx = self.find_roles(role)
                    idx_l = self.find_roles(l)
                    idx_r = self.find_roles(r)
                    f_m = np.argmax(state[idx])
                    f_l = np.argmax(state[idx_l])
                    f_r = np.argmax(state[idx_r])
                    treelet = (idx[f_m], idx_l[f_l], idx_r[f_r])
                    pair_l = (idx[f_m], idx_l[f_l])
                    pair_r = (idx[f_m], idx_r[f_r])

                    if treelet in stat['treelets']:
                        stat['treelets'][treelet] += p
                    else:
                        stat['treelets'][treelet] = p

                    if pair_l in stat['binding_pairs']:
                        stat['binding_pairs'][pair_l] += p
                    else:
                        stat['binding_pairs'][pair_l] = p

                    if pair_r in stat['binding_pairs']:
                        stat['binding_pairs'][pair_r] += p
                    else:
                        stat['binding_pairs'][pair_r] = p

        return stat

    def get_prefix(self):
        prefix = []
        for sent in self.corpus['sentence']:
            for si in range(len(sent)):
                curr_prefix = sent[:(si+1)]
                if curr_prefix not in prefix:
                    prefix.append(curr_prefix)

        # sort by length
        prefix.sort(key=len)

        return prefix

    def add_corpus_stat(self):

        if not hasattr(self, 'corpus'):
            sys.exit(
                'Create the corpus first by running the following command:\nGscNet.generate_corpus().\n')

        elif 'prob_sent' not in self.corpus:
            sys.exit(
                'Add a key named "prob_sent" with an appropriate probability distribution as its value to the dictionary, GscNet.corpus.')

        self.corpus['prob'] = {}
        self.corpus['prob']['bindings'] = np.zeros(self.num_bindings)
        self.corpus['prob']['binding_pairs'] = {}
        self.corpus['prob']['treelets'] = {}
        self.corpus['prob']['trees'] = {}
        self.corpus['prob']['bindings_0'] = np.zeros(self.num_bindings)
        self.corpus['prob']['binding_pairs_0'] = {}
        self.corpus['prob']['treelets_0'] = {}
        self.corpus['prob']['trees_0'] = {}

        # binding probability
        # prob_sent: change format
        prob_bindings = np.zeros(self.num_bindings)
        for si, state in enumerate(self.corpus['target']):
            p = self.corpus['prob_sent'][si]
            prob_bindings += p * state

            gp_key = tuple(np.where(state == 1)[0])
            self.corpus['prob']['trees'][gp_key] = p
            self.corpus['prob']['trees_0'][gp_key] = p

        self.corpus['prob']['bindings'] = prob_bindings
        self.corpus['prob']['bindings_0'] = prob_bindings

        self.corpus['prob']['treelets'] = {}
        self.corpus['prob']['binding_pairs'] = {}
        for role in self.role_names:
            if not self.hg.roles.is_terminal(role):
                self.corpus['prob']['treelets'][role] = {}
                self.corpus['prob']['binding_pairs'][role] = {'l': {}, 'r': {}}
                for si, state in enumerate(self.corpus['target']):
                    p = self.corpus['prob_sent'][si]
                    daughters = self.hg.roles.get_daughters(role)
                    l = daughters['l']
                    r = daughters['r']
                    idx = self.find_roles(role)
                    idx_l = self.find_roles(l)
                    idx_r = self.find_roles(r)
                    f_m = np.argmax(state[idx])
                    f_l = np.argmax(state[idx_l])
                    f_r = np.argmax(state[idx_r])
                    treelet = (idx[f_m], idx_l[f_l], idx_r[f_r])
                    pair_l = (idx[f_m], idx_l[f_l])
                    pair_r = (idx[f_m], idx_r[f_r])

                    if treelet in self.corpus['prob']['treelets'][role]:
                        self.corpus['prob']['treelets'][role][treelet] += p
                    else:
                        self.corpus['prob']['treelets'][role][treelet] = p

                    if pair_l in self.corpus['prob']['binding_pairs'][role]['l']:
                        self.corpus['prob']['binding_pairs'][role]['l'][pair_l] += p
                    else:
                        self.corpus['prob']['binding_pairs'][role]['l'][pair_l] = p

                    if pair_r in self.corpus['prob']['binding_pairs'][role]['r']:
                        self.corpus['prob']['binding_pairs'][role]['r'][pair_r] += p
                    else:
                        self.corpus['prob']['binding_pairs'][role]['r'][pair_r] = p

                    if treelet in self.corpus['prob']['treelets_0']:
                        self.corpus['prob']['treelets_0'][treelet] += p
                    else:
                        self.corpus['prob']['treelets_0'][treelet] = p

                    if pair_l in self.corpus['prob']['binding_pairs_0']:
                        self.corpus['prob']['binding_pairs_0'][pair_l] += p
                    else:
                        self.corpus['prob']['binding_pairs_0'][pair_l] = p

                    if pair_r in self.corpus['prob']['binding_pairs_0']:
                        self.corpus['prob']['binding_pairs_0'][pair_r] += p
                    else:
                        self.corpus['prob']['binding_pairs_0'][pair_r] = p

    def generate(self, log_trace=False):
        '''Generate and return a grid point (list)'''

        self.reset()
        self.set_state(mu=self.ep, sd=self.train_opts['init_noise_mag'])
        if self.opts['use_runC']:
            self.runC(self.train_opts['dur'], log_trace=log_trace)
        else:
            self.run(self.train_opts['dur'], log_trace=log_trace)
        gp = self.read_grid_point(disp=False)
        return gp

    def check_grammaticality(self, gp=None):
        '''Return the grammaticality (bool) of a grid point gp (list)'''

        if gp is None:
            gp = self.read_grid_point()

        self.set_discrete_state(gp)
        # return np.any(self.corpus['target'].dot(self.actC) == self.num_roles)
        check = (2 * self.corpus['target'] - 1).dot(
            2 * self.actC - 1) == self.num_bindings
        gram = np.any(check)
        if gram:
            idx = np.where(check)[0][0]
        else:
            idx = None
        return gram, idx

    def average_filler_bias(self):

        if self.opts['use_second_order_bias']:

            bC = np.diag(self.WC).copy()
            WC0 = self.WC - np.diag(bC)

            # bC = np.tile(self.vec2mat(bC).mean(axis=1), self.num_roles)
            # self.WC = WC0 + np.diag(bC)
            # self._set_weights()

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:

                    roots = self.hg.g.get_roots() + [self.hg.g.opts['f_root']]
                    rid = self.find_roles(self.role_names[-1])
                    rid = [ii for ii in rid if ii in self.find_fillers(roots)]
                    # rid = self.find_roles(self.role_names[-1])  # top brick role
                    bC[rid] -= 2.   # NOTE: second-order bias = 2 * first-order bias

            idx = self.train_opts['idx_mask_bias2']

            mask = np.ones(self.num_bindings)
            mask[idx] = np.nan
            fbias_avg = np.nanmean(self.vec2mat(bC * mask), axis=1)
            bC_new = np.tile(fbias_avg, self.num_roles)
            bC_new[idx] = bC[idx]

            if 'free_update_null' in self.train_opts:
                if self.train_opts['free_update_null']:
                    idx_null = self.find_fillers(self.hg.g.opts['null'])
                    bC_new[idx_null] = bC[idx_null]

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:
                    bC_new[rid] += 2.

            self.WC = WC0 + np.diag(bC_new)
            self._set_weights()

        else:

            # self.bC = np.tile(
            #     self.vec2mat(self.bC).mean(axis=1), self.num_roles)
            # self._set_biases()

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:
                    roots = self.hg.g.get_roots() + [self.hg.g.opts['f_root']]
                    rid = self.find_roles(self.role_names[-1])
                    rid = [ii for ii in rid if ii in self.find_fillers(roots)]
                    # rid = self.find_roles(self.role_names[-1])  # top brick role
                    bC[rid] -= 1.

            idx = self.train_opts['idx_mask_bias1']
            mask = np.ones(self.num_bindings)
            mask[idx] = np.nan
            fbias_avg = np.nanmean(self.vec2mat(bC * mask), axis=1)
            bC_new = np.tile(fbias_avg, self.num_roles)
            bC_new[idx] = bC[idx]

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:
                    bC_new[rid] += 1.

            self.bC = bC_new
            self._set_weights()

    def average_weight(self):

        WC_L = 0.
        WC_R = 0.
        WC_S = 0.   # sister roles
        count_L = 0
        count_R = 0
        count_S = 0
        for role in self.role_names:
            if not self.hg.roles.is_terminal(role):
                daughters = self.hg.roles.get_daughters(role)
                daughter_l = daughters['l'][0]
                daughter_r = daughters['r'][0]
                idx = self.find_roles(role)
                idx_l = self.find_roles(daughter_l)
                idx_r = self.find_roles(daughter_r)
                count_L += 1
                count_R += 1
                count_S += 1
                WC_L += self.WC[np.ix_(idx, idx_l)]
                WC_R += self.WC[np.ix_(idx, idx_r)]
                WC_S += self.WC[np.ix_(idx_l, idx_r)]

        WC_L /= float(count_L)
        WC_R /= float(count_R)
        WC_S /= float(count_S)

        for role in self.role_names:
            if not self.hg.roles.is_terminal(role):
                daughters = self.hg.roles.get_daughters(role)
                daughter_l = daughters['l'][0]
                daughter_r = daughters['r'][0]
                idx = self.find_roles(role)
                idx_l = self.find_roles(daughter_l)
                idx_r = self.find_roles(daughter_r)
                self.WC[np.ix_(idx, idx_l)] = WC_L
                self.WC[np.ix_(idx_l, idx)] = WC_L.T
                self.WC[np.ix_(idx, idx_r)] = WC_R
                self.WC[np.ix_(idx_r, idx)] = WC_R.T
                # In the default setting, this will be 0.
                self.WC[np.ix_(idx_l, idx_r)] = WC_S
                self.WC[np.ix_(idx_r, idx_l)] = WC_S.T

        self._set_weights()

    def average_weight2(self):

        WC_L = 0.
        WC_R = 0.
        WC_S = 0.   # sister roles
        count_L = 0
        count_R = 0
        count_S = 0
        for role in self.role_names:
            if not self.hg.roles.is_terminal(role):
                daughters = self.hg.roles.get_daughters(role)
                daughter_l = daughters['l'][0]
                daughter_r = daughters['r'][0]
                idx = self.find_roles(role)
                idx_l = self.find_roles(daughter_l)
                idx_r = self.find_roles(daughter_r)
                count_L += 1
                count_R += 1
                count_S += 1
                WC_L += self.WC[np.ix_(idx, idx_l)]
                WC_R += self.WC[np.ix_(idx, idx_r)]
                WC_S += self.WC[np.ix_(idx_l, idx_r)]

        WC_L /= float(count_L)
        WC_R /= float(count_R)
        WC_S /= float(count_S)

        WC_avg = np.zeros(self.WC.shape)

        for role in self.role_names:
            if not self.hg.roles.is_terminal(role):
                daughters = self.hg.roles.get_daughters(role)
                daughter_l = daughters['l'][0]
                daughter_r = daughters['r'][0]
                idx = self.find_roles(role)
                idx_l = self.find_roles(daughter_l)
                idx_r = self.find_roles(daughter_r)
                WC_avg[np.ix_(idx, idx_l)] = WC_L
                WC_avg[np.ix_(idx_l, idx)] = WC_L.T
                WC_avg[np.ix_(idx, idx_r)] = WC_R
                WC_avg[np.ix_(idx_r, idx)] = WC_R.T
                # In the default setting, this will be 0.
                WC_avg[np.ix_(idx_l, idx_r)] = WC_S
                WC_avg[np.ix_(idx_r, idx_l)] = WC_S.T

        return WC_avg

    # ===============================================================
    # NOTE: Check conditional probability distributions
    # ===============================================================

    def get_cond_prob(self):

        prefix = {}
        for si, sent in enumerate(self.corpus['sentence']):
            for wi in range(1, len(sent)+1):
                sent_bidx = np.where(self.corpus['target'][si] == 1)[0]
                rnames = ['(1,%d)' % wj for wj in range(1, wi + 1)]
                role_bidx = self.find_roles(rnames)
                temp = [bid for bid in sent_bidx if bid in role_bidx]
                pref = tuple([self.binding_names[bid] for bid in temp])
                if pref not in prefix.keys():
                    prefix[pref] = self.corpus['prob']['trees'].copy()
        # print('1', prefix)

        for pref in prefix:
            # print('2', prefix)
            bidx = self.find_bindings(list(pref))
            for parse_key in self.corpus['prob']['trees']:
                # print(parse_key)
                if not set(bidx).issubset(list(parse_key)):
                    prefix[pref][parse_key] = 0.

        # ambiguous terminals
        prefix0 = {}
        for pref in prefix:
            bnames = []
            for bname in list(pref):
                fname, rname = bname.split(self.hg.opts['bsep'])
                if self.hg.opts['sep'] in fname:
                    fname = fname.split(self.hg.opts['sep'])[0]
                bnames.append(fname + self.hg.opts['bsep'] + rname)
            if tuple(bnames) not in prefix0:
                prefix0[tuple(bnames)] = {}
                for parse_key in prefix[pref]:
                    prefix0[tuple(bnames)][parse_key] = 0.
            for parse_key in prefix[pref]:
                prefix0[tuple(bnames)][parse_key] += prefix[pref][parse_key]

        prefix = prefix0

        # Normalize
        for pref in prefix:
            psum = 0.
            for key, val in prefix[pref].items():
                psum += val
            for key in prefix[pref]:
                prefix[pref][key] /= psum

        return prefix

    def run_prefix(self, prefix, update_q_discrete=False, log_trace=False):
        for wi, fname in enumerate(prefix):
            self.run_word(
                fname, wi + 1, update_q_discrete=update_q_discrete, log_trace=log_trace)
            self.store.append({'actC': self.actC, 'q': self.q})

    def run_word(self, fname, wpos, symmetric=True, update_q_discrete=False, log_trace=False):

        q_max_backup = self.opts['q_max']
        bname = fname + self.hg.opts['bsep'] + '(1,%d)' % wpos
        qinc = self.qpolicy[wpos] - self.qpolicy[wpos - 1]
        self.opts['q_max'] = self.qpolicy[wpos]
        # print(bname)
        self.set_input(bname)
        if self.train_opts['update_scale_constants']:
            self.update_scale_constants(pos=wpos, symmetric=symmetric)
        if update_q_discrete:
            update_q = False
            self.q = self.qpolicy[wpos] * np.ones(self.num_roles)
        else:
            update_q = True

        if self.opts['use_runC']:
            self.runC(
                np.max(qinc) / self.opts['q_rate'], log_trace=log_trace, update_q=update_q)
        else:
            self.run(np.max(qinc) / self.opts['q_rate'],
                     log_trace=log_trace, update_q=update_q)
        self.opts['q_max'] = q_max_backup

    def run_wrapup(self, update_q_discrete=False, log_trace=False, clear_input=True):
        # self.opts['q_max'] = q_max
        dur = np.max(self.opts['q_max'] - self.q)
        if clear_input:
            self.clear_input()
        if self.train_opts['update_scale_constants']:
            self.update_scale_constants(pos=0)
        if update_q_discrete:
            update_q = False
            self.q = self.opts['q_max'] * np.ones(self.num_roles)
        else:
            update_q = True

        # experimental
        if self.train_opts['apply_wrapup_scale_constants']:
            self.update_scale_constants(pos=1)

        if self.opts['use_runC']:
            self.runC(dur / self.opts['q_rate'],
                      log_trace=log_trace, update_q=update_q)
        else:
            self.run(dur / self.opts['q_rate'],
                     log_trace=log_trace, update_q=update_q)
        self.store.append({'actC': self.actC, 'qvec': self.q})

    # def run_sent(self, sent, use_multiple_timescale=False,
    #              update_scale_constants=False, symmetric=True, scaling_factor=None,
    #              update_q_mask=True, decay_factor=0., wrapup_clear_input=False,
    #              use_type=True, disp=True,
    #              null_input_extend_pos=True, null_input_extend_lv=True,
    #              estr_null=2.0, plot_state=False):

    #     if 'look_ahead' not in self.opts:
    #         look_ahead = 0
    #     else:
    #         look_ahead = self.opts['look_ahead']

    #     maxlen = self.hg.opts['max_sent_len']
    #     if scaling_factor is None:
    #         scaling_factor = self.opts['scaling_factor']

    #     if not update_q_mask:
    #         self.update_q_mask(pos=0)

    #     self.reset(mu=self.ep, sd=0.02)
    #     if use_multiple_timescale:
    #         self.opts['scaling_factor'] = 0.5
    #         print('Scaling_factor =', self.opts['scaling_factor'])
    #         self.update_scale_constants(lv=1, pos=999, scale_type='lv')

    #     for ii, bname in enumerate(sent):
    #         # self.clear_input()
    #         if update_scale_constants:
    #             self.update_scale_constants(pos=ii + 1, symmetric=symmetric)
    #             # heatmap(self.vec2mat(self.scale_constants), xticklabels='', yticklabels='')
    #             # self.read_state(self.scale_constants)
    #         # self.plot_state(self.scale_constants)
    #         if update_q_mask:
    #             if hasattr(self, 'update_q_mask'):
    #                 self.update_q_mask(pos=ii + 1 + look_ahead)

    #         self.extC *= decay_factor
    #         self.set_input(bname, use_type=use_type, cumulative=True)
    #         if self.opts['use_runC']:
    #             self.runC((self.qpolicy[ii + 1] - self.qpolicy[ii]) / self.opts['q_rate'])
    #         else:
    #             self.run((self.qpolicy[ii + 1] - self.qpolicy[ii]) / self.opts['q_rate'])

    #         if plot_state:
    #             self.plot_tree(figsize=(18, 6))

    #         # heatmap(self.vec2mat(self.extC), xticklabels='', yticklabels='')
    #         # heatmap(self.vec2mat(self.scale_constants), xticklabels='', yticklabels='')
    #         # print(ii + 1, bname)
    #         # self.plot_trace('scale_constants')
    #         # plt.show()

    #     # heatmap(self.vec2mat(self.scale_constants))

    #     # temporary
    #     if 'decay_last' in self.opts:
    #         if self.opts['decay_last']:
    #             self.extC *= decay_factor
    #         else:
    #             pass
    #     else:
    #         self.extC *= decay_factor

    #     if wrapup_clear_input:
    #         self.clear_input()
    #     if update_scale_constants:
    #         self.update_scale_constants(pos=0)
    #     else:
    #         if use_multiple_timescale:
    #             self.update_scale_constants(pos=0)
    #     if update_q_mask:
    #         if hasattr(self, 'update_q_mask'):
    #             self.update_q_mask(pos=0)

    #     if len(sent) < maxlen:
    #         # null_input = [self.hg.opts['f_empty'] + self.hg.opts['bsep'] + '(1,{})'.format(jj)
    #         #               for jj in range(len(sent) + 1, maxlen + 1)]
    #         # self.set_input(null_input, use_type=False, cumulative=True)
    #         set_null_input(self, estr=estr_null, pos=len(sent) + 1,
    #             extend_pos=null_input_extend_pos, extend_lv=null_input_extend_lv,
    #             cumulative=True)  # CHECK

    #         # print(null_input)
    #     if self.opts['use_runC']:
    #         # self.runC((self.opts['q_max'] - self.qpolicy[:ii+1].sum()) / self.opts['q_rate'])
    #         self.runC((self.opts['q_max'] - min(self.q)) / self.opts['q_rate'])
    #     else:
    #         # self.run((self.opts['q_max'] - self.qpolicy[:ii+1].sum()) / self.opts['q_rate'])
    #         self.run((self.opts['q_max'] - min(self.q)) / self.opts['q_rate'])
    #     if disp:
    #         print(sent)
    #         self.plot_tree2(scale=1.5)

    def run_sent(self, sent, word_rt=None, use_multiple_timescale=False,
                 update_scale_constants=False, symmetric=True, scaling_factor=None,
                 update_q_mask=True, decay_factor=0., wrapup_clear_input=False,
                 use_type=True, disp=True,
                 null_input_extend_pos=True, null_input_extend_lv=True,
                 estr_null=2.0, plot_state=False):

        word_rt0 = []
        for ii, bname in enumerate(sent):
            word_rt0.append(
                (self.qpolicy[ii + 1] - self.qpolicy[ii]) / self.opts['q_rate'])

        if word_rt is not None:
            assert len(sent) == len(word_rt)
        else:
            word_rt = word_rt0.copy()

        if 'look_ahead' not in self.opts:
            look_ahead = 0
        else:
            look_ahead = self.opts['look_ahead']

        maxlen = self.hg.opts['max_sent_len']
        if scaling_factor is None:
            scaling_factor = self.opts['scaling_factor']

        if not update_q_mask:
            self.update_q_mask(pos=0)

        self.reset(mu=self.ep, sd=0.02)
        if use_multiple_timescale:
            self.opts['scaling_factor'] = 0.5
            print('Scaling_factor =', self.opts['scaling_factor'])
            self.update_scale_constants(lv=1, pos=999, scale_type='lv')

        for ii, bname in enumerate(sent):
            # self.clear_input()
            if update_scale_constants:
                self.update_scale_constants(pos=ii + 1, symmetric=symmetric)
                # heatmap(self.vec2mat(self.scale_constants), xticklabels='', yticklabels='')
                # self.read_state(self.scale_constants)
            # self.plot_state(self.scale_constants)
            if update_q_mask:
                if hasattr(self, 'update_q_mask'):
                    self.update_q_mask(pos=ii + 1 + look_ahead)

            self.extC *= decay_factor
            self.set_input(bname, use_type=use_type, cumulative=True)
            if self.opts['use_runC']:
                if word_rt[ii] <= word_rt0[ii]:
                    self.runC(word_rt[ii])
                else:
                    self.runC(word_rt0[ii])
                    self.opts['q_rate'] = 0.
                    self.runC(word_rt[ii] - word_rt0[ii])
                    self.opts['q_rate'] = 1.
            else:
                if word_rt[ii] <= word_rt0[ii]:
                    self.run(word_rt[ii])
                else:
                    self.run(word_rt0[ii])
                    self.opts['q_rate'] = 0.
                    self.run(word_rt[ii] - word_rt0[ii])
                    self.opts['q_rate'] = 1.

            if plot_state:
                self.plot_tree(figsize=(18, 6))

            # heatmap(self.vec2mat(self.extC), xticklabels='', yticklabels='')
            # heatmap(self.vec2mat(self.scale_constants), xticklabels='', yticklabels='')
            # print(ii + 1, bname)
            # self.plot_trace('scale_constants')
            # plt.show()

        # heatmap(self.vec2mat(self.scale_constants))

        # temporary
        if 'decay_last' in self.opts:
            if self.opts['decay_last']:
                self.extC *= decay_factor
            else:
                pass
        else:
            self.extC *= decay_factor

        if wrapup_clear_input:
            self.clear_input()
        if update_scale_constants:
            self.update_scale_constants(pos=0)
        else:
            if use_multiple_timescale:
                self.update_scale_constants(pos=0)
        if update_q_mask:
            if hasattr(self, 'update_q_mask'):
                self.update_q_mask(pos=0)

        if len(sent) < maxlen:
            # null_input = [self.hg.opts['f_empty'] + self.hg.opts['bsep'] + '(1,{})'.format(jj)
            #               for jj in range(len(sent) + 1, maxlen + 1)]
            # self.set_input(null_input, use_type=False, cumulative=True)
            set_null_input(self, estr=estr_null, pos=len(sent) + 1,
                           extend_pos=null_input_extend_pos, extend_lv=null_input_extend_lv,
                           cumulative=True)  # CHECK

            # print(null_input)
        if self.opts['use_runC']:
            # self.runC((self.opts['q_max'] - self.qpolicy[:ii+1].sum()) / self.opts['q_rate'])
            self.runC((self.opts['q_max'] - min(self.q)) / self.opts['q_rate'])
        else:
            # self.run((self.opts['q_max'] - self.qpolicy[:ii+1].sum()) / self.opts['q_rate'])
            self.run((self.opts['q_max'] - min(self.q)) / self.opts['q_rate'])
        if disp:
            print(sent)
            self.plot_tree2(scale=1.5)

    def estimate_prob_inc(self, prefix, num_trials=40, progress=0, update_q_discrete=False):
        # NOW

        corpus = {}
        corpus['target'] = []
        corpus['count'] = []
        corpus['prob_sent'] = []
        self.actC_list = []

        for trial_id in range(num_trials):

            if progress > 0:
                if (trial_id + 1) % progress == 0:
                    print('[%04d]' % (trial_id + 1), end='')
                    if (trial_id + 1) % (10 * progress) == 0:
                        print('')

            self.reset(mu=self.ep, sd=self.train_opts['init_noise_mag'])
            # self.opts['q_max'] = 15.
            # self.set_state(mu=self.ep, sd=self.train_opts['init_noise_mag'])
            if len(prefix) > 0:
                self.run_prefix(prefix, update_q_discrete=update_q_discrete)
                self.actC_list.append(list(self.store[-1]['actC']))
            else:
                self.actC_list.append(list(self.actC))

            self.run_wrapup(update_q_discrete=update_q_discrete)
            gp = self.read_grid_point(disp=False)
            idx = self.find_bindings(gp)
            self.set_discrete_state(gp)

            if list(self.actC) not in corpus['target']:
                corpus['target'].append(list(self.actC))
                corpus['count'].append(1)
            else:
                idx = corpus['target'].index(list(self.actC))
                corpus['count'][idx] += 1

        corpus['target'] = np.array(corpus['target'])
        corpus['count'] = np.array(corpus['count'])
        corpus['prob_sent'] = corpus['count'] / corpus['count'].sum()

        stat = self.get_corpus_stat(corpus)
        return stat, np.array(self.actC_list)

        # self.actC_list = np.array(self.actC_list)
        # self.prob_bindings = actC_mean / num_trials

    def estimate_prob_inc_jax(self, prefix, num_trials=40, progress=0, update_q_discrete=False, rng_seed=None):
        """
        JAX-accelerated version of estimate_prob_inc.

        Runs all trials in parallel on GPU for massive speedup.
        Falls back to original CPU version if JAX is not available.

        Args:
            prefix: List of filler names for the prefix
            num_trials: Number of trials to run in parallel
            progress: Progress reporting interval
            update_q_discrete: Boolean for q update mode
            rng_seed: Random seed for reproducibility

        Returns:
            stat: Corpus statistics (same format as original)
            actC_list: Array of activation states (num_trials, num_bindings)
        """
        if not JAX_AVAILABLE:
            print("JAX not available, falling back to CPU version")
            return self.estimate_prob_inc(prefix, num_trials, progress, update_q_discrete)

        print(f"Running {num_trials} trials in parallel on GPU...")
        t0 = time.time()

        # Extract network parameters for JAX
        net_params = _extract_net_params_for_jax(self)

        # Generate random keys for each trial
        if rng_seed is None:
            rng_seed = np.random.randint(0, 1000000)
        rng = jax.random.PRNGKey(rng_seed)
        rng_keys = jax.random.split(rng, num_trials)

        # Run all trials in parallel on GPU
        actC_batch, grid_point_batch = _run_trials_batched_jax(
            rng_keys, net_params, prefix, update_q_discrete
        )

        # Convert back to numpy for compatibility with existing code
        actC_batch = np.array(actC_batch)
        grid_point_batch = np.array(grid_point_batch)

        print(f"GPU execution time: {time.time() - t0:.3f}s")

        # Process results (same as original - aggregate unique states)
        # CRITICAL FIX: Use grid points (discrete) not continuous actC for aggregation
        # Convert grid point indices to one-hot actC vectors (like CPU version does)
        corpus = {}
        corpus['target'] = []
        corpus['count'] = []
        corpus['prob_sent'] = []
        actC_list = []

        for trial_id in range(num_trials):
            # Store continuous actC for return value
            actC_list.append(list(actC_batch[trial_id]))
            # Convert grid point (filler indices per role) to one-hot actC
            grid_point = grid_point_batch[trial_id]
            actC_discrete = np.zeros(self.num_bindings)
            for role_idx, filler_idx in enumerate(grid_point):
                # the binding name ordering is binding_names = [f + bsep + r for r in self.role_names for f in self.filler_names]
                # = [f0/r0, f1/r0, f2/r0, ..., f0/r1, f1/r1, ...]
                binding_idx = role_idx * self.num_fillers + int(filler_idx)
                actC_discrete[binding_idx] = 1.0

            # Aggregate using discrete states (like CPU version)
            if list(actC_discrete) not in corpus['target']:
                corpus['target'].append(list(actC_discrete))
                corpus['count'].append(1)
            else:
                idx = corpus['target'].index(list(actC_discrete))
                corpus['count'][idx] += 1

        corpus['target'] = np.array(corpus['target'])
        corpus['count'] = np.array(corpus['count'])
        corpus['prob_sent'] = corpus['count'] / corpus['count'].sum()

        stat = self.get_corpus_stat(corpus)
        return stat, np.array(actC_list)

    def ema_stat(self, stat_new, stat_old, weight=None):

        if weight is None:
            weight = self.train_opts['ema_stat_weight']

        res = {}

        for obj in stat_new:

            res[obj] = {}

            keys1 = [key for key in stat_new[obj]]
            keys2 = [key for key in stat_old[obj]]

            keys_all = list(set(keys1 + keys2))

            for key in keys_all:

                if key not in stat_new[obj]:
                    p = 1e-15
                else:
                    p = max(1e-15, stat_new[obj][key])

                if key not in stat_old[obj]:
                    q = 1e-15
                else:
                    q = max(1e-15, stat_old[obj][key])

                if self.train_opts['ema_trees_only']:
                    if obj == "trees":
                        res[obj][key] = (1 - weight) * p + weight * q
                    else:
                        if key in stat_new[obj]:
                            res[obj][key] = p
                else:
                    res[obj][key] = (1 - weight) * p + weight * q

        return res

    def cost(self, stat_P, stat_Q):

        kl = {}
        xent = {}
        err = {}
        err_log = {}

        for obj in stat_P:

            kl[obj] = 0.
            xent[obj] = 0.
            err[obj] = {}
            err_log[obj] = {}

            keys1 = [key for key in stat_P[obj]]
            keys2 = [key for key in stat_Q[obj]]

            if (obj == "trees") and (self.train_opts['use_err_gram_only']):
                keys_all = keys1
            else:
                keys_all = list(set(keys1 + keys2))

            if obj == "trees":
                ent_p = 0.
                ent_q = 0.
                n_trees = len(keys_all)
                p_unif = np.ones(n_trees) / n_trees
                ent_max = (-p_unif * np.log(p_unif)).sum()

            for key in keys_all:

                if key not in stat_P[obj]:
                    p = 1e-15
                else:
                    p = max(1e-15, stat_P[obj][key])

                if key not in stat_Q[obj]:
                    q = 1e-15
                else:
                    q = max(1e-15, stat_Q[obj][key])

                # if np.isclose(p, 1e-15) and np.isclose(q, 1e-15):
                #     print('CHECK')

                kl[obj] += p * (np.log(p) - np.log(q))
                xent[obj] += (-p * np.log(q))
                err[obj][key] = p - q
                elog = np.log(p) - np.log(q)
                err_log[obj][key] = np.sign(elog) * min(
                    abs(elog) * self.train_opts['err_log_scaler'],
                    self.train_opts['err_log_max'])

                if obj == 'trees':
                    if key in keys1:
                        # Consider only grammatical parses
                        ent_p += (-p * np.log(p))
                        ent_q += (-q * np.log(q))

        # Normalize entropy (entropy: [0, 1])
        if abs(ent_max) > 0:
            ent_p /= ent_max
            ent_q /= ent_max
        else:
            ent_p = 0
            ent_q = 0
        err['ent_diff'] = ent_p - ent_q         # experimental
        err_log['ent_diff'] = ent_p - ent_q     # use the same

        # ADD terminal binding xent to treelets
        if ('bindings' in stat_P) and ('bindings' in stat_Q):
            keys1 = [key for key in stat_P['bindings']]
            keys2 = [key for key in stat_Q['bindings']]
            keys_all = list(set(keys1 + keys2))
            for key in keys_all:
                if key in self.find_roles(self.hg.roles.get_terminals()):
                    if key not in stat_P['bindings']:
                        p = 1e-15
                    else:
                        p = max(1e-15, stat_P['bindings'][key])

                    if key not in stat_Q['bindings']:
                        q = 1e-15
                    else:
                        q = max(1e-15, stat_Q['bindings'][key])

                    # if np.isclose(p, 1e-15) and np.isclose(q, 1e-15):
                    #     print('CHECK')

                    kl['bindings'] += p * (np.log(p) - np.log(q))
                    xent['bindings'] += (-p * np.log(q))
                    # err[obj][key] = p - q

        return kl, xent, err, err_log

    def cost_grad(self, err, extC_token):
        # CHECK: Use net.extC directly instead of extC_token to update net.estr
        # (for the case of using cumulative input)

        # print('extC_token', extC_token)
        # print([self.binding_names[ii] for ii, val in enumerate(extC_token) if val == 1])

        rnames_terminal = self.hg.roles.get_terminals()
        idx_terminal = self.find_roles(rnames_terminal)

        dWC = np.zeros(self.WC.shape)
        dbC = np.zeros(self.bC.shape)
        destr = np.zeros(self.estr.shape)
        dq = np.zeros(self.num_roles)

        if self.train_opts['bias1_only']:

            keys_tree = [key for key in err['trees']]
            if self.train_opts['num_tree_update'] is not None:
                idx_keys = np.random.choice(
                    len(keys_tree), self.train_opts['num_tree_update'])
                keys_tree = [keys_tree[ii] for ii in idx_keys]

            keys_treelet = [key for key in err['treelets']]
            if self.train_opts['num_treelet_update'] is not None:
                idx_keys = np.random.choice(
                    len(keys_treelet), self.train_opts['num_treelet_update'])
                keys_treelet = [keys_treelet[ii] for ii in idx_keys]

            keys_binding = []
            for key in keys_treelet:
                keys_binding += list(key)

            if self.train_opts['coef']['trees'] > 0.:
                for key, val in err['trees'].items():

                    if key in keys_tree:  # pwc: new

                        if self.train_opts['err_tree_positive_only']:
                            val = max(val, 0.)

                        state = np.zeros(self.num_bindings)
                        state[list(key)] = 1.
                        # dbC += state * self.train_opts['mask0'] * val * self.train_opts['coef']['trees']
                        dbC += state * val * self.train_opts['coef']['trees']

                        if self.train_opts['update_estr']:
                            if self.train_opts['update_estr_terminals_only']:
                                idx_tb = [ii for ii in list(
                                    key) if ii in idx_terminal]
                            else:
                                idx_tb = list(key)
                            destr[idx_tb] += extC_token[idx_tb] * \
                                val * \
                                self.train_opts['coef']['trees']  # * actC[idx_tb]

            if self.train_opts['coef']['treelets'] > 0.:
                for key, val in err['treelets'].items():

                    if key in keys_treelet:  # pwc: new
                        key = list(key)
                        dbC[key[0]] += val * \
                            self.train_opts['coef']['treelets']

                        if self.train_opts['update_estr']:
                            if not self.train_opts['update_estr_terminals_only']:
                                destr[key] += extC_token[key] * \
                                    val * self.train_opts['coef']['treelets']

                for key, val in err['bindings'].items():

                    if key in keys_binding:
                        if key in idx_terminal:
                            dbC[key] += val * \
                                self.train_opts['coef']['treelets']

                            if self.train_opts['update_estr']:
                                destr[key] += extC_token[key] * val * \
                                    self.train_opts['coef']['treelets']  # * actC[idx_tb]

                                # print('bname =', self.binding_names[key])
                                # print('extC =', extC_token[key])
                                # print('val =', val)
                                # print('grad =', extC_token[key] * val *
                                #       self.train_opts['coef']['treelets'])
                                # # print('2', destr)

            if self.train_opts['coef']['binding_pairs'] > 0.:
                for key, val in err['binding_pairs'].items():
                    key = list(key)
                    dbC[key[0]] += val * \
                        self.train_opts['coef']['binding_pairs']
                    dbC[key[1]] += val * \
                        self.train_opts['coef']['binding_pairs']

            if self.train_opts['coef']['bindings'] > 0.:
                for key, val in err['bindings'].items():
                    dbC[key] += val * self.train_opts['coef']['bindings']
                    if self.train_opts['update_estr']:
                        destr[key] += extC_token[key] * val * \
                            self.train_opts['coef']['bindings']  # * actC[idx_tb]

            # ENTROPY (use parse structures)
            if self.train_opts['coef_q'] > 0.:
                dq = -err['ent_diff'] * self.train_opts['coef_q']
                # print(dq)
        else:

            keys_tree = [key for key in err['trees']]
            if self.train_opts['num_tree_update'] is not None:
                idx_keys = np.random.choice(
                    len(keys_tree), self.train_opts['num_tree_update'])
                keys_tree = [keys_tree[ii] for ii in idx_keys]

            keys_treelet = [key for key in err['treelets']]
            if self.train_opts['num_treelet_update'] is not None:
                idx_keys = np.random.choice(
                    len(keys_treelet), self.train_opts['num_treelet_update'])
                keys_treelet = [keys_treelet[ii] for ii in idx_keys]

            keys_binding = []
            for key in keys_treelet:
                keys_binding += list(key)

            if self.train_opts['coef']['trees'] > 0.:
                for key, val in err['trees'].items():

                    if key in keys_tree:  # pwc: new

                        if self.train_opts['err_tree_positive_only']:
                            val = max(val, 0.)

                        state = np.zeros(self.num_bindings)
                        state[list(key)] = 1.
                        dWC += np.outer(state, state) * \
                            self.train_opts['mask0'] * val * \
                            self.train_opts['coef']['trees']

                        if self.train_opts['update_estr']:
                            if self.train_opts['update_estr_terminals_only']:
                                idx_tb = [ii for ii in list(
                                    key) if ii in idx_terminal]
                            else:
                                idx_tb = list(key)
                            destr[idx_tb] += extC_token[idx_tb] * \
                                val * \
                                self.train_opts['coef']['trees']  # * actC[idx_tb]

            if self.train_opts['coef']['treelets'] > 0.:
                for key, val in err['treelets'].items():

                    if key in keys_treelet:  # pwc: new
                        key = list(key)

                        if not self.train_opts['bias_only']:
                            dWC[key[0], key[1]] += val * \
                                self.train_opts['coef']['treelets']
                            dWC[key[1], key[0]] += val * \
                                self.train_opts['coef']['treelets']
                            dWC[key[0], key[2]] += val * \
                                self.train_opts['coef']['treelets']
                            dWC[key[2], key[0]] += val * \
                                self.train_opts['coef']['treelets']

                        dWC[key[0], key[0]] += val * \
                            self.train_opts['coef']['treelets']

                        if self.train_opts['update_estr']:
                            if not self.train_opts['update_estr_terminals_only']:
                                destr[key] += extC_token[key] * \
                                    val * self.train_opts['coef']['treelets']

                for key, val in err['bindings'].items():

                    if key in keys_binding:
                        if key in idx_terminal:
                            dWC[key, key] += val * \
                                self.train_opts['coef']['treelets']

                            if self.train_opts['update_estr']:
                                destr[key] += extC_token[key] * val * \
                                    self.train_opts['coef']['treelets']  # * actC[idx_tb]

                                # print('bname =', self.binding_names[key])
                                # print('extC =', extC_token[key])
                                # print('val =', val)
                                # print('grad =', extC_token[key] * val *
                                #       self.train_opts['coef']['treelets'])
                                # # print('2', destr)

            if self.train_opts['coef']['binding_pairs'] > 0.:
                for key, val in err['binding_pairs'].items():
                    key = list(key)
                    dWC[key[0], key[1]] += val * \
                        self.train_opts['coef']['binding_pairs']
                    dWC[key[1], key[0]] += val * \
                        self.train_opts['coef']['binding_pairs']

            if self.train_opts['coef']['bindings'] > 0.:
                for key, val in err['bindings'].items():
                    dWC[key, key] += val * self.train_opts['coef']['bindings']
                    if self.train_opts['update_estr']:
                        destr[key] += extC_token[key] * val * \
                            self.train_opts['coef']['bindings']  # * actC[idx_tb]

            # ENTROPY (use parse structures)
            if self.train_opts['coef_q'] > 0.:
                dq = -err['ent_diff'] * self.train_opts['coef_q']
                # print(dq)

        return dWC, destr, dq, dbC

    def update_train_opts(self, train_opts):

        for key in train_opts:
            if key in self.train_opts:
                self.train_opts[key] = train_opts[key]
                if key in ['update_sister_harmony', 'update_gram_only']:
                    self.train_opts['mask0'] = self.get_mask0()
            else:
                sys.exit('`{}` is not supported.'.format(key))

    def initialize(self, train_opts=None):

        self.WC = self.params_backup['WC'].copy()
        self.bC = self.params_backup['bC'].copy()
        self.estr = self.params_backup['estr'].copy()
        self.qpolicy = self.params_backup['qpolicy'].copy()
        self._set_weights()
        self._set_biases()
        self.update_bowl_strength()
        self.ep = self.params_backup['ep'].copy()
        # self.get_ep(method=self.opts['ep_method'])
        self.epoch_num = 0
        self.store = []

        # every grammatical parse was generated over the course of learning
        self.nonzero_all0 = False
        # every grammatical parse was generated during the last epoch
        self.nonzero_all1 = False
        # number of treelet frames to update in each iteration (asynchronous update)
        self.num_treelets_update = max(self.num_roles//4, 1)

        # Set train_opts to default values
        self.train_opts = {}
        self.train_opts['report_cycle'] = 1
        self.train_opts['weight_decay'] = False
        self.train_opts['weight_decay_to'] = ['default', 'average'][0]
        # learning rate will be multiplied
        self.train_opts['weight_decay_factor'] = 0.001
        self.train_opts['update_sister_harmony'] = False
        self.train_opts['bias2_only'] = False
        self.train_opts['free_update_null'] = False
        self.train_opts['num_treelet_update'] = None
        self.train_opts['num_tree_update'] = None      # or integer
        # when set to True, updates non-zero weights only
        self.train_opts['update_gram_only'] = False
        self.train_opts['ema_stat_weight'] = 0.
        self.train_opts['ema_trees_only'] = False
        # when computing errors, ignore ungrammatical structures
        self.train_opts['use_err_gram_only'] = False
        # consider only positive error for tree probabilities
        self.train_opts['err_tree_positive_only'] = False
        self.train_opts['parallel_parser_train'] = False
        # each prob diff would be 0.999 - 0.001
        self.train_opts['parallel_parser_dWC_scaler'] = 0.1
        # ratio of paralle parsing to production (depricated)
        self.train_opts['parallel_parser_sample_ratio'] = 0.1
        self.train_opts['parallel_parser_num_trials'] = 0
        self.train_opts['parallel_parser_sample_uniform'] = True
        self.train_opts['apply_wrapup_scale_constants'] = False
        self.train_opts['adaptive_training'] = False
        self.train_opts['asynchronous_update'] = False
        # use this only with the default coef setting
        self.train_opts['asynchronous_update_choose_errmax'] = False
        self.train_opts['use_err_log'] = False
        self.train_opts['err_log_scaler'] = 0.1
        self.train_opts['err_log_max'] = 1   # clipping
        self.train_opts['use_err_avg'] = False
        self.train_opts['lrate'] = 0.01
        self.train_opts['num_trials'] = 20
        self.train_opts['num_epochs'] = 10
        self.train_opts['pseudocount'] = 1e-15
        self.train_opts['epsilon'] = 1e-15
        self.train_opts['trace_varnames'] = [
            'kl_trees',
            'kl_treelets',
            'kl_binding_pairs',
            'kl_bindings',
            'xent_trees',
            'xent_treelets',
            'xent_binding_pairs',
            'xent_bindings',
            # 'xent_terminal_bindings',
            'prob_sent',
            'acc',
            'WC',
            # 'bC',
            'estr',
            # 'qpolicy',
            'lrate',
            'num_trials',
        ]

        coef = {}
        coef['trees'] = 1.
        coef['treelets'] = 0.
        coef['binding_pairs'] = 0.
        coef['bindings'] = 0.
        # NOTE: normalize coef
        self.train_opts['coef'] = coef
        self.train_opts['bias1_only'] = False
        self.train_opts['bias_only'] = False   # if coef['treelets'] > 0,
        self.train_opts['update_estr'] = True
        self.train_opts['update_bowl_strength'] = True
        self.train_opts['ep_method'] = ['newton', 'integration'][1]
        self.train_opts['use_actval'] = False
        # 15.
        self.train_opts['dur'] = self.opts['q_max'] / self.opts['q_rate']
        self.train_opts['init_noise_mag'] = 0.02
        self.train_opts['update_w'] = True
        self.train_opts['coef_q'] = 0.
        self.train_opts['coef_weights_for_parser'] = 1.
        self.train_opts['scale_dWC_parser'] = 0.1
        self.train_opts['update_scale_constants'] = True
        self.train_opts['update_estr_terminals_only'] = True
        self.train_opts['average_weight'] = False
        self.train_opts['average_filler_bias'] = False
        self.train_opts['optimizer'] = ['sgd', 'adam'][0]

        self.train_opts['mask0'] = self.get_mask0()

        # mask_bias
        # NOTE: Harmony values of illegitimate bindings are assumed to be
        # smaller than or equal to -4.
        # idx_mask_bias1 = np.diag(self.bC) <= -4.
        # mask_avg_bias1 = np.ones(self.num_bindings)
        # mask_avg_bias1[idx_mask_bias1] = np.nan
        # self.train_opts['mask_avg_bias1'] = mask_avg_bias1
        # idx_mask_bias2 = np.diag(self.WC) <= -8.
        # mask_avg_bias2 = np.ones(self.num_bindings)
        # mask_avg_bias2[idx_mask_bias2] = np.nan
        # self.train_opts['mask_avg_bias2'] = mask_avg_bias2
        self.train_opts['idx_mask_bias1'] = np.diag(self.bC) <= -4.
        self.train_opts['idx_mask_bias2'] = np.diag(self.WC) <= -8.

        # Update train_opts
        if train_opts is not None:
            self.update_train_opts(train_opts)

        if len(self.train_opts['trace_varnames']) > 0:
            self.traces_train = {}
            for key in self.train_opts['trace_varnames']:
                self.traces_train[key] = []

        if self.train_opts['optimizer'] == 'adam':
            self.optim = {}
            self.optim['M_WC'] = np.zeros_like(self.WC)
            self.optim['M_bC'] = np.zeros_like(self.bC)
            self.optim['R_WC'] = np.zeros_like(self.WC)
            self.optim['R_bC'] = np.zeros_like(self.bC)
            self.optim['beta1'] = .9
            self.optim['beta2'] = .999
            self.optim['eps'] = 1e-8

    def get_mask0(self):

        if self.train_opts['update_gram_only']:
            mask0 = abs(np.sign(self.WC))
            # allow the udpate of second-order bias of every binding
            np.fill_diagonal(mask0, 1)
        else:
            rnames_terminal = self.hg.roles.get_terminals()
            idx_terminal = self.find_roles(rnames_terminal)
            mask0 = np.zeros(self.WC.shape)
            for role in self.role_names:
                idx = self.find_roles(role)
                mask0[np.ix_(idx, idx)] = 1.
                if not self.hg.roles.is_terminal(role):
                    daughters = self.hg.roles.get_daughters(role)
                    idx_l = self.find_roles(daughters['l'])
                    idx_r = self.find_roles(daughters['r'])
                    mask0[np.ix_(idx, idx_l)] = 1.
                    mask0[np.ix_(idx_l, idx)] = 1.
                    mask0[np.ix_(idx, idx_r)] = 1.
                    mask0[np.ix_(idx_r, idx)] = 1.
                    if self.train_opts['update_sister_harmony']:
                        mask0[np.ix_(idx_l, idx_r)] = 1.
                        mask0[np.ix_(idx_r, idx_l)] = 1.

        return mask0

    def update_traces_train(self, log):

        for key in self.train_opts['trace_varnames']:
            if key == 'WC':
                self.traces_train[key].append(log[key].flatten(order='F'))
            else:
                self.traces_train[key].append(log[key])

    def clear_traces_train(self):

        for key in self.train_opts['trace_varnames']:
            self.traces_train[key] = []

    def train2(self, prefix_list=None, prefix_weights=None,
               train_opts=None, savefilename=None, log_ema_stat=True):

        if hasattr(self, 'traces_train'):
            for key, val in self.traces_train.items():
                if type(val).__module__ == np.__name__:
                    self.traces_train[key] = list(val)

        if train_opts is not None:
            self.update_train_opts(train_opts)

        if prefix_list is None:
            prefix_list = [[]]

        if prefix_weights is None:
            prefix_weights = np.ones(len(prefix_list))
            prefix_weights /= prefix_weights.sum()

        maxlen_prefix = 0
        for prefix in prefix_list:
            maxlen_prefix = max(maxlen_prefix, len(prefix))

        for _ in range(self.train_opts['num_epochs']):

            self.epoch_num += 1

            # mask = net.params_backup['WC'].astype(bool).astype(float)
            mask = np.ones(self.WC.shape)
            dWC = np.zeros(self.WC.shape)
            dbC = np.zeros(self.bC.shape)
            # FOR NOW: use same commitment strength for all roles
            dqpolicy = np.zeros(self.qpolicy.shape)
            destr = np.zeros(self.estr.shape)
            xent = {}
            xent['trees'] = 0.
            xent['treelets'] = 0.
            xent['binding_pairs'] = 0.
            xent['bindings'] = 0.
            kl = {}
            kl['trees'] = 0.
            kl['treelets'] = 0.
            kl['binding_pairs'] = 0.
            kl['bindings'] = 0.

            # update weights
            prob_sent_report_list = []

            if self.train_opts['parallel_parser_train']:
                dWC_parse, acc, dbC_parse = self.train_parallel_parsing()
                dWC += dWC_parse
                dbC += dbC_parse

            for pi, prefix in enumerate(prefix_list):

                if prefix_weights[pi] > 0:

                    if len(prefix) > 0:
                        scale_dWC = self.train_opts['scale_dWC_parser']
                        prefix_bnames = [ftype + self.hg.opts['bsep'] + '(1,{})'.format(wi + 1)
                                         for wi, ftype in enumerate(prefix)]
                    else:
                        scale_dWC = 1.0
                        prefix_bnames = []

                    stat_P = self.get_corpus_stat(
                        self.subset_corpus(prefix_bnames))
                    # TEST: change estimate_prob_inc to estimate_prob_inc_jax
                    if JAX_AVAILABLE:
                        stat_Q, actC_set = self.estimate_prob_inc_jax(
                            prefix=prefix, num_trials=self.train_opts['num_trials'])
                    else:
                        stat_Q, actC_set = self.estimate_prob_inc(
                            prefix=prefix, num_trials=self.train_opts['num_trials'])
                    if self.train_opts['ema_stat_weight'] > 0:
                        if hasattr(self, 'stat_Q_prev'):
                            stat_Q_new = self.ema_stat(
                                stat_new=stat_Q, stat_old=self.stat_Q_prev, weight=None)
                        else:
                            stat_Q_new = stat_Q
                    else:
                        stat_Q_new = stat_Q

                    self.clear_input()
                    if len(prefix_bnames) > 0:
                        # currently, one word at a time
                        prefix_bnames = prefix_bnames[-1]
                        self.set_input(prefix_bnames)
                    extC_token = self.extC.astype(bool).astype(int)

                    kl_curr, xent_curr, err, err_log = self.cost(
                        stat_P, stat_Q_new)
                    self.stat_Q_prev = stat_Q_new  #

                    if self.train_opts['use_err_avg']:
                        err_avg = {}
                        for key1, _ in err.items():
                            err_avg[key1] = {}
                            if isinstance(err[key1], dict):
                                for key2, _ in err[key1].items():
                                    err_avg[key1][key2] = (
                                        err[key1][key2] + err_log[key1][key2])/2
                            else:
                                err_avg[key1] = (err[key1] + err_log[key1])/2
                        dWC_curr, destr_curr, dq_curr, dbC_curr = self.cost_grad(
                            err_avg, extC_token)
                    elif self.train_opts['use_err_log']:
                        dWC_curr, destr_curr, dq_curr, dbC_curr = self.cost_grad(
                            err_log, extC_token)
                    else:
                        dWC_curr, destr_curr, dq_curr, dbC_curr = self.cost_grad(
                            err, extC_token)

                    dWC += dWC_curr * scale_dWC * prefix_weights[pi]
                    dbC += dbC_curr * scale_dWC * prefix_weights[pi]
                    if len(prefix) > 0:
                        destr += destr_curr * prefix_weights[pi]
                        if self.train_opts['coef_q'] > 0:
                            dqpolicy[len(prefix)] += dq_curr * \
                                prefix_weights[pi]
                    for key in xent:
                        xent[key] += xent_curr[key]
                    for key in kl:
                        kl[key] += kl_curr[key]

                    prob_sent_report = np.zeros(len(self.corpus['target']))
                    if self.train_opts['ema_stat_weight'] > 0 and log_ema_stat:
                        for si, state in enumerate(self.corpus['target']):
                            gp_key = tuple(np.where(state == 1)[0])
                            if gp_key in stat_Q_new['trees']:
                                prob_sent_report[si] = stat_Q_new['trees'][gp_key]
                    else:
                        for si, state in enumerate(self.corpus['target']):
                            gp_key = tuple(np.where(state == 1)[0])
                            if gp_key in stat_Q['trees']:
                                prob_sent_report[si] = stat_Q['trees'][gp_key]
                    prob_sent_report_list.append(list(prob_sent_report))

            # if len(dqvec_dict):
            #     for len_prefix in range(1, maxlen_prefix + 1):
            #         qvec = self.qpolicy[len_prefix] + dqvec_dict[len_prefix]
            #         self.qpolicy[len_prefix] = np.maximum(self.qpolicy[len_prefix - 1], qvec)

            if self.train_opts['asynchronous_update']:

                if self.train_opts['asynchronous_update_choose_errmax']:
                    temp = np.zeros(len(self.role_names))
                    for ri, rname in enumerate(self.role_names):
                        idx = self.find_roles(rname)
                        for key, val in err['treelets'].items():
                            if key[0] in idx:
                                temp[ri] += abs(val)

                        if self.hg.roles.is_terminal(rname):
                            for key, val in err['bindings'].items():
                                if key in idx:
                                    temp[ri] += abs(val)

                    rid_candidates = np.argwhere(
                        temp == np.amax(temp)).flatten()
                    role_idx_list = np.random.choice(
                        rid_candidates, self.num_treelets_update, replace=False)

                else:
                    role_idx_list = np.random.choice(
                        self.num_roles, self.num_treelets_update, replace=False)

                maskbC_update = np.zeros(self.num_bindings)
                rnames = [self.role_names[rid] for rid in role_idx_list]
                idx = self.find_roles(rnames)
                maskbC_update[idx] = 1.

                maskWC_update = np.zeros(
                    (self.num_bindings, self.num_bindings))
                treelet_list = []
                for rid in role_idx_list:
                    r_daughters = self.hg.roles.get_daughters(
                        self.role_names[rid])
                    treelet_list.append(
                        [self.role_names[rid]] + r_daughters['l'] + r_daughters['r'])

                for treelet in treelet_list:
                    idx = self.find_roles(treelet)
                    maskWC_update[np.ix_(idx, idx)] = 1.
            else:
                maskWC_update = np.ones((self.num_bindings, self.num_bindings))
                maskbC_update = np.ones(self.num_bindings)

            if self.train_opts['update_w']:
                # print('epoch num=', epi, destr)

                # TODO: Add the weight decay term to different settings
                #     : Currently, the term was added only to the default setting case
                if ('weight_decay' in self.train_opts) and self.train_opts['weight_decay']:
                    if self.train_opts['weight_decay_to'] == 'default':
                        ref = self.params_backup['WC']
                    else:
                        ref = self.average_weight2()
                    weight_decay = - \
                        self.train_opts['weight_decay_factor'] * \
                        (self.WC - ref)
                else:
                    weight_decay = np.zeros(self.WC.shape)

                if not (('bias2_only' in self.train_opts) and self.train_opts['bias2_only']):
                    if self.train_opts['optimizer'] == 'adam':
                        # TODO: Add the weight decay term
                        self.optim['M_WC'] = self.optim['beta1'] * \
                            self.optim['M_WC'] + \
                            (1. - self.optim['beta1']) * dWC
                        self.optim['R_WC'] = self.optim['beta2'] * \
                            self.optim['R_WC'] + \
                            (1. - self.optim['beta2']) * dWC**2
                        m_k_hat_WC = self.optim['M_WC'] / \
                            (1. - self.optim['beta1']**self.epoch_num)
                        r_k_hat_WC = self.optim['R_WC'] / \
                            (1. - self.optim['beta2']**self.epoch_num)
                        self.WC += self.train_opts['lrate'] * m_k_hat_WC / \
                            (np.sqrt(r_k_hat_WC) + self.optim['eps'])
                        self._set_weights()
                    else:
                        self.WC += self.train_opts['lrate'] * \
                            (dWC + weight_decay) * maskWC_update
                        self._set_weights()

                if self.train_opts['bias1_only']:
                    self.bC += self.train_opts['lrate'] * dbC * maskbC_update
                    self._set_biases()

                if not self.opts['use_second_order_bias']:
                    if self.train_opts['optimizer'] == 'adam':
                        self.optim['M_bC'] = self.optim['beta1'] * \
                            self.optim['M_bC'] + \
                            (1. - self.optim['beta1']) * dbC
                        self.optim['R_bC'] = self.optim['beta2'] * \
                            self.optim['R_bC'] + \
                            (1. - self.optim['beta2']) * dbC**2
                        m_k_hat_bC = self.optim['M_bC'] / \
                            (1. - self.optim['beta1']**self.epoch_num)
                        r_k_hat_bC = self.optim['R_bC'] / \
                            (1. - self.optim['beta2']**self.epoch_num)
                        self.bC += self.train_opts['lrate'] * m_k_hat_bC / \
                            (np.sqrt(r_k_hat_bC) + self.optim['eps'])
                        self._set_biases()
                    else:
                        # update
                        self.bC += self.train_opts['lrate'] * \
                            dbC * maskbC_update
                        self._set_biases()

                if self.train_opts['update_estr']:
                    self.estr += self.train_opts['lrate'] * destr

                if self.train_opts['average_weight']:
                    self.average_weight()

                if self.train_opts['average_filler_bias']:
                    self.average_filler_bias()

                if self.train_opts['update_bowl_strength']:
                    self.update_bowl_strength()

                if self.train_opts['coef_q'] > 0.:
                    qpolicy = self.qpolicy + \
                        self.train_opts['lrate'] * dqpolicy
                    for ii in range(1, len(self.qpolicy)):
                        qpolicy[ii] = max(qpolicy[ii], self.qpolicy[ii - 1])
                    self.qpolicy = qpolicy

                self.reset()    # reset q val
                self.get_ep(method=self.train_opts['ep_method'])

            # print('Check', np.max(abs(dWC)))

            dWC_max = np.max(abs(dWC))
            dbC_max = np.max(abs(dbC))

            if 'report_cycle' in self.train_opts:
                report_cycle = self.train_opts['report_cycle']
            else:
                report_cycle = 1

            if self.epoch_num % report_cycle == 0:
                print('[{:04d}]'.format(self.epoch_num), end='')
                print('{:.3f}'.format(kl['trees']).rjust(9), end=' | ')
                # print('{:.3f}'.format(xent['trees']).rjust(9), end='')
                # print('{:.3f}'.format(xent['treelets']).rjust(9), end=' | ')
                # print('{:.3f}'.format(xent['binding_pairs']).rjust(9), end='')
                # print('{:.3f}'.format(xent['bindings']).rjust(9), end=' | ')
                for prob_sent_report in prob_sent_report_list:
                    print(' '.join([
                        '{:.3f}'.format(prob)
                        for pi, prob in enumerate(prob_sent_report)
                        if pi < 6]), end='')
                # for prob_sent_report in prob_sent_report_list:
                #     print(' '.join(['{:.3f}'.format(prob) for prob in prob_sent_report]), end=' | ')
                # print(' '.join(['{:.3f}'.format(np.array(q).mean()) for q in self.qpolicy]), end=' ')
                # print('< {:.1f}'.format(self.opts['q_max']))

                prob_sum = 0.
                for prob_sent_report in prob_sent_report_list:
                    for prob in prob_sent_report:
                        prob_sum += prob
                print(' | {:.3f}'.format(prob_sum), end='')

                if self.train_opts['parallel_parser_train']:
                    print(' | {:.3f}'.format(acc), end='')

                print(' | {:.3f} {:.3f}'.format(dWC_max, dbC_max))

            prob_sent_report_list = np.array(prob_sent_report_list)

            log = {}
            log['WC'] = self.WC
            log['bC'] = self.bC
            log['estr'] = self.estr.copy()
            # print(max(self.estr))
            log['prob_sent'] = prob_sent_report_list.flatten(order='C')
            log['acc'] = prob_sent_report_list.sum(axis=1)
            for key in xent:
                log['xent_' + key] = xent[key]
            for key in kl:
                log['kl_' + key] = kl[key]

            log['lrate'] = self.train_opts['lrate']
            log['num_trials'] = self.train_opts['num_trials']

            self.update_traces_train(log)

            if self.train_opts['adaptive_training']:
                if (not self.nonzero_all0) and np.all(np.array(self.traces_train['prob_sent']).sum(axis=0) > 0):
                    # Over the course of training, every parse tree was generated at once.
                    self.nonzero_all0 = True
                    self.train_opts['lrate'] *= 0.1
                if (not self.nonzero_all1) and np.all(self.traces_train['prob_sent'][-1] > 0):
                    # At the last iteration, every parse tree was generated.
                    self.nonzero_all1 = True
                    self.train_opts['num_trials'] *= 2

        if hasattr(self, 'traces_train'):
            for key, val in self.traces_train.items():
                if isinstance(val, list):
                    self.traces_train[key] = np.array(val)

        if savefilename is not None:
            save_model(self, savefilename)

    def test(self, num_trials=10):

        num_sent = len(self.corpus['sentence'])
        max_sent_len = self.hg.opts['max_sent_len']
        test_res = np.zeros((num_trials, num_sent))
        dur = self.train_opts['dur']

        f_empty_type = self.hg.g.get_types(self.hg.g.opts['f_empty'])[0]
        null_binding_input = f_empty_type + self.hg.opts['bsep'] + '(1,{})'

        for si, sent in enumerate(self.corpus['sentence']):
            sent_acc = 0.
            if len(sent) < max_sent_len:
                null_input = [
                    null_binding_input.format(ii)
                    for ii in range(len(sent) + 1, max_sent_len + 1)]
                sent += null_input

            targ = self.corpus['target'][si]

            for ti in range(num_trials):
                self.reset(self.ep, 0.02)
                self.set_input(sent)
                if self.opts['use_runC']:
                    self.runC(dur)
                else:
                    self.run(dur)
                gp = self.read_grid_point()
                self.set_discrete_state(gp)
                if self.actC.dot(targ) == self.num_roles:
                    test_res[ti, si] = 1.
                    sent_acc += 1.
                else:
                    test_res[ti, si] = 0.

            sent = ' '.join([bname.split('/')[0] for bname in sent])

            print('Sentence {:d} ACC = {:.3f} ({:s})'.format(
                si, sent_acc/num_trials, sent))


def save_model(net, filename):
    f = open(filename, 'wb')
    pickle.dump(net, f)
    f.close()


def load_model(filename):
    f = open(filename, 'rb')
    net = pickle.load(f)
    f.close()
    return net


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + \
            (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        # Anchor the last smoothed value
        last = smoothed_val

    return np.array(smoothed)


def maxeig(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    return max(eigvals)


def unique(fillers):
    '''Returns (list) of unique names of fillers (list of str)'''

    fillers = list(set(fillers))
    fillers.sort()
    return fillers


def plot_TP(vec1, vec2, figsize=None, label=None):
    '''Plots the outer product of vec1 and vec2.

    Following a convention used in Smolensky & Legendre (2006),
    the outer product is plotted in a group of units (circles)
    arranged in a matrix form. Circles with two borders indicate
    that their corresponding components have negative values.
    Circles with single borders indicate those components have
    positive values. Face color represents the magnitude of the
    components. Note that the left column and the top row present
    vec1 and vec2, respectively.

    Args:
        vec1 and vec2: (1d NumPy arrays)
        figsize: (tuple) figure size

    Examples:
        >> vec1 = np.array([-0.2, 0.1])
        >> vec2 = np.array([0.2, 0.7, -0.1])
        >> plot_TP(vec1, vec2)
    '''

    nrow = vec1.shape[0]
    ncol = vec2.shape[0]
    radius = 0.4

    arr = np.zeros((nrow + 1, ncol + 1))
    arr[1:, 1:] = np.outer(vec1, vec2)
    arr[0, 1:] = vec2
    arr[1:, 0] = vec1

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    for ii in range(nrow + 1):
        for jj in range(ncol + 1):
            if (ii == 0) and (jj == 0):
                continue
            if (ii == 0) or (jj == 0):
                alpha = 1  # 0.3
            else:
                alpha = 1

            if arr[ii, jj] >= 0:
                curr_unit = plt.Circle(
                    (jj, -ii), radius,
                    color=plt.cm.gray(1 - abs(arr[ii, jj])),
                    alpha=alpha)
                ax.add_artist(curr_unit)
                curr_unit = plt.Circle(
                    (jj, -ii), radius,
                    color='k', fill=False)
                ax.add_artist(curr_unit)
            else:
                curr_unit = plt.Circle(
                    (jj, -ii), radius,
                    color='k', fill=False)
                ax.add_artist(curr_unit)
                curr_unit = plt.Circle(
                    (jj, -ii), radius - 0.1,
                    color=plt.cm.gray(1 - abs(arr[ii, jj])),
                    alpha=alpha)
                ax.add_artist(curr_unit)
                curr_unit = plt.Circle(
                    (jj, -ii), radius - 0.1,
                    color='k', fill=False)
                ax.add_artist(curr_unit)

    ax.axis([
        0 - radius - 0.6, ncol + radius + 0.6,
        - nrow - radius - 0.6, 0 + radius + 0.6])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def heatmap(data, xlabel=None, ylabel=None,
            xticklabels=None, yticklabels=None,
            xtick=True, ytick=True, val_range=None,
            rotate_xticklabels=True, disp=True,
            grayscale=False, cmap='Reds', colorbar=True,
            figsize=None, fontsize=16):
    """Plots data (2d NumPy array) in a heatmap.

    Args:
        data: (2d NumPy array) of numeric values
        xlabel: (str) x-axis label
        ylabel: (str) y-axis label
        xticklabels: (list) of str
        yticklabels: (list) of str
        xtick: (bool) Plot tick markers on x-axis
        ytick: (bool) Plot tick markers on y-axis
        rotate_xticklabels: (bool) 'True' to rotate xtick labels (90 deg)
        val_range: (tuple) of two numeric values
        cmap: (str) name of color map, default: 'Reds'
        grayscale: (bool) whether to use grayscale colors
        colorbar: (bool) whether to show colorbar
        figsize: (tuple) figure size

    Examples:
        >> heatmap(data, val_range=(0, 1))  # fix colorbar scale
    """

    if grayscale:
        cmap = plt.cm.get_cmap("gray_r")
    else:
        cmap = plt.cm.get_cmap(cmap)

    if figsize is None:
        plt.subplots()
    else:
        plt.subplots(figsize=figsize)

    if val_range is not None:
        plt.imshow(data, cmap=cmap, vmin=val_range[0], vmax=val_range[1],
                   interpolation="nearest", aspect='auto')
    else:
        plt.imshow(data, cmap=cmap, interpolation="nearest", aspect='auto')

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)

    if xticklabels is None:
        xticklabels = list(range(data.shape[1]))
        xticklabels = [str(x) for x in xticklabels]

    if rotate_xticklabels:
        plt.xticks(
            np.arange(len(xticklabels)), xticklabels,
            rotation='vertical')
    else:
        plt.xticks(np.arange(len(xticklabels)), xticklabels)

    if yticklabels is None:
        yticklabels = list(range(data.shape[0]))
        yticklabels = [str(x) for x in yticklabels]

    plt.yticks(np.arange(len(yticklabels)), yticklabels)

    if xtick is False:
        plt.tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom='off',       # ticks along the bottom edge are off
            top='off',          # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
    if ytick is False:
        plt.tick_params(
            axis='y',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            left='off',         # ticks along the bottom edge are off
            right='off',        # ticks along the top edge are off
            labelleft='off')    # labels along the bottom edge are off

    if colorbar:
        plt.colorbar()

    if disp:
        plt.show()


def compute_cossim(vec1, vec2):
    '''Computes cosine of the angle between vec1 and vec2 (1d NumPy arrays).'''

    return (vec1.dot(vec2) /
            (np.sqrt(vec1.T.dot(vec1)) * np.sqrt(vec2.T.dot(vec2))))


def compute_similarity(mat):
    '''Computes pairwise similarity (dot products) of column vectors in mat.

    Args:
        mat: 2d NumPy array

    Returns:
        A 2d NumPy array in which the (i,j)-th component is a dot product
        of the i-th and j-th column vectors of mat.
    '''

    n_cols = mat.shape[1]
    sim_mat = np.eye(n_cols)
    for ii in range(n_cols - 1):
        for jj in range(1, n_cols):
            sim_mat[jj, ii] = sim_mat[ii, jj] = mat[:, ii].dot(mat[:, jj])
    return sim_mat


def encode_symbols(num_symbols, coord='dist', dp=0., dim=None, seed=None):
    """Generates vector encodings of num_symbols symbols.

    Column vectors are the encodings of symbols.

    Args:
        num_symbols: int, number of symbols to encode
        coord: string, 'dist' or (distributed representation, default)
            or 'local' (local representation)
        dp: float (0 [default] <= dp <= 1) or 2D-numpy array of
            pairwise similarity (dot product)
        dim: int, number of dimensions to encode a symbol.
            must not be smaller than [num_symbols]

        dp and dim will be ignored if coord is set to 'local' or 'C'.

    Returns:
        A 2d NumPy array. Each column vector is a unique representation
        of a symbol.

    Usage:
        >>> gsc.encode_symbols(2)
        >>> gsc.encode_symbols(3, dp=0.3, dim=5)
        >>> gsc.encode_symbols(3, seed=100)
    """

    if coord == 'local' or coord == 'C':
        sym_mat = np.eye(num_symbols)

    else:
        if dim is None:
            dim = num_symbols
        else:
            if dim < num_symbols:
                message = ("The [dim] value must be same as or "
                           "greater than the [num_symbols] value.")
                sys.exit(message)

        if isinstance(dp, numbers.Number):
            # if dp is number, convert it to a 2d NumPy array in which
            # all diagonal components have the value of dp and all
            # off-diagonal components have a value of 0.
            dp = (dp * np.ones((num_symbols, num_symbols)) +
                  (1 - dp) * np.eye(num_symbols, num_symbols))

        sym_mat = dot_products(dp_mat=dp, dim=dim, seed=seed)

    return sym_mat


def dot_products(dp_mat, dim, max_iter=100000, seed=None, tol=1e-6):
    """Returns a 2D NumPy arrays of random numbers (float) such that
    pairwise dot products of column vectors are close to dp_mat
    (2d NumPy array [square matrix]) within tolerance (1e-6).

    Don Matthias wrote the original script in MATLAB for the LDNet program.
    He explains how this program works as follows:

    Given square matrix dpMatrix of dimension N-by-N, find N
    dim-dimensional unit vectors whose pairwise dot products match
    dpMatrix. Results are returned in the columns of M. itns is the
    number of iterations of search required, and may be ignored.

    Algorithm: Find a matrix M such that M'*M = dpMatrix. This is done
    via gradient descent on a cost function that is the square of the
    frobenius norm of (M'*M-dpMatrix).

    NOTE: It has trouble finding more than about 16 vectors, possibly for
    dumb numerical reasons (like stepsize and tolerance), which might be
    fixable if necessary.

    Args:
        dim: (int) dimensionality of vectors
        dp_mat: (2d NumPy arrays) of pairwise dot products (similarity)
        max_iter: (int) maximum number of iterations
        seed: (int) seed number for reproducibility

    Returns:
        A dim-by-num_symbols NumPy array of floats. Column vectors
        are the representation vectors of num_symbols unique symbols.

    Precondition:
        dp_mat must be a symmetric square matrix.
        dim must be equal to or greater than num of columns of dp_mat.
    """

    # TOL = 1e-6
    num_symbols = dp_mat.shape[0]

    if dim < num_symbols:
        sys.exit('dim must be equal to or greater than num_symbols.')

    if seed is not None:
        np.random.seed(seed)

    # if not (dp_mat.T == dp_mat).all():
    if not np.allclose(dp_mat.T, dp_mat):
        sys.exit('dot_products: dp_mat must be symmetric')

    if (np.diag(dp_mat) != 1).any():
        sys.exit(('dot_products: dp_mat must have '
                  'all ones on the main diagonal'))

    sym_mat = np.random.uniform(
        size=dim * num_symbols).reshape(dim, num_symbols, order='F')
    min_step = .1
    converged = False

    for iter_num in range(1, max_iter + 1):
        inc = sym_mat.dot(sym_mat.T.dot(sym_mat) - dp_mat)
        step = min(min_step, .01 / abs(inc).max())
        sym_mat = sym_mat - step * inc
        max_diff = abs(sym_mat.T.dot(sym_mat) - dp_mat).max()
        if max_diff <= tol:
            converged = True
            break

    if not converged:
        print("Didn't converge after {} iterations".format(max_iter))

    return sym_mat


def find_nearest_vector(array, value):
    idx = np.array([np.linalg.norm(x+y) for (x, y) in array-value]).argmin()
    return array[idx]


def compute_dist(net, ref_points, metric=['euclidean', 'cos', 'dp'][0]):
    act_trace = net.C2N(net.traces['actC'].T).T
    ref_points = net.C2N(ref_points.T).T
    dist = np.zeros((act_trace.shape[0], ref_points.shape[0]))
    if metric == 'euclidean':
        for pi, point in enumerate(ref_points):
            dist[:, pi] = np.sqrt(
                ((act_trace - point[None, :])**2).sum(axis=1))
        dist /= net.num_units
    elif metric == 'dp':
        for pi, point in enumerate(ref_points):
            dist[:, pi] = (act_trace * point[None, :]).sum(axis=1)
        dist /= net.num_roles
    elif metric == 'cos':
        for pi, point in enumerate(ref_points):
            dp = (act_trace * point[None, :]).sum(axis=1)
            mag1 = np.sqrt(np.sum(act_trace ** 2, axis=1))
            mag2 = np.sqrt(np.sum(point ** 2))
            dist[:, pi] = dp / (mag1 * mag2)

    return dist


# Zipf distribution
def zipf(N, s=1):
    res = np.zeros(N)
    for k in range(1, N + 1):
        res[k-1] = (1/k**s)
    return res / res.sum()


def plot_train_result(net, weight=0., normalize=False, ylim_kl=None, ylim_acc=[0., 1.],
                      linewidth=1, legend=True, savefilename_prefix=None, log_y=False):

    nsent_per_iteration = net.train_opts['num_trials'] + \
        net.train_opts['parallel_parser_num_trials']

    # Plot KL divergence
    xval = np.arange(
        len(net.traces_train['kl_trees'])) * nsent_per_iteration
    # KL was computed using ema prob estimate (Do not smooth again)
    plt.plot(xval, net.traces_train['kl_trees'], linewidth=linewidth)
    plt.grid()
    plt.xlabel('# of sentences', fontsize=15)
    plt.ylabel('KL divergence', fontsize=15)
    if ylim_kl is not None:
        plt.ylim(ylim_kl)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if savefilename_prefix is not None:
        plt.savefig(savefilename_prefix + '-kl.pdf')
    # plt.show()

    # Plot accuracy
    xval = np.arange(len(net.traces_train['acc'])) * nsent_per_iteration
    plt.plot(xval, smooth(
        net.traces_train['acc'], weight=weight), linewidth=linewidth)
    plt.ylim(0, 1)
    plt.xlabel('# of sentences', fontsize=15)
    plt.ylabel('Production accuracy', fontsize=15)
    plt.grid()
    if ylim_acc is not None:
        plt.ylim(ylim_acc)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if savefilename_prefix is not None:
        plt.savefig(savefilename_prefix + '-acc.pdf')
    # plt.show()

    if savefilename_prefix is None:
        savefilename = None
    else:
        savefilename = savefilename_prefix + '-prob.pdf'
    plot_prob_trees_trace(net, weight=weight, normalize=normalize,
                          savefilename=savefilename, linewidth=linewidth, legend=legend, log_y=log_y)


def plot_prob_trees_trace(
        net, normalize=False, weight=0.,
        xunit=[None, 'num_trials'][1], savefilename=None,
        legend=True, linewidth=1, log_y=False):

    nsent_per_iteration = net.train_opts['num_trials'] + \
        net.train_opts['parallel_parser_num_trials']

    sent0 = []
    for sent in net.corpus['sentence']:
        sent0.append(' '.join([bname.split('/')[0] for bname in sent]))

    ptarg = net.corpus['prob_sent']
    yy = net.traces_train['prob_sent']

    if xunit is None:
        xx = np.arange(len(yy))
        xlab = '# of updates'
    elif xunit == 'num_trials':
        # It is assumed that num_trials was fixed over the course of training
        xx = np.arange(len(yy)) * nsent_per_iteration
        xlab = '# of sentences'

    if normalize:
        acc = net.traces_train['acc']  # 2d-array
        # first_nonzero = np.where(net.traces_train['acc'] > 0)[0][0]
        yy = yy / (acc + 1e-15)  # prevent zero division

    yy = smooth(yy, weight)

    if log_y:
        yy = np.log(yy)

    for si, sent in enumerate(net.corpus['sentence']):
        if log_y:
            yy1 = np.log(ptarg[si])
        else:
            yy1 = ptarg[si]
        plt.axhline(yy1, linestyle='--', c='C%d' % (si % 10))
        plt.plot(xx, yy[:, si],  # / yy.sum(axis=1),
                 label=sent0[si], color='C%d' % (si % 10), linewidth=linewidth)

    ylab = 'Sentence probability'
    if log_y:
        ylab = 'Log sentence probability'

    if normalize:
        ylab += '\n(normalized)'
    plt.ylabel(ylab, fontsize=15)
    plt.xlabel(xlab, fontsize=15)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if savefilename is not None:
        plt.savefig(savefilename)
    # plt.show()


def parse(net, estr=2, slen=None, apply_time_constant=False, null1=False,
          trained_only=True, uniform=True):

    if slen is None:
        min_slen = 1
        max_slen = net.hg.opts['max_sent_len']
    else:
        min_slen = slen
        max_slen = slen
    net.estr = np.ones(net.num_bindings) * estr

    if trained_only:
        n_sent = len(net.corpus['sentence'])
        if uniform:
            p = np.ones(n_sent) / n_sent
        else:
            p = net.corpus['prob_sent']

        if net.hg.opts['use_same_len']:
            f_empty_type = net.hg.g.get_types(net.hg.g.opts['f_empty'])[0]
        else:
            print('CHECK')
            f_empty_type = net.hg.g.opts['null']

        idx = np.random.choice(n_sent, 1, replace=True, p=p)
        sent = net.corpus['sentence'][idx[0]]
        targ = net.corpus['target'][idx[0]]
        sent += [f_empty_type + net.hg.opts['bsep'] + '(1,{})'.format(ii)
                 for ii in range(len(sent) + 1, net.hg.opts['max_sent_len'] + 1)]
    else:
        sent, targ, p = net.generate_sentence(
            add_null_input=True, min_sent_len=min_slen, max_sent_len=max_slen)

    if null1:
        sent = sent[:min_slen+1]
    print('Input sentence =', ' '.join(
        [bname.split('/')[0] for bname in sent]))
    net.reset(net.ep, 0.02)

    if apply_time_constant:
        net.opts['scale_type'] = 'lv'
        net.opts['scaling_factor'] = 0.1
        net.update_scale_constants(pos=1, lv=1)

    net.set_input(sent)
    net.runC(15)
    net.plot_tree2(scale=1.5)
    if hasattr(net, 'get_discrete_state'):
        actC = net.get_discrete_state(net.read_grid_point())
    else:
        idx = self.find_bindings(binding_names)
        actC = np.zeros(self.num_bindings)
        actC[idx] = 1.0

    if np.allclose(actC, targ):
        print('Correct')
    else:
        print('False')

    return net


def parse2(net, dq, num_trials=1, estr=2, estr_null=1, slen=None,
           decay_factor=0., scaling_factor=2, scaling_symmetric=False,
           null_input_extend_pos=True, null_input_extend_lv=True, plot_tree=False):

    # dq = np.array([0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1])

    if slen is None:
        min_slen = 1
        max_slen = net.hg.opts['max_sent_len']
    else:
        min_slen = slen
        max_slen = slen
    sent, targ, p = net.generate_sentence(
        add_null_input=False, min_sent_len=min_slen, max_sent_len=max_slen)

    net.estr = np.ones(net.num_bindings) * estr
    net.opts['scale_type'] = 'pos'
    net.opts['scaling_factor'] = scaling_factor

    print('Input sentence =', ' '.join(
        [bname.split('/')[0] for bname in sent]))

    for ti in range(num_trials):

        net.reset(net.ep, 0.02)

        for bi, bname in enumerate(sent):
            net.update_scale_constants(bi + 1, symmetric=scaling_symmetric)
            net.extC *= decay_factor
            net.set_input(bname, cumulative=True)
            # print(bname)
            net.runC(dq[bi])
            # net.plot_tree2(scale=1.5)

        # null_input = ['@/(1,{})'.format(ii) for ii in range(len(sent) + 1, net.hg.opts['max_sent_len'] + 1)]
        # net.update_scale_constants(pos=0)
        # net.set_input(null_input)
        set_null_input(net, estr=estr_null, pos=len(sent) + 1,
                       extend_pos=null_input_extend_pos, extend_lv=null_input_extend_lv)
        net.update_scale_constants(pos=0)
        net.runC(net.opts['q_max'] - net.q[0])

        if plot_tree:
            net.plot_tree2(scale=1.5)
        net.set_discrete_state(net.read_grid_point())
        if np.allclose(net.actC, targ):
            print('Correct')
        else:
            print('False')

    return net


def convert_sentence(sent, term2word):

    sent0 = []
    for bname in sent:
        fname, rname = bname.split('/')
        if fname in term2word:
            fnames = term2word[fname]
        else:
            print("Can't find {} in `term2word`".format(fname))

        bnames = [fname + '/' + rname for fname in fnames]
        sent0.append(bnames)

    return sent0


def test_parse_inc(net, dq, num_sent=None, num_trials=10,
                   estr=2, estr_null=2, term2word=None,
                   symmetric=False,
                   decay_factor=0.5, scaling_factor=2,
                   update_q_mask=True,
                   update_scale_constants=False,
                   use_multiple_timescale=False,
                   wrapup_clear_input=False,
                   null_input_extend_pos=True,
                   null_input_extend_lv=True, disp=False):

    if num_sent is None:
        num_sent = len(net.corpus['sentence'])
    else:
        num_sent = min(num_sent, len(net.corpus['sentence']))
    max_sent_len = net.hg.opts['max_sent_len']
    res = {}

    f_empty_type = net.hg.g.get_types(net.hg.opts['f_empty'])

    net.qpolicy = dq.cumsum()
    net.qpolicy = np.insert(net.qpolicy, 0, 0.)

    for si in range(num_sent):

        sent = net.corpus['sentence'][si]
        targ = net.corpus['target'][si]
        sent_acc = 0.

        # Remove the empty position filler in 'sent'
        sent0 = [bname for bname in sent
                 if bname.split(net.hg.opts['bsep'])[0] not in f_empty_type]

        if term2word is not None:
            sent0 = convert_sentence(sent=sent0, term2word=term2word)

        res[si] = {}
        res[si]['sentence'] = sent0
        res[si]['parse_corr'] = targ
        res[si]['acc'] = 0.
        res[si]['parse_incorr'] = []
        # print(sent0)

        for ti in range(num_trials):
            net.run_sent(
                sent0, decay_factor=decay_factor,
                symmetric=symmetric,
                update_q_mask=update_q_mask,
                update_scale_constants=update_scale_constants,
                use_multiple_timescale=use_multiple_timescale,
                wrapup_clear_input=wrapup_clear_input,
                null_input_extend_pos=null_input_extend_pos,
                null_input_extend_lv=null_input_extend_lv, disp=disp)

            net.set_discrete_state(net.read_grid_point())
            if np.allclose(net.actC, targ):
                sent_acc += 1.
            else:
                res[si]['parse_incorr'].append(net.actC)

        res[si]['acc'] = sent_acc/num_trials
        res[si]['parse_incorr'] = np.array(res[si]['parse_incorr'])

        sent = ' '.join([bname.split('/')[0] for bname in sent])
        print('Sentence {:d} ACC = {:.3f} ({:s})'.format(
            si, res[si]['acc'], sent))

    return res


def get_weight_matrix(net, rname1, rname2):
    # from rname1 to rname2
    bid1 = net.find_roles(rname1)
    bid2 = net.find_roles(rname2)
    return net.WC[np.ix_(bid2, bid1)]


def plot_weight_matrix(net, rname1, rname2):
    # from rname1 to rname2
    WC = get_weight_matrix(net, rname1, rname2)
    heatmap(WC,
            xticklabels=net.filler_names,
            yticklabels=net.filler_names,
            xlabel=rname1,
            ylabel=rname2,
            rotate_xticklabels=True)


def plot_bias(net):
    if net.opts['use_second_order_bias']:
        bCmat = net.vec2mat(np.diag(net.WC))
        heatmap(bCmat,
                xticklabels=net.role_names,
                yticklabels=net.filler_names,
                xlabel='Roles',
                ylabel='Fillers',
                rotate_xticklabels=True)


def report_stat(net, decimals=3):
    if hasattr(net, 'traces_train'):
        if 'prob_sent' in net.traces_train:
            for si, sent in enumerate(net.corpus['sentence']):
                sent = [bname.split('/')[0] for bname in sent]
                pval = '{0:.{1}f}'.format(
                    net.corpus['prob_sent'][si], decimals)
                qval = '{0:.{1}f}'.format(
                    net.traces_train['prob_sent'][-1, si], decimals)
                print('Sentence {:d}: p = {}, q = {} ({})'.format(
                    si + 1, pval, qval, ' '.join(sent)))


def report_train_result(net, num_epochs=100, report_nsent=7, decimals=4):

    print('Number of fillers = {}'.format(net.num_fillers))
    print('Number of roles = {}'.format(net.num_roles))
    print('Number of bindings = {}'.format(net.num_bindings))

    print('Maximum sentence length = {}'.format(net.hg.opts['max_sent_len']))
    print('Number of sentences = {}'.format(len(net.corpus['sentence'])))

    print('beta (bowl strength) = {:.3f}'.format(net.opts['bowl_strength']))
    print('m (competition strength) = {:.3f}'.format(net.opts['m']))
    print('dt = {}'.format(net.opts['dt_init']))
    print('dq/dt = {}'.format(net.opts['q_rate']))
    print('T = {:.3f}'.format(net.opts['T_init']))

    print('KL: M = {:.3f}, SD = {:.3f}'.format(
        net.traces_train['kl_trees'][-num_epochs:].mean(),
        net.traces_train['kl_trees'][-num_epochs:].std()))
    print('ACC: M = {:.3f}, SD = {:.3f}'.format(
        net.traces_train['acc'][-num_epochs:].mean(),
        net.traces_train['acc'][-num_epochs:].std()))

    num_sent = min(report_nsent, len(net.corpus['sentence']))
    for si in range(num_sent):
        sent = net.corpus['sentence'][si]
        sent = [bname.split('/')[0] for bname in sent]
        pval = '{0:.{1}f}'.format(net.corpus['prob_sent'][si], decimals)
        qval = '{0:.{1}f}'.format(
            net.traces_train['prob_sent'][-num_epochs:, si].mean(axis=0), decimals)
        print('Sentence {:d}: p = {}, q = {} ({})'.format(
            si + 1, pval, qval, ' '.join(sent)))


def compare_models(
        net_list, ylim_kl=None, ylim_acc=[0., 1.], weight_kl=0.,
        weight_acc=0., weight_prob_sent=0., normalize=False, net_labs=None):
    # Temporary

    if net_labs is not None:
        assert len(net_labs) == len(net_list)
    else:
        net_labs = ['M{:02d}'.format(ii + 1) for ii in range(len(net_list))]

    # KL divergence
    for neti, net in enumerate(net_list):
        nsent_per_iteration = net.train_opts['num_trials'] + \
            net.train_opts['parallel_parser_num_trials']
        xval = np.arange(
            len(net.traces_train['kl_trees'])) * nsent_per_iteration
        plt.plot(xval, smooth(net.traces_train['kl_trees'], weight_kl),
                 label=net_labs[neti])
    plt.grid()
    plt.ylabel('KL divergence', fontsize=15)
    plt.xlabel('# of sentences', fontsize=15)
    plt.legend()
    if ylim_kl is not None:
        plt.ylim(ylim_kl[0], ylim_kl[1])
    plt.show()

    # accuracy
    xval = np.arange(len(net.traces_train['acc'])) * nsent_per_iteration
    for neti, net in enumerate(net_list):
        nsent_per_iteration = net.train_opts['num_trials'] + \
            net.train_opts['parallel_parser_num_trials']
        xval = np.arange(len(net.traces_train['acc'])) * nsent_per_iteration
        plt.plot(xval, smooth(net.traces_train['acc'], weight=weight_acc),
                 label=net_labs[neti])
    plt.ylim(ylim_acc[0], ylim_acc[1])
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('# of sentences', fontsize=15)
    plt.grid()
    plt.legend()
    plt.show()

    for neti, net in enumerate(net_list):
        plt.title(net_labs[neti])
        plot_prob_trees_trace(net, normalize=normalize,
                              weight=weight_prob_sent)
        plt.show()

        for si, sent in enumerate(net.corpus['sentence']):
            sent = [bname.split('/')[0] for bname in sent]
            print('Sentence {:d}: p = {:.3f}, q = {:.3f} ({})'.format(
                si + 1,
                net.corpus['prob_sent'][si],
                net.traces_train['prob_sent'][-1, si],
                ' '.join(sent)))


def set_null_input(
        net, pos, estr=1, extend_pos=False, extend_lv=False,
        cumulative=False):  # , use_type=False):

    if extend_lv:
        lv = range(1, net.hg.opts['max_sent_len'] + 1)
    else:
        lv = [1]

    bnames = []

    for ll in lv:

        if extend_pos:
            pos0 = range(pos - ll + 1, net.hg.opts['max_sent_len'] - ll + 2)
        else:
            pos0 = [pos]

        # print(ll, pos0)

        for pp in pos0:

            if pp >= 1:

                rname = '({},{})'.format(ll, pp)

                if (ll > 1) and (pp == 1):
                    fname = net.hg.opts['f_root']

                elif (ll > 1) and (pp > 1):
                    fname = net.hg.opts['f_empty_copy']
                else:
                    fname = net.hg.opts['f_empty']

                bname = fname + net.hg.opts['bsep'] + rname
                bnames.append(bname)

    # print(bnames)
    net.estr_backup = net.estr.copy()
    net.estr = np.ones(net.num_bindings) * estr
    net.set_input(bnames, cumulative=cumulative, use_type=False)
    net.estr = net.estr_backup.copy()


def train_inc_parser(net, num_trials, lrate=0.1):
    bsep = net.hg.opts['bsep']
    max_slen = net.hg.opts['max_sent_len']
    acc = []
    if 'acc_inc' not in net.traces_train:
        net.traces_train['acc_inc'] = []
    else:
        net.traces_train['acc_inc'] = list(net.traces_train['acc_inc'])

    for ti in range(num_trials):
        pos_wrong = -1
        sent, targ, p = net.generate_sentence(add_null_input=False)
        net.run_sent(sent, disp=False)
        gp = net.read_grid_point(disp=False)

        net.set_discrete_state(gp)
        if np.allclose(net.actC, targ):
            acc.append(1.)
        else:
            acc.append(0.)

        for bi, bname in enumerate(sent):
            ftype1 = bname.split(bsep)[0]  # ignore tokens
            fname2 = gp[bi].split(bsep)[0]
            ftype2 = net.hg.g.get_types(fname2)[0]
            if ftype1 != ftype2:
                # print(ftype1, ftype2)
                pos_wrong = bi
                break

        if pos_wrong >= 0:
            #             print('FOUND')
            #             print(sent, bi + 1, ftype1, ftype2)
            #             for ii in range(pos_wrong, max_slen):

            net.qpolicy[:pos_wrong + 1] -= 0.01 * \
                np.arange(pos_wrong + 1)/(pos_wrong + 1) * lrate  # previous
            net.qpolicy[pos_wrong + 1] += 0.01 * lrate  # previous
#             for ii in range(0, pos_wrong):
# #                 print('Updating')
#                 net.qpolicy[ii + 1] -= 0.01 * (ii + 1) * lrate
#                 print(net.qpolicy)

            for ii in range(1, len(net.qpolicy)):
                if net.qpolicy[ii-1] >= net.qpolicy[ii]:
                    net.qpolicy[ii] = net.qpolicy[ii-1]
            # for ii in range(1, len(net.qpolicy)):
            #     if net.qpolicy[-ii-1] >= net.qpolicy[-ii]:
            #         net.qpolicy[-ii-1] = net.qpolicy[-ii]

            for ii in range(len(net.qpolicy)):
                if net.qpolicy[ii] < 0:
                    net.qpolicy[ii] = 0
                if net.qpolicy[ii] >= net.opts['q_max']:
                    net.qpolicy[ii] = net.opts['q_max']

            net.qpolicy[0] = 0.

            print(ti, net.qpolicy)
        else:
            pass
            # net.qpolicy += 0.001 * np.arange(len(net.qpolicy)) * lrate

        # net.qpolicy[0] = max(min(0, net.qpolicy[0]), 0)
#         for ii in range(1, len(net.qpolicy)):
# #             print(-ii, -ii-1)
#             if net.qpolicy[-ii-1] >= net.qpolicy[-ii]:
#                 net.qpolicy[-ii] = net.qpolicy[-ii-1]
#         for ii in range(1, len(net.qpolicy)):
# #             print(-ii, -ii-1)
#             if net.qpolicy[ii-1] >= net.qpolicy[ii]:
#                 net.qpolicy[ii] = net.qpolicy[ii-1]
        # print(net.qpolicy)

    acc = np.array(acc)
    print('ACC = {:.3f}'.format(acc.mean()))
    # acc = acc.cumsum() / np.ones(num_trials).cumsum()
    net.traces_train['acc_inc'] += list(acc)
    net.traces_train['acc_inc'] = np.array(net.traces_train['acc_inc'])


def compute_metric(net, actC_trace, treelet):
    '''Return a trace of the similarity distance from each activation state to 
    the grid point of the treelet. The similarity distance is the normalized
    dot product; this is equivalent to the average of the activation values of 
    the constituent bindings of the treelet.'''
    roles = []
    for bname in treelet:
        roles.append(bname.split('/')[1])

    idx = net.find_bindings(treelet)
    dp = actC_trace[:, idx].sum(axis=1) / len(roles)

    return dp


def get_treelet_frame(net, rname):
    '''Return a list of roles that form a treelet whose mother position is `rname`'''
    children = net.hg.roles.get_daughters(rname)
    return [children['l'][0], children['r'][0], rname]


def get_treelets(net, rules, rname):
    '''Return all grammatical treelets at the position `rname`'''

    roleset = get_treelet_frame(net, rname)
    treelets = []

    for rule in rules:

        treelet = []

        if rule['d1'] is not None:
            treelet.append(rule['d1'] + '/' + roleset[0])

        if rule['d2'] is not None:
            treelet.append(rule['d2'] + '/' + roleset[1])

        treelet.append(rule['m'] + '/' + roleset[2])
        treelets.append(treelet)

    return treelets


def compute_treelet_act_trace(net, actC_trace, rules, rname):
    #     rules = net.hg.g.get_rules()
    treeletset = get_treelets(net, rules, rname)

    # bug in Grammar --- redundant rules
#     treeletset = set(tuple(treelet) for treelet in treeletset)
#     treeletset = [list(treelet) for treelet in treeletset]

    dp_all = []
    for treelet in treeletset:
        dp = compute_metric(net, actC_trace, treelet)
        dp_all.append(dp)

    return np.array(dp_all).T


# def rule2str(rule, add_prob=False, suppress_pos=False):
#     '''Print a rule in a succinct form'''
#     str1 = ''

#     if not suppress_pos:
#         if rule['d1'] is not None:
#             str1 += rule['d1']
#         str1 += '({})'.format(rule['m'])
#         if rule['d2'] is not None:
#             str1 += rule['d2']
#     else:
#         if rule['d1'] is not None:
#             str1 += rule['d1'].split(':')[0]
#         str1 += '({})'.format(rule['m'].split(':')[0])
#         if rule['d2'] is not None:
#             str1 += rule['d2'].split(':')[0]

#     if add_prob:
#         if rule['p'] is not None:
#             str1 += ' ({:.3f})'.format(rule['p'])

#     return str1


def rule2str(rule, add_prob=False, suppress_pos=False):
    '''Print a rule in a succinct form'''
    str1 = ''

    if not suppress_pos:
        str1 += rule['m']
        str1 += '('
        if rule['d1'] is not None:
            str1 += rule['d1']
        str1 += ','
        if rule['d2'] is not None:
            str1 += rule['d2']
        str1 += ')'
    else:
        str1 += rule['m'].split(':')[0]
        str1 += '('
        if rule['d1'] is not None:
            str1 += rule['d1'].split(':')[0]
        str1 += ','
        if rule['d2'] is not None:
            str1 += rule['d2'].split(':')[0]
        str1 += ')'

    if add_prob:
        if rule['p'] is not None:
            str1 += ' ({:.3f})'.format(rule['p'])

    return str1


def create_rule_labels(rules, add_prob=False, suppress_pos=False):
    labs = []
    for rule in rules:
        labs.append(rule2str(rule, add_prob=add_prob,
                    suppress_pos=suppress_pos))

    return labs


def check_acc(net, targ):
    gp1 = net.read_grid_point()
    gp2 = net.read_grid_point(actC=targ)
    return set(gp1) == set(gp2)


def get_windows(policy, sent):
    # Create windows info for annotation
    windows = []
    for wi, word in enumerate(sent):
        windows.append((policy[wi:wi+2].tolist(), word.split('/')[0]))

    return windows


def plot_treelet_act_trace(
        net, rname, num_treelets=4,
        tmin=0, tmax=1000, downsampling=30,
        suppress_pos=True, add_prob=False, legend_pos=None):

    # bug somewhere in gsc.py: contain multiple copies of augmented rules
    # remove redundancy
    rules0 = net.hg.g.get_rules()
    rules = []
    for rule in rules0:
        if rule not in rules:
            rules.append(rule)

    labs = create_rule_labels(rules, add_prob=add_prob,
                              suppress_pos=suppress_pos)

    idx = (net.traces['t'] >= tmin) * (net.traces['t'] <= tmax)
    actC_trace = net.traces['actC'][idx, :]
    dp_all = compute_treelet_act_trace(net, actC_trace, rules, rname)

    temp = np.argsort(dp_all.sum(axis=0))
    focus_idx = temp[::-1][:num_treelets]
    labs_focus = [labs[ii] for ii in focus_idx]

    plt.plot(net.traces['t'][idx][::downsampling],
             dp_all[::downsampling, focus_idx])
    if legend_pos is not None:
        plt.legend(labs_focus, loc=legend_pos)
    else:
        plt.legend(labs_focus)
    plt.ylim(0, 1)
    plt.xlabel('Time')
    plt.ylabel('Treelet activation at {}'.format(rname))

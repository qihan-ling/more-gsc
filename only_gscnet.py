import time
import numpy as np
import sys
import numbers
import copy
import pickle
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

    def _set_encodings(self):

        self.encodings = {}

        self.encodings['filler_names'] = None
        self.encodings['role_names'] = None

        self.encodings['seed_f'] = None
        self.encodings['seed_r'] = None

        self.encodings['coord_f'] = 'N'
        self.encodings['coord_r'] = 'N'

        self.encodings['dim_f'] = None
        self.encodings['dim_r'] = None

        self.encodings['F'] = None
        self.encodings['R'] = None

        self.encodings['dp_f'] = 0.
        self.encodings['dp_r'] = 0.

        self.encodings['similarity'] = None

    def _update_encodings(self, encodings):
        """Updates encodings."""
        if encodings is not None:
            for key in encodings:
                if key in self.encodings:
                    self.encodings[key] = encodings[key]
                else:
                    sys.exit('Cannot find `{}` in encodings.'.format(key))

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

    def _add_names(self):

        fnames = [val for rule in self.rules for key, val in rule.items()
                  if (key != 'p') and (val is not None)]
        fnames = list(set(fnames))
        fnames.sort()

        if self.opts['add_null']:
            fnames.append(self.opts['null'])

        self.filler_names = fnames

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
                self.bC[rid] += 1.

            self._set_biases()

    def bias2weight(self):
        '''Set recurrent weights given bias values in conceptual coordinates'''

        self.WC = self.WC + np.diag(2 * self.bC)
        self.bC = np.zeros(self.num_bindings)
        self._set_weights()
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

    def set_state(self, mu, sd=0.):

        noise_vec = np.random.normal(
            loc=0., scale=sd, size=self.num_bindings)
        self.actC = mu + noise_vec
        self.actCmat = self.vec2mat()
        self.act = self.C2N()

    def backup_parameters(self):

        self.params_backup = {}
        self.params_backup['encodings'] = copy.deepcopy(self.encodings)
        self.params_backup['WC'] = self.WC.copy()
        self.params_backup['bC'] = self.bC.copy()
        self.params_backup['estr'] = self.estr.copy()
        self.params_backup['ep'] = self.ep.copy()
        if hasattr(self, 'qpolicy'):
            self.params_backup['qpolicy'] = self.qpolicy.copy()

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

    def update_train_opts(self, train_opts):

        for key in train_opts:
            if key in self.train_opts:
                self.train_opts[key] = train_opts[key]
                if key in ['update_sister_harmony', 'update_gram_only']:
                    self.train_opts['mask0'] = self.get_mask0()
            else:
                sys.exit('`{}` is not supported.'.format(key))

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
        t_post = time.time()

        # Store continuous actC for return value
        actC_list = actC_batch.tolist()  # Fast batch conversion

        # OPTIMIZATION 1: Vectorized one-hot encoding
        # Instead of looping, use advanced indexing
        # grid_point_batch shape: (num_trials, num_roles)
        # We want: actC_discrete[trial, role_idx * num_fillers + filler_idx] = 1.0

        actC_discrete_batch = np.zeros((num_trials, self.num_bindings))
        role_indices = np.arange(self.num_roles)  # [0, 1, 2, ..., num_roles-1]

        for trial_id in range(num_trials):
            binding_indices = role_indices * self.num_fillers + \
                grid_point_batch[trial_id].astype(int)
            actC_discrete_batch[trial_id, binding_indices] = 1.0

        # OPTIMIZATION 2: Use dictionary with tuple keys for O(1) lookup Instead of list membership testing and list.index() which are O(n)

        state_counts = {}  # {tuple(grid_point): count}

        for trial_id in range(num_trials):
            # Use grid_point as hashable key (faster than comparing full one-hot vectors)
            gp_key = tuple(grid_point_batch[trial_id])

            if gp_key in state_counts:
                state_counts[gp_key] += 1
            else:
                state_counts[gp_key] = 1

        # Convert to corpus format
        corpus = {}
        corpus['target'] = []
        corpus['count'] = []
        for gp_key, count in state_counts.items():
            # Reconstruct one-hot from grid_point
            actC_discrete = np.zeros(self.num_bindings)
            for role_idx, filler_idx in enumerate(gp_key):
                # the binding name ordering is binding_names = [f + bsep + r for r in self.role_names for f in self.filler_names]
                # = [f0/r0, f1/r0, f2/r0, ..., f0/r1, f1/r1, ...]
                binding_idx = role_idx * self.num_fillers + int(filler_idx)
                actC_discrete[binding_idx] = 1.0

            # Aggregate using discrete states (like CPU version)
            corpus['target'].append(list(actC_discrete))
            corpus['count'].append(count)

        corpus['target'] = np.array(corpus['target'])
        corpus['count'] = np.array(corpus['count'])
        corpus['prob_sent'] = corpus['count'] / corpus['count'].sum()
        print(f"Post-processing time: {time.time() - t_post:.3f}s")

        stat = self.get_corpus_stat(corpus)
        return stat, np.array(actC_list)

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

    def find_roles(self, rnames):

        if not isinstance(rnames, list):
            rnames = [rnames]
        return [idx for idx, rname in enumerate(self.role_names)
                if rname in rnames]

    def find_bindings(self, bnames):

        if not isinstance(bnames, list):
            bnames = [bnames]

        return [bi for bi, bname in enumerate(self.binding_names)
                if bname in bnames]

    def find_fillers(self, fnames):
        '''Returns (list) of indices for fnames (str or list of str)'''

        if not isinstance(fnames, list):
            fnames = [fnames]
        return [fi for fi, fname in enumerate(self.filler_names)
                if fname in fnames]

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

    def update_traces_train(self, log):

        for key in self.train_opts['trace_varnames']:
            if key == 'WC':
                self.traces_train[key].append(log[key].flatten(order='F'))
            else:
                self.traces_train[key].append(log[key])

    def save_model(net, filename):
        f = open(filename, 'wb')
        pickle.dump(net, f)
        f.close()

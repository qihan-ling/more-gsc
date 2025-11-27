import time
import numpy as np
import sys
import numbers
import copy
import pickle
import matplotlib.pyplot as plt
try:
    from scipy import sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available - sparse matrices disabled")
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
from only_datastructure_speedup_sap import PCFG, Node, HarmonicGrammar, BrickRole


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


def load_model(filename):
    f = open(filename, 'rb')
    net = pickle.load(f)
    f.close()
    return net


def save_model(net, filename):
    f = open(filename, 'wb')
    pickle.dump(net, f)
    f.close()


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
                # message = ("The [dim] value must be same as or "
                #            "greater than the [num_symbols] value.")
                # sys.exit(message)
                print(
                    f"  Warning: Using dim={dim} < num_symbols={num_symbols}")
                print(
                    f"  This creates a compressed representation to avoid memory issues.")
                print(
                    f"  The optimization will try to approximate similarity structure in lower dimensions.")

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
        For exact solutions, dim must be >= num_symbols.
        For approximate solutions with dim < num_symbols, special cases are handled.
    """

    # TOL = 1e-6
    num_symbols = dp_mat.shape[0]

    # if dim < num_symbols:
    #     sys.exit('dim must be equal to or greater than num_symbols.')

    if seed is not None:
        np.random.seed(seed)

    # SPECIAL CASE: Orthogonal encodings (dp=0 off-diagonal) with dim < num_symbols
    # This is common for large grammars where memory is limited
    is_orthogonal = np.allclose(dp_mat, np.eye(num_symbols))

    if dim < num_symbols:
        if is_orthogonal:
            # For orthogonal case: use random unit vectors in lower dimensions
            # They won't be perfectly orthogonal but will be approximately so
            print(
                f"  Using approximate orthogonal encodings: {dim}D for {num_symbols} symbols")
            print(
                f"  Memory saved: {(num_symbols**2 - dim**2) * 8 / 1e9:.2f} GB per matrix")

            sym_mat = np.random.randn(dim, num_symbols)
            # Normalize columns to unit length
            sym_mat = sym_mat / np.linalg.norm(sym_mat, axis=0, keepdims=True)

            # Report approximation quality
            actual_dp = sym_mat.T.dot(sym_mat)
            max_off_diag = np.max(
                np.abs(actual_dp - np.diag(np.diag(actual_dp))))
            print(
                f"  Approximation quality: max off-diagonal dot product = {max_off_diag:.4f}")

            return sym_mat
        else:
            # Non-orthogonal case with dim < num_symbols: use random projection
            print(
                f"  Warning: Using random projection for dim={dim} < num_symbols={num_symbols}")
            print(f"  Similarity structure will be approximated, not exact.")

            # Use random projection - won't match dp_mat exactly but is the best we can do
            sym_mat = np.random.randn(dim, num_symbols)
            sym_mat = sym_mat / np.linalg.norm(sym_mat, axis=0, keepdims=True)

            return sym_mat

    # STANDARD CASE: dim >= num_symbols, exact solution possible

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
    print(f"rules are {rules}")
    labs = create_rule_labels(rules, add_prob=add_prob,
                              suppress_pos=suppress_pos)
    print(f"labs are {labs}")
    idx = (net.traces['t'] >= tmin) * (net.traces['t'] <= tmax)
    actC_trace = net.traces['actC'][idx, :]
    dp_all = compute_treelet_act_trace(net, actC_trace, rules, rname)

    temp = np.argsort(dp_all.sum(axis=0))
    focus_idx = temp[::-1][:num_treelets]
    print(f"focus_idx is {focus_idx}")
    labs_focus = [labs[ii] for ii in focus_idx]
    print(f"labs_focus is {labs_focus}")
    plt.plot(net.traces['t'][idx][::downsampling],
             dp_all[::downsampling, focus_idx])
    if legend_pos is not None:
        plt.legend(labs_focus, loc=legend_pos)
    else:
        plt.legend(labs_focus)
    plt.ylim(0, 1)
    plt.xlabel('Time')
    plt.ylabel('Treelet activation at {}'.format(rname))


def create_rule_labels(rules, add_prob=False, suppress_pos=False):
    labs = []
    for rule in rules:
        labs.append(rule2str(rule, add_prob=add_prob,
                    suppress_pos=suppress_pos))

    return labs


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


def compute_metric(net, actC_trace, treelet):
    '''Return a trace of the similarity distance from each activation state to 
    the grid point of the treelet. The similarity distance is the normalized
    dot product; this is equivalent to the average of the activation values of 
    the constituent bindings of the treelet.'''
    roles = []
    for bname in treelet:
        roles.append(bname.split('/')[1])

    idx = net.find_bindings_fast(treelet)
    dp = actC_trace[:, idx].sum(axis=1) / len(roles)

    return dp


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
# =============================================================================
# JAX-ACCELERATED TRIAL EXECUTION
# =============================================================================
# These functions enable GPU-accelerated parallel trial execution


if JAX_AVAILABLE:
    from functools import partial

    @partial(jit, static_argnums=(6, 7, 8, 9))
    def _adam_update_jax(W, dW, M, R, step, mask, lrate, beta1, beta2, eps):
        """
        JIT-compiled Adam update with internal step tracking.

        Args:
            W: Current weights (JAX array)
            dW: Gradients (JAX array)
            M, R: Momentum states (JAX arrays)
            step: Step counter (JAX scalar)
            mask: Update mask (JAX array or None)
            lrate, beta1, beta2, eps: Hyperparameters (static)

        Returns:
            (W_new, M_new, R_new, step_new) - all JAX arrays
        """
        # Increment step
        step = step + 1

        # Update momentum
        M = beta1 * M + (1.0 - beta1) * dW
        R = beta2 * R + (1.0 - beta2) * (dW ** 2)

        # Bias correction
        m_hat = M / (1.0 - beta1 ** step)
        r_hat = R / (1.0 - beta2 ** step)

        # Compute update
        update = lrate * m_hat / (jnp.sqrt(r_hat) + eps)

        # Apply mask if provided
        if mask is not None:
            update = update * mask

        # Update weights
        W_new = W + update

        return W_new, M, R, step

    @partial(jit, static_argnums=(3,))
    def _sgd_update_jax(W, dW, mask, lrate):
        """JIT-compiled SGD update."""
        if mask is not None:
            update = lrate * dW * mask
        else:
            update = lrate * dW
        return W + update

    @jit
    def _lazy_s_multiply(C, C_T, vector, scale_constants):
        """
        Compute (C @ C.T) @ vector WITHOUT materializing C @ C.T.

        This is the KEY optimization that eliminates the 18 TB S matrix!

        Instead of:
            S = C @ C.T              # 18 TB matrix!
            result = S @ vector      # Matrix-vector multiply

        We compute:
            temp = C.T @ vector      # Small operation
            result = C @ temp        # Small operation

        Mathematically equivalent but uses ZERO extra memory!
        """
        temp = jnp.dot(C_T, vector)
        result = jnp.dot(C, temp)
        return scale_constants * result

    @partial(jit, static_argnums=(10,))
    def _dynamics_step_jax(actC, WC, bC, extC, bowl_center, bowl_strength,
                           scale_constants, C, C_T, N, num_fillers, dt, T,
                           q, q_max, q_rate, m, rng_key):
        """
        Single dynamics step for equilibrium finding.
        JIT-compiled for GPU acceleration.
        """
        # Compute gradient (HGradC logic)
        # Grammar term
        hgrad_g = jnp.dot(WC, actC) + bC + extC
        # Bowl term
        hgrad_b = bowl_strength * (bowl_center - actC)
        # Commitment term
        hgrad_q0 = -2 * jnp.repeat(q, num_fillers) * \
            actC * (1 - actC) * (1 - 2*actC)
        # Uniqueness term
        actCmat = actC.reshape((num_fillers, -1), order='F')
        ssq = jnp.sum(actCmat ** 2, axis=0)
        hgrad_q1 = -4 * m * actC * jnp.repeat(ssq - 1, num_fillers)

        hgrad = hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1

        # Project through similarity: gradC = C @ (C.T @ hgrad)
        gradC = _lazy_s_multiply(C, C_T, hgrad, scale_constants)

        # Euler integration
        actC_new = actC + dt * gradC

        # Add noise
        rng_key, subkey = jax.random.split(rng_key)
        num_units = N.shape[1]
        noise = jnp.sqrt(2 * T * dt) * jax.random.normal(subkey,
                                                         shape=(num_units,), dtype=jnp.float32)
        noiseC = jnp.sqrt(scale_constants) * jnp.dot(C, noise)
        actC_new = actC_new + noiseC

        # Update q
        q_new = q + q_rate * dt
        q_new = jnp.maximum(jnp.minimum(q_new, q_max), 0)

        return actC_new, q_new, rng_key

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
            # S = net_params['S'] avoid storing S, using C_T to do lazy multiplication
            C_T = net_params['C_T']
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
            # OLD (creates 18 TB S matrix): gradC = scale_const * (S @ HGradC_val)
            # NEW (lazy): gradC = scale_const * C @ (C.T @ HGradC_val)
            temp = C_T @ HGradC_val
            gradC = scale_const * (C @ temp)

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
        # NOTE: This function should only be called when net.WC is NOT sparse
        # The calling function (estimate_prob_inc_jax) checks for sparse matrices
        # and falls back to CPU version to avoid OOM errors
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
            # 'S': jnp.array(net.S),  # Inverse similarity matrix
            'C': jnp.array(net.C),
            'C_T': jnp.array(net.C_T),
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
        self.use_jax = JAX_AVAILABLE and self.opts.get('use_jax', True)
        self._add_change_of_basis_matrices()
        dur = time.time() - t0
        print('{} s for generating encodings'.format(dur))

        t0 = time.time()
        if self.hg is not None:
            # self._precompute_fast_lookups()
            self._precompute_fastER_lookups()
        # Auto-detect sparse matrix need
        if self.opts['use_sparse_wc'] is None:
            self.opts['use_sparse_wc'] = (
                self.num_bindings > self.opts['sparse_wc_threshold'] and
                SCIPY_AVAILABLE and
                not self.use_jax  # Sparse not yet supported with JAX
            )

        # Add parameters ==========================================
        if self.use_jax:
            if self.opts['use_sparse_wc']:
                sys.exit("ERROR: Sparse WC matrices not yet supported with JAX. "
                         "Please use CPU mode (use_jax=False) for large grammars.")

            print("Initializing parameters on GPU with JAX...")
            # Use float32 for GPU efficiency (good balance of speed/precision)
            self.WC = jnp.zeros(
                (self.num_bindings, self.num_bindings), dtype=jnp.float32)
            self.bC = jnp.zeros(self.num_bindings, dtype=jnp.float32)
            self.estr = jnp.ones(
                self.num_bindings, dtype=jnp.float32) * self.opts['init_estr']

            # Initialize JAX random key for efficient random number generation
            self.rng_key = jax.random.PRNGKey(seed if seed is not None else 0)

            # Initialize Adam states on GPU
            self.optim = {
                'M_WC': jnp.zeros_like(self.WC),
                'R_WC': jnp.zeros_like(self.WC),
                'M_bC': jnp.zeros_like(self.bC),
                'R_bC': jnp.zeros_like(self.bC),
                'step_WC': jnp.array(0, dtype=jnp.float32),
                'step_bC': jnp.array(0, dtype=jnp.float32),
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8
            }

            print(f"  ✓ WC on GPU: {self.WC.shape} ({self.WC.dtype})")
            print(f"  ✓ Memory: {self.WC.nbytes / 1e9:.2f} GB")
        else:
            print("Initializing parameters on CPU with NumPy...")
            # Use sparse matrix for large grammars
            if self.opts['use_sparse_wc']:
                dense_size_gb = self.num_bindings ** 2 * 8 / 1e9
                print(
                    f"  Large grammar detected ({self.num_bindings} bindings)")
                print(
                    f"  Dense WC would be {dense_size_gb:.1f} GB - using SPARSE matrix!")
                # self.WC = sparse.lil_matrix(
                # Use dok_matrix (Dictionary of Keys) for construction - much more memory efficient
                # than lil_matrix when doing many incremental updates
                print(f"  Using dok_matrix for memory-efficient construction...")
                self.WC = sparse.dok_matrix(
                    (self.num_bindings, self.num_bindings), dtype=np.float64)

                self.use_sparse = True
            else:
                self.WC = np.zeros((self.num_bindings, self.num_bindings))
                self.use_sparse = False

            self.bC = np.zeros(self.num_bindings)
            self.estr = self.opts['init_estr'] * np.ones(self.num_bindings)
            # Optimizer states will be initialized in initialize() method
            # to avoid OOM during model construction
            # Initialize Adam states on CPU
            # if self.opts['use_sparse_wc']:
            #     # Sparse optimizer states
            #     print("  Initializing sparse optimizer states...")
            #     self.optim = {
            #         'M_WC': sparse.lil_matrix((self.num_bindings, self.num_bindings), dtype=np.float64),
            #         'R_WC': sparse.lil_matrix((self.num_bindings, self.num_bindings), dtype=np.float64),
            #         'M_bC': np.zeros(self.num_bindings),
            #         'R_bC': np.zeros(self.num_bindings),
            #         'step_WC': 0,
            #         'step_bC': 0,
            #         'beta1': 0.9,
            #         'beta2': 0.999,
            #         'eps': 1e-8
            #     }
            # else:
            #     self.optim = {
            #         'M_WC': np.zeros_like(self.WC),
            #         'R_WC': np.zeros_like(self.WC),
            #         'M_bC': np.zeros_like(self.bC),
            #         'R_bC': np.zeros_like(self.bC),
            #         'step_WC': 0,
            #         'step_bC': 0,
            #         'beta1': 0.9,
            #         'beta2': 0.999,
            #         'eps': 1e-8
            #     }
        ############ ORIGINAL CODE COMMENTED OUT#######
        # self.WC = np.zeros((self.num_bindings, self.num_bindings))
        # self.bC = np.zeros(self.num_bindings)
        # self.estr = self.opts['init_estr'] * np.ones(self.num_bindings)
        if hg is not None:
            print("DEBUG: _build_model starts")
            self._build_model()
            print("DEBUG: _adjust_default_param_vals starts")
            self._adjust_default_param_vals()
            if self.opts['use_second_order_bias']:
                print("DEBUG: bias2weight starts")
                self.bias2weight()
            # Note: WC already converted to CSR in _build_model() for efficiency
            # Final memory report
            if self.opts['use_sparse_wc']:
                # print("  Converting WC matrix to CSR format...")
                # self.WC = self.WC.tocsr()
                # self.optim['M_WC'] = self.optim['M_WC'].tocsr()
                # self.optim['R_WC'] = self.optim['R_WC'].tocsr()
                nnz = self.WC.nnz
                total = self.num_bindings ** 2
                sparsity = 100 * (1 - nnz / total)
                memory_dense_gb = total * 8 / 1e9
                memory_sparse_mb = (nnz * (8 + 8)) / 1e6  # value + indices
                print(
                    #     f"  ✓ WC sparsity: {sparsity:.4f}% ({nnz:,} non-zero out of {total:,})")
                    f"  ✓ Final WC sparsity: {sparsity:.4f}% ({nnz:,} non-zero out of {total:,})")
                print(
                    f"  ✓ Memory saved: {memory_dense_gb:.1f} GB (dense) → {memory_sparse_mb:.1f} MB (sparse)")
                print(
                    f"  ✓ Reduction: {memory_dense_gb * 1000 / memory_sparse_mb:.0f}x")

        dur = time.time() - t0
        print('{} s for initializing parameter values'.format(dur))

        if self.use_jax:
            # JAX version: create state variables on GPU
            self.extC = jnp.zeros(self.num_bindings, dtype=jnp.float32)
            self.ext = self.C2N(actC=self.extC)
            self._set_bowl_parameters()

            self.q = jnp.ones(
                self.num_roles, dtype=jnp.float32) * self.opts['q_init']
            self.T = self.opts['T_init']
            self.dt = self.opts['dt_init']

            # Add state variables on GPU
            self.t = 0.
            self.actC = jnp.zeros(self.num_bindings, dtype=jnp.float32)
            self.actCmat = self.vec2mat(self.actC)
            self.act = self.C2N()
            self.update_scale_constants(pos=0)
        else:
            # NumPy version: CPU arrays
            self.extC = np.zeros(self.num_bindings)
            self.ext = self.C2N(actC=self.extC)
            self._set_bowl_parameters()

            self.q = self.opts['q_init'] * np.ones(self.num_roles)
            self.T = self.opts['T_init']
            self.dt = self.opts['dt_init']

            # Add state variables
            self.t = 0.
            self.actC = np.zeros(self.num_bindings)
            self.actCmat = self.vec2mat(self.actC)
            self.act = self.C2N()
            self.update_scale_constants(pos=0)

        t0 = time.time()
        print("  Computing equilibrium point (ep)...")
        self.get_ep(method=self.opts['ep_method'])
        dur = time.time() - t0
        print('  {} s for finding a global equilibrium point'.format(dur))

        self.set_state(mu=self.ep)
        if qpolicy is None:
            self.qpolicy = np.arange(self.hg.opts['max_sent_len'] + 1)
        else:
            self.qpolicy = qpolicy
        self.backup_parameters()

    def _precompute_fastER_lookups(self):
        '''Pre-compute index lookup structures for O(1) access.

        OPTIMIZED VERSION:
        - Combined role loop for role_to_binding + daughter indices
        - Efficient dict comprehensions
        - Complexity: O(R + F + B) where R=roles, F=fillers, B=bindings
        '''
        import time
        t0 = time.time()

        # Verify Phase 1 is complete
        if not hasattr(self.hg.roles, 'role_name_to_idx'):
            print("WARNING: BrickRole Phase 1 not found!")
            return

        num_roles = self.hg.num_roles
        num_fillers = self.hg.num_fillers
        num_bindings = self.hg.num_bindings

        # ============ COMBINED LOOP: Role bindings + daughter indices ============

        self.role_to_binding_indices = {}
        self.role_daughter_binding_indices = {}

        # Cache terminal status and daughter indices for faster access
        role_is_terminal = self.hg.roles.role_is_terminal
        daughter_l_idx = self.hg.roles.role_daughter_l_idx
        daughter_r_idx = self.hg.roles.role_daughter_r_idx

        for ri in range(num_roles):
            # Build role→binding mapping
            start = ri * num_fillers
            end = start + num_fillers
            binding_indices = np.arange(start, end, dtype=np.int32)
            self.role_to_binding_indices[ri] = binding_indices

            # Build daughter binding indices for non-terminal roles
            if not role_is_terminal[ri]:
                dl_idx = daughter_l_idx[ri]
                dr_idx = daughter_r_idx[ri]

                if dl_idx >= 0 and dr_idx >= 0:
                    # Compute daughter binding indices using already-computed arrays
                    self.role_daughter_binding_indices[ri] = {
                        'l': np.arange(dl_idx * num_fillers,
                                       (dl_idx + 1) * num_fillers,
                                       dtype=np.int32),
                        'r': np.arange(dr_idx * num_fillers,
                                       (dr_idx + 1) * num_fillers,
                                       dtype=np.int32),
                        'self': binding_indices
                    }

        # ============ FILLER TO BINDING INDICES ============

        # Can't combine with role loop (different iteration count)
        self.filler_to_binding_indices = {
            fi: np.arange(fi, num_bindings, num_fillers, dtype=np.int32)
            for fi in range(num_fillers)
        }

        # ============ NAME TO INDEX MAPPINGS ============

        # Reference existing mappings (no computation)
        self.role_name_to_idx = self.hg.roles.role_name_to_idx
        self.filler_name_to_idx = self.hg.g.filler_name_to_idx

        # Build binding name→index mapping
        self.binding_name_to_idx = {
            name: bi for bi, name in enumerate(self.hg.binding_names)
        }

        self.role_name_to_binding_indices = {}
        # Build filler_name -> binding_indices mapping (matching slow find_fillers logic)
        self.filler_name_to_binding_indices = {}
        for bi, bname in enumerate(self.binding_names):
            role_name = bname.split('/')[1]
            if role_name not in self.role_name_to_binding_indices:
                self.role_name_to_binding_indices[role_name] = []
            filler_name = bname.split('/')[0]
            if filler_name not in self.filler_name_to_binding_indices:
                self.filler_name_to_binding_indices[filler_name] = []
            self.filler_name_to_binding_indices[filler_name].append(bi)

            self.role_name_to_binding_indices[role_name].append(bi)

        # Convert lists to numpy arrays
        for role_name in self.role_name_to_binding_indices:
            self.role_name_to_binding_indices[role_name] = np.array(
                self.role_name_to_binding_indices[role_name], dtype=np.int32)

        # Convert lists to numpy arrays
        for filler_name in self.filler_name_to_binding_indices:
            self.filler_name_to_binding_indices[filler_name] = np.array(
                self.filler_name_to_binding_indices[filler_name], dtype=np.int32)

        elapsed = time.time() - t0
        print(f"✓ GscNet fast lookups built in {elapsed:.3f}s "
              f"({num_roles} roles, {num_fillers} fillers, {num_bindings} bindings)")

    def _precompute_fast_lookups(self):
        '''Pre-compute index lookup structures for O(1) access.'''

        print("Pre-computing GscNet fast lookups...")

        # Verify Phase 1 is complete
        if not hasattr(self.hg.roles, 'role_name_to_idx'):
            print("WARNING: Phase 1 not found!")
            return

        # Binding indices for each role
        self.role_to_binding_indices = {}
        for ri in range(self.hg.num_roles):
            start = ri * self.hg.num_fillers
            end = start + self.hg.num_fillers
            self.role_to_binding_indices[ri] = np.arange(
                start, end, dtype=np.int32)

        # Binding indices for each filler
        self.filler_to_binding_indices = {}
        for fi in range(self.hg.num_fillers):
            self.filler_to_binding_indices[fi] = np.arange(
                fi, self.hg.num_bindings, self.hg.num_fillers, dtype=np.int32
            )

        self.role_name_to_binding_indices = {}
        for bi, bname in enumerate(self.binding_names):
            role_name = bname.split('/')[1]
            if role_name not in self.role_name_to_binding_indices:
                self.role_name_to_binding_indices[role_name] = []
            self.role_name_to_binding_indices[role_name].append(bi)

        # Convert lists to numpy arrays
        for role_name in self.role_name_to_binding_indices:
            self.role_name_to_binding_indices[role_name] = np.array(
                self.role_name_to_binding_indices[role_name], dtype=np.int32)

        # Build filler_name -> binding_indices mapping (matching slow find_fillers logic)
        self.filler_name_to_binding_indices = {}
        for bi, bname in enumerate(self.binding_names):
            filler_name = bname.split('/')[0]
            if filler_name not in self.filler_name_to_binding_indices:
                self.filler_name_to_binding_indices[filler_name] = []
            self.filler_name_to_binding_indices[filler_name].append(bi)

        # Convert lists to numpy arrays
        for filler_name in self.filler_name_to_binding_indices:
            self.filler_name_to_binding_indices[filler_name] = np.array(
                self.filler_name_to_binding_indices[filler_name], dtype=np.int32)

        # Use HG's pre-computed mappings (from Phase 1)
        self.role_name_to_idx = self.hg.roles.role_name_to_idx
        self.filler_name_to_idx = self.hg.g.filler_name_to_idx

        # Binding name to index
        self.binding_name_to_idx = {}
        for bi in range(self.hg.num_bindings):
            self.binding_name_to_idx[self.hg.binding_names[bi]] = bi

        # Pre-compute daughter indices (for training loops)
        self.role_daughter_binding_indices = {}
        for ri in range(self.hg.num_roles):
            if not self.hg.roles.role_is_terminal[ri]:
                dl_idx = self.hg.roles.role_daughter_l_idx[ri]
                dr_idx = self.hg.roles.role_daughter_r_idx[ri]

                if dl_idx >= 0 and dr_idx >= 0:
                    self.role_daughter_binding_indices[ri] = {
                        'l': self.role_to_binding_indices[dl_idx],
                        'r': self.role_to_binding_indices[dr_idx],
                        'self': self.role_to_binding_indices[ri]
                    }

        print("✓ GscNet fast lookups complete!")

    def find_roles_fast(self, rnames):
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
        if not isinstance(rnames, list):
            rnames = [rnames]
        if len(rnames) == 1:
            return self.role_name_to_binding_indices.get(rnames[0], np.array([], dtype=np.int32))
        result = []
        for rname in rnames:
            if rname in self.role_name_to_binding_indices:
                result.append(self.role_name_to_binding_indices[rname])
        return np.concatenate(result) if result else np.array([], dtype=np.int32)

    def find_fillers_fast(self, fnames):
        '''Fast O(1) version of find_fillers.'''
        if not isinstance(fnames, list):
            fnames = [fnames]
        if len(fnames) == 1:
            return self.filler_name_to_binding_indices.get(fnames[0], np.array([], dtype=np.int32))

        result = []
        for fname in fnames:
            if fname in self.filler_name_to_binding_indices:
                result.append(self.filler_name_to_binding_indices[fname])

        return np.concatenate(result) if result else np.array([], dtype=np.int32)

    def find_bindings_fast(self, bnames):
        '''Fast O(1) version of find_bindings.'''
        if not isinstance(bnames, list):
            bnames = [bnames]

        return [self.binding_name_to_idx[bn] for bn in bnames if bn in self.binding_name_to_idx]

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
        # JAX default
        self.opts['use_jax'] = JAX_AVAILABLE
        # Sparse matrix support for large grammars
        # Will be auto-enabled if num_bindings > 100,000
        self.opts['use_sparse_wc'] = None  # None = auto-detect
        # Threshold for auto-enabling sparse
        self.opts['sparse_wc_threshold'] = 100000

    def _update_opts(self, opts):
        # Update opts

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
        """Create change-of-basis matrices WITHOUT materializing S.

        CRITICAL: S = C @ C.T would be 18 TB for large grammars!
        We never materialize S - instead compute C @ (C.T @ v) on-the-fly.
        """

        print("Computing change-of-basis matrices...")
        t0 = time.time()
        # Compute N and C (these are manageable sizes)
        N = np.kron(self.R, self.F)
        # if N.shape[0] == N.shape[1]:
        #     C = np.linalg.inv(N)
        # else:
        #     C = np.linalg.pinv(N)
        print(
            f"  N shape: {N.shape} ({N.shape[0] * N.shape[1] * 8 / 1e9:.2f} GB)")
        print(f"  Kronecker product took {time.time() - t0:.2f} s")

        # Compute pseudo-inverse efficiently using Kronecker product property
        # OPTIMIZATION: pinv(kron(R, F)) = kron(pinv(R), pinv(F))
        # This is MUCH faster for large grammars!
        t0 = time.time()
        print(f"  Computing pseudo-inverse using fast Kronecker decomposition...")

        # Compute pinv of small matrices R and F separately
        R_pinv = np.linalg.pinv(self.R, rcond=1e-10)
        F_pinv = np.linalg.pinv(self.F, rcond=1e-10)

        # Reconstruct C using Kronecker product (much faster!)
        C = np.kron(R_pinv, F_pinv)

        dur = time.time() - t0
        print(
            f"  Fast pseudo-inverse took {dur:.2f} s (vs. minutes for direct method)")
        print(f"  C shape: {C.shape}")

        self.N = N
        self.C = C

        # Compute Gc (small matrix: num_units × num_units)
        print("DEBUG _add_change_of_basis_matrices")
        # self.Gc = self.C.T.dot(self.C)

        # Reshape C for filler/role access
        self.C_reshaped = self.C.reshape(
            (self.num_fillers, self.num_roles, self.num_units), order='F'
        )

        # CRITICAL: DON'T CREATE S!
        # The old code did: self.S = self.C.dot(self.C.T)
        # With 1.5M bindings, S would be 1.5M × 1.5M = 18 TB!
        #
        # Instead, we compute C @ (C.T @ v) on-the-fly in update_stateC()
        # This uses ZERO extra memory and is faster!

        # Pre-compute C.T for efficiency (still small)
        self.C_T = self.C.T

        # Convert to JAX if using GPU
        if self.use_jax:  # QI's TODO: use_jax not defined yet
            print("  Converting change-of-basis matrices to GPU...")
            # Use float32 for GPU efficiency (or float64 if precision critical)
            self.C = jnp.array(self.C, dtype=jnp.float32)
            self.C_T = jnp.array(self.C_T, dtype=jnp.float32)
            self.N = jnp.array(self.N, dtype=jnp.float32)
            print(f"  ✓ C matrix on GPU: {self.C.shape} ({self.C.dtype})")

        print(
            f"  ✓ Change-of-basis complete (C: {self.C.shape}, no S materialization)")

        # Memory saved:
        # Old: S = 1.5M × 1.5M × 4 bytes = 9 TB (float32)
        # New: 0 bytes (S never created!)
        bytes_saved = self.num_bindings ** 2 * 4
        print(
            f"  ✓ Memory saved by not creating S: {bytes_saved / 1e12:.1f} TB")

    def _build_model(self):
        # Initialize the model by setting weight and bias parameters to
        # some default values specified in HG.
        # NOTE: Complex competition rules and null rules were removed temporarily.
        print(f"  Building weight model from {len(self.hg.rules)} HG rules...")
        t_start = time.time()
        # max_sent_len = self.hg.opts['max_sent_len']
        # use_hnf = self.hg.g.opts['use_hnf']
        role_system = self.hg.opts['role_system']
        roles = self.hg.roles
        bsep = self.hg.opts['bsep']

        H_root_illegitimate = self.hg.opts['H_root_illegitimate']
        H_terminal_illegitimate = self.hg.opts['H_terminal_illegitimate']
        H_nonterminal_illegitimate = self.hg.opts['H_nonterminal_illegitimate']
        H_copy_illegitimate = self.hg.opts['H_copy_illegitimate']

        # self.WC = np.zeros((self.num_bindings, self.num_bindings))

        # t1 = time.time()
        # Binary and copy rules =========================
        # for rule in self.hg.subset_rules(['binary', 'copy']):
        print(f"    Processing binary and copy rules...")
        binary_copy_rules = self.hg.subset_rules(['binary', 'copy'])
        print(f"      Found {len(binary_copy_rules):,} binary/copy rules")

        # OPTIMIZATION: Pre-compute filler indices to avoid string concat + lookup
        f1_indices = {}  # Cache filler1 indices by filler name
        f2_indices = {}  # Cache filler2 indices by filler name

        is_sparse = hasattr(self, 'use_sparse') and self.use_sparse

        update_count = 0
        last_report_time = time.time()
        for rule_idx, rule in enumerate(binary_copy_rules):
            if rule_idx > 0 and rule_idx % 10000 == 0:
                elapsed = time.time() - last_report_time
                rate = 10000 / elapsed if elapsed > 0 else 0
                print(
                    f"      Processed {rule_idx:,}/{len(binary_copy_rules):,} rules ({rate:.0f} rules/s, {update_count:,} weight updates)")
                last_report_time = time.time()
                update_count = 0

            # Pre-compute binding indices for this rule's fillers
            f1 = rule['f1']
            f2 = rule['f2']
            H = rule['H']

            # Cache the indices for each role for this filler
            if f1 not in f1_indices:
                f1_indices[f1] = {ri: self.binding_name_to_idx.get(f1 + bsep + roles.role_names[ri], -1)
                                  for ri in range(len(roles.role_names))}
            if f2 not in f2_indices:
                f2_indices[f2] = {ri: self.binding_name_to_idx.get(f2 + bsep + roles.role_names[ri], -1)
                                  for ri in range(len(roles.role_names))}

            # Now process roles without string operations
            # for role in roles.role_names:
            for ri in range(len(self.hg.role_names)):
                # if roles.is_bracketed(role) == rule['br']:
                # role = roles.role_names[ri]
                if self.hg.roles.role_is_bracketed[ri] == rule['br']:
                    # mother_roles = roles.get_mothers(role)
                    # focus_mother_roles = mother_roles[rule['rel']]
                    idx1 = f1_indices[f1].get(ri, -1)
                    if idx1 < 0:
                        continue
                    focus_mother_roles_indices = self.hg.roles.role_mothers_idx[rule['rel']][ri]
                    for focus_mother_roles_ind in focus_mother_roles_indices:
                        # focus_mother_role = self.role_names[focus_mother_roles_ind]
                        # if focus_mother_role in roles.role_names:
                        #     b1name = rule['f1'] + bsep + role
                        #     b2name = rule['f2'] + bsep + focus_mother_role
                        #     self.set_weight(b1name, b2name, rule['H'],
                        #                     cumulative=True, c2n=False)
                        # abandon code above, code below Direct matrix updates bypassing set_weight()
                        if focus_mother_roles_ind >= len(roles.role_names):
                            continue
                        idx2 = f2_indices[f2].get(focus_mother_roles_ind, -1)
                        if idx2 < 0:
                            continue

                        # Direct matrix update without string operations
                        if is_sparse:
                            self.WC[idx1, idx2] = self.WC[idx1, idx2] + H
                            self.WC[idx2, idx1] = self.WC[idx2, idx1] + H
                        else:
                            self.WC[idx1, idx2] += H
                            self.WC[idx2, idx1] += H
                        update_count += 1
        # dur = time.time() - t1
        # print('{} ms for implementing binrary HG rules'.format(dur))

        # Competition rules =========================
        print(f"    Processing competition rules...")
        cumulative = False
        # for rule in self.hg.subset_rules('competition'):
        competition_rules = self.hg.subset_rules('competition')
        for rule in competition_rules:
            r1, r2 = rule['rel'].split('/')
            if r1 == 'ub' and r2 == 'ub':
                # for role in roles.role_names:
                for ri in range(len(self.hg.role_names)):
                    # if not roles.is_bracketed(role):
                    if not self.hg.roles.role_is_bracketed[ri]:
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
        print(f"    Processing unary rules...")
        self.bC = np.zeros(self.num_bindings)
        if self.hg.opts['unary_base'] == 'filler':
            unary_rules = self.hg.subset_rules('unary')
            for rule in unary_rules:
                self.set_filler_bias(rule['f1'], rule['H'], c2n=False)
        else:
            sys.exit('CHECK "unary_base"!')

        # Additional constraints (penalty for ungrammatical bindings)
        if H_root_illegitimate < 0:
            # for rname in roles.role_names:
            for ri in range(len(roles.role_names)):
                rname = roles.role_names[ri]
                if role_system == 'brick_role':
                    # lv, pos = roles.str2tuple(rname)
                    lv, pos = roles.role_tuples[ri]
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
            # for rname in roles.role_names:
            for ri in range(len(roles.role_names)):
                rname = roles.role_names[ri]
                if role_system == 'brick_role':
                    # lv, pos = roles.str2tuple(rname)
                    lv, pos = roles.role_tuples[ri]
                    if lv != 1:
                        # bnames = [f + bsep + rname
                        #           for f in self.hg.g.get_fillers()
                        #           if self.hg.g.is_terminal(f)]
                        terminal_fi = np.where(self.hg.g.filler_is_terminal)[0]
                        bnames = [self.hg.g.filler_names[fi] +
                                  bsep + rname for fi in terminal_fi]
                        self.set_bias(
                            bnames, H_terminal_illegitimate, c2n=False)

        if H_nonterminal_illegitimate < 0:
            # for rname in roles.role_names:
            for ri in range(len(roles.role_names)):
                rname = roles.role_names[ri]
                if role_system == 'brick_role':
                    # lv, pos = roles.str2tuple(rname)
                    lv, pos = roles.role_tuples[ri]
                    if lv == 1:
                        # bnames = [f + bsep + rname
                        #           for f in self.hg.g.get_fillers()
                        #           if (not self.hg.g.is_terminal(f) and
                        #               f != self.hg.g.opts['null'])]
                        null_idx = self.hg.g.filler_name_to_idx.get(
                            self.hg.g.opts['null'], -1)
                        nonterminal_fi = np.where(
                            ~self.hg.g.filler_is_terminal)[0]
                        if null_idx >= 0:
                            nonterminal_fi = nonterminal_fi[nonterminal_fi != null_idx]
                        bnames = [self.hg.g.filler_names[fi] +
                                  bsep + rname for fi in nonterminal_fi]
                        self.set_bias(
                            bnames, H_nonterminal_illegitimate, c2n=False)
        print(f"    Adding grammatical constraints...")
        if H_copy_illegitimate < 0:
            # for rname in roles.role_names:
            for ri in range(len(roles.role_names)):
                rname = roles.role_names[ri]
                if role_system == 'brick_role':
                    # lv, pos = roles.str2tuple(rname)
                    lv, pos = roles.role_tuples[ri]
                    if lv == 1:
                        # bnames = [f + bsep + rname
                        #           for f in self.hg.g.get_fillers()
                        #           if self.hg.g.is_copy(f)]
                        copy_fi = np.where(self.hg.g.filler_is_copy)[0]
                        bnames = [self.hg.g.filler_names[fi] +
                                  bsep + rname for fi in copy_fi]
                        self.set_bias(bnames, H_copy_illegitimate, c2n=False)
        # Convert WC to CSR BEFORE matrix multiplication (critical for performance!)
        if hasattr(self, 'use_sparse') and self.use_sparse:
            print(f"    Converting WC from dok_matrix to CSR for efficient operations...")
            t_convert = time.time()
            self.WC = self.WC.tocsr()
            nnz = self.WC.nnz
            total = self.num_bindings ** 2
            sparsity = 100 * (1 - nnz / total)
            print(
                f"      WC: {nnz:,} non-zero elements ({sparsity:.4f}% sparse)")
            print(f"      Conversion took {time.time() - t_convert:.2f}s")

        print(f"    Setting weights...")
        self._set_weights()
        print(f"    Setting biases...")
        self._set_biases()
        dur = time.time() - t_start
        print(f"  ✓ Weight model built in {dur:.2f}s ({dur/60:.2f} min)")

    def _adjust_default_param_vals(self, method='Newton'):

        if self.hg.opts['use_same_len']:
            # Adjust bias values of root bindings before adding expansion rules
            # and bias values of newly added empty bindings.

            bC = self.bC.copy()

            if not self.hg.opts['add1_to_root']:
                # default

                if not np.isclose(self.hg.opts['H_root_illegitimate'], 0.):

                    for fname in self.hg.get_roots():
                        # for rname in self.role_names:
                        for ri in range(len(self.hg.role_names)):
                            rname = self.role_names[ri]
                            bname = fname + self.hg.opts['bsep'] + rname
                            idx = self.find_bindings_fast(bname)
                            # lv, pos = self.hg.roles.str2tuple(rname)
                            lv, pos = self.hg.roles.role_tuples[ri]
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
                        # for rname in self.role_names:
                        for ri in range(len(self.hg.role_names)):
                            rname = self.role_names[ri]
                            bname = fname + self.hg.opts['bsep'] + rname
                            idx = self.find_bindings_fast(bname)
                            # lv, pos = self.hg.roles.str2tuple(rname)
                            lv, pos = self.hg.roles.role_tuples[ri]
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
                # rid = self.find_roles(self.role_names[-1])
                rid = self.find_roles_fast(self.role_names[-1])
                # rid = [ii for ii in rid if ii in self.find_fillers(roots)]
                root_filler_indices = self.find_fillers_fast(roots)
                rid = [ii for ii in rid if ii in root_filler_indices]
                self.bC[rid] += 1.

            self._set_biases()

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

        idx1 = self.find_bindings_fast(bname1)
        idx2 = self.find_bindings_fast(bname2)
        # For sparse matrices, we need special handling
        is_sparse = hasattr(self, 'use_sparse') and self.use_sparse

        if not cumulative:
            if symmetric:
                if is_sparse:
                    # Sparse: set each element individually
                    # dok_matrix efficiently handles this pattern
                    for i in idx1:
                        for j in idx2:
                            self.WC[i, j] = weight
                            self.WC[j, i] = weight
                else:
                    # Dense: can use fancy indexing
                    self.WC[idx1, idx2] = self.WC[idx2, idx1] = weight
            else:
                if is_sparse:
                    for i in idx2:
                        for j in idx1:
                            self.WC[i, j] = weight
                else:
                    self.WC[idx2, idx1] = weight
        else:
            # WC = np.zeros(self.WC.shape)
            if symmetric:
                # WC[idx1, idx2] = WC[idx2, idx1] = weight
                # self.WC += WC
                if is_sparse:
                    # Sparse: accumulate each element individually
                    for i in idx1:
                        for j in idx2:
                            self.WC[i, j] = self.WC[i, j] + weight
                            self.WC[j, i] = self.WC[j, i] + weight
                else:
                    # Dense: can use fancy indexing with +=
                    self.WC[idx1, idx2] += weight
                    self.WC[idx2, idx1] += weight
            else:
                # WC[idx2, idx1] = weight
                # self.WC += WC
                if is_sparse:
                    for i in idx2:
                        for j in idx1:
                            self.WC[i, j] = self.WC[i, j] + weight
                else:
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

        idx = self.find_bindings_fast(binding_name)
        self.bC[idx] = bias
        if c2n:
            self._set_biases()

    def bias2weight(self):
        '''Set recurrent weights given bias values in conceptual coordinates'''

        # Add diagonal - works for both dense and sparse matrices
        if hasattr(self, 'use_sparse') and self.use_sparse:
            # Sparse matrix: add to diagonal efficiently
            diag_values = 2 * self.bC
            # FIXED: Use setdiag() for fast diagonal modification on CSR matrices
            # setdiag() is optimized for all scipy sparse formats
            current_diag = self.WC.diagonal()
            new_diag = current_diag + diag_values
            # Set diagonal elements individually (works with any sparse format)
            self.WC.setdiag(new_diag)
            self.WC.setdiag(self.WC.diagonal() + diag_values)
        else:
            # Dense matrix: standard numpy operation
            self.WC = self.WC + np.diag(2 * self.bC)
        self.bC = np.zeros(self.num_bindings)
        self._set_weights()
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

        NOTE: For large sparse grammars, W is never used (network works in
        conceptual coordinates). Skip computation to save memory.
        '''
        # Skip for sparse matrices - W is never used and causes OOM
        if hasattr(self, 'use_sparse') and self.use_sparse:
            print("      Skipping W computation (not needed for sparse matrices)")
            self.W = None
            return

        self.W = self.C.T.dot(self.WC).dot(self.C)

    def _set_biases(self):
        '''Converts bC to b.

        bC: b_c, bias vector for conceptual coordinates
        b : b_n, bias vector for neural coordinates

        NOTE: For large sparse grammars, b is never used (network works in
        conceptual coordinates). Skip computation to save memory.
        '''
        # Skip for sparse matrices - b is never used
        if hasattr(self, 'use_sparse') and self.use_sparse:
            print("      Skipping b computation (not needed for sparse matrices)")
            self.b = None
            return

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

    def check_divergence(self, tol=2.):
        if self.use_jax:
            return jnp.max(self.actC) > tol
        return max(self.actC) > tol

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

        # For JAX: use JIT-compiled loop for maximum speed
        if self.use_jax and not log_trace and tol is None and (self.opts['T_decay_rate'] <= 0):
            # Pure JAX fast path: JIT-compiled dynamics loop
            num_steps = int(duration / self.dt)

            def body_fun(i, carry):
                actC, q, rng_key = carry
                actC_new, q_new, rng_key_new = _dynamics_step_jax(
                    actC, self.WC, self.bC, self.extC, self.bowl_center,
                    self.opts['bowl_strength'], self.scale_constants,
                    self.C, self.C_T, self.N, self.num_fillers,
                    self.dt, self.T, q, self.opts['q_max'],
                    self.opts['q_rate'] if update_q else 0.0,
                    self.opts['m'], rng_key
                )
                return (actC_new, q_new, rng_key_new)
            # Run JIT-compiled loop
            init_carry = (self.actC, self.q, self.rng_key)
            final_carry = jax.lax.fori_loop(0, num_steps, body_fun, init_carry)
            self.actC, self.q, self.rng_key = final_carry
            # Update derived quantities
            self.actCmat = self.vec2mat()
            self.t += num_steps * self.dt
        else:
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
        if self.use_jax:
            self.q = self.opts['q_init'] * \
                jnp.ones(self.num_roles, dtype=jnp.float32)
            # Reset JAX random key to respect np.random.seed() calls
            # Use current numpy random state to generate a new JAX key
            self.rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        else:
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
                self.runC(dur, log_trace=False)
            else:
                self.run(dur)
            # if plot:
            #     self.plot_trace('actC')
            self.ep = self.actC.copy()
            self.opts['T_init'] = T_init_backup
            self.opts['q_rate'] = q_rate_backup

        self.q = q_backup.copy()

    def extend_rvec(self, rvec):
        return np.tile(
            rvec, (self.num_fillers, 1)).flatten('F')

    def add_noiseC(self):

        if self.use_jax:
            # JAX version: for GPU compatibility
            # JAX version: use JAX random with proper key splitting
            self.rng_key, subkey = jax.random.split(self.rng_key)
            noise = jnp.sqrt(2 * self.T * self.dt) * \
                jax.random.normal(subkey, shape=(
                    self.num_units,), dtype=jnp.float32)
            noiseC = jnp.sqrt(self.scale_constants) * \
                self.N2C(noise)  # rescaling noise
            self.actC += noiseC
        else:
            # NumPy version
            noise = np.sqrt(2 * self.T * self.dt) * \
                np.random.randn(self.num_units)
            noiseC = np.sqrt(self.scale_constants) * \
                self.N2C(noise)  # rescaling noise
            self.actC += noiseC

    def update_q(self):

        if hasattr(self, 'q_mask'):
            self.q += self.opts['q_rate'] * self.q_mask * self.dt
        else:
            self.q += self.opts['q_rate'] * self.dt
        if self.use_jax:
            self.q = jnp.maximum(
                jnp.minimum(self.q, self.opts['q_max']), 0)
        else:
            self.q = np.maximum(
                np.minimum(self.q, self.opts['q_max']), 0)

    def set_random_state(self, minact=0, maxact=1):

        if self.use_jax:
            # JAX version: use JAX random for GPU compatibility
            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.actC = jax.random.uniform(
                subkey, shape=(self.num_bindings,),
                minval=minact, maxval=maxact, dtype=jnp.float32)
        else:
            # NumPy version
            self.actC = np.random.uniform(
                minact, maxact, size=self.num_bindings)
        self.actCmat = self.vec2mat()
        self.act = self.C2N(self.actC)

    def update_stateC(self):
        """
        Update state using lazy S computation.

        KEY OPTIMIZATION: Instead of S @ v, compute C @ (C.T @ v)
        This avoids materializing the 18 TB S matrix!
        """
        # Compute gradient in conceptual space
        hgrad = self.HGradC()

        # OLD (doesn't work - S is too large):
        # gradC = self.scale_constants * self.S.dot(hgrad)

        # NEW (no S needed):
        # gradC = scale_constants * (C @ C.T) @ hgrad
        #       = scale_constants * C @ (C.T @ hgrad)  ← compute this way!

        if self.use_jax:
            # JAX version: JIT-compiled, all on GPU
            gradC = _lazy_s_multiply(
                self.C, self.C_T, hgrad, self.scale_constants
            )
            self.t += self.dt
            self.actC = self.actC + self.dt * gradC
        else:
            # NumPy version: on CPU
            temp = self.C_T.dot(hgrad)
            gradC = self.C.dot(temp)
            gradC = self.scale_constants * gradC
            self.t += self.dt
            self.actC = self.actC + self.dt * gradC

        # Add noise
        self.add_noiseC()
        self.actCmat = self.vec2mat()
        ########### ORIGINAL CODE BELOW ###########
        # # ToDo:
        # # (1) adaptive stepsize
        # # (2) different time scales -> consider dt as a vector (scale_constants)
        # #     (scale_constants for noise?)
        # # (3) clamp

        # gradC = self.scale_constants * self.S.dot(self.HGradC())

        # # adaptive stepsize
        # # update_dt()  # --> dt as a vector

        # self.t += self.dt             # update time
        # self.actC = self.actC + self.dt * gradC    # Euler integration
        # self.add_noiseC()

        # # if self.clamped:
        # #     self.act = self.act_clamped()

        # # Update actC and actCmat which are needed to compute Hq0Grad and Hq1Grad
        # # self.actC = self.N2C()
        # self.actCmat = self.vec2mat()

    def _set_bowl_parameters(self):
        '''Sets bowl parameters to default values. Default values
        must be updated after setting the weight and bias values.'''

        if isinstance(self.opts['bowl_center'], numbers.Number):
            if self.use_jax:
                self.bowl_center = (self.opts['bowl_center'] *
                                    jnp.ones(self.num_bindings, dtype=jnp.float32))
            else:
                self.bowl_center = (self.opts['bowl_center'] *
                                    np.ones(self.num_bindings))
        else:
            if self.use_jax:
                self.bowl_center = jnp.array(
                    self.opts['bowl_center'], dtype=jnp.float32)
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
                # for rname in self.role_names:
                #     idx = self.find_roles(rname)
                #     lv0, pos0 = self.hg.roles.str2tuple(rname)
                for ri in range(len(self.hg.role_names)):
                    idx = self.role_to_binding_indices[ri]
                    lv0, pos0 = self.hg.roles.role_tuples[ri]

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
                if self.use_jax:
                    self.scale_constants = jnp.ones(
                        self.num_bindings, dtype=jnp.float32)
                    self.scale_constants_q = jnp.array(
                        weights, dtype=jnp.float32)
                else:
                    self.scale_constants = np.ones(self.num_bindings)
                    self.scale_constants_q = weights
            else:
                if self.use_jax:
                    self.scale_constants = jnp.array(
                        weights, dtype=jnp.float32)
                    self.scale_constants_q = jnp.ones(
                        self.num_bindings, dtype=jnp.float32)
                else:
                    self.scale_constants = weights
                    self.scale_constants_q = np.ones(self.num_bindings)

        else:
            # Not yet implemneted
            if self.use_jax:
                self.scale_constants = jnp.ones(
                    self.num_bindings, dtype=jnp.float32)
            else:
                self.scale_constants = np.ones(self.num_bindings)

    def set_state(self, mu, sd=0.):

        if self.use_jax:
            # JAX version: use JAX random for GPU compatibility
            self.rng_key, subkey = jax.random.split(self.rng_key)
            noise_vec = jax.random.normal(
                subkey, shape=(self.num_bindings,), dtype=jnp.float32) * sd
        else:
            # NumPy version
            noise_vec = np.random.normal(
                loc=0., scale=sd, size=self.num_bindings)
        self.actC = mu + noise_vec
        self.actCmat = self.vec2mat()
        self.act = self.C2N()

    def backup_parameters(self):

        self.params_backup = {}
        self.params_backup['encodings'] = copy.deepcopy(self.encodings)
        if self.use_jax:
            # JAX arrays are immutable, so we can just store references
            # Or use jnp.array() to create copies that stay on GPU
            self.params_backup['WC'] = jnp.array(self.WC)
            self.params_backup['bC'] = jnp.array(self.bC)
            self.params_backup['estr'] = jnp.array(self.estr)
            self.params_backup['ep'] = jnp.array(self.ep)
            if hasattr(self, 'qpolicy'):
                # qpolicy is typically NumPy even in JAX mode
                self.params_backup['qpolicy'] = self.qpolicy.copy()
        else:
            # NumPy version
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
        print(f"Generating corpus with {nsamples} samples...")
        t_start = time.time()
        sentences = []
        targets = []
        pvals = []
        counts = []
        for i in range(nsamples):
            # Progress reporting
            if i > 0 and i % 500 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed
                remaining = (nsamples - i) / rate if rate > 0 else 0
                print(
                    f"  Generated {i}/{nsamples} samples ({rate:.1f} samples/s, {remaining/60:.1f} min remaining)")
            sentence, target, p = self.generate_sentence(
                min_sent_len=min_sent_len,
                max_sent_len=max_sent_len,
                use_type=use_type)
            # print(f"sentence: {sentence}")
            # print(f"target: {target}")
            # print(f"p: {p}")
            if i == 0:
                print(
                    f"first sentence took {time.time() - t_start:.1f}s to generate")
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
        dur = time.time() - t_start
        unique_sents = len(sentences)
        print(f"✓ Corpus generation complete in {dur:.1f}s ({dur/60:.1f} min)")
        print(f"  {unique_sents} unique sentences from {nsamples} samples")

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

    def get_discrete_state(self, binding_names):

        idx = self.find_bindings_fast(binding_names)
        actC = np.zeros(self.num_bindings)
        actC[idx] = 1.0
        return actC

    def set_discrete_state(self, binding_names):

        idx = self.find_bindings_fast(binding_names)
        self.actC = np.zeros(self.num_bindings)
        self.actC[idx] = 1.0
        self.actCmat = self.vec2mat()
        self.act = self.C2N()

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

            # for role in self.role_names:
            #     if not self.hg.roles.is_terminal(role):
            #         daughters = self.hg.roles.get_daughters(role)
            #         l = daughters['l']
            #         r = daughters['r']
            #         idx = self.find_roles(role)
            #         idx_l = self.find_roles(l)
            #         idx_r = self.find_roles(r)
            for ri in range(self.hg.num_roles):
                if not self.hg.roles.role_is_terminal[ri]:  # O(1) - GOOD!
                    # O(1) - GOOD!
                    daughters_info = self.role_daughter_binding_indices[ri]
                    if daughters_info:
                        idx = daughters_info['self']    # O(1) - GOOD!
                        idx_l = daughters_info['l']      # O(1) - GOOD!
                        idx_r = daughters_info['r']
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

    def run_prefix(self, prefix, update_q_discrete=False, log_trace=False):
        """Run through a sequence of prefix words.

        Args:
            prefix: List of filler names for the prefix
            update_q_discrete: Boolean for q update mode
            log_trace: Whether to log traces
        """
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

    def read_grid_point(self, actC=None, disp=False):

        if actC is None:
            actCmat = self.vec2mat(actC=self.actC)
        else:
            actCmat = self.vec2mat(actC=actC)
        if isinstance(actCmat, jax.Array):
            winner_idx = jnp.argmax(actCmat, axis=0)
        else:
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
        self.train_opts['idx_mask_bias1'] = self.bC <= -4.
        # self.train_opts['idx_mask_bias2'] = np.diag(self.WC) <= -8.

        # CRITICAL: Use .diagonal() for sparse matrices to avoid densification
        if hasattr(self, 'use_sparse') and self.use_sparse:
            self.train_opts['idx_mask_bias2'] = self.WC.diagonal() <= -8.
        else:
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
            # self.optim['M_WC'] = np.zeros_like(self.WC)
            # self.optim['M_bC'] = np.zeros_like(self.bC)
            # Handle sparse matrices properly
            if hasattr(self, 'use_sparse') and self.use_sparse:
                # For sparse matrices, create sparse optimizer states
                print("  Initializing sparse optimizer states for Adam...")
                from scipy import sparse
                # Use CSR format directly for efficiency (WC is already in CSR format)
                self.optim['M_WC'] = sparse.csr_matrix(
                    self.WC.shape, dtype=np.float64)
                self.optim['R_WC'] = sparse.csr_matrix(
                    self.WC.shape, dtype=np.float64)
            else:
                # Dense matrices
                self.optim['M_WC'] = np.zeros_like(self.WC)
                self.optim['R_WC'] = np.zeros_like(self.WC)
            self.optim['R_bC'] = np.zeros_like(self.bC)
            self.optim['step_WC'] = 0
            self.optim['step_bC'] = 0
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
            # mask0 = abs(np.sign(self.WC))
            # # allow the udpate of second-order bias of every binding
            # np.fill_diagonal(mask0, 1)
            # Get binary mask indicating non-zero elements
            if hasattr(self, 'use_sparse') and self.use_sparse:
                # For sparse matrices, use sign() method which preserves sparsity
                mask0 = self.WC.sign()
                if mask0 is None:  # sign() not available, use abs(...)
                    mask0 = abs(self.WC).astype(bool).astype(float)
                else:
                    mask0 = abs(mask0)
                # FIXED: Use dok_matrix instead of lil_matrix for memory efficiency
                # Convert to dok for diagonal modification (more memory efficient than lil)
                mask0 = mask0.todok()
                # Set diagonal elements to 1
                for i in range(min(mask0.shape)):
                    mask0[i, i] = 1
            else:
                mask0 = abs(np.sign(self.WC))
                # allow the udpate of second-order bias of every binding
                np.fill_diagonal(mask0, 1)
        else:
            # rnames_terminal = self.hg.roles.get_terminals()
            # idx_terminal = self.find_roles(rnames_terminal)
            # Use sparse zeros for sparse WC
            # CRITICAL FIX: For large grammars, use vectorized COO construction instead of nested loops
            if hasattr(self, 'use_sparse') and self.use_sparse:
                print("    Building mask0 using vectorized COO construction...")
                import time
                # FIXED: Build mask0 in BATCHES to avoid holding 10+ billion entries in memory
                t_start = time.time()
                # Process roles in batches, building and accumulating CSR matrices incrementally
                mask0 = sparse.csr_matrix(self.WC.shape, dtype=np.float64)

                # Collect all (row, col) pairs first using vectorized operations
                batch_size = 10  # Process 10 roles at a time
                non_terminal_roles = [ri for ri in range(len(self.hg.role_names))
                                      if not self.hg.roles.role_is_terminal[ri]]
                total_roles = len(non_terminal_roles)

                for batch_start in range(0, total_roles, batch_size):
                    batch_end = min(batch_start + batch_size, total_roles)
                    role_indices = non_terminal_roles[batch_start:batch_end]

                    if batch_start % 50 == 0:
                        print(
                            f"      Processing batch {batch_start//batch_size + 1}/{(total_roles + batch_size - 1)//batch_size} (roles {batch_start}-{batch_end}/{total_roles})...")

                    # Collect pairs for this batch only
                    row_list = []
                    col_list = []

                    for ri in role_indices:
                        indices = self.get_role_and_daughter_indices_fast(ri)
                        if indices != None:
                            idx = np.array(indices['self'])
                            idx_l = np.array(indices['l'])
                            idx_r = np.array(indices['r'])

                            # idx × idx (self-role)
                            rows_self, cols_self = np.meshgrid(
                                idx, idx, indexing='ij')
                            row_list.append(rows_self.ravel())
                            col_list.append(cols_self.ravel())

                            # idx × idx_l (parent-left) + symmetric
                            rows_pl, cols_pl = np.meshgrid(
                                idx, idx_l, indexing='ij')
                            row_list.extend([rows_pl.ravel(), cols_pl.ravel()])
                            col_list.extend([cols_pl.ravel(), rows_pl.ravel()])

                            # idx × idx_r (parent-right) + symmetric
                            rows_pr, cols_pr = np.meshgrid(
                                idx, idx_r, indexing='ij')
                            row_list.extend([rows_pr.ravel(), cols_pr.ravel()])
                            col_list.extend([cols_pr.ravel(), rows_pr.ravel()])

                            # Sister harmony (if enabled)
                            if self.train_opts['update_sister_harmony']:
                                rows_s, cols_s = np.meshgrid(
                                    idx_l, idx_r, indexing='ij')
                                row_list.extend(
                                    [rows_s.ravel(), cols_s.ravel()])
                                col_list.extend(
                                    [cols_s.ravel(), rows_s.ravel()])

                    # Build COO for this batch and add to mask0
                    if row_list:
                        batch_rows = np.concatenate(row_list)
                        batch_cols = np.concatenate(col_list)
                        batch_data = np.ones(len(batch_rows), dtype=np.float64)

                        batch_coo = sparse.coo_matrix(
                            (batch_data, (batch_rows, batch_cols)),
                            shape=self.WC.shape,
                            dtype=np.float64
                        )

                        # Add to cumulative mask (CSR addition handles duplicates)
                        mask0 = mask0 + batch_coo.tocsr()

                        # Explicitly free batch memory
                        del batch_rows, batch_cols, batch_data, batch_coo, row_list, col_list

                # Normalize: set all non-zero values to 1 (removing duplicate counts)
                mask0.data = np.ones_like(mask0.data)
                print(
                    f"      Total mask0 construction: {time.time() - t_start:.2f}s")
                print(f"      mask0 has {mask0.nnz:,} non-zero entries")
            else:
                # Dense path (original code using np.ix_)
                mask0 = np.zeros(self.WC.shape)
                for ri in range(len(self.hg.role_names)):
                    if not self.hg.roles.role_is_terminal[ri]:
                        indices = self.get_role_and_daughter_indices_fast(ri)
                        if indices != None:
                            idx = indices['self']
                            idx_l = indices['l']
                            idx_r = indices['r']
                            mask0[np.ix_(idx, idx)] = 1.
                            mask0[np.ix_(idx, idx_l)] = 1.
                            mask0[np.ix_(idx_l, idx)] = 1.
                            mask0[np.ix_(idx, idx_r)] = 1.
                            mask0[np.ix_(idx_r, idx)] = 1.
                            if self.train_opts['update_sister_harmony']:
                                mask0[np.ix_(idx_l, idx_r)] = 1.
                                mask0[np.ix_(idx_r, idx_l)] = 1.

        # CRITICAL: Ensure mask0 is in CSR format for fast element access during training
        # CSR format is optimized for element access like mask0[i,j] which happens
        # millions of times in the training loop
        if hasattr(self, 'use_sparse') and self.use_sparse and sparse.issparse(mask0):
            if not sparse.isspmatrix_csr(mask0):
                print("    Converting mask0 to CSR for fast training access...")
                mask0 = mask0.tocsr()

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
            if self.use_jax:
                prefix_weights = jnp.ones(len(prefix_list), dtype=jnp.float32)
                prefix_weights = prefix_weights / prefix_weights.sum()
            else:
                prefix_weights = np.ones(len(prefix_list))
                prefix_weights /= prefix_weights.sum()

        maxlen_prefix = 0
        for prefix in prefix_list:
            maxlen_prefix = max(maxlen_prefix, len(prefix))

        for _ in range(self.train_opts['num_epochs']):

            self.epoch_num += 1

            # mask = net.params_backup['WC'].astype(bool).astype(float)
            # mask = np.ones(self.WC.shape)
            # Initialize gradients with correct array type
            if self.use_jax:
                dWC = jnp.zeros(self.WC.shape, dtype=jnp.float32)
                dbC = jnp.zeros(self.bC.shape, dtype=jnp.float32)
                destr = jnp.zeros(self.estr.shape, dtype=jnp.float32)
                dqpolicy = jnp.zeros(self.qpolicy.shape, dtype=jnp.float32)
            else:
                # For sparse WC, use sparse gradient accumulator
                if hasattr(self, 'use_sparse') and self.use_sparse:
                    dWC = sparse.dok_matrix(self.WC.shape, dtype=np.float64)
                else:
                    dWC = np.zeros(self.WC.shape)
                dbC = np.zeros(self.bC.shape)
                destr = np.zeros(self.estr.shape)
                dqpolicy = np.zeros(self.qpolicy.shape)
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
                        stat_Q = self.estimate_prob_inc_jax(
                            prefix=prefix, num_trials=self.train_opts['num_trials'],
                            progress=10)  # Report every 10 trials
                    else:
                        stat_Q, actC_set = self.estimate_prob_inc(
                            prefix=prefix, num_trials=self.train_opts['num_trials'],
                            progress=10)  # Report every 10 trials
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
                    # Efficient boolean→int conversion for JAX
                    if self.use_jax:
                        extC_token = (self.extC != 0).astype(jnp.int32)
                    else:
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
                    # for ri, rname in enumerate(self.role_names):
                    #     idx = self.find_roles(rname)
                    for ri in range(len(self.hg.role_names)):
                        idx = self.role_to_binding_indices[ri]
                        rname = self.hg.role_names[ri]
                        for key, val in err['treelets'].items():
                            if key[0] in idx:
                                temp[ri] += abs(val)

                        # if self.hg.roles.is_terminal(rname):
                        if self.hg.roles.role_is_terminal[ri]:
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
                # rnames = [self.role_names[rid] for rid in role_idx_list]
                # idx = self.find_roles(rnames)
                if self.use_jax:
                    maskbC_update = jnp.zeros(
                        self.num_bindings, dtype=jnp.float32)
                    maskWC_update = jnp.zeros(
                        (self.num_bindings, self.num_bindings), dtype=jnp.float32)
                else:
                    maskbC_update = np.zeros(self.num_bindings)
                    # Use sparse mask for sparse WC
                    if hasattr(self, 'use_sparse') and self.use_sparse:
                        maskWC_update = sparse.dok_matrix(
                            (self.num_bindings, self.num_bindings), dtype=np.float64)
                    else:
                        maskWC_update = np.zeros(
                            (self.num_bindings, self.num_bindings))
                idx = np.concatenate([self.role_to_binding_indices[rid]
                                      for rid in role_idx_list])
                if self.use_jax:
                    maskbC_update = maskbC_update.at[idx].set(1.0)
                else:
                    maskbC_update[idx] = 1.
                # treelet_list = []
                for rid in role_idx_list:
                    #     r_daughters = self.hg.roles.get_daughters(
                    #         self.role_names[rid])
                    #     treelet_list.append(
                    #         [self.role_names[rid]] + r_daughters['l'] + r_daughters['r'])

                    # for treelet in treelet_list:
                    #     idx = self.find_roles(treelet)
                    if rid in self.role_daughter_binding_indices[rid]:
                        daughter_info = self.role_daughter_binding_indices[rid]
                        idx_self = daughter_info['self']
                        idx_l = daughter_info['l']
                        idx_r = daughter_info['r']
                        idx = np.concatenate([idx_self, idx_l, idx_r])
                    else:
                        idx = self.role_to_binding_indices[rid]
                    if self.use_jax:
                        # For JAX: use .at indexing
                        idx_i, idx_j = jnp.meshgrid(idx, idx, indexing='ij')
                        maskWC_update = maskWC_update.at[idx_i.flatten(
                        ), idx_j.flatten()].set(1.0)
                    else:
                        # FIXED: Avoid np.ix_() for sparse matrices (causes densification)
                        if hasattr(self, 'use_sparse') and self.use_sparse:
                            for i in idx:
                                for j in idx:
                                    maskWC_update[i, j] = 1.
                        else:
                            maskWC_update[np.ix_(idx, idx)] = 1.
            else:
                if self.use_jax:
                    maskWC_update = jnp.ones(
                        (self.num_bindings, self.num_bindings), dtype=jnp.float32)
                    maskbC_update = jnp.ones(
                        self.num_bindings, dtype=jnp.float32)
                else:
                    # For sparse matrices, avoid creating 2.85 TB mask of all ones
                    # Instead, set mask to None and check later
                    if hasattr(self, 'use_sparse') and self.use_sparse:
                        # No mask needed (all ones = no masking)
                        maskWC_update = None
                        maskbC_update = np.ones(self.num_bindings)
                    else:
                        maskWC_update = np.ones(
                            (self.num_bindings, self.num_bindings))
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
                    # Use sparse zeros for sparse WC
                    if hasattr(self, 'use_sparse') and self.use_sparse:
                        weight_decay = sparse.dok_matrix(
                            self.WC.shape, dtype=np.float64)
                    else:
                        weight_decay = np.zeros(self.WC.shape)

                if not (('bias2_only' in self.train_opts) and self.train_opts['bias2_only']):
                    if self.train_opts['optimizer'] == 'adam':
                        if self.use_jax:
                            # Ensure gradients are JAX arrays
                            if not isinstance(weight_decay, jnp.ndarray):
                                weight_decay = jnp.array(
                                    weight_decay, dtype=jnp.float32)

                            # JAX Adam update (JIT-compiled, all on GPU)
                            self.WC, self.optim['M_WC'], self.optim['R_WC'], self.optim['step_WC'] = \
                                _adam_update_jax(
                                    self.WC,
                                    dWC + weight_decay,  # Include weight decay
                                    self.optim['M_WC'],
                                    self.optim['R_WC'],
                                    self.optim['step_WC'],
                                    maskWC_update,
                                    self.train_opts['lrate'],
                                    self.optim['beta1'],
                                    self.optim['beta2'],
                                    self.optim['eps']
                            )
                        else:
                            # NumPy Adam update (original code)
                            # if self.train_opts['optimizer'] == 'adam':
                            # TODO: Add the weight decay term

                            # CRITICAL FIX: Convert dWC from DOK to CSR before arithmetic
                            # Mixing DOK and CSR silently produces incorrect results!
                            if hasattr(self, 'use_sparse') and self.use_sparse:
                                dWC = dWC.tocsr()

                            self.optim['M_WC'] = self.optim['beta1'] * \
                                self.optim['M_WC'] + \
                                (1. - self.optim['beta1']) * dWC
                            # Element-wise square - works with sparse
                            if hasattr(self, 'use_sparse') and self.use_sparse:
                                # Sparse-compatible power
                                dWC_squared = dWC.power(2)
                            else:
                                dWC_squared = dWC**2
                            self.optim['R_WC'] = self.optim['beta2'] * \
                                self.optim['R_WC'] + \
                                (1. - self.optim['beta2']) * dWC_squared
                            m_k_hat_WC = self.optim['M_WC'] / \
                                (1. - self.optim['beta1']**self.epoch_num)
                            r_k_hat_WC = self.optim['R_WC'] / \
                                (1. - self.optim['beta2']**self.epoch_num)
                            # Sparse sqrt - use power(0.5) which is sparse-compatible
                            if hasattr(self, 'use_sparse') and self.use_sparse:
                                r_k_hat_WC_sqrt = r_k_hat_WC.power(
                                    0.5)  # Sparse sqrt via power
                            else:
                                r_k_hat_WC_sqrt = np.sqrt(r_k_hat_WC)
                            self.WC += self.train_opts['lrate'] * m_k_hat_WC / \
                                (r_k_hat_WC_sqrt + self.optim['eps'])
                        self._set_weights()
                    else:
                        # SGD update - handle sparse mask

                        # CRITICAL FIX: Convert dWC from DOK to CSR before arithmetic
                        if hasattr(self, 'use_sparse') and self.use_sparse:
                            dWC = dWC.tocsr()
                            if not isinstance(weight_decay, (int, float)) and hasattr(weight_decay, 'tocsr'):
                                weight_decay = weight_decay.tocsr()

                        if maskWC_update is None:
                            # No mask (all ones) - apply update directly
                            self.WC += self.train_opts['lrate'] * \
                                (dWC + weight_decay)
                        else:
                            # Apply mask
                            self.WC += self.train_opts['lrate'] * \
                                (dWC + weight_decay) * maskWC_update
                        self._set_weights()

                if self.train_opts['bias1_only']:
                    self.bC += self.train_opts['lrate'] * dbC * maskbC_update
                    self._set_biases()
                if not self.opts['use_second_order_bias']:
                    if self.train_opts['optimizer'] == 'adam':
                        if self.use_jax:

                            self.bC, self.optim['M_bC'], self.optim['R_bC'], self.optim['step_bC'] = \
                                _adam_update_jax(
                                    self.bC, dbC, self.optim['M_bC'], self.optim['R_bC'],
                                    self.optim['step_bC'], maskbC_update,
                                    self.train_opts['lrate'], self.optim['beta1'],
                                    self.optim['beta2'], self.optim['eps']
                            )
                        else:
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

            # dWC_max = np.max(abs(dWC))
            # Compute max gradients - handle sparse matrices to avoid densification
            if hasattr(self, 'use_sparse') and self.use_sparse:
                # For sparse matrices, use .max() method which doesn't densify
                # abs() on sparse matrix preserves sparsity
                dWC_max = abs(dWC).max() if dWC.nnz > 0 else 0.0
            else:
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

        if self.use_jax:
            dWC = jnp.zeros((self.num_bindings, self.num_bindings))
            dbC = jnp.zeros(self.num_bindings)
        else:
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

        import time as time_module
        for trial_id in range(num_trials):

            if progress > 0:
                if (trial_id + 1) % progress == 0:
                    print('[%04d]' % (trial_id + 1), end='', flush=True)
                    if (trial_id + 1) % (10 * progress) == 0:
                        print('')

            trial_start = time_module.time()
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
            idx = self.find_bindings_fast(gp)
            self.set_discrete_state(gp)

            if list(self.actC) not in corpus['target']:
                corpus['target'].append(list(self.actC))
                corpus['count'].append(1)
            else:
                idx = corpus['target'].index(list(self.actC))
                corpus['count'][idx] += 1

            # Report timing for every 100th trial
            if progress > 0 and (trial_id + 1) % (10 * progress) == 0:
                trial_time = time_module.time() - trial_start
                print(f' [{trial_time:.2f}s/trial]')

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

        # Check if WC is sparse - JAX doesn't support sparse matrices yet
        from scipy import sparse
        if sparse.issparse(self.WC):
            print("WARNING: WC is sparse matrix - JAX acceleration not supported with sparse matrices.")
            print("         Falling back to CPU version. Consider using use_jax=True during initialization")
            print("         for JAX support, or use estimate_prob_inc() directly for CPU mode.")
            return self.estimate_prob_inc(prefix, num_trials, progress, update_q_discrete)

        # print(f"Running {num_trials} trials in parallel on GPU...")
        # t0 = time.time()

        # Extract network parameters for JAX
        net_params = _extract_net_params_for_jax(self)

        # Generate random keys for each trial
        if rng_seed is None:
            rng_seed = np.random.randint(0, 1000000)
        rng = jax.random.PRNGKey(rng_seed)
        rng_keys = jax.random.split(rng, num_trials)

        # Run all trials in parallel on GPU, ignore actC_batch from the output
        _, grid_point_batch = _run_trials_batched_jax(
            rng_keys, net_params, prefix, update_q_discrete
        )

        # Convert back to numpy for compatibility with existing code
        # actC_batch = np.array(actC_batch)
        grid_point_batch = np.array(grid_point_batch)

        # print(f"GPU execution time: {time.time() - t0:.3f}s")

        # Process results (same as original - aggregate unique states)
        # CRITICAL FIX: Use grid points (discrete) not continuous actC for aggregation
        # Convert grid point indices to one-hot actC vectors (like CPU version does)
        # t_post = time.time()

        # Store continuous actC for return value
        # actC_list = actC_batch.tolist()  # Fast batch conversion

        # OPTIMIZATION 1: Vectorized one-hot encoding
        # Instead of looping, use advanced indexing
        # grid_point_batch shape: (num_trials, num_roles)
        # We want: actC_discrete[trial, role_idx * num_fillers + filler_idx] = 1.0

        actC_discrete_batch = np.zeros((num_trials, self.num_bindings))
        role_indices = np.arange(self.num_roles)  # [0, 1, 2, ..., num_roles-1]
        state_counts = {}  # {tuple(grid_point): count}
        for trial_id in range(num_trials):
            binding_indices = role_indices * self.num_fillers + \
                grid_point_batch[trial_id].astype(int)
            actC_discrete_batch[trial_id, binding_indices] = 1.0

        # OPTIMIZATION 2: Use dictionary with tuple keys for O(1) lookup Instead of list membership testing and list.index() which are O(n)

        # for trial_id in range(num_trials):
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
        # print(f"Post-processing time: {time.time() - t_post:.3f}s")

        stat = self.get_corpus_stat(corpus)
        return stat

    def get_role_and_daughter_indices_fast(self, role_idx):
        """
        Fast lookup for a role and its daughters' binding indices.

        This is a common pattern in training loops:
            for role in self.role_names:
                idx = self.find_roles(role)
                daughters = self.hg.roles.get_daughters(role)
                idx_l = self.find_roles(daughters['l'][0])
                idx_r = self.find_roles(daughters['r'][0])

        Becomes:
            for ri in range(len(self.hg.role_names)):
                indices = self.get_role_and_daughter_indices_fast(ri)
                idx = indices['self']
                idx_l = indices['l']
                idx_r = indices['r']

        Args:
            role_idx: int - index of role (0 to num_roles-1)

        Returns:
            dict or None: {'self': array, 'l': array, 'r': array} or None if terminal
        """
        return self.role_daughter_binding_indices.get(role_idx)

    def find_roles_by_filler_fast(self, filler_indices):
        """
        Find all bindings that have the given filler(s).

        Useful pattern: Get all bindings for specific fillers across all roles.

        Args:
            filler_indices: int or list of int - filler index/indices

        Returns:
            np.array: binding indices
        """
        if not isinstance(filler_indices, list):
            filler_indices = [filler_indices]

        result = []
        for fi in filler_indices:
            if fi in self.filler_to_binding_indices:
                result.append(self.filler_to_binding_indices[fi])

        if result:
            return np.concatenate(result)
        else:
            return np.array([], dtype=np.int32)

    def clear_input(self):

        if self.use_jax:
            self.extC = jnp.zeros(self.num_bindings, dtype=jnp.float32)
        else:
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

        idx = self.find_bindings_fast(binding_names)
        if self.use_jax:
            # JAX version: use JAX arrays
            curr_extC = jnp.zeros(self.num_bindings, dtype=jnp.float32)
            if len(idx) > 0:
                if isinstance(idx, list):
                    curr_extC = curr_extC.at[np.array(idx)].set(1.0)
                else:
                    curr_extC = curr_extC.at[idx].set(1.0)
            self.extC = self.extC + self.estr * curr_extC
        else:
            # NumPy version
            curr_extC = np.zeros(self.num_bindings)
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
            rname_terminal_indices = np.where(
                self.hg.roles.role_is_terminal)[0]
            terminal_rnames = [self.role_names[ri]
                               for ri in rname_terminal_indices]
            terminal_rnames = set(terminal_rnames)
            for key in keys_all:
                # if key in self.find_roles(self.hg.roles.get_terminals()):
                if key in set(terminal_rnames):
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

        # rnames_terminal = self.hg.roles.get_terminals()
        rname_terminal = np.where(self.hg.roles.role_is_terminal)[0]
        # idx_terminal = self.find_roles(rnames_terminal)
        idx_terminal = np.concatenate([self.role_to_binding_indices[ri]
                                       for ri in rname_terminal])

        # Initialize gradients with correct array type
        if self.use_jax:
            dWC = jnp.zeros(self.WC.shape, dtype=jnp.float32)
            dbC = jnp.zeros(self.bC.shape, dtype=jnp.float32)
            destr = jnp.zeros(self.estr.shape, dtype=jnp.float32)
            dq = jnp.zeros(self.num_roles, dtype=jnp.float32)
        else:
            # For sparse WC, use sparse gradient accumulator
            if hasattr(self, 'use_sparse') and self.use_sparse:
                dWC = sparse.dok_matrix(self.WC.shape, dtype=np.float64)
            else:
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

                        key_idx = np.array(list(key), dtype=np.int32)
                        if self.use_jax:
                            state = jnp.zeros(
                                self.num_bindings, dtype=jnp.float32)
                            state = state.at[key_idx].set(1.0)
                        else:
                            state = np.zeros(self.num_bindings)
                            state[key_idx] = 1.
                        # dbC += state * self.train_opts['mask0'] * val * self.train_opts['coef']['trees']
                        if self.use_jax:
                            dbC = dbC + state * val * \
                                self.train_opts['coef']['trees']
                        else:
                            dbC += state * val * \
                                self.train_opts['coef']['trees']

                        if self.train_opts['update_estr']:
                            if self.train_opts['update_estr_terminals_only']:
                                idx_tb = [ii for ii in list(
                                    key) if ii in idx_terminal]
                            else:
                                idx_tb = list(key)
                            idx_tb = np.array(idx_tb, dtype=np.int32)
                            if self.use_jax:
                                destr = destr.at[idx_tb].add(
                                    extC_token[idx_tb] * val * self.train_opts['coef']['trees'])
                            else:
                                destr[idx_tb] += extC_token[idx_tb] * \
                                    val * self.train_opts['coef']['trees']

            if self.train_opts['coef']['treelets'] > 0.:
                for key, val in err['treelets'].items():

                    if key in keys_treelet:  # pwc: new
                        key = np.array(list(key), dtype=np.int32)
                        if self.use_jax:
                            dbC = dbC.at[key[0]].add(
                                val * self.train_opts['coef']['treelets'])
                        else:
                            dbC[key[0]] += val * \
                                self.train_opts['coef']['treelets']

                        if self.train_opts['update_estr']:
                            if not self.train_opts['update_estr_terminals_only']:
                                if self.use_jax:
                                    destr = destr.at[key].add(
                                        extC_token[key] * val * self.train_opts['coef']['treelets'])
                                else:
                                    destr[key] += extC_token[key] * \
                                        val * \
                                        self.train_opts['coef']['treelets']

                for key, val in err['bindings'].items():

                    if key in keys_binding:
                        if key in idx_terminal:
                            if self.use_jax:
                                dbC = dbC.at[key].add(
                                    val * self.train_opts['coef']['treelets'])
                            else:
                                dbC[key] += val * \
                                    self.train_opts['coef']['treelets']

                            if self.train_opts['update_estr']:
                                if self.use_jax:
                                    destr = destr.at[key].add(
                                        extC_token[key] * val * self.train_opts['coef']['treelets'])
                                else:
                                    destr[key] += extC_token[key] * val * \
                                        self.train_opts['coef']['treelets']

                                # print('bname =', self.binding_names[key])
                                # print('extC =', extC_token[key])
                                # print('val =', val)
                                # print('grad =', extC_token[key] * val *
                                #       self.train_opts['coef']['treelets'])
                                # # print('2', destr)

            if self.train_opts['coef']['binding_pairs'] > 0.:
                for key, val in err['binding_pairs'].items():
                    key = np.array(list(key), dtype=np.int32)
                    if self.use_jax:
                        dbC = dbC.at[key[0]].add(
                            val * self.train_opts['coef']['binding_pairs'])
                        dbC = dbC.at[key[1]].add(
                            val * self.train_opts['coef']['binding_pairs'])
                    else:
                        dbC[key[0]] += val * \
                            self.train_opts['coef']['binding_pairs']
                        dbC[key[1]] += val * \
                            self.train_opts['coef']['binding_pairs']

            if self.train_opts['coef']['bindings'] > 0.:
                for key, val in err['bindings'].items():
                    if self.use_jax:
                        dbC = dbC.at[key].add(
                            val * self.train_opts['coef']['bindings'])
                    else:
                        dbC[key] += val * self.train_opts['coef']['bindings']
                    if self.train_opts['update_estr']:
                        if self.use_jax:
                            destr = destr.at[key].add(
                                extC_token[key] * val * self.train_opts['coef']['bindings'])
                        else:
                            destr[key] += extC_token[key] * val * \
                                self.train_opts['coef']['bindings']

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

                        key_idx = np.array(list(key), dtype=np.int32)
                        if self.use_jax:
                            state = jnp.zeros(
                                self.num_bindings, dtype=jnp.float32)
                            state = state.at[key_idx].set(1.0)
                            dWC = dWC + jnp.outer(state, state) * \
                                self.train_opts['mask0'] * val * \
                                self.train_opts['coef']['trees']
                        else:
                            # state = np.zeros(self.num_bindings)
                            # state[key_idx] = 1.
                            # dWC += np.outer(state, state) * \
                            #     self.train_opts['mask0'] * val * \
                            #     self.train_opts['coef']['trees']
                            # For sparse matrices, avoid np.outer() which creates dense matrix
                            if hasattr(self, 'use_sparse') and self.use_sparse:
                                # Compute gradient coefficient
                                coef_val = val * \
                                    self.train_opts['coef']['trees']
                                # Directly update sparse matrix at (i,j) for all i,j in key_idx
                                # This is equivalent to outer(state, state) but sparse-friendly
                                for i in key_idx:
                                    for j in key_idx:
                                        # Check mask0 before updating (mask0 is sparse)
                                        if self.train_opts['mask0'][i, j] != 0:
                                            dWC[i, j] = dWC[i, j] + coef_val
                            else:
                                state = np.zeros(self.num_bindings)
                                state[key_idx] = 1.
                                dWC += np.outer(state, state) * \
                                    self.train_opts['mask0'] * val * \
                                    self.train_opts['coef']['trees']

                        if self.train_opts['update_estr']:
                            if self.train_opts['update_estr_terminals_only']:
                                idx_tb = [ii for ii in list(
                                    key) if ii in idx_terminal]
                            else:
                                idx_tb = list(key)
                            idx_tb = np.array(idx_tb, dtype=np.int32)
                            if self.use_jax:
                                destr = destr.at[idx_tb].add(
                                    extC_token[idx_tb] * val * self.train_opts['coef']['trees'])
                            else:
                                destr[idx_tb] += extC_token[idx_tb] * \
                                    val * self.train_opts['coef']['trees']

            if self.train_opts['coef']['treelets'] > 0.:
                for key, val in err['treelets'].items():

                    if key in keys_treelet:  # pwc: new
                        key = np.array(list(key), dtype=np.int32)
                        coef_val = val * self.train_opts['coef']['treelets']
                        if not self.train_opts['bias_only']:
                            if self.use_jax:
                                dWC = dWC.at[key[0], key[1]].add(coef_val)
                                dWC = dWC.at[key[1], key[0]].add(coef_val)
                                dWC = dWC.at[key[0], key[2]].add(coef_val)
                                dWC = dWC.at[key[2], key[0]].add(coef_val)
                            else:
                                dWC[key[0], key[1]] += coef_val
                                dWC[key[1], key[0]] += coef_val
                                dWC[key[0], key[2]] += coef_val
                                dWC[key[2], key[0]] += coef_val

                        if self.use_jax:
                            dWC = dWC.at[key[0], key[0]].add(coef_val)
                        else:
                            dWC[key[0], key[0]] += coef_val

                        if self.train_opts['update_estr']:
                            if not self.train_opts['update_estr_terminals_only']:
                                if self.use_jax:
                                    destr = destr.at[key].add(
                                        extC_token[key] * val * self.train_opts['coef']['treelets'])
                                else:
                                    destr[key] += extC_token[key] * \
                                        val * \
                                        self.train_opts['coef']['treelets']

                for key, val in err['bindings'].items():

                    if key in keys_binding:
                        if key in idx_terminal:
                            coef_val = val * \
                                self.train_opts['coef']['treelets']
                            if self.use_jax:
                                dWC = dWC.at[key, key].add(coef_val)
                            else:
                                dWC[key, key] += coef_val

                            if self.train_opts['update_estr']:

                                if self.use_jax:
                                    destr = destr.at[key].add(
                                        extC_token[key] * coef_val)
                                else:
                                    destr[key] += extC_token[key] * coef_val

                                # print('bname =', self.binding_names[key])
                                # print('extC =', extC_token[key])
                                # print('val =', val)
                                # print('grad =', extC_token[key] * val *
                                #       self.train_opts['coef']['treelets'])
                                # # print('2', destr)

            if self.train_opts['coef']['binding_pairs'] > 0.:
                for key, val in err['binding_pairs'].items():
                    key = list(key)
                    coef_val = val * self.train_opts['coef']['binding_pairs']
                    if self.use_jax:
                        dWC = dWC.at[key[0], key[1]].add(coef_val)
                        dWC = dWC.at[key[1], key[0]].add(coef_val)
                    else:
                        dWC[key[0], key[1]] += coef_val
                        dWC[key[1], key[0]] += coef_val

            if self.train_opts['coef']['bindings'] > 0.:
                for key, val in err['bindings'].items():
                    coef_val = val * self.train_opts['coef']['bindings']
                    if self.use_jax:
                        dWC = dWC.at[key, key].add(coef_val)
                    else:
                        dWC[key, key] += coef_val
                    if self.train_opts['update_estr']:
                        if self.use_jax:
                            destr = destr.at[key].add(
                                extC_token[key] * coef_val)
                        else:
                            destr[key] += extC_token[key] * coef_val

            # ENTROPY (use parse structures)
            if self.train_opts['coef_q'] > 0.:
                dq = -err['ent_diff'] * self.train_opts['coef_q']
                # print(dq)

        return dWC, destr, dq, dbC

    def average_weight2(self):

        WC_L = 0.
        WC_R = 0.
        WC_S = 0.   # sister roles
        count_L = 0
        count_R = 0
        count_S = 0
        # for role in self.role_names:
        #     if not self.hg.roles.is_terminal(role):
        #         daughters = self.hg.roles.get_daughters(role)
        #         daughter_l = daughters['l'][0]
        #         daughter_r = daughters['r'][0]
        #         idx = self.find_roles(role)
        #         idx_l = self.find_roles(daughter_l)
        #         idx_r = self.find_roles(daughter_r)
        for ri in range(len(self.hg.role_names)):
            if not self.hg.roles.role_is_terminal[ri]:
                indices = self.get_role_and_daughter_indices_fast(ri)
                if indices != None:
                    idx = indices['self']
                    idx_l = indices['l']
                    idx_r = indices['r']
                    count_L += 1
                    count_R += 1
                    count_S += 1
                    WC_L += self.WC[np.ix_(idx, idx_l)]
                    WC_R += self.WC[np.ix_(idx, idx_r)]
                    WC_S += self.WC[np.ix_(idx_l, idx_r)]

        WC_L /= float(count_L)
        WC_R /= float(count_R)
        WC_S /= float(count_S)

        # Use sparse zeros for sparse WC
        if hasattr(self, 'use_sparse') and self.use_sparse:
            WC_avg = sparse.dok_matrix(self.WC.shape, dtype=np.float64)
        else:
            WC_avg = np.zeros(self.WC.shape)

        # for role in self.role_names:
        #     if not self.hg.roles.is_terminal(role):
        #         daughters = self.hg.roles.get_daughters(role)
        #         daughter_l = daughters['l'][0]
        #         daughter_r = daughters['r'][0]
        #         idx = self.find_roles(role)
        #         idx_l = self.find_roles(daughter_l)
        #         idx_r = self.find_roles(daughter_r)
        for ri in range(len(self.hg.role_names)):
            if not self.hg.roles.role_is_terminal[ri]:
                indices = self.get_role_and_daughter_indices_fast(ri)
                if indices != None:
                    idx = indices['self']
                    idx_l = indices['l']
                    idx_r = indices['r']
                   # FIXED: Avoid np.ix_() for sparse matrices (causes densification)
                    if hasattr(self, 'use_sparse') and self.use_sparse:
                        # Explicit loops to avoid densification
                        for i_pos, i in enumerate(idx):
                            for j_pos, j in enumerate(idx_l):
                                WC_avg[i, j] = WC_L[i_pos, j_pos]
                                WC_avg[j, i] = WC_L[j_pos, i_pos]  # Transpose
                        for i_pos, i in enumerate(idx):
                            for j_pos, j in enumerate(idx_r):
                                WC_avg[i, j] = WC_R[i_pos, j_pos]
                                WC_avg[j, i] = WC_R[j_pos, i_pos]  # Transpose
                        # Sister harmony (usually 0 in default setting)
                        if WC_S != 0:
                            for i_pos, i in enumerate(idx_l):
                                for j_pos, j in enumerate(idx_r):
                                    WC_avg[i, j] = WC_S[i_pos, j_pos]
                                    # Transpose
                                    WC_avg[j, i] = WC_S[j_pos, i_pos]
                    else:
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
        # for role in self.role_names:
        #     if not self.hg.roles.is_terminal(role):
        #         daughters = self.hg.roles.get_daughters(role)
        #         daughter_l = daughters['l'][0]
        #         daughter_r = daughters['r'][0]
        #         idx = self.find_roles(role)
        #         idx_l = self.find_roles(daughter_l)
        #         idx_r = self.find_roles(daughter_r)
        for ri in range(len(self.hg.role_names)):
            if not self.hg.roles.role_is_terminal[ri]:
                indices = self.get_role_and_daughter_indices_fast(ri)
                if indices != None:
                    idx = indices['self']
                    idx_l = indices['l']
                    idx_r = indices['r']
                    count_L += 1
                    count_R += 1
                    count_S += 1
                    WC_L += self.WC[np.ix_(idx, idx_l)]
                    WC_R += self.WC[np.ix_(idx, idx_r)]
                    WC_S += self.WC[np.ix_(idx_l, idx_r)]

        WC_L /= float(count_L)
        WC_R /= float(count_R)
        WC_S /= float(count_S)

        # for role in self.role_names:
        #     if not self.hg.roles.is_terminal(role):
        #         daughters = self.hg.roles.get_daughters(role)
        #         daughter_l = daughters['l'][0]
        #         daughter_r = daughters['r'][0]
        #         idx = self.find_roles(role)
        #         idx_l = self.find_roles(daughter_l)
        #         idx_r = self.find_roles(daughter_r)
        for ri in range(len(self.hg.role_names)):
            if not self.hg.roles.role_is_terminal[ri]:
                indices = self.get_role_and_daughter_indices_fast(ri)
                if indices != None:
                    idx = indices['self']
                    idx_l = indices['l']
                    idx_r = indices['r']
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

            # Extract diagonal - works for both dense and sparse
            if hasattr(self, 'use_sparse') and self.use_sparse:
                bC = self.WC.diagonal().copy()
            else:
                bC = np.diag(self.WC).copy()

            # Subtract diagonal - works for both dense and sparse
            if hasattr(self, 'use_sparse') and self.use_sparse:
                # Sparse: set diagonal to zero
                WC0 = self.WC.copy()
                # Set diagonal elements to 0 individually (works with any sparse format)
                WC0.setdiag(0)
            else:
                # Dense: standard subtraction
                WC0 = self.WC - np.diag(bC)

            # bC = np.tile(self.vec2mat(bC).mean(axis=1), self.num_roles)
            # self.WC = WC0 + np.diag(bC)
            # self._set_weights()

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:

                    roots = self.hg.g.get_roots() + [self.hg.g.opts['f_root']]
                    # rid = self.find_roles(self.role_names[-1])
                    # rid = [
                    #     ii for ii in rid if ii in self.find_fillers(roots)]
                    # rid = self.find_roles(self.role_names[-1])  # top brick role
                    root_filler_indices = self.find_fillers_fast(roots)
                    role_idx = len(self.hg.role_names) - 1
                    role_binding_idx = self.role_to_binding_indices[role_idx]
                    root_bindings = np.concatenate([
                        self.filler_to_binding_indices[fi]
                        for fi in root_filler_indices
                    ])
                    rid = np.intersect1d(role_binding_idx, root_bindings)
                    bC[rid] -= 2.   # NOTE: second-order bias = 2 * first-order bias

            idx = self.train_opts['idx_mask_bias2']

            mask = np.ones(self.num_bindings)
            mask[idx] = np.nan
            fbias_avg = np.nanmean(self.vec2mat(bC * mask), axis=1)
            bC_new = np.tile(fbias_avg, self.num_roles)
            bC_new[idx] = bC[idx]

            if 'free_update_null' in self.train_opts:
                if self.train_opts['free_update_null']:
                    # idx_null = self.find_fillers(self.hg.g.opts['null'])
                    idx_null = self.find_fillers_fast(self.hg.g.opts['null'])
                    bC_new[idx_null] = bC[idx_null]

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:
                    bC_new[rid] += 2.

            # Add new diagonal - works for both dense and sparse
            if hasattr(self, 'use_sparse') and self.use_sparse:
                # Sparse: set new diagonal
                # FIXED: Use setdiag() for fast diagonal modification on CSR matrices
                WC0.setdiag(bC_new)
                self.WC = WC0
            else:
                # Dense: standard addition
                self.WC = WC0 + np.diag(bC_new)

            self._set_weights()

        else:

            # self.bC = np.tile(
            #     self.vec2mat(self.bC).mean(axis=1), self.num_roles)
            # self._set_biases()

            if 'add1_to_root' in self.hg.opts:
                if self.hg.opts['add1_to_root']:
                    roots = self.hg.g.get_roots() + [self.hg.g.opts['f_root']]
                    # rid = self.find_roles(self.role_names[-1])
                    root_filler_indices = self.find_fillers_fast(roots)
                    role_idx = len(self.hg.role_names) - 1
                    role_binding_idx = self.role_to_binding_indices[role_idx]
                    root_bindings = np.concatenate([
                        self.filler_to_binding_indices[fi]
                        for fi in root_filler_indices
                    ])
                    rid = np.intersect1d(role_binding_idx, root_bindings)
                    # rid = [
                    #     ii for ii in root_filler_indices if ii in self.find_fillers(roots)]
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
        # eigvals, eigvecs = np.linalg.eigh(self.WC)
        # eig_max = max(eigvals)

        # For sparse matrices, use scipy.sparse.linalg.eigsh to compute only largest eigenvalue
        if hasattr(self, 'use_sparse') and self.use_sparse:
            from scipy.sparse.linalg import eigsh
            print("    Computing largest eigenvalue of sparse WC for bowl strength...")
            try:
                # Compute only the largest eigenvalue (k=1, which='LA')
                # This is MUCH faster than computing all eigenvalues
                eig_max = eigsh(self.WC, k=1, which='LA',
                                return_eigenvectors=False)[0]
                print(f"      Largest eigenvalue: {eig_max:.6f}")
            except Exception as e:
                print(
                    f"      Warning: eigsh failed ({e}), using default bowl strength")
                # Fallback: use a conservative estimate based on matrix norm
                eig_max = 0.0  # Will be overridden by other conditions or default
        else:
            # Dense matrices: use full eigenvalue decomposition
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
        if self.use_jax:
            ssq = jnp.sum(actCmat ** 2, axis=0)
        else:
            ssq = np.sum(actCmat ** 2, axis=0)
        hgrad_q1 = -4 * self.opts['m'] * actC * self.extend_rvec(rvec=ssq - 1)
        return (hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1)
 #####################################
 # PLOT

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
                    self.runC(word_rt[ii], log_trace=False)
                else:
                    self.runC(word_rt0[ii], log_trace=False)
                    self.opts['q_rate'] = 0.
                    self.runC(word_rt[ii] - word_rt0[ii], log_trace=False)
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
            self.runC((self.opts['q_max'] - min(self.q)) /
                      self.opts['q_rate'], log_trace=False)
        else:
            # self.run((self.opts['q_max'] - self.qpolicy[:ii+1].sum()) / self.opts['q_rate'])
            self.run((self.opts['q_max'] - min(self.q)) / self.opts['q_rate'])
        if disp:
            print(sent)
            self.plot_tree2(scale=1.5)

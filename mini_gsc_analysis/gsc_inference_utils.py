"""Shared inference and metric utilities for probing trained GscNet models.

This module supplies the building blocks used by the per-set ``probe_<set>.py``
scripts:

* ``load_net``: load a pickled ``GscNet`` checkpoint.
* ``run_sentence``: run a sentence through the network, capture per-step traces
  and per-word snapshots. Cached by sentence string so each unique sentence
  runs at most once across an analysis.
* ``[D]`` extractors (sentence-agnostic per-sentence difficulty metrics):
  ``D_surp``, ``D_H``, ``D_settle``, ``D_ent``, ``D_burst``, ``D_path``.
  Each provides ``..._total(net, run)`` and ``..._per_word(net, run)``.
* ``[J]`` extractors (joint per-pair divergence): final-state cosine distance,
  mean per-role KL, argmax-mismatch fraction, log-prob gap (= ``|D_surp|`` gap),
  peak per-word divergence.
* ``sample_control_pairs``: random pair sampler from ``net.corpus`` (or random
  POS sequences) excluding user-specified target sentences.
* Statistics helpers for Design A and Design B (whole-sequence and pooled
  critical-window null).

The framework targets ``GscNet`` instances created by ``only_gscnet_speedup``
(used by ``SAP_analysis/train_*.py``). That class does NOT define ``H``,
``Hg`` or related harmony methods, so we recompute them post-hoc from
``actC``, ``q``, ``extC`` using the same decomposition as ``gsc.py``.
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Path setup so this module can be imported either from the repo root or
# from inside ``mini_gsc_analysis/``.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_net(model_path: str):
    """Load a pickled ``GscNet`` checkpoint.

    Re-implements ``only_gscnet_speedup.load_model`` so callers do not need to
    import that module directly. The pickled object brings in its own class
    references; we just need ``only_gscnet_speedup`` (and friends) to be
    importable, which the ``sys.path`` adjustment above ensures.
    """
    with open(model_path, "rb") as fh:
        net = pickle.load(fh)
    return net


# ---------------------------------------------------------------------------
# Sentence string / binding name helpers
# ---------------------------------------------------------------------------


def pos_tokens(sentence_str: str) -> list[str]:
    """Split a whitespace-separated POS sequence string into a token list."""
    return sentence_str.strip().split()


def bnames_for_sentence(net, pos_seq: Sequence[str]) -> list[str]:
    """Return the list of binding names ``f/(1,wpos)`` for a POS sequence."""
    bsep = net.hg.opts["bsep"]
    return [f"{tok}{bsep}(1,{i + 1})" for i, tok in enumerate(pos_seq)]


def sentence_key(pos_seq: Sequence[str]) -> str:
    """Canonical string key for caching runs."""
    return " ".join(pos_seq)


# ---------------------------------------------------------------------------
# Harmony reconstruction (post-hoc, from traced actC / q / extC)
#
# GscNet in only_gscnet_speedup does not define H / Hg / Hb / Hq0 / Hq1 as
# methods on the class. We recreate the same decomposition used in gsc.py,
# verified by working backward from HGradC.
# ---------------------------------------------------------------------------


def _extend_rvec(net, rvec: np.ndarray) -> np.ndarray:
    """Vectorised version of net.extend_rvec compatible with batched arrays.

    For a 1-D rvec of shape (num_roles,), returns a (num_bindings,) array.
    For a 2-D rvec of shape (T, num_roles), returns a (T, num_bindings) array.
    """
    rvec = np.asarray(rvec)
    if rvec.ndim == 1:
        return np.tile(rvec, (net.num_fillers, 1)).flatten("F")
    if rvec.ndim == 2:
        T = rvec.shape[0]
        out = np.empty((T, net.num_bindings))
        for i in range(T):
            out[i] = np.tile(rvec[i], (net.num_fillers, 1)).flatten("F")
        return out
    raise ValueError(f"unsupported rvec shape {rvec.shape}")


def _vec2mat(net, actC: np.ndarray) -> np.ndarray:
    """Reshape ``actC`` (num_bindings,) into (num_fillers, num_roles)."""
    return np.asarray(actC).reshape(
        (net.num_fillers, net.num_roles), order="F"
    )


def compute_H(net, actC: np.ndarray, q: np.ndarray,
              extC: np.ndarray) -> float:
    """Recompute the total harmony H = Hg + Hb + Hq0 + Hq1.

    Same decomposition used in ``gsc.py``: derived by integrating
    ``HGradC`` in ``only_gscnet_speedup``.
    """
    actC = np.asarray(actC, dtype=float)
    q = np.asarray(q, dtype=float)
    extC = np.asarray(extC, dtype=float)
    Hg = (0.5 * float(actC @ (net.WC @ actC))
          + float(net.bC @ actC)
          + float(extC @ actC))
    bowl_center = np.asarray(net.bowl_center, dtype=float)
    Hb = (-0.5 * float(net.opts["bowl_strength"])
          * float(np.sum((actC - bowl_center) ** 2)))
    q_ext = _extend_rvec(net, q)
    Hq0 = -float(np.sum(q_ext * actC ** 2 * (1 - actC) ** 2))
    actCmat = _vec2mat(net, actC)
    ssq = np.sum(actCmat ** 2, axis=0)
    Hq1 = -float(net.opts["m"]) * float(np.sum((ssq - 1) ** 2))
    return Hg + Hb + Hq0 + Hq1


def compute_HGradC(net, actC: np.ndarray, q: np.ndarray,
                   extC: np.ndarray) -> np.ndarray:
    """Recompute the harmony gradient using the same formula as HGradC."""
    actC = np.asarray(actC, dtype=float)
    q = np.asarray(q, dtype=float)
    extC = np.asarray(extC, dtype=float)
    actCmat = _vec2mat(net, actC)
    bowl_center = np.asarray(net.bowl_center, dtype=float)
    hgrad_g = net.WC @ actC + np.asarray(net.bC) + extC
    hgrad_b = float(net.opts["bowl_strength"]) * (bowl_center - actC)
    q_ext = _extend_rvec(net, q)
    hgrad_q0 = -2 * q_ext * actC * (1 - actC) * (1 - 2 * actC)
    ssq = np.sum(actCmat ** 2, axis=0)
    hgrad_q1 = (-4 * float(net.opts["m"])
                * actC * _extend_rvec(net, ssq - 1))
    return np.asarray(hgrad_g + hgrad_b + hgrad_q0 + hgrad_q1, dtype=float)


def compute_H_trace(net, actC_trace: np.ndarray, q_trace: np.ndarray,
                    extC_trace: np.ndarray) -> np.ndarray:
    """Vectorised post-hoc harmony for a full trace.

    ``actC_trace``, ``q_trace`` and ``extC_trace`` should each be 2-D arrays
    over time, returned by ``run_sentence``. Returns a 1-D array of harmony
    values.
    """
    actC_trace = np.asarray(actC_trace, dtype=float)
    q_trace = np.asarray(q_trace, dtype=float)
    extC_trace = np.asarray(extC_trace, dtype=float)
    T = actC_trace.shape[0]
    H = np.empty(T)
    for i in range(T):
        H[i] = compute_H(net, actC_trace[i], q_trace[i], extC_trace[i])
    return H


def compute_grad_norm_trace(net, actC_trace: np.ndarray,
                            q_trace: np.ndarray,
                            extC_trace: np.ndarray) -> np.ndarray:
    """L2 norm of the harmony gradient at each step of the trace."""
    actC_trace = np.asarray(actC_trace, dtype=float)
    q_trace = np.asarray(q_trace, dtype=float)
    extC_trace = np.asarray(extC_trace, dtype=float)
    T = actC_trace.shape[0]
    out = np.empty(T)
    for i in range(T):
        out[i] = float(np.linalg.norm(
            compute_HGradC(net, actC_trace[i], q_trace[i], extC_trace[i])))
    return out


# ---------------------------------------------------------------------------
# Run a single sentence through the network with caching
# ---------------------------------------------------------------------------


@dataclass
class SentenceRun:
    """Container for everything we capture about one sentence's processing.

    Attributes:
        pos_seq: list of POS strings, e.g. ``['DT','NN','VBZ','DT','NN']``.
        n_words: sentence length.
        per_word: list of ``n_words + 1`` dicts with snapshots after each
            ``run_word`` call (and one extra after ``run_wrapup``):

                ``actC``: (num_bindings,) state at end of word.
                ``q``: (num_roles,) commitment vector.
                ``H``: float, harmony value at end of word.
                ``grad_norm``: float, L2 norm of HGradC at end.
                ``trace_slice``: (start, stop) into the per-step arrays.
                ``fname``: POS string, or ``'<wrapup>'``.

        traces: dict with ``'t'``, ``'actC'``, ``'q'``, ``'extC'``,
            ``'H'``, ``'grad_norm'`` arrays of shape (T_total, ...).
        seed: numpy seed used for the run.
    """

    pos_seq: list[str]
    n_words: int
    per_word: list[dict] = field(default_factory=list)
    traces: dict = field(default_factory=dict)
    seed: int = 0


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import jax  # noqa: F401
        # The jax key is reseeded inside net.reset() from numpy state.
    except Exception:
        pass


def run_sentence(net, pos_seq: Sequence[str], seed: int,
                 cache: Optional[dict] = None,
                 reset_sd: float = 0.01) -> SentenceRun:
    """Run a POS-token sequence through the network and capture traces.

    The same ``(sentence_string, seed)`` always returns the same result
    (cached). The cache is a plain dict supplied by the caller; pass the
    same dict across many calls to deduplicate work.
    """
    pos_seq = list(pos_seq)
    key = (sentence_key(pos_seq), int(seed))
    if cache is not None and key in cache:
        return cache[key]

    _set_seed(seed)
    if not hasattr(net, "ep") or net.ep is None:
        raise RuntimeError(
            "net.ep is not set; was the model initialised before saving?")

    net.reset(mu=net.ep, sd=reset_sd)
    net.store = []
    # Make sure the variables we need are tracked.
    desired = ["t", "actC", "q", "extC"]
    cur = list(net.opts.get("trace_varnames", []))
    for v in desired:
        if v not in cur:
            cur.append(v)
    net.opts["trace_varnames"] = cur
    net.initialize_traces(trace_list=desired)

    per_word: list[dict] = []

    def _trace_len() -> int:
        v = net.traces.get("t", [])
        return len(v) if isinstance(v, list) else int(v.shape[0])

    for wi, fname in enumerate(pos_seq):
        before = _trace_len()
        net.run_word(fname, wi + 1, log_trace=True)
        after = _trace_len()
        per_word.append({
            "fname": fname,
            "wpos": wi + 1,
            "actC": np.asarray(net.actC, dtype=float).copy(),
            "q": np.asarray(net.q, dtype=float).copy(),
            "extC": np.asarray(net.extC, dtype=float).copy(),
            "trace_slice": (before, after),
        })

    before = _trace_len()
    net.run_wrapup(log_trace=True)
    after = _trace_len()
    per_word.append({
        "fname": "<wrapup>",
        "wpos": len(pos_seq) + 1,
        "actC": np.asarray(net.actC, dtype=float).copy(),
        "q": np.asarray(net.q, dtype=float).copy(),
        "extC": np.asarray(net.extC, dtype=float).copy(),
        "trace_slice": (before, after),
    })

    traces = {k: np.asarray(net.traces[k]) for k in desired}
    H_trace = compute_H_trace(net, traces["actC"], traces["q"], traces["extC"])
    grad_trace = compute_grad_norm_trace(
        net, traces["actC"], traces["q"], traces["extC"])
    traces["H"] = H_trace
    traces["grad_norm"] = grad_trace

    for entry in per_word:
        a, b = entry["trace_slice"]
        if b > a:
            entry["H"] = float(H_trace[b - 1])
            entry["grad_norm"] = float(grad_trace[b - 1])
        else:
            entry["H"] = float("nan")
            entry["grad_norm"] = float("nan")

    run = SentenceRun(
        pos_seq=pos_seq, n_words=len(pos_seq),
        per_word=per_word, traces=traces, seed=int(seed),
    )
    if cache is not None:
        cache[key] = run
    return run


# ---------------------------------------------------------------------------
# [D] per-sentence difficulty metrics
#
# Every metric provides:
#   <name>_per_word(net, run) -> np.ndarray of length n_words.
#   <name>_total(net, run)    -> float (= mean of per-word values, with the
#                                appropriate normalisation already applied).
#
# All metrics are sentence-agnostic: they depend only on ``net`` (global
# constants) and ``run`` (this sentence's traces).
# ---------------------------------------------------------------------------


def _slice_trace(run: SentenceRun, key: str, wpos_idx: int) -> np.ndarray:
    """Per-step slice of the trace ``key`` covering one ``run_word`` call.

    ``wpos_idx`` is 0-indexed into ``run.per_word`` (so values 0..n_words-1
    correspond to the n words; ``n_words`` is the wrap-up slot).
    """
    a, b = run.per_word[wpos_idx]["trace_slice"]
    return run.traces[key][a:b]


def D_H_per_word(net, run: SentenceRun) -> np.ndarray:
    """Per-word harmony deficit: -H_final(word) / num_bindings.

    Higher = harder. Final-of-word harmony, normalised so values are
    cross-set comparable.
    """
    nb = float(net.num_bindings)
    out = np.array([-entry["H"] / nb for entry in run.per_word[:run.n_words]])
    return out


def D_H_total(net, run: SentenceRun) -> float:
    return float(np.mean(D_H_per_word(net, run)))


def D_settle_per_word(net, run: SentenceRun) -> np.ndarray:
    """Per-word settling difficulty: log(1 + mean ||grad H||) within run_word.

    Higher = the model spent the word in a region with steep harmony gradient
    (i.e. far from a fixed point). ``log(1+x)`` keeps the metric on a tame
    scale across phenomena.
    """
    out = np.empty(run.n_words)
    for w in range(run.n_words):
        gn = _slice_trace(run, "grad_norm", w)
        out[w] = np.log1p(float(np.mean(gn))) if len(gn) else float("nan")
    return out


def D_settle_total(net, run: SentenceRun) -> float:
    return float(np.nanmean(D_settle_per_word(net, run)))


def _per_role_distribution(actCmat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Square-and-normalise: p_r ∝ actCmat[:, r]**2.

    This matches the Hq1 quantisation constraint (sum_b actCmat[b,r]**2 ≈ 1).
    """
    sq = actCmat ** 2
    s = sq.sum(axis=0, keepdims=True)
    return sq / np.maximum(s, eps)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def D_ent_per_word(net, run: SentenceRun) -> np.ndarray:
    """Per-word mean per-role entropy of the per-role filler distribution.

    Direct readout of "how many parses are being considered simultaneously".
    Computed by averaging over the per-step entropies inside ``run_word``.
    """
    out = np.empty(run.n_words)
    for w in range(run.n_words):
        actC_steps = _slice_trace(run, "actC", w)
        if len(actC_steps) == 0:
            out[w] = float("nan")
            continue
        ents = []
        for actC in actC_steps:
            p = _per_role_distribution(_vec2mat(net, actC))
            ents.append(np.mean([_entropy(p[:, r]) for r in range(p.shape[1])]))
        out[w] = float(np.mean(ents))
    return out


def D_ent_total(net, run: SentenceRun) -> float:
    return float(np.nanmean(D_ent_per_word(net, run)))


def D_path_per_word(net, run: SentenceRun) -> np.ndarray:
    """Per-word integrated state-space path length.

    sum_t ||actC(t+1) - actC(t)||_2 within run_word, normalised by
    sqrt(num_bindings).
    """
    nb = np.sqrt(float(net.num_bindings))
    out = np.empty(run.n_words)
    for w in range(run.n_words):
        actC_steps = _slice_trace(run, "actC", w)
        if len(actC_steps) <= 1:
            out[w] = 0.0
            continue
        diffs = np.diff(actC_steps, axis=0)
        out[w] = float(np.sum(np.linalg.norm(diffs, axis=1))) / nb
    return out


def D_path_total(net, run: SentenceRun) -> float:
    return float(np.mean(D_path_per_word(net, run)))


def D_burst_per_word(net, run: SentenceRun,
                     threshold: Optional[float] = None) -> np.ndarray:
    """Per-word reanalysis-burst count.

    Counts how many times ||actC(t+1) - actC(t)|| / dt crosses a threshold
    upward. When ``threshold`` is ``None`` we use the 95th percentile of
    velocities across the whole sentence trace as a within-sentence
    self-calibrated threshold (a coarse but cross-sentence-comparable
    fallback).
    """
    actC_full = run.traces["actC"]
    if len(actC_full) <= 1:
        return np.zeros(run.n_words)
    diffs_full = np.diff(actC_full, axis=0)
    speeds_full = np.linalg.norm(diffs_full, axis=1)
    if threshold is None:
        threshold = (float(np.percentile(speeds_full, 95))
                     if len(speeds_full) else 0.0)
    out = np.zeros(run.n_words)
    for w in range(run.n_words):
        a, b = run.per_word[w]["trace_slice"]
        if b - a <= 1:
            continue
        ds = np.linalg.norm(np.diff(actC_full[a:b], axis=0), axis=1)
        crossings = np.sum((ds[1:] > threshold) & (ds[:-1] <= threshold))
        crossings += int(ds[0] > threshold)
        out[w] = float(crossings)
    return out


def D_burst_total(net, run: SentenceRun) -> float:
    return float(np.mean(D_burst_per_word(net, run)))


def D_surp_per_word(net, run: SentenceRun, eps: float = 1e-9) -> np.ndarray:
    """Per-word surprisal proxy.

    This is a cheap, sentence-agnostic stand-in for ``-log p(w_t | w_<t)``.
    For each word ``w`` we read out the per-role distribution from the
    end-of-word ``actCmat``, locate the role at level 1 / position ``w``
    (the terminal role for that word) and report ``-log p(filler_w)``
    where ``filler_w`` is the actual POS played at that position.

    Interpretation: if the model is confident in the right filler at the
    terminal role for word ``w`` after processing it, surprisal is low;
    otherwise high. This is much cheaper than running ``estimate_prob_inc``
    per prefix and is also more robust to the absence of ``run_prefix``
    on ``only_gscnet_speedup``.
    """
    out = np.empty(run.n_words)
    role_names = list(net.role_names)
    filler_names = list(net.filler_names)
    for w in range(run.n_words):
        rname = f"(1,{w + 1})"
        try:
            r_idx = role_names.index(rname)
        except ValueError:
            out[w] = float("nan")
            continue
        actC = run.per_word[w]["actC"]
        actCmat = _vec2mat(net, actC)
        p = _per_role_distribution(actCmat)
        try:
            f_idx = filler_names.index(run.per_word[w]["fname"])
        except ValueError:
            out[w] = float("nan")
            continue
        out[w] = float(-np.log(max(p[f_idx, r_idx], eps)))
    return out


def D_surp_total(net, run: SentenceRun) -> float:
    return float(np.nanmean(D_surp_per_word(net, run)))


# Public dictionary of all [D] metrics for iteration.
D_METRICS = {
    "D_surp": (D_surp_total, D_surp_per_word),
    "D_H": (D_H_total, D_H_per_word),
    "D_settle": (D_settle_total, D_settle_per_word),
    "D_ent": (D_ent_total, D_ent_per_word),
    "D_burst": (D_burst_total, D_burst_per_word),
    "D_path": (D_path_total, D_path_per_word),
}


# ---------------------------------------------------------------------------
# [J] joint per-pair divergence metrics
# ---------------------------------------------------------------------------


def J_final_cosine(net, run_a: SentenceRun, run_b: SentenceRun) -> float:
    """1 - cosine similarity between the final actC of A and B."""
    a = run_a.per_word[-1]["actC"]
    b = run_b.per_word[-1]["actC"]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(1.0 - float(a @ b) / (na * nb))


def J_final_l2(net, run_a: SentenceRun, run_b: SentenceRun) -> float:
    """L2 distance between final actC, normalised by sqrt(num_bindings)."""
    a = run_a.per_word[-1]["actC"]
    b = run_b.per_word[-1]["actC"]
    return float(np.linalg.norm(a - b) / np.sqrt(net.num_bindings))


def J_mean_role_kl(net, run_a: SentenceRun, run_b: SentenceRun,
                   eps: float = 1e-9) -> float:
    """Mean over roles of KL(p_A_r || p_B_r) using square-and-normalise dists."""
    p = _per_role_distribution(_vec2mat(net, run_a.per_word[-1]["actC"]))
    q = _per_role_distribution(_vec2mat(net, run_b.per_word[-1]["actC"]))
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=0)))


def J_argmax_mismatch(net, run_a: SentenceRun, run_b: SentenceRun) -> float:
    """Fraction of roles where the argmax filler differs at end of run."""
    pa = _vec2mat(net, run_a.per_word[-1]["actC"])
    pb = _vec2mat(net, run_b.per_word[-1]["actC"])
    return float(np.mean(np.argmax(pa, axis=0) != np.argmax(pb, axis=0)))


def J_logp_gap(net, run_a: SentenceRun, run_b: SentenceRun) -> float:
    """|D_surp_total(A) - D_surp_total(B)| -- equals |log p_model(A) - log p(B)|.

    Conventionally part of the log-prob test. We expose it as J for plotting
    parity with the other J metrics, and in cross-set ranking it duplicates
    the D_surp Design A statistic (reported under both for transparency).
    """
    return float(abs(D_surp_total(net, run_a) - D_surp_total(net, run_b)))


def J_peak_perword_gap(net, run_a: SentenceRun, run_b: SentenceRun) -> float:
    """max over words of |actC_A(t) - actC_B(t)|_2, normalised by sqrt(NB).

    Captures the peak per-word state-space divergence regardless of sentence
    length. For sentences of different length we align by word index (1..n)
    and stop at min(len_A, len_B).
    """
    n = min(run_a.n_words, run_b.n_words)
    if n == 0:
        return 0.0
    nb = np.sqrt(float(net.num_bindings))
    diffs = []
    for w in range(n):
        a = run_a.per_word[w]["actC"]
        b = run_b.per_word[w]["actC"]
        diffs.append(float(np.linalg.norm(a - b)) / nb)
    return float(np.max(diffs)) if diffs else 0.0


J_METRICS = {
    "J_final_cosine": J_final_cosine,
    "J_final_l2": J_final_l2,
    "J_mean_role_kl": J_mean_role_kl,
    "J_argmax_mismatch": J_argmax_mismatch,
    "J_logp_gap": J_logp_gap,
    "J_peak_perword_gap": J_peak_perword_gap,
}


# ---------------------------------------------------------------------------
# Control-pool sampling
# ---------------------------------------------------------------------------


def corpus_pos_sentences(net) -> list[list[str]]:
    """All sentences in ``net.corpus`` returned as POS-token lists."""
    out = []
    for sent in net.corpus["sentence"]:
        toks = [bn.split(net.hg.opts["bsep"])[0] for bn in sent]
        out.append(toks)
    return out


def sample_control_pairs(
    net,
    n_pairs: int,
    exclude_sentences: Iterable[str],
    source: str = "corpus",
    seed: int = 0,
    length_pool: Optional[Iterable[int]] = None,
) -> list[tuple[list[str], list[str]]]:
    """Sample random sentence pairs for the null distribution.

    Args:
        net: trained GscNet.
        n_pairs: how many pairs to draw.
        exclude_sentences: iterable of canonical sentence strings (space-joined
            POS sequences) that must not appear in any control pair.
        source: ``'corpus'`` (default) draws unique pairs from
            ``net.corpus['sentence']``. ``'random_pos'`` synthesises random
            POS sequences using filler names found in the corpus.
        seed: RNG seed for sampling.
        length_pool: when ``source='random_pos'``, the lengths to draw from.
            Defaults to lengths observed in ``net.corpus``.
    """
    rng = np.random.RandomState(seed)
    excl = set(exclude_sentences)
    pairs: list[tuple[list[str], list[str]]] = []
    if source == "corpus":
        all_sents = corpus_pos_sentences(net)
        eligible = [s for s in all_sents if sentence_key(s) not in excl]
        if len(eligible) < 2:
            raise RuntimeError(
                f"only {len(eligible)} eligible control sentences; need >= 2")
        max_unique_pairs = len(eligible) * (len(eligible) - 1) // 2
        seen = set()
        attempts = 0
        max_attempts = max(20 * n_pairs, 1000)
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            i, j = rng.choice(len(eligible), size=2, replace=False)
            a = eligible[i]
            b = eligible[j]
            kpair = tuple(sorted([sentence_key(a), sentence_key(b)]))
            if kpair in seen:
                continue
            seen.add(kpair)
            pairs.append((a, b))
        if len(pairs) < n_pairs:
            print(
                f"warning: only sampled {len(pairs)} unique pairs out of "
                f"{n_pairs} requested ({max_unique_pairs} possible)")
        return pairs
    elif source == "random_pos":
        all_sents = corpus_pos_sentences(net)
        if length_pool is None:
            length_pool = sorted({len(s) for s in all_sents})
        # Use ALL POS tokens that appear in the corpus across positions.
        token_pool = sorted({tok for s in all_sents for tok in s})
        for _ in range(n_pairs):
            la = int(rng.choice(list(length_pool)))
            lb = int(rng.choice(list(length_pool)))
            a = list(rng.choice(token_pool, size=la))
            b = list(rng.choice(token_pool, size=lb))
            if sentence_key(a) in excl or sentence_key(b) in excl:
                continue
            pairs.append((a, b))
        return pairs
    else:
        raise ValueError(f"unknown source: {source!r}")


# ---------------------------------------------------------------------------
# Statistics: target vs control
# ---------------------------------------------------------------------------


def target_vs_control_stats(target: float, controls: np.ndarray) -> dict:
    """Compute z, one-sided percentile, and basic descriptive stats."""
    controls = np.asarray(controls, dtype=float)
    controls = controls[~np.isnan(controls)]
    if controls.size == 0:
        return {"z": float("nan"), "percentile": float("nan"),
                "p_one_sided": float("nan"),
                "control_mean": float("nan"), "control_std": float("nan"),
                "n_control": 0, "target": target}
    mu = float(np.mean(controls))
    sd = float(np.std(controls, ddof=1)) if controls.size > 1 else 0.0
    z = (target - mu) / sd if sd > 0 else float("nan")
    # one-sided upper-tail p-value: fraction of controls >= target
    p = float(np.mean(controls >= target))
    pct = float(np.mean(controls < target) * 100.0)
    return {"z": float(z), "percentile": pct, "p_one_sided": p,
            "control_mean": mu, "control_std": sd,
            "n_control": int(controls.size), "target": float(target)}


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    if a.size == 1 and b.size == 1:
        return float("nan")
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    pooled = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b)
                     / max(a.size + b.size - 2, 1))
    if pooled <= 0:
        return float("nan")
    return float((mu_a - mu_b) / pooled)


# ---------------------------------------------------------------------------
# Design A and Design B test runners
# ---------------------------------------------------------------------------


def whole_sequence_gap(d_total_a: float, d_total_b: float) -> float:
    """Design A statistic: |D_total(A) - D_total(B)|."""
    return float(abs(d_total_a - d_total_b))


def critical_window_value(per_word: np.ndarray, d: int, spillover: int) -> float:
    """Mean of ``per_word`` over the window ``[d-1, d-1+spillover]`` (1-indexed d).

    Returns NaN if the window falls outside the array.
    """
    per_word = np.asarray(per_word, dtype=float)
    start = d - 1
    stop = min(start + spillover + 1, per_word.size)
    if start < 0 or start >= per_word.size:
        return float("nan")
    return float(np.nanmean(per_word[start:stop]))


def critical_region_target_gap(
    per_word_a: np.ndarray, d_a: int,
    per_word_b: np.ndarray, d_b: int,
    spillover: int,
) -> float:
    """Design B target statistic for one pair and one D metric."""
    va = critical_window_value(per_word_a, d_a, spillover)
    vb = critical_window_value(per_word_b, d_b, spillover)
    return float(abs(va - vb))


def pooled_window_null(
    per_word_a: np.ndarray, per_word_b: np.ndarray, spillover: int,
) -> np.ndarray:
    """Design B null statistic for one control pair, one D metric.

    Returns one value per ``(t1, t2)`` window-anchor tuple.
    """
    pa = np.asarray(per_word_a, dtype=float)
    pb = np.asarray(per_word_b, dtype=float)
    w = spillover + 1
    if pa.size < w or pb.size < w:
        return np.array([], dtype=float)
    out = []
    for t1 in range(pa.size - w + 1):
        va = float(np.nanmean(pa[t1:t1 + w]))
        for t2 in range(pb.size - w + 1):
            vb = float(np.nanmean(pb[t2:t2 + w]))
            out.append(abs(va - vb))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Pair-level extraction (used by per-set scripts)
# ---------------------------------------------------------------------------


@dataclass
class TargetPair:
    """User-supplied target pair specification."""

    name: str
    sentence_a: list[str]
    sentence_b: list[str]
    d_a: int
    d_b: int
    spillover: int = 2


def _per_word(metric_name: str, net, run: SentenceRun) -> np.ndarray:
    return D_METRICS[metric_name][1](net, run)


def _total(metric_name: str, net, run: SentenceRun) -> float:
    return D_METRICS[metric_name][0](net, run)


def extract_pair(net, sent_a: Sequence[str], sent_b: Sequence[str],
                 seed: int, cache: dict) -> dict:
    """Run both sentences and compute every D and J value (raw)."""
    run_a = run_sentence(net, sent_a, seed=seed, cache=cache)
    run_b = run_sentence(net, sent_b, seed=seed, cache=cache)
    out = {
        "sentence_a": sentence_key(sent_a),
        "sentence_b": sentence_key(sent_b),
        "n_words_a": run_a.n_words,
        "n_words_b": run_b.n_words,
        "D_total": {},
        "D_per_word_a": {},
        "D_per_word_b": {},
        "J": {},
    }
    for name in D_METRICS:
        out["D_total"][name + "_a"] = _total(name, net, run_a)
        out["D_total"][name + "_b"] = _total(name, net, run_b)
        out["D_per_word_a"][name] = _per_word(name, net, run_a).tolist()
        out["D_per_word_b"][name] = _per_word(name, net, run_b).tolist()
    for name, fn in J_METRICS.items():
        out["J"][name] = fn(net, run_a, run_b)
    return out


# ---------------------------------------------------------------------------
# Aggregation across many pairs
# ---------------------------------------------------------------------------


def aggregate_design_a(target_records: list[dict],
                       control_records: list[dict]) -> dict:
    """Build the Design A summary for every D metric.

    Each ``target_records[i]`` and ``control_records[i]`` is the dict returned
    by ``extract_pair``.
    """
    out = {}
    for name in D_METRICS:
        target_gaps = np.array([
            abs(r["D_total"][name + "_a"] - r["D_total"][name + "_b"])
            for r in target_records
        ], dtype=float)
        control_gaps = np.array([
            abs(r["D_total"][name + "_a"] - r["D_total"][name + "_b"])
            for r in control_records
        ], dtype=float)
        per_target = []
        for i, t in enumerate(target_gaps):
            stats = target_vs_control_stats(float(t), control_gaps)
            stats["pair_name"] = target_records[i].get("pair_name", f"target_{i}")
            per_target.append(stats)
        out[name] = {
            "per_target": per_target,
            "control_mean": float(np.nanmean(control_gaps))
            if control_gaps.size else float("nan"),
            "control_std": float(np.nanstd(control_gaps, ddof=1))
            if control_gaps.size > 1 else float("nan"),
            "n_control": int(control_gaps.size),
            "control_gaps": control_gaps.tolist(),
            "target_gaps": target_gaps.tolist(),
            "cohen_d_target_vs_control": cohen_d(target_gaps, control_gaps),
        }
    # Joint metrics: same idea but the metric is already a per-pair scalar.
    out_j = {}
    for name in J_METRICS:
        target_vals = np.array([r["J"][name] for r in target_records],
                               dtype=float)
        control_vals = np.array([r["J"][name] for r in control_records],
                                dtype=float)
        per_target = []
        for i, t in enumerate(target_vals):
            stats = target_vs_control_stats(float(t), control_vals)
            stats["pair_name"] = target_records[i].get("pair_name", f"target_{i}")
            per_target.append(stats)
        out_j[name] = {
            "per_target": per_target,
            "control_mean": float(np.nanmean(control_vals))
            if control_vals.size else float("nan"),
            "control_std": float(np.nanstd(control_vals, ddof=1))
            if control_vals.size > 1 else float("nan"),
            "n_control": int(control_vals.size),
            "control_values": control_vals.tolist(),
            "target_values": target_vals.tolist(),
            "cohen_d_target_vs_control": cohen_d(target_vals, control_vals),
        }
    return {"D": out, "J": out_j}


def aggregate_design_b(
    target_records: list[dict],
    control_records: list[dict],
    target_specs: list[TargetPair],
    control_spillover: int = 2,
) -> dict:
    """Build the Design B summary for every D metric.

    The control null is pooled over ``(control_pair, t1, t2)`` window anchors.
    A separate null is computed per ``spillover`` value: the per-target group
    is matched to the null sharing the same ``spillover``.
    """
    spillovers_target = sorted({sp.spillover for sp in target_specs})
    out = {}
    for name in D_METRICS:
        per_target_all = []
        nulls_by_spillover = {}
        for sp in spillovers_target:
            ctrl_pool = []
            for r in control_records:
                pa = np.asarray(r["D_per_word_a"][name], dtype=float)
                pb = np.asarray(r["D_per_word_b"][name], dtype=float)
                ctrl_pool.append(pooled_window_null(pa, pb, sp))
            ctrl_null = (np.concatenate(ctrl_pool)
                         if ctrl_pool else np.array([], dtype=float))
            nulls_by_spillover[sp] = ctrl_null
        for i, spec in enumerate(target_specs):
            r = target_records[i]
            pa = np.asarray(r["D_per_word_a"][name], dtype=float)
            pb = np.asarray(r["D_per_word_b"][name], dtype=float)
            tg = critical_region_target_gap(pa, spec.d_a, pb, spec.d_b,
                                            spec.spillover)
            stats = target_vs_control_stats(
                tg, nulls_by_spillover[spec.spillover])
            stats["pair_name"] = spec.name
            stats["spillover"] = spec.spillover
            stats["window_a"] = [spec.d_a, spec.d_a + spec.spillover]
            stats["window_b"] = [spec.d_b, spec.d_b + spec.spillover]
            per_target_all.append(stats)
        out[name] = {
            "per_target": per_target_all,
            "null_size_by_spillover": {
                str(sp): int(arr.size) for sp, arr in nulls_by_spillover.items()
            },
            "null_mean_by_spillover": {
                str(sp): (float(np.nanmean(arr)) if arr.size else float("nan"))
                for sp, arr in nulls_by_spillover.items()
            },
            "null_std_by_spillover": {
                str(sp): (float(np.nanstd(arr, ddof=1))
                          if arr.size > 1 else float("nan"))
                for sp, arr in nulls_by_spillover.items()
            },
        }
    return {"D": out}


# ---------------------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------------------


def shared_prefix_check(target_records: list[dict],
                         metric_names: Iterable[str] = ("D_H", "D_path"),
                         tol: float = 1e-6) -> dict:
    """Verify that for any target pair sharing a leading sub-prefix, the
    per-word metric values for those positions agree exactly (up to tol).

    Non-zero pre-edit gaps flag a determinism bug.
    """
    out = {}
    for r in target_records:
        a = r["sentence_a"].split()
        b = r["sentence_b"].split()
        prefix_len = 0
        for tok_a, tok_b in zip(a, b):
            if tok_a == tok_b:
                prefix_len += 1
            else:
                break
        if prefix_len == 0:
            continue
        info = {"shared_prefix_len": prefix_len, "violations": {}}
        for m in metric_names:
            pa = np.asarray(r["D_per_word_a"][m])
            pb = np.asarray(r["D_per_word_b"][m])
            if min(pa.size, pb.size) < prefix_len:
                continue
            diffs = np.abs(pa[:prefix_len] - pb[:prefix_len])
            if np.any(diffs > tol):
                info["violations"][m] = diffs.tolist()
        out[r.get("pair_name", "?")] = info
    return out


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)


def read_json(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# High-level per-set driver
# ---------------------------------------------------------------------------


def run_per_set_analysis(
    set_name: str,
    model_path: str,
    target_pairs: list[TargetPair],
    n_control: int = 100,
    control_seed: int = 0,
    run_seed: int = 1024,
    control_source: str = "corpus",
    out_dir: Optional[str] = None,
    fig_dir: Optional[str] = None,
    make_plots: bool = True,
) -> dict:
    """End-to-end per-set analysis.

    Steps:
        1. Load model.
        2. Run each target sentence (deduplicated).
        3. Sample ``n_control`` control pairs and run each.
        4. Compute Design A and Design B summaries.
        5. Write ``results/<set>_results.json`` and figures (if requested).
    """
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "results")
    fig_dir = fig_dir or os.path.join(os.path.dirname(__file__), "figs",
                                       set_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[{set_name}] loading model from {model_path}")
    net = load_net(model_path)
    cache: dict = {}

    target_keys = {sentence_key(sp.sentence_a) for sp in target_pairs}
    target_keys.update(sentence_key(sp.sentence_b) for sp in target_pairs)

    print(f"[{set_name}] running {len(target_pairs)} target pair(s)")
    target_records = []
    for sp in target_pairs:
        rec = extract_pair(net, sp.sentence_a, sp.sentence_b,
                           seed=run_seed, cache=cache)
        rec["pair_name"] = sp.name
        rec["d_a"] = sp.d_a
        rec["d_b"] = sp.d_b
        rec["spillover"] = sp.spillover
        rec["is_target"] = True
        target_records.append(rec)

    print(f"[{set_name}] sampling {n_control} control pairs (source={control_source})")
    control_pairs = sample_control_pairs(
        net, n_control, exclude_sentences=target_keys,
        source=control_source, seed=control_seed,
    )
    print(f"[{set_name}] running {len(control_pairs)} control pairs")
    control_records = []
    for ci, (a, b) in enumerate(control_pairs):
        rec = extract_pair(net, a, b, seed=run_seed, cache=cache)
        rec["pair_name"] = f"ctrl_{ci:03d}"
        rec["is_target"] = False
        control_records.append(rec)

    print(f"[{set_name}] aggregating Design A")
    design_a = aggregate_design_a(target_records, control_records)
    print(f"[{set_name}] aggregating Design B")
    design_b = aggregate_design_b(target_records, control_records,
                                  target_pairs)

    sanity = shared_prefix_check(target_records)

    payload = {
        "set_name": set_name,
        "model_path": model_path,
        "n_control_requested": n_control,
        "n_control_drawn": len(control_pairs),
        "control_source": control_source,
        "run_seed": run_seed,
        "control_seed": control_seed,
        "target_pairs": [
            {"name": sp.name,
             "sentence_a": sentence_key(sp.sentence_a),
             "sentence_b": sentence_key(sp.sentence_b),
             "d_a": sp.d_a, "d_b": sp.d_b, "spillover": sp.spillover}
            for sp in target_pairs
        ],
        "design_a": design_a,
        "design_b": design_b,
        "sanity_shared_prefix": sanity,
        "raw_target_records": target_records,
        "raw_control_records": control_records,
    }

    out_path = os.path.join(out_dir, f"{set_name}_results.json")
    write_json(out_path, payload)
    print(f"[{set_name}] wrote {out_path}")

    if make_plots:
        try:
            _render_per_set_figures(set_name, payload, fig_dir, target_pairs)
        except Exception as exc:
            print(f"[{set_name}] plot rendering failed: {exc}")

    return payload


def _render_per_set_figures(set_name: str, payload: dict, fig_dir: str,
                            target_pairs: list[TargetPair]) -> None:
    """Render the per-set Design A histograms and Design B per-word plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    # --- Design A histograms -----------------------------------------------
    da = payload["design_a"]["D"]
    metric_names = list(D_METRICS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, name in zip(axes, metric_names):
        controls = np.asarray(da[name]["control_gaps"], dtype=float)
        ax.hist(controls, bins=20, color="#888", alpha=0.7,
                label=f"control (n={controls.size})")
        for stats in da[name]["per_target"]:
            ax.axvline(stats["target"], color="red", linestyle="--",
                       label=f"{stats['pair_name']} z={stats['z']:.2f}")
        ax.set_title(f"Design A: {name}")
        ax.set_xlabel("|D_total(A) - D_total(B)|")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
    fig.suptitle(f"{set_name} | Design A: whole-sequence gaps")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{set_name}_design_a.png"), dpi=150)
    plt.close(fig)

    # --- Design A J-metrics ------------------------------------------------
    dj = payload["design_a"]["J"]
    j_names = list(J_METRICS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, name in zip(axes, j_names):
        controls = np.asarray(dj[name]["control_values"], dtype=float)
        ax.hist(controls, bins=20, color="#558", alpha=0.7,
                label=f"control (n={controls.size})")
        for stats in dj[name]["per_target"]:
            ax.axvline(stats["target"], color="red", linestyle="--",
                       label=f"{stats['pair_name']} z={stats['z']:.2f}")
        ax.set_title(f"Design A (J): {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
    fig.suptitle(f"{set_name} | joint divergence (J) per pair")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{set_name}_design_a_J.png"), dpi=150)
    plt.close(fig)

    # --- Design B per-word envelope plots ---------------------------------
    target_records = payload["raw_target_records"]
    control_records = payload["raw_control_records"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, name in zip(axes, metric_names):
        # Build envelope from per-pair per-word |D_a - D_b| (length-aligned to
        # min of the two sentences). For unequal-length pairs we just plot the
        # mean over the shorter one.
        ctrl_curves = []
        for r in control_records:
            pa = np.asarray(r["D_per_word_a"][name], dtype=float)
            pb = np.asarray(r["D_per_word_b"][name], dtype=float)
            n = min(pa.size, pb.size)
            if n == 0:
                continue
            ctrl_curves.append(np.abs(pa[:n] - pb[:n]))
        max_len = max((len(c) for c in ctrl_curves), default=0)
        if max_len > 0:
            padded = np.full((len(ctrl_curves), max_len), np.nan)
            for i, c in enumerate(ctrl_curves):
                padded[i, :len(c)] = c
            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            xs = np.arange(1, max_len + 1)
            ax.plot(xs, mean_curve, color="#888", label="control mean")
            ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                            color="#888", alpha=0.3, label="+/-1 SD")
        for i, sp in enumerate(target_pairs):
            r = target_records[i]
            pa = np.asarray(r["D_per_word_a"][name], dtype=float)
            pb = np.asarray(r["D_per_word_b"][name], dtype=float)
            n = min(pa.size, pb.size)
            if n == 0:
                continue
            xs = np.arange(1, n + 1)
            ax.plot(xs, np.abs(pa[:n] - pb[:n]), label=sp.name)
            window_a = (sp.d_a, sp.d_a + sp.spillover)
            ax.axvspan(window_a[0] - 0.4, window_a[1] + 0.4, color="red",
                       alpha=0.10)
        ax.set_title(f"Design B: per-word |D_a - D_b| | {name}")
        ax.set_xlabel("word position")
        ax.set_ylabel(name)
        ax.legend(fontsize=7)
    fig.suptitle(f"{set_name} | Design B: per-word envelope (red shading = "
                 "target window for sentence A)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{set_name}_design_b.png"), dpi=150)
    plt.close(fig)

    # --- Sanity check: shared-prefix gap should be ~0 ---------------------
    sanity = payload.get("sanity_shared_prefix", {})
    if sanity:
        fig, ax = plt.subplots(figsize=(8, 5))
        for pair_name, info in sanity.items():
            for m, diffs in info.get("violations", {}).items():
                xs = np.arange(1, len(diffs) + 1)
                ax.plot(xs, diffs, label=f"{pair_name}/{m}")
        ax.set_title(f"{set_name} | shared-prefix sanity: violations")
        ax.set_xlabel("word position (within shared prefix)")
        ax.set_ylabel("|D_a - D_b|")
        if ax.has_data():
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no shared-prefix violations",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{set_name}_sanity.png"), dpi=150)
        plt.close(fig)

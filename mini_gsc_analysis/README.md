# mini_gsc_analysis

Quantitative probing framework for trained GscNet models.

Tests **Hypothesis 1**: parsing-difficulty gap inside the target critical pair
(e.g. `DT NN VBZ DT NN` vs `DT NNS VBZ DT NN` for agreement) is significantly
larger than within random control pairs.

The framework also captures joint divergence metrics `J(A, B)` for the
**parallel hypothesis** (are the two parses different?) and saves the raw
per-pair table so a future correlation analysis (Hypothesis 2) can be done
without re-running the simulations.

## Layout

```
mini_gsc_analysis/
    gsc_inference_utils.py         shared core (loader, run_sentence,
                                   D / J extractors, sampling, stats, plots)
    probe_agreement.py             one script per phenomenon
    probe_classicgp_mvpr.py
    probe_classicgp_nps.py
    probe_classicgp_npz.py
    probe_high_low_attachment.py
    probe_relative_clause.py
    probe_cross_set.py             aggregate per-set results
    probe_correlation.py           optional: J vs |D_gap| scatter analysis
    results/                       <set>_results.json + cross_set_summary.json
    figs/<set>/                    Design-A / Design-B / sanity plots
    figs/cross_set/                cross-set heatmaps and bar charts
    figs/correlation/              optional scatters
```

## Per-sentence difficulty metrics (D)

* `D_surp`: per-word surprisal, derived from end-of-word `actCmat[level=1, position=w]`.
* `D_H`: harmony deficit (`-H` per word, normalised by `num_bindings`).
* `D_settle`: log-mean of `||grad H||` within each `run_word`.
* `D_ent`: mean per-role entropy over the per-step `actCmat` distributions.
* `D_burst`: count of high-velocity bursts in `actC` per word.
* `D_path`: integrated state-space path length per word, normalised by
  `sqrt(num_bindings)`.

## Joint divergence metrics (J)

* `J_final_cosine`: `1 - cos(actC_A_final, actC_B_final)`
* `J_final_l2`: L2 distance, normalised by `sqrt(num_bindings)`
* `J_mean_role_kl`: mean KL between per-role filler distributions
* `J_argmax_mismatch`: fraction of roles where final argmax differs
* `J_logp_gap`: `|D_surp_total(A) - D_surp_total(B)|`
* `J_peak_perword_gap`: peak per-word L2 (truncated to shorter sentence)

## Test designs

* **Design A** -- whole-sequence: `G_target = |D_total(A) - D_total(B)|`
  vs control gaps `|D_total(s_i) - D_total(s_j)|`.
* **Design B** -- critical-region: target = `|mean_window_A D_a - mean_window_B D_b|`,
  using each sentence's own window `[d_s, d_s + spillover]`. Null is pooled
  over `(control_pair, t1, t2)` independent window-anchor tuples and is
  computed separately per `spillover` value (attachment uses spillover=1, all
  others use spillover=2).

## Running on the cluster

The trained `.pkl` checkpoints live under `SAP_analysis/`. Each per-set
script defaults to the file produced by the corresponding `train_*.py`
script. Override with `--model-path /path/to/model.pkl` if needed.

```sh
# 1. Run each phenomenon individually (independent; can be parallelised).
python -m mini_gsc_analysis.probe_agreement
python -m mini_gsc_analysis.probe_classicgp_mvpr
python -m mini_gsc_analysis.probe_classicgp_nps
python -m mini_gsc_analysis.probe_classicgp_npz
python -m mini_gsc_analysis.probe_high_low_attachment
python -m mini_gsc_analysis.probe_relative_clause

# 2. Aggregate across sets.
python -m mini_gsc_analysis.probe_cross_set

# 3. Optional: complementary correlation analysis.
python -m mini_gsc_analysis.probe_correlation
```

Each per-set script accepts:

```
--model-path PATH         override default checkpoint path
--n-control N             number of control pairs (default 100)
--control-source corpus   sample from net.corpus (default) or random_pos
--control-seed N          RNG seed for control sampling (default 0)
--run-seed N              RNG seed used for every sentence run (default 1024)
--no-plots                skip per-set figure rendering
```

## Caching, reproducibility, sanity checks

* Each per-set script runs every unique sentence at most once thanks to a
  cache keyed by `(sentence_string, seed)`.
* All sentence runs in a single per-set call use the same `--run-seed`, which
  is then sequenced into per-sentence numpy state via `np.random.seed`. This
  gives shared-seed determinism: for any target pair sharing a leading
  sub-prefix (e.g. agreement positions 1 = `DT`, both sentences) the per-word
  `D` values must agree exactly. The framework reports any violations under
  `sanity_shared_prefix` in the JSON output and renders them as a separate
  plot.
* All metrics that involve dividing by `num_bindings` / `MAXLEN` are
  pre-normalised in `gsc_inference_utils.py`, so cross-set rankings (z-scores
  in particular) are unitless and directly comparable.

## Output JSON schema (per set)

```jsonc
{
  "set_name": "agreement",
  "model_path": "...",
  "target_pairs": [{"name": ..., "sentence_a": ..., "sentence_b": ...,
                    "d_a": 3, "d_b": 3, "spillover": 2}],
  "design_a": {
    "D": {"D_surp": {"per_target": [...], "control_mean": ..., ...}, ...},
    "J": {...}
  },
  "design_b": {
    "D": {"D_surp": {"per_target": [...], "null_size_by_spillover": {...}, ...}, ...}
  },
  "sanity_shared_prefix": {...},
  "raw_target_records": [...],   // preserved for correlation analysis
  "raw_control_records": [...]
}
```

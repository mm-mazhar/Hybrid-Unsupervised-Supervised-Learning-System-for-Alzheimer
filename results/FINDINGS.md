# Validation findings

Produced by `research/f_validation.ipynb`. Every number below is read from the CSVs in this
directory; none is typed by hand. This document supersedes the performance claims in notebooks
`c_clustering` through `hybrid_inference_system_b` and in the manuscript as submitted.

---

## Bottom line

**The cluster-based ("persona") machinery does not improve prediction over a plain linear model.**

On the 746-row external holdout, evaluated once, with every model given identical features,
identical rows and person-level splits:

| Rank | Model | MAE | 95% CI | R² |
| ---: | :--- | ---: | :--- | ---: |
| 1 | Lasso | **32.73** | 30.96 – 34.53 | 0.544 |
| 2 | ElasticNet | 32.75 | 30.98 – 34.52 | 0.543 |
| 3 | Ridge | 32.77 | 31.01 – 34.57 | 0.542 |
| 4 | HistGradientBoosting | 33.28 | 31.44 – 34.97 | 0.537 |
| 5 | Random forest (global, no clusters) | 33.77 | 31.93 – 35.59 | 0.517 |
| 6 | Strategy C (oversampled) | 33.88 | 32.04 – 35.76 | 0.509 |
| 7 | Strategy B (cluster feature) | 33.91 | 32.09 – 35.72 | 0.515 |
| 8 | Hybrid router | 34.18 | 32.32 – 35.95 | 0.508 |
| 9 | Strategy A (per-cluster specialists) | 34.59 | 32.74 – 36.36 | 0.498 |
| 10 | Mean baseline | 49.89 | 47.45 – 52.40 | -0.000 |

Paired bootstrap, each cluster-aware model against Lasso, MAE difference with CI excluding zero in
every case (`17_holdout_contrasts.csv`):

- Strategy A: **+1.86** worse (0.73 – 2.97)
- Hybrid router: **+1.45** worse (0.42 – 2.45)
- Strategy B: **+1.19** worse (0.26 – 2.16)
- Strategy C: **+1.16** worse (0.15 – 2.17)

Repeated grouped cross-validation (5 folds × 3 repeats, 15 folds) agrees: the best model is
HistGradientBoosting at MAE 32.23, Lasso is 32.93, and the cluster-aware arms sit at or below the
conventional benchmarks. Strategy B is the only one statistically indistinguishable from Lasso
(diff -0.01, CI -0.28 – 0.27); the rest are worse with CIs excluding zero.

The direction is consistent across both evaluations. This is not a marginal call.

---

## What was rebuilt, and why the old numbers cannot be used

Two leaks upstream of the ones the reviewer raised:

1. **K-Means, the scaler and the one-hot encoding were fitted on all 2,889 training rows.** Every
   row that later served as "test" in strategies A–D had already helped define its own cluster.
   `kmeans.predict()` appears nowhere in the original code, and the fitted objects were never saved.
2. **647 individuals contribute two rows each** (a 2016 and a 2021 outcome) with byte-identical
   predictors once `Year`/`PredictionYear`/`UID` are dropped. Splits were row-level, so roughly 36%
   of test rows had their own twin sitting in training.

The new pipeline fits preprocessing, scaling, encoding and K-Means on training rows only, splits on
`UID` throughout (`GroupKFold`/`GroupShuffleSplit`), restores `PredictionYear` as a feature so the
two rows per person are no longer identical, and asserts loudly if a `UID` ever spans a fold.

`test.csv` is person-disjoint from `train.csv` (0 shared UIDs) and was touched exactly once,
in section 7.

| Partition | Rows | Persons | With 2 rows | Target mean (SD) |
| :--- | ---: | ---: | ---: | :--- |
| train.csv (development) | 2,889 | 2,242 | 647 | 146.1 (59.1) |
| test.csv (external holdout) | 746 | 570 | 176 | 146.3 (61.1) |

---

## The reviewer's points, answered

### 1. "No accuracy, discrimination or calibration; no CV scheme described; no conventional benchmark"

Now all present, and all in `results/`.

- Partitioning scheme documented in `01_partition_summary.csv` and section 1.
- Repeated grouped CV: `09_cv_folds.csv` (150 rows), `10_cv_summary.csv` (mean ± SD over 15 folds).
- Conventional benchmarks: Ridge, Lasso, ElasticNet, all alpha-tuned inside each training fold.
- Bootstrap 95% CIs on every headline number; paired bootstrap for every contrast that carries an
  argument.
- Discrimination and calibration, on a pre-specified dichotomisation at the training-set lower
  quartile (`14_binary_cv_summary.csv`):

| Model | ROC-AUC | PR-AUC | Brier | Sens. | Spec. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Logistic regression** | **0.833** | 0.633 | **0.139** | 0.387 | 0.941 |
| Random forest, cluster-aware | 0.831 | 0.631 | 0.148 | 0.599 | 0.864 |
| Random forest, global | 0.831 | 0.629 | 0.148 | 0.607 | 0.859 |
| Prior baseline | 0.500 | 0.257 | 0.191 | 0.000 | 1.000 |

Logistic regression is best on both discrimination and calibration. The reviewer named it
specifically; it wins.

### 2. "Oversampling leakage"

**Did not occur.** `d_strategyC.ipynb` splits at cell 6 and calls `resample()` on training rows only
at cell 8. This was a documentation failure, not a methods failure — but see the two real leaks
above, which are worse and which the reviewer did not catch.

### 3. "Findings are well-replicated gerontology; that the method reveals hidden structure must be demonstrated"

Demonstrated, and it does not. See the bottom-line table. The matched comparison the manuscript
asserts but never ran now exists, and the cluster machinery loses it.

### 4. "No external validation; no evidence clusters are stable; no evidence assignment improves anything"

External validation now exists (section 7). Stability was measured three ways
(`07_stability_summary.csv`):

| Check | Adjusted Rand Index |
| :--- | :--- |
| Across 20 random seeds, pairwise | 0.488 ± 0.175 |
| Person-level bootstrap, 100 resamples | 0.468 ± 0.131 |
| **2003 wave vs 2012 wave features** | **0.078** |

An ARI near 0.5 across seeds means roughly half the partition structure is arbitrary. An ARI of
0.078 across measurement waves means the partition is essentially unrelated to itself when computed
from a different wave of the same people — the specific thing the reviewer asked about.

The k-selection diagnostics say the same (`02_k_selection.csv`): silhouette never exceeds **0.055**
at any k from 2 to 10, and is **0.033** at the k=6 used throughout the project. For k ≥ 3 the
smallest cluster is always a single row. There is no strong cluster structure in this feature space
to find.

One thing did come out clean: the saved K-Means reproduces the training cluster proportions on the
holdout to within 0.8 percentage points (`03_cluster_shares.csv`). The 69%-collapse-to-cluster-0
seen in `hybrid_inference_system_b.ipynb` was a bug in that notebook's unscaled nearest-centroid
router, not a real distribution shift.

---

## Claims that must be dropped

| Claim as published | Status |
| :--- | :--- |
| "The Hybrid System significantly outperformed the baseline Global Model" | **False on a matched comparison.** Hybrid 34.18 vs Lasso 32.73; the hybrid is worse, CI excludes zero. |
| Hybrid MAE 34.76 beats AutoGluon 35.19 | **Not a comparison.** Different feature sets (157 raw vs 67 engineered), different treatment of `Year`, different rows (746 holdout vs a 578-row split of train). Recorded in `19_historical_numbers.csv` as non-comparable. |
| Cluster 1 (wealthy): Strategy B reduced error ~7 points | **Reversed.** On the holdout the hybrid loses to the linear model by 5.9 MAE in that cluster (`18_holdout_per_cluster.csv`). |
| Per-cluster R² ≈ 0.59–0.60 as evidence of quality | **Invalid comparison.** Within-cluster R² cannot be compared to whole-sample R²; restricting to a cluster shrinks outcome variance. Report within-cluster MAE instead. |
| Triage classifier's 93% "validates that phenotypes are distinct" | **Circular.** It predicts K-Means labels from the same 158 features K-Means clustered on. Reproduced here at 88.6%. Against a 34.8% majority-class baseline that is real learnability of a decision boundary, not evidence the phenotypes exist. |
| Phenotypes "identifiable using basic demographic questions" | **False.** Restricted to the 15 genuinely intake-observable features (age, gender, education, marital status, self-rated health, urbanicity), accuracy drops to **59.3%**, balanced accuracy **0.434** (`15_triage_reframed.csv`). The intake-triage story does not hold. |
| "Immediate clinical utility" via persona assignment at intake | **Unsupported**, and now contradicted by the two rows above. |
| `c_clustering.ipynb` performs PCA | **Does not exist in the code.** Already corrected in the README. |
| Triage lives in `g_triage_classifier.ipynb` | **No such file.** It is `d_strategyD.ipynb`. Already corrected. |

---

## What survives

Not nothing:

- A leakage-safe, person-level, fully specified pipeline with a benchmark ladder and uncertainty on
  every number — which is what the reviewer said was missing.
- A well-calibrated logistic model for the dichotomised outcome: AUC 0.833, Brier 0.139, beating
  both random forest variants on calibration.
- A clean negative result on heterogeneity-aware modelling in this cohort, with the stability
  statistics that explain *why* it fails (there is no stable partition to route on).
- A documented, reproducible correction of a router bug that was silently sending 69% of the
  holdout to one cluster.

---

## Options

**A. Reframe and resubmit as a negative/methodological result.** "Does heterogeneity-aware
modelling add predictive value over conventional baselines in MHAS?" Answer: no, and here is the
cluster-instability evidence for why. Honest, defensible, and the evidence is already in `results/`.
The reviewer's "requires substantial work on its exposition" still applies — this is a rewrite, not
a revision.

**B. Pivot to the within-person design.** The 647 duplicate-UID pairs (2016 vs 2021 outcome for the
same person) are a horizon/change design currently being treated as a nuisance, and the unused
`delta_*` / `changed_*` features in `d_featEng.ipynb` are exactly the cross-wave signal a
gerontology reviewer would find novel. Different paper, plausibly a stronger one.

**C. Withdraw.** The original decision criterion was "withdraw if the matched comparison shows the
hybrid does not beat Ridge." It doesn't. If the paper cannot be reframed away from the persona
claim, this is the honest call.

The persona framing itself is not recoverable. Options A and B both require abandoning it.

---

## Reproducing

```bash
uv venv --python=3.11 .venv && uv sync
uv run jupyter nbconvert --to notebook --execute research/f_validation.ipynb --inplace
```

Helper code lives in `research/utils/validation.py`. Fitted models are written to
`models/validated/` (gitignored). Figures are in `visualization/f_validation/`.

| File | Contents |
| :--- | :--- |
| `01_partition_summary.csv` | Row/person counts and target distribution per partition |
| `02_k_selection.csv` | Inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, k = 2–10 |
| `03_cluster_shares.csv` | Train vs holdout cluster proportions |
| `04`–`07_stability_*.csv` | Seed, bootstrap and cross-wave ARI |
| `08_tuning.csv` | Selected hyperparameters |
| `09`–`12_cv_*.csv` | Fold-level results, summary, paired contrasts, per-cluster |
| `13`–`14_binary_*.csv` | Discrimination and calibration |
| `15_triage_reframed.csv` | Triage accuracy by feature set |
| `16`–`18_holdout_*.csv` | External holdout results, contrasts, per-cluster |
| `19_historical_numbers.csv` | Previously published figures, annotated as non-comparable |
| `20_findings.csv` | Machine-readable version of this document's headline answers |

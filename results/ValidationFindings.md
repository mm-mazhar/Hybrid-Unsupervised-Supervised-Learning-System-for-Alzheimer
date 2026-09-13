# MHAS cognitive-health manuscript — validation findings (Sep 2026)

Repo: `C:\Users\maz\dev\Projects_\alzheimer`, branch `validation/reviewer-rebuttal`, commit `2aced9c`.
Full evidence: `results/` in that repo (20 CSVs + `FINDINGS.md` + a script that re-checks every
documented figure against the CSVs).

## Context

A journal reviewer rejected the manuscript (K-Means "personas" + per-cluster Random Forests on the
MHAS cohort) for reporting no model performance, no partitioning scheme, and no conventional
benchmark; for possible oversampling leakage; for findings that are already well-replicated
gerontology; and for unsupported claims of clinical utility.

`research/f_validation.ipynb` rebuilds the pipeline leakage-free and runs the matched comparison the
manuscript asserted but never performed.

## Bottom line

**The persona/hybrid contribution does not hold up.** On the 746-row person-disjoint holdout,
evaluated once, every model on identical rows and features:

| Model | MAE | 95% CI | R² |
| :--- | ---: | :--- | ---: |
| Lasso | **32.73** | 30.96–34.53 | 0.544 |
| ElasticNet | 32.75 | 30.98–34.52 | 0.543 |
| Ridge | 32.77 | 31.01–34.57 | 0.542 |
| HistGradientBoosting | 33.28 | 31.44–34.97 | 0.537 |
| Random forest (global) | 33.77 | 31.93–35.59 | 0.517 |
| Strategy C (oversampled) | 33.88 | 32.04–35.76 | 0.509 |
| Strategy B (cluster feature) | 33.91 | 32.09–35.72 | 0.515 |
| Hybrid router | 34.18 | 32.32–35.95 | 0.508 |
| Strategy A (specialists) | 34.59 | 32.74–36.36 | 0.498 |
| Mean baseline | 49.89 | 47.45–52.40 | -0.000 |

Paired bootstrap vs Lasso: Strategy A +1.86 (CI 0.73–2.97), hybrid +1.45 (0.42–2.45), Strategy B
+1.19 (0.26–2.16), Strategy C +1.16 (0.15–2.17). Every CI excludes zero. Repeated grouped CV
(15 folds) gives the same ordering.

## Why it fails: the clusters are not a stable partition

| Check | Adjusted Rand Index |
| :--- | :--- |
| Across 20 seeds, pairwise | 0.488 ± 0.175 |
| Person-level bootstrap (100) | 0.468 ± 0.131 |
| 2003 wave vs 2012 wave features | **0.078** |

Silhouette never exceeds 0.055 for any k in 2–10, and is 0.033 at the k=6 used throughout. For
k ≥ 3 the smallest cluster is always a single row. There is no strong structure in this feature
space to route on.

## Leaks found (two the reviewer missed)

1. K-Means, the scaler and the one-hot encoding were fitted on all 2,889 training rows — every
   "test" row had helped define its own cluster. `kmeans.predict()` existed nowhere; the fitted
   objects were never saved.
2. 647 individuals contribute two rows each (2016 + 2021 outcome) with identical predictors, and
   splits were row-level, so ~36% of test rows had their twin in training.

Also: the hybrid router used an unscaled nearest-centroid rather than the fitted K-Means, sending
69% of the holdout to one cluster. Fixed — the saved K-Means reproduces training cluster shares on
the holdout to within 0.8pp.

The oversampling the reviewer flagged was **clean**: `d_strategyC` split before resampling.

## Claims that must be dropped

- "Hybrid significantly outperformed the global baseline" — false on a matched comparison.
- "Hybrid 34.76 beats AutoGluon 35.19" — not a comparison (different features, rows, and handling
  of `Year`).
- "Cluster 1 (wealthy): Strategy B reduced error ~7 points" — reversed; the hybrid loses to the
  linear model by 5.9 MAE there.
- Per-cluster R² ≈ 0.59 as evidence of quality — within-cluster R² is not comparable to
  whole-sample R².
- Triage 93% "validates the phenotypes" — circular (predicts K-Means labels from the same 158
  features). Reproduced at 88.6% vs a 34.8% majority baseline.
- Phenotypes "identifiable from basic demographic questions" — restricted to the 15 genuinely
  intake-observable features, accuracy falls to 59.3% (balanced accuracy 0.434).
- "Immediate clinical utility" via intake persona assignment — unsupported.

## What survives

- A leakage-safe, person-level pipeline with a benchmark ladder and uncertainty on every number —
  exactly what the reviewer said was missing.
- A well-calibrated logistic model for the dichotomised outcome: AUC 0.833, Brier 0.139, beating
  both random-forest variants on calibration.
- A clean negative result with the stability statistics that explain why heterogeneity-aware
  modelling fails here.

## Options (undecided as of this writing)

- **A. Reframe as a negative/methodological result** — "does heterogeneity-aware modelling add
  predictive value over conventional baselines in MHAS?" Answer: no, with the instability evidence
  for why. Requires a rewrite, not a revision.
- **B. Pivot to the within-person design** — the 647 duplicate-UID pairs (2016 vs 2021 outcome for
  the same person) are a horizon/change design currently treated as a nuisance; the unused
  `delta_*` / `changed_*` features in `d_featEng.ipynb` are the cross-wave signal a gerontology
  reviewer would find novel.
- **C. Withdraw** — the pre-agreed criterion was "withdraw if the hybrid does not beat Ridge." It
  doesn't.

The persona framing is not recoverable; A and B both require abandoning it.
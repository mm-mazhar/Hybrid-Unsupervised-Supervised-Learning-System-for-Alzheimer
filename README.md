# Cognitive Health Prediction: A Hybrid Unsupervised & Supervised Learning System For Alzheimer's Disease

## 📌 Project Overview
This research project aims to predict **Cognitive Health Scores** (`composite_score`) in an aging population. 

Traditional regression models often fail to capture the distinct etiology of cognitive decline across different socioeconomic and health profiles. Instead of a "one-size-fits-all" approach, this project implements a **Hybrid Machine Learning System** that:
1.  **Discovers Phenotypes:** Uses Unsupervised Learning (K-Means) to identify distinct patient personas (e.g., "The Frail", "The Ultra-Wealthy", "The Professionals").
2.  **Adaptive Modeling:** Routes patients to specific predictive models optimized for their phenotype.
3.  **Explainability:** Uses SHAP to inspect the drivers of prediction within each group.

> **Status.** A leakage-safe re-evaluation of this pipeline
> (`research/f_validation.ipynb`, write-up in [`results/FINDINGS.md`](results/FINDINGS.md))
> found that the hybrid persona system does **not** outperform a plain linear model on a matched
> comparison, and that the discovered clusters are not stable across seeds, resamples, or
> measurement waves. Quote figures from `results/`, not from notebooks 1–7.

---

## 📂 Repository Structure & Workflow

The analysis is divided into sequential notebooks, each representing a distinct phase of the research pipeline.

| Sequence | File Name | Description |
| :--- | :--- | :--- |
| **1. Unsupervised** | `research/c_clustering.ipynb` | **Phenotype Discovery.** Performs Data Cleaning, Scaling/One-Hot Encoding, and K-Means Clustering. Identifies 5 core clusters and 1 outlier group. |
| **2. Strategy A** (Predictive Modeling) | `research/d_strategyA.ipynb` | **Specialist Models.** Trains separate Random Forest regressors for each cluster ("Divide and Conquer"). |
| **3. Strategy B (Global Meta Feature)** | `research/d_strategyB.ipynb` | **Global Meta-Model.** Trains one global model with `cluster_id` as a feature. (Best for small groups like "The Ultra-Wealthy"). |
| **4. Strategy C (Cluster Over Sampling)** | `research/d_strategyC.ipynb` | **Oversampling.** Uses resampling techniques to boost the signal of minority groups. (Best for "The Frail & Vulnerable"). |
| **5. Strtegy D (Validation - Triage Classifier)** | `research/d_strategyD.ipynb` | **Triage System.** Trains a Classifier to predict the Cluster ID based on raw data. Validates that phenotypes are distinct (93% Accuracy). |
| **6. Deployment** | `research/hybrid_inference_system_a.ipynb` and `research/hybrid_inference_system_b.ipynb` | **Final Inference Engine.** The system which Routes new patients to the best model (A, B, or C) based on their profile. |
| **7. Strategy E (XAI)** | `research/d_strategyE.ipynb` | **Explainability.** Uses SHAP values to prove that drivers of health (e.g., Income vs. ADLs) differ fundamentally across clusters. |
| **8. Validation & Benchmarking** | `research/f_validation.ipynb` | **Leakage-safe re-evaluation.** Person-level (`UID`) splits, clustering fitted on training rows only, a conventional benchmark ladder (Ridge/Lasso/ElasticNet/logistic regression), repeated grouped cross-validation, cluster stability statistics, and one matched evaluation on the held-out `test.csv`. **Supersedes the performance figures reported by notebooks 1–7.** Results are written to `results/`. |

> **Note on notebooks 1–7.** They are retained for provenance, but their reported
> metrics should not be quoted. They fit K-Means, the scaler and the one-hot
> encoding on all available rows, and they split at row level even though 647
> individuals contribute two rows each with identical predictors. Use
> `research/f_validation.ipynb` and `results/` for any performance claim.

---

## 🧠 Methodology: The Hybrid System

> **Superseded.** The design below is the system as originally built. A leakage-safe
> re-evaluation (`research/f_validation.ipynb`) found that it does not improve prediction
> over a plain linear model, and that the clusters it routes on are not a stable partition
> of the data. The description is kept for provenance; see **Key Results** for the
> figures that hold up.

No single modeling strategy was best for every subgroup in the original experiments, so a
**Hybrid Router** was built to select an approach per patient:

### The 5 Phenotypes (Clusters)
*   **Cluster 0 (The Frail & Vulnerable):** High physical impairment, high depression.
*   **Cluster 1 (The Ultra-Wealthy):** Extremely high capital income, small sample size ($N=57$).
*   **Cluster 2 (Working Middle Class):** Average income, currently working.
*   **Cluster 3 (Non-Working Middle Class):** Relies on spousal income, homemakers.
*   **Cluster 5 (High-Earning Professionals):** High education, high salary, very healthy.

### The Routing Logic
| Patient Profile | Strategy Used | Original justification (not reproduced under a matched comparison) |
| :--- | :--- | :--- |
| **Cluster 0 (Frail)** | **Strategy C (Oversampling)** | Oversampling was reported to reduce MAE by ~4.5 points. |
| **Cluster 1 (Wealthy)** | **Strategy B (Global Context)** | $N=57$ was thought too small for a specialist; global context was reported to reduce MAE by ~7 points. |
| **Clusters 2, 3, 5** | **Strategy A (Specialists)** | Specialist models were reported to be most accurate on the large, distinct groups. |

---

## 📊 Key Results

All figures below come from `research/f_validation.ipynb` and are written to `results/`.
Full write-up: **[`results/FINDINGS.md`](results/FINDINGS.md)**.

### External holdout — 746 person-disjoint rows, every model on the same rows, evaluated once

| Model | MAE | 95% CI | R² |
| :--- | ---: | :--- | ---: |
| **Lasso** | **32.73** | 30.96 – 34.53 | 0.544 |
| ElasticNet | 32.75 | 30.98 – 34.52 | 0.543 |
| Ridge | 32.77 | 31.01 – 34.57 | 0.542 |
| HistGradientBoosting | 33.28 | 31.44 – 34.97 | 0.537 |
| Random forest (global) | 33.77 | 31.93 – 35.59 | 0.517 |
| Strategy C (oversampled) | 33.88 | 32.04 – 35.76 | 0.509 |
| Strategy B (cluster feature) | 33.91 | 32.09 – 35.72 | 0.515 |
| Hybrid router | 34.18 | 32.32 – 35.95 | 0.508 |
| Strategy A (specialists) | 34.59 | 32.74 – 36.36 | 0.498 |
| Mean baseline | 49.89 | 47.45 – 52.40 | -0.000 |

**No cluster-aware model beats the conventional benchmark.** Paired bootstrap against Lasso:
Strategy A +1.86 MAE (CI 0.73 – 2.97), hybrid +1.45 (0.42 – 2.45), Strategy B +1.19 (0.26 – 2.16),
Strategy C +1.16 (0.15 – 2.17) — every CI excludes zero. Repeated grouped cross-validation
(15 folds) gives the same ordering.

### Cluster stability

| Check | Adjusted Rand Index |
| :--- | :--- |
| Across 20 seeds, pairwise | 0.488 ± 0.175 |
| Person-level bootstrap (100) | 0.468 ± 0.131 |
| 2003 wave vs 2012 wave features | **0.078** |

Silhouette never exceeds 0.055 for any $k \in [2, 10]$, and is 0.033 at the $k=6$ used
throughout. The partition is not stable across seeds, resamples, or measurement waves.

### Discrimination and calibration (outcome dichotomised at the training lower quartile)

| Model | ROC-AUC | Brier | Sensitivity | Specificity |
| :--- | ---: | ---: | ---: | ---: |
| **Logistic regression** | **0.833** | **0.139** | 0.387 | 0.941 |
| Random forest, cluster-aware | 0.831 | 0.148 | 0.599 | 0.864 |
| Random forest, global | 0.831 | 0.148 | 0.607 | 0.859 |

### Triage classification

The classifier reaches **88.6%** accuracy predicting cluster membership — but from the same 158
features K-Means clustered on, against a 34.8% majority-class baseline. That measures how learnable
the K-Means boundary is, not whether the phenotypes are real.

Restricted to the 15 genuinely intake-observable features (age, gender, education, marital status,
self-rated health, urbanicity), accuracy falls to **59.3%** (balanced accuracy 0.434). Assigning
personas from basic demographic questions does not work.

---

## 🛠️ Installation & Usage

This project uses Python. We recommend using `uv` for fast package management, or standard `pip`.

### Prerequisites
*   Python 3.9+
*   JupyterLab

### Setup
```bash
# 1. Clone the repository
- git clone https://github.com/yourusername/cognitive-health-hybrid.git

- cd cognitive-health-hybrid

# 2. Create virtual environment

[uv](https://github.com/astral-sh/uv) is a fast Python package manager and environment tool recommended for this project.

**Install uv**
- You can install `uv` using pip:

- pip install uv

- uv venv --python=3.11 .venv
- uv sync
```

# Cognitive Health Prediction: A Hybrid Unsupervised & Supervised Learning System

## 📌 Project Overview
This research project aims to predict **Cognitive Health Scores** (`composite_score`) in an aging population. 

Traditional regression models often fail to capture the distinct etiology of cognitive decline across different socioeconomic and health profiles. Instead of a "one-size-fits-all" approach, this project implements a **Hybrid Machine Learning System** that:
1.  **Discovers Phenotypes:** Uses Unsupervised Learning (K-Means) to identify distinct patient personas (e.g., "The Frail", "The Ultra-Wealthy", "The Professionals").
2.  **Adaptive Modeling:** Routes patients to specific predictive models optimized for their phenotype.
3.  **Explainability:** Uses SHAP to validate the distinct biological and socioeconomic drivers for each group.

---

## 📂 Repository Structure & Workflow

The analysis is divided into sequential notebooks, each representing a distinct phase of the research pipeline.

| Sequence | File Name | Description |
| :--- | :--- | :--- |
| **1. Unsupervised** | `research/c_clustering.ipynb` | **Phenotype Discovery.** Performs Data Cleaning, PCA/Scaling, and K-Means Clustering. Identifies 5 core clusters and 1 outlier group. |
| **2. Strategy A** (Predictive Modeling) | `research/d_strategyA.ipynb` | **Specialist Models.** Trains separate Random Forest regressors for each cluster ("Divide and Conquer"). |
| **3. Strategy B (Global Meta Feature)** | `research/d_strategyB.ipynb` | **Global Meta-Model.** Trains one global model with `cluster_id` as a feature. (Best for small groups like "The Ultra-Wealthy"). |
| **4. Strategy C (Cluster Over Sampling)** | `research/d_strategyC.ipynb` | **Oversampling.** Uses resampling techniques to boost the signal of minority groups. (Best for "The Frail & Vulnerable"). |
| **5. Strtegy D (Validation - Triage Classifier)** | `research/d_strategyD.ipynb` | **Triage System.** Trains a Classifier to predict the Cluster ID based on raw data. Validates that phenotypes are distinct (93% Accuracy). |
| **6. Deployment** | `research/hybrid_inference_system_a.ipynb` and `research/hybrid_inference_system_b.ipynb` | **Final Inference Engine.** The system which Routes new patients to the best model (A, B, or C) based on their profile. |
| **7. Strategy E (XAI)** | `research/d_strategyE.ipynb` | **Explainability.** Uses SHAP values to prove that drivers of health (e.g., Income vs. ADLs) differ fundamentally across clusters. |

---

## 🧠 Methodology: The Hybrid System

Through rigorous experimentation, we determined that no single modeling strategy works best for everyone. We constructed a **Hybrid Router** that dynamically selects the best mathematical approach:

### The 5 Phenotypes (Clusters)
*   **Cluster 0 (The Frail & Vulnerable):** High physical impairment, high depression.
*   **Cluster 1 (The Ultra-Wealthy):** Extremely high capital income, small sample size ($N=57$).
*   **Cluster 2 (Working Middle Class):** Average income, currently working.
*   **Cluster 3 (Non-Working Middle Class):** Relies on spousal income, homemakers.
*   **Cluster 5 (High-Earning Professionals):** High education, high salary, very healthy.

### The Routing Logic
| Patient Profile | Strategy Used | Justification |
| :--- | :--- | :--- |
| **Cluster 0 (Frail)** | **Strategy C (Oversampling)** | Standard models ignore "sickness" signals in favor of the healthy majority. Oversampling reduced Error (MAE) by **~4.5 points**. |
| **Cluster 1 (Wealthy)** | **Strategy B (Global Context)** | Sample size ($N=57$) was too small for a specialist model. Global context stabilized predictions, reducing Error by **~7 points**. |
| **Clusters 2, 3, 5** | **Strategy A (Specialists)** | Large, distinct populations. Specialist models provided the highest accuracy by isolating specific lifestyle patterns. |

---

## 📊 Key Results

The Hybrid System significantly outperformed the baseline Global Model on unseen test data.

### Performance by Group (Mean Absolute Error)
*   **Cluster 0 (Frail):** 29.08 (vs Baseline 35.19)
*   **Cluster 1 (Wealthy):** 31.14 (vs Baseline 47.00+)
*   **Cluster 5 (Pros):** 28.63 (Highest Accuracy, $R^2 \approx 0.60$)

### Triage Classification
The Triage Classifier (`g_triage_classifier.ipynb`) achieved **93% Accuracy**, validating that these phenotypes are clinically distinct and identifiable using basic demographic questions.

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

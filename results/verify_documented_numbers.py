"""Cross-check every number quoted in README.md and results/FINDINGS.md against results/*.csv.

Run from the repository root:  python tests/test_docs_match_results.py
Exits non-zero if any documented figure has drifted from the CSVs that produced it.
"""
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

def load(name):
    with open(RES / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def num(x):
    return round(float(x), 4)

failures, checks = [], 0

def check(label, expected, doc_text, fmt="{:.2f}"):
    """Assert the formatted expected value literally appears in doc_text."""
    global checks
    checks += 1
    s = fmt.format(expected)
    if s not in doc_text:
        failures.append(f"{label}: expected substring {s!r} not found")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
findings = (RES / "FINDINGS.md").read_text(encoding="utf-8")
both = readme + "\n" + findings

# --- holdout table ---------------------------------------------------------
holdout = {r["model"]: r for r in load("16_holdout_results.csv")}
for model, row in holdout.items():
    for doc, name in ((readme, "README"), (findings, "FINDINGS")):
        check(f"{name} {model} mae", num(row["mae"]), doc)
        check(f"{name} {model} mae_lo", num(row["mae_lo"]), doc)
        check(f"{name} {model} mae_hi", num(row["mae_hi"]), doc)
        check(f"{name} {model} r2", num(row["r2"]), doc, "{:.3f}")

# ranking claim: lasso is best by MAE, dummy worst
order = sorted(holdout.values(), key=lambda r: float(r["mae"]))
if order[0]["model"] != "1b_lasso":
    failures.append(f"claim 'Lasso is best on holdout' is wrong: {order[0]['model']}")
checks += 1

# every cluster-aware model must be worse than every linear model
CLUSTER = {"4_strategy_A_specialists", "5_strategy_B_cluster_feature",
           "6_strategy_C_oversampled", "7_hybrid_router"}
LINEAR = {"1a_ridge", "1b_lasso", "1c_elasticnet"}
for c in CLUSTER:
    for l in LINEAR:
        checks += 1
        if not float(holdout[c]["mae"]) > float(holdout[l]["mae"]):
            failures.append(f"claim 'no cluster-aware model beats a linear model' violated: {c} <= {l}")

# --- holdout contrasts vs lasso -------------------------------------------
for row in load("17_holdout_contrasts.csv"):
    if row["reference"] != "1b_lasso":
        continue
    for doc, name in ((readme, "README"), (findings, "FINDINGS")):
        check(f"{name} contrast {row['model']} diff", num(row["mae_diff"]), doc)
        check(f"{name} contrast {row['model']} lo", num(row["ci_lo"]), doc)
        check(f"{name} contrast {row['model']} hi", num(row["ci_hi"]), doc)
    checks += 1
    if row["verdict"] != "reference better":
        failures.append(f"claim 'every CI excludes zero vs lasso' violated: {row['model']} -> {row['verdict']}")
    checks += 1
    if float(row["ci_lo"]) <= 0:
        failures.append(f"CI for {row['model']} does not exclude zero: lo={row['ci_lo']}")

# --- stability -------------------------------------------------------------
stab = {r["check"]: r for r in load("07_stability_summary.csv")}
seed = stab["seed (20 seeds, pairwise)"]
boot = stab["person-level bootstrap (100)"]
wave = stab["cross-wave (2003 vs 2012)"]
for doc, name in ((readme, "README"), (findings, "FINDINGS")):
    check(f"{name} seed ari", num(seed["ari_mean"]), doc, "{:.3f}")
    check(f"{name} seed sd", num(seed["ari_sd"]), doc, "{:.3f}")
    check(f"{name} boot ari", num(boot["ari_mean"]), doc, "{:.3f}")
    check(f"{name} boot sd", num(boot["ari_sd"]), doc, "{:.3f}")
    check(f"{name} wave ari", num(wave["ari_mean"]), doc, "{:.3f}")

# --- k selection -----------------------------------------------------------
ks = load("02_k_selection.csv")
max_sil = max(float(r["silhouette"]) for r in ks)
sil_k6 = next(float(r["silhouette"]) for r in ks if r["k"] == "6")
checks += 2
if not max_sil < 0.055:
    failures.append(f"claim 'silhouette never exceeds 0.055' wrong: max={max_sil}")
if round(sil_k6, 3) != 0.033:
    failures.append(f"claim 'silhouette 0.033 at k=6' wrong: {sil_k6}")
checks += 1
singletons = [r["k"] for r in ks if int(r["k"]) >= 3 and int(r["smallest_cluster"]) != 1]
if singletons:
    failures.append(f"claim 'for k>=3 smallest cluster is always 1' wrong at k={singletons}")

# --- binary arm ------------------------------------------------------------
binary = {r["model"]: r for r in load("14_binary_cv_summary.csv")}
for model in ("1_logistic_regression", "2_random_forest_global", "3_random_forest_cluster_aware"):
    r = binary[model]
    for doc, name in ((readme, "README"), (findings, "FINDINGS")):
        check(f"{name} {model} auc", num(r["roc_auc_mean"]), doc, "{:.3f}")
        check(f"{name} {model} brier", num(r["brier_mean"]), doc, "{:.3f}")
        check(f"{name} {model} sens", num(r["sensitivity_mean"]), doc, "{:.3f}")
        check(f"{name} {model} spec", num(r["specificity_mean"]), doc, "{:.3f}")
checks += 1
best_auc = max(binary.values(), key=lambda r: float(r["roc_auc_mean"]))["model"]
if best_auc != "1_logistic_regression":
    failures.append(f"claim 'logistic regression best on AUC' wrong: {best_auc}")
checks += 1
best_brier = min(binary.values(), key=lambda r: float(r["brier_mean"]))["model"]
if best_brier != "1_logistic_regression":
    failures.append(f"claim 'logistic regression best on Brier' wrong: {best_brier}")

# --- triage ----------------------------------------------------------------
triage = {r["feature_set"]: r for r in load("15_triage_reframed.csv")}
allf = triage["all_features (as published)"]
intake = triage["intake_observable_only"]
base = triage["majority-class baseline"]
for doc, name in ((readme, "README"), (findings, "FINDINGS")):
    check(f"{name} triage all", float(allf["accuracy_mean"]) * 100, doc, "{:.1f}")
    check(f"{name} triage intake", float(intake["accuracy_mean"]) * 100, doc, "{:.1f}")
    check(f"{name} triage baseline", float(base["accuracy_mean"]) * 100, doc, "{:.1f}")
    check(f"{name} triage intake balacc", num(intake["balanced_accuracy_mean"]), doc, "{:.3f}")
check("FINDINGS n_features all", int(allf["n_features"]), findings, "{:d}")
check("FINDINGS n_features intake", int(intake["n_features"]), findings, "{:d}")
check("README n_features all", int(allf["n_features"]), readme, "{:d}")
check("README n_features intake", int(intake["n_features"]), readme, "{:d}")

# --- partitions ------------------------------------------------------------
parts = {r["partition"]: r for r in load("01_partition_summary.csv")}
tr, te = parts["train (development)"], parts["test.csv (external holdout)"]
check("FINDINGS train rows", int(tr["n_rows"]), findings, "{:,d}")
check("FINDINGS train persons", int(tr["n_persons"]), findings, "{:,d}")
check("FINDINGS train dupes", int(tr["persons_with_2_rows"]), findings, "{:d}")
check("FINDINGS test rows", int(te["n_rows"]), findings, "{:d}")
check("FINDINGS test persons", int(te["n_persons"]), findings, "{:d}")
check("FINDINGS test dupes", int(te["persons_with_2_rows"]), findings, "{:d}")

# --- CV summary ------------------------------------------------------------
cv = {r["model"]: r for r in load("10_cv_summary.csv")}
check("FINDINGS cv hgb", num(cv["3_hist_gradient_boosting"]["mae_mean"]), findings)
check("FINDINGS cv lasso", num(cv["1b_lasso"]["mae_mean"]), findings)
checks += 1
cv_best = min(cv.values(), key=lambda r: float(r["mae_mean"]))["model"]
if cv_best != "3_hist_gradient_boosting":
    failures.append(f"claim 'HGB best in CV' wrong: {cv_best}")
checks += 1
n_folds = {r["n_folds"] for r in cv.values()}
if n_folds != {"15"}:
    failures.append(f"claim '15 folds' wrong: {n_folds}")
checks += 1
b_vs_lasso = next(r for r in load("11_cv_contrasts.csv")
                  if r["model"] == "5_strategy_B_cluster_feature" and r["reference"] == "1b_lasso")
if b_vs_lasso["verdict"] != "inconclusive":
    failures.append(f"claim 'Strategy B indistinguishable from Lasso in CV' wrong: {b_vs_lasso['verdict']}")
check("FINDINGS cv B diff", num(b_vs_lasso["mae_diff"]), findings)

# --- per-cluster holdout ---------------------------------------------------
pc = {r["cluster"]: r for r in load("18_holdout_per_cluster.csv")}
c1 = pc["1"]
check("FINDINGS cluster1 gap", abs(float(c1["mae_advantage_hybrid"])), findings, "{:.1f}")
checks += 1
losing = [k for k, r in pc.items() if float(r["mae_advantage_hybrid"]) < 0]
if len(losing) != 4:
    failures.append(f"claim 'hybrid loses to linear in 4 of 5 clusters' wrong: {len(losing)} ({losing})")

# --- cluster shares --------------------------------------------------------
shares = load("03_cluster_shares.csv")
max_gap = max(float(r["abs_diff"]) for r in shares)
checks += 1
if not max_gap <= 0.008:
    failures.append(f"claim 'within 0.8pp' wrong: max abs_diff={max_gap}")

# --- historical ------------------------------------------------------------
hist = {r["source"]: r for r in load("19_historical_numbers.csv")}
check("FINDINGS hist hybrid mae", num(hist["hybrid_inference_system_b.ipynb"]["reported_mae"]), findings)
check("FINDINGS hist autogluon mae", num(hist["train.ipynb AutoGluon WeightedEnsemble_L2"]["reported_mae"]), findings)
checks += 1
if any(r["comparable"] != "False" for r in hist.values()):
    failures.append("historical rows should all be flagged non-comparable")

# --- no superseded claims survive -----------------------------------------
BANNED = ["g_triage_classifier", "29.08", "31.14", "28.63",
          "significantly outperformed", "Uses SHAP to validate", "PCA/Scaling"]
for term in BANNED:
    checks += 1
    if term in readme:
        failures.append(f"superseded claim still in README: {term!r}")

print(f"{checks} assertions checked")
if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all consistent with results/*.csv")

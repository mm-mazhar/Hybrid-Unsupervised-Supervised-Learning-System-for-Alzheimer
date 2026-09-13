"""Leakage-safe validation helpers for the MHAS cognitive-health pipeline.

This module exists to answer specific reviewer objections to the manuscript. Each
helper corresponds to something the original notebooks either did incorrectly or
did not do at all:

*   ``build_preprocessor`` / ``CategoricalAligner`` -- fit preprocessing on train
    only and transform unseen data safely. The original ``c_clustering.ipynb``
    fitted the scaler, the one-hot encoding and K-Means on every available row.
*   ``person_level_split`` / ``person_level_folds`` -- partition by ``UID`` rather
    than by row. 647 training UIDs contribute two rows each (a 2016 and a 2021
    outcome) whose predictors are byte-identical, so a row-level split leaks a
    patient's twin across the boundary.
*   ``ClusterSpace`` -- an explicit, persistable scaler + encoder + K-Means bundle
    with a real ``predict`` path. Nothing in the original pipeline saved these
    objects, which forced the inference notebooks to approximate the router with
    unscaled nearest-centroid distances.
*   ``cluster_stability`` -- seed, bootstrap and cross-wave agreement statistics.
*   ``regression_metrics`` / ``bootstrap_ci`` / ``paired_bootstrap_diff`` --
    metrics with uncertainty, so "strategy X beats strategy Y" carries an
    interval rather than a single number.

Conventions follow the rest of ``research/utils``: scikit-learn compatible
transformers, pandas in and pandas out, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from .tranformerColumnsHighNA import DropColumnsHighNA
from .transformerDataTypesConversion import (
    FloatToCategoryTransformer,
    ObjectToCategoryTransformer,
)
from .transformerDropColumns import ColumnDropper
from .transformerDropLowVarNum import IdentifyAndDropLowVarNum
from .transformerImputeMissingValues import MissingValueImputer

SEED = 42
TARGET = "composite_score"
GROUP_KEY = "UID"

# Redundant / near-duplicate insurance and affect columns, carried over verbatim
# from c_clustering.ipynb cell 7 so the feature space stays comparable to the
# original analysis. UID is dropped here too, hence stash_groups() must run first.
COLS_TO_DROP: list[str] = [
    "UID",
    "imss_03",
    "imss_12",
    "issste_03",
    "issste_12",
    "pem_def_mar_03",
    "pem_def_mar_12",
    "insur_private_03",
    "insur_private_12",
    "insur_other_03",
    "insur_other_12",
    "seg_pop_12",
    "Tired_03",
    "Tired_12",
    "Happy_03",
    "Happy_12",
]

THRESHOLD_MISSING = 70.0
NUM_STRATEGY = "median"
CAT_STRATEGY = "mode"
THRESHOLD_QUASI_CONSTANT = 1e-8
THRESHOLD_RATIO = 0.1
MAX_UNIQUE = 50


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
class CategoricalAligner(BaseEstimator, TransformerMixin):
    """Force categorical columns onto the category set learned during ``fit``.

    ``MissingValueImputer`` fills a category column with the *training* mode. If
    the conversion to ``category`` dtype happens first, pandas refuses to insert
    a value that is not already among that column's categories, and any category
    present in test but not in train silently becomes NaN. Running this
    transformer after the dtype conversions makes the train/test category sets
    identical and remaps unseen levels to the training mode.

    The count of remapped values is exposed as ``n_unseen_`` so it can be
    reported rather than hidden.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalAligner":
        self.categories_: dict[str, list[Any]] = {}
        self.fallback_: dict[str, Any] = {}
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                cats = list(X[col].cat.categories)
                self.categories_[col] = cats
                mode = X[col].mode()
                self.fallback_[col] = mode.iloc[0] if len(mode) else cats[0]
        self.n_unseen_: dict[str, int] = {}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.n_unseen_ = {}
        for col, cats in self.categories_.items():
            if col not in X.columns:
                continue
            values = X[col].astype(object)
            unseen = ~values.isin(cats) & values.notna()
            n_unseen = int(unseen.sum())
            if n_unseen:
                self.n_unseen_[col] = n_unseen
                values = values.where(~unseen, self.fallback_[col])
            values = values.fillna(self.fallback_[col])
            X[col] = pd.Categorical(values, categories=cats)
        return X


class NumericDtypeAligner(BaseEstimator, TransformerMixin):
    """Cast numeric columns to the dtype seen during ``fit``.

    Imputation can leave an integer-valued column as int in one partition and
    float in the other. Harmless for the models, but it breaks strict
    train/test schema assertions, so normalise it.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NumericDtypeAligner":
        self.dtypes_ = {
            col: X[col].dtype
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, dtype in self.dtypes_.items():
            if col in X.columns and X[col].dtype != dtype:
                X[col] = X[col].astype(dtype)
        return X


def build_preprocessor(
    cols_to_drop: Sequence[str] | None = None,
    threshold_missing: float = THRESHOLD_MISSING,
    num_strategy: str = NUM_STRATEGY,
    cat_strategy: str = CAT_STRATEGY,
    quasi_constant_threshold: float = THRESHOLD_QUASI_CONSTANT,
    threshold_ratio: float = THRESHOLD_RATIO,
    max_unique: int = MAX_UNIQUE,
) -> Pipeline:
    """Assemble the cleaning pipeline used for every model in the comparison.

    Same transformers and thresholds as ``c_clustering.ipynb`` cell 14, with two
    deliberate changes:

    1.  Imputation runs *before* the dtype conversions. The original order made
        ``fit`` on train / ``transform`` on test raise a ``TypeError`` on the
        first categorical column whose test values differ from train, which is
        why the original notebooks never separated fit from transform.
    2.  ``SpecificColumnCategorizer`` for ``Year`` is gone. The caller drops the
        raw outcome year and keeps ``PredictionYear`` (the 4- or 9-year horizon)
        as a numeric feature. Horizon is known at prediction time and is what
        distinguishes a person's two rows.
    """
    cols_to_drop = list(COLS_TO_DROP if cols_to_drop is None else cols_to_drop)
    return Pipeline(
        [
            ("1_drop_columns", ColumnDropper(columns_to_drop=cols_to_drop)),
            ("2_drop_high_na", DropColumnsHighNA(threshold=threshold_missing)),
            (
                "3_impute",
                MissingValueImputer(
                    num_strategy=num_strategy, cat_strategy=cat_strategy
                ),
            ),
            (
                "4_object_to_category",
                ObjectToCategoryTransformer(
                    threshold_ratio=threshold_ratio, max_unique=max_unique
                ),
            ),
            ("5_float_to_category", FloatToCategoryTransformer()),
            ("6_align_categories", CategoricalAligner()),
            ("7_align_numeric_dtypes", NumericDtypeAligner()),
            (
                "8_drop_low_variance",
                IdentifyAndDropLowVarNum(
                    quasi_constant_threshold=quasi_constant_threshold
                ),
            ),
        ]
    )


def stash_groups(
    df: pd.DataFrame, group_key: str = GROUP_KEY
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(df, groups)`` with the grouping key preserved as a Series.

    ``COLS_TO_DROP`` removes ``UID``, so the person identifier has to be pulled
    out before preprocessing or the person-level split becomes impossible. This
    is the step whose absence produced the twin-row leakage.
    """
    if group_key not in df.columns:
        raise KeyError(f"{group_key!r} not in dataframe columns")
    return df, df[group_key].copy()


def split_features_target(
    df: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target].copy()


def column_kinds(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into ``(numeric, categorical)`` name lists."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


# --------------------------------------------------------------------------
# Person-level partitioning
# --------------------------------------------------------------------------
def person_level_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Single train/validation split where no ``UID`` spans the boundary."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def person_level_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = SEED,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, fold, train_idx, test_idx)`` for repeated grouped CV.

    ``GroupKFold`` is deterministic, so repeats are generated by shuffling the
    group labels through a permutation before splitting. The original notebooks
    used a single ``random_state=42`` split everywhere and reported no variance
    across resamples at all.
    """
    rng = np.random.default_rng(random_state)
    group_values = pd.Index(groups.unique())
    for repeat in range(n_repeats):
        permuted = rng.permutation(group_values)
        remap = {g: i for i, g in enumerate(permuted)}
        shuffled = groups.map(remap)
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups=shuffled)
        ):
            yield repeat, fold, train_idx, test_idx


def assert_no_group_leakage(
    groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Raise if any person appears on both sides of a split."""
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise AssertionError(
            f"{len(overlap)} UIDs appear in both train and test, e.g. "
            f"{sorted(overlap)[:5]}"
        )


def partition_summary(
    name: str, y: pd.Series, groups: pd.Series
) -> dict[str, Any]:
    """Descriptive row for the partitioning table the reviewer asked for."""
    counts = groups.value_counts()
    return {
        "partition": name,
        "n_rows": int(len(y)),
        "n_persons": int(groups.nunique()),
        "rows_per_person_mean": round(float(counts.mean()), 3),
        "persons_with_2_rows": int((counts == 2).sum()),
        "target_mean": round(float(y.mean()), 3),
        "target_sd": round(float(y.std()), 3),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
@dataclass
class ClusterSpace:
    """Scaler + one-hot encoder + K-Means, fitted on training rows only.

    Replaces the original approach of ``StandardScaler().fit_transform`` and
    ``pd.get_dummies`` over the whole dataset followed by reading
    ``kmeans.labels_``. ``get_dummies`` cannot map unseen data onto a fixed
    column space, which is why no ``predict`` path existed; ``OneHotEncoder``
    with ``handle_unknown="ignore"`` can.
    """

    n_clusters: int
    random_state: int = SEED
    n_init: int = 10
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)

    def _design_matrix(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        numeric = X[self.numeric_cols]
        categorical = X[self.categorical_cols].astype(object)
        if fit:
            scaled = self.scaler_.fit_transform(numeric)
            encoded = self.encoder_.fit_transform(categorical)
        else:
            scaled = self.scaler_.transform(numeric)
            encoded = self.encoder_.transform(categorical)
        return np.hstack([scaled, encoded])

    def fit(self, X: pd.DataFrame) -> "ClusterSpace":
        if not self.numeric_cols and not self.categorical_cols:
            self.numeric_cols, self.categorical_cols = column_kinds(X)
        self.scaler_ = StandardScaler()
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="first"
        )
        design = self._design_matrix(X, fit=True)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.kmeans_.fit(design)
        self.n_features_ = design.shape[1]
        self.n_samples_fit_ = design.shape[0]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._design_matrix(X, fit=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign cluster labels to unseen rows. Absent from the original code."""
        return self.kmeans_.predict(self.transform(X))

    @property
    def labels_(self) -> np.ndarray:
        return self.kmeans_.labels_


def choose_k(
    design: np.ndarray,
    k_range: Sequence[int] = range(2, 11),
    random_state: int = SEED,
    n_init: int = 10,
    sample_size: int | None = 2000,
) -> pd.DataFrame:
    """Diagnostics per candidate k.

    The original analysis picked k=6 from a visually smooth elbow curve and
    reported no internal validity index. Silhouette, Calinski-Harabasz and
    Davies-Bouldin are all cheap here and give the choice something defensible.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(design)
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(
                    silhouette_score(
                        design,
                        labels,
                        sample_size=sample_size,
                        random_state=random_state,
                    )
                ),
                "calinski_harabasz": float(
                    calinski_harabasz_score(design, labels)
                ),
                "davies_bouldin": float(davies_bouldin_score(design, labels)),
                "smallest_cluster": int(np.bincount(labels).min()),
                "largest_cluster": int(np.bincount(labels).max()),
            }
        )
    return pd.DataFrame(rows)


def seed_stability(
    design: np.ndarray,
    n_clusters: int,
    seeds: Sequence[int],
    n_init: int = 10,
) -> pd.DataFrame:
    """Pairwise adjusted Rand index across K-Means refits with different seeds."""
    labelings = {
        seed: KMeans(
            n_clusters=n_clusters, random_state=int(seed), n_init=n_init
        ).fit_predict(design)
        for seed in seeds
    }
    rows = []
    seed_list = list(labelings)
    for i, a in enumerate(seed_list):
        for b in seed_list[i + 1 :]:
            rows.append(
                {
                    "seed_a": int(a),
                    "seed_b": int(b),
                    "ari": float(
                        adjusted_rand_score(labelings[a], labelings[b])
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_stability(
    X: pd.DataFrame,
    groups: pd.Series,
    reference_labels: np.ndarray,
    n_clusters: int,
    n_boot: int = 100,
    random_state: int = SEED,
    n_init: int = 10,
) -> pd.DataFrame:
    """Cluster agreement under person-level bootstrap resampling.

    Resampling is done over ``UID`` rather than rows, so a person is drawn whole.
    Each replicate refits the full scaler/encoder/K-Means stack and is compared,
    on the rows both partitions share, against the reference labeling.
    """
    rng = np.random.default_rng(random_state)
    unique_groups = groups.unique()
    numeric, categorical = column_kinds(X)
    reference = pd.Series(reference_labels, index=X.index)
    rows = []
    for b in range(n_boot):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = groups.isin(set(drawn))
        idx = X.index[mask.to_numpy()]
        space = ClusterSpace(
            n_clusters=n_clusters,
            random_state=int(rng.integers(0, 10_000)),
            n_init=n_init,
            numeric_cols=numeric,
            categorical_cols=categorical,
        ).fit(X.loc[idx])
        rows.append(
            {
                "replicate": b,
                "n_rows": int(len(idx)),
                "ari": float(
                    adjusted_rand_score(reference.loc[idx], space.labels_)
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_wave_stability(
    X: pd.DataFrame,
    n_clusters: int,
    suffix_a: str = "_03",
    suffix_b: str = "_12",
    random_state: int = SEED,
    n_init: int = 10,
) -> dict[str, Any]:
    """Do the 2003 and 2012 waves imply the same partition?

    Clusters the ``_03`` block and the ``_12`` block independently and compares
    the two labelings by ARI. This is the direct answer to "no evidence the
    clusters are stable across samples or waves", using columns already present
    in the data.
    """
    out: dict[str, Any] = {}
    labelings = {}
    for tag, suffix in (("wave_03", suffix_a), ("wave_12", suffix_b)):
        cols = [c for c in X.columns if c.endswith(suffix)]
        numeric, categorical = column_kinds(X[cols])
        space = ClusterSpace(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            numeric_cols=numeric,
            categorical_cols=categorical,
        ).fit(X[cols])
        labelings[tag] = space.labels_
        out[f"n_cols_{tag}"] = len(cols)
    out["ari_wave03_vs_wave12"] = float(
        adjusted_rand_score(labelings["wave_03"], labelings["wave_12"])
    )
    out["labels"] = labelings
    return out


def cluster_share_table(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> pd.DataFrame:
    """Cluster prevalence in train vs holdout.

    The original router sent 69% of the holdout to cluster 0 against 23% of
    training rows. This table shows whether a correct ``predict`` reproduces the
    training distribution.
    """
    train = pd.Series(train_labels).value_counts(normalize=True).sort_index()
    test = pd.Series(test_labels).value_counts(normalize=True).sort_index()
    table = pd.DataFrame(
        {"train_share": train, "holdout_share": test}
    ).fillna(0.0)
    table.index.name = "cluster"
    table["abs_diff"] = (table["train_share"] - table["holdout_share"]).abs()
    return table.round(4)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = SEED,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI for one metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(random_state)
    point = float(metric(y_true, y_pred))
    n = len(y_true)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def metrics_with_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_boot: int = 2000,
    random_state: int = SEED,
) -> dict[str, Any]:
    """MAE / RMSE / R^2 each with a 95% bootstrap interval."""
    metrics = {
        "mae": mean_absolute_error,
        "rmse": lambda a, b: float(np.sqrt(mean_squared_error(a, b))),
        "r2": r2_score,
    }
    out: dict[str, Any] = {"n": int(len(y_true))}
    for name, fn in metrics.items():
        point, lo, hi = bootstrap_ci(
            y_true, y_pred, fn, n_boot=n_boot, random_state=random_state
        )
        out[name] = round(point, 4)
        out[f"{name}_lo"] = round(lo, 4)
        out[f"{name}_hi"] = round(hi, 4)
    return out


def paired_bootstrap_diff(
    y_true: Sequence[float],
    y_pred_a: Sequence[float],
    y_pred_b: Sequence[float],
    metric: Callable[[np.ndarray, np.ndarray], float] = mean_absolute_error,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = SEED,
) -> dict[str, float]:
    """CI for ``metric(a) - metric(b)`` on the same rows.

    Both models are scored on identical bootstrap draws, which is what makes the
    comparison paired. For MAE a negative difference means model ``a`` is better.
    The interval is what turns "the hybrid beats the baseline" into a claim with
    a stated uncertainty.
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], a[idx]) - metric(y_true[idx], b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "diff": float(metric(y_true, a) - metric(y_true, b)),
        "diff_lo": float(lo),
        "diff_hi": float(hi),
        "favours_a": bool(hi < 0),
        "favours_b": bool(lo > 0),
        "inconclusive": bool(lo <= 0 <= hi),
    }


def per_cluster_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    clusters: Sequence[int],
) -> pd.DataFrame:
    """Within-cluster error.

    MAE and RMSE are on the outcome scale and comparable across subgroups. R^2
    is *not*: it is scaled by the variance of whichever subgroup it is computed
    on, and restricting to a cluster shrinks that variance. The original
    ``d_strategyA`` notebook compared a within-cluster R^2 of 0.59 against a
    whole-sample 0.38 and read it as improvement. R^2 is reported here only for
    continuity, with ``outcome_sd`` alongside so the artifact is visible.
    """
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "cluster": np.asarray(clusters),
        }
    )
    rows = []
    for cluster, chunk in frame.groupby("cluster"):
        rows.append(
            {
                "cluster": cluster,
                "n": int(len(chunk)),
                "outcome_sd": round(float(chunk["y_true"].std()), 4),
                "mae": round(
                    float(mean_absolute_error(chunk["y_true"], chunk["y_pred"])),
                    4,
                ),
                "rmse": round(
                    float(
                        np.sqrt(
                            mean_squared_error(chunk["y_true"], chunk["y_pred"])
                        )
                    ),
                    4,
                ),
                "r2_not_comparable": round(
                    float(r2_score(chunk["y_true"], chunk["y_pred"])), 4
                )
                if len(chunk) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# --------------------------------------------------------------------------
# Oversampling
# --------------------------------------------------------------------------
def oversample_by_cluster(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: Sequence[int],
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Balance cluster sizes by resampling with replacement.

    Same procedure as ``d_strategyC.ipynb`` cell 8, which was already correct:
    the caller must pass training rows only. Kept as a function so the ordering
    is enforced by the call site rather than by a comment.
    """
    clusters = np.asarray(clusters)
    frame = X.copy()
    frame["__y"] = y.to_numpy()
    frame["__cluster"] = clusters
    target_size = int(pd.Series(clusters).value_counts().max())
    parts = []
    for cluster, chunk in frame.groupby("__cluster"):
        parts.append(
            resample(
                chunk,
                replace=True,
                n_samples=target_size,
                random_state=random_state,
            )
        )
    balanced = pd.concat(parts).sample(
        frac=1.0, random_state=random_state
    )
    y_bal = balanced.pop("__y")
    cluster_bal = balanced.pop("__cluster").to_numpy()
    return balanced, y_bal, cluster_bal


# --------------------------------------------------------------------------
# Strategy estimators
# --------------------------------------------------------------------------
class ClusterSpecialistRegressor(BaseEstimator):
    """Strategy A: one model per cluster ("divide and conquer").

    Requires cluster labels at fit and predict time. Clusters smaller than
    ``min_cluster_size`` fall back to a model fitted on all rows, so a tiny
    group such as the N=57 high-income cluster cannot silently produce a
    specialist fitted on a handful of observations.
    """

    def __init__(self, base_estimator, min_cluster_size: int = 30):
        self.base_estimator = base_estimator
        self.min_cluster_size = min_cluster_size

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Sequence[int]):
        clusters = np.asarray(clusters)
        self.fallback_ = clone(self.base_estimator).fit(X, y)
        self.models_: dict[Any, Any] = {}
        self.fallback_clusters_: list[Any] = []
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if mask.sum() < self.min_cluster_size:
                self.fallback_clusters_.append(cluster)
                continue
            self.models_[cluster] = clone(self.base_estimator).fit(
                X[mask], y[mask]
            )
        return self

    def predict(self, X: pd.DataFrame, clusters: Sequence[int]) -> np.ndarray:
        clusters = np.asarray(clusters)
        out = np.empty(len(X), dtype=float)
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            model = self.models_.get(cluster, self.fallback_)
            out[mask] = model.predict(X[mask])
        return out


def route_predictions(
    X: pd.DataFrame,
    clusters: Sequence[int],
    predictors: dict[str, Callable[[pd.DataFrame, np.ndarray], np.ndarray]],
    routes: dict[Any, str],
    default: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Send each cluster's rows to its assigned strategy.

    Mirrors the deployed hybrid (cluster 0 to the oversampled model, the small
    high-income cluster to the global cluster-aware model, the rest to
    specialists) but expects labels from ``ClusterSpace.predict`` rather than the
    unscaled nearest-centroid approximation used in
    ``hybrid_inference_system_a/b.ipynb``, where raw income terms dominated the
    Euclidean distance.

    ``predictors`` maps a strategy name to a callable taking ``(rows, labels)``,
    which lets Strategy B receive its cluster-augmented frame while the others
    take plain features. Returns predictions and the strategy used per row.
    """
    clusters = np.asarray(clusters)
    out = np.empty(len(X), dtype=float)
    used = np.empty(len(X), dtype=object)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        key = routes.get(cluster, default)
        out[mask] = predictors[key](X[mask], clusters[mask])
        used[mask] = key
    return out, used

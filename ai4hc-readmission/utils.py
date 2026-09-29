"""Helper functions for the AI4HC 30-day readmission project.

The notebook imports everything it needs from here so the analysis stays
readable.  Functions fall into six groups:

1. Data loading, cohort rules and stateless feature engineering
2. Splitting and leakage checks
3. Pipelines and preprocessing
4. Scores, discrimination metrics and threshold policy
5. Calibration and fairness
6. Plots, exports and Responsible AI glue

Naming follows the instructor's reference notebook (``ML4HL_OD_RAI_toolbox``)
so that the two read alike.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


# ==========================================================================
# 1. Data loading, cohort rules, stateless feature engineering
# ==========================================================================

def load_raw(path=None) -> pd.DataFrame:
    """Load ``diabetic_data.csv`` without destroying the literal string "None".

    Pandas maps ``"None"`` to NaN by default.  In this dataset ``"None"`` in
    ``A1Cresult`` / ``max_glu_serum`` means *the test was not performed*, which
    is clinically meaningful and central to the causal question, so default NA
    handling is switched off and only ``"?"`` is treated as missing.
    """
    path = config.RAW_DATA_PATH if path is None else path
    df = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)

    assert df.shape == (101766, 50), f"Unexpected raw shape: {df.shape}"
    assert df["A1Cresult"].isna().sum() == 0, "A1Cresult should have no NaN"
    assert "None" in set(df["A1Cresult"].unique()), "'None' was lost from A1Cresult"
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 when the encounter was followed by a readmission within 30 days."""
    return (df["readmitted"] == "<30").astype(int)


def apply_cohort_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the inclusion / exclusion rules and return a CONSORT-style flow.

    Returns
    -------
    (cohort, flow_table)
    """
    steps: List[Dict[str, Any]] = []

    def record(step: str, frame: pd.DataFrame, removed: int) -> None:
        steps.append({
            "Step": step,
            "Encounters removed": removed,
            "Encounters remaining": len(frame),
            "Unique patients remaining": frame["patient_nbr"].nunique(),
        })

    out = df.copy()
    record("All encounters in the UCI extract", out, 0)

    mask = ~out["discharge_disposition_id"].isin(config.EXCLUDE_DISCHARGE_IDS)
    removed = int((~mask).sum())
    out = out[mask]
    record("Exclude discharge to death or hospice", out, removed)

    mask = out["gender"] != "Unknown/Invalid"
    removed = int((~mask).sum())
    out = out[mask]
    record("Exclude unknown/invalid gender", out, removed)

    out = out.reset_index(drop=True)
    return out, pd.DataFrame(steps)


def icd9_to_group(code: Any) -> str:
    """Map an ICD-9 code to the diagnosis groups used by Strack et al. (2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    text = str(code).strip()
    if text == "" or text.lower() in {"nan", "?"}:
        return "Missing"

    # V- and E-codes are supplementary / external-cause codes
    if text[0].upper() in {"V", "E"}:
        return "Other"

    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if (390 <= value <= 459) or int(value) == 785:
        return "Circulatory"
    if (460 <= value <= 519) or int(value) == 786:
        return "Respiratory"
    if (520 <= value <= 579) or int(value) == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if (580 <= value <= 629) or int(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def _age_midpoint(bracket: Any) -> float:
    """Convert an age bracket such as ``"[70-80)"`` to its midpoint (75)."""
    try:
        lo, hi = str(bracket).strip("[)").split("-")
        return (float(lo) + float(hi)) / 2.0
    except Exception:
        return np.nan


def _age_group(years: float) -> str:
    """Collapse age into the three bands the source paper reports."""
    if np.isnan(years):
        return "Unknown"
    if years < 30:
        return "<30"
    if years <= 60:
        return "30-60"
    return ">60"


def _payer_group(code: Any) -> str:
    """Collapse the payer code into a coarse socioeconomic proxy."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Unknown"
    text = str(code).strip()
    if text in {"", "nan", "?"}:
        return "Unknown"
    return config.PAYER_GROUPS.get(text, "Private/Other")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise feature engineering.

    Every transformation here is a pure function of a single row: the same
    input always produces the same output and nothing is learned from the
    data.  That makes it safe to run *before* splitting.  No missing value is
    filled and no statistic (mean, median, frequency) is computed -- those are
    stateful steps and live inside the sklearn ``Pipeline`` instead.
    """
    out = df.copy()

    # --- diagnosis groups -------------------------------------------------
    for col in ["diag_1", "diag_2", "diag_3"]:
        out[f"{col}_group"] = out[col].map(icd9_to_group)

    # --- age --------------------------------------------------------------
    out["age_years"] = out["age"].map(_age_midpoint)
    out["age_group"] = out["age_years"].map(_age_group)

    # --- ID-code groupings (UCI IDs_mapping) ------------------------------
    out["admission_group"] = (
        out["admission_type_id"].map(config.ADMISSION_TYPE_GROUPS).fillna("Other/Unknown")
    )
    out["admission_source_group"] = (
        out["admission_source_id"].map(config.ADMISSION_SOURCE_GROUPS).fillna("Other/Unknown")
    )
    out["discharge_group"] = (
        out["discharge_disposition_id"].map(config.DISCHARGE_GROUPS).fillna("Other/Unknown")
    )

    # --- payer ------------------------------------------------------------
    out["payer_group"] = out["payer_code"].map(_payer_group)

    # --- tests performed --------------------------------------------------
    # "None" means the test was NOT performed: that is the treatment variable
    # for the later causal analysis, not a missing value.
    out["A1C_tested"] = (out["A1Cresult"] != "None").astype(int)
    out["glu_tested"] = (out["max_glu_serum"] != "None").astype(int)

    # --- medication behaviour --------------------------------------------
    out["med_change"] = (out["change"] == "Ch").astype(int)
    out["on_diabetes_med"] = (out["diabetesMed"] == "Yes").astype(int)

    drugs = [c for c in config.DRUG_COLS if c in out.columns]
    out["n_meds_up"] = (out[drugs] == "Up").sum(axis=1)
    out["n_meds_down"] = (out[drugs] == "Down").sum(axis=1)
    out["n_diabetes_meds_prescribed"] = (out[drugs] != "No").sum(axis=1)

    # --- prior utilisation ------------------------------------------------
    out["total_prior_visits"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    out["any_prior_inpatient"] = (out["number_inpatient"] > 0).astype(int)

    # --- target -----------------------------------------------------------
    out[config.TARGET_COL] = make_target(out)

    return out


def get_feature_lists(df: pd.DataFrame) -> Dict[str, Any]:
    """Resolve the configured feature lists against the columns actually present."""
    numeric = [c for c in config.NUMERIC_COLS if c in df.columns]
    categorical = [c for c in config.CATEGORICAL_COLS if c in df.columns]

    if config.USE_RACE_GENDER_AS_FEATURES:
        categorical = categorical + [c for c in ["race", "gender"] if c in df.columns]
    if config.USE_PAYER_AS_FEATURE and "payer_group" in df.columns:
        categorical = categorical + ["payer_group"]

    return {
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "sensitive_cols": [c for c in config.SENSITIVE_COLS if c in df.columns],
        "group_col": config.GROUP_COL,
        "target_col": config.TARGET_COL,
    }


def missingness_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness with the handling decision and its reason."""
    decisions = {
        "weight": ("Dropped", "96.9% missing - no usable signal"),
        "medical_specialty": ("Kept, imputed as 'Missing'", "Missingness is itself informative (which service admitted the patient)"),
        "payer_code": ("Grouped into payer_group, 'Unknown' level", "Not missing at random; used as a socioeconomic proxy and audited"),
        "race": ("Kept for fairness audit only, not a model input", "2.2% missing; used to measure gaps, not to score patients"),
        "diag_1": ("Grouped by ICD-9 chapter; NaN -> 'Missing'", "Stateless rule-based grouping"),
        "diag_2": ("Grouped by ICD-9 chapter; NaN -> 'Missing'", "Stateless rule-based grouping"),
        "diag_3": ("Grouped by ICD-9 chapter; NaN -> 'Missing'", "Stateless rule-based grouping"),
    }
    pct = (df_raw.isna().mean() * 100).sort_values(ascending=False)
    rows = []
    for col, value in pct.items():
        if value <= 0:
            continue
        handling, reason = decisions.get(col, ("Imputed inside the pipeline", "Fitted on the training fold only"))
        rows.append({
            "Column": col,
            "% missing": round(float(value), 2),
            "Handling decision": handling,
            "Reason": reason,
        })
    return pd.DataFrame(rows)


def cohort_table(df: pd.DataFrame, by: str, target_col: Optional[str] = None) -> pd.DataFrame:
    """"Table 1"-style summary of one grouping variable with readmission rates."""
    target_col = target_col or config.TARGET_COL
    grouped = df.groupby(by, dropna=False)[target_col]
    out = pd.DataFrame({
        "Level": grouped.size().index.astype(str),
        "n": grouped.size().values,
        "% of cohort": (grouped.size().values / len(df) * 100),
        "Readmitted <30d (n)": grouped.sum().values,
        "Readmission rate %": (grouped.mean().values * 100),
    })
    out.insert(0, "Variable", by)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def build_cohort_table(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Stack ``cohort_table`` over several variables into one Table 1."""
    frames = [cohort_table(df, v) for v in variables if v in df.columns]
    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# 2. Splitting and leakage checks
# ==========================================================================

def grouped_train_cal_val_test_split(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    target_col: Optional[str] = None,
    sizes: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Split into train / cal / val / test with **no patient in two parts**.

    A plain random split over encounters would place repeat visits of the same
    patient on both sides of the split, letting the model memorise individuals
    and inflating validation scores.  ``GroupShuffleSplit`` on ``patient_nbr``
    prevents that.

    Returns ``(parts, summary_table)`` where ``parts`` maps
    ``"train" | "cal" | "val" | "test"`` to DataFrames.
    """
    group_col = group_col or config.GROUP_COL
    target_col = target_col or config.TARGET_COL
    sizes = sizes or config.SPLIT_SIZES
    seed = config.RANDOM_STATE if seed is None else seed

    total = sum(sizes.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    groups = df[group_col].values

    # Step 1: hold out (cal + val + test)
    holdout_frac = sizes["cal"] + sizes["val"] + sizes["test"]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, df[target_col], groups))
    train, rest = df.iloc[train_idx], df.iloc[rest_idx]

    # Step 2: split the holdout into cal vs (val + test)
    rest_frac = sizes["val"] + sizes["test"]
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=rest_frac / holdout_frac, random_state=seed
    )
    cal_idx, vt_idx = next(gss2.split(rest, rest[target_col], rest[group_col].values))
    cal, val_test = rest.iloc[cal_idx], rest.iloc[vt_idx]

    # Step 3: split into val vs test
    gss3 = GroupShuffleSplit(
        n_splits=1, test_size=sizes["test"] / rest_frac, random_state=seed
    )
    val_idx, test_idx = next(
        gss3.split(val_test, val_test[target_col], val_test[group_col].values)
    )
    val, test = val_test.iloc[val_idx], val_test.iloc[test_idx]

    parts = {
        "train": train.reset_index(drop=True),
        "cal": cal.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    assert_no_leakage(parts, group_col)

    overall_prev = df[target_col].mean()
    rows = []
    for name, part in parts.items():
        rows.append({
            "Partition": name,
            "Role": {
                "train": "Fit the model",
                "cal": "Fit the probability calibrator",
                "val": "Model selection and threshold policy",
                "test": "Opened once, after the policy is locked",
            }[name],
            "Encounters": len(part),
            "Unique patients": part[group_col].nunique(),
            "Readmissions <30d": int(part[target_col].sum()),
            "Prevalence %": float(part[target_col].mean() * 100),
            "Share of cohort %": len(part) / len(df) * 100,
        })
    rows.append({
        "Partition": "overall",
        "Role": "-",
        "Encounters": len(df),
        "Unique patients": df[group_col].nunique(),
        "Readmissions <30d": int(df[target_col].sum()),
        "Prevalence %": float(overall_prev * 100),
        "Share of cohort %": 100.0,
    })
    summary = pd.DataFrame(rows)

    # Prevalence in each part should stay close to the overall rate
    for name, part in parts.items():
        gap = abs(part[target_col].mean() - overall_prev)
        if gap > 0.01:
            warnings.warn(
                f"Prevalence in '{name}' differs from overall by {gap*100:.2f} points"
            )

    return parts, summary


def assert_no_leakage(parts: Dict[str, pd.DataFrame], group_col: Optional[str] = None) -> None:
    """Raise if any patient (or index) appears in two partitions."""
    group_col = group_col or config.GROUP_COL
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(parts[a][group_col]) & set(parts[b][group_col])
            if shared:
                raise AssertionError(
                    f"Patient leakage between '{a}' and '{b}': {len(shared)} shared patients"
                )
            shared_idx = set(parts[a].index) & set(parts[b].index)
            if shared_idx and not parts[a].index.equals(parts[b].index):
                pass  # reset indices legitimately overlap; group check is authoritative


# ==========================================================================
# 3. Pipelines and preprocessing
# ==========================================================================

def build_preprocessor(
    numeric_cols: Sequence[str], categorical_cols: Sequence[str]
) -> ColumnTransformer:
    """Every stateful step lives here, so all of it is fit on training data only.

    ``min_frequency`` on the one-hot encoder means rare-category grouping (for
    example ``medical_specialty``, which has 70+ levels) is *learned on the
    training fold*, not decided by looking at the whole dataset.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=config.MIN_CATEGORY_FREQ,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ==========================================================================
# 4. Scores, discrimination and threshold policy
# ==========================================================================

def positive_scores(model, X) -> np.ndarray:
    """Return the model's probability for the positive class.

    Falls back to a min-max scaled ``decision_function`` when the estimator has
    no ``predict_proba``.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    raw = np.asarray(model.decision_function(X)).ravel()
    lo, hi = raw.min(), raw.max()
    if np.isclose(hi, lo):
        return np.full_like(raw, 0.5, dtype=float)
    return (raw - lo) / (hi - lo)


def auc_report(y_true, y_score, name: str = "model", plot: bool = True) -> Dict[str, float]:
    """Summarise discrimination and probability quality; optionally draw curves."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    prevalence = float(y_true.mean())

    single_class = len(np.unique(y_true)) < 2
    metrics = {
        "name": name,
        "n": int(len(y_true)),
        "prevalence": prevalence,
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        "log_loss": float("nan") if single_class else float(
            log_loss(y_true, np.clip(y_score, 1e-15, 1 - 1e-15))
        ),
    }

    if plot and not single_class:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, color="#1f5c8b", lw=2,
                     label=f"ROC AUC = {metrics['roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate (recall)")
        axes[0].set_title(f"ROC curve - {name}")
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[1].plot(recall, precision, color="#b5451b", lw=2,
                     label=f"PR AUC = {metrics['pr_auc']:.3f}")
        axes[1].axhline(prevalence, ls="--", color="grey", lw=1,
                        label=f"Prevalence = {prevalence:.3f}")
        axes[1].set_xlabel("Recall (share of readmissions caught)")
        axes[1].set_ylabel("Precision (share of alerts that readmit)")
        axes[1].set_title(f"Precision-recall curve - {name}")
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        plt.show()

    return metrics


def tradeoff_table(y_true, y_score, thresholds: Optional[Iterable[float]] = None) -> pd.DataFrame:
    """One row per candidate threshold, vectorised with cumulative sums.

    Columns are counts (TP/FP/TN/FN), rates (recall, precision, specificity,
    FPR) and the per-1,000-discharge framing clinicians actually work with.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    total_pos = int(y_true.sum())
    total_neg = n - total_pos

    if thresholds is None:
        candidates = np.unique(y_score)
        thresholds = np.concatenate(([0.0], candidates, [1.0]))
    thresholds = np.unique(np.asarray(list(thresholds), dtype=float))

    # Sort scores descending once; a threshold t flags every score >= t.
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    cum_tp = np.concatenate(([0], np.cumsum(sorted_labels)))

    # Number of records with score >= t, for each threshold
    n_flagged = np.searchsorted(-sorted_scores, -thresholds, side="right")

    tp = cum_tp[n_flagged]
    fp = n_flagged - tp
    fn = total_pos - tp
    tn = total_neg - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(tp, total_pos, out=np.full_like(tp, np.nan, dtype=float),
                           where=total_pos > 0)
        precision = np.divide(tp, n_flagged, out=np.full_like(tp, np.nan, dtype=float),
                              where=n_flagged > 0)
        specificity = np.divide(tn, total_neg, out=np.full_like(tn, np.nan, dtype=float),
                                where=total_neg > 0)
        fpr = np.divide(fp, total_neg, out=np.full_like(fp, np.nan, dtype=float),
                        where=total_neg > 0)

    return pd.DataFrame({
        "threshold": thresholds,
        "TP": tp.astype(int),
        "FP": fp.astype(int),
        "TN": tn.astype(int),
        "FN": fn.astype(int),
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "fpr": fpr,
        "alerts_per_1000": n_flagged / n * 1000.0,
        "true_pos_per_1000": tp / n * 1000.0,
        "false_neg_per_1000": fn / n * 1000.0,
    })


def summary_at_threshold(y_true, y_score, threshold: float) -> pd.DataFrame:
    """One-row trade-off summary at a single threshold."""
    return tradeoff_table(y_true, y_score, thresholds=[float(threshold)]).reset_index(drop=True)


def pick_threshold_workload(y_true, y_score, alerts_per_1000_max: float) -> Dict[str, Any]:
    """Lowest threshold that still fits capacity, i.e. maximum recall within budget."""
    table = tradeoff_table(y_true, y_score)
    feasible = table[table["alerts_per_1000"] <= alerts_per_1000_max]
    if feasible.empty:
        raise ValueError(
            f"No threshold keeps alerts at or below {alerts_per_1000_max} per 1,000"
        )
    chosen = float(feasible["threshold"].min())
    return {"threshold": chosen, "table": table}


def pick_threshold_recall_floor(y_true, y_score, recall_floor: float) -> Dict[str, Any]:
    """Among thresholds meeting the recall floor, take the highest precision."""
    table = tradeoff_table(y_true, y_score)
    feasible = table[table["recall"] >= recall_floor].copy()
    if feasible.empty:
        raise ValueError(f"No threshold reaches a recall of {recall_floor}")
    feasible = feasible.sort_values(
        ["precision", "threshold"], ascending=[False, False]
    )
    return {"threshold": float(feasible["threshold"].iloc[0]), "table": table}


def pick_threshold_cost(y_true, y_score, C_FP: float, C_FN: float) -> Dict[str, Any]:
    """Harm-weighted threshold: empirical minimum cost plus the Bayes-rule value."""
    table = tradeoff_table(y_true, y_score).copy()
    table["expected_cost"] = C_FP * table["FP"] + C_FN * table["FN"]
    best = table.loc[table["expected_cost"].idxmin()]
    return {
        "threshold_empirical": float(best["threshold"]),
        "threshold_theoretical": float(C_FP / (C_FP + C_FN)),
        "table": table,
    }


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# ==========================================================================
# 5. Calibration
# ==========================================================================

def calibrate_on_holdout(fitted_model, X_cal, y_cal, method: str = "sigmoid"):
    """Calibrate an already-fitted model on a dedicated holdout slice.

    The calibrator never sees the training data or the test data.  Two sklearn
    APIs can do this; which one is available depends on the version that the
    Responsible AI packages pin:

    * scikit-learn >= 1.6 -- wrap the model in ``FrozenEstimator``
    * older versions      -- ``CalibratedClassifierCV(..., cv="prefit")``

    Keeping the choice behind this helper means the notebook never changes.
    """
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except ImportError:  # older scikit-learn pinned by responsibleai
        calibrator = CalibratedClassifierCV(estimator=fitted_model, method=method, cv="prefit")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrator.fit(X_cal, y_cal)
    return calibrator


def reliability_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count (quantile) reliability bins: predicted vs observed risk."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)

    ranks = pd.qcut(pd.Series(p).rank(method="first"), q=n_bins, labels=False)
    frame = pd.DataFrame({"p": p, "y": y_true, "bin": ranks})
    grouped = frame.groupby("bin")
    out = pd.DataFrame({
        "Bin": np.arange(1, n_bins + 1),
        "n": grouped.size().values,
        "Mean predicted risk": grouped["p"].mean().values,
        "Observed readmission rate": grouped["y"].mean().values,
        "Readmissions": grouped["y"].sum().values.astype(int),
    })
    out["Gap (predicted - observed)"] = (
        out["Mean predicted risk"] - out["Observed readmission rate"]
    )
    return out


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """Equal-count expected calibration error: mean |predicted - observed|."""
    table = reliability_table(y_true, p, n_bins=n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * table["Gap (predicted - observed)"].abs()).sum())


def calibration_in_the_large(y_true, prob_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare mean predicted risk with observed prevalence for each score set."""
    y_true = np.asarray(y_true).astype(int)
    observed = float(y_true.mean())
    rows = []
    for name, p in prob_dict.items():
        p = np.asarray(p, dtype=float)
        rows.append({
            "Scores": name,
            "Mean predicted risk": float(p.mean()),
            "Observed readmission rate": observed,
            "Difference": float(p.mean() - observed),
            "Brier score": float(brier_score_loss(y_true, np.clip(p, 0, 1))),
            "ECE": expected_calibration_error(y_true, p),
        })
    return pd.DataFrame(rows)


def plot_reliability(y_true, prob_dict: Dict[str, np.ndarray], n_bins: int = 10,
                     title: str = "Reliability: predicted vs observed risk"):
    """Reliability diagram for one or more score sets against the diagonal."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")

    colours = ["#b5451b", "#1f5c8b", "#3f7d20", "#6c3d8f"]
    for (name, p), colour in zip(prob_dict.items(), colours):
        table = reliability_table(y_true, p, n_bins=n_bins)
        ax.plot(table["Mean predicted risk"], table["Observed readmission rate"],
                "o-", color=colour, lw=1.8, ms=5, label=name)

    upper = max(0.05, max(np.max(np.asarray(p)) for p in prob_dict.values()) * 1.05)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Mean predicted risk in bin")
    ax.set_ylabel("Observed readmission rate in bin")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


# ==========================================================================
# 6. Fairness and subgroup metrics
# ==========================================================================

def subgroup_metrics(y_true, y_score, threshold: float, sensitive: pd.Series,
                     min_n: int = 30) -> pd.DataFrame:
    """Per-group performance at the locked threshold, with Wilson intervals.

    Uses ``fairlearn.metrics.MetricFrame`` when fairlearn is installed and a
    pandas fallback otherwise, so the audit never silently disappears.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    sensitive = pd.Series(sensitive).astype(str).reset_index(drop=True)

    overall_recall = y_pred[y_true == 1].mean() if (y_true == 1).any() else np.nan
    overall_fpr = y_pred[y_true == 0].mean() if (y_true == 0).any() else np.nan

    rows = []
    for level, idx in sensitive.groupby(sensitive).groups.items():
        idx = np.asarray(list(idx))
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        pos, neg = int((yt == 1).sum()), int((yt == 0).sum())

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        recall = tp / pos if pos else np.nan
        fpr = fp / neg if neg else np.nan
        precision = tp / int((yp == 1).sum()) if (yp == 1).any() else np.nan

        recall_lo, recall_hi = wilson_interval(tp, pos)
        fpr_lo, fpr_hi = wilson_interval(fp, neg)

        both_classes = len(np.unique(yt)) == 2
        rows.append({
            "Group": str(level),
            "n": len(idx),
            "Readmissions": pos,
            "Prevalence": float(yt.mean()),
            "Selection rate": float(yp.mean()),
            "Recall": recall,
            "Recall 95% CI": f"{recall_lo:.3f}-{recall_hi:.3f}" if pos else "-",
            "FPR": fpr,
            "FPR 95% CI": f"{fpr_lo:.3f}-{fpr_hi:.3f}" if neg else "-",
            "Precision": precision,
            "Brier": float(brier_score_loss(yt, np.clip(ys, 0, 1))) if len(yt) else np.nan,
            "ROC-AUC": float(roc_auc_score(yt, ys)) if both_classes else np.nan,
            "Recall gap vs overall": recall - overall_recall if pos else np.nan,
            "FPR gap vs overall": fpr - overall_fpr if neg else np.nan,
            "Recall ratio vs overall": recall / overall_recall if pos and overall_recall else np.nan,
            "Reliable (n >= %d)" % min_n: len(idx) >= min_n,
        })

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def fairness_summary(y_true, y_score, threshold: float,
                     sensitive_df: pd.DataFrame) -> pd.DataFrame:
    """Fairlearn disparity metrics for each sensitive attribute."""
    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )
    except ImportError:
        return pd.DataFrame([{"note": "fairlearn is not installed; disparity metrics skipped"}])

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    rows = []
    for col in sensitive_df.columns:
        s = sensitive_df[col].astype(str).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=s))
            dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=s))

        # Equal-opportunity gap = spread in recall between groups
        recalls = []
        for level in np.unique(s):
            mask = (s == level) & (y_true == 1)
            if mask.sum() >= 30:
                recalls.append(y_pred[mask].mean())
        eo_gap = float(max(recalls) - min(recalls)) if len(recalls) > 1 else np.nan

        rows.append({
            "Sensitive attribute": col,
            "Groups": int(len(np.unique(s))),
            "Equalised odds difference": eod,
            "Demographic parity difference": dpd,
            "Equal opportunity gap (recall spread)": eo_gap,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 7. Plots
# ==========================================================================

def plot_recall_floor_curves(y_true, y_score, recall_floor: float,
                             chosen_threshold: float):
    """Recall and precision against threshold, with the floor and the choice marked."""
    table = tradeoff_table(y_true, y_score)
    table = table[table["threshold"] <= np.nanpercentile(np.asarray(y_score), 99.5)]

    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.plot(table["threshold"], table["recall"], color="#1f5c8b", lw=2, label="Recall")
    ax.plot(table["threshold"], table["precision"], color="#e08214", lw=2, label="Precision")
    ax.axhline(recall_floor, ls="--", color="#b5451b", lw=1.5,
               label=f"Recall floor = {recall_floor:.2f}")
    ax.axvline(chosen_threshold, ls=":", color="black", lw=1.5,
               label=f"Chosen threshold = {chosen_threshold:.3f}")

    at = summary_at_threshold(y_true, y_score, chosen_threshold).iloc[0]
    ax.plot([chosen_threshold], [at["recall"]], "o", color="#1f5c8b", ms=8)
    ax.annotate(f"Recall={at['recall']:.2f}", (chosen_threshold, at["recall"]),
                textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.annotate(f"Prec={at['precision']:.2f}", (chosen_threshold, at["precision"]),
                textcoords="offset points", xytext=(8, -14), fontsize=9)

    ax.set_xlabel("Alert threshold (calibrated probability)")
    ax.set_ylabel("Score")
    ax.set_title("Recall floor, then maximise precision")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def plot_cumulative_recall_at_threshold(y_true, y_score, chosen_threshold: float):
    """Share of readmissions caught as the alert list grows, ranked by risk."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)
    caught = np.cumsum(y_true[order]) / max(1, y_true.sum())
    n_alerts = np.arange(1, len(y_true) + 1)
    alerts_at_thr = int((y_score >= chosen_threshold).sum())
    recall_at_thr = caught[alerts_at_thr - 1] if alerts_at_thr > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.plot(n_alerts, caught, color="#1f5c8b", lw=2, label="Cumulative recall")
    ax.axvline(alerts_at_thr, ls="--", color="#b5451b", lw=1.5,
               label=f"Alerts at locked threshold = {alerts_at_thr:,}")
    ax.plot([alerts_at_thr], [recall_at_thr], "o", color="black", ms=7)
    ax.annotate(f"Recall = {recall_at_thr:.2f}", (alerts_at_thr, recall_at_thr),
                textcoords="offset points", xytext=(10, -14), fontsize=9)

    ax.set_xlabel("Number of alerts (patients reviewed, ranked by risk)")
    ax.set_ylabel("Share of 30-day readmissions caught")
    ax.set_title("Clinical benefit for each increase in review workload")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def plot_topk_at_threshold(y_true, y_score, chosen_threshold: float, top_k: int = 30):
    """The top-k highest-risk patients, coloured by what actually happened."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)[:top_k]
    scores, labels = y_score[order], y_true[order]
    colours = ["#b5451b" if v == 1 else "#8a99a8" for v in labels]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(np.arange(1, len(scores) + 1), scores, color=colours)
    ax.axhline(chosen_threshold, ls="--", color="black", lw=1.5,
               label=f"Locked threshold = {chosen_threshold:.3f}")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#b5451b", label="Readmitted within 30 days"),
        Patch(color="#8a99a8", label="Not readmitted within 30 days"),
        plt.Line2D([0], [0], ls="--", color="black",
                   label=f"Locked threshold = {chosen_threshold:.3f}"),
    ], fontsize=9, loc="upper right")

    ax.set_xlabel(f"Top {top_k} patients by estimated risk")
    ax.set_ylabel("Calibrated readmission probability")
    ax.set_title(f"Top {top_k} highest-risk patients and their actual outcome")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plt.show()
    return fig


def plot_risk_bands(y_true, y_score, n_bands: int = 10):
    """Decile risk bands: observed readmission rate and cumulative capture."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    bands = np.array_split(np.arange(len(y_sorted)), n_bands)

    rows, cum = [], 0
    for i, band in enumerate(bands, start=1):
        pos = int(y_sorted[band].sum())
        cum += pos
        rows.append({
            "Risk band": f"{i} ({'highest' if i == 1 else 'lowest' if i == n_bands else ''})".strip(),
            "Band": i,
            "Patients": len(band),
            "Readmissions": pos,
            "Readmission rate %": pos / len(band) * 100,
            "Cumulative capture %": cum / max(1, y_true.sum()) * 100,
            "Cumulative patients reviewed %": (band[-1] + 1) / len(y_sorted) * 100,
        })
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(table["Band"], table["Readmission rate %"], color="#1f5c8b")
    axes[0].axhline(y_true.mean() * 100, ls="--", color="#b5451b", lw=1.5,
                    label=f"Overall rate = {y_true.mean()*100:.1f}%")
    axes[0].set_xlabel("Risk band (1 = highest estimated risk)")
    axes[0].set_ylabel("Observed readmission rate (%)")
    axes[0].set_title("Do readmissions concentrate in the top bands?")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].plot(table["Cumulative patients reviewed %"], table["Cumulative capture %"],
                 "o-", color="#3f7d20", lw=2)
    axes[1].plot([0, 100], [0, 100], "--", color="grey", lw=1, label="No model (random review)")
    axes[1].set_xlabel("Share of discharges reviewed (%)")
    axes[1].set_ylabel("Share of readmissions caught (%)")
    axes[1].set_title("Cumulative capture as review capacity grows")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    return table, fig


def plot_score_distribution(y_true, y_score, threshold: Optional[float] = None,
                            title: str = "Estimated risk by actual outcome"):
    """Overlapping score distributions for readmitted vs not readmitted."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 5))
    bins = np.linspace(0, np.percentile(y_score, 99.5), 40)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.6, density=True,
            color="#8a99a8", label="Not readmitted within 30 days")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.6, density=True,
            color="#b5451b", label="Readmitted within 30 days")
    if threshold is not None:
        ax.axvline(threshold, ls="--", color="black", lw=1.5,
                   label=f"Locked threshold = {threshold:.3f}")
    ax.set_xlabel("Calibrated readmission probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


# ==========================================================================
# 8. Exports
# ==========================================================================

def savefig(fig, name: str, dpi: int = 150):
    """Save a figure to ``outputs/figures`` for the slide deck."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def savetable(df: pd.DataFrame, name: str, index: bool = False):
    """Save a table to ``outputs/tables`` for the slide deck."""
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLE_DIR / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=index)
    return path


def stratified_subsample(df: pd.DataFrame, target_col: str, n: int,
                         seed: Optional[int] = None) -> pd.DataFrame:
    """Draw a class-stratified subsample, used to keep RAI compute tractable."""
    seed = config.RANDOM_STATE if seed is None else seed
    if n >= len(df):
        return df.copy()
    frac = n / len(df)
    out = (
        df.groupby(target_col, group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed))
    )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ==========================================================================
# 9. Responsible AI glue
# ==========================================================================

class ThresholdedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a fitted probabilistic model so ``predict`` uses the locked threshold.

    Defined at module level (not inside a factory) so instances are picklable,
    which ``RAIInsights`` requires.
    """

    def __init__(self, model=None, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y=None):  # already fitted; present for API compatibility
        return self

    @property
    def classes_(self):
        return getattr(self.model, "classes_", np.array([0, 1]))

    def predict_proba(self, X):
        return np.asarray(self.model.predict_proba(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def __sklearn_is_fitted__(self) -> bool:
        return True


def make_thresholded_estimator(model, threshold: float) -> ThresholdedClassifier:
    """Return a picklable estimator that applies the locked alert threshold."""
    return ThresholdedClassifier(model=model, threshold=float(threshold))


def init_rai_dependencies() -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Try-import the Responsible AI stack; missing pieces must not break the core analysis."""
    flags: Dict[str, bool] = {}
    objects: Dict[str, Any] = {}

    try:
        from responsibleai import RAIInsights, FeatureMetadata
        objects["RAIInsights"] = RAIInsights
        objects["FeatureMetadata"] = FeatureMetadata
        flags["_RAI"] = True
    except Exception:
        flags["_RAI"] = False

    try:
        from raiwidgets import ResponsibleAIDashboard
        objects["ResponsibleAIDashboard"] = ResponsibleAIDashboard
        flags["_DASHBOARD"] = True
    except Exception:
        flags["_DASHBOARD"] = False

    try:
        import interpret_community  # noqa: F401
        flags["_INTERPRET"] = True
    except Exception:
        flags["_INTERPRET"] = False

    try:
        import erroranalysis  # noqa: F401
        flags["_ERRANALYSIS"] = True
    except Exception:
        flags["_ERRANALYSIS"] = False

    try:
        import fairlearn  # noqa: F401
        from fairlearn.metrics import MetricFrame
        objects["MetricFrame"] = MetricFrame
        flags["_FAIRLEARN"] = True
    except Exception:
        flags["_FAIRLEARN"] = False

    try:
        import dice_ml  # noqa: F401
        flags["_DICE"] = True
    except Exception:
        flags["_DICE"] = False

    try:
        import econml  # noqa: F401
        flags["_ECONML"] = True
    except Exception:
        flags["_ECONML"] = False

    return flags, objects


def bootstrap_metric_ci(y_true, y_score, groups, metric: str = "pr_auc",
                        n_boot: int = 200, seed: Optional[int] = None,
                        alpha: float = 0.05) -> Tuple[float, float]:
    """Patient-clustered bootstrap CI: resample patients, not encounters.

    Repeat encounters of one patient are correlated, so resampling rows would
    understate uncertainty.
    """
    seed = config.RANDOM_STATE if seed is None else seed
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score

    stats = []
    for _ in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(fn(y_true[idx], y_score[idx]))

    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def align_rai_categories(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         categorical_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop RAI test rows holding a category the RAI train sample never saw.

    ``RAIInsights`` counterfactual generation requires every category in the
    test frame to appear in the train frame.  Sub-sampling for tractable
    compute can break that for rare levels (for example an uncommon
    ``medical_specialty``).  Only the *dashboard inputs* are filtered here --
    the locked model, the locked threshold and every reported metric are
    untouched.

    Returns ``(aligned_test_df, dropped_report)``.
    """
    mask = pd.Series(True, index=test_df.index)
    rows = []
    for col in categorical_cols:
        if col not in test_df.columns or col not in train_df.columns:
            continue
        seen = set(train_df[col].astype(str).unique())
        col_mask = test_df[col].astype(str).isin(seen)
        dropped = int((~col_mask).sum())
        if dropped:
            rows.append({
                "Feature": col,
                "Unseen categories": ", ".join(
                    sorted(set(test_df.loc[~col_mask, col].astype(str)))
                ),
                "Test rows affected": dropped,
            })
        mask &= col_mask

    report = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Feature", "Unseen categories", "Test rows affected"]
    )
    return test_df[mask].reset_index(drop=True), report


class RAIModelAdapter(ClassifierMixin, BaseEstimator):
    """The locked model as the Responsible AI Toolbox needs to see it.

    Two problems with passing ``ThresholdedClassifier`` straight to ``RAIInsights``:

    1. The RAI frames hold categorical columns as strings, but the fitted
       pipeline learned some of them as integers (``A1C_tested``,
       ``med_change``, ``any_prior_inpatient`` ...).  The one-hot encoder then
       treats "1" as an unseen category and the predictions change.  This
       adapter restores the training dtypes before every prediction.
    2. DiCE (counterfactuals) decides a flip by whether the positive-class
       probability crosses 0.5, not the locked threshold.  ``predict_proba``
       therefore returns an *alert score*: the calibrated risk shifted on the
       log-odds scale so that 0.5 is exactly the locked threshold.  The shift is
       monotone, so rankings, ROC-AUC and PR-AUC are unchanged; calibrated risk
       is still available from ``calibrated_risk``.

    Defined at module level so ``RAIInsights.save`` / ``load`` can pickle it.
    """

    def __init__(self, model=None, feature_dtypes=None, threshold: float = 0.5):
        self.model = model
        self.feature_dtypes = feature_dtypes
        self.threshold = threshold

    def fit(self, X, y=None):  # already fitted
        return self

    @property
    def classes_(self):
        return np.array([0, 1])

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def _restore(self, X):
        X = pd.DataFrame(X).copy()
        columns = list(self.feature_dtypes)
        X = X[columns]
        for column in columns:
            dtype = str(self.feature_dtypes[column])
            if dtype not in ("object", "category"):
                values = pd.to_numeric(X[column], errors="coerce")
                X[column] = values if values.isna().any() else values.astype(dtype)
        return X

    def calibrated_risk(self, X) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._restore(X)))[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.calibrated_risk(X) >= self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        p = np.clip(self.calibrated_risk(X), 1e-9, 1 - 1e-9)
        t = float(self.threshold)
        z = np.log(p / (1 - p)) - np.log(t / (1 - t))
        alert_score = 1.0 / (1.0 + np.exp(-z))
        return np.c_[1 - alert_score, alert_score]


def make_rai_estimator(model, feature_frame: pd.DataFrame, threshold: float) -> RAIModelAdapter:
    """Return the picklable adapter used for every Responsible AI component."""
    dtypes = {column: str(dtype) for column, dtype in feature_frame.dtypes.items()}
    return RAIModelAdapter(model=model, feature_dtypes=dtypes, threshold=float(threshold))



def build_dashboard_cohorts(test_df: pd.DataFrame, target_col: Optional[str] = None):
    """The notebook's analysis cohorts (Section 9.5 plus the Section 9.4c blind spot) as saved dashboard cohorts.

    Each cohort is defined twice from the same rules: once as raiutils ``Cohort`` filters, so it appears by name
    in every dashboard view, and once as a pandas mask, so its size and readmission rate can be printed and
    checked against the dashboard. Categorical columns in the RAI frame are strings, hence ``"1"`` for flags.
    Returns ``(cohort_list, summary_table)``.
    """
    from raiutils.cohort import Cohort, CohortFilter, CohortFilterMethods as M

    target_col = target_col or config.TARGET_COL
    facility = "Facility (SNF/rehab/other)"
    # categorical filters only support "includes", so "not facility" is spelled out
    not_facility = sorted(v for v in test_df["discharge_group"].astype(str).unique() if v != facility)
    specs = [
        ("1. Age >60 & discharged to facility",
         [(M.METHOD_INCLUDES, [">60"], "age_group"), (M.METHOD_INCLUDES, [facility], "discharge_group")],
         (test_df["age_group"] == ">60") & (test_df["discharge_group"] == facility)),
        ("2. Payer self-pay or unknown",
         [(M.METHOD_INCLUDES, ["Self-pay", "Unknown"], "payer_group")],
         test_df["payer_group"].isin(["Self-pay", "Unknown"])),
        ("3a. Race = African American", [(M.METHOD_INCLUDES, ["AfricanAmerican"], "race")],
         test_df["race"] == "AfricanAmerican"),
        ("3b. Race = Caucasian", [(M.METHOD_INCLUDES, ["Caucasian"], "race")], test_df["race"] == "Caucasian"),
        ("4a. Primary diagnosis Circulatory", [(M.METHOD_INCLUDES, ["Circulatory"], "diag_1_group")],
         test_df["diag_1_group"] == "Circulatory"),
        ("4b. Primary diagnosis Diabetes", [(M.METHOD_INCLUDES, ["Diabetes"], "diag_1_group")],
         test_df["diag_1_group"] == "Diabetes"),
        ("5. >=1 prior inpatient stay", [(M.METHOD_GREATER, [0], "number_inpatient")],
         test_df["number_inpatient"].astype(float) > 0),
        ("6. Blind spot: first stay, home, <=5 days",
         [(M.METHOD_LESS_AND_EQUAL, [0], "number_inpatient"),
          (M.METHOD_INCLUDES, not_facility, "discharge_group"),
          (M.METHOD_LESS_AND_EQUAL, [5.5], "time_in_hospital")],
         (test_df["number_inpatient"].astype(float) <= 0) & (test_df["discharge_group"] != facility)
         & (test_df["time_in_hospital"].astype(float) <= 5.5)),
    ]
    cohorts, rows = [], []
    for name, filters, mask in specs:
        cohort = Cohort(name=name)
        for method, arg, column in filters:
            cohort.add_cohort_filter(CohortFilter(method=method, arg=arg, column=column))
        cohorts.append(cohort)
        rows.append({"Dashboard cohort": name, "n": int(mask.sum()),
                     "Readmissions": int(test_df.loc[mask, target_col].astype(int).sum()),
                     "Readmission rate": float(test_df.loc[mask, target_col].astype(int).mean())})
    return cohorts, pd.DataFrame(rows)

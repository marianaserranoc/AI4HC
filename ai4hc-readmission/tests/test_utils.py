"""Tests for the helper functions in utils.py.

Run from the repository root:   pytest -q
The tests that need the real dataset are skipped when data/raw/diabetic_data.csv is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import utils  # noqa: E402

HAS_DATA = config.RAW_DATA_PATH.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="data/raw/diabetic_data.csv not present")


# ---------------------------------------------------------------------------
# Feature engineering and feature lists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code, group", [
    ("250.83", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"), ("486", "Respiratory"),
    ("786", "Respiratory"), ("562", "Digestive"), ("820", "Injury"), ("715", "Musculoskeletal"),
    ("599", "Genitourinary"), ("174", "Neoplasms"), ("V45", "Other"), ("E888", "Other"),
    (None, "Missing"), (np.nan, "Missing"), ("?", "Missing"),
])
def test_icd9_grouping(code, group):
    assert utils.icd9_to_group(code) == group


def _toy_encounters(n=6):
    rng = np.random.default_rng(0)
    row = {
        "encounter_id": range(n), "patient_nbr": [1, 1, 2, 3, 4, 5][:n],
        "race": "Caucasian", "gender": "Female", "age": "[70-80)",
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
        "time_in_hospital": 3, "payer_code": ["MC", "MD", "SP", np.nan, "BC", "?"][:n],
        "medical_specialty": np.nan, "num_lab_procedures": 40, "num_procedures": 1,
        "num_medications": 12, "number_outpatient": 0, "number_emergency": 1,
        "number_inpatient": [0, 2, 0, 1, 0, 3][:n],
        "diag_1": "250.01", "diag_2": "428", "diag_3": "V45", "number_diagnoses": 7,
        "max_glu_serum": "None", "A1Cresult": ["None", ">8", "Norm", "None", ">7", "None"][:n],
        "change": ["No", "Ch", "No", "Ch", "No", "No"][:n], "diabetesMed": "Yes",
        "readmitted": ["NO", "<30", ">30", "<30", "NO", "NO"][:n],
    }
    df = pd.DataFrame(row)
    for drug in config.DRUG_COLS:
        df[drug] = rng.choice(["No", "Steady", "Up", "Down"], n)
    return df


def test_engineer_features_is_row_wise_and_correct():
    df = _toy_encounters()
    out = utils.engineer_features(df)
    # target: only "<30" is positive
    assert out[config.TARGET_COL].tolist() == [0, 1, 0, 1, 0, 0]
    # "None" means the HbA1c test was not performed
    assert out["A1C_tested"].tolist() == [0, 1, 1, 0, 1, 0]
    assert out["med_change"].tolist() == [0, 1, 0, 1, 0, 0]
    assert out["any_prior_inpatient"].tolist() == [0, 1, 0, 1, 0, 1]
    assert out["age_years"].eq(75).all() and out["age_group"].eq(">60").all()
    assert out["payer_group"].tolist() == ["Medicare", "Medicaid", "Self-pay", "Unknown", "Private/Other", "Unknown"]
    # stateless: each row's features do not depend on the other rows
    single = utils.engineer_features(df.iloc[[3]])
    pd.testing.assert_frame_equal(single.reset_index(drop=True), out.iloc[[3]].reset_index(drop=True))


def test_identifiers_and_audit_columns_are_never_features():
    df = utils.engineer_features(_toy_encounters())
    lists = utils.get_feature_lists(df)
    features = set(lists["numeric_cols"]) | set(lists["categorical_cols"])
    for forbidden in ["encounter_id", "patient_nbr", "race", "gender", "readmitted", config.TARGET_COL]:
        assert forbidden not in features, forbidden


# ---------------------------------------------------------------------------
# Splitting and leakage
# ---------------------------------------------------------------------------

def test_assert_no_leakage_raises_on_shared_patient():
    parts = {"train": pd.DataFrame({"patient_nbr": [1, 2]}), "test": pd.DataFrame({"patient_nbr": [2, 3]})}
    with pytest.raises(AssertionError):
        utils.assert_no_leakage(parts, "patient_nbr")


def test_grouped_split_has_no_shared_patients():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"patient_nbr": rng.integers(0, 400, 2000)})
    df[config.TARGET_COL] = rng.random(2000) < 0.11
    parts, summary = utils.grouped_train_cal_val_test_split(df, "patient_nbr", config.TARGET_COL)
    ids = [set(p["patient_nbr"]) for p in parts.values()]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j]
    assert sum(len(p) for p in parts.values()) == len(df)


# ---------------------------------------------------------------------------
# Metrics and threshold policy
# ---------------------------------------------------------------------------

def test_tradeoff_table_counts():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    row = utils.tradeoff_table(y, s, thresholds=[0.35]).iloc[0]
    assert (row.TP, row.FP, row.FN, row.TN) == (2, 1, 0, 2)
    assert row.alerts_per_1000 == pytest.approx(600.0)
    assert row.recall == pytest.approx(1.0) and row.precision == pytest.approx(2 / 3)


def test_workload_threshold_respects_capacity():
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.11).astype(int)
    s = rng.random(5000) * 0.3 + 0.2 * y
    thr = utils.pick_threshold_workload(y, s, 250)["threshold"]
    assert (s >= thr).mean() * 1000 <= 250


def test_wilson_interval_contains_estimate():
    lo, hi = utils.wilson_interval(48, 114)
    assert 0 <= lo < 48 / 114 < hi <= 1
    assert all(np.isnan(utils.wilson_interval(0, 0)))


# ---------------------------------------------------------------------------
# Responsible AI glue
# ---------------------------------------------------------------------------

def test_rai_adapter_restores_dtypes_and_aligns_threshold():
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"flag": rng.integers(0, 2, 500), "grp": rng.choice(["a", "b"], 500)})
    y = ((X["flag"] == 1) & (rng.random(500) < 0.6)).astype(int)
    model = Pipeline([("prep", ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ["flag", "grp"])])),
                      ("lr", LogisticRegression())]).fit(X, y)
    thr = 0.2
    adapter = utils.make_rai_estimator(model, X, thr)

    X_str = X.astype(str)           # how the RAI frames store categoricals
    X_str["race"] = "x"             # audit-only column must be ignored
    expected = (model.predict_proba(X)[:, 1] >= thr).astype(int)
    assert np.array_equal(adapter.predict(X_str), expected)
    assert np.array_equal(adapter.predict_proba(X_str).argmax(axis=1), expected)


def test_dashboard_cohorts_match_masks():
    pytest.importorskip("raiutils")
    test_df = pd.DataFrame({
        "age_group": [">60", ">60", "30-60"], "discharge_group": ["Facility (SNF/rehab/other)", "Home", "Home"],
        "payer_group": ["Unknown", "Medicare", "Self-pay"], "race": ["Caucasian", "AfricanAmerican", "Caucasian"],
        "diag_1_group": ["Diabetes", "Circulatory", "Other"], "number_inpatient": [0, 2, 0],
        "time_in_hospital": [3, 8, 5], config.TARGET_COL: [1, 0, 1],
    })
    cohorts, table = utils.build_dashboard_cohorts(test_df)
    assert len(cohorts) == len(table) == 8
    sizes = dict(zip(table["Dashboard cohort"], table["n"]))
    assert sizes["1. Age >60 & discharged to facility"] == 1
    assert sizes["2. Payer self-pay or unknown"] == 2
    assert sizes["6. Blind spot: first stay, home, <=5 days"] == 1


# ---------------------------------------------------------------------------
# Real data (skipped without the CSV)
# ---------------------------------------------------------------------------

@needs_data
def test_load_raw_keeps_none_and_cohort_size():
    raw = utils.load_raw()
    assert "None" in set(raw["A1Cresult"])
    cohort, flow = utils.apply_cohort_rules(raw)
    assert len(cohort) == 99_340 and cohort["patient_nbr"].nunique() == 69_987


@needs_data
def test_real_split_is_leak_free_and_reproducible():
    cohort, _ = utils.apply_cohort_rules(utils.load_raw())
    df = utils.engineer_features(cohort)
    a, _ = utils.grouped_train_cal_val_test_split(df)
    b, _ = utils.grouped_train_cal_val_test_split(df)
    utils.assert_no_leakage(a)
    assert all(a[k]["encounter_id"].equals(b[k]["encounter_id"]) for k in a)
    assert len(a["test"]) == 14_885

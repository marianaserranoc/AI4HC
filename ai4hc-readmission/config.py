"""Central configuration for the AI4HC 30-day readmission project.

Every constant that controls the analysis lives here so the notebook never
hard-codes a path, a seed or a policy value.  Paths are built with ``pathlib``
relative to this file, so the project runs from any working directory.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = ROOT / "data" / "raw" / "diabetic_data.csv"
IDS_MAPPING_PATH = ROOT / "data" / "raw" / "IDs_mapping.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
RAI_DIR = OUTPUT_DIR / "rai_insights"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Target and cohort definition
# --------------------------------------------------------------------------
TARGET_COL = "readmit_30d"
GROUP_COL = "patient_nbr"

#: Discharge dispositions that end in death or hospice; such encounters cannot
#: be meaningfully readmitted, so they are removed (as in Strack et al. 2014).
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

#: Columns dropped by name before modelling.  These are stateless, rule-based
#: decisions justified by the data dictionary, not by fitting on the data.
DROP_COLS = [
    "encounter_id",              # identifier
    "weight",                    # 96.9% missing
    "examide",                   # constant ("No" everywhere)
    "citoglipton",               # constant ("No" everywhere)
    "acetohexamide",             # ~0% non-"No"
    "troglitazone",              # ~0% non-"No"
    "glimepiride-pioglitazone",  # ~0% non-"No"
    "metformin-rosiglitazone",   # ~0% non-"No"
    "metformin-pioglitazone",    # ~0% non-"No"
]

#: Diabetes drug columns retained as features after DROP_COLS is applied.
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "tolazamide", "insulin", "glyburide-metformin",
    "glipizide-metformin",
]

# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
#: Patient-grouped split.  A separate calibration slice keeps the calibrator
#: off the training data, and the validation slice carries threshold choice.
SPLIT_SIZES = {"train": 0.60, "cal": 0.10, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
SENSITIVE_COLS = ["race", "gender", "age_group", "payer_group"]

#: Race and gender are audited but not used as model inputs (see README).
USE_RACE_GENDER_AS_FEATURES = False
#: Payer group is a socioeconomic proxy; the notebook runs both variants.
USE_PAYER_AS_FEATURE = True

#: OneHotEncoder min_frequency.  Rare-category grouping is therefore learned
#: on the training fold only -- a key leakage control.
MIN_CATEGORY_FREQ = 100

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "age_years",
    "total_prior_visits", "n_meds_up", "n_meds_down",
    "n_diabetes_meds_prescribed",
]

CATEGORICAL_COLS = [
    "admission_group", "admission_source_group", "discharge_group",
    "medical_specialty", "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "A1C_tested", "glu_tested", "med_change",
    "on_diabetes_med", "any_prior_inpatient", "insulin", "metformin",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone",
]

# --------------------------------------------------------------------------
# Threshold policy  [fixed BEFORE the test set is opened]
# --------------------------------------------------------------------------
MAX_ALERTS_PER_1000 = 250
MIN_VALIDATION_RECALL = 0.50
FP_HARM_POINTS = 1.0
FN_HARM_RATIOS = [2.0, 5.0, 10.0]

#: Fairness tolerance: subgroup recall / FPR gap vs. overall, in points.
FAIRNESS_TOLERANCE = 0.10

# --------------------------------------------------------------------------
# Responsible AI
# --------------------------------------------------------------------------
RAI_TRAIN_SAMPLE = 5000  # [tunable] sized so rai_insights.compute() stays near ~10 minutes
RAI_TEST_SAMPLE = 2000   # [tunable] counterfactual generation dominates RAI runtime
RAI_TREATMENTS = ["A1C_tested", "med_change"]
RAI_HETEROGENEITY = ["diag_1_group", "age_group"]
RAI_CF_FEATURES_TO_VARY = [
    "A1C_tested", "med_change", "insulin", "discharge_group", "num_medications",
]

# --------------------------------------------------------------------------
# ID code groupings (UCI IDs_mapping.csv)
# --------------------------------------------------------------------------
ADMISSION_TYPE_GROUPS = {
    1: "Emergency/Urgent", 2: "Emergency/Urgent", 7: "Emergency/Urgent",
    3: "Elective",
    4: "Other/Unknown", 5: "Other/Unknown", 6: "Other/Unknown", 8: "Other/Unknown",
}

ADMISSION_SOURCE_GROUPS = {
    7: "Emergency room",
    1: "Referral", 2: "Referral", 3: "Referral",
    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer",
    18: "Transfer", 22: "Transfer", 25: "Transfer", 26: "Transfer",
}

DISCHARGE_GROUPS = {
    1: "Home",
    6: "Home with health service", 8: "Home with health service",
    3: "Facility (SNF/rehab/other)", 2: "Facility (SNF/rehab/other)",
    22: "Facility (SNF/rehab/other)", 5: "Facility (SNF/rehab/other)",
    4: "Facility (SNF/rehab/other)", 23: "Facility (SNF/rehab/other)",
    24: "Facility (SNF/rehab/other)", 9: "Facility (SNF/rehab/other)",
    10: "Facility (SNF/rehab/other)", 15: "Facility (SNF/rehab/other)",
    16: "Facility (SNF/rehab/other)", 17: "Facility (SNF/rehab/other)",
    27: "Facility (SNF/rehab/other)", 28: "Facility (SNF/rehab/other)",
    29: "Facility (SNF/rehab/other)", 30: "Facility (SNF/rehab/other)",
    7: "Left against medical advice",
    12: "Other/Unknown", 18: "Other/Unknown", 25: "Other/Unknown",
    26: "Other/Unknown",
}

PAYER_GROUPS = {
    "MC": "Medicare", "MD": "Medicaid",
    "SP": "Self-pay",
}

#: Plain-English column labels used when a trade-off table is displayed.
THRESHOLD_DISPLAY_LABELS = {
    "threshold": "Alert threshold",
    "TP": "Readmissions correctly flagged",
    "FP": "Alerts on non-readmitted patients",
    "TN": "Correctly not flagged",
    "FN": "Missed readmissions",
    "recall": "Recall (share of readmissions caught)",
    "precision": "Precision (share of alerts that readmit)",
    "specificity": "Specificity",
    "fpr": "False positive rate",
    "alerts_per_1000": "Alerts per 1,000 discharges",
    "true_pos_per_1000": "Readmissions caught per 1,000",
    "false_neg_per_1000": "Readmissions missed per 1,000",
}

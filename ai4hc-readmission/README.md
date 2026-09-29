# Predicting 30-Day Readmission Risk for Hospitalised Diabetic Patients

**AI for Healthcare (AI4HC) — Group Assignment 1 · Responsible AI Toolbox** · IE University, September 2026

## Introduction and purpose

About **1 in 9** diabetic inpatient stays ends with the patient readmitted within 30 days, and many of those returns
are considered preventable with better discharge planning. A transitional-care team cannot call every discharged
patient, so someone has to decide who gets the extra help.

**Clinical question.** At the moment of discharge, which diabetic inpatients are at highest risk of readmission
within 30 days, so that a nurse team with capacity for **250 reviews per 1,000 discharges** can prioritise them for
medication reconciliation, diabetes education and an early follow-up call?

The model is **decision support, not a diagnosis**: it orders a worklist, a clinician can always override it, and a
low score never means a patient is safe. The notebook audits the model end to end with the Microsoft Responsible AI
Toolbox (error analysis, fairness, feature importance, counterfactuals and causal analysis) and turns the findings
into clinician-ready recommendations, deployment risks and monitoring triggers.

## Project structure

```
ai4hc-readmission/
├── README.md                          this file
├── environment.yml                    conda environment, RAI stack pinned
├── diabetes_readmission_rai.ipynb     the full analysis, 10 sections, all outputs saved (keep it at the root)
├── utils.py                           helper functions (see below)
├── config.py                          every constant: paths, seed, features, policy values
├── .gitignore
├── data/raw/                          put diabetic_data.csv here (not tracked)
├── outputs/
│   ├── figures/                       every figure the notebook produces, for the slides
│   ├── tables/                        every table as CSV
│   ├── rai_screenshots/               screenshots of each dashboard view, from the notebook's saved analysis
│   └── rai_insights/                  the computed RAIInsights object, written by Section 9.3 (not tracked)
├── presentation/                      the final deck (.pptx) and its Canva link
└── tests/test_utils.py                pytest tests for utils.py
```

`utils.py` holds data loading and cohort rules, stateless feature engineering, the patient-grouped split and
leakage check, the preprocessing pipeline, metrics and threshold policies, calibration, fairness tables, plots,
exports, and the Responsible AI glue: the dashboard model adapter and the saved dashboard cohorts. Run the
notebook from the repository root, so `config.py` and `utils.py` import and `outputs/` lands in the right place.

## Install and environment

Prerequisites: conda or mamba, and Jupyter.

```bash
conda env create -f environment.yml
conda activate ai4hc_readmit
python -m ipykernel install --user --name ai4hc_readmit --display-name ai4hc_readmit
```

The Responsible AI Toolbox (`responsibleai` / `raiwidgets` 0.36) supports only an older scientific stack:
**Python 3.10, pandas < 2.0, numpy ≤ 1.26.2, scikit-learn ≤ 1.5.1, dice-ml 0.11**. `environment.yml` pins exactly
these. Do not install the toolbox with `pip install --no-deps` on top of a newer stack; the dashboard then fails
to render or computes wrong results.

**Run locally, not on Google Colab.** The dashboard is a small web server on the machine running the notebook, and
its `localhost` link only opens in a browser on that same machine. Colab also cannot install pandas < 2.0.

## Quickstart

1. Download `diabetic_data.csv` from the UCI repository (link below) and place it at `data/raw/diabetic_data.csv`.
2. Put `utils.py` and `config.py` in the same folder as the notebook.
3. Activate the environment and start Jupyter: `jupyter lab`.
4. Open `diabetes_readmission_rai.ipynb`, select the `ai4hc_readmit` kernel, and run all cells top to bottom.

A full run takes about 20 minutes, 13 of them in `rai_insights.compute()` (Section 9.3). The dashboard opens inline in
Section 9.3; if it does not appear, open the printed `http://localhost:<port>` link in a browser. To reopen it later
without recomputing:

```python
import utils, config
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard
rai = RAIInsights.load(str(config.RAI_DIR))
cohorts, _ = utils.build_dashboard_cohorts(rai.test)
ResponsibleAIDashboard(rai, cohort_list=cohorts)
```

`import utils` must come first: it defines the model adapter stored inside the saved analysis.

### Tests

```bash
pytest -q
```

26 tests cover ICD-9 grouping, stateless feature engineering, that identifiers and audit columns never become
features, the leakage check and the patient-grouped split, the trade-off table and capacity-constrained threshold,
Wilson intervals, the dashboard model adapter and the saved dashboard cohorts. Two tests use the real data and are
skipped when `data/raw/diabetic_data.csv` is absent.

## Data description

- **Source.** UCI *Diabetes 130-US Hospitals for Years 1999–2008*: 101,766 inpatient encounters from 130 US
  hospitals, 50 columns. Licence CC BY 4.0. Reference: Strack et al. (2014), *BioMed Research International*,
  Article ID 781670. Data: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- **Cohort.** Encounters ending in death or hospice (2,423) and with unknown gender (3) are excluded, leaving
  **99,340 encounters from 69,987 patients**.
- **Target.** `readmit_30d` = 1 when `readmitted == "<30"`. Prevalence **11.4%**. Readmissions after 30 days count
  as negatives, because the 30-day quality metric does not count them.
- **Features.** 13 numeric and 21 categorical inputs describing the stay that is ending and the prior year's
  utilisation, all known at discharge. `race` and `gender` are kept for the fairness audit but are never model
  inputs. `payer_group` is a model input and is audited as a socioeconomic proxy.
- **Caveats.** The literal string `"None"` in `A1Cresult` means *test not performed* and is preserved on load.
  `weight` is 97% missing and dropped. `payer_code` is 40% missing, not at random. The label only sees
  **in-network** readmissions, so positives are under-counted. The data is from 1999–2008 and has no dates or site
  identifier, so temporal and external validation are not possible.

## Evaluation and thresholding

- **Split.** Patient-grouped 60 / 10 / 15 / 15 (train / calibration / validation / test) with `GroupShuffleSplit`
  on `patient_nbr`, so no patient appears in two partitions (23% of patients have repeat encounters). Asserted in
  code.
- **Pipeline and baselines.** All imputation, scaling and one-hot encoding (with rare-category grouping) sits inside
  a scikit-learn `Pipeline`, fitted on training data only. Majority-class and prevalence-only baselines are defined
  before any tuning.
- **Model.** Logistic regression, chosen over gradient boosting by a pre-declared rule (boosting must improve
  validation PR-AUC by ≥ 0.01; it improved by 0.009). Validation ROC-AUC 0.660, PR-AUC 0.207.
- **Calibration.** Sigmoid (Platt) calibration fitted on the dedicated calibration slice. Brier 0.0988 against a
  0.1024 baseline; expected calibration error 0.011; also checked within subgroups.
- **Threshold.** Three methods compared on validation: workload-capped (≤ 250 alerts per 1,000), recall floor
  (≥ 50%), and harm-ratio sensitivity. The joint target was infeasible, so the pre-declared fallback applies:
  maximise recall within capacity. **Locked threshold 0.137**, before the test set was opened.
- **Test result (evaluated once).** ROC-AUC 0.653, PR-AUC 0.207, recall 41.9%, precision 19.1% at 250 alerts per
  1,000: about 48 readmissions caught, 66 missed and 5 reviews per readmission caught.
- **Post-hoc analyses** (8.4.1–8.5, 9.4d) use the test set only to check recommendations, and are labelled as such;
  none of them changed the locked model or threshold.

## Responsible AI and reproducibility

- **Dashboard.** `RAIInsights` with all six components: data analysis, model overview and fairness, error analysis,
  feature importance, counterfactuals (DiCE) and causal analysis (EconML). It runs on a class-stratified sample
  (5,000 train, 1,997 test). Eight analysis cohorts are saved into the dashboard, including the error-analysis
  blind spot (first stay, sent home, ≤ 5 days).
- **Model adapter.** The dashboard receives the locked model through `utils.RAIModelAdapter`, which restores the
  training data types before predicting and applies the locked threshold. Inside the dashboard, "probability" is an
  alert score where **0.5 equals the locked threshold**, not a 50% risk.
- **Key findings.** The model is a hospital-history model. Its lowest-error dashboard group has 0% recall. Fairness
  gaps run by sex, age and payer rather than race. Dropping `payer_group` does not help, while per-group thresholds
  do. The HbA1c causal effect points protective but is only hypothesis-generating. Details, risks and monitoring
  triggers are in Sections 9.6–9.8 and 10.
- **Governance.** Human override always available; subgroup recall and false-positive rate monitored monthly with
  alert triggers; annual refit that repeats the full audit; a model card; and a sunset clause if clinician
  acceptance stays below 50%.
- **Reproducibility.** Random seed **42** everywhere (split, models, sub-sampling, bootstrap). Environment pinned in
  `environment.yml`. DiCE and EconML are not fully seeded, so counterfactual and causal figures can vary slightly
  between runs; the reported figures and screenshots come from the same saved run.

*Academic exercise for the AI4HC course. Not a validated clinical tool; do not use it for decisions about real
patients.*

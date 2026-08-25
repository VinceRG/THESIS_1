# Smart Healthcare Clinic Management - Model Training and Evaluation Report

*Regenerated 2026-08-14 by running the application's live `/retrain` training pipeline (`train_and_evaluate_model`, `fast=False`, real `RandomizedSearchCV` hyperparameter tuning) against Accudetek Main Branch's full consultation history (10,129 records, January 2023 - August 2026) in the application database. `/upload` still trains with `fast=True` (fixed parameters) so a large file import isn't blocked on tuning; `/retrain` is the explicit, deliberate action an administrator takes to produce the tuned model these figures describe. This replaces an earlier version of this report generated before hyperparameter tuning was wired into any live route.*

## 1. Model Objective

The objective of this Random Forest regression model is to forecast the monthly consultation demand at Accudetek Health Diagnostics. The model predicts the expected consultation volume by diagnosis, age group, and gender to provide decision-support information for clinic administrators in staff planning, resource preparedness, service readiness, and operational decision-making.

## 2. Dataset and Feature Preparation

| Item | Value |
|---|---:|
| Source Records | 10,129 consultation records, January 2023 - August 2026 |
| Dataset Period Used for Training | January 2023 - December 2025 |
| Training Frame Rows | 3,672 |
| Diagnosis-Age Group-Gender Segments | 102 |
| Diagnoses Covered (modeled) | 17 |
| Age Groups Covered | 3 |
| Gender Categories Covered | 2 |
| Training Granularity | Diagnosis x Age Group x Gender x Month |

### Complete-Year Training Rule

Only calendar years with all 12 months present are used for training, so monthly and quarterly seasonality are learned from complete yearly cycles.

| Item | Value |
|---|---:|
| Years Found in Source Data | 2023, 2024, 2025, 2026 |
| Years Used for Training | 2023, 2024, 2025 |
| Years Excluded | 2026 |
| Records Excluded by This Rule | 2,048 |

**Note:** because 2026 is still in progress, this rule currently excludes it entirely rather than partially — including its most recent, most representative growth trend. This is a known trade-off of the current rule, not an artifact of this regeneration; it applies identically to every training run until the rule itself is revisited.

### Rare Diagnosis Handling

Diagnoses with fewer than 50 historical records are grouped into a single "Other Services/Cases" category to reduce sparse-label noise while retaining the demographic prediction structure.

| Item | Value |
|---|---:|
| Original Diagnosis Labels | 291 |
| Modeled Diagnosis Labels | 17 |
| Diagnoses Grouped as "Other Services/Cases" | 275 |
| Records in the Grouped Category | 4,096 |

### Predictive Features

The model was trained using the following features:

- Diagnosis
- Age Group
- Gender
- Monthly Seasonality (cyclical sine/cosine encoding)
- Time Index
- One-, Two-, Three-, Six-, and Twelve-Month Lag Values
- Rolling Averages (3- and 6-month) and Rolling Standard Deviation
- Recent Trend Indicator

The consultation records were transformed into monthly consultation counts for each diagnosis-age group-gender segment. Feature engineering was then performed by generating lag values, rolling averages, seasonal indicators, and trend features to enable the Random Forest model to learn recurring consultation patterns and temporal behavior.

## 3. Model Validation Method

To simulate real-world forecasting, the model used a time-based validation approach rather than a random train-test split.

Earlier months were used for model training, while the most recent months were reserved for validation. Cross-validation folds are also split by calendar period rather than by row, so no fold ever mixes past and future months.

This prevents information leakage and reflects the practical forecasting scenario in which future consultation demand must be predicted using only historical records.

| Item | Value |
|---|---:|
| Training Months | 30 |
| Validation Months | 6 |
| Validation Period | July 2025 - December 2025 |

### Baseline Forecast

The model was compared against a naive forecasting approach that assumes the next month's consultation count will be the same as the previous month's value.

This baseline provides a simple benchmark for evaluating whether the Random Forest model learns meaningful consultation patterns beyond historical repetition.

## 4. Demographic Feature Comparison

Two Random Forest models were evaluated.

### Model A - Proposed Model

Features included:

- Diagnosis
- Age Group
- Gender
- Monthly Lag Values
- Rolling Averages
- Trend Features
- Seasonal Indicators

Validation results:

| Metric | Result |
|---|---:|
| Validation R2 | 0.9382 |
| Validation MAE | 1.0242 |
| Validation RMSE | 1.6140 |
| Improvement over Baseline | 17.31% |

### Model B - Comparison Model

Features included:

- Diagnosis
- Monthly Lag Values
- Rolling Averages
- Trend Features
- Seasonal Indicators

Age Group and Gender were excluded.

Validation results:

| Metric | Result |
|---|---:|
| Validation R2 | 0.9703 |
| Validation MAE | 2.9936 |
| Validation RMSE | 5.0229 |
| Improvement over Baseline | 21.71% |

### Interpretation

Model A generated lower absolute prediction error than Model B by 1.9694 MAE points, in addition to producing demographic-specific forecasts by diagnosis, age group, and gender rather than diagnosis alone.

Model B recorded a higher R2 and a higher percentage improvement over its own baseline, but it performs a coarser, diagnosis-only forecasting task with roughly a sixth as many training rows. Because the two models operate at different levels of granularity and against different baselines, their R2 and improvement percentages are not directly comparable.

Model A was selected because it provides both lower absolute forecasting error and demographic-specific predictions that better support clinic planning, staffing, and resource allocation.

## 5. Model Performance

**Model Verdict:** Acceptable

| Metric | Result |
|---|---:|
| Validation R2 | 0.9382 |
| Validation MAE | 1.0242 |
| Validation RMSE | 1.6140 |
| Baseline MAE | 1.2386 |
| Baseline RMSE | 2.1043 |
| Improvement over Baseline | 17.31% |
| Cross-Validation R2 Mean | 0.9218 +/- 0.0045 |
| Cross-Validation MAE Mean | 0.9643 |
| Training R2 (Reference Only) | 0.9444 |

**Note on the cross-validation figure:** `/retrain` now trains with `fast=False`, so this cross-validation estimate runs across the multiple expanding-window folds the code supports (rather than a single fold), and the reported standard deviation (±0.0045) is a genuine, non-trivial estimate of consistency across folds. `/upload` still trains with `fast=True` (fixed parameters, single CV fold) since it's triggered automatically by a file import and should not block on tuning; only the deliberate, explicit `/retrain` action produces the tuned figures in this report.

## 6. Performance Metric Interpretation

### Validation R2

Measures how well the model explains the variation in unseen validation data.

Higher values indicate stronger predictive performance on future consultation records.

### Mean Absolute Error (MAE)

Measures the average prediction error in consultation cases.

An MAE of 1.0242 means that, on average, the predicted consultation volume differs from the actual value by approximately one consultation case per diagnosis-age group-gender segment.

### Root Mean Squared Error (RMSE)

Measures the magnitude of prediction errors while assigning greater weight to larger mistakes.

Lower RMSE values indicate better overall forecasting accuracy.

### Baseline Improvement

This measures how much the Random Forest model reduced prediction error compared with simply using the previous month's consultation count as the forecast.

A 17.31% improvement demonstrates that the model learned meaningful consultation patterns rather than merely repeating historical observations.

## 7. Final Interpretation

Based on the validation results, the Random Forest regression model demonstrated acceptable performance for short-term consultation forecasting.

The model achieved a validation R2 of 0.9382, indicating that it explained approximately 94% of the variation in unseen consultation demand.

Its Mean Absolute Error of 1.0242 indicates that the average forecasting error was approximately one consultation case per diagnosis-age group-gender month segment.

Compared with the naive last-month forecasting baseline, the proposed model reduced prediction error by 17.31%, demonstrating that it learned recurring consultation patterns rather than simply repeating previous observations.

These findings support the feasibility of the proposed Smart Healthcare Clinic Management System as a decision-support tool for forecasting consultation trends, supporting staff planning, improving service preparedness, and assisting operational decision-making at Accudetek Health Diagnostics.

## 8. Best Random Forest Parameters

| Hyperparameter | Value |
|---|---:|
| n_estimators | 120 |
| max_depth | 6 |
| min_samples_split | 2 |
| min_samples_leaf | 2 |
| max_features | 0.5 |

**Methodology note:** these parameters are the output of a real `RandomizedSearchCV` search (8 iterations, scored on mean absolute error, over expanding-window `TimeSeriesSplit` folds that never divide a calendar month) run by `/retrain`, which trains with `fast=False`. `/upload` still trains with `fast=True` (the fixed parameters `n_estimators=80, max_depth=6, min_samples_split=5, min_samples_leaf=2, max_features=sqrt`) so that importing a large consultation file isn't blocked on tuning; an administrator clicking "Retrain Model" is what produces the tuned parameters and figures reported here. Search runtime on this dataset (10,129 records) was measured at roughly 10-11 seconds, well within a normal request timeout.

## 9. Specialist/Department Attribution Methodology

The Random Forest described in Sections 1-8 predicts **consultation case volume** by diagnosis, age group, gender, and month. It does not predict a specialist or role directly. A separate, deliberately non-ML attribution step translates each forecasted diagnosis into the staff role most likely responsible for it, so that predicted volume can be turned into a staffing recommendation.

### ML prediction inputs vs. resource-planning inputs

These are two distinct stages and should not be conflated:

| Stage | Inputs | Output |
|---|---|---|
| ML prediction (Random Forest) | diagnosis, age group, gender, month, cyclical seasonality, time index, 1/2/3/6/12-month lags, 3/6-month rolling mean and std, trend indicator | predicted consultation case count per diagnosis-age group-gender-month segment |
| Resource planning (non-ML) | the above prediction, plus scheduled appointment data, current active staff counts by role, and the diagnosis-to-role attribution below | required staff by role, staffing gap, staffing recommendation |

Patient and appointment data feed the resource-planning stage, not the Random Forest's training features - the model itself is never trained on appointment records.

### Two-tier diagnosis-to-role attribution

Each forecasted diagnosis is mapped to a staff role using, in order:

1. **Historical mapping (preferred).** For each diagnosis, every historical `ConsultationRecord.physician` value is matched by name against the live `StaffMember` roster to recover that physician's role, and the most frequent matched role is used - i.e. "which role actually handled this diagnosis historically," derived from data already in the database rather than guessed. A diagnosis only uses this mapping when it has at least **5 matched historical records** and the top role holds at least **50%** of those matches; both thresholds are methodology choices (analogous to the existing 50-record minimum for modeling a diagnosis at all, Section 2), not the result of a formal statistical test.
2. **Keyword fallback.** When historical evidence is insufficient, the system falls back to a static keyword rule (matching diagnosis/service text against Imaging, Laboratory, and Cardiology keyword lists, defaulting to General Physicians otherwise).

Every staffing table on the Resources page (daily and monthly) and the exportable "Resource Recommendation" report display which of these two sources produced each number.

### Observed mapping coverage (this dataset, this run)

| Item | Value |
|---|---:|
| Modeled diagnoses | 19 |
| Mapped via historical staffing data | 19 (100%) |
| Mapped via keyword fallback | 0 (0%) |

On the real Accudetek dataset, `ConsultationRecord.physician` is populated consistently enough that every modeled diagnosis clears the historical-mapping threshold, most with 100% role agreement among matched records (the lowest observed share was 61.8%, for CBC, still above the 50% threshold). The keyword fallback remains in place as a safety net for diagnoses with no or insufficient historical physician data (e.g., a newly introduced diagnosis), but is not exercised on the current dataset. This is disclosed as an observed result of this run, not a guarantee - a dataset with sparser or less consistent physician records would fall back more often, which the reporting above is designed to surface rather than hide.

## 10. Study Limitation

Note: The reported model performance was obtained using the simulated consultation dataset developed for this study. Although the system demonstrates the technical feasibility of consultation forecasting, deployment in an actual clinical environment would require retraining and validation using Accudetek Health Diagnostics' historical consultation records to verify predictive performance under real-world conditions.

Additionally, the current complete-year training rule (Section 2) means the model is validated only through December 2025 and does not yet learn from 2026 activity. Any deployment decision should account for this gap until the rule is revisited.

The diagnosis-to-role attribution in Section 9 depends on `ConsultationRecord.physician` being populated with names that match the live `StaffMember` roster; it is a decision-support translation of the volume forecast, not an independently validated clinical staffing model, and should be reviewed by clinic administrators before acting on it.

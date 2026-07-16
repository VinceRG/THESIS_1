# Smart Healthcare Clinic Management - Model Training and Evaluation Report

## 1. Model Objective

The objective of this Random Forest regression model is to forecast the monthly consultation demand at Accudetek Health Diagnostics. The model predicts the expected consultation volume by diagnosis, age group, and gender to provide decision-support information for clinic administrators in staff planning, resource preparedness, service readiness, and operational decision-making.

## 2. Dataset and Feature Preparation

| Item | Value |
|---|---:|
| Dataset Period | January 2023 - December 2025 |
| Training Frame Rows | 1296 |
| Diagnosis-Age Group-Gender Segments | 36 |
| Diagnoses Covered | 6 |
| Age Groups Covered | 3 |
| Gender Categories Covered | 2 |
| Training Granularity | Diagnosis x Age Group x Gender x Month |

### Predictive Features

The model was trained using the following features:

- Diagnosis
- Age Group
- Gender
- Monthly Seasonality
- Time Index
- One-, Two-, Three-, Six-, and Twelve-Month Lag Values
- Rolling Averages
- Recent Trend Indicators

The consultation records were transformed into monthly consultation counts for each diagnosis-age group-gender segment. Feature engineering was then performed by generating lag values, rolling averages, seasonal indicators, and trend features to enable the Random Forest model to learn recurring consultation patterns and temporal behavior.

## 3. Model Validation Method

To simulate real-world forecasting, the model used a time-based validation approach rather than a random train-test split.

Earlier months were used for model training, while the most recent months were reserved for validation.

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
| Validation R2 | 0.7232 |
| Validation MAE | 1.9252 |
| Validation RMSE | 2.6726 |
| Improvement over Baseline | 29.88% |

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
| Validation R2 | 0.8508 |
| Validation MAE | 5.4518 |
| Validation RMSE | 6.4459 |
| Improvement over Baseline | 9.56% |

### Interpretation

Model A generated lower prediction error than Model B by 3.5266 MAE points while producing detailed forecasts by diagnosis, age group, and gender.

Although Model B achieved a higher R2, it performed a simpler diagnosis-level forecasting task. Since the two models operate at different levels of granularity, their R2 values should not be interpreted as a direct measure of superiority.

Model A was selected because it provides lower forecasting error and produces demographic-specific predictions that better support clinic planning, staffing, and resource allocation.

## 5. Model Performance

**Model Verdict:** Acceptable

| Metric | Result |
|---|---:|
| Validation R2 | 0.7232 |
| Validation MAE | 1.9252 |
| Validation RMSE | 2.6726 |
| Baseline MAE | 2.7454 |
| Baseline RMSE | 3.6177 |
| Improvement over Baseline | 29.88% |
| Cross-Validation R2 Mean | 0.6846 +/- 0.0267 |
| Cross-Validation MAE Mean | 2.1259 |
| Training R2 (Reference Only) | 0.8028 |

## 6. Performance Metric Interpretation

### Validation R2

Measures how well the model explains the variation in unseen validation data.

Higher values indicate stronger predictive performance on future consultation records.

### Mean Absolute Error (MAE)

Measures the average prediction error in consultation cases.

An MAE of 1.9252 means that, on average, the predicted consultation volume differs from the actual value by approximately two consultation cases per diagnosis-age group-gender segment.

### Root Mean Squared Error (RMSE)

Measures the magnitude of prediction errors while assigning greater weight to larger mistakes.

Lower RMSE values indicate better overall forecasting accuracy.

### Baseline Improvement

This measures how much the Random Forest model reduced prediction error compared with simply using the previous month's consultation count as the forecast.

A 29.88% improvement demonstrates that the model learned meaningful consultation patterns rather than merely repeating historical observations.

## 7. Final Interpretation

Based on the validation results, the Random Forest regression model demonstrated acceptable performance for short-term consultation forecasting.

The model achieved a validation R2 of 0.7232, indicating that it explained approximately 72% of the variation in unseen consultation demand.

Its Mean Absolute Error of 1.9252 indicates that the average forecasting error was approximately two consultation cases per diagnosis-age group-gender month segment.

Compared with the naive last-month forecasting baseline, the proposed model reduced prediction error by 29.88%, demonstrating that it successfully learned recurring consultation patterns rather than simply repeating previous observations.

These findings support the feasibility of the proposed Smart Healthcare Clinic Management System as a decision-support tool for forecasting consultation trends, supporting staff planning, improving service preparedness, and assisting operational decision-making at Accudetek Health Diagnostics.

## 8. Best Random Forest Parameters

| Hyperparameter | Value |
|---|---:|
| n_estimators | 300 |
| max_depth | 10 |
| min_samples_split | 10 |
| min_samples_leaf | 2 |
| max_features | log2 |

## 9. Study Limitation

Note: The reported model performance was obtained using the simulated consultation dataset developed for this study. Although the system demonstrates the technical feasibility of consultation forecasting, deployment in an actual clinical environment would require retraining and validation using Accudetek Health Diagnostics' historical consultation records to verify predictive performance under real-world conditions.
